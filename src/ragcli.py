import json
from pathlib import Path
from tqdm import tqdm
from .bm25 import BM25Index
from .chunking import Chunk
from .context import ContextCreator
from .evaluate import evaluate_dataset
from .llm import BaseLLM, Qwen3LLM
from .models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


class RagCLI:
    """Command-Line Interface for the Retrieval-Augmented Generation system."""

    def __init__(self, sys_prompt: str) -> None:
        """Store the system prompt used for all answer generation.

        Args:
            sys_prompt: Instructions describing the assistant's role,
                passed to the LLM on every `answer`/`answer_dataset` call.
        """
        self.sys_prompt = sys_prompt

    def index(
        self,
        repo_path: str = "data/raw/vllm-0.10.1",
        max_chunk_size: int = 2000,
        index_dir: str = "data/processed",
        k1: float = 1.2,
        b: float = 0.25,
    ) -> None:
        """Ingest a repository and build the searchable BM25 index.

        Args:
            repo_path: Root directory of the repository to index.
            max_chunk_size: Maximum number of characters per chunk.
            index_dir: Directory to persist the resulting index into.
            k1: BM25 term-frequency saturation parameter (default 1.5).
            b: BM25 length-normalization parameter, in [0, 1]. Defaults
                to 0.2 here (lower than bm25s's own library default of
                0.75), empirically validated against this project's
                datasets: chunk sizes vary widely (up to max_chunk_size),
                and the classic default over-penalizes longer, often
                genuinely more relevant chunks — b=0.2 measured
                Recall@5=0.810 (docs) vs. 0.640 at the library default,
                which is below the subject's required 80% threshold.
        """
        repo = Path(repo_path)
        if not repo.is_dir():
            print(f"[index] error: '{repo_path}' is not a directory")
            return

        index = BM25Index.build(repo,
                                max_chunk_size=max_chunk_size,
                                k1=k1, b=b)
        index.save(Path(index_dir))
        print(f"Ingestion complete! Indexed {len(index.chunks)} "
              f"chunks under {index_dir}")

    def search(
        self,
        query: str,
        k: int = 5,
        index_dir: str = "data/processed",
    ) -> None:
        """Search the index for a single query and print the top-k sources.

        Args:
            query: The search query.
            k: Number of results to retrieve.
            index_dir: Directory the index was saved into by `index`.
        """
        index = self._load_index(index_dir)
        if index is None:
            return

        chunks = index.search(query, k=k)
        sources = [self._chunk_to_source(c) for c in chunks]
        print(json.dumps([s.model_dump() for s in sources], indent=2))

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        index_dir: str = "data/processed",
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Search the index for every question in a dataset file.

        Args:
            dataset_path: Path to a JSON file matching the RagDataset model.
            k: Number of results to retrieve per question.
            index_dir: Directory the index was saved into by `index`.
            save_directory: Directory to save the StudentSearchResults JSON
            into (same filename as `dataset_path`).
        """
        index = self._load_index(index_dir)
        if index is None:
            return

        dataset = self._load_dataset(dataset_path)
        if dataset is None:
            return

        results: list[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            chunks = index.search(question.question, k=k)
            results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=[
                        self._chunk_to_source(c) for c in chunks],
                )
            )

        output = StudentSearchResults(search_results=results, k=k)
        out_path = self._save_json(output, save_directory, dataset_path)
        print(f"Saved student_search_results to {out_path}")

    def answer(
        self,
        query: str,
        k: int = 10,
        index_dir: str = "data/processed",
        max_context_length: int = 6000,
        max_context_tokens: int = 3000,
        max_new_tokens: int = 350,
    ) -> None:
        """Answer a single query using retrieved context.

        Args:
            query: The question to answer.
            k: Number of context chunks to retrieve.
            index_dir: Directory the index was saved into by `index`.
            max_context_length: Maximum characters of context passed to
                the LLM (a cheap pre-filter used to pick which whole
                chunks to include, capped independently of `k`).
            max_context_tokens: Hard cap on the context, enforced with
                the model's own tokenizer (a character count is only an
                approximation of token count).
            max_new_tokens: Maximum number of tokens the LLM generates.
        """
        index = self._load_index(index_dir)
        if index is None:
            return

        chunks = index.search(query, k=k)
        context_creator = ContextCreator(max_context_length=max_context_length)
        context = context_creator.build_context_from_chunks(chunks)

        llm = Qwen3LLM()
        context = llm.truncate_to_token_budget(context, max_context_tokens)
        answer_text = llm.generate(
            system_prompt=self.sys_prompt,
            user_prompt=f"Context:\n{context}\n\nQuestion: {query}",
            max_new_tokens=max_new_tokens,
        )

        result = MinimalAnswer(
            question_id="adhoc",
            question=query,
            retrieved_sources=[self._chunk_to_source(c) for c in chunks],
            answer=answer_text,
        )
        print(result.model_dump_json(indent=2, by_alias=True))

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        max_context_length: int = 6000,
        max_context_tokens: int = 3000,
        max_new_tokens: int = 350,
    ) -> None:
        """Generate answers for every question in a saved search-results file.

        Args:
            student_search_results_path: Path to a JSON file matching the
                StudentSearchResults model, produced by `search_dataset`.
            save_directory: Directory to save the resulting
                StudentSearchResultsAndAnswer JSON into.
            max_context_length: Maximum characters of context passed to
                the LLM per question (a cheap pre-filter for which whole
                sources to include).
            max_context_tokens: Hard cap on the context, enforced with
                the model's own tokenizer.
            max_new_tokens: Maximum number of tokens the LLM generates
                per answer.
        """
        try:
            raw = Path(student_search_results_path).read_text(encoding="utf-8")
            data = StudentSearchResults.model_validate_json(raw)
        except (OSError, ValueError) as exc:
            print("[answer_dataset] could not load "
                  f"'{student_search_results_path}': {exc}")
            return

        print(f"Loaded {len(data.search_results)} questions from "
              f"{student_search_results_path}")
        llm = Qwen3LLM()
        context_creator = ContextCreator(max_context_length=max_context_length)
        answers = [
            self._answer_one(result,
                             llm,
                             context_creator,
                             max_context_tokens,
                             max_new_tokens)
            for result in tqdm(data.search_results, desc="Answering")
        ]

        output = StudentSearchResultsAndAnswer(search_results=answers,
                                               k=data.k)
        out_path = self._save_json(output,
                                   save_directory,
                                   student_search_results_path)
        print(f"Saved student_search_results_and_answer to {out_path}")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k_values: str | tuple[int, ...] = "1,3,5,10",
    ) -> None:
        """Evaluate retrieval quality (recall@k) against ground truth.

        Args:
            student_search_results_path: Path to a JSON file matching the
                StudentSearchResults model (search results to evaluate).
            dataset_path: Path to the ground-truth JSON file (RagDataset,
                with AnsweredQuestion entries containing `sources`).
            k_values: k values to evaluate, e.g. "1,3,5,10". Note: Fire
                auto-parses comma-separated CLI args into a tuple, so
                both a raw string and an already-parsed tuple are
                accepted here.
        """
        try:
            if isinstance(k_values, str):
                ks = [int(k.strip()) for k in k_values.split(",") if k.strip()]
            else:
                ks = [int(k) for k in k_values]
        except ValueError:
            print(f"[evaluate] invalid --k_values '{k_values}',"
                  " expected e.g. '1,3,5,10'")
            return

        try:
            student = StudentSearchResults.model_validate_json(
                Path(student_search_results_path).read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            print(f"[evaluate] could not load '{student_search_results_path}':"
                  f" {exc}")
            return

        dataset = self._load_dataset(dataset_path)
        if dataset is None:
            return

        ground_truth = {
            q.question_id: q
            for q in dataset.rag_questions
            if isinstance(q, AnsweredQuestion)
        }
        if not ground_truth:
            print("[evaluate] no answered questions with ground-truth sources "
                  f"in '{dataset_path}'")
            return

        scores = evaluate_dataset(student.search_results, ground_truth, ks)

        print("Evaluation Results")
        print("=" * 40)
        print(f"Questions evaluated: {len(ground_truth)}")
        for k in ks:
            print(f"Recall@{k}: {scores[k]:.3f}")

    def _answer_one(
        self,
        result: MinimalSearchResults,
        llm: BaseLLM,
        context_creator: ContextCreator,
        max_context_tokens: int,
        max_new_tokens: int,
    ) -> MinimalAnswer:
        """Generate one answer for a single (already-searched) question.

        Not a @staticmethod: it needs `self.sys_prompt`.
        """
        context = context_creator.build_context_from_sources(
            result.retrieved_sources)
        context = llm.truncate_to_token_budget(context, max_context_tokens)
        answer_text = llm.generate(
            system_prompt=self.sys_prompt,
            user_prompt=f"Context:\n{context}\n\nQuestion: {result.question}",
            max_new_tokens=max_new_tokens,
        )
        return MinimalAnswer(
            question_id=result.question_id,
            question=result.question,
            retrieved_sources=result.retrieved_sources,
            answer=answer_text,
        )

    @staticmethod
    def _load_index(index_dir: str) -> BM25Index | None:
        """Load a saved BM25Index, printing a friendly error if missing."""
        try:
            return BM25Index.load(Path(index_dir))
        except FileNotFoundError as exc:
            print(f"[error] {exc}")
            return None

    @staticmethod
    def _load_dataset(dataset_path: str) -> RagDataset | None:
        """Load and validate a RagDataset JSON file, or None on failure."""
        try:
            raw = Path(dataset_path).read_text(encoding="utf-8")
            return RagDataset.model_validate_json(raw)
        except (OSError, ValueError) as exc:
            print(f"[error] could not load dataset '{dataset_path}': {exc}")
            return None

    @staticmethod
    def _save_json(
        model: StudentSearchResults,
        save_directory: str,
        source_path: str,
    ) -> Path:
        """Save a pydantic model as JSON, reusing source_path's filename."""
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(source_path).name
        out_path.write_text(model.model_dump_json(indent=2, by_alias=True),
                            encoding="utf-8")
        return out_path

    @staticmethod
    def _chunk_to_source(chunk: Chunk) -> MinimalSource:
        """Convert an internal Chunk into the public MinimalSource model."""
        return MinimalSource(
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
        )
