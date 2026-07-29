import json
from pathlib import Path
import fire
from tqdm import tqdm
from .bm25 import BM25Index
from .chunking import Chunk
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

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a codebase. "
    "Answer only using the information given in the context below. "
    "If the context does not contain the answer, say you don't know — "
    "do not fall back on prior knowledge, and do not guess. "
    "Never invent URLs, links, or file paths: only mention a URL or path "
    "if it appears verbatim in the context. If you want to point to a "
    "source, just state its file path in plain prose (e.g. 'see "
    "docs/foo.md'). The context is organized into blocks starting with "
    "a line like '# Source: <path>' — that marker is formatting for you "
    "to read, not something to repeat; never copy that literal line into "
    "your answer. "
    "Be concise and self-contained: someone reading only your answer, "
    "without the original question, should understand it."
)


def _chunk_to_source(chunk: Chunk) -> MinimalSource:
    """Convert an internal Chunk into the public MinimalSource model."""
    return MinimalSource(
        file_path=chunk.file_path,
        first_character_index=chunk.first_character_index,
        last_character_index=chunk.last_character_index,
    )


def _assemble_context(parts: list[str], max_context_length: int) -> str:
    """Join formatted source blocks, capped at `max_context_length` characters.

    Greedily includes whole blocks until the next one would exceed the
    budget (so we never cut a source awkwardly mid-block). If even the
    first block alone is too big, it is hard-truncated so the LLM still
    gets *some* context instead of none.

    Args:
        parts: Already-formatted "# Source: ...\\n<text>" blocks, ordered
            by relevance (most relevant first).
        max_context_length: Maximum total number of characters to keep.

    Returns:
        The assembled context string, joined by blank lines.
    """
    if max_context_length <= 0 or not parts:
        return ""

    included: list[str] = []
    total = 0
    for part in parts:
        separator_len = 2 if included else 0  # "\n\n" between blocks
        if total + separator_len + len(part) <= max_context_length:
            included.append(part)
            total += separator_len + len(part)
        elif not included:
            included.append(part[:max_context_length])
            break
        else:
            break
    return "\n\n".join(included)


def _build_context(chunks: list[Chunk], max_context_length: int = 6000) -> str:
    """Format retrieved chunks into a context block for the LLM prompt.

    Args:
        chunks: Retrieved chunks, most relevant first.
        max_context_length: Maximum total characters of context to pass
            to the LLM (independent of how many chunks were retrieved
            for evaluation purposes) — a large context makes CPU
            prefill very slow, so this is capped by default.
    """
    parts = [f"# Source: {chunk.file_path}\n{chunk.text}" for chunk in chunks]
    return _assemble_context(parts, max_context_length)


def _read_source_text(source: MinimalSource) -> str:
    """Re-read the exact text span of a MinimalSource from disk.

    Used by `answer_dataset`, which only has file_path + character
    offsets on disk (no chunk text), so the original repository files
    must still be available at the recorded paths.

    Args:
        source: The source to read.

    Returns:
        The text span, or an empty string if the file can no longer be
        read (e.g. moved/deleted since indexing) instead of crashing.
    """
    try:
        text = Path(source.file_path).read_text(encoding="utf-8")
        return text[source.first_character_index:source.last_character_index]
    except (OSError, UnicodeDecodeError) as exc:
        print(f"[answer_dataset] could not re-read {source.file_path}: {exc}")
        return ""


def _build_context_from_sources(
    sources: list[MinimalSource], max_context_length: int = 6000
) -> str:
    """Format ground-truth/retrieved sources (re-read from disk) as context.

    Args:
        sources: Sources to read and format, most relevant first.
        max_context_length: Maximum total characters of context to pass
            to the LLM.
    """
    parts = []
    for source in sources:
        text = _read_source_text(source)
        if text:
            parts.append(f"# Source: {source.file_path}\n{text}")
    return _assemble_context(parts, max_context_length)


class RagCLI:
    """Command-Line Interface for the Retrieval-Augmented Generation system."""

    def index(
        self,
        repo_path: str = "data/raw/vllm-0.10.1",
        max_chunk_size: int = 2000,
        index_dir: str = "data/processed",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Ingest a repository and build the searchable BM25 index.

        Args:
            repo_path: Root directory of the repository to index.
            max_chunk_size: Maximum number of characters per chunk.
            index_dir: Directory to persist the resulting index into.
            k1: BM25 term-frequency saturation parameter (default 1.5).
            b: BM25 length-normalization parameter, in [0, 1] (default
                0.75). Lower it if long chunks seem unfairly penalized.
        """
        repo = Path(repo_path)
        if not repo.is_dir():
            print(f"[index] error: '{repo_path}' is not a directory")
            return

        index = BM25Index.build(repo,
                                max_chunk_size=max_chunk_size,
                                k1=k1,
                                b=b)
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
        sources = [_chunk_to_source(c) for c in chunks]
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
                    retrieved_sources=[_chunk_to_source(c) for c in chunks],
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
        context = _build_context(chunks, max_context_length=max_context_length)
        llm = Qwen3LLM()
        context = llm.truncate_to_token_budget(context, max_context_tokens)
        answer_text = llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=f"Context:\n{context}\n\nQuestion: {query}",
            max_new_tokens=max_new_tokens,
        )

        result = MinimalAnswer(
            question_id="adhoc",
            question=query,
            retrieved_sources=[_chunk_to_source(c) for c in chunks],
            answer=answer_text,
        )
        print(result.model_dump_json(indent=2))

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
        answers = [
            self._answer_one(result,
                             llm,
                             max_context_length,
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
            print(f"[evaluate] invalid --k_values '{k_values}', "
                  "expected e.g. '1,3,5,10'")
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
        out_path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        return out_path

    @staticmethod
    def _answer_one(
        result: MinimalSearchResults,
        llm: BaseLLM,
        max_context_length: int,
        max_context_tokens: int,
        max_new_tokens: int,
    ) -> MinimalAnswer:
        """Generate one answer for a single (already-searched) question."""
        context = _build_context_from_sources(
            result.retrieved_sources, max_context_length=max_context_length
        )
        context = llm.truncate_to_token_budget(context, max_context_tokens)
        answer_text = llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=f"Context:\n{context}\n\nQuestion: {result.question}",
            max_new_tokens=max_new_tokens,
        )
        return MinimalAnswer(
            question_id=result.question_id,
            question=result.question,
            retrieved_sources=result.retrieved_sources,
            answer=answer_text,
        )


if __name__ == "__main__":
    fire.Fire(RagCLI)
