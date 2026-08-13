
import pickle
import re
from pathlib import Path
import bm25s  # type: ignore
from tqdm import tqdm  # type: ignore
from .chunking import Chunk, chunk_file

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "can", "could", "did", "do", "does", "doing", "done", "for", "from",
    "had", "has", "have", "having", "how", "i", "if", "in", "into", "is",
    "it", "its", "of", "on", "or", "our", "should", "so", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those",
    "to", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",
})


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 matching.

    Splits on non-alphanumeric characters and lowercases everything.
    Also splits identifiers written in ``snake_case`` or ``camelCase``
    into their sub-words, so a query like "openai server" can match a
    symbol like ``OpenAIServer`` or ``openai_server`` in the code.
    Common English stopwords (see :data:`_STOPWORDS`) are dropped, since
    they add noise rather than discriminating signal.

    Args:
        text: Raw text to tokenize.

    Returns:
        A list of lowercase tokens, with stopwords removed.
    """
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        tokens.append(raw.lower())

        parts = raw.split("_")
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts if p)

        camel_parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z]|$)", raw)
        if len(camel_parts) > 1:
            tokens.extend(p.lower() for p in camel_parts if p)
    return [t for t in tokens if t not in _STOPWORDS]


class BM25Index:
    """A BM25 retrieval index over a list of source-code/text chunks."""

    _INDEX_SUBDIR = "bm25s_index"
    _CHUNKS_FILE = "chunks.pkl"
    _KNOWN_TEXT_FILENAMES = frozenset({"LICENSE"})

    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Initialize the index, optionally building it right away.

        Args:
            chunks: The chunks to index. If provided, the BM25 model is
                fit immediately; otherwise call :meth:`build` or
                :meth:`load` before calling :meth:`search`.
            k1: BM25 term-frequency saturation parameter (default 1.5).
                Higher values let repeated term matches keep
                contributing more to the score.
            b: BM25 length-normalization parameter, in [0, 1] (default
                0.75). 0 disables length normalization entirely; 1
                fully normalizes by document length. Lower it if longer
                chunks are being unfairly penalized.
        """
        self.chunks: list[Chunk] = chunks or []
        self.k1 = k1
        self.b = b
        self._bm25: bm25s.BM25 | None = None
        if self.chunks:
            self._fit()

    def _fit(self) -> None:
        """Tokenize all chunks and fit the underlying BM25 model."""
        tokenized_corpus = [tokenize(chunk.text) for chunk in self.chunks]
        self._bm25 = bm25s.BM25(k1=self.k1, b=self.b, method="robertson")
        self._bm25.index(tokenized_corpus, show_progress=False)

    @classmethod
    def build(
        cls,
        repo_path: Path,
        max_chunk_size: int = 2000,
        extensions: tuple[str, ...] = (
            ".py", ".md", ".txt", ".rst", ".yaml", ".yml", ".toml", ".cfg",
            ".ini"),
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        """Walk a repository, chunk every matching file, and build the index.

        Args:
            repo_path: Root directory of the repository to index.
            max_chunk_size: Maximum number of characters per chunk.
            extensions: File extensions to include. Files with no
                extension at all (e.g. ``LICENSE``) are matched
                separately against :data:`_KNOWN_TEXT_FILENAMES`, since
                an empty ``path.suffix`` can never match an extensions
                tuple no matter how it's configured.
            k1: BM25 term-frequency saturation parameter.
            b: BM25 length-normalization parameter.

        Returns:
            A fitted :class:`BM25Index`.
        """
        matching_files = [
            path
            for path in sorted(repo_path.rglob("*"))
            if path.is_file()
            and (
                path.suffix in extensions or path.name in
                cls._KNOWN_TEXT_FILENAMES)
        ]

        chunks: list[Chunk] = []
        for path in tqdm(matching_files, desc="Chunking files"):
            chunks.extend(chunk_file(path, max_chunk_size=max_chunk_size))
        return cls(chunks, k1=k1, b=b)

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """Return the top-k chunks most relevant to ``query``.

        Args:
            query: The natural-language or code query string.
            k: Maximum number of chunks to return.

        Returns:
            The top-k chunks, ranked by descending BM25 score. Returns
            an empty list if the index has not been built yet, or if
            ``query`` is empty/degenerate.

        Raises:
            ValueError: If ``k`` is negative.
        """
        if k < 0:
            raise ValueError("k must be >= 0")
        if self._bm25 is None or not self.chunks or k == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        k_effective = min(k, len(self.chunks))
        results, _scores = self._bm25.retrieve(
            [query_tokens], k=k_effective, show_progress=False
        )
        return [self.chunks[int(i)] for i in results[0]]

    def save(self, index_dir: Path) -> None:
        """Persist the index (chunks + BM25 statistics) to disk.

        Args:
            index_dir: Directory to write the index files into. Created
                if it does not already exist.

        Raises:
            OSError: If the directory cannot be created or written to.
        """
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / self._CHUNKS_FILE).open("wb") as f:
            pickle.dump(self.chunks, f)
        if self._bm25 is not None:
            self._bm25.save(
                str(index_dir / self._INDEX_SUBDIR), show_progress=False)

    @classmethod
    def load(cls, index_dir: Path) -> "BM25Index":
        """Load a previously saved index from disk.

        Args:
            index_dir: Directory previously written by :meth:`save`.

        Returns:
            The restored :class:`BM25Index`.

        Raises:
            FileNotFoundError: If the index files are missing.
        """
        chunks_path = index_dir / cls._CHUNKS_FILE
        bm25s_path = index_dir / cls._INDEX_SUBDIR
        if not chunks_path.exists() or not bm25s_path.exists():
            raise FileNotFoundError(
                f"No index found under {index_dir}. Run `index` first."
            )
        index = cls()
        with chunks_path.open("rb") as f:
            index.chunks = pickle.load(f)
        index._bm25 = bm25s.BM25.load(
            str(bm25s_path), load_corpus=False, show_progress=False
        )
        index.k1 = index._bm25.k1
        index.b = index._bm25.b
        return index
