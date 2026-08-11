from pathlib import Path
from .chunking import Chunk
from .models import MinimalSource


class ContextCreator:
    """Formats retrieved chunks/sources into a character-capped LLM context."""

    def __init__(self, max_context_length: int = 6000) -> None:
        """Initialize the creator.

        Args:
            max_context_length: Maximum total characters of context to
                assemble. This is a cheap pre-filter for which whole
                chunks/sources to include; a further, exact token-based
                cap is applied separately via the LLM's own tokenizer
                (see `BaseLLM.truncate_to_token_budget`).
        """
        self.max_context_length = max_context_length

    def build_context_from_chunks(self, chunks: list[Chunk]) -> str:
        """Format retrieved chunks (already holding their text) as context.

        Args:
            chunks: Retrieved chunks, most relevant first.
        """
        parts = [
            f"# Source: {chunk.file_path}\n{chunk.text}" for chunk in chunks
            ]
        return self._assemble_context(parts)

    def build_context_from_sources(self, sources: list[MinimalSource]) -> str:
        """Format sources (re-read from disk) as context.

        Used when only `MinimalSource` (file_path + offsets, no cached
        text) is available, e.g. in `answer_dataset`, which loads
        previously-saved search results rather than fresh chunks.

        Args:
            sources: Sources to read and format, most relevant first.
        """
        parts = []
        for source in sources:
            text = self._read_source_text(source)
            if text:
                parts.append(f"# Source: {source.file_path}\n{text}")
        return self._assemble_context(parts)

    def _assemble_context(self, parts: list[str]) -> str:
        """Join formatted source blocks, capped at `self.max_context_length`.

        Greedily includes whole blocks until the next one would exceed
        the budget (so a source is never cut awkwardly mid-block). If
        even the first block alone is too big, it is hard-truncated so
        the LLM still gets *some* context instead of none.

        Args:
            parts: Already-formatted "# Source: ...\\n<text>" blocks,
                ordered by relevance (most relevant first).

        Returns:
            The assembled context string, blocks joined by blank lines.
        """
        if self.max_context_length <= 0 or not parts:
            return ""

        included: list[str] = []
        total = 0
        for part in parts:
            separator_len = 2 if included else 0  # "\n\n" between blocks
            if total + separator_len + len(part) <= self.max_context_length:
                included.append(part)
                total += separator_len + len(part)
            elif not included:
                included.append(part[: self.max_context_length])
                break
            else:
                break
        return "\n\n".join(included)

    @staticmethod
    def _read_source_text(source: MinimalSource) -> str:
        """Re-read the exact text span of a MinimalSource from disk.

        This one has no need for `self.max_context_length` (or any
        instance state), so it stays a `@staticmethod` legitimately.

        Args:
            source: The source to read.

        Returns:
            The text span, or an empty string if the file can no longer
            be read (e.g. moved/deleted since indexing) instead of
            crashing.
        """
        try:
            text = Path(source.file_path).read_text(encoding="utf-8")
            return text[
                source.first_character_index:source.last_character_index
                ]
        except (OSError, UnicodeDecodeError) as exc:
            print(f"[context] could not re-read {source.file_path}: {exc}")
            return ""
