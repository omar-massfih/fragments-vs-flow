"""Shared data models for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExtractedPage:
    """One extracted page from a source document."""

    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    """Extracted text and metadata for one source document."""

    source: str
    document_id: str
    input_path: Path
    output_path: Path
    pages: tuple[ExtractedPage, ...]
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        """Return the number of extracted pages."""
        return len(self.pages)

    def combined_text(self) -> str:
        """Return the full extracted document text with page markers."""
        sections: list[str] = []
        for page in self.pages:
            page_text = page.text.rstrip("\n")
            if page_text:
                sections.append(f"[PAGE {page.page_number}]\n{page_text}")
            else:
                sections.append(f"[PAGE {page.page_number}]")
        return "\n".join(sections)

    def to_payload(self) -> dict[str, object]:
        """Return the extracted document in a stable payload format."""
        return {
            "source": self.source,
            "document_id": self.document_id,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "page_count": self.page_count,
            "metadata": self.metadata,
            "pages": [
                {"page_number": page.page_number, "text": page.text}
                for page in self.pages
            ],
            "text": self.combined_text(),
        }


@dataclass(frozen=True)
class ChunkRecord:
    """One chunk derived from an extracted source document."""

    source: str
    document_id: str
    text: str
    page_start: int
    page_end: int
    token_start: int
    token_end: int
    chunk_index: int
    chunk_id: str
    total_chunks: int
    headers: dict[str, str] = field(default_factory=dict)
    doc_id: str | None = None
    source_path: str = ""
    window_size: int = 0
    overlap: int = 0
    boundary_snapped: bool = False

    def to_payload(self) -> dict[str, object]:
        """Return the chunk in a stable payload format."""
        return {
            "source": self.source,
            "document_id": self.document_id,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
            "total_chunks": self.total_chunks,
            "headers": self.headers,
            "doc_id": self.doc_id or "",
            "source_path": self.source_path,
            "window_size": self.window_size,
            "overlap": self.overlap,
            "boundary_snapped": self.boundary_snapped,
        }


@dataclass(frozen=True)
class RetrievedChunk:
    """One ranked chunk returned from vector retrieval."""

    source: str
    document_id: str
    text: str
    page_start: int | None
    page_end: int | None
    token_start: int
    token_end: int
    chunk_index: int
    chunk_id: str
    total_chunks: int
    headers: dict[str, str] = field(default_factory=dict)
    score: float = 0.0
    rank: int = 0
    doc_id: str | None = None
    source_path: str = ""
    window_size: int = 0
    overlap: int = 0
    boundary_snapped: bool = False

    def to_payload(self) -> dict[str, object]:
        """Return the retrieved chunk in a stable payload format."""
        return {
            "source": self.source,
            "document_id": self.document_id,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
            "total_chunks": self.total_chunks,
            "headers": self.headers,
            "score": self.score,
            "rank": self.rank,
            "doc_id": self.doc_id or "",
            "source_path": self.source_path,
            "window_size": self.window_size,
            "overlap": self.overlap,
            "boundary_snapped": self.boundary_snapped,
        }
