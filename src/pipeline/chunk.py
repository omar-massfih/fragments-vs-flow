"""Chunking step helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from src.models import ChunkRecord, ExtractedDocument, ExtractedPage
from src.paths import SourcePaths
import tiktoken
from tqdm import tqdm


PAGE_MARKER_PATTERN = re.compile(r"^\[\[?PAGE (\d+)\]?\]$")
LEGACY_PAGE_MARKER_PATTERN = re.compile(r"\[\[?PAGE (\d+)\]?\]")
TOKEN_PATTERN = re.compile(r"\S+")
MARKDOWN_HEADER_PATTERN = re.compile(r"(?m)^\s{0,3}#{1,3}\s+\S")
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
TEXTBOOK_ENCODING = "cl100k_base"
VALID_CHUNK_MODES = {"token-window", "per-page", "textbook-markdown"}


@dataclass(frozen=True)
class ChunkingConfig:
    """Configurable chunking parameters."""

    mode: str = "token-window"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_CHUNK_OVERLAP

    def __post_init__(self) -> None:
        """Validate chunk size and overlap."""
        if self.mode not in VALID_CHUNK_MODES:
            raise ValueError(
                "mode must be 'token-window', 'per-page', or 'textbook-markdown'",
            )
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.overlap < 0:
            raise ValueError("overlap must be zero or greater")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")


@dataclass(frozen=True)
class TextbookSection:
    """Prepared textbook section metadata used during chunking."""

    headers: dict[str, str]
    section_text: str
    section_token_start: int
    chunk_texts: tuple[str, ...]


def default_chunking_config(source: str) -> ChunkingConfig:
    """Return source-specific default chunking settings."""
    return ChunkingConfig(
        mode="textbook-markdown" if source == "textbooks" else "token-window",
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_CHUNK_OVERLAP,
    )


def generate_doc_id(source: str, document_id: str) -> str:
    """Generate a stable document ID without depending on filesystem paths."""
    identity = f"{source.strip().lower()}:{document_id.strip()}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def build_chunk_id(
    document: ExtractedDocument,
    chunk_index: int,
    token_start: int,
    token_end: int,
) -> str:
    """Build a stable chunk ID from the document name and chunk index."""
    _ = token_start
    _ = token_end
    return f"{document.document_id}:{chunk_index}"


def parse_extracted_text(text: str) -> tuple[tuple[tuple[int, str], ...], str]:
    """Parse extracted text with page markers into ordered page payloads."""
    pages: list[tuple[int, str]] = []
    current_page: int | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        marker_match = PAGE_MARKER_PATTERN.match(raw_line.strip())
        if marker_match:
            if current_page is not None:
                pages.append((current_page, "\n".join(current_lines).strip()))
            current_page = int(marker_match.group(1))
            current_lines = []
            continue
        current_lines.append(raw_line)

    if current_page is not None:
        pages.append((current_page, "\n".join(current_lines).strip()))

    source_text = "extracted_text_with_pages" if pages else "plain_text_without_pages"
    return tuple(pages), source_text


def load_extracted_document(text_path: Path, source: str) -> ExtractedDocument:
    """Load one extracted text file back into an extracted document record."""
    raw_text = text_path.read_text(encoding="utf-8")
    parsed_pages, parsed_source = parse_extracted_text(raw_text)
    if parsed_pages:
        pages = tuple(
            ExtractedPage(page_number=page_number, text=page_text)
            for page_number, page_text in parsed_pages
        )
    else:
        pages = (ExtractedPage(page_number=1, text=raw_text.strip()),)

    return ExtractedDocument(
        source=source,
        document_id=text_path.stem,
        input_path=text_path.with_suffix(".pdf"),
        output_path=text_path,
        pages=pages,
        metadata={
            "file_name": text_path.name,
            "file_stem": text_path.stem,
            "source": source,
            "loaded_from": parsed_source,
        },
    )


def load_source_documents(source_paths: SourcePaths) -> list[ExtractedDocument]:
    """Load all extracted text documents for one source from disk."""
    return [
        load_extracted_document(text_path, source=source_paths.source)
        for text_path in sorted(source_paths.text_dir.glob("*.txt"))
        if text_path.is_file()
    ]


def chunk_document(
    document: ExtractedDocument,
    config: ChunkingConfig | None = None,
    show_progress: bool = False,
) -> list[ChunkRecord]:
    """Chunk one extracted document into stable chunk records."""
    chunking_config = config or default_chunking_config(document.source)
    if chunking_config.mode == "per-page":
        return chunk_document_per_page(document)
    if chunking_config.mode == "textbook-markdown":
        if document.source != "textbooks":
            raise ValueError("textbook-markdown mode is only supported for textbooks")
        return chunk_textbook_document(
            document,
            config=chunking_config,
            show_progress=show_progress,
        )
    return chunk_document_token_window(document, config=chunking_config)


def chunk_document_token_window(
    document: ExtractedDocument,
    config: ChunkingConfig,
) -> list[ChunkRecord]:
    """Chunk one extracted document with a simple token sliding window."""
    tokens = tokenize_document(document)
    if not tokens:
        return []

    step = config.chunk_size - config.overlap
    doc_id = generate_doc_id(document.source, document.document_id)
    chunks: list[ChunkRecord] = []
    for chunk_index, token_start in enumerate(range(0, len(tokens), step)):
        token_end = min(token_start + config.chunk_size, len(tokens))
        chunk_slice = tokens[token_start:token_end]
        if not chunk_slice:
            continue
        page_numbers = [page_number for _, page_number in chunk_slice]
        chunks.append(
            ChunkRecord(
                source=document.source,
                document_id=document.document_id,
                text=" ".join(token for token, _ in chunk_slice),
                page_start=min(page_numbers),
                page_end=max(page_numbers),
                token_start=token_start,
                token_end=token_end,
                chunk_index=chunk_index,
                chunk_id=build_chunk_id(document, chunk_index, token_start, token_end),
                total_chunks=0,
                headers={},
                doc_id=doc_id,
                source_path=legacy_source_path(document),
                window_size=config.chunk_size,
                overlap=config.overlap,
                boundary_snapped=False,
            ),
        )
        if token_end == len(tokens):
            break

    total_chunks = len(chunks)
    return [
        ChunkRecord(
            source=chunk.source,
            document_id=chunk.document_id,
            text=chunk.text,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            token_start=chunk.token_start,
            token_end=chunk.token_end,
            chunk_index=chunk.chunk_index,
            chunk_id=chunk.chunk_id,
            total_chunks=total_chunks,
            headers=chunk.headers,
            doc_id=chunk.doc_id,
            source_path=chunk.source_path,
            window_size=chunk.window_size,
            overlap=chunk.overlap,
            boundary_snapped=chunk.boundary_snapped,
        )
        for chunk in chunks
    ]


def chunk_textbook_document(
    document: ExtractedDocument,
    config: ChunkingConfig,
    show_progress: bool = False,
) -> list[ChunkRecord]:
    """Chunk textbooks using the old structure-aware textbook behavior."""
    full_text = document.combined_text()
    if not full_text.strip():
        return []
    clean_text, page_spans = build_clean_text_and_page_spans(full_text)
    if not MARKDOWN_HEADER_PATTERN.search(clean_text):
        return chunk_document_token_window(document, config=config)

    tokenizer = tiktoken.get_encoding(TEXTBOOK_ENCODING)
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TEXTBOOK_ENCODING,
        chunk_size=config.chunk_size,
        chunk_overlap=config.overlap,
    )
    textbook_sections = prepare_textbook_sections(
        clean_text=clean_text,
        markdown_docs=markdown_splitter.split_text(clean_text),
        recursive_splitter=recursive_splitter,
        tokenizer=tokenizer,
    )
    doc_id = generate_doc_id(document.source, document.document_id)
    source_path = legacy_source_path(document)

    chunks: list[ChunkRecord] = []
    search_start = 0
    chunk_iterator: list[TextbookSection] | object = textbook_sections
    if show_progress:
        chunk_iterator = tqdm(
            textbook_sections,
            desc="Chunking textbook chunks",
            unit="section",
        )
    for section in chunk_iterator:
        previous_chunk_tokens: list[int] | None = None
        previous_chunk_token_end = 0
        for chunk_text in section.chunk_texts:
            chunk_tokens = tokenizer.encode(chunk_text)
            page_start, page_end, next_search_start = get_page_range_for_text(
                chunk_text=chunk_text,
                clean_text=clean_text,
                page_spans=page_spans,
                search_start=search_start,
            )
            if previous_chunk_tokens is None:
                local_token_start = 0
            else:
                local_token_start = previous_chunk_token_end - shared_token_overlap(
                    previous_chunk_tokens,
                    chunk_tokens,
                    max_overlap=config.overlap,
                )
            local_token_end = local_token_start + len(chunk_tokens)
            token_start = section.section_token_start + local_token_start
            token_end = section.section_token_start + local_token_end
            if next_search_start >= 0:
                search_start = next_search_start
            previous_chunk_tokens = chunk_tokens
            previous_chunk_token_end = local_token_end
            chunks.append(
                ChunkRecord(
                    source=document.source,
                    document_id=document.document_id,
                    text=chunk_text,
                    page_start=page_start,
                    page_end=page_end,
                    token_start=token_start,
                    token_end=token_end,
                    chunk_index=len(chunks),
                    chunk_id=build_chunk_id(
                        document=document,
                        chunk_index=len(chunks),
                        token_start=token_start,
                        token_end=token_end,
                    ),
                    total_chunks=0,
                    headers=section.headers,
                    doc_id=doc_id,
                    source_path=source_path,
                    window_size=config.chunk_size,
                    overlap=config.overlap,
                    boundary_snapped=False,
                ),
            )

    total_chunks = len(chunks)
    return [
        ChunkRecord(
            source=chunk.source,
            document_id=chunk.document_id,
            text=chunk.text,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            token_start=chunk.token_start,
            token_end=chunk.token_end,
            chunk_index=chunk.chunk_index,
            chunk_id=chunk.chunk_id,
            total_chunks=total_chunks,
            headers=chunk.headers,
            doc_id=chunk.doc_id,
            source_path=chunk.source_path,
            window_size=chunk.window_size,
            overlap=chunk.overlap,
            boundary_snapped=chunk.boundary_snapped,
        )
        for chunk in chunks
    ]


def chunk_document_per_page(document: ExtractedDocument) -> list[ChunkRecord]:
    """Chunk one extracted document into one chunk per extracted page."""
    page_chunks: list[ChunkRecord] = []
    token_start = 0
    non_empty_pages = [page for page in document.pages if TOKEN_PATTERN.findall(page.text)]
    total_chunks = len(non_empty_pages)
    doc_id = generate_doc_id(document.source, document.document_id)

    for chunk_index, page in enumerate(non_empty_pages):
        page_tokens = TOKEN_PATTERN.findall(page.text)
        token_end = token_start + len(page_tokens)
        page_chunks.append(
            ChunkRecord(
                source=document.source,
                document_id=document.document_id,
                text=page.text.strip(),
                page_start=page.page_number,
                page_end=page.page_number,
                token_start=token_start,
                token_end=token_end,
                chunk_index=chunk_index,
                chunk_id=build_chunk_id(
                    document=document,
                    chunk_index=chunk_index,
                    token_start=token_start,
                    token_end=token_end,
                ),
                total_chunks=total_chunks,
                headers={},
                doc_id=doc_id,
                source_path=legacy_source_path(document),
                window_size=0,
                overlap=0,
                boundary_snapped=False,
            ),
        )
        token_start = token_end

    return page_chunks


def tokenize_document(document: ExtractedDocument) -> list[tuple[str, int]]:
    """Tokenize an extracted document while preserving page numbers."""
    tokens: list[tuple[str, int]] = []
    for page in document.pages:
        page_tokens = TOKEN_PATTERN.findall(page.text)
        tokens.extend((token, page.page_number) for token in page_tokens)
    return tokens


def extract_page_ranges(text: str) -> dict[int, tuple[int | None, int | None]]:
    """Extract page number ranges from legacy and rebuild page markers."""
    page_markers = list(LEGACY_PAGE_MARKER_PATTERN.finditer(text))
    if not page_markers:
        return {0: (None, None)}

    page_ranges: dict[int, tuple[int | None, int | None]] = {}
    for index, match in enumerate(page_markers):
        page_num = int(match.group(1))
        start_pos = match.start()
        end_pos = page_markers[index + 1].start() if index + 1 < len(page_markers) else len(text)
        for pos in range(start_pos, end_pos):
            if pos not in page_ranges:
                page_ranges[pos] = (page_num, page_num)
    return page_ranges


def get_page_range_for_text(
    chunk_text: str,
    clean_text: str,
    page_spans: list[tuple[int, int, int]],
    search_start: int = 0,
) -> tuple[int | None, int | None, int]:
    """Find the page range and next ordered search position for one textbook chunk."""
    matched_text = None
    chunk_start = -1
    for candidate_text in _candidate_chunk_texts(chunk_text):
        chunk_start = clean_text.find(candidate_text, search_start)
        if chunk_start != -1:
            matched_text = candidate_text
            break
    if chunk_start == -1 or matched_text is None:
        return (None, None, -1)
    chunk_end = chunk_start + len(matched_text)
    pages_in_chunk: set[int] = set()
    for span_start, span_end, page_number in page_spans:
        if span_end <= chunk_start:
            continue
        if span_start >= chunk_end:
            break
        pages_in_chunk.add(page_number)
    if not pages_in_chunk:
        return (None, None, chunk_end)
    return (min(pages_in_chunk), max(pages_in_chunk), chunk_end)


def prepare_textbook_sections(
    clean_text: str,
    markdown_docs: list[object],
    recursive_splitter: RecursiveCharacterTextSplitter,
    tokenizer: tiktoken.Encoding,
) -> list[TextbookSection]:
    """Prepare textbook sections with token offsets and split chunks.

    This avoids repeatedly re-encoding full-text prefixes for every chunk.
    """
    sections: list[TextbookSection] = []
    search_start = 0
    token_cursor = 0

    for markdown_doc in markdown_docs:
        headers = dict(getattr(markdown_doc, "metadata", {}) or {})
        section_text = str(getattr(markdown_doc, "page_content", markdown_doc))
        section_start = clean_text.find(section_text, search_start)
        if section_start >= 0:
            token_cursor += len(tokenizer.encode(clean_text[search_start:section_start]))
            section_token_start = token_cursor
            search_start = section_start + len(section_text)
        else:
            section_token_start = token_cursor

        section_token_length = len(tokenizer.encode(section_text))
        chunk_texts = tuple(recursive_splitter.split_text(section_text))
        sections.append(
            TextbookSection(
                headers=headers,
                section_text=section_text,
                section_token_start=section_token_start,
                chunk_texts=chunk_texts,
            ),
        )
        token_cursor = section_token_start + section_token_length

    return sections


def remove_page_markers(text: str) -> str:
    """Remove page markers before structure-aware textbook chunking."""
    return re.sub(r"\[\[?PAGE \d+\]?\]\n?", "", text)


def build_clean_text_and_page_spans(full_text: str) -> tuple[str, list[tuple[int, int, int]]]:
    """Build page-marker-stripped text plus ordered page spans in that text."""
    parsed_pages, _ = parse_extracted_text(full_text)
    if not parsed_pages:
        clean_text = remove_page_markers(full_text)
        return clean_text, []

    clean_parts: list[str] = []
    page_spans: list[tuple[int, int, int]] = []
    cursor = 0
    total_pages = len(parsed_pages)
    for index, (page_number, page_text) in enumerate(parsed_pages):
        clean_parts.append(page_text)
        if page_text:
            page_spans.append((cursor, cursor + len(page_text), page_number))
            cursor += len(page_text)
        if index < total_pages - 1:
            clean_parts.append("\n")
            cursor += 1
    return "".join(clean_parts), page_spans


def _candidate_chunk_texts(chunk_text: str) -> tuple[str, ...]:
    """Return exact and normalized chunk-text variants for page-range lookup."""
    candidates = [chunk_text]
    normalized_newlines = re.sub(r"[ \t]+\n", "\n", chunk_text)
    if normalized_newlines not in candidates:
        candidates.append(normalized_newlines)
    collapsed_blank_lines = re.sub(r"\n{3,}", "\n\n", normalized_newlines)
    if collapsed_blank_lines not in candidates:
        candidates.append(collapsed_blank_lines)
    return tuple(candidates)


def shared_token_overlap(
    previous_tokens: list[int],
    current_tokens: list[int],
    max_overlap: int,
) -> int:
    """Return the actual token overlap between consecutive textbook chunks."""
    if max_overlap <= 0:
        return 0

    max_candidate = min(max_overlap, len(previous_tokens), len(current_tokens))
    for overlap_size in range(max_candidate, 0, -1):
        if previous_tokens[-overlap_size:] == current_tokens[:overlap_size]:
            return overlap_size
    return 0


def legacy_source_path(document: ExtractedDocument) -> str:
    """Return the old-style relative source path used in chunk metadata."""
    return f"data/output/{document.source}/text/{document.document_id}.txt"


def chunk_source_documents(
    documents: list[ExtractedDocument],
    config: ChunkingConfig | None = None,
) -> list[ChunkRecord]:
    """Chunk all extracted documents for one source corpus."""
    chunk_records: list[ChunkRecord] = []
    for document in documents:
        chunk_records.extend(chunk_document(document, config=config))
    return chunk_records


def chunk_output_path(source_paths: SourcePaths, document: ExtractedDocument) -> Path:
    """Return the chunk output path for one extracted document."""
    return source_paths.chunks_dir / f"{document.document_id}.json"


def build_chunk_document_payload(
    document: ExtractedDocument,
    chunks: list[ChunkRecord],
    config: ChunkingConfig,
) -> dict[str, object]:
    """Build the per-document chunk payload written to disk."""
    document_metadata: dict[str, object] = {
        "source_path": legacy_source_path(document),
        "document_id": document.document_id,
        "doc_id": generate_doc_id(document.source, document.document_id),
        "source": document.source,
        "mode": config.mode,
        "total_chunks": len(chunks),
    }
    if config.mode == "per-page":
        document_metadata["page_count"] = document.page_count
        document_metadata["chunk_unit"] = "page"
    elif config.mode == "textbook-markdown":
        document_metadata["max_tokens"] = config.chunk_size
        document_metadata["overlap"] = config.overlap
        document_metadata["encoding"] = TEXTBOOK_ENCODING
        document_metadata["strategy"] = "textbook"
    else:
        document_metadata["chunk_size"] = config.chunk_size
        document_metadata["overlap"] = config.overlap
        document_metadata["chunk_unit"] = "token"

    return {
        "document_metadata": document_metadata,
        "chunks": [chunk.to_payload() for chunk in chunks],
    }


def write_chunk_document(
    source_paths: SourcePaths,
    document: ExtractedDocument,
    chunks: list[ChunkRecord],
    config: ChunkingConfig | None = None,
) -> Path:
    """Write one document's chunks to a deterministic JSON file."""
    chunking_config = config or default_chunking_config(document.source)
    output_path = chunk_output_path(source_paths, document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_chunk_document_payload(document, chunks, chunking_config)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_chunk_source_documents(
    source_paths: SourcePaths,
    documents: list[ExtractedDocument],
    config: ChunkingConfig | None = None,
    show_progress: bool = False,
) -> list[Path]:
    """Chunk and write all extracted documents for one source corpus."""
    output_paths: list[Path] = []
    document_items: list[ExtractedDocument] | object = sorted(
        documents,
        key=lambda item: item.document_id,
    )
    if show_progress:
        document_items = tqdm(
            document_items,
            desc="Chunking documents",
            unit="document",
        )
    for document in document_items:
        chunking_config = config or default_chunking_config(document.source)
        chunks = chunk_document(
            document,
            config=chunking_config,
            show_progress=show_progress,
        )
        chunk_output_file = write_chunk_document(
            source_paths=source_paths,
            document=document,
            chunks=chunks,
            config=chunking_config,
        )
        output_paths.append(chunk_output_file)
    return output_paths
