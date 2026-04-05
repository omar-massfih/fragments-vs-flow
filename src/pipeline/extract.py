"""Extraction step helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.logger import get_logger
from src.models import ExtractedDocument
from src.paths import SourcePaths
from src.providers.pdf import PdfExtractor

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ExtractionWriteResult:
    """Outcome for one extracted source document write."""

    document: ExtractedDocument
    was_written: bool


def extract_document(
    pdf_path: Path,
    output_path: Path,
    source: str,
    extractor: PdfExtractor,
    show_progress: bool = False,
) -> ExtractedDocument:
    """Extract one source PDF into an in-memory document record."""
    pages = extractor.extract_pages(pdf_path, show_progress=show_progress)
    return ExtractedDocument(
        source=source,
        document_id=pdf_path.stem,
        input_path=pdf_path,
        output_path=output_path,
        pages=tuple(pages),
        metadata={
            "file_name": pdf_path.name,
            "file_stem": pdf_path.stem,
            "source": source,
        },
    )


def render_extracted_text(document: ExtractedDocument) -> str:
    """Render extracted pages into the text file format used by later steps."""
    return document.combined_text()


def build_extracted_payload(document: ExtractedDocument) -> dict[str, object]:
    """Build one extracted-document payload for downstream serialization."""
    return document.to_payload()


def discover_source_pdfs(source_paths: SourcePaths) -> list[Path]:
    """Return all PDF inputs for one source in stable name order."""
    return sorted(
        path
        for path in source_paths.input_dir.glob("*.pdf")
        if path.is_file()
    )


def source_text_output_path(source_paths: SourcePaths, pdf_path: Path) -> Path:
    """Return the extracted text output path for one source document."""
    return source_paths.text_dir / f"{pdf_path.stem}.txt"


def write_extracted_document(
    document: ExtractedDocument,
    overwrite: bool = False,
) -> bool:
    """Write one extracted document to disk.

    Args:
        document: Extracted document to write.
        overwrite: Whether to replace an existing extracted text file.

    Returns:
        ``True`` when a file was written, otherwise ``False`` when an existing
        file was kept.
    """
    document.output_path.parent.mkdir(parents=True, exist_ok=True)
    if document.output_path.exists() and not overwrite:
        return False

    document.output_path.write_text(
        render_extracted_text(document),
        encoding="utf-8",
    )
    return True


def extract_source_documents(
    source_paths: SourcePaths,
    extractor: PdfExtractor,
    overwrite: bool = False,
    show_progress: bool = False,
) -> list[ExtractionWriteResult]:
    """Extract and write all PDFs for one workspace source."""
    results: list[ExtractionWriteResult] = []
    pdf_paths = discover_source_pdfs(source_paths)
    LOGGER.info(
        (
            f"Extracting {len(pdf_paths)} PDF file(s) from "
            f"{source_paths.input_dir} -> {source_paths.text_dir}"
        ),
    )
    for index, pdf_path in enumerate(pdf_paths, start=1):
        LOGGER.info(
            f"[{index}/{len(pdf_paths)}] Extracting {pdf_path.name}",
        )
        document = extract_document(
            pdf_path=pdf_path,
            output_path=source_text_output_path(source_paths, pdf_path),
            source=source_paths.source,
            extractor=extractor,
            show_progress=show_progress,
        )
        was_written = write_extracted_document(
            document=document,
            overwrite=overwrite,
        )
        action = "wrote" if was_written else "skipped"
        LOGGER.info(
            f"    {action}: {document.output_path.name} ({document.page_count} page(s))",
        )
        results.append(
            ExtractionWriteResult(document=document, was_written=was_written),
        )
    return results
