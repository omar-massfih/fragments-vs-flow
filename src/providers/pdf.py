"""PDF extraction provider interfaces."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Protocol
import warnings

from src.models import ExtractedPage


class PdfExtractor(Protocol):
    """Interface for extracting ordered text from one PDF file."""

    def extract_pages(
        self,
        pdf_path: Path,
        show_progress: bool = False,
    ) -> list[ExtractedPage]:
        """Extract text from a PDF file page by page."""


class PdfPlumberExtractor:
    """PDF extractor backed by pdfplumber."""

    def extract_pages(
        self,
        pdf_path: Path,
        show_progress: bool = False,
    ) -> list[ExtractedPage]:
        """Extract text from one PDF file page by page."""
        import pdfplumber
        from tqdm import tqdm

        pages: list[ExtractedPage] = []
        with pdfplumber.open(pdf_path) as pdf:
            page_iterator = enumerate(pdf.pages, start=1)
            if show_progress:
                page_iterator = enumerate(
                    tqdm(
                        pdf.pages,
                        total=len(pdf.pages),
                        desc=f"Pages {pdf_path.stem}",
                        unit="page",
                        leave=False,
                    ),
                    start=1,
                )
            for index, page in page_iterator:
                extracted_text = page.extract_text() or ""
                pages.append(
                    ExtractedPage(
                        page_number=index,
                        text=extracted_text.strip(),
                    ),
                )
        return pages


class DoclingPageExtractor:
    """Shared Docling per-page extractor with source-specific rendering."""

    def __init__(self, quiet: bool = True) -> None:
        """Initialize the Docling extractor."""
        from docling.document_converter import DocumentConverter

        self.converter = DocumentConverter()
        self.quiet = quiet

        if self.quiet:
            self._configure_quiet_mode()

    def extract_pages(
        self,
        pdf_path: Path,
        show_progress: bool = False,
    ) -> list[ExtractedPage]:
        """Extract slide text one PDF page at a time."""
        from tqdm import tqdm

        total_pages = self._get_page_count(pdf_path)
        if total_pages == 0:
            return []

        page_numbers: object = range(1, total_pages + 1)
        if show_progress:
            page_numbers = tqdm(
                page_numbers,
                desc=f"Pages {pdf_path.stem}",
                unit="page",
                leave=False,
            )

        pages: list[ExtractedPage] = []
        for page_number in page_numbers:
            result = self._convert_quietly(pdf_path, (int(page_number), int(page_number)))
            page_text = self._render_page_text(result.document)
            pages.append(ExtractedPage(page_number=int(page_number), text=page_text))
        return pages

    def _configure_quiet_mode(self) -> None:
        """Suppress low-signal warnings and logs from Docling dependencies."""
        warnings.filterwarnings(
            "ignore",
            message=r".*Parameter `strict_text` has been deprecated.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Parameter `strict_text` has been deprecated.*",
            category=UserWarning,
        )
        logging.getLogger().setLevel(logging.ERROR)
        for name in ("docling", "pypdf", "pdfminer", "transformers"):
            logging.getLogger(name).setLevel(logging.ERROR)

    def _render_page_text(self, document: object) -> str:
        """Convert one Docling document page into extracted text."""
        raise NotImplementedError

    def _get_page_count(self, file_path: Path) -> int:
        """Read the PDF page count for per-page extraction."""
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(file_path))
            return len(reader.pages)
        except Exception as exc:
            raise ValueError(f"Unable to read PDF page count: {file_path}") from exc

    def _convert_quietly(self, path: Path, page_range: tuple[int, int]) -> object:
        """Run one quiet Docling conversion on a single page."""
        if not self.quiet:
            return self.converter.convert(str(path), page_range=page_range)

        from contextlib import redirect_stderr, redirect_stdout

        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            return self.converter.convert(str(path), page_range=page_range)


class SlideDoclingPageExtractor(DoclingPageExtractor):
    """Docling per-page extractor for slide PDFs."""

    def _render_page_text(self, document: object) -> str:
        """Render slide content using Docling plain-text export."""
        if hasattr(document, "export_to_text"):
            return str(document.export_to_text())
        raise AttributeError("Docling document must support text export")


class TextbookDoclingPageExtractor(DoclingPageExtractor):
    """Docling per-page markdown extractor for textbook PDFs."""

    def _render_page_text(self, document: object) -> str:
        """Use Docling markdown only for textbook extraction."""
        if hasattr(document, "export_to_markdown"):
            return str(document.export_to_markdown())
        raise AttributeError("Docling document must support markdown export")


def create_pdf_extractor(source: str | None = None) -> PdfExtractor:
    """Create the default PDF extractor backend."""
    if source == "textbooks":
        return TextbookDoclingPageExtractor()
    if source == "slides":
        return SlideDoclingPageExtractor()
    return PdfPlumberExtractor()
