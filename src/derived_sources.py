"""Shared helpers for derived corpus namespaces."""

from __future__ import annotations


RENARRATED_SOURCE = "slides_renarrated"
TEXTBOOKS_FRAGMENTED_SOURCE = "textbooks_fragmented"

DERIVED_SOURCE_LABEL_SOURCE: dict[str, str] = {
    RENARRATED_SOURCE: "slides",
    TEXTBOOKS_FRAGMENTED_SOURCE: "textbooks",
}


def resolve_effective_label_source(source: str) -> str:
    """Resolve the test-set label namespace for one source."""
    normalized_source = source.strip().lower()
    return DERIVED_SOURCE_LABEL_SOURCE.get(normalized_source, normalized_source)


def resolve_label_fields(source: str) -> tuple[str, str]:
    """Resolve labeled-test-set field names for one source."""
    effective_source = resolve_effective_label_source(source)
    if effective_source == "slides":
        return "slides_labels", "slide_chunk_ids"
    if effective_source == "textbooks":
        return "textbooks_labels", "textbook_chunk_ids"
    raise ValueError(f"Unsupported source for label resolution: {source}")
