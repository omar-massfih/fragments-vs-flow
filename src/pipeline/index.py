"""Indexing step helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.models import ChunkRecord
from src.paths import SourcePaths
from src.providers.embeddings import EmbeddingProvider
from src.providers.vector_db import VectorStore


def chunk_collection_name(source_paths: SourcePaths) -> str:
    """Return the vector collection name for one source corpus."""
    return source_paths.source


def load_chunk_document(chunk_path: Path) -> list[ChunkRecord]:
    """Load one per-document chunk JSON file from disk."""
    payload = json.loads(chunk_path.read_text(encoding="utf-8"))
    return [
        ChunkRecord(
            source=chunk["source"],
            document_id=chunk["document_id"],
            text=chunk["text"],
            page_start=chunk["page_start"],
            page_end=chunk["page_end"],
            token_start=chunk["token_start"],
            token_end=chunk["token_end"],
            chunk_index=chunk["chunk_index"],
            chunk_id=chunk["chunk_id"],
            total_chunks=chunk["total_chunks"],
            headers=chunk["headers"],
            doc_id=chunk.get("doc_id"),
            source_path=str(chunk.get("source_path", "")),
            window_size=int(chunk.get("window_size", 0)),
            overlap=int(chunk.get("overlap", 0)),
            boundary_snapped=bool(chunk.get("boundary_snapped", False)),
        )
        for chunk in payload["chunks"]
    ]


def load_source_chunks(source_paths: SourcePaths) -> list[ChunkRecord]:
    """Load all chunk records for one source corpus from disk."""
    chunk_records: list[ChunkRecord] = []
    for chunk_path in sorted(source_paths.chunks_dir.glob("*.json")):
        if chunk_path.is_file():
            chunk_records.extend(load_chunk_document(chunk_path))
    return chunk_records


def build_index_payloads(chunks: list[ChunkRecord]) -> list[dict[str, object]]:
    """Build vector-store payloads for chunk records."""
    return [chunk.to_payload() for chunk in chunks]


def index_chunks(
    chunks: list[ChunkRecord],
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    collection_name: str,
) -> None:
    """Embed chunk text and upsert the results into the vector store."""
    if not chunks:
        return

    embeddings = embedding_provider.embed_texts([chunk.text for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise ValueError("embedding count must match chunk count")

    vector_store.upsert(
        collection_name=collection_name,
        ids=[chunk.chunk_id for chunk in chunks],
        vectors=embeddings,
        payloads=build_index_payloads(chunks),
    )
