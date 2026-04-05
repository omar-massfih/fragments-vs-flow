"""Retrieval step helpers."""

from __future__ import annotations

from datetime import datetime, UTC
import json
from pathlib import Path

from src.models import RetrievedChunk
from src.paths import SourcePaths
from src.pipeline.index import chunk_collection_name
from src.providers.embeddings import EmbeddingProvider
from src.providers.vector_db import VectorStore
from tqdm import tqdm


def retrieve_source_chunks(
    source_paths: SourcePaths,
    query_text: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    top_k: int = 3,
) -> list[RetrievedChunk]:
    """Retrieve ranked chunks for one source corpus.

    Args:
        source_paths: Resolved source workspace paths.
        query_text: Query text to search for.
        embedding_provider: Provider used to embed the query text.
        vector_store: Vector backend used to search the indexed collection.
        top_k: Number of ranked chunks to return.

    Returns:
        A ranked list of retrieved chunk records.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    query = query_text.strip()
    if not query:
        raise ValueError("query_text cannot be empty")

    query_embeddings = embedding_provider.embed_texts([query])
    if len(query_embeddings) != 1:
        raise ValueError("query embedding count must be exactly one")

    hits = vector_store.search(
        collection_name=chunk_collection_name(source_paths),
        query_vector=query_embeddings[0],
        top_k=top_k,
    )
    return build_retrieved_chunks(hits)


def build_retrieved_chunks(hits: list[dict[str, object]]) -> list[RetrievedChunk]:
    """Convert normalized vector-store hits into retrieved chunk records."""
    return [
        RetrievedChunk(
            source=str(hit["source"]),
            document_id=str(hit["document_id"]),
            text=str(hit["text"]),
            page_start=_optional_int(hit.get("page_start")),
            page_end=_optional_int(hit.get("page_end")),
            token_start=int(hit["token_start"]),
            token_end=int(hit["token_end"]),
            chunk_index=int(hit["chunk_index"]),
            chunk_id=str(hit["chunk_id"]),
            total_chunks=int(hit["total_chunks"]),
            headers=dict(hit.get("headers", {})),
            score=float(hit.get("score", 0.0)),
            rank=rank,
            doc_id=str(hit.get("doc_id", "")),
            source_path=str(hit.get("source_path", "")),
            window_size=int(hit.get("window_size", 0)),
            overlap=int(hit.get("overlap", 0)),
            boundary_snapped=bool(hit.get("boundary_snapped", False)),
        )
        for rank, hit in enumerate(hits, start=1)
    ]


def build_retrieval_chunk_payloads(
    chunks: list[RetrievedChunk],
) -> list[dict[str, object]]:
    """Convert retrieved chunks into the current retrieval-only export shape."""
    return [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "doc_id": chunk.doc_id or chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "distance": chunk.score,
            "source_path": chunk.source_path,
            "headers": chunk.headers,
            "window_size": chunk.window_size,
            "overlap": chunk.overlap,
            "token_start": chunk.token_start,
            "token_end": chunk.token_end,
            "boundary_snapped": chunk.boundary_snapped,
        }
        for chunk in chunks
    ]


def run_retrieval_queries(
    source_paths: SourcePaths,
    queries: list[dict[str, object]],
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    top_k: int,
    show_progress: bool = False,
) -> list[dict[str, object]]:
    """Run retrieval for a list of test queries and normalize the results."""
    results: list[dict[str, object]] = []
    query_items: list[dict[str, object]] | object = queries
    if show_progress:
        query_items = tqdm(
            queries,
            desc="Retrieving queries",
            unit="query",
        )
    for query_item in query_items:
        query_text = str(query_item["question"])
        retrieved_chunks = retrieve_source_chunks(
            source_paths=source_paths,
            query_text=query_text,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            top_k=top_k,
        )
        results.append(
            {
                "query": query_text,
                "chunks": build_retrieval_chunk_payloads(retrieved_chunks),
                "status": "RETRIEVED",
                "question_id": str(query_item["question_id"]),
            },
        )
    return results


def retrieval_output_path(source_paths: SourcePaths, top_k: int) -> Path:
    """Return the output path for retrieval-only JSON exports."""
    return source_paths.retrieval_dir / f"retrieval_with_k_{top_k}.json"


def build_retrieval_export_payload(
    source_paths: SourcePaths,
    results: list[dict[str, object]],
    top_k: int,
    collection_name: str,
    embedding_deployment: str,
    embedding_provider: str | None = None,
    timestamp: str | None = None,
) -> dict[str, object]:
    """Build the current top-level retrieval-only export payload."""
    successful = sum(1 for result in results if result.get("status") == "RETRIEVED")
    failed = len(results) - successful
    return {
        "metadata": {
            "timestamp": timestamp or datetime.now(UTC).replace(tzinfo=None).isoformat(),
            "embedding_deployment": embedding_deployment,
            "embedding_model": embedding_deployment,
            "embedding_provider": embedding_provider,
            "top_k": top_k,
            "collection_name": collection_name,
            "db_path": str(source_paths.milvus_dir),
            "retrieve_only": True,
        },
        "status": "SUCCESS",
        "results": results,
        "summary": {
            "total_queries": len(results),
            "successful": successful,
            "failed": failed,
        },
    }


def write_retrieval_results(
    source_paths: SourcePaths,
    results: list[dict[str, object]],
    top_k: int,
    collection_name: str,
    embedding_deployment: str,
    embedding_provider: str | None = None,
    timestamp: str | None = None,
) -> Path:
    """Write retrieval-only results in the current JSON export shape."""
    output_path = retrieval_output_path(source_paths, top_k)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_retrieval_export_payload(
        source_paths=source_paths,
        results=results,
        top_k=top_k,
        collection_name=collection_name,
        embedding_deployment=embedding_deployment,
        embedding_provider=embedding_provider,
        timestamp=timestamp,
    )
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _optional_int(value: object) -> int | None:
    """Normalize nullable numeric fields returned by the vector store."""
    if value is None:
        return None
    return int(value)
