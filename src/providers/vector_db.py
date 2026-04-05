"""Vector database interfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol


class VectorStore(Protocol):
    """Interface for storing chunk embeddings in a vector collection."""

    def upsert(
        self,
        collection_name: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, object]],
    ) -> None:
        """Insert or update vectors and payloads."""

    def record_count(self, collection_name: str) -> int:
        """Return the number of stored records for one collection."""

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[dict[str, object]]:
        """Return ranked vector-search hits for one collection."""


class MilvusLiteVectorStore:
    """Milvus Lite-backed vector store for source-specific collections."""

    def __init__(self, storage_path: Path, client: object | None = None) -> None:
        """Create the Milvus Lite client for one source-specific storage path."""
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_file_path = self.storage_path / "milvus_lite.db"
        self.client = client or self._create_client()

    def _create_client(self) -> object:
        """Create the default Milvus Lite client."""
        from pymilvus import MilvusClient

        return MilvusClient(str(self.db_file_path))

    def upsert(
        self,
        collection_name: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, object]],
    ) -> None:
        """Insert or update vectors and payloads in a Milvus Lite collection."""
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError("ids, vectors, and payloads must have the same length")
        if not ids:
            return

        embedding_dim = len(vectors[0])
        self._ensure_collection(collection_name, embedding_dim)
        records = [
            self._build_record(record_id, vector, payload)
            for record_id, vector, payload in zip(ids, vectors, payloads, strict=True)
        ]
        self.client.upsert(collection_name=collection_name, data=records)

    def record_count(self, collection_name: str) -> int:
        """Return the row count reported by Milvus for one collection."""
        stats = self.client.get_collection_stats(collection_name)
        return int(stats.get("row_count", 0))

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[dict[str, object]]:
        """Search one Milvus collection and normalize the hit payloads."""
        raw_results = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=top_k,
            anns_field="embedding",
            output_fields=[
                "chunk_id",
                "source",
                "document_id",
                "text",
                "page_start",
                "page_end",
                "token_start",
                "token_end",
                "chunk_index",
                "total_chunks",
                "headers",
                "doc_id",
                "source_path",
                "window_size",
                "overlap",
                "boundary_snapped",
            ],
            search_params={"metric_type": "COSINE"},
        )
        hits = raw_results[0] if raw_results else []
        return [self._normalize_search_hit(hit) for hit in hits]

    def _ensure_collection(self, collection_name: str, embedding_dim: int) -> None:
        """Create the collection and vector index on first use."""
        if self.client.has_collection(collection_name):
            return

        from pymilvus import CollectionSchema, DataType, FieldSchema
        from pymilvus.milvus_client.index import IndexParams

        fields = [
            FieldSchema(
                name="chunk_id",
                dtype=DataType.VARCHAR,
                max_length=255,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="page_start", dtype=DataType.INT64),
            FieldSchema(name="page_end", dtype=DataType.INT64),
            FieldSchema(name="token_start", dtype=DataType.INT64),
            FieldSchema(name="token_end", dtype=DataType.INT64),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="total_chunks", dtype=DataType.INT64),
            FieldSchema(name="headers", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="source_path", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="window_size", dtype=DataType.INT64),
            FieldSchema(name="overlap", dtype=DataType.INT64),
            FieldSchema(name="boundary_snapped", dtype=DataType.BOOL),
        ]
        schema = CollectionSchema(
            fields=fields,
            description="Chunk embeddings for semantic retrieval",
        )
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
        )

        index_params = IndexParams()
        index_params.add_index(
            field_name="embedding",
            index_type="FLAT",
            metric_type="COSINE",
        )
        self.client.create_index(
            collection_name=collection_name,
            index_params=index_params,
        )

    def _build_record(
        self,
        record_id: str,
        vector: list[float],
        payload: dict[str, object],
    ) -> dict[str, object]:
        """Convert one chunk payload into the stored Milvus row shape."""
        return {
            "chunk_id": record_id,
            "embedding": vector,
            "source": str(payload.get("source", "")),
            "document_id": str(payload.get("document_id", "")),
            "text": str(payload.get("text", "")),
            "page_start": self._int_or_sentinel(payload.get("page_start")),
            "page_end": self._int_or_sentinel(payload.get("page_end")),
            "token_start": self._int_or_default(payload.get("token_start")),
            "token_end": self._int_or_default(payload.get("token_end")),
            "chunk_index": self._int_or_default(payload.get("chunk_index")),
            "total_chunks": self._int_or_default(payload.get("total_chunks")),
            "headers": json.dumps(payload.get("headers", {}), sort_keys=True),
            "doc_id": str(payload.get("doc_id", "")),
            "source_path": str(payload.get("source_path", "")),
            "window_size": self._int_or_default(payload.get("window_size")),
            "overlap": self._int_or_default(payload.get("overlap")),
            "boundary_snapped": bool(payload.get("boundary_snapped", False)),
        }

    def _int_or_default(self, value: object, default: int = 0) -> int:
        """Return one integer value with a deterministic fallback."""
        if value is None:
            return default
        return int(value)

    def _int_or_sentinel(self, value: object, sentinel: int = -1) -> int:
        """Return one integer value or a Milvus-safe sentinel for null pages."""
        if value is None:
            return sentinel
        return int(value)

    def _normalize_search_hit(self, hit: dict[str, object]) -> dict[str, object]:
        """Convert one raw Milvus search hit into a stable payload."""
        entity = hit.get("entity", {})
        if not isinstance(entity, dict):
            entity = {}
        chunk_id = entity.get("chunk_id", hit.get("id", ""))
        headers_raw = entity.get("headers", "{}")
        headers = json.loads(headers_raw) if isinstance(headers_raw, str) else {}

        return {
            "chunk_id": str(chunk_id),
            "source": str(entity.get("source", "")),
            "document_id": str(entity.get("document_id", "")),
            "text": str(entity.get("text", "")),
            "page_start": self._sentinel_to_none(entity.get("page_start")),
            "page_end": self._sentinel_to_none(entity.get("page_end")),
            "token_start": self._int_or_default(entity.get("token_start")),
            "token_end": self._int_or_default(entity.get("token_end")),
            "chunk_index": self._int_or_default(entity.get("chunk_index")),
            "total_chunks": self._int_or_default(entity.get("total_chunks")),
            "headers": headers,
            "doc_id": str(entity.get("doc_id", "")),
            "source_path": str(entity.get("source_path", "")),
            "window_size": self._int_or_default(entity.get("window_size")),
            "overlap": self._int_or_default(entity.get("overlap")),
            "boundary_snapped": bool(entity.get("boundary_snapped", False)),
            "score": float(hit.get("distance", hit.get("score", 0.0))),
        }

    def _sentinel_to_none(self, value: object, sentinel: int = -1) -> int | None:
        """Convert Milvus page sentinels back into nullable page values."""
        if value is None:
            return None
        converted = int(value)
        return None if converted == sentinel else converted


class PlaceholderVectorStore:
    """Temporary vector store used for explicit placeholder-only tests."""

    def __init__(self, storage_path: Path) -> None:
        """Store the configured path for later backend wiring."""
        self.storage_path = storage_path

    def upsert(
        self,
        collection_name: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, object]],
    ) -> None:
        """Reject vector writes until the real backend is wired in."""
        raise NotImplementedError(
            (
                "Vector storage is not implemented yet for collection "
                f"'{collection_name}' with {len(ids)} record(s)."
            ),
        )

    def record_count(self, collection_name: str) -> int:
        """Reject collection statistics until the real backend is wired in."""
        raise NotImplementedError(
            f"Vector storage is not implemented yet for '{collection_name}'.",
        )

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[dict[str, object]]:
        """Reject retrieval until the real backend is wired in."""
        raise NotImplementedError(
            (
                "Vector storage is not implemented yet for collection "
                f"'{collection_name}' and top_k={top_k}."
            ),
        )


def create_vector_store(storage_path: Path, client: object | None = None) -> VectorStore:
    """Create the default vector store backend."""
    return MilvusLiteVectorStore(storage_path=storage_path, client=client)
