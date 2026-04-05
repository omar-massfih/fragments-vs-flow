"""Re-narrativization pipeline for derived slide corpora."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from src.derived_sources import RENARRATED_SOURCE
from src.models import ChunkRecord
from src.paths import SourcePaths, WorkspacePaths
from src.pipeline.evaluate import (
    SampleEvaluator,
    evaluate_generation_results,
    write_evaluation_results,
)
from src.pipeline.generate import (
    load_retrieval_results as load_saved_retrieval_results,
    run_generation_queries,
    write_generation_results,
)
from src.pipeline.index import chunk_collection_name, index_chunks, load_source_chunks
from src.pipeline.ir_metrics import (
    compute_ir_metrics,
    ir_metrics_results_dir,
    write_ir_metrics_results,
)
from src.pipeline.retrieve import run_retrieval_queries, write_retrieval_results
from src.providers.embeddings import EmbeddingProvider, create_embedding_provider
from src.providers.llm import LlmProvider, create_llm_provider
from src.providers.vector_db import VectorStore, create_vector_store
from src.services.renarrator import Renarrator, create_renarrator
from tqdm import tqdm

DEFAULT_RENARRATIVE_K_VALUES = [1, 3, 5, 7, 10]
RENARRATIVE_START_STAGES = ("rewrite", "retrieve", "generate", "evaluate", "ir-metrics")


@dataclass(frozen=True)
class RenarratedDocumentSummary:
    """Summary of one rewritten chunk document."""

    document_id: str
    input_path: Path
    output_path: Path
    chunk_count: int
    fallback_count: int


def build_renarrated_source_paths(workspace_paths: WorkspacePaths) -> SourcePaths:
    """Return path bundle for the derived renarrativized slide corpus."""
    source_output_dir = workspace_paths.output_root / RENARRATED_SOURCE
    return SourcePaths(
        source=RENARRATED_SOURCE,
        input_dir=workspace_paths.input_root / "slides",
        output_dir=source_output_dir,
        text_dir=source_output_dir / "text",
        chunks_dir=source_output_dir / "chunks",
        generation_dir=workspace_paths.output_root / "generation" / RENARRATED_SOURCE,
        retrieval_dir=workspace_paths.output_root / "retrieval" / RENARRATED_SOURCE,
        milvus_dir=workspace_paths.milvus_root / RENARRATED_SOURCE,
        results_dir=workspace_paths.results_root / RENARRATED_SOURCE,
    )


def run_renarrative_pipeline(
    *,
    workspace_paths: WorkspacePaths,
    model_id: str | None = None,
    k_values: list[int] | None = None,
    start_at: str = "rewrite",
    renarrator: Renarrator | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
    generation_llm_provider: LlmProvider | None = None,
    evaluator: SampleEvaluator | None = None,
) -> dict[str, object]:
    """Rewrite slide chunks into narrative form and run the downstream pipeline."""
    if start_at not in RENARRATIVE_START_STAGES:
        raise ValueError(
            "start_at must be one of "
            f"{', '.join(RENARRATIVE_START_STAGES)}",
        )

    k_values = list(k_values or DEFAULT_RENARRATIVE_K_VALUES)
    original_source_paths = workspace_paths.for_source("slides")
    renarrated_source_paths = build_renarrated_source_paths(workspace_paths)
    original_chunk_paths = sorted(original_source_paths.chunks_dir.glob("*.json"))
    if not original_chunk_paths:
        raise FileNotFoundError(
            f"Missing slide chunk files under {original_source_paths.chunks_dir}. "
            "Run chunk --source slides first.",
        )
    if not workspace_paths.test_set_with_labels_path.exists():
        raise FileNotFoundError(
            f"Missing labeled test set: {workspace_paths.test_set_with_labels_path}",
        )

    test_set_payload = json.loads(
        workspace_paths.test_set_with_labels_path.read_text(encoding="utf-8"),
    )
    if not isinstance(test_set_payload, list):
        raise ValueError("test_set_with_labels.json must contain a JSON list")

    stage_index = RENARRATIVE_START_STAGES.index(start_at)
    start_stage = lambda stage_name: RENARRATIVE_START_STAGES.index(stage_name) >= stage_index
    renarrator_model_id = model_id or ""

    document_summaries: list[RenarratedDocumentSummary] = []
    if start_at == "rewrite":
        reset_renarrative_outputs(renarrated_source_paths)
        active_renarrator = renarrator or create_renarrator(model_id=model_id)
        renarrator_model_id = active_renarrator.model_id
        document_summaries = rewrite_slide_chunk_documents(
            input_chunk_paths=original_chunk_paths,
            output_source_paths=renarrated_source_paths,
            renarrator=active_renarrator,
            show_progress=True,
        )

        rewritten_chunks = load_source_chunks(renarrated_source_paths)
        active_embedding_provider = embedding_provider or create_embedding_provider(
            show_progress=True,
        )
        active_vector_store = vector_store or create_vector_store(renarrated_source_paths.milvus_dir)
        collection_name = chunk_collection_name(renarrated_source_paths)
        index_chunks(
            chunks=rewritten_chunks,
            embedding_provider=active_embedding_provider,
            vector_store=active_vector_store,
            collection_name=collection_name,
        )
    else:
        collection_name = chunk_collection_name(renarrated_source_paths)
        active_embedding_provider = None
        active_vector_store = None
        if start_stage("retrieve"):
            active_embedding_provider = embedding_provider or create_embedding_provider(
                show_progress=True,
            )
            active_vector_store = vector_store or create_vector_store(renarrated_source_paths.milvus_dir)

    retrieval_paths: dict[str, str] = {}
    generation_paths: dict[str, str] = {}
    evaluation_paths: dict[str, str] = {}
    evaluation_summary_paths: dict[str, str] = {}
    evaluation_summaries: dict[str, dict[str, object]] = {}
    active_generation_llm = None
    for top_k in k_values:
        if start_stage("retrieve"):
            if active_embedding_provider is None or active_vector_store is None:
                raise RuntimeError("retrieve stage requires embedding provider and vector store")
            retrieval_results = run_retrieval_queries(
                source_paths=renarrated_source_paths,
                queries=test_set_payload,
                embedding_provider=active_embedding_provider,
                vector_store=active_vector_store,
                top_k=top_k,
                show_progress=True,
            )
            retrieval_path = write_retrieval_results(
                source_paths=renarrated_source_paths,
                results=retrieval_results,
                top_k=top_k,
                collection_name=collection_name,
                embedding_deployment=getattr(
                    active_embedding_provider,
                    "model_id",
                    "",
                ),
                embedding_provider=getattr(
                    active_embedding_provider,
                    "provider_name",
                    None,
                ),
            )
        else:
            retrieval_results = load_saved_retrieval_results(
                source_paths=renarrated_source_paths,
                top_k=top_k,
            )
            retrieval_path = renarrated_source_paths.retrieval_dir / f"retrieval_with_k_{top_k}.json"
        retrieval_paths[str(top_k)] = str(retrieval_path)

        if start_stage("generate"):
            if active_generation_llm is None:
                active_generation_llm = generation_llm_provider or create_llm_provider(
                )
            generation_results = run_generation_queries(
                source_paths=renarrated_source_paths,
                retrieval_results=retrieval_results,
                llm_provider=active_generation_llm,
                show_progress=True,
            )
            generation_path = write_generation_results(
                source_paths=renarrated_source_paths,
                results=generation_results,
                top_k=top_k,
                model_id=getattr(
                    active_generation_llm,
                    "model_id",
                    "",
                ),
                provider=getattr(
                    active_generation_llm,
                    "provider_name",
                    None,
                ),
            )
        else:
            generation_path = renarrated_source_paths.generation_dir / f"answers_with_k_{top_k}.json"
        generation_paths[str(top_k)] = str(generation_path)

        if start_stage("evaluate"):
            evaluation_payload = evaluate_generation_results(
                source_paths=renarrated_source_paths,
                top_k=top_k,
                labeled_test_set_path=workspace_paths.test_set_with_labels_path,
                evaluator=evaluator,
                show_progress=True,
            )
            evaluation_path, _summary_path = write_evaluation_results(
                source_paths=renarrated_source_paths,
                top_k=top_k,
                evaluation_payload=evaluation_payload,
            )
            per_k_summary_path = write_renarrative_evaluation_summary(
                source_paths=renarrated_source_paths,
                top_k=top_k,
                evaluation_summary=evaluation_payload["summary"],
            )
            evaluation_paths[str(top_k)] = str(evaluation_path)
            evaluation_summary_paths[str(top_k)] = str(per_k_summary_path)
            evaluation_summaries[str(top_k)] = dict(evaluation_payload["summary"])

    overall_evaluation_summary_path = (
        write_renarrative_overall_summary(
            source_paths=renarrated_source_paths,
            k_values=k_values,
            per_k_summaries=evaluation_summaries,
        )
        if start_stage("evaluate")
        else renarrated_source_paths.results_dir / "SUMMARY.json"
    )

    if start_stage("ir-metrics"):
        ir_metrics_payload = compute_ir_metrics(
            source_paths=renarrated_source_paths,
            labeled_test_set_path=workspace_paths.test_set_with_labels_path,
            k_values=k_values,
            label_source="slides",
        )
        ir_json_path, ir_csv_path, ir_summary_path = write_ir_metrics_results(
            source_paths=renarrated_source_paths,
            ir_metrics_payload=ir_metrics_payload,
        )
    else:
        ir_metrics_dir = ir_metrics_results_dir(renarrated_source_paths)
        ir_json_path = ir_metrics_dir / "ir_metrics.json"
        ir_csv_path = ir_metrics_dir / "ir_metrics.csv"
        ir_summary_path = ir_metrics_dir / "SUMMARY.json"

    fallback_count = sum(summary.fallback_count for summary in document_summaries)
    rewritten_count = sum(summary.chunk_count for summary in document_summaries) - fallback_count
    summary_vector_store = active_vector_store
    if summary_vector_store is None and renarrated_source_paths.milvus_dir.exists():
        summary_vector_store = create_vector_store(renarrated_source_paths.milvus_dir)
    record_count = (
        summary_vector_store.record_count(collection_name)
        if summary_vector_store is not None
        else 0
    )
    return {
        "course": workspace_paths.course,
        "data_root": str(workspace_paths.data_root),
        "source": RENARRATED_SOURCE,
        "model_id": renarrator_model_id,
        "start_at": start_at,
        "chunk_input_dir": str(original_source_paths.chunks_dir),
        "chunk_output_dir": str(renarrated_source_paths.chunks_dir),
        "rewritten_chunk_count": rewritten_count,
        "fallback_original_text_count": fallback_count,
        "total_chunk_count": sum(summary.chunk_count for summary in document_summaries),
        "documents": [
            {
                "document_id": summary.document_id,
                "input_path": str(summary.input_path),
                "output_path": str(summary.output_path),
                "chunk_count": summary.chunk_count,
                "fallback_count": summary.fallback_count,
            }
            for summary in document_summaries
        ],
        "milvus_dir": str(renarrated_source_paths.milvus_dir),
        "collection_name": collection_name,
        "record_count": record_count,
        "k_values": k_values,
        "retrieval_paths": retrieval_paths,
        "generation_paths": generation_paths,
        "evaluation_paths": evaluation_paths,
        "evaluation_summary_paths": evaluation_summary_paths,
        "evaluation_summary_path": str(overall_evaluation_summary_path),
        "ir_metrics_json": str(ir_json_path),
        "ir_metrics_csv": str(ir_csv_path),
        "ir_metrics_summary": str(ir_summary_path),
    }


def rewrite_slide_chunk_documents(
    *,
    input_chunk_paths: list[Path],
    output_source_paths: SourcePaths,
    renarrator: Renarrator,
    show_progress: bool = True,
) -> list[RenarratedDocumentSummary]:
    """Rewrite all original slide chunk documents into a derived corpus."""
    summaries: list[RenarratedDocumentSummary] = []
    output_source_paths.chunks_dir.mkdir(parents=True, exist_ok=True)
    document_items: list[Path] | object = input_chunk_paths
    if show_progress:
        document_items = tqdm(
            input_chunk_paths,
            desc="Re-narrativizing documents",
            unit="document",
        )
    for input_chunk_path in document_items:
        payload = json.loads(input_chunk_path.read_text(encoding="utf-8"))
        chunks = payload.get("chunks", [])
        if not isinstance(chunks, list):
            raise ValueError(f"Chunk payload must contain a list under 'chunks': {input_chunk_path}")

        fallback_count = 0
        rewritten_chunks: list[dict[str, object] | None] = [None] * len(chunks)
        chunk_items = list(enumerate(chunks))
        chunk_progress = (
            tqdm(
                total=len(chunk_items),
                desc=f"Chunks {input_chunk_path.stem}",
                unit="chunk",
                leave=False,
            )
            if show_progress
            else None
        )
        try:
            for index, chunk in chunk_items:
                chunk_index, rewritten_chunk, used_fallback = _renarrate_chunk_payload(
                    renarrator,
                    chunk,
                    index,
                )
                fallback_count += int(used_fallback)
                rewritten_chunks[chunk_index] = rewritten_chunk
                if chunk_progress is not None:
                    chunk_progress.update(1)
        finally:
            if chunk_progress is not None:
                chunk_progress.close()

        document_metadata = dict(payload.get("document_metadata", {}))
        document_metadata["source"] = RENARRATED_SOURCE
        document_metadata["derived_from_source"] = "slides"
        document_metadata["renarrated"] = True

        output_path = output_source_paths.chunks_dir / input_chunk_path.name
        output_path.write_text(
            json.dumps(
                {
                    "document_metadata": document_metadata,
                    "chunks": [chunk for chunk in rewritten_chunks if chunk is not None],
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )
        summaries.append(
            RenarratedDocumentSummary(
                document_id=str(document_metadata.get("document_id", input_chunk_path.stem)),
                input_path=input_chunk_path,
                output_path=output_path,
                chunk_count=len(rewritten_chunks),
                fallback_count=fallback_count,
            ),
        )
    return summaries


def _renarrate_chunk_payload(
    renarrator: Renarrator,
    chunk: object,
    index: int,
) -> tuple[int, dict[str, object], bool]:
    """Rewrite one chunk payload while preserving stable ordering metadata."""
    if not isinstance(chunk, dict):
        raise ValueError("Chunk rows must be JSON objects")

    result = renarrator.renarrate_chunk(
        chunk_text=str(chunk.get("text", "")),
        metadata=chunk,
    )
    rewritten_chunk = dict(chunk)
    rewritten_chunk["source"] = RENARRATED_SOURCE
    rewritten_chunk["text"] = result.text
    return index, rewritten_chunk, result.used_fallback


def load_renarrated_chunks(source_paths: SourcePaths) -> list[ChunkRecord]:
    """Load rewritten chunk records for the derived corpus."""
    return load_source_chunks(source_paths)


def reset_renarrated_milvus_storage(source_paths: SourcePaths) -> None:
    """Remove the previous derived Milvus Lite database before reindexing."""
    db_file_path = source_paths.milvus_dir / "milvus_lite.db"
    if db_file_path.exists():
        db_file_path.unlink()


def reset_renarrative_outputs(source_paths: SourcePaths) -> None:
    """Clear derived-corpus artifacts before a fresh renarrative run."""
    for directory in (
        source_paths.chunks_dir,
        source_paths.retrieval_dir,
        source_paths.generation_dir,
        source_paths.results_dir,
        ir_metrics_results_dir(source_paths),
    ):
        if directory.exists():
            shutil.rmtree(directory)
    reset_renarrated_milvus_storage(source_paths)


def renarrative_evaluation_summary_path(source_paths: SourcePaths, top_k: int) -> Path:
    """Return the per-k evaluation summary path for renarrative runs."""
    return source_paths.results_dir / f"SUMMARY_with_k_{top_k}.json"


def write_renarrative_evaluation_summary(
    source_paths: SourcePaths,
    top_k: int,
    evaluation_summary: dict[str, object],
) -> Path:
    """Write one dedicated per-k evaluation summary file."""
    summary_path = renarrative_evaluation_summary_path(source_paths, top_k)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(evaluation_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary_path


def write_renarrative_overall_summary(
    source_paths: SourcePaths,
    k_values: list[int],
    per_k_summaries: dict[str, dict[str, object]],
) -> Path:
    """Write the shared evaluation summary as one multi-k renarrative summary."""
    summary_path = source_paths.results_dir / "SUMMARY.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source_paths.source,
        "k_values": k_values,
        "per_k": per_k_summaries,
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary_path
