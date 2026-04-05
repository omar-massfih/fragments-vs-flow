"""Entry point for the clean pipeline rebuild."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Sequence
import warnings


def configure_runtime_noise_suppression() -> None:
    """Suppress third-party warnings and low-signal runtime logs."""
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API\..*",
        category=UserWarning,
        module=r"pymilvus\.client(\..*)?",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Deprecated call to `pkg_resources\.declare_namespace.*",
        category=DeprecationWarning,
    )


configure_runtime_noise_suppression()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.logger import emit_json, get_logger
from src.pipeline.chunk import (
    ChunkingConfig,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    VALID_CHUNK_MODES,
    default_chunking_config,
    load_source_documents,
    write_chunk_source_documents,
)
from src.pipeline.evaluate import (
    evaluate_generation_results,
    evaluate_oracle_results,
    load_generated_answers,
    load_ground_truth_map,
    write_evaluation_results,
    write_oracle_evaluation_results,
)
from src.pipeline.extract import extract_source_documents
from src.pipeline.generate import (
    load_retrieval_results,
    run_generation_queries,
    write_generation_results,
)
from src.pipeline.ir_metrics import compute_ir_metrics, write_ir_metrics_results
from src.pipeline.index import chunk_collection_name, index_chunks, load_source_chunks
from src.pipeline.renarrative import RENARRATIVE_START_STAGES, run_renarrative_pipeline
from src.pipeline.retrieve import (
    retrieve_source_chunks,
    run_retrieval_queries,
    write_retrieval_results,
)
from src.pipeline.textbook_fragmentation import (
    TEXTBOOK_FRAGMENTATION_START_STAGES,
    run_textbook_fragmentation_pipeline,
)
from src.paths import WorkspacePaths
from src.providers.embeddings import create_embedding_provider
from src.providers.llm import create_llm_provider
from src.providers.pdf import create_pdf_extractor
from src.providers.vector_db import create_vector_store

logger = get_logger("app")


def _create_pdf_extractor_for_source(source: str) -> object:
    """Create a PDF extractor while tolerating older zero-arg test doubles."""
    try:
        signature = inspect.signature(create_pdf_extractor)
    except (TypeError, ValueError):
        return create_pdf_extractor(source=source)

    parameters = signature.parameters.values()
    accepts_source_keyword = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == "source"
        for parameter in parameters
    )
    if accepts_source_keyword:
        return create_pdf_extractor(source=source)
    return create_pdf_extractor()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Clean rebuild of the thesis RAG pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command")

    commands = (
        "paths",
        "extract",
        "chunk",
        "index",
        "retrieve",
        "generate",
        "evaluate",
        "ir-metrics",
        "fragment-textbooks",
        "renarrative",
        "pipeline",
    )
    for command in commands:
        command_parser = subparsers.add_parser(command, help=f"Run {command} step")
        if command == "paths":
            command_parser.add_argument(
                "--course",
                required=True,
                help="Course workspace to resolve",
            )
            command_parser.add_argument(
                "--source",
                choices=["slides", "textbooks"],
                required=True,
                help="Corpus source to resolve",
            )
            command_parser.add_argument(
                "--data-root",
                default=None,
                help="Optional shared data root containing multiple course folders",
            )
            continue

        command_parser.add_argument(
            "--course",
            required=True,
            help="Course workspace to run",
        )
        command_parser.add_argument(
            "--data-root",
            default=None,
            help="Optional shared data root containing multiple course folders",
        )
        if command in {"renarrative", "fragment-textbooks"}:
            command_parser.add_argument(
                "--k-values",
                type=int,
                nargs="+",
                default=[1, 3, 5, 7, 10],
                help="K values to include in the renarrativized evaluation run",
            )
            command_parser.add_argument(
                "--model-id",
                default=None,
                help="Optional watsonx model override for the derived-corpus rewrite step",
            )
            command_parser.add_argument(
                "--start-at",
                choices=(
                    list(RENARRATIVE_START_STAGES)
                    if command == "renarrative"
                    else list(TEXTBOOK_FRAGMENTATION_START_STAGES)
                ),
                default="rewrite" if command == "renarrative" else "fragment",
                help=(
                    "Resume the renarrative pipeline from a later stage"
                    if command == "renarrative"
                    else "Resume the fragmented-textbook pipeline from a later stage"
                ),
            )
            continue
        command_parser.add_argument(
            "--source",
            choices=["slides", "textbooks"],
            required=True,
            help="Corpus source to run",
        )
        if command in {"extract", "pipeline"}:
            command_parser.add_argument(
                "--overwrite",
                action="store_true",
                help="Overwrite extracted text files that already exist",
            )
        if command in {"chunk", "pipeline"}:
            command_parser.add_argument(
                "--chunk-mode",
                choices=sorted(VALID_CHUNK_MODES),
                default=None,
                help=(
                    "Chunking mode override. "
                    "Defaults to token-window for slides and textbook-markdown for textbooks"
                ),
            )
            command_parser.add_argument(
                "--chunk-size",
                type=int,
                default=DEFAULT_CHUNK_SIZE,
                help="Maximum number of tokens per chunk",
            )
            command_parser.add_argument(
                "--overlap",
                type=int,
                default=DEFAULT_CHUNK_OVERLAP,
                help="Number of overlapping tokens between neighboring chunks",
            )
        if command in {"retrieve", "generate", "evaluate", "pipeline"}:
            command_parser.add_argument(
                "--top-k",
                type=int,
                default=3,
                help="Number of chunks to retrieve",
            )
        if command in {"ir-metrics", "pipeline"}:
            command_parser.add_argument(
                "--k-values",
                type=int,
                nargs="+",
                default=[1, 3, 5, 7, 10],
                help="K values to include in IR metrics calculation",
            )
        if command == "retrieve":
            command_parser.add_argument(
                "--query",
                default=None,
                help="Optional query text for direct retrieval instead of test-set export",
            )
        if command == "evaluate":
            command_parser.add_argument(
                "--oracle",
                action="store_true",
                help="Evaluate oracle answers generated directly from labeled chunks",
            )
    return parser


def build_paths_payload(workspace_paths: WorkspacePaths, source: str) -> dict[str, str]:
    """Build a JSON-safe payload for resolved workspace and source paths."""
    source_paths = workspace_paths.for_source(source)
    return {
        "course": workspace_paths.course,
        "data_root": str(workspace_paths.data_root),
        "course_root": str(workspace_paths.course_root),
        "input_root": str(workspace_paths.input_root),
        "output_root": str(workspace_paths.output_root),
        "milvus_root": str(workspace_paths.milvus_root),
        "results_root": str(workspace_paths.results_root),
        "test_set_root": str(workspace_paths.test_set_root),
        "test_set_with_labels_path": str(workspace_paths.test_set_with_labels_path),
        "source": source_paths.source,
        "source_input_dir": str(source_paths.input_dir),
        "source_output_dir": str(source_paths.output_dir),
        "text_dir": str(source_paths.text_dir),
        "chunks_dir": str(source_paths.chunks_dir),
        "generation_dir": str(source_paths.generation_dir),
        "retrieval_dir": str(source_paths.retrieval_dir),
        "milvus_dir": str(source_paths.milvus_dir),
        "results_dir": str(source_paths.results_dir),
    }


def run_extract_command(args: argparse.Namespace) -> int:
    """Run source-level PDF extraction for one workspace source."""
    workspace_paths = WorkspacePaths.from_course(
        course=args.course,
        repo_root=Path(__file__).resolve().parent,
        data_root=args.data_root,
    )
    source_paths = workspace_paths.for_source(args.source)
    extractor = _create_pdf_extractor_for_source(args.source)
    results = extract_source_documents(
        source_paths=source_paths,
        extractor=extractor,
        overwrite=args.overwrite,
        show_progress=True,
    )
    payload = {
        "course": workspace_paths.course,
        "source": source_paths.source,
        "input_dir": str(source_paths.input_dir),
        "text_dir": str(source_paths.text_dir),
        "overwrite": args.overwrite,
        "document_count": len(results),
        "written_count": sum(1 for result in results if result.was_written),
        "skipped_count": sum(1 for result in results if not result.was_written),
        "documents": [
            {
                "document_id": result.document.document_id,
                "input_path": str(result.document.input_path),
                "output_path": str(result.document.output_path),
                "page_count": result.document.page_count,
                "was_written": result.was_written,
            }
            for result in results
        ],
    }
    emit_json(payload)
    return 0


def run_chunk_command(args: argparse.Namespace) -> int:
    """Run source-level chunking for one workspace source."""
    workspace_paths = WorkspacePaths.from_course(
        course=args.course,
        repo_root=Path(__file__).resolve().parent,
        data_root=args.data_root,
    )
    source_paths = workspace_paths.for_source(args.source)
    default_config = default_chunking_config(args.source)
    chunking_config = ChunkingConfig(
        mode=args.chunk_mode or default_config.mode,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    documents = load_source_documents(source_paths)
    output_paths = write_chunk_source_documents(
        source_paths=source_paths,
        documents=documents,
        config=chunking_config,
        show_progress=True,
    )
    payload = {
        "course": workspace_paths.course,
        "source": source_paths.source,
        "text_dir": str(source_paths.text_dir),
        "chunks_dir": str(source_paths.chunks_dir),
        "mode": chunking_config.mode,
        "chunk_size": chunking_config.chunk_size,
        "overlap": chunking_config.overlap,
        "document_count": len(documents),
        "written_count": len(output_paths),
        "documents": [
            {
                "document_id": document.document_id,
                "text_path": str(document.output_path),
                "chunk_path": str(source_paths.chunks_dir / f"{document.document_id}.json"),
            }
            for document in sorted(documents, key=lambda item: item.document_id)
        ],
    }
    emit_json(payload)
    return 0


def run_index_command(args: argparse.Namespace) -> int:
    """Run source-level indexing for one workspace source."""
    workspace_paths = WorkspacePaths.from_course(
        course=args.course,
        repo_root=Path(__file__).resolve().parent,
        data_root=args.data_root,
    )
    source_paths = workspace_paths.for_source(args.source)
    chunks = load_source_chunks(source_paths)
    collection_name = chunk_collection_name(source_paths)
    if chunks:
        embedding_provider = create_embedding_provider(show_progress=True)
        vector_store = create_vector_store(source_paths.milvus_dir)
        index_chunks(
            chunks=chunks,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            collection_name=collection_name,
        )
        record_count = vector_store.record_count(collection_name)
    else:
        record_count = 0
    payload = {
        "course": workspace_paths.course,
        "source": source_paths.source,
        "chunks_dir": str(source_paths.chunks_dir),
        "milvus_dir": str(source_paths.milvus_dir),
        "collection_name": collection_name,
        "chunk_count": len(chunks),
        "record_count": record_count,
    }
    emit_json(payload)
    return 0


def run_retrieve_command(args: argparse.Namespace) -> int:
    """Run retrieval for one source corpus."""
    workspace_paths = WorkspacePaths.from_course(
        course=args.course,
        repo_root=Path(__file__).resolve().parent,
        data_root=args.data_root,
    )
    source_paths = workspace_paths.for_source(args.source)
    collection_name = chunk_collection_name(source_paths)

    if args.query:
        embedding_provider = create_embedding_provider(show_progress=False)
        vector_store = create_vector_store(source_paths.milvus_dir)
        retrieved_chunks = retrieve_source_chunks(
            source_paths=source_paths,
            query_text=args.query,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            top_k=args.top_k,
        )
        payload = {
            "course": workspace_paths.course,
            "source": source_paths.source,
            "query": args.query,
            "top_k": args.top_k,
            "collection_name": collection_name,
            "milvus_dir": str(source_paths.milvus_dir),
            "result_count": len(retrieved_chunks),
            "results": [chunk.to_payload() for chunk in retrieved_chunks],
        }
        emit_json(payload)
        return 0

    test_set_path = workspace_paths.test_set_root / "test_set.json"
    if not test_set_path.exists():
        raise FileNotFoundError(
            (
                f"Missing {test_set_path}. "
                "Use --query for direct retrieval or create test_set/test_set.json."
            ),
        )

    embedding_provider = create_embedding_provider(show_progress=False)
    vector_store = create_vector_store(source_paths.milvus_dir)
    test_set_payload = json.loads(
        test_set_path.read_text(encoding="utf-8"),
    )
    if not isinstance(test_set_payload, list):
        raise ValueError("test_set.json must contain a JSON list of query items")
    results = run_retrieval_queries(
        source_paths=source_paths,
        queries=test_set_payload,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        top_k=args.top_k,
        show_progress=True,
    )
    output_path = write_retrieval_results(
        source_paths=source_paths,
        results=results,
        top_k=args.top_k,
        collection_name=collection_name,
        embedding_deployment=getattr(embedding_provider, "model_id", ""),
        embedding_provider=getattr(
            embedding_provider,
            "provider_name",
            None,
        ),
    )
    successful = sum(1 for result in results if result["status"] == "RETRIEVED")
    payload = {
        "course": workspace_paths.course,
        "source": source_paths.source,
        "top_k": args.top_k,
        "collection_name": collection_name,
        "test_set_path": str(test_set_path),
        "retrieval_path": str(output_path),
        "milvus_dir": str(source_paths.milvus_dir),
        "query_count": len(results),
        "successful": successful,
        "failed": len(results) - successful,
    }
    emit_json(payload)
    return 0


def run_generate_command(args: argparse.Namespace) -> int:
    """Run answer generation from saved retrieval results."""
    workspace_paths = WorkspacePaths.from_course(
        course=args.course,
        repo_root=Path(__file__).resolve().parent,
        data_root=args.data_root,
    )
    source_paths = workspace_paths.for_source(args.source)
    retrieval_results = load_retrieval_results(
        source_paths=source_paths,
        top_k=args.top_k,
    )
    logger.info(
        "Starting generation: source=%s top_k=%s items=%s",
        source_paths.source,
        args.top_k,
        len(retrieval_results),
    )
    llm_provider = create_llm_provider()
    results = run_generation_queries(
        source_paths=source_paths,
        retrieval_results=retrieval_results,
        llm_provider=llm_provider,
        show_progress=True,
    )
    output_path = write_generation_results(
        source_paths=source_paths,
        results=results,
        top_k=args.top_k,
        model_id=getattr(llm_provider, "model_id", ""),
        provider=getattr(llm_provider, "provider_name", None),
    )
    successful = sum(1 for result in results if result["status"] == "SUCCESS")
    payload = {
        "course": workspace_paths.course,
        "source": source_paths.source,
        "top_k": args.top_k,
        "retrieval_path": str(source_paths.retrieval_dir / f"retrieval_with_k_{args.top_k}.json"),
        "generation_path": str(output_path),
        "query_count": len(results),
        "successful": successful,
        "failed": len(results) - successful,
    }
    emit_json(payload)
    return 0


def run_evaluate_command(args: argparse.Namespace) -> int:
    """Run evaluation from saved generated answers."""
    workspace_paths = WorkspacePaths.from_course(
        course=args.course,
        repo_root=Path(__file__).resolve().parent,
        data_root=args.data_root,
    )
    source_paths = workspace_paths.for_source(args.source)
    if args.oracle:
        oracle_inputs = load_ground_truth_map(workspace_paths.test_set_with_labels_path)
        logger.info(
            "Starting evaluation: source=%s oracle=%s items=%s",
            source_paths.source,
            True,
            len(oracle_inputs),
        )
        llm_provider = create_llm_provider()
        evaluation_payload = evaluate_oracle_results(
            source_paths=source_paths,
            labeled_test_set_path=workspace_paths.test_set_with_labels_path,
            llm_provider=llm_provider,
            show_progress=True,
        )
        output_path, summary_path = write_oracle_evaluation_results(
            source_paths=source_paths,
            evaluation_payload=evaluation_payload,
        )
        summary = evaluation_payload["summary"]
        payload = {
            "course": workspace_paths.course,
            "source": source_paths.source,
            "oracle": True,
            "evaluation_path": str(output_path),
            "summary_path": str(summary_path),
            "query_count": summary["total_queries"],
            "successful": summary["successful"],
            "failed": summary["failed"],
        }
    else:
        generated_answers = load_generated_answers(source_paths=source_paths, top_k=args.top_k)
        logger.info(
            "Starting evaluation: source=%s oracle=%s top_k=%s items=%s",
            source_paths.source,
            False,
            args.top_k,
            len(generated_answers),
        )
        evaluation_payload = evaluate_generation_results(
            source_paths=source_paths,
            top_k=args.top_k,
            labeled_test_set_path=workspace_paths.test_set_with_labels_path,
            show_progress=True,
        )
        output_path, summary_path = write_evaluation_results(
            source_paths=source_paths,
            top_k=args.top_k,
            evaluation_payload=evaluation_payload,
        )
        summary = evaluation_payload["summary"]
        payload = {
            "course": workspace_paths.course,
            "source": source_paths.source,
            "oracle": False,
            "top_k": args.top_k,
            "generation_path": str(source_paths.generation_dir / f"answers_with_k_{args.top_k}.json"),
            "evaluation_path": str(output_path),
            "summary_path": str(summary_path),
            "query_count": summary["total_queries"],
            "successful": summary["successful"],
            "failed": summary["failed"],
        }
    emit_json(payload)
    return 0


def run_ir_metrics_command(args: argparse.Namespace) -> int:
    """Run IR metrics calculation from generated or retrieval outputs."""
    workspace_paths = WorkspacePaths.from_course(
        course=args.course,
        repo_root=Path(__file__).resolve().parent,
        data_root=args.data_root,
    )
    source_paths = workspace_paths.for_source(args.source)
    ir_metrics_payload = compute_ir_metrics(
        source_paths=source_paths,
        labeled_test_set_path=workspace_paths.test_set_with_labels_path,
        k_values=args.k_values,
    )
    json_path, csv_path, summary_path = write_ir_metrics_results(
        source_paths=source_paths,
        ir_metrics_payload=ir_metrics_payload,
    )
    payload = {
        "course": workspace_paths.course,
        "source": source_paths.source,
        "k_values": args.k_values,
        "ir_metrics_json": str(json_path),
        "ir_metrics_csv": str(csv_path),
        "summary_path": str(summary_path),
        "num_questions": ir_metrics_payload["summary"]["metadata"]["num_questions"],
        "corpus_metrics": ir_metrics_payload["summary"]["corpus_metrics"],
    }
    emit_json(payload)
    return 0


def run_renarrative_command(args: argparse.Namespace) -> int:
    """Run the one-shot slide renarrativization experiment pipeline."""
    workspace_paths = WorkspacePaths.from_course(
        course=args.course,
        repo_root=Path(__file__).resolve().parent,
        data_root=args.data_root,
    )
    payload = run_renarrative_pipeline(
        workspace_paths=workspace_paths,
        model_id=args.model_id,
        k_values=args.k_values,
        start_at=args.start_at,
    )
    emit_json(payload)
    return 0


def run_fragment_textbooks_command(args: argparse.Namespace) -> int:
    """Run the one-shot fragmented-textbook experiment pipeline."""
    workspace_paths = WorkspacePaths.from_course(
        course=args.course,
        repo_root=Path(__file__).resolve().parent,
        data_root=args.data_root,
    )
    payload = run_textbook_fragmentation_pipeline(
        workspace_paths=workspace_paths,
        model_id=args.model_id,
        k_values=args.k_values,
        start_at=args.start_at,
    )
    emit_json(payload)
    return 0


def _resolve_pipeline_k_values(args: argparse.Namespace) -> list[int]:
    """Resolve which IR k-values the pipeline should compute.

    The pipeline only produces retrieval and generation artifacts for one
    `top_k` value per run. If the user did not explicitly override the CLI
    default IR set, align IR metrics with that same `top_k`.
    """
    default_k_values = [1, 3, 5, 7, 10]
    if list(args.k_values) == default_k_values:
        return [args.top_k]
    return list(args.k_values)


def run_pipeline_command(args: argparse.Namespace) -> int:
    """Run the full pipeline in one fixed end-to-end order."""
    workspace_paths = WorkspacePaths.from_course(
        course=args.course,
        repo_root=Path(__file__).resolve().parent,
        data_root=args.data_root,
    )
    source_paths = workspace_paths.for_source(args.source)
    logger.info(
        "Starting pipeline: source=%s top_k=%s",
        source_paths.source,
        args.top_k,
    )
    pipeline_k_values = _resolve_pipeline_k_values(args)

    step_order = [
        "extract",
        "chunk",
        "index",
        "retrieve",
        "generate",
        "evaluate",
        "ir-metrics",
    ]
    completed_steps: list[str] = []
    step_outputs: dict[str, dict[str, object]] = {}

    try:
        logger.info("Pipeline step: extract")
        extractor = _create_pdf_extractor_for_source(args.source)
        extract_results = extract_source_documents(
            source_paths=source_paths,
            extractor=extractor,
            overwrite=args.overwrite,
            show_progress=True,
        )
        step_outputs["extract"] = {
            "document_count": len(extract_results),
            "written_count": sum(1 for result in extract_results if result.was_written),
            "skipped_count": sum(1 for result in extract_results if not result.was_written),
            "overwrite": args.overwrite,
            "reused_existing_outputs": not args.overwrite,
        }
        completed_steps.append("extract")

        logger.info("Pipeline step: chunk")
        default_config = default_chunking_config(args.source)
        chunking_config = ChunkingConfig(
            mode=args.chunk_mode or default_config.mode,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        documents = load_source_documents(source_paths)
        chunk_output_paths = write_chunk_source_documents(
            source_paths=source_paths,
            documents=documents,
            config=chunking_config,
            show_progress=True,
        )
        step_outputs["chunk"] = {
            "document_count": len(documents),
            "written_count": len(chunk_output_paths),
            "mode": chunking_config.mode,
            "chunk_size": chunking_config.chunk_size,
            "overlap": chunking_config.overlap,
            "recomputed": True,
        }
        completed_steps.append("chunk")

        logger.info("Pipeline step: index")
        chunks = load_source_chunks(source_paths)
        embedding_provider = create_embedding_provider(show_progress=True)
        vector_store = create_vector_store(source_paths.milvus_dir)
        collection_name = chunk_collection_name(source_paths)
        index_chunks(
            chunks=chunks,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            collection_name=collection_name,
        )
        step_outputs["index"] = {
            "chunk_count": len(chunks),
            "record_count": vector_store.record_count(collection_name),
            "collection_name": collection_name,
            "recomputed": True,
        }
        completed_steps.append("index")

        logger.info("Pipeline step: retrieve")
        test_set_path = workspace_paths.test_set_root / "test_set.json"
        if not test_set_path.exists():
            raise FileNotFoundError(
                (
                    f"Missing {test_set_path}. "
                    "Create test_set/test_set.json before running pipeline."
                ),
            )
        test_set_payload = json.loads(test_set_path.read_text(encoding="utf-8"))
        if not isinstance(test_set_payload, list):
            raise ValueError("test_set.json must contain a JSON list of query items")
        retrieval_embedding_provider = create_embedding_provider(show_progress=False)
        retrieval_results = run_retrieval_queries(
            source_paths=source_paths,
            queries=test_set_payload,
            embedding_provider=retrieval_embedding_provider,
            vector_store=create_vector_store(source_paths.milvus_dir),
            top_k=args.top_k,
            show_progress=True,
        )
        retrieval_output_path = write_retrieval_results(
            source_paths=source_paths,
            results=retrieval_results,
            top_k=args.top_k,
            collection_name=collection_name,
            embedding_deployment=getattr(retrieval_embedding_provider, "model_id", ""),
            embedding_provider=getattr(retrieval_embedding_provider, "provider_name", None),
        )
        retrieval_successful = sum(
            1 for result in retrieval_results if result["status"] == "RETRIEVED"
        )
        step_outputs["retrieve"] = {
            "query_count": len(retrieval_results),
            "successful": retrieval_successful,
            "failed": len(retrieval_results) - retrieval_successful,
            "top_k": args.top_k,
            "retrieval_path": str(retrieval_output_path),
            "recomputed": True,
        }
        completed_steps.append("retrieve")

        logger.info("Pipeline step: generate")
        logger.info(
            "Starting generation: source=%s top_k=%s items=%s",
            source_paths.source,
            args.top_k,
            len(retrieval_results),
        )
        generation_llm_provider = create_llm_provider()
        generation_results = run_generation_queries(
            source_paths=source_paths,
            retrieval_results=retrieval_results,
            llm_provider=generation_llm_provider,
            show_progress=True,
        )
        generation_output_path = write_generation_results(
            source_paths=source_paths,
            results=generation_results,
            top_k=args.top_k,
            model_id=getattr(generation_llm_provider, "model_id", ""),
            provider=getattr(generation_llm_provider, "provider_name", None),
        )
        generation_successful = sum(
            1 for result in generation_results if result["status"] == "SUCCESS"
        )
        step_outputs["generate"] = {
            "query_count": len(generation_results),
            "successful": generation_successful,
            "failed": len(generation_results) - generation_successful,
            "top_k": args.top_k,
            "generation_path": str(generation_output_path),
            "recomputed": True,
        }
        completed_steps.append("generate")

        logger.info("Pipeline step: evaluate")
        logger.info(
            "Starting evaluation: source=%s oracle=%s top_k=%s",
            source_paths.source,
            False,
            args.top_k,
        )
        evaluation_payload = evaluate_generation_results(
            source_paths=source_paths,
            top_k=args.top_k,
            labeled_test_set_path=workspace_paths.test_set_with_labels_path,
            show_progress=True,
        )
        evaluation_output_path, evaluation_summary_path = write_evaluation_results(
            source_paths=source_paths,
            top_k=args.top_k,
            evaluation_payload=evaluation_payload,
        )
        evaluation_summary = evaluation_payload["summary"]
        step_outputs["evaluate"] = {
            "query_count": evaluation_summary["total_queries"],
            "successful": evaluation_summary["successful"],
            "failed": evaluation_summary["failed"],
            "top_k": args.top_k,
            "evaluation_path": str(evaluation_output_path),
            "summary_path": str(evaluation_summary_path),
            "recomputed": True,
        }
        completed_steps.append("evaluate")

        logger.info("Pipeline step: ir-metrics")
        ir_metrics_payload = compute_ir_metrics(
            source_paths=source_paths,
            labeled_test_set_path=workspace_paths.test_set_with_labels_path,
            k_values=pipeline_k_values,
        )
        ir_json_path, ir_csv_path, ir_summary_path = write_ir_metrics_results(
            source_paths=source_paths,
            ir_metrics_payload=ir_metrics_payload,
        )
        step_outputs["ir-metrics"] = {
            "k_values": pipeline_k_values,
            "ir_metrics_json": str(ir_json_path),
            "ir_metrics_csv": str(ir_csv_path),
            "summary_path": str(ir_summary_path),
            "recomputed": True,
        }
        completed_steps.append("ir-metrics")
    except Exception as exc:
        raise RuntimeError(
            f"Pipeline failed during step '{step_order[len(completed_steps)]}'"
        ) from exc

    emit_json(
        {
            "course": workspace_paths.course,
            "source": source_paths.source,
            "pipeline_order": step_order,
            "stops_on_failure": True,
            "reuse_policy": {
                "extract": "reuse existing text files unless --overwrite is set",
                "chunk": "recompute chunk JSON from current extracted text",
                "index": "recompute vector collection from current chunk JSON",
                "retrieve": "recompute retrieval export from current index and test set",
                "generate": "recompute generation export from current retrieval export",
                "evaluate": "recompute evaluation outputs from current generation export",
                "ir-metrics": (
                    "recompute IR summaries from current retrieval or generation outputs"
                ),
            },
            "completed_steps": completed_steps,
            "outputs": step_outputs,
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "paths":
        workspace_paths = WorkspacePaths.from_course(
            course=args.course,
            repo_root=Path(__file__).resolve().parent,
            data_root=args.data_root,
        )
        emit_json(build_paths_payload(workspace_paths, args.source))
        return 0

    if args.command == "extract":
        return run_extract_command(args)

    if args.command == "chunk":
        return run_chunk_command(args)

    if args.command == "index":
        return run_index_command(args)

    if args.command == "retrieve":
        return run_retrieve_command(args)

    if args.command == "generate":
        return run_generate_command(args)

    if args.command == "evaluate":
        return run_evaluate_command(args)

    if args.command == "ir-metrics":
        return run_ir_metrics_command(args)

    if args.command == "fragment-textbooks":
        return run_fragment_textbooks_command(args)

    if args.command == "pipeline":
        return run_pipeline_command(args)

    if args.command == "renarrative":
        return run_renarrative_command(args)

    raise NotImplementedError(f"Command '{args.command}' is not implemented yet.")


if __name__ == "__main__":
    raise SystemExit(main())
