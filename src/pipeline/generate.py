"""Generation step helpers."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

from src.models import RetrievedChunk
from src.paths import SourcePaths
from src.providers.llm import LlmProvider
from src.services.prompts import build_generation_prompt
from tqdm import tqdm


def generate_answer(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    llm_provider: LlmProvider,
) -> dict[str, object]:
    """Generate one answer from retrieved chunks.

    Args:
        question: User question to answer.
        retrieved_chunks: Ranked chunks used as context.
        llm_provider: LLM backend used to generate the answer.

    Returns:
        A generation payload with the rendered prompt and answer text.
    """
    prompt = build_generation_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )
    answer = llm_provider.generate_text(prompt)
    return {
        "question": question,
        "prompt": prompt,
        "answer": answer,
        "retrieved_chunks": [chunk.to_payload() for chunk in retrieved_chunks],
    }


def generation_output_path(source_paths: SourcePaths, top_k: int) -> Path:
    """Return the output path for generated answer JSON exports."""
    return source_paths.generation_dir / f"answers_with_k_{top_k}.json"


def retrieval_input_path(source_paths: SourcePaths, top_k: int) -> Path:
    """Return the retrieval export path used as generation input."""
    return source_paths.retrieval_dir / f"retrieval_with_k_{top_k}.json"


def load_retrieval_results(source_paths: SourcePaths, top_k: int) -> list[dict[str, object]]:
    """Load retrieval export results for one source and top-k setting."""
    input_path = retrieval_input_path(source_paths, top_k)
    if not input_path.exists():
        raise FileNotFoundError(
            (
                f"Missing {input_path}. "
                "Run retrieve first to create retrieval_with_k_<k>.json."
            ),
        )

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("retrieval results payload must contain a JSON list under 'results'")
    return results


def run_generation_queries(
    source_paths: SourcePaths,
    retrieval_results: list[dict[str, object]],
    llm_provider: LlmProvider,
    show_progress: bool = False,
) -> list[dict[str, object]]:
    """Generate answers for a list of retrieval results in input order."""
    progress = (
        tqdm(total=len(retrieval_results), desc="Generating answers", unit="query")
        if show_progress
        else None
    )
    indexed_results: list[dict[str, object]] = []
    try:
        for result_item in retrieval_results:
            indexed_results.append(
                _generate_query_result(
                    source_paths=source_paths,
                    result_item=result_item,
                    llm_provider=llm_provider,
                ),
            )
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    return indexed_results


def _generate_query_result(
    source_paths: SourcePaths,
    result_item: dict[str, object],
    llm_provider: LlmProvider,
) -> dict[str, object]:
    """Generate one query result while preserving stable ordering."""
    query = str(result_item["query"])
    question_id = result_item.get("question_id")
    retrieved_chunks = build_retrieved_chunks_from_payloads(
        source=source_paths.source,
        chunk_payloads=list(result_item.get("chunks", [])),
    )

    try:
        generation = generate_answer(
            question=query,
            retrieved_chunks=retrieved_chunks,
            llm_provider=llm_provider,
        )
        output_item: dict[str, object] = {
            "query": query,
            "answer": generation["answer"],
            "chunks": generation["retrieved_chunks"],
            "status": "SUCCESS",
        }
        if question_id is not None:
            output_item["question_id"] = str(question_id)
    except Exception as exc:
        output_item = {
            "query": query,
            "answer": None,
            "chunks": [chunk.to_payload() for chunk in retrieved_chunks],
            "status": "ERROR",
            "error": str(exc),
        }
        if question_id is not None:
            output_item["question_id"] = str(question_id)

    return output_item


def build_generation_export_payload(
    source_paths: SourcePaths,
    results: list[dict[str, object]],
    top_k: int,
    model_id: str,
    provider: str | None = None,
    timestamp: str | None = None,
) -> dict[str, object]:
    """Build the current top-level generation export payload."""
    successful = sum(1 for result in results if result.get("status") == "SUCCESS")
    failed = len(results) - successful
    return {
        "metadata": {
            "timestamp": timestamp or datetime.now(UTC).replace(tzinfo=None).isoformat(),
            "model_id": model_id,
            "provider": provider,
            "top_k": top_k,
            "source": source_paths.source,
        },
        "status": "SUCCESS",
        "results": results,
        "summary": {
            "total_queries": len(results),
            "successful": successful,
            "failed": failed,
        },
    }


def write_generation_results(
    source_paths: SourcePaths,
    results: list[dict[str, object]],
    top_k: int,
    model_id: str,
    provider: str | None = None,
    timestamp: str | None = None,
) -> Path:
    """Write generated answers in the current JSON export shape."""
    output_path = generation_output_path(source_paths, top_k)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_generation_export_payload(
        source_paths=source_paths,
        results=results,
        top_k=top_k,
        model_id=model_id,
        provider=provider,
        timestamp=timestamp,
    )
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def build_retrieved_chunks_from_payloads(
    source: str,
    chunk_payloads: list[dict[str, object]],
) -> list[RetrievedChunk]:
    """Convert retrieval-export chunk payloads back into retrieved chunk records."""
    return [
        RetrievedChunk(
            source=source,
            document_id=str(chunk_payload.get("document_id", chunk_payload["doc_id"])),
            text=str(chunk_payload["text"]),
            page_start=_optional_int(chunk_payload.get("page_start")),
            page_end=_optional_int(chunk_payload.get("page_end")),
            token_start=int(chunk_payload["token_start"]),
            token_end=int(chunk_payload["token_end"]),
            chunk_index=int(chunk_payload["chunk_index"]),
            chunk_id=str(chunk_payload["chunk_id"]),
            total_chunks=int(chunk_payload["total_chunks"]),
            headers=dict(chunk_payload.get("headers", {})),
            score=float(chunk_payload.get("distance", 0.0)),
            rank=index,
            doc_id=str(chunk_payload.get("doc_id", "")),
            source_path=str(chunk_payload.get("source_path", "")),
            window_size=int(chunk_payload.get("window_size", 0)),
            overlap=int(chunk_payload.get("overlap", 0)),
            boundary_snapped=bool(chunk_payload.get("boundary_snapped", False)),
        )
        for index, chunk_payload in enumerate(chunk_payloads, start=1)
    ]


def _optional_int(value: object) -> int | None:
    """Normalize nullable numeric fields from retrieval exports."""
    if value is None:
        return None
    return int(value)
