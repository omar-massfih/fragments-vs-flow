"""IR metrics step helpers."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any

from src.derived_sources import resolve_label_fields
from src.paths import SourcePaths


def ir_metrics_results_dir(source_paths: SourcePaths) -> Path:
    """Return the current-style results directory for IR metrics outputs."""
    return source_paths.results_dir.parent / "ir_metrics" / source_paths.source


def ir_metrics_output_path(source_paths: SourcePaths) -> Path:
    """Return the output path for IR metrics JSON results."""
    return ir_metrics_results_dir(source_paths) / "ir_metrics.json"


def ir_metrics_csv_output_path(source_paths: SourcePaths) -> Path:
    """Return the output path for IR metrics CSV results."""
    return ir_metrics_results_dir(source_paths) / "ir_metrics.csv"


def ir_metrics_summary_path(source_paths: SourcePaths) -> Path:
    """Return the output path for the compact IR metrics summary."""
    return ir_metrics_results_dir(source_paths) / "SUMMARY.json"


def load_generation_results(source_paths: SourcePaths, top_k: int) -> list[dict[str, object]]:
    """Load one generated-answer export used for IR metrics calculation."""
    generation_path = source_paths.generation_dir / f"answers_with_k_{top_k}.json"
    if not generation_path.exists():
        raise FileNotFoundError(
            f"Missing {generation_path}. Run generate first to create answers_with_k_<k>.json.",
        )

    payload = json.loads(generation_path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("generation results payload must contain a JSON list under 'results'")
    return results


def load_retrieval_results(source_paths: SourcePaths, top_k: int) -> list[dict[str, object]]:
    """Load one retrieval-only export used for IR metrics calculation."""
    retrieval_path = source_paths.retrieval_dir / f"retrieval_with_k_{top_k}.json"
    if not retrieval_path.exists():
        raise FileNotFoundError(
            f"Missing {retrieval_path}. Run retrieve first to create retrieval_with_k_<k>.json.",
        )

    payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("retrieval results payload must contain a JSON list under 'results'")
    return results


def load_ranked_results(source_paths: SourcePaths, top_k: int) -> list[dict[str, object]]:
    """Load ranked retrieval contexts from the newest available ranked artifact."""
    generation_path = source_paths.generation_dir / f"answers_with_k_{top_k}.json"
    retrieval_path = source_paths.retrieval_dir / f"retrieval_with_k_{top_k}.json"

    if generation_path.exists() and retrieval_path.exists():
        if generation_path.stat().st_mtime > retrieval_path.stat().st_mtime:
            return load_generation_results(source_paths, top_k)
        return load_retrieval_results(source_paths, top_k)
    if generation_path.exists():
        return load_generation_results(source_paths, top_k)
    return load_retrieval_results(source_paths, top_k)


def load_labeled_test_set(labeled_test_set_path: Path) -> dict[str, dict[str, object]]:
    """Load the labeled test set keyed by question ID."""
    if not labeled_test_set_path.exists():
        raise FileNotFoundError(f"Missing labeled test set: {labeled_test_set_path}")

    payload = json.loads(labeled_test_set_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("test_set_with_labels.json must contain a JSON list")

    return {
        str(item["question_id"]): item
        for item in payload
        if isinstance(item, dict) and item.get("question_id") is not None
    }


def compute_ir_metrics(
    source_paths: SourcePaths,
    labeled_test_set_path: Path,
    k_values: list[int] | None = None,
    label_source: str | None = None,
) -> dict[str, object]:
    """Compute IR metrics from generated outputs and labeled chunk IDs."""
    selected_k_values = k_values or [1, 3, 5, 7, 10]
    effective_label_source = label_source or source_paths.source
    test_set = load_labeled_test_set(labeled_test_set_path)
    per_question_results: list[dict[str, object]] = []
    corpus_scores: dict[str, float] = {}

    for top_k in selected_k_values:
        generation_results = load_ranked_results(source_paths, top_k)
        hits: list[int] = []
        primary_hits: list[int] = []
        context_precisions: list[float] = []
        context_recalls: list[float] = []
        mrr_scores: list[float] = []
        ndcg_scores: list[float] = []

        for item in generation_results:
            question_id = str(item.get("question_id", ""))
            if question_id not in test_set:
                continue

            test_item = test_set[question_id]
            labels = select_labels_for_source(test_item, effective_label_source)
            gold_chunks = {
                chunk_id for chunk_id, label in labels.items()
                if label >= 1
            }
            retrieved_chunk_ids = [
                str(chunk.get("chunk_id", ""))
                for chunk in item.get("chunks", [])
            ]

            hit = calculate_hit_at_k(retrieved_chunk_ids, labels, top_k)
            primary_hit = calculate_primary_at_k(retrieved_chunk_ids, labels, top_k)
            context_precision = calculate_context_precision_at_k(
                retrieved_chunk_ids,
                labels,
                top_k,
            )
            context_recall = calculate_context_recall_at_k(
                retrieved_chunk_ids,
                labels,
                top_k,
            )
            hits.append(hit)
            primary_hits.append(primary_hit)
            context_precisions.append(context_precision)
            context_recalls.append(context_recall)
            mrr_score = calculate_mrr_at_k(retrieved_chunk_ids, gold_chunks, k=top_k)
            ndcg_score = calculate_ndcg_at_k(retrieved_chunk_ids, labels, k=top_k)
            mrr_scores.append(mrr_score)
            ndcg_scores.append(ndcg_score)

            per_question_results.append(
                {
                    "question_id": question_id,
                    "complexity": test_item.get("complexity", ""),
                    "reasoning_type": test_item.get("reasoning_type", ""),
                    "metric": f"Hit@{top_k}",
                    "value": hit,
                },
            )
            per_question_results.append(
                {
                    "question_id": question_id,
                    "complexity": test_item.get("complexity", ""),
                    "reasoning_type": test_item.get("reasoning_type", ""),
                    "metric": f"Primary@{top_k}",
                    "value": primary_hit,
                },
            )
            per_question_results.append(
                {
                    "question_id": question_id,
                    "complexity": test_item.get("complexity", ""),
                    "reasoning_type": test_item.get("reasoning_type", ""),
                    "metric": f"ContextPrecision@{top_k}",
                    "value": context_precision,
                },
            )
            per_question_results.append(
                {
                    "question_id": question_id,
                    "complexity": test_item.get("complexity", ""),
                    "reasoning_type": test_item.get("reasoning_type", ""),
                    "metric": f"ContextRecall@{top_k}",
                    "value": context_recall,
                },
            )
            per_question_results.append(
                {
                    "question_id": question_id,
                    "complexity": test_item.get("complexity", ""),
                    "reasoning_type": test_item.get("reasoning_type", ""),
                    "metric": f"MRR@{top_k}",
                    "value": mrr_score,
                },
            )
            per_question_results.append(
                {
                    "question_id": question_id,
                    "complexity": test_item.get("complexity", ""),
                    "reasoning_type": test_item.get("reasoning_type", ""),
                    "metric": f"nDCG@{top_k}",
                    "value": ndcg_score,
                },
            )

        if hits:
            corpus_scores[f"Hit@{top_k}"] = sum(hits) / len(hits)
        if primary_hits:
            corpus_scores[f"Primary@{top_k}"] = sum(primary_hits) / len(primary_hits)
        if context_precisions:
            corpus_scores[f"ContextPrecision@{top_k}"] = (
                sum(context_precisions) / len(context_precisions)
            )
        if context_recalls:
            corpus_scores[f"ContextRecall@{top_k}"] = (
                sum(context_recalls) / len(context_recalls)
            )
        if mrr_scores:
            corpus_scores[f"MRR@{top_k}"] = sum(mrr_scores) / len(mrr_scores)
        if ndcg_scores:
            corpus_scores[f"nDCG@{top_k}"] = sum(ndcg_scores) / len(ndcg_scores)

    return build_ir_metrics_payload(
        source_paths=source_paths,
        k_values=selected_k_values,
        results=per_question_results,
        corpus_scores=corpus_scores,
    )


def compute_ir_metrics_rows_for_question(
    source_paths: SourcePaths,
    labeled_test_set_path: Path,
    question_id: str,
    k_values: list[int] | None = None,
    label_source: str | None = None,
) -> list[dict[str, object]]:
    """Compute IR metric rows for one question across selected k values."""
    selected_k_values = k_values or [1, 3, 5, 7, 10]
    effective_label_source = label_source or source_paths.source
    test_set = load_labeled_test_set(labeled_test_set_path)
    if question_id not in test_set:
        raise ValueError(f"question_id={question_id} was not found in the labeled test set")

    test_item = test_set[question_id]
    labels = select_labels_for_source(test_item, effective_label_source)
    gold_chunks = {
        chunk_id for chunk_id, label in labels.items()
        if label >= 1
    }
    per_question_results: list[dict[str, object]] = []

    for top_k in selected_k_values:
        ranked_results = load_ranked_results(source_paths, top_k)
        item = next(
            (
                candidate
                for candidate in ranked_results
                if str(candidate.get("question_id", "")) == question_id
            ),
            None,
        )
        if item is None:
            raise ValueError(
                f"question_id={question_id} was not found in ranked results for top_k={top_k}",
            )

        retrieved_chunk_ids = [
            str(chunk.get("chunk_id", ""))
            for chunk in item.get("chunks", [])
        ]
        hit = calculate_hit_at_k(retrieved_chunk_ids, labels, top_k)
        primary_hit = calculate_primary_at_k(retrieved_chunk_ids, labels, top_k)
        context_precision = calculate_context_precision_at_k(
            retrieved_chunk_ids,
            labels,
            top_k,
        )
        context_recall = calculate_context_recall_at_k(
            retrieved_chunk_ids,
            labels,
            top_k,
        )
        mrr_score = calculate_mrr_at_k(retrieved_chunk_ids, gold_chunks, k=top_k)
        ndcg_score = calculate_ndcg_at_k(retrieved_chunk_ids, labels, k=top_k)

        for metric, value in (
            (f"Hit@{top_k}", hit),
            (f"Primary@{top_k}", primary_hit),
            (f"ContextPrecision@{top_k}", context_precision),
            (f"ContextRecall@{top_k}", context_recall),
            (f"MRR@{top_k}", mrr_score),
            (f"nDCG@{top_k}", ndcg_score),
        ):
            per_question_results.append(
                {
                    "question_id": question_id,
                    "complexity": test_item.get("complexity", ""),
                    "reasoning_type": test_item.get("reasoning_type", ""),
                    "metric": metric,
                    "value": value,
                },
            )

    return per_question_results


def write_ir_metrics_results(
    source_paths: SourcePaths,
    ir_metrics_payload: dict[str, object],
) -> tuple[Path, Path, Path]:
    """Write IR metrics JSON, CSV, and summary files."""
    json_path = ir_metrics_output_path(source_paths)
    csv_path = ir_metrics_csv_output_path(source_paths)
    summary_path = ir_metrics_summary_path(source_paths)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(
        json.dumps(ir_metrics_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    results = list(ir_metrics_payload["results"])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["question_id", "complexity", "reasoning_type", "metric", "value"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    summary_path.write_text(
        json.dumps(ir_metrics_payload["summary"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return json_path, csv_path, summary_path


def build_ir_metrics_payload_from_results(
    source_paths: SourcePaths,
    k_values: list[int],
    results: list[dict[str, object]],
    timestamp: str | None = None,
) -> dict[str, object]:
    """Build an IR metrics payload from an existing flat results list."""
    corpus_scores = _build_corpus_scores_from_results(results, k_values)
    return build_ir_metrics_payload(
        source_paths=source_paths,
        k_values=k_values,
        results=results,
        corpus_scores=corpus_scores,
        timestamp=timestamp,
    )


def build_ir_metrics_payload(
    source_paths: SourcePaths,
    k_values: list[int],
    results: list[dict[str, object]],
    corpus_scores: dict[str, float],
    timestamp: str | None = None,
) -> dict[str, object]:
    """Build the top-level IR metrics payload."""
    run_timestamp = timestamp or datetime.now(UTC).replace(tzinfo=None).isoformat()
    breakdowns = _build_breakdowns(results, corpus_scores)
    return {
        "metadata": {
            "generated_at_utc": run_timestamp,
            "source": source_paths.source,
            "k_values": k_values,
            "num_questions": len({result["question_id"] for result in results}),
        },
        "status": "SUCCESS",
        "corpus_scores": corpus_scores,
        "breakdowns": breakdowns,
        "per_question_results": _build_per_question_results(results),
        "results": results,
        "summary": _build_summary_payload(
            source_paths=source_paths,
            k_values=k_values,
            results=results,
            corpus_scores=corpus_scores,
            breakdowns=breakdowns,
            timestamp=run_timestamp,
        ),
    }


def select_labels_for_source(test_item: dict[str, object], source: str) -> dict[str, int]:
    """Select the labeled chunk dictionary for one source corpus."""
    label_field, _ = resolve_label_fields(source)
    labels = test_item.get(label_field, {})

    if not isinstance(labels, dict):
        return {}
    return {str(chunk_id): int(label) for chunk_id, label in labels.items()}


def calculate_hit_at_k(retrieved_chunk_ids: list[str], labels: dict[str, int], k: int) -> int:
    """Return 1 if any relevant chunk appears in top-k, else 0."""
    top_k_chunks = retrieved_chunk_ids[:k]
    gold_chunks = {
        chunk_id for chunk_id, label in labels.items()
        if label >= 1
    }
    return 1 if set(top_k_chunks) & gold_chunks else 0


def calculate_primary_at_k(
    retrieved_chunk_ids: list[str],
    labels: dict[str, int],
    k: int,
) -> int:
    """Return 1 if any primary chunk appears in top-k, else 0."""
    top_k_chunks = retrieved_chunk_ids[:k]
    primary_chunks = {
        chunk_id for chunk_id, label in labels.items()
        if label == 2
    }
    return 1 if set(top_k_chunks) & primary_chunks else 0


def calculate_mrr_at_k(
    retrieved_chunk_ids: list[str],
    gold_chunks: set[str],
    k: int = 10,
) -> float:
    """Calculate reciprocal rank of the first relevant chunk in top-k."""
    for index, chunk_id in enumerate(retrieved_chunk_ids[:k], start=1):
        if chunk_id in gold_chunks:
            return 1.0 / index
    return 0.0


def calculate_ndcg_at_k(
    retrieved_chunk_ids: list[str],
    labels: dict[str, int],
    k: int = 10,
) -> float:
    """Calculate graded nDCG at k."""
    dcg = 0.0
    for index, chunk_id in enumerate(retrieved_chunk_ids[:k], start=1):
        relevance = labels.get(chunk_id, 0)
        if relevance > 0:
            dcg += relevance / math.log2(index + 1)

    ideal_relevances = sorted(
        [label for label in labels.values() if label > 0],
        reverse=True,
    )
    if not ideal_relevances:
        return 0.0

    idcg = 0.0
    for index, relevance in enumerate(ideal_relevances[:k], start=1):
        idcg += relevance / math.log2(index + 1)
    if idcg == 0:
        return 0.0
    return min(dcg / idcg, 1.0)


def calculate_context_precision_at_k(
    retrieved_chunk_ids: list[str],
    labels: dict[str, int],
    k: int,
) -> float:
    """Calculate deterministic ID-based context precision at K."""
    if k <= 0:
        return 0.0
    top_k = retrieved_chunk_ids[:k]
    gold_chunks = {
        chunk_id for chunk_id, label in labels.items()
        if label >= 1
    }
    relevant_in_topk = len(set(top_k) & gold_chunks)
    return relevant_in_topk / k


def calculate_context_recall_at_k(
    retrieved_chunk_ids: list[str],
    labels: dict[str, int],
    k: int,
) -> float:
    """Calculate deterministic ID-based context recall at K."""
    top_k = retrieved_chunk_ids[:k]
    gold_chunks = {
        chunk_id for chunk_id, label in labels.items()
        if label >= 1
    }
    if not gold_chunks:
        return 0.0
    relevant_in_topk = len(set(top_k) & gold_chunks)
    return relevant_in_topk / len(gold_chunks)


def _build_per_question_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Group flat metric rows into one per-question results structure."""
    grouped: dict[str, dict[str, object]] = {}
    for result in results:
        question_id = str(result["question_id"])
        entry = grouped.setdefault(
            question_id,
            {
                "question_id": question_id,
                "complexity": result.get("complexity", ""),
                "reasoning_type": result.get("reasoning_type", ""),
                "num_gold_chunks": None,
                "results": {},
            },
        )
        entry["results"][str(result["metric"])] = result["value"]
    return [grouped[key] for key in sorted(grouped)]


def _build_breakdowns(
    results: list[dict[str, object]],
    corpus_scores: dict[str, float],
) -> dict[str, dict[str, dict[str, float]]]:
    """Build grouped averages by complexity and reasoning type."""
    by_complexity: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_reasoning_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for result in results:
        metric = str(result["metric"])
        value = float(result["value"])
        complexity = str(result.get("complexity", "") or "")
        reasoning_type = str(result.get("reasoning_type", "") or "")
        if complexity:
            by_complexity[metric][complexity].append(value)
        if reasoning_type:
            by_reasoning_type[metric][reasoning_type].append(value)

    metric_order = list(corpus_scores.keys())
    return {
        "by_complexity": {
            metric: {
                category: round(sum(values) / len(values), 4)
                for category, values in sorted(by_complexity.get(metric, {}).items())
                if values
            }
            for metric in metric_order
            if by_complexity.get(metric)
        },
        "by_reasoning_type": {
            metric: {
                category: round(sum(values) / len(values), 4)
                for category, values in sorted(by_reasoning_type.get(metric, {}).items())
                if values
            }
            for metric in metric_order
            if by_reasoning_type.get(metric)
        },
    }


def _build_corpus_scores_from_results(
    results: list[dict[str, object]],
    k_values: list[int],
) -> dict[str, float]:
    """Derive corpus-level metric averages from flat per-question rows."""
    metric_values: dict[str, list[float]] = defaultdict(list)
    for result in results:
        metric = str(result["metric"])
        metric_values[metric].append(float(result["value"]))

    ordered_metrics = [
        metric
        for top_k in k_values
        for metric in (
            f"Hit@{top_k}",
            f"Primary@{top_k}",
            f"ContextPrecision@{top_k}",
            f"ContextRecall@{top_k}",
            f"MRR@{top_k}",
            f"nDCG@{top_k}",
        )
        if metric in metric_values
    ]
    return {
        metric: sum(metric_values[metric]) / len(metric_values[metric])
        for metric in ordered_metrics
    }


def _build_summary_payload(
    source_paths: SourcePaths,
    k_values: list[int],
    results: list[dict[str, object]],
    corpus_scores: dict[str, float],
    breakdowns: dict[str, dict[str, dict[str, float]]],
    timestamp: str,
) -> dict[str, object]:
    """Build the current-style IR summary payload."""
    num_questions = len({str(result["question_id"]) for result in results})
    by_complexity = _format_summary_group(
        grouped_breakdowns=breakdowns["by_complexity"],
        results=results,
        group_key="complexity",
        metric_order=list(corpus_scores.keys()),
    )
    by_reasoning_type = _format_summary_group(
        grouped_breakdowns=breakdowns["by_reasoning_type"],
        results=results,
        group_key="reasoning_type",
        metric_order=list(corpus_scores.keys()),
    )
    failure_analysis = _build_failure_analysis(results)
    derived_metrics = _build_derived_metrics(
        corpus_scores=corpus_scores,
        by_complexity=by_complexity,
        by_reasoning_type=by_reasoning_type,
        num_questions=num_questions,
        failure_analysis=failure_analysis,
    )

    return {
        "metadata": {
            "timestamp": timestamp,
            "corpus": source_paths.source,
            "num_questions": num_questions,
            "k_values": k_values,
        },
        "corpus_metrics": {
            metric: round(value, 4)
            for metric, value in corpus_scores.items()
        },
        "by_complexity": by_complexity,
        "by_reasoning_type": by_reasoning_type,
        "failure_analysis": failure_analysis,
        "derived_metrics": derived_metrics,
    }


def _format_summary_group(
    grouped_breakdowns: dict[str, dict[str, float]],
    results: list[dict[str, object]],
    group_key: str,
    metric_order: list[str],
) -> dict[str, dict[str, float | int]]:
    """Format grouped averages into the summary structure with counts."""
    counts: dict[str, set[str]] = defaultdict(set)
    for result in results:
        category = str(result.get(group_key, "") or "")
        if category:
            counts[category].add(str(result["question_id"]))

    formatted: dict[str, dict[str, float | int]] = {}
    for category in sorted(counts):
        metrics: dict[str, float | int] = {"n": len(counts[category])}
        for metric in metric_order:
            metric_values = grouped_breakdowns.get(metric, {})
            if category in metric_values:
                metrics[metric] = metric_values[category]
        formatted[category] = metrics
    return formatted


def _build_failure_analysis(results: list[dict[str, object]]) -> dict[str, object]:
    """Build summary failure diagnostics from per-question metric rows."""
    zero_retrieval: list[str] = []
    low_mrr: list[dict[str, object]] = []
    for result in results:
        if result["metric"] != "MRR@10":
            continue
        question_id = str(result["question_id"])
        value = float(result["value"])
        if value == 0.0:
            zero_retrieval.append(question_id)
        elif value < 0.15:
            approx_rank = int(round(1.0 / value)) if value > 0 else 0
            low_mrr.append(
                {
                    "question_id": question_id,
                    "MRR@10": round(value, 4),
                    "approx_rank": approx_rank,
                },
            )
    return {
        "zero_retrieval": zero_retrieval,
        "low_mrr": low_mrr,
    }


def _build_derived_metrics(
    corpus_scores: dict[str, float],
    by_complexity: dict[str, dict[str, float | int]],
    by_reasoning_type: dict[str, dict[str, float | int]],
    num_questions: int,
    failure_analysis: dict[str, object],
) -> dict[str, object]:
    """Build derived corpus-level IR metrics."""
    complexity_mrrs = {
        category: float(values.get("MRR@10", 0.0))
        for category, values in by_complexity.items()
    }
    best_complexity = max(complexity_mrrs, key=complexity_mrrs.get) if complexity_mrrs else None
    worst_complexity = min(complexity_mrrs, key=complexity_mrrs.get) if complexity_mrrs else None
    fact_mrr = float(by_reasoning_type.get("Fact_Lookup", {}).get("MRR@10", 0.0))
    multihop_mrr = float(by_reasoning_type.get("Multi-hop", {}).get("MRR@10", 0.0))
    zero_retrieval = list(failure_analysis.get("zero_retrieval", []))
    low_mrr = list(failure_analysis.get("low_mrr", []))

    return {
        "hit10_mrr10_gap": round(corpus_scores.get("Hit@10", 0.0) - corpus_scores.get("MRR@10", 0.0), 4),
        "fact_vs_multihop_gap": round(fact_mrr - multihop_mrr, 4),
        "best_complexity": best_complexity,
        "worst_complexity": worst_complexity,
        "best_complexity_mrr": round(complexity_mrrs.get(best_complexity, 0.0), 4),
        "worst_complexity_mrr": round(complexity_mrrs.get(worst_complexity, 0.0), 4),
        "zero_retrieval_rate": round(len(zero_retrieval) / num_questions, 4) if num_questions else 0.0,
        "low_ranking_rate": round(len(low_mrr) / num_questions, 4) if num_questions else 0.0,
    }
