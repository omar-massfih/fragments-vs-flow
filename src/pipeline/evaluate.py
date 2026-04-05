"""RAGAS-based evaluation helpers."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import UTC, datetime
import inspect
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Protocol

from src.config import PipelineConfig
from src.derived_sources import resolve_label_fields
from src.models import ChunkRecord, RetrievedChunk
from src.paths import SourcePaths
from src.pipeline.generate import generate_answer
from src.pipeline.index import load_source_chunks
from src.providers.llm import LlmProvider
from tqdm import tqdm


logger = logging.getLogger(__name__)


class AsyncLoopRunner:
    """Run async coroutines on one persistent background event loop."""

    def __init__(self) -> None:
        """Start a dedicated background event loop thread."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        """Own the background event loop for submitted coroutines."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coroutine: Any) -> Any:
        """Submit one coroutine to the background loop and wait for its result."""
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result()

    def close(self) -> None:
        """Stop the background loop and join the worker thread."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()


class SampleEvaluator(Protocol):
    """Protocol for one-sample evaluation backends."""

    def evaluate_sample(
        self,
        question: str,
        ground_truth: str,
        retrieved_contexts: list[str],
        response: str,
    ) -> dict[str, float | None]:
        """Evaluate one generated answer against one reference answer."""

    def close(self) -> None:
        """Release resources held by the evaluator, if any."""


@dataclass(frozen=True)
class EvaluationSampleResult:
    """Evaluation result for one question-answer pair."""

    question_id: str
    question: str
    ground_truth: str
    answer: str
    chunks: list[dict[str, object]]
    retrieved_contexts: list[str]
    retrieved_chunk_ids: list[str]
    retrieval_mode: str
    k: int
    context_precision: float | None
    context_recall: float | None
    faithfulness: float | None
    answer_relevancy: float | None
    answer_correctness: float | None
    retrieval_time_s: float
    generation_time_s: float
    evaluation_time_s: float
    complexity: str | None = None
    reasoning_type: str | None = None
    error: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Return the result in a stable JSON payload format."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "complexity": self.complexity,
            "reasoning_type": self.reasoning_type,
            "retrieval_mode": self.retrieval_mode,
            "k": self.k,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "retrieved_contexts": self.retrieved_contexts,
            "answer": self.answer,
            "context_char_len": sum(len(context) for context in self.retrieved_contexts),
            "retrieval_time_s": self.retrieval_time_s,
            "generation_time_s": self.generation_time_s,
            "evaluation_time_s": self.evaluation_time_s,
            "total_time_s": (
                self.retrieval_time_s + self.generation_time_s + self.evaluation_time_s
            ),
            "chunks": self.chunks,
            "metrics": {
                "context_precision": self.context_precision,
                "context_recall": self.context_recall,
                "faithfulness": self.faithfulness,
                "answer_relevancy": self.answer_relevancy,
                "answer_correctness": self.answer_correctness,
            },
            "status": "ERROR" if self.error else "SUCCESS",
            "error": self.error,
        }


class RagasEvaluator:
    """RAGAS evaluation backend mirroring the top-level pipeline."""

    def __init__(
        self,
        llm: object,
        embeddings: object,
        strictness: int = 1,
        close_targets: list[object] | None = None,
        *,
        provider_name: str | None = None,
        model_id: str | None = None,
        embedding_provider_name: str | None = None,
        embedding_model_id: str | None = None,
    ) -> None:
        """Initialize RAGAS metrics."""
        from ragas.metrics.collections import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        self.context_precision_metric = ContextPrecision(llm=llm)
        self.context_recall_metric = ContextRecall(llm=llm)
        self.faithfulness_metric = Faithfulness(llm=llm)
        self.answer_relevancy_metric = AnswerRelevancy(
            llm=llm,
            embeddings=embeddings,
            strictness=strictness,
        )
        self.answer_correctness_metric = AnswerCorrectness(
            llm=llm,
            embeddings=embeddings,
        )
        self._close_targets = list(close_targets or [])
        self._runner = AsyncLoopRunner()
        self.provider_name = provider_name
        self.model_id = model_id
        self.embedding_provider_name = embedding_provider_name
        self.embedding_model_id = embedding_model_id

    def evaluate_sample(
        self,
        question: str,
        ground_truth: str,
        retrieved_contexts: list[str],
        response: str,
    ) -> dict[str, float | None]:
        """Evaluate one sample with RAGAS metrics."""
        results: dict[str, float | None] = {
            "context_precision": None,
            "context_recall": None,
            "faithfulness": None,
            "answer_relevancy": None,
            "answer_correctness": None,
        }

        if retrieved_contexts:
            results["context_precision"] = _score_metric(
                self.context_precision_metric.ascore(
                    user_input=question,
                    reference=ground_truth,
                    retrieved_contexts=retrieved_contexts,
                ),
                "context precision",
                run_coroutine=self._runner.run,
            )
            results["context_recall"] = _score_metric(
                self.context_recall_metric.ascore(
                    user_input=question,
                    reference=ground_truth,
                    retrieved_contexts=retrieved_contexts,
                ),
                "context recall",
                run_coroutine=self._runner.run,
            )
            results["faithfulness"] = _score_metric(
                self.faithfulness_metric.ascore(
                    user_input=question,
                    response=response,
                    retrieved_contexts=retrieved_contexts,
                ),
                "faithfulness",
                run_coroutine=self._runner.run,
            )

        results["answer_relevancy"] = _score_metric(
            self.answer_relevancy_metric.ascore(
                user_input=question,
                response=response,
            ),
            "answer relevancy",
            run_coroutine=self._runner.run,
        )
        results["answer_correctness"] = _score_metric(
            self.answer_correctness_metric.ascore(
                user_input=question,
                response=response,
                reference=ground_truth,
            ),
            "answer correctness",
            run_coroutine=self._runner.run,
        )
        return results

    def close(self) -> None:
        """Close shared clients owned by this evaluator."""
        seen_ids: set[int] = set()
        for target in self._close_targets:
            target_id = id(target)
            if target_id in seen_ids:
                continue
            seen_ids.add(target_id)
            close = getattr(target, "close", None)
            if callable(close):
                close_result = close()
                if inspect.isawaitable(close_result):
                    self._runner.run(close_result)
                continue
            aclose = getattr(target, "aclose", None)
            if callable(aclose):
                self._runner.run(aclose())
        self._runner.close()


def create_ragas_evaluator(
    config: PipelineConfig | None = None,
    llm: object | None = None,
    embeddings: object | None = None,
    strictness: int = 1,
) -> SampleEvaluator:
    """Create the default RAGAS evaluation backend."""
    if llm is not None and embeddings is not None:
        return RagasEvaluator(llm=llm, embeddings=embeddings, strictness=strictness)

    pipeline_config = config or PipelineConfig.from_env()
    ragas_llm, ragas_embeddings, close_targets = _build_ragas_clients(pipeline_config)
    return RagasEvaluator(
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        strictness=strictness,
        close_targets=close_targets,
        provider_name=pipeline_config.evaluation_provider,
        model_id=pipeline_config.evaluation_model,
        embedding_provider_name=pipeline_config.embedding_provider,
        embedding_model_id=pipeline_config.embedding_model,
    )


def _build_ragas_clients(config: PipelineConfig) -> tuple[object, object, list[object]]:
    """Build RAGAS-compatible LLM and embedding clients plus cleanup targets."""
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory

    if config.evaluation_provider == "azure_openai":
        from openai import AsyncAzureOpenAI

        azure_client = AsyncAzureOpenAI(
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            azure_endpoint=config.azure_openai_endpoint,
        )
        ragas_llm = llm_factory(
            config.evaluation_model,
            client=azure_client,
            provider="openai",
            max_tokens=8192,
            temperature=0,
        )
        ragas_embeddings = embedding_factory(
            "openai",
            model=config.embedding_model,
            client=azure_client,
        )
        return ragas_llm, ragas_embeddings, [azure_client]

    if config.evaluation_provider == "openai":
        from openai import AsyncOpenAI

        openai_client = AsyncOpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url or None,
        )
        ragas_llm = llm_factory(
            config.evaluation_model,
            client=openai_client,
            provider="openai",
            max_tokens=8192,
            temperature=0,
        )
        ragas_embeddings = embedding_factory(
            "openai",
            model=config.embedding_model,
            client=openai_client,
        )
        return ragas_llm, ragas_embeddings, [openai_client]

    raise ValueError(
        "Current RAGAS collection metrics require an OpenAI-compatible evaluation "
        "backend. Set EVALUATION_PROVIDER to azure_openai or openai.",
    )


def _build_evaluation_chat_model(config: PipelineConfig) -> object:
    """Build the configured judge LLM for RAGAS."""
    if config.evaluation_provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            api_key=config.azure_openai_api_key,
            azure_endpoint=config.azure_openai_endpoint,
            azure_deployment=config.evaluation_model,
            model=config.evaluation_model,
            api_version=config.azure_openai_api_version,
            temperature=0,
            max_completion_tokens=8192,
        )

    if config.evaluation_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url or None,
            model=config.evaluation_model,
            temperature=0,
            max_completion_tokens=8192,
        )

    if config.evaluation_provider == "watsonx":
        from langchain_ibm import ChatWatsonx

        return ChatWatsonx(
            model_id=config.evaluation_model,
            url=config.watsonx_url,
            apikey=config.watsonx_apikey,
            project_id=config.watsonx_project_id,
            params={
                "temperature": 0,
                "max_new_tokens": 8192,
                "top_p": 1.0,
            },
        )

    raise ValueError(f"Unsupported evaluation provider: {config.evaluation_provider}")


def _build_evaluation_embeddings_model(config: PipelineConfig) -> object:
    """Build the configured embeddings backend for RAGAS."""
    if config.embedding_provider == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            api_key=config.azure_openai_api_key,
            azure_endpoint=config.azure_openai_endpoint,
            azure_deployment=config.embedding_model,
            model=config.embedding_model,
            api_version=config.azure_openai_api_version,
        )

    if config.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url or None,
            model=config.embedding_model,
        )

    if config.embedding_provider == "watsonx":
        from langchain_ibm import WatsonxEmbeddings

        return WatsonxEmbeddings(
            model_id=config.embedding_model,
            url=config.watsonx_url,
            apikey=config.watsonx_apikey,
            project_id=config.watsonx_project_id,
        )

    raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")


def _build_evaluator_metadata(evaluator: SampleEvaluator) -> dict[str, object]:
    """Extract stable provider metadata from one evaluator when available."""
    metadata: dict[str, object] = {}
    provider_name = getattr(evaluator, "provider_name", None)
    model_id = getattr(evaluator, "model_id", None)
    embedding_provider_name = getattr(evaluator, "embedding_provider_name", None)
    embedding_model_id = getattr(evaluator, "embedding_model_id", None)
    if provider_name:
        metadata["evaluation_provider"] = str(provider_name)
    if model_id:
        metadata["evaluation_model"] = str(model_id)
    if embedding_provider_name:
        metadata["embedding_provider"] = str(embedding_provider_name)
    if embedding_model_id:
        metadata["embedding_model"] = str(embedding_model_id)
    return metadata


def _build_generation_metadata(llm_provider: LlmProvider) -> dict[str, object]:
    """Extract stable provider metadata from one generation provider."""
    metadata: dict[str, object] = {}
    provider_name = getattr(llm_provider, "provider_name", None)
    model_id = getattr(llm_provider, "model_id", None)
    if provider_name:
        metadata["generation_provider"] = str(provider_name)
    if model_id:
        metadata["generation_model"] = str(model_id)
    return metadata


def _create_default_evaluator(
    *,
    config: PipelineConfig | None,
    strictness: int,
) -> SampleEvaluator:
    """Create one evaluator while tolerating older patched factory signatures."""
    try:
        return create_ragas_evaluator(
            config=config,
            strictness=strictness,
        )
    except TypeError as exc:
        if "unexpected keyword argument 'config'" not in str(exc):
            raise
        return create_ragas_evaluator(strictness=strictness)


def generation_input_path(source_paths: SourcePaths, top_k: int) -> Path:
    """Return the generated-answer input path for evaluation."""
    return source_paths.generation_dir / f"answers_with_k_{top_k}.json"


def evaluation_output_path(source_paths: SourcePaths, top_k: int) -> Path:
    """Return the output path for detailed evaluation results."""
    return source_paths.results_dir / f"evaluation_with_k_{top_k}.json"


def evaluation_summary_path(source_paths: SourcePaths) -> Path:
    """Return the output path for the compact evaluation summary."""
    return source_paths.results_dir / "SUMMARY.json"


def evaluation_summary_with_k_path(source_paths: SourcePaths, top_k: int) -> Path:
    """Return the output path for one top-k-specific evaluation summary."""
    return source_paths.results_dir / f"SUMMARY_with_k_{top_k}.json"


def oracle_evaluation_output_path(source_paths: SourcePaths) -> Path:
    """Return the output path for detailed oracle evaluation results."""
    return source_paths.results_dir / "evaluation_oracle.json"


def oracle_evaluation_summary_path(source_paths: SourcePaths) -> Path:
    """Return the output path for the oracle evaluation summary."""
    return source_paths.results_dir / "SUMMARY_oracle.json"


def load_generated_answers(source_paths: SourcePaths, top_k: int) -> list[dict[str, object]]:
    """Load generated answers from the standard generation export."""
    input_path = generation_input_path(source_paths, top_k)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run generate first to create answers_with_k_<k>.json.",
        )

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("generation results payload must contain a JSON list under 'results'")
    return results


def load_ground_truth_map(labeled_test_set_path: Path) -> dict[str, dict[str, object]]:
    """Load ground-truth answers keyed by question ID."""
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


def load_evaluation_payload(output_path: Path) -> dict[str, object]:
    """Load one saved evaluation payload from disk."""
    if not output_path.exists():
        raise FileNotFoundError(f"Missing evaluation export: {output_path}")

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation export must contain a JSON object: {output_path}")
    return payload


def evaluate_generation_results(
    source_paths: SourcePaths,
    top_k: int,
    labeled_test_set_path: Path,
    evaluator: SampleEvaluator | None = None,
    config: PipelineConfig | None = None,
    strictness: int = 1,
    show_progress: bool = False,
) -> dict[str, object]:
    """Evaluate generated answers against ground-truth answers with RAGAS."""
    generated_answers = load_generated_answers(source_paths, top_k)
    ground_truth_map = load_ground_truth_map(labeled_test_set_path)
    created_evaluator = evaluator is None
    active_evaluator = evaluator or _create_default_evaluator(
        config=config,
        strictness=strictness,
    )
    try:
        results = _evaluate_generation_results(
            top_k=top_k,
            generated_answers=generated_answers,
            ground_truth_map=ground_truth_map,
            evaluator=active_evaluator,
            show_progress=show_progress,
        )
        return build_evaluation_payload(
            source_paths=source_paths,
            top_k=top_k,
            results=results,
            metadata=_build_evaluator_metadata(active_evaluator),
        )
    finally:
        if created_evaluator:
            _close_evaluator(active_evaluator)


def evaluate_generated_answer_item(
    *,
    item: dict[str, object],
    top_k: int,
    labeled_test_set_path: Path,
    evaluator: SampleEvaluator | None = None,
    config: PipelineConfig | None = None,
    strictness: int = 1,
) -> EvaluationSampleResult:
    """Evaluate one generated-answer row against the labeled test set."""
    results = evaluate_generated_answer_items(
        items=[item],
        top_k=top_k,
        labeled_test_set_path=labeled_test_set_path,
        evaluator=evaluator,
        config=config,
        strictness=strictness,
    )
    return results[0]


def evaluate_generated_answer_items(
    *,
    items: list[dict[str, object]],
    top_k: int,
    labeled_test_set_path: Path,
    evaluator: SampleEvaluator | None = None,
    config: PipelineConfig | None = None,
    strictness: int = 1,
) -> list[EvaluationSampleResult]:
    """Evaluate a selected subset of generated-answer rows in input order."""
    ground_truth_map = load_ground_truth_map(labeled_test_set_path)
    created_evaluator = evaluator is None
    active_evaluator = evaluator or _create_default_evaluator(
        config=config,
        strictness=strictness,
    )
    try:
        return _evaluate_generation_results(
            top_k=top_k,
            generated_answers=items,
            ground_truth_map=ground_truth_map,
            evaluator=active_evaluator,
        )
    finally:
        if created_evaluator:
            _close_evaluator(active_evaluator)


def write_evaluation_results(
    source_paths: SourcePaths,
    top_k: int,
    evaluation_payload: dict[str, object],
) -> tuple[Path, Path]:
    """Write detailed evaluation results plus per-k and combined summaries."""
    output_path = evaluation_output_path(source_paths, top_k)
    per_k_summary_path = evaluation_summary_with_k_path(source_paths, top_k)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(evaluation_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    per_k_summary_path.write_text(
        json.dumps(evaluation_payload["summary"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary_path = write_combined_evaluation_summary(source_paths)
    return output_path, summary_path


def rebuild_evaluation_summaries(source_paths: SourcePaths) -> tuple[Path, dict[str, Path]]:
    """Rebuild per-k and combined summaries from existing evaluation exports."""
    per_k_summaries = load_evaluation_summaries_by_k(source_paths)
    if not per_k_summaries:
        raise FileNotFoundError(
            f"Missing evaluation exports under {source_paths.results_dir}. "
            "Expected at least one evaluation_with_k_<k>.json file.",
        )

    per_k_paths: dict[str, Path] = {}
    for top_k, summary in per_k_summaries.items():
        summary_path = evaluation_summary_with_k_path(source_paths, int(top_k))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        per_k_paths[top_k] = summary_path

    summary_path = write_combined_evaluation_summary(source_paths)
    return summary_path, per_k_paths


def write_combined_evaluation_summary(source_paths: SourcePaths) -> Path:
    """Write one shared summary spanning every available evaluation_with_k file."""
    summary_path = evaluation_summary_path(source_paths)
    per_k_summaries = load_evaluation_summaries_by_k(source_paths)
    payload = {
        "source": source_paths.source,
        "k_values": [int(top_k) for top_k in per_k_summaries],
        "per_k": per_k_summaries,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary_path


def build_evaluation_payload_from_result_payloads(
    source_paths: SourcePaths,
    top_k: int,
    result_payloads: list[dict[str, object]],
    metadata: dict[str, object] | None = None,
    timestamp: str | None = None,
) -> dict[str, object]:
    """Build one evaluation payload from already-serialized result rows."""
    results = [
        evaluation_result_from_payload(result_payload)
        for result_payload in result_payloads
    ]
    return build_evaluation_payload(
        source_paths=source_paths,
        top_k=top_k,
        results=results,
        metadata=metadata,
        timestamp=timestamp,
    )


def load_evaluation_summaries_by_k(source_paths: SourcePaths) -> dict[str, dict[str, object]]:
    """Load per-k evaluation summaries from detailed evaluation exports."""
    per_k_summaries: dict[str, dict[str, object]] = {}
    for output_path in sorted(
        source_paths.results_dir.glob("evaluation_with_k_*.json"),
        key=_evaluation_output_path_sort_key,
    ):
        top_k = parse_top_k_from_evaluation_output_path(output_path)
        if top_k is None:
            continue
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        summary = payload.get("summary")
        if isinstance(summary, dict):
            per_k_summaries[str(top_k)] = summary
    return per_k_summaries


def evaluation_result_from_payload(result_payload: dict[str, object]) -> EvaluationSampleResult:
    """Deserialize one saved result row into an evaluation sample result."""
    metrics = result_payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    chunks = result_payload.get("chunks", [])
    if not isinstance(chunks, list):
        chunks = []
    contexts = result_payload.get("retrieved_contexts", [])
    if not isinstance(contexts, list):
        contexts = []
    chunk_ids = result_payload.get("retrieved_chunk_ids", [])
    if not isinstance(chunk_ids, list):
        chunk_ids = []

    return EvaluationSampleResult(
        question_id=str(result_payload.get("question_id", "")),
        question=str(result_payload.get("question", "")),
        ground_truth=str(result_payload.get("ground_truth", "")),
        answer=str(result_payload.get("answer", "")),
        chunks=[chunk for chunk in chunks if isinstance(chunk, dict)],
        retrieved_contexts=[str(context) for context in contexts],
        retrieved_chunk_ids=[str(chunk_id) for chunk_id in chunk_ids],
        retrieval_mode=str(result_payload.get("retrieval_mode", "vector")),
        k=int(result_payload.get("k", 0)),
        context_precision=_optional_float(metrics.get("context_precision")),
        context_recall=_optional_float(metrics.get("context_recall")),
        faithfulness=_optional_float(metrics.get("faithfulness")),
        answer_relevancy=_optional_float(metrics.get("answer_relevancy")),
        answer_correctness=_optional_float(metrics.get("answer_correctness")),
        retrieval_time_s=float(result_payload.get("retrieval_time_s", 0.0) or 0.0),
        generation_time_s=float(result_payload.get("generation_time_s", 0.0) or 0.0),
        evaluation_time_s=float(result_payload.get("evaluation_time_s", 0.0) or 0.0),
        complexity=_optional_str(result_payload.get("complexity")),
        reasoning_type=_optional_str(result_payload.get("reasoning_type")),
        error=_optional_str(result_payload.get("error")),
    )


def parse_top_k_from_evaluation_output_path(output_path: Path) -> int | None:
    """Extract top-k from an evaluation_with_k_<k>.json filename."""
    prefix = "evaluation_with_k_"
    suffix = ".json"
    name = output_path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    raw_value = name.removeprefix(prefix).removesuffix(suffix)
    try:
        return int(raw_value)
    except ValueError:
        return None


def _evaluation_output_path_sort_key(output_path: Path) -> tuple[int, str]:
    """Sort evaluation files numerically by top-k when possible."""
    top_k = parse_top_k_from_evaluation_output_path(output_path)
    if top_k is None:
        return (2**31 - 1, output_path.name)
    return (top_k, output_path.name)


def evaluate_oracle_results(
    source_paths: SourcePaths,
    labeled_test_set_path: Path,
    llm_provider: LlmProvider,
    evaluator: SampleEvaluator | None = None,
    config: PipelineConfig | None = None,
    strictness: int = 1,
    show_progress: bool = False,
) -> dict[str, object]:
    """Evaluate oracle-mode answers built from labeled evidence chunks with RAGAS."""
    """Evaluate oracle-mode answers built from labeled evidence chunks."""
    ground_truth_map = load_ground_truth_map(labeled_test_set_path)
    chunk_map = {chunk.chunk_id: chunk for chunk in load_source_chunks(source_paths)}
    created_evaluator = evaluator is None
    active_evaluator = evaluator or _create_default_evaluator(
        config=config,
        strictness=strictness,
    )
    try:
        results = _evaluate_oracle_results(
            source_paths=source_paths,
            ground_truth_map=ground_truth_map,
            chunk_map=chunk_map,
            llm_provider=llm_provider,
            evaluator=active_evaluator,
            show_progress=show_progress,
        )
        return build_oracle_evaluation_payload(
            source_paths=source_paths,
            results=results,
            metadata={
                **_build_evaluator_metadata(active_evaluator),
                **_build_generation_metadata(llm_provider),
            },
        )
    finally:
        if created_evaluator:
            _close_evaluator(active_evaluator)


def write_oracle_evaluation_results(
    source_paths: SourcePaths,
    evaluation_payload: dict[str, object],
) -> tuple[Path, Path]:
    """Write detailed oracle evaluation results and summary JSON."""
    output_path = oracle_evaluation_output_path(source_paths)
    summary_path = oracle_evaluation_summary_path(source_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(evaluation_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(evaluation_payload["summary"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path, summary_path


def _evaluate_generation_results(
    top_k: int,
    generated_answers: list[dict[str, object]],
    ground_truth_map: dict[str, dict[str, object]],
    evaluator: SampleEvaluator,
    show_progress: bool = False,
) -> list[EvaluationSampleResult]:
    """Evaluate one generation export in input order."""
    progress = (
        tqdm(total=len(generated_answers), desc="Evaluating answers", unit="query")
        if show_progress
        else None
    )
    indexed_results: list[EvaluationSampleResult] = []

    try:
        for item in generated_answers:
            indexed_results.append(
                _evaluate_generation_item(
                    item=item,
                    top_k=top_k,
                    ground_truth_map=ground_truth_map,
                    evaluator=evaluator,
                ),
            )
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    return indexed_results


def _evaluate_oracle_results(
    source_paths: SourcePaths,
    ground_truth_map: dict[str, dict[str, object]],
    chunk_map: dict[str, ChunkRecord],
    llm_provider: LlmProvider,
    evaluator: SampleEvaluator,
    show_progress: bool,
) -> list[EvaluationSampleResult]:
    """Evaluate oracle-mode answers in input order."""
    ground_truth_items = list(ground_truth_map.items())
    progress = (
        tqdm(total=len(ground_truth_items), desc="Evaluating oracle answers", unit="query")
        if show_progress
        else None
    )
    indexed_results: list[EvaluationSampleResult] = []

    try:
        for question_id, item in ground_truth_items:
            indexed_results.append(
                _evaluate_oracle_item(
                    source_paths=source_paths,
                    question_id=question_id,
                    item=item,
                    chunk_map=chunk_map,
                    llm_provider=llm_provider,
                    evaluator=evaluator,
                ),
            )
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    return indexed_results


def _evaluate_generation_item(
    item: dict[str, object],
    top_k: int,
    ground_truth_map: dict[str, dict[str, object]],
    evaluator: SampleEvaluator,
) -> EvaluationSampleResult:
    """Evaluate one generated answer item while isolating per-item failures."""
    question_id = str(item.get("question_id", ""))
    reference_item = ground_truth_map.get(question_id)
    question = str(item.get("query", ""))
    answer = _normalize_text(item.get("answer"))
    chunks = list(item.get("chunks", []))
    contexts = [str(chunk.get("text", "")) for chunk in chunks]
    chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
    base_error = _optional_str(item.get("error"))

    if reference_item is None:
        return EvaluationSampleResult(
            question_id=question_id,
            question=question,
            ground_truth="",
            answer=answer,
            chunks=chunks,
            retrieved_contexts=contexts,
            retrieved_chunk_ids=chunk_ids,
            retrieval_mode="vector",
            k=top_k,
            retrieval_time_s=0.0,
            generation_time_s=0.0,
            evaluation_time_s=0.0,
            error=base_error or f"Missing ground truth for question_id={question_id}",
        )

    question = str(reference_item.get("question", question))
    ground_truth = str(reference_item.get("ground_truth", "")).strip()
    try:
        evaluation_start = time.perf_counter()
        metrics = evaluator.evaluate_sample(
            question=question,
            ground_truth=ground_truth,
            retrieved_contexts=contexts,
            response=answer,
        )
        evaluation_time = time.perf_counter() - evaluation_start
        error = base_error
    except Exception as exc:
        metrics = {}
        evaluation_time = 0.0
        error = str(exc)

    return EvaluationSampleResult(
        question_id=question_id,
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        chunks=chunks,
        retrieved_contexts=contexts,
        retrieved_chunk_ids=chunk_ids,
        retrieval_mode="vector",
        k=top_k,
        context_precision=metrics.get("context_precision"),
        context_recall=metrics.get("context_recall"),
        faithfulness=metrics.get("faithfulness"),
        answer_relevancy=metrics.get("answer_relevancy"),
        answer_correctness=metrics.get("answer_correctness"),
        retrieval_time_s=0.0,
        generation_time_s=0.0,
        evaluation_time_s=evaluation_time,
        complexity=_optional_str(reference_item.get("complexity")),
        reasoning_type=_optional_str(reference_item.get("reasoning_type")),
        error=error,
    )


def _evaluate_oracle_item(
    source_paths: SourcePaths,
    question_id: str,
    item: dict[str, object],
    chunk_map: dict[str, ChunkRecord],
    llm_provider: LlmProvider,
    evaluator: SampleEvaluator,
) -> EvaluationSampleResult:
    """Evaluate one oracle sample while isolating per-item failures."""
    question = str(item.get("question", ""))
    ground_truth = str(item.get("ground_truth", "")).strip()
    retrieval_time = 0.0
    generation_time = 0.0
    evaluation_time = 0.0
    chunks: list[dict[str, object]] = []
    contexts: list[str] = []
    oracle_chunk_ids: list[str] = []
    answer = ""

    try:
        oracle_chunk_ids = resolve_oracle_chunk_ids(
            source=source_paths.source,
            labeled_item=item,
        )
        retrieval_start = time.perf_counter()
        oracle_chunks = build_oracle_chunks(
            source_paths=source_paths,
            chunk_map=chunk_map,
            chunk_ids=oracle_chunk_ids,
        )
        retrieval_time = time.perf_counter() - retrieval_start

        generation_start = time.perf_counter()
        generation = generate_answer(
            question=question,
            retrieved_chunks=oracle_chunks,
            llm_provider=llm_provider,
        )
        generation_time = time.perf_counter() - generation_start
        answer = _normalize_text(generation.get("answer"))
        chunks = [chunk.to_payload() for chunk in oracle_chunks]
        contexts = [chunk.text for chunk in oracle_chunks]

        evaluation_start = time.perf_counter()
        metrics = evaluator.evaluate_sample(
            question=question,
            ground_truth=ground_truth,
            retrieved_contexts=contexts,
            response=answer,
        )
        evaluation_time = time.perf_counter() - evaluation_start
        error = None
    except Exception as exc:
        metrics = {}
        error = str(exc)

    return EvaluationSampleResult(
        question_id=question_id,
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        chunks=chunks,
        retrieved_contexts=contexts,
        retrieved_chunk_ids=oracle_chunk_ids,
        retrieval_mode="oracle",
        k=len(oracle_chunk_ids),
        context_precision=metrics.get("context_precision"),
        context_recall=metrics.get("context_recall"),
        faithfulness=metrics.get("faithfulness"),
        answer_relevancy=metrics.get("answer_relevancy"),
        answer_correctness=metrics.get("answer_correctness"),
        retrieval_time_s=retrieval_time,
        generation_time_s=generation_time,
        evaluation_time_s=evaluation_time,
        complexity=_optional_str(item.get("complexity")),
        reasoning_type=_optional_str(item.get("reasoning_type")),
        error=error,
    )


def build_evaluation_payload(
    source_paths: SourcePaths,
    top_k: int,
    results: list[EvaluationSampleResult],
    metadata: dict[str, object] | None = None,
    timestamp: str | None = None,
) -> dict[str, object]:
    """Build the top-level standard evaluation payload."""
    summary_timestamp = timestamp or datetime.now(UTC).replace(tzinfo=None).isoformat()
    return {
        "metadata": {
            "timestamp": summary_timestamp,
            "source": source_paths.source,
            "top_k": top_k,
            "evaluation_type": "standard",
            **(metadata or {}),
        },
        "status": "SUCCESS",
        "results": [result.to_payload() for result in results],
        "summary": _build_summary(
            source=source_paths.source,
            top_k=top_k,
            timestamp=summary_timestamp,
            results=results,
        ),
    }


def build_oracle_evaluation_payload(
    source_paths: SourcePaths,
    results: list[EvaluationSampleResult],
    metadata: dict[str, object] | None = None,
    timestamp: str | None = None,
) -> dict[str, object]:
    """Build the top-level oracle evaluation payload."""
    summary_timestamp = timestamp or datetime.now(UTC).replace(tzinfo=None).isoformat()
    return {
        "metadata": {
            "timestamp": summary_timestamp,
            "source": source_paths.source,
            "evaluation_type": "oracle",
            **(metadata or {}),
        },
        "status": "SUCCESS",
        "results": [result.to_payload() for result in results],
        "summary": _build_summary(
            source=source_paths.source,
            top_k=None,
            timestamp=summary_timestamp,
            results=results,
        ),
    }


def resolve_oracle_chunk_ids(source: str, labeled_item: dict[str, object]) -> list[str]:
    """Resolve oracle chunk IDs for one labeled test-set item and source."""
    label_field, explicit_field = resolve_label_fields(source)

    labels = labeled_item.get(label_field, {})
    if isinstance(labels, dict) and labels:
        resolved = [
            str(chunk_id)
            for chunk_id, label in labels.items()
            if int(label) >= 1
        ]
        if resolved:
            return resolved

    explicit_ids = labeled_item.get(explicit_field, [])
    if isinstance(explicit_ids, list) and explicit_ids:
        return [str(chunk_id) for chunk_id in explicit_ids]

    raise ValueError(f"No oracle chunk IDs found for source={source}")


def build_oracle_chunks(
    source_paths: SourcePaths,
    chunk_map: dict[str, ChunkRecord],
    chunk_ids: list[str],
) -> list[RetrievedChunk]:
    """Build retrieved-chunk records from oracle chunk IDs."""
    oracle_chunks: list[RetrievedChunk] = []
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        if chunk_id not in chunk_map:
            raise KeyError(
                f"Oracle chunk_id={chunk_id} not found in {source_paths.chunks_dir}",
            )
        chunk = chunk_map[chunk_id]
        oracle_chunks.append(
            RetrievedChunk(
                source=chunk.source,
                document_id=chunk.document_id,
                text=chunk.text,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                token_start=chunk.token_start,
                token_end=chunk.token_end,
                chunk_index=chunk.chunk_index,
                chunk_id=chunk.chunk_id,
                total_chunks=chunk.total_chunks,
                headers=chunk.headers,
                score=1.0,
                rank=rank,
            ),
        )
    return oracle_chunks


def _score_metric(
    coroutine: Any,
    metric_name: str,
    run_coroutine: Any = None,
) -> float | None:
    """Evaluate one RAGAS metric and normalize failures to `None`."""
    try:
        runner = run_coroutine or _run_coroutine_sync
        result = runner(coroutine)
        return float(getattr(result, "value", result))
    except Exception as exc:
        logger.error("%s evaluation failed: %s", metric_name, exc)
        return None


def _build_summary(
    source: str,
    top_k: int | None,
    timestamp: str,
    results: list[EvaluationSampleResult],
) -> dict[str, object]:
    """Build summary statistics for evaluation outputs."""
    metrics = (
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
    )
    summary: dict[str, object] = {
        "timestamp": timestamp,
        "source": source,
        "total_queries": len(results),
        "successful": sum(1 for result in results if result.error is None),
        "failed": sum(1 for result in results if result.error is not None),
        "averages": {
            metric: _average_metric(results, metric)
            for metric in metrics
        },
    }
    if top_k is not None:
        summary["top_k"] = top_k
    return summary


def _average_metric(results: list[EvaluationSampleResult], metric_name: str) -> float | None:
    """Compute one average metric value across non-null scores."""
    values = [
        getattr(result, metric_name)
        for result in results
        if getattr(result, metric_name) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _optional_str(value: object) -> str | None:
    """Normalize nullable fields into optional strings."""
    if value is None:
        return None
    return str(value)


def _optional_float(value: object) -> float | None:
    """Normalize nullable metric values into optional floats."""
    if value is None:
        return None
    return float(value)


def _normalize_text(value: object) -> str:
    """Normalize nullable answer values into strings."""
    if value is None:
        return ""
    return str(value).strip()


def _close_evaluator(evaluator: SampleEvaluator) -> None:
    """Best-effort cleanup for evaluators that hold resources."""
    close = getattr(evaluator, "close", None)
    if callable(close):
        close()


def _run_coroutine_sync(coroutine: Any) -> Any:
    """Run one coroutine synchronously, even inside an active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    result_future: Future[Any] = Future()

    def runner() -> None:
        try:
            result_future.set_result(asyncio.run(coroutine))
        except BaseException as exc:  # pragma: no cover - defensive cross-thread propagation
            result_future.set_exception(exc)

    import threading

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    return result_future.result()
