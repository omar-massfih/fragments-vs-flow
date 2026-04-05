"""Chunk re-narrativization helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.config import DEFAULT_WATSONX_MODEL_ID, PipelineConfig
from src.providers.llm import LlmProvider, create_llm_provider


DEFAULT_RENARRATION_MAX_TOKENS = 500
DEFAULT_RENARRATION_TEMPERATURE = 0.3
DEFAULT_RENARRATION_MAX_ATTEMPTS = 3
DEFAULT_RENARRATION_RETRY_DELAYS_S = (2.0, 4.0)


@dataclass(frozen=True)
class RenarrationResult:
    """Outcome of re-narrativizing one chunk."""

    text: str
    used_fallback: bool


class Renarrator:
    """Rewrite fragmented slide chunks into narrative prose."""

    def __init__(
        self,
        llm_provider: LlmProvider,
        *,
        model_id: str = DEFAULT_WATSONX_MODEL_ID,
        max_attempts: int = DEFAULT_RENARRATION_MAX_ATTEMPTS,
        retry_delays_s: tuple[float, ...] = DEFAULT_RENARRATION_RETRY_DELAYS_S,
    ) -> None:
        """Store provider and retry policy."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.llm_provider = llm_provider
        self.model_id = model_id
        self.max_attempts = max_attempts
        self.retry_delays_s = retry_delays_s

    def build_prompt(self, chunk_text: str, metadata: dict[str, Any] | None = None) -> str:
        """Build the slide-to-narrative rewrite prompt."""
        metadata = metadata or {}
        page_info = ""
        if metadata.get("page_start"):
            page_info = f" from slide {metadata['page_start']}"

        return f"""You are a technical writing assistant. Your task is to rewrite fragmented slide text into flowing narrative prose.

CRITICAL RULES:
- Output ONLY the rewritten narrative text itself
- Do NOT include any notes, comments, or explanations about the rewriting process
- Do NOT add phrases like "Note:", "Re-narrativized version:", "This renarration...", etc.
- Do NOT add introductory or concluding statements
- Start writing the narrative content immediately

CONTENT RULES:
- Preserve ALL technical terms, definitions, and formulas exactly as written
- Convert bullet points into complete, grammatically correct sentences
- Add natural transitions between concepts for logical flow
- Maintain the same technical depth - do not add new information
- Use present tense and active voice where appropriate
- Remove slide-specific formatting (bullets, slide numbers, etc.)

INPUT TEXT{page_info}:
{chunk_text}

OUTPUT (narrative text only, no preamble):"""

    def renarrate_chunk(
        self,
        chunk_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> RenarrationResult:
        """Rewrite one chunk, falling back to the original text after repeated failures."""
        normalized_original = chunk_text.strip()
        if not normalized_original:
            raise ValueError("chunk_text cannot be empty")

        prompt = self.build_prompt(chunk_text=normalized_original, metadata=metadata)
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                rewritten = self.llm_provider.generate_text(prompt).strip()
                if not rewritten:
                    raise ValueError("Empty response from model")
                return RenarrationResult(text=rewritten, used_fallback=False)
            except Exception as exc:  # pragma: no cover - retry branch exercised in tests
                last_error = exc
                if attempt == self.max_attempts:
                    break
                delay_index = min(attempt - 1, len(self.retry_delays_s) - 1)
                time.sleep(self.retry_delays_s[delay_index])

        _ = last_error
        return RenarrationResult(text=normalized_original, used_fallback=True)


def create_renarrator(model_id: str | None = None) -> Renarrator:
    """Create the default env-configured renarrator."""
    pipeline_config = PipelineConfig.from_env() if model_id is None else None
    selected_model_id = (
        model_id
        or getattr(pipeline_config, "generation_model", "")
        or DEFAULT_WATSONX_MODEL_ID
    )
    llm_provider = create_llm_provider(
        config=pipeline_config,
        model_id=selected_model_id,
        temperature=DEFAULT_RENARRATION_TEMPERATURE,
        max_tokens=DEFAULT_RENARRATION_MAX_TOKENS,
    )
    return Renarrator(llm_provider=llm_provider, model_id=selected_model_id)
