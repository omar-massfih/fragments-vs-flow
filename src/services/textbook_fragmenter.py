"""Textbook chunk fragmentation helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.config import DEFAULT_WATSONX_MODEL_ID, PipelineConfig
from src.providers.llm import LlmProvider, create_llm_provider


DEFAULT_FRAGMENTATION_MAX_TOKENS = 450
DEFAULT_FRAGMENTATION_TEMPERATURE = 0.2
DEFAULT_FRAGMENTATION_MAX_ATTEMPTS = 3
DEFAULT_FRAGMENTATION_RETRY_DELAYS_S = (2.0, 4.0)


@dataclass(frozen=True)
class TextbookFragmentationResult:
    """Outcome of fragmenting one textbook chunk."""

    text: str
    used_fallback: bool


class TextbookFragmenter:
    """Rewrite textbook chunks into concise study-note fragments."""

    def __init__(
        self,
        llm_provider: LlmProvider,
        *,
        model_id: str = DEFAULT_WATSONX_MODEL_ID,
        max_attempts: int = DEFAULT_FRAGMENTATION_MAX_ATTEMPTS,
        retry_delays_s: tuple[float, ...] = DEFAULT_FRAGMENTATION_RETRY_DELAYS_S,
    ) -> None:
        """Store provider and retry policy."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.llm_provider = llm_provider
        self.model_id = model_id
        self.max_attempts = max_attempts
        self.retry_delays_s = retry_delays_s

    def build_prompt(self, chunk_text: str, metadata: dict[str, Any] | None = None) -> str:
        """Build the textbook-to-fragments rewrite prompt."""
        metadata = metadata or {}
        section_hint = ""
        headers = metadata.get("headers", {})
        if isinstance(headers, dict):
            non_empty_headers = [str(value).strip() for value in headers.values() if str(value).strip()]
            if non_empty_headers:
                section_hint = f"\nSECTION CONTEXT: {' | '.join(non_empty_headers)}"

        return f"""You are a technical writing assistant. Your task is to rewrite textbook prose into fragment-style study bullets.

CRITICAL RULES:
- Output ONLY the rewritten fragment text itself
- Do NOT include notes, comments, or explanations about the rewriting process
- Do NOT add introductory or concluding statements
- Keep the content in the same order as the input

CONTENT RULES:
- Preserve ALL technical terms, definitions, formulas, numbers, and named concepts exactly
- Keep the same chunk boundaries and information content
- Convert flowing prose into concise bullet-point or fragment style
- Remove transitions, connective prose, and rhetorical framing
- Do NOT add new information or delete information
- Prefer short, study-note style fragments over full narrative sentences

INPUT TEXT:{section_hint}
{chunk_text}

OUTPUT (fragment-style text only, no preamble):"""

    def fragment_chunk(
        self,
        chunk_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> TextbookFragmentationResult:
        """Rewrite one textbook chunk, falling back to the original text on failure."""
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
                return TextbookFragmentationResult(text=rewritten, used_fallback=False)
            except Exception as exc:  # pragma: no cover
                last_error = exc
                if attempt == self.max_attempts:
                    break
                delay_index = min(attempt - 1, len(self.retry_delays_s) - 1)
                time.sleep(self.retry_delays_s[delay_index])

        _ = last_error
        return TextbookFragmentationResult(text=normalized_original, used_fallback=True)


def create_textbook_fragmenter(model_id: str | None = None) -> TextbookFragmenter:
    """Create the default env-configured textbook fragmenter."""
    pipeline_config = PipelineConfig.from_env() if model_id is None else None
    selected_model_id = (
        model_id
        or getattr(pipeline_config, "generation_model", "")
        or DEFAULT_WATSONX_MODEL_ID
    )
    llm_provider = create_llm_provider(
        config=pipeline_config,
        model_id=selected_model_id,
        temperature=DEFAULT_FRAGMENTATION_TEMPERATURE,
        max_tokens=DEFAULT_FRAGMENTATION_MAX_TOKENS,
    )
    return TextbookFragmenter(llm_provider=llm_provider, model_id=selected_model_id)
