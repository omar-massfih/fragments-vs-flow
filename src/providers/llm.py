"""LLM provider interfaces for answer generation."""

from __future__ import annotations

from typing import Any, Protocol

from src.config import PipelineConfig


DEFAULT_GENERATION_TEMPERATURE = 0.0
DEFAULT_GENERATION_MAX_TOKENS = 1000
DEFAULT_GENERATION_TOP_P = 1.0


class LlmProvider(Protocol):
    """Interface for answer-generation LLM backends."""

    provider_name: str
    model_id: str

    def generate_text(self, prompt: str) -> str:
        """Generate one answer from a prompt."""

    async def agenerate_text(self, prompt: str) -> str:
        """Generate one answer from a prompt asynchronously."""


class ChatModelLlmProvider:
    """Generic chat-model-backed answer-generation adapter."""

    def __init__(
        self,
        client: object,
        *,
        provider_name: str = "watsonx",
        model_id: str = "",
    ) -> None:
        """Store the initialized chat client and provider metadata."""
        self.client = client
        self.provider_name = provider_name
        self.model_id = model_id

    def generate_text(self, prompt: str) -> str:
        """Generate one answer string from the provider response."""
        response = self.client.invoke(prompt)
        return _normalize_response_text(response)

    async def agenerate_text(self, prompt: str) -> str:
        """Generate one answer string through the async provider surface."""
        response = await self.client.ainvoke(prompt)
        return _normalize_response_text(response)


class WatsonxLlmProvider(ChatModelLlmProvider):
    """watsonx-backed answer-generation adapter."""


class AzureOpenAILlmProvider(ChatModelLlmProvider):
    """Azure OpenAI-backed answer-generation adapter."""


class OpenAILlmProvider(ChatModelLlmProvider):
    """OpenAI-backed answer-generation adapter."""


class PlaceholderLlmProvider:
    """Temporary LLM provider used for explicit placeholder-only tests."""

    provider_name = "placeholder"
    model_id = ""

    def generate_text(self, prompt: str) -> str:
        """Reject generation calls until the real backend is wired in."""
        raise NotImplementedError(
            f"Generation is not implemented yet for prompt length {len(prompt)}.",
        )

    async def agenerate_text(self, prompt: str) -> str:
        """Reject async generation calls until the real backend is wired in."""
        raise NotImplementedError(
            f"Generation is not implemented yet for prompt length {len(prompt)}.",
        )


def _normalize_response_text(response: Any) -> str:
    """Normalize provider responses into stripped text."""
    if hasattr(response, "content"):
        return str(response.content).strip()
    return str(response).strip()


def create_llm_provider(
    config: PipelineConfig | None = None,
    client: object | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    temperature: float = DEFAULT_GENERATION_TEMPERATURE,
    max_tokens: int = DEFAULT_GENERATION_MAX_TOKENS,
    top_p: float = DEFAULT_GENERATION_TOP_P,
) -> LlmProvider:
    """Create the configured answer-generation backend."""
    pipeline_config = config or PipelineConfig.from_env()
    selected_provider = provider or pipeline_config.generation_provider
    selected_model_id = model_id or pipeline_config.generation_model

    if client is not None:
        if selected_provider == "watsonx":
            return WatsonxLlmProvider(
                client=client,
                provider_name=selected_provider,
                model_id=selected_model_id,
            )
        if selected_provider == "azure_openai":
            return AzureOpenAILlmProvider(
                client=client,
                provider_name=selected_provider,
                model_id=selected_model_id,
            )
        if selected_provider == "openai":
            return OpenAILlmProvider(
                client=client,
                provider_name=selected_provider,
                model_id=selected_model_id,
            )
        raise ValueError(f"Unsupported generation provider: {selected_provider}")

    if selected_provider == "watsonx":
        from langchain_ibm import ChatWatsonx

        watsonx_client = ChatWatsonx(
            model_id=selected_model_id,
            url=pipeline_config.watsonx_url,
            apikey=pipeline_config.watsonx_apikey,
            project_id=pipeline_config.watsonx_project_id,
            params={
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "top_p": top_p,
            },
        )
        return WatsonxLlmProvider(
            client=watsonx_client,
            provider_name=selected_provider,
            model_id=selected_model_id,
        )

    if selected_provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI

        azure_client = AzureChatOpenAI(
            api_key=pipeline_config.azure_openai_api_key,
            azure_endpoint=pipeline_config.azure_openai_endpoint,
            azure_deployment=selected_model_id,
            api_version=pipeline_config.azure_openai_api_version,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_tokens,
        )
        return AzureOpenAILlmProvider(
            client=azure_client,
            provider_name=selected_provider,
            model_id=selected_model_id,
        )

    if selected_provider == "openai":
        from langchain_openai import ChatOpenAI

        openai_client = ChatOpenAI(
            api_key=pipeline_config.openai_api_key,
            base_url=pipeline_config.openai_base_url or None,
            model=selected_model_id,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_tokens,
        )
        return OpenAILlmProvider(
            client=openai_client,
            provider_name=selected_provider,
            model_id=selected_model_id,
        )

    raise ValueError(f"Unsupported generation provider: {selected_provider}")
