"""Embedding provider interfaces."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol

from src.config import PipelineConfig
from tqdm import tqdm


class EmbeddingProvider(Protocol):
    """Interface for converting text batches into embedding vectors."""

    provider_name: str
    model_id: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""


class RetryingEmbeddingProvider:
    """Embedding adapter with batching and bounded retries."""

    def __init__(
        self,
        client: object,
        *,
        provider_name: str = "azure_openai",
        model_id: str = "",
        batch_size: int = 32,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        show_progress: bool = False,
    ) -> None:
        """Store the embedding client and retry policy."""
        self.client = client
        self.provider_name = provider_name
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.show_progress = show_progress

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches with simple retry handling."""
        if not texts:
            return []

        embeddings: list[list[float]] = []
        batch_starts = range(0, len(texts), self.batch_size)
        if self.show_progress:
            batch_starts = tqdm(
                batch_starts,
                total=(len(texts) + self.batch_size - 1) // self.batch_size,
                desc="Embedding batches",
                unit="batch",
            )
        for batch_start in batch_starts:
            batch = texts[batch_start : batch_start + self.batch_size]
            embeddings.extend(self._embed_batch(batch))
        return embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch with bounded retries."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.embed_documents(texts)
            except Exception as exc:  # pragma: no cover - retry branch tested via fake client
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_delay_seconds)
        assert last_error is not None
        raise RuntimeError(
            "Embedding batch failed after retries",
        ) from last_error


class AzureEmbeddingProvider(RetryingEmbeddingProvider):
    """Azure OpenAI embedding adapter with batched retries."""


class OpenAIEmbeddingProvider(RetryingEmbeddingProvider):
    """OpenAI embedding adapter with batched retries."""


class WatsonxEmbeddingProvider(RetryingEmbeddingProvider):
    """watsonx embedding adapter with batched retries."""


@dataclass
class OpenAIEmbeddingClientAdapter:
    """Small adapter for the official OpenAI embeddings client."""

    client: object
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts through the OpenAI embeddings API."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]


class PlaceholderEmbeddingProvider:
    """Temporary embedding provider used until indexing is implemented."""

    provider_name = "placeholder"
    model_id = ""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Reject embedding calls until the real backend is wired in."""
        raise NotImplementedError(
            f"Embeddings are not implemented yet for {len(texts)} text(s).",
        )


def create_embedding_provider(
    config: PipelineConfig | None = None,
    client: object | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    batch_size: int = 32,
    max_retries: int = 3,
    retry_delay_seconds: float = 1.0,
    show_progress: bool = False,
) -> EmbeddingProvider:
    """Create the configured embedding provider backend."""
    pipeline_config = config or PipelineConfig.from_env()
    selected_provider = provider or pipeline_config.embedding_provider
    selected_model_id = model_id or pipeline_config.embedding_model

    provider_kwargs = {
        "provider_name": selected_provider,
        "model_id": selected_model_id,
        "batch_size": batch_size,
        "max_retries": max_retries,
        "retry_delay_seconds": retry_delay_seconds,
        "show_progress": show_progress,
    }

    if client is not None:
        if selected_provider == "azure_openai":
            return AzureEmbeddingProvider(client=client, **provider_kwargs)
        if selected_provider == "openai":
            return OpenAIEmbeddingProvider(client=client, **provider_kwargs)
        if selected_provider == "watsonx":
            return WatsonxEmbeddingProvider(client=client, **provider_kwargs)
        raise ValueError(f"Unsupported embedding provider: {selected_provider}")

    if selected_provider == "azure_openai":
        try:
            from langchain_openai import AzureOpenAIEmbeddings

            azure_client = AzureOpenAIEmbeddings(
                api_key=pipeline_config.azure_openai_api_key,
                azure_endpoint=pipeline_config.azure_openai_endpoint,
                azure_deployment=selected_model_id,
                model=selected_model_id,
                api_version=pipeline_config.azure_openai_api_version,
            )
        except ModuleNotFoundError:
            from openai import AzureOpenAI

            azure_client = OpenAIEmbeddingClientAdapter(
                client=AzureOpenAI(
                    api_key=pipeline_config.azure_openai_api_key,
                    azure_endpoint=pipeline_config.azure_openai_endpoint,
                    api_version=pipeline_config.azure_openai_api_version,
                ),
                model_name=selected_model_id,
            )
        return AzureEmbeddingProvider(client=azure_client, **provider_kwargs)

    if selected_provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings

            openai_client = OpenAIEmbeddings(
                api_key=pipeline_config.openai_api_key,
                base_url=pipeline_config.openai_base_url or None,
                model=selected_model_id,
            )
        except ModuleNotFoundError:
            from openai import OpenAI

            openai_client = OpenAIEmbeddingClientAdapter(
                client=OpenAI(
                    api_key=pipeline_config.openai_api_key,
                    base_url=pipeline_config.openai_base_url or None,
                ),
                model_name=selected_model_id,
            )
        return OpenAIEmbeddingProvider(client=openai_client, **provider_kwargs)

    if selected_provider == "watsonx":
        from langchain_ibm import WatsonxEmbeddings

        watsonx_client = WatsonxEmbeddings(
            model_id=selected_model_id,
            url=pipeline_config.watsonx_url,
            apikey=pipeline_config.watsonx_apikey,
            project_id=pipeline_config.watsonx_project_id,
        )
        return WatsonxEmbeddingProvider(client=watsonx_client, **provider_kwargs)

    raise ValueError(f"Unsupported embedding provider: {selected_provider}")
