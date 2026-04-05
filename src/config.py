"""Configuration models and environment loading for the pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_AZURE_OPENAI_API_VERSION = "2024-02-15-preview"
DEFAULT_WATSONX_MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
DEFAULT_GENERATION_PROVIDER = "watsonx"
DEFAULT_EVALUATION_PROVIDER = "azure_openai"
DEFAULT_EMBEDDING_PROVIDER = "azure_openai"
VALID_PROVIDERS = {"watsonx", "azure_openai", "openai"}


def _normalize_provider(value: str, env_key: str) -> str:
    """Normalize and validate one configured provider name."""
    normalized = value.strip().lower()
    if normalized not in VALID_PROVIDERS:
        raise ValueError(
            f"{env_key} must be one of {', '.join(sorted(VALID_PROVIDERS))}",
        )
    return normalized


def _resolve_role_model(
    *,
    role_label: str,
    provider: str,
    configured_value: str,
    watsonx_value: str,
    watsonx_env_key: str,
    azure_value: str,
    azure_env_key: str,
) -> tuple[str, list[str]]:
    """Resolve one role-specific model setting plus any missing env names."""
    if configured_value:
        return configured_value, []
    if provider == "watsonx":
        return watsonx_value, [watsonx_env_key] if not watsonx_value else []
    if provider == "azure_openai":
        return azure_value, [azure_env_key] if not azure_value else []
    return "", [role_label]


@dataclass(frozen=True)
class PipelineConfig:
    """Provider configuration for the pipeline."""

    generation_provider: str
    evaluation_provider: str
    embedding_provider: str
    generation_model: str
    evaluation_model: str
    embedding_model: str
    openai_api_key: str
    openai_base_url: str
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_embedding_deployment: str
    azure_openai_deployment_name: str
    azure_openai_api_version: str
    watsonx_apikey: str
    watsonx_url: str
    watsonx_project_id: str
    watsonx_model_id: str
    watsonx_embedding_model_id: str

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "PipelineConfig":
        """Load pipeline configuration from environment variables."""
        if env_file is None:
            load_dotenv()
        else:
            load_dotenv(dotenv_path=Path(env_file), override=False)

        generation_provider = _normalize_provider(
            os.getenv("GENERATION_PROVIDER", DEFAULT_GENERATION_PROVIDER),
            "GENERATION_PROVIDER",
        )
        evaluation_provider = _normalize_provider(
            os.getenv("EVALUATION_PROVIDER", DEFAULT_EVALUATION_PROVIDER),
            "EVALUATION_PROVIDER",
        )
        embedding_provider = _normalize_provider(
            os.getenv("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER),
            "EMBEDDING_PROVIDER",
        )

        if evaluation_provider == "watsonx":
            raise ValueError(
                "EVALUATION_PROVIDER=watsonx is not supported in the thesis artifact. "
                "Use azure_openai or openai.",
            )

        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()

        azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        azure_openai_deployment_name = os.getenv(
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "",
        ).strip()
        azure_openai_embedding_deployment = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            azure_openai_deployment_name,
        ).strip()
        azure_openai_api_version = os.getenv(
            "AZURE_OPENAI_API_VERSION",
            DEFAULT_AZURE_OPENAI_API_VERSION,
        ).strip()

        watsonx_apikey = os.getenv("WATSONX_APIKEY", "").strip()
        watsonx_url = os.getenv("WATSONX_URL", "").strip()
        watsonx_project_id = os.getenv("WATSONX_PROJECT_ID", "").strip()
        watsonx_model_id = os.getenv(
            "WATSONX_MODEL_ID",
            DEFAULT_WATSONX_MODEL_ID,
        ).strip()
        watsonx_embedding_model_id = os.getenv(
            "WATSONX_EMBEDDING_MODEL_ID",
            "",
        ).strip()

        generation_model, missing_generation = _resolve_role_model(
            role_label="GENERATION_MODEL",
            provider=generation_provider,
            configured_value=os.getenv("GENERATION_MODEL", "").strip(),
            watsonx_value=watsonx_model_id,
            watsonx_env_key="WATSONX_MODEL_ID",
            azure_value=azure_openai_deployment_name,
            azure_env_key="AZURE_OPENAI_DEPLOYMENT_NAME",
        )
        evaluation_model, missing_evaluation = _resolve_role_model(
            role_label="EVALUATION_MODEL",
            provider=evaluation_provider,
            configured_value=os.getenv("EVALUATION_MODEL", "").strip(),
            watsonx_value=watsonx_model_id,
            watsonx_env_key="WATSONX_MODEL_ID",
            azure_value=azure_openai_deployment_name,
            azure_env_key="AZURE_OPENAI_DEPLOYMENT_NAME",
        )
        embedding_model, missing_embedding = _resolve_role_model(
            role_label="EMBEDDING_MODEL",
            provider=embedding_provider,
            configured_value=os.getenv("EMBEDDING_MODEL", "").strip(),
            watsonx_value=watsonx_embedding_model_id,
            watsonx_env_key="WATSONX_EMBEDDING_MODEL_ID",
            azure_value=azure_openai_embedding_deployment,
            azure_env_key="AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        )

        missing: list[str] = []
        missing.extend(missing_generation)
        missing.extend(missing_evaluation)
        missing.extend(missing_embedding)

        selected_providers = {generation_provider, evaluation_provider, embedding_provider}
        if "azure_openai" in selected_providers:
            required_azure_values = {
                "AZURE_OPENAI_API_KEY": azure_openai_api_key,
                "AZURE_OPENAI_ENDPOINT": azure_openai_endpoint,
            }
            for key, value in required_azure_values.items():
                if not value:
                    missing.append(key)

        if "watsonx" in selected_providers:
            required_watsonx_values = {
                "WATSONX_APIKEY": watsonx_apikey,
                "WATSONX_URL": watsonx_url,
                "WATSONX_PROJECT_ID": watsonx_project_id,
            }
            for key, value in required_watsonx_values.items():
                if not value:
                    missing.append(key)

        if "openai" in selected_providers and not openai_api_key:
            missing.append("OPENAI_API_KEY")

        deduped_missing = sorted(set(item for item in missing if item))
        if deduped_missing:
            raise ValueError(
                "Missing required environment variables: " + ", ".join(deduped_missing),
            )

        return cls(
            generation_provider=generation_provider,
            evaluation_provider=evaluation_provider,
            embedding_provider=embedding_provider,
            generation_model=generation_model,
            evaluation_model=evaluation_model,
            embedding_model=embedding_model,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            azure_openai_api_key=azure_openai_api_key,
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_embedding_deployment=azure_openai_embedding_deployment,
            azure_openai_deployment_name=azure_openai_deployment_name,
            azure_openai_api_version=azure_openai_api_version,
            watsonx_apikey=watsonx_apikey,
            watsonx_url=watsonx_url,
            watsonx_project_id=watsonx_project_id,
            watsonx_model_id=watsonx_model_id,
            watsonx_embedding_model_id=watsonx_embedding_model_id,
        )
