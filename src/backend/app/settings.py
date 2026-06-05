from __future__ import annotations

from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LlmProvider = Literal["openai", "azure"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8", extra="ignore")

    # Plug-and-play: set LLM_PROVIDER=openai or LLM_PROVIDER=azure
    llm_provider: LlmProvider = "openai"

    # OpenAI (platform API) — required when llm_provider=openai
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-5-mini"
    openai_chat_max_tokens: int = 64000
    openai_embed_model: str = "text-embedding-3-small"

    # Azure OpenAI — required when llm_provider=azure
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_embed_api_version: str = "2024-02-01"
    azure_openai_embed_deployment: str = "text-embedding-3-small"
    azure_openai_chat_deployment: str = "gpt-5-mini"
    azure_openai_chat_api_version: str = "2024-12-01-preview"

    # Weaviate
    weaviate_url: str = "http://localhost:8080"
    weaviate_grpc_host: str = "localhost"
    weaviate_grpc_port: int = 50051
    weaviate_api_key: str | None = None
    datasets_collection: str = "Dataset"

    # Demo: faster technical agent (RLM with lower limits, 1 resource/dataset, cap datasets)
    agora_demo_mode: bool = False

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @model_validator(mode="after")
    def _validate_active_provider(self) -> "Settings":
        if self.llm_provider == "openai":
            if not (self.openai_api_key or "").strip():
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        elif self.llm_provider == "azure":
            if not (self.azure_openai_api_key or "").strip():
                raise ValueError("AZURE_OPENAI_API_KEY is required when LLM_PROVIDER=azure")
            if not (self.azure_openai_endpoint or "").strip():
                raise ValueError("AZURE_OPENAI_ENDPOINT is required when LLM_PROVIDER=azure")
        return self

    @property
    def uses_openai(self) -> bool:
        return self.llm_provider == "openai"

    @property
    def uses_azure(self) -> bool:
        return self.llm_provider == "azure"

    @property
    def chat_pricing_key(self) -> str:
        """Model/deployment name used for token pricing lookups."""
        if self.uses_azure:
            return self.azure_openai_chat_deployment
        return self.openai_chat_model

    @property
    def embed_pricing_key(self) -> str:
        if self.uses_azure:
            return self.azure_openai_embed_deployment
        return self.openai_embed_model


settings = Settings()
