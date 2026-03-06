from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8", extra="ignore")

    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_embed_api_version: str = "2024-02-01"  # from env AZURE_OPENAI_EMBED_API_VERSION
    azure_openai_embed_deployment: str = "text-embedding-3-small"
    azure_openai_chat_deployment: str = "gpt-5-mini"

    # Weaviate
    weaviate_url: str = "http://localhost:8080"
    weaviate_grpc_host: str = "localhost"
    weaviate_grpc_port: int = 50051
    weaviate_api_key: str | None = None
    datasets_collection: str = "Dataset"


settings = Settings()
