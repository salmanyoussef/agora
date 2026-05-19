from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI (platform API)
    openai_api_key: str
    openai_chat_model: str = "gpt-5-mini"
    openai_chat_max_tokens: int = 64000
    openai_embed_model: str = "text-embedding-3-small"

    # Weaviate
    weaviate_url: str = "http://localhost:8080"
    weaviate_grpc_host: str = "localhost"
    weaviate_grpc_port: int = 50051
    weaviate_api_key: str | None = None
    datasets_collection: str = "Dataset"

    # Demo: faster technical agent (RLM with lower limits, 1 resource/dataset, cap datasets)
    agora_demo_mode: bool = False


settings = Settings()
