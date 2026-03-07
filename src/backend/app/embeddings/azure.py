from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Any
import logging

from openai import AzureOpenAI


logger = logging.getLogger(__name__)

_shared_client: Optional["AzureEmbeddingClient"] = None


@dataclass
class AzureEmbeddingClient:
    azure_endpoint: str
    api_key: str
    deployment: str
    api_version: str

    def __post_init__(self) -> None:
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

    def _log_usage(self, resp: Any, batch_size: int) -> None:
        usage: Optional[Any] = getattr(resp, "usage", None)
        if not usage:
            logger.info("Azure embeddings batch completed: texts=%d", batch_size)
            return

        # AzureOpenAI may expose usage as attrs or as a dict; handle both.
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        if prompt_tokens is None and isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")

        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is None and isinstance(usage, dict):
            total_tokens = usage.get("total_tokens")

        logger.info(
            "Azure embeddings batch completed: texts=%d, prompt_tokens=%s, total_tokens=%s",
            batch_size,
            prompt_tokens,
            total_tokens,
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Azure OpenAI: embeddings.create(model=<deployment>, input=[...])
        logger.info("Calling Azure embeddings: texts=%d", len(texts))
        resp = self.client.embeddings.create(model=self.deployment, input=texts)
        self._log_usage(resp, batch_size=len(texts))
        return [d.embedding for d in resp.data]

    def close(self) -> None:
        """Close the underlying OpenAI client and release connections."""
        if hasattr(self.client, "close"):
            self.client.close()
            logger.debug("Azure embedding client closed")


def get_embedding_client() -> "AzureEmbeddingClient":
    """Return a shared Azure embedding client (lazy singleton). Reuse avoids leaking SSL connections."""
    global _shared_client
    if _shared_client is not None:
        return _shared_client
    from app.settings import settings
    _shared_client = AzureEmbeddingClient(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        deployment=settings.azure_openai_embed_deployment,
        api_version=settings.azure_openai_embed_api_version,
    )
    return _shared_client
