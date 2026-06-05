"""
Embedding clients for OpenAI and Azure OpenAI (module path kept for imports).

Use get_embedding_client() — picks the active provider from settings.llm_provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable
import logging

from openai import AzureOpenAI, OpenAI

logger = logging.getLogger(__name__)

_shared_client: Optional["EmbeddingClient"] = None


def _usage_to_dict(usage: Any) -> Optional[Dict[str, int]]:
    if not usage:
        return None
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    if prompt_tokens is None and isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None and isinstance(usage, dict):
        total_tokens = usage.get("total_tokens")
    if prompt_tokens is None and total_tokens is None:
        return None
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


@runtime_checkable
class EmbeddingClient(Protocol):
    def embed_texts(self, texts: List[str]) -> Tuple[List[List[float]], Optional[Dict[str, int]]]: ...
    def close(self) -> None: ...


@dataclass
class OpenAIEmbeddingClient:
    api_key: str
    model: str
    provider_label: str = "OpenAI"

    def __post_init__(self) -> None:
        self.client = OpenAI(api_key=self.api_key)

    def _log_usage(self, resp: Any, batch_size: int) -> None:
        usage: Optional[Any] = getattr(resp, "usage", None)
        label = self.provider_label
        if not usage:
            logger.info("%s embeddings batch completed: texts=%d model=%s", label, batch_size, self.model)
            return
        d = _usage_to_dict(usage)
        if d:
            logger.info(
                "%s embeddings batch completed: texts=%d model=%s prompt_tokens=%s total_tokens=%s",
                label,
                batch_size,
                self.model,
                d.get("prompt_tokens"),
                d.get("total_tokens"),
            )
        else:
            logger.info("%s embeddings batch completed: texts=%d model=%s", label, batch_size, self.model)

    def embed_texts(self, texts: List[str]) -> Tuple[List[List[float]], Optional[Dict[str, int]]]:
        logger.info("Calling %s embeddings: texts=%d model=%s", self.provider_label, len(texts), self.model)
        resp = self.client.embeddings.create(model=self.model, input=texts)
        self._log_usage(resp, batch_size=len(texts))
        embeddings = [d.embedding for d in resp.data]
        usage_dict = _usage_to_dict(getattr(resp, "usage", None))
        return (embeddings, usage_dict)

    def close(self) -> None:
        if hasattr(self.client, "close"):
            self.client.close()
            logger.debug("%s embedding client closed", self.provider_label)


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
            logger.info("Azure embeddings batch completed: texts=%d deployment=%s", batch_size, self.deployment)
            return
        d = _usage_to_dict(usage)
        if d:
            logger.info(
                "Azure embeddings batch completed: texts=%d deployment=%s prompt_tokens=%s total_tokens=%s",
                batch_size,
                self.deployment,
                d.get("prompt_tokens"),
                d.get("total_tokens"),
            )
        else:
            logger.info("Azure embeddings batch completed: texts=%d deployment=%s", batch_size, self.deployment)

    def embed_texts(self, texts: List[str]) -> Tuple[List[List[float]], Optional[Dict[str, int]]]:
        logger.info("Calling Azure embeddings: texts=%d deployment=%s", len(texts), self.deployment)
        resp = self.client.embeddings.create(model=self.deployment, input=texts)
        self._log_usage(resp, batch_size=len(texts))
        embeddings = [d.embedding for d in resp.data]
        usage_dict = _usage_to_dict(getattr(resp, "usage", None))
        return (embeddings, usage_dict)

    def close(self) -> None:
        if hasattr(self.client, "close"):
            self.client.close()
            logger.debug("Azure embedding client closed")


def get_embedding_client() -> EmbeddingClient:
    """Return a shared embedding client for the configured LLM_PROVIDER."""
    global _shared_client
    if _shared_client is not None:
        return _shared_client

    from app.settings import settings

    if settings.uses_azure:
        _shared_client = AzureEmbeddingClient(
            azure_endpoint=settings.azure_openai_endpoint or "",
            api_key=settings.azure_openai_api_key or "",
            deployment=settings.azure_openai_embed_deployment,
            api_version=settings.azure_openai_embed_api_version,
        )
    else:
        _shared_client = OpenAIEmbeddingClient(
            api_key=settings.openai_api_key or "",
            model=settings.openai_embed_model,
        )
    return _shared_client


def reset_embedding_client() -> None:
    """Close and clear the singleton (e.g. after provider change in tests)."""
    global _shared_client
    if _shared_client is not None:
        _shared_client.close()
        _shared_client = None
