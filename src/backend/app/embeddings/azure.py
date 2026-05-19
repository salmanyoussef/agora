"""
OpenAI embedding client (legacy module path: app.embeddings.azure).

Uses OPENAI_API_KEY and OPENAI_EMBED_MODEL from settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

_shared_client: Optional["OpenAIEmbeddingClient"] = None


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


@dataclass
class OpenAIEmbeddingClient:
    api_key: str
    model: str

    def __post_init__(self) -> None:
        self.client = OpenAI(api_key=self.api_key)

    def _log_usage(self, resp: Any, batch_size: int) -> None:
        usage: Optional[Any] = getattr(resp, "usage", None)
        if not usage:
            logger.info("OpenAI embeddings batch completed: texts=%d model=%s", batch_size, self.model)
            return
        d = _usage_to_dict(usage)
        if d:
            logger.info(
                "OpenAI embeddings batch completed: texts=%d model=%s prompt_tokens=%s total_tokens=%s",
                batch_size,
                self.model,
                d.get("prompt_tokens"),
                d.get("total_tokens"),
            )
        else:
            logger.info("OpenAI embeddings batch completed: texts=%d model=%s", batch_size, self.model)

    def embed_texts(self, texts: List[str]) -> Tuple[List[List[float]], Optional[Dict[str, int]]]:
        logger.info("Calling OpenAI embeddings: texts=%d model=%s", len(texts), self.model)
        resp = self.client.embeddings.create(model=self.model, input=texts)
        self._log_usage(resp, batch_size=len(texts))
        embeddings = [d.embedding for d in resp.data]
        usage_dict = _usage_to_dict(getattr(resp, "usage", None))
        return (embeddings, usage_dict)

    def close(self) -> None:
        if hasattr(self.client, "close"):
            self.client.close()
            logger.debug("OpenAI embedding client closed")


# Backward-compatible alias for type hints / imports
AzureEmbeddingClient = OpenAIEmbeddingClient


def get_embedding_client() -> OpenAIEmbeddingClient:
    global _shared_client
    if _shared_client is not None:
        return _shared_client
    from app.settings import settings

    _shared_client = OpenAIEmbeddingClient(
        api_key=settings.openai_api_key,
        model=settings.openai_embed_model,
    )
    return _shared_client
