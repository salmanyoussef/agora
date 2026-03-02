from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Any
import logging

from openai import AzureOpenAI


logger = logging.getLogger(__name__)


@dataclass
class AzureEmbeddingClient:
    azure_endpoint: str
    api_key: str
    deployment: str

    def __post_init__(self) -> None:
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version="2024-02-01",
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
