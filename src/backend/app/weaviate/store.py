from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlparse
import logging

import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import Auth

from app.settings import settings


logger = logging.getLogger(__name__)


@dataclass
class WeaviateStore:
    collection_name: str = settings.datasets_collection

    def connect(self):
        """
        Connect to Weaviate using the v4 client.

        settings.weaviate_url should look like:
          - http://localhost:8080
          - http://weaviate:8080   (when backend runs in docker-compose)
          - https://your-host       (port optional; defaults to 443 for https)

        IMPORTANT: Do NOT pass "host:port" into http_host. Pass only hostname,
        and pass the port separately. (Otherwise you end up with host:port:port.)
        """
        u = urlparse(settings.weaviate_url)

        http_host = u.hostname or "localhost"
        http_secure = (u.scheme == "https")
        http_port = u.port or (443 if http_secure else 8080)

        logger.info(
            "Connecting to Weaviate: http_host=%s, http_port=%d, http_secure=%s, grpc_host=%s, grpc_port=%d",
            http_host,
            http_port,
            http_secure,
            settings.weaviate_grpc_host,
            settings.weaviate_grpc_port,
        )

        kwargs = dict(
            http_host=http_host,
            http_port=http_port,
            http_secure=http_secure,
            grpc_host=settings.weaviate_grpc_host,
            grpc_port=settings.weaviate_grpc_port,
            grpc_secure=False,
        )

        if settings.weaviate_api_key:
            kwargs["auth_credentials"] = Auth.api_key(settings.weaviate_api_key)

        return weaviate.connect_to_custom(**kwargs)

    def ensure_collection(self) -> None:
        client = self.connect()
        try:
            if client.collections.exists(self.collection_name):
                logger.info("Weaviate collection already exists: %s", self.collection_name)
                return

            logger.info("Creating Weaviate collection: %s", self.collection_name)
            client.collections.create(
                name=self.collection_name,
                vector_config=wvc.config.Configure.Vectors.self_provided(),
                properties=[
                    wvc.config.Property(name="dataset_id", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="description", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="organization", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="url", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="tags", data_type=wvc.config.DataType.TEXT_ARRAY),
                ],
            )
        finally:
            client.close()

    def upsert_many(self, rows: Iterable[Tuple[Dict[str, Any], List[float]]]) -> int:
        self.ensure_collection()
        client = self.connect()
        try:
            col = client.collections.use(self.collection_name)

            objs: List[wvc.data.DataObject] = []
            for props, vec in rows:
                objs.append(wvc.data.DataObject(properties=props, vector=vec))

            if not objs:
                logger.info("Weaviate upsert_many called with empty batch")
                return 0

            logger.info("Weaviate upsert_many: inserting batch_size=%d", len(objs))
            res = col.data.insert_many(objs)

            if hasattr(res, "has_errors") and res.has_errors:
                logger.warning("Weaviate insert_many reported errors")

            if hasattr(res, "uuids") and res.uuids is not None:
                inserted = len(res.uuids)
                logger.info("Weaviate insert_many succeeded: inserted=%d", inserted)
                return inserted

            logger.info("Weaviate insert_many completed without uuids; assuming inserted=%d", len(objs))
            return len(objs)
        finally:
            client.close()

    def search(
        self,
        query_text: str,
        query_vector: List[float],
        k: int = 5,
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        self.ensure_collection()
        client = self.connect()
        try:
            logger.info("Weaviate hybrid search: k=%d, alpha=%.2f", k, alpha)
            col = client.collections.use(self.collection_name)
            resp = col.query.hybrid(
                query=query_text,
                vector=query_vector,
                alpha=alpha,
                query_properties=["title", "description", "organization", "tags", "content"],
                limit=k,
                return_metadata=wvc.query.MetadataQuery(distance=True, score=True),
            )

            out: List[Dict[str, Any]] = []
            for obj in resp.objects:
                props = dict(obj.properties) if obj.properties else {}
                md = obj.metadata
                out.append(
                    {
                        **props,
                        "_distance": getattr(md, "distance", None),
                        "_score": getattr(md, "score", None),
                    }
                )
            logger.info("Weaviate search returned %d objects", len(out))
            return out
        finally:
            client.close()


    def count(self) -> int:
        self.ensure_collection()
        client = self.connect()
        try:
            logger.info("Weaviate count called for collection=%s", self.collection_name)
            col = client.collections.use(self.collection_name)
            agg = col.aggregate.over_all(total_count=True)
            count = int(agg.total_count or 0)
            logger.info("Weaviate count result: %d", count)
            return count
        finally:
            client.close()

    def sample(self, limit: int = 20) -> List[Dict[str, Any]]:
        self.ensure_collection()
        client = self.connect()
        try:
            logger.info("Weaviate sample called: limit=%d", limit)
            col = client.collections.use(self.collection_name)
            resp = col.query.fetch_objects(limit=limit)
            out: List[Dict[str, Any]] = []
            for obj in resp.objects:
                props = dict(obj.properties) if obj.properties else {}
                out.append(props)
            logger.info("Weaviate sample returned %d objects", len(out))
            return out
        finally:
            client.close()