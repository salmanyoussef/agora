from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set, Tuple
from urllib.parse import urlparse
import logging
import uuid

import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
from weaviate.exceptions import UnexpectedStatusCodeError

from app.settings import settings


logger = logging.getLogger(__name__)

# Stable namespace for deterministic UUIDs when no existing object is found.
_DATASET_UUID_NS = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")


def _dataset_uuid(dataset_id: str) -> uuid.UUID:
    return uuid.uuid5(_DATASET_UUID_NS, dataset_id.strip())


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

    def _existing_uuids_by_dataset_id(
        self,
        col: Any,
        dataset_ids: List[str],
    ) -> Dict[str, uuid.UUID]:
        if not dataset_ids:
            return {}
        resp = col.query.fetch_objects(
            filters=Filter.by_property("dataset_id").contains_any(dataset_ids),
            limit=len(dataset_ids),
        )
        out: Dict[str, uuid.UUID] = {}
        for obj in resp.objects or []:
            props = obj.properties or {}
            did = props.get("dataset_id")
            if isinstance(did, str) and did and did not in out:
                out[did] = obj.uuid
        return out

    def upsert_many(self, rows: Iterable[Tuple[Dict[str, Any], List[float]]]) -> int:
        """Insert or update objects keyed by dataset_id (one Weaviate object per catalogue id)."""
        self.ensure_collection()
        client = self.connect()
        try:
            col = client.collections.use(self.collection_name)
            row_list = list(rows)
            if not row_list:
                logger.info("Weaviate upsert_many called with empty batch")
                return 0

            dataset_ids = [props.get("dataset_id", "") for props, _ in row_list if props.get("dataset_id")]
            existing = self._existing_uuids_by_dataset_id(col, dataset_ids)

            upserted = 0
            logger.info("Weaviate upsert_many: batch_size=%d", len(row_list))
            for props, vec in row_list:
                did = props.get("dataset_id")
                if not did:
                    continue
                did_str = str(did)
                if did_str in existing:
                    col.data.replace(
                        uuid=existing[did_str],
                        properties=props,
                        vector=vec,
                    )
                else:
                    obj_uuid = _dataset_uuid(did_str)
                    try:
                        col.data.insert(uuid=obj_uuid, properties=props, vector=vec)
                    except UnexpectedStatusCodeError:
                        # Object may already exist under the deterministic id (e.g. retry).
                        col.data.replace(uuid=obj_uuid, properties=props, vector=vec)
                upserted += 1

            logger.info("Weaviate upsert_many succeeded: upserted=%d", upserted)
            return upserted
        finally:
            client.close()

    def delete_stale_except(self, keep_dataset_ids: Set[str]) -> int:
        """Remove indexed datasets whose id is no longer on data.gouv.fr."""
        self.ensure_collection()
        client = self.connect()
        removed = 0
        try:
            col = client.collections.use(self.collection_name)
            logger.info("Weaviate delete_stale_except: scanning collection=%s", self.collection_name)
            for obj in col.iterator():
                props = obj.properties or {}
                did = props.get("dataset_id")
                if isinstance(did, str) and did and did not in keep_dataset_ids:
                    col.data.delete_by_id(obj.uuid)
                    removed += 1
            logger.info("Weaviate delete_stale_except: removed=%d", removed)
            return removed
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
                query_properties=["content"],
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