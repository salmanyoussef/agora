# Weaviate pre-seeded with the AGORA catalogue index (~73k datasets, BYOV embeddings).
# SEED_URL must point to a tar.gz of the Weaviate data directory (weaviate_backup.tar.gz).
# The data is baked into the image: no persistent disk needed, redeploys re-seed automatically.
# Run this as a PRIVATE service only — anonymous auth is enabled.
FROM alpine:3.20 AS seed
ARG SEED_URL
RUN apk add --no-cache curl \
    && mkdir -p /seed \
    && curl -fSL "$SEED_URL" -o /tmp/backup.tar.gz \
    && tar -xzf /tmp/backup.tar.gz -C /seed \
    && rm /tmp/backup.tar.gz

FROM cr.weaviate.io/semitechnologies/weaviate:1.36.0
COPY --from=seed /seed /var/lib/weaviate

ENV PERSISTENCE_DATA_PATH=/var/lib/weaviate
ENV AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
ENV QUERY_DEFAULTS_LIMIT=25
ENV CLUSTER_HOSTNAME=node1
