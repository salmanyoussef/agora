#!/usr/bin/env bash
# AGORA reviewer demo — Azure Container Apps deployment (reproducible runbook).
# Prereqs: az login done; weaviate_backup.tar.gz available locally.
# Usage: ./aca_deploy.sh /path/to/weaviate_backup.tar.gz
set -euo pipefail

BACKUP="${1:?usage: aca_deploy.sh /path/to/weaviate_backup.tar.gz}"
RG=agora-demo-rg
LOC=francecentral
ACR=agorademoacr$RANDOM          # must be globally unique, lowercase alphanum
ENVNAME=agora-demo-env
APP=agora-demo
SA=agorademosa$RANDOM            # storage account for budget-file persistence
TAG=v1
REPO_SRC="$(cd "$(dirname "$0")/../.." && pwd)"   # .../src

az group create -n "$RG" -l "$LOC"
az acr create -n "$ACR" -g "$RG" --sku Basic --admin-enabled true

# 1) Backend image (context: src/, includes frontend + Deno for RLM)
az acr build -r "$ACR" -t agora-backend:$TAG -f "$REPO_SRC/Dockerfile.backend" "$REPO_SRC"

# 2) Seeded Weaviate image (context: temp dir with backup tarball)
CTX=$(mktemp -d)
cp "$REPO_SRC/infra/weaviate-seeded/Dockerfile.aca" "$CTX/Dockerfile"
cp "$BACKUP" "$CTX/weaviate_backup.tar.gz"
az acr build -r "$ACR" -t agora-weaviate:$TAG "$CTX"
rm -rf "$CTX"

# 3) Container Apps environment + Azure Files share for /data (budget persistence)
az containerapp env create -n "$ENVNAME" -g "$RG" -l "$LOC"
az storage account create -n "$SA" -g "$RG" -l "$LOC" --sku Standard_LRS
KEY=$(az storage account keys list -n "$SA" -g "$RG" --query '[0].value' -o tsv)
az storage share create --name agoradata --account-name "$SA" --account-key "$KEY"
az containerapp env storage set -n "$ENVNAME" -g "$RG" --storage-name agoradata \
  --azure-file-account-name "$SA" --azure-file-account-key "$KEY" \
  --azure-file-share-name agoradata --access-mode ReadWrite

# 4) Fill the YAML template and create the app
ENV_ID=$(az containerapp env show -n "$ENVNAME" -g "$RG" --query id -o tsv)
ACR_SERVER=$(az acr show -n "$ACR" --query loginServer -o tsv)
ACR_USER=$(az acr credential show -n "$ACR" --query username -o tsv)
ACR_PASS=$(az acr credential show -n "$ACR" --query 'passwords[0].value' -o tsv)
ADMIN_TOKEN=$(openssl rand -hex 24)
: "${AZURE_OPENAI_ENDPOINT:?export AZURE_OPENAI_ENDPOINT first}"
: "${AZURE_OPENAI_API_KEY:?export AZURE_OPENAI_API_KEY first}"

sed -e "s|__ENV_ID__|$ENV_ID|" -e "s|__ACR_SERVER__|$ACR_SERVER|g" \
    -e "s|__ACR_USER__|$ACR_USER|" -e "s|__ACR_PASSWORD__|$ACR_PASS|" \
    -e "s|__AZURE_OPENAI_KEY__|$AZURE_OPENAI_API_KEY|" \
    -e "s|__AZURE_OPENAI_ENDPOINT__|$AZURE_OPENAI_ENDPOINT|" \
    -e "s|__ADMIN_TOKEN__|$ADMIN_TOKEN|" -e "s|__TAG__|$TAG|g" \
    "$(dirname "$0")/containerapp.yaml.template" > /tmp/agora-app.yaml

az containerapp create -n "$APP" -g "$RG" --yaml /tmp/agora-app.yaml
rm -f /tmp/agora-app.yaml

echo "Admin token: $ADMIN_TOKEN"
echo "Public URL: https://$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
