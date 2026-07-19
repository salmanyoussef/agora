#!/usr/bin/env bash
# AGORA ICSOC reviewer demo — run inside Azure Cloud Shell (bash).
#   curl -sL https://raw.githubusercontent.com/salmanyoussef/agora/deploy/reviewer-demo/src/infra/aca/cloudshell_deploy.sh | bash
# Uses: subscription UDST-ACADEMICS-CAPSTONE-Project, resource group am_rg_east2_o3mini.
# Reads the Azure OpenAI key directly from the o3miniapi resource — no secrets typed anywhere.
set -euo pipefail

SUB_NAME="UDST-ACADEMICS-CAPSTONE-Project"
RG="am_rg_east2_o3mini"
BRANCH="deploy/reviewer-demo"
REPO="https://github.com/salmanyoussef/agora.git"
APP="agora-demo"
ENVNAME="agora-demo-env"
TAG="v1"

az account set --subscription "$SUB_NAME"
SUBID=$(az account show --query id -o tsv)
SFX=$(echo "$SUBID" | tr -d - | cut -c1-8)
ACR="agoraicsoc$SFX"
SA="agorasa$SFX"
LOC=$(az group show -n "$RG" --query location -o tsv)
echo "== Subscription $SUBID | RG $RG ($LOC) | ACR $ACR =="

# --- Azure OpenAI endpoint + key (from existing resource in the RG) ---
AOAI_NAME=$(az cognitiveservices account list -g "$RG" --query "[?contains(properties.endpoint,'o3miniapi')].name | [0]" -o tsv)
[ -z "$AOAI_NAME" ] && AOAI_NAME=$(az cognitiveservices account list -g "$RG" --query "[0].name" -o tsv)
AOAI_ENDPOINT=$(az cognitiveservices account show -n "$AOAI_NAME" -g "$RG" --query properties.endpoint -o tsv)
AOAI_KEY=$(az cognitiveservices account keys list -n "$AOAI_NAME" -g "$RG" --query key1 -o tsv)
echo "== Azure OpenAI resource: $AOAI_NAME ($AOAI_ENDPOINT) =="

# --- Registry ---
az acr show -n "$ACR" -g "$RG" >/dev/null 2>&1 || az acr create -n "$ACR" -g "$RG" --sku Basic --admin-enabled true -o none

# --- Sources ---
WORK=$(mktemp -d)
git clone -q --depth 1 -b "$BRANCH" "$REPO" "$WORK/agora"
git clone -q --depth 1 -b weaviate-backup-data "$REPO" "$WORK/bk"
( cd "$WORK/bk" && cat part.* > weaviate_backup.tar.gz && sha256sum -c SHA256SUM )

# --- Build images in ACR (cloud build, no local docker) ---
echo "== Building backend image (long: torch/unstructured) =="
az acr build -r "$ACR" -t "agora-backend:$TAG" -f "$WORK/agora/src/Dockerfile.backend" "$WORK/agora/src"
echo "== Building seeded Weaviate image =="
WCTX=$(mktemp -d)
cp "$WORK/agora/src/infra/weaviate-seeded/Dockerfile.aca" "$WCTX/Dockerfile"
mv "$WORK/bk/weaviate_backup.tar.gz" "$WCTX/"
az acr build -r "$ACR" -t "agora-weaviate:$TAG" "$WCTX"

# --- Container Apps environment + Azure Files for budget persistence ---
az extension add --name containerapp --upgrade -y 2>/dev/null || true
az provider register -n Microsoft.App --wait 2>/dev/null || true
az containerapp env show -n "$ENVNAME" -g "$RG" >/dev/null 2>&1 || az containerapp env create -n "$ENVNAME" -g "$RG" -l "$LOC" -o none
az storage account show -n "$SA" -g "$RG" >/dev/null 2>&1 || az storage account create -n "$SA" -g "$RG" -l "$LOC" --sku Standard_LRS -o none
SKEY=$(az storage account keys list -n "$SA" -g "$RG" --query '[0].value' -o tsv)
az storage share create --name agoradata --account-name "$SA" --account-key "$SKEY" -o none
az containerapp env storage set -n "$ENVNAME" -g "$RG" --storage-name agoradata \
  --azure-file-account-name "$SA" --azure-file-account-key "$SKEY" \
  --azure-file-share-name agoradata --access-mode ReadWrite -o none

# --- App (backend public + weaviate localhost sidecar) ---
ENV_ID=$(az containerapp env show -n "$ENVNAME" -g "$RG" --query id -o tsv)
ACR_SERVER=$(az acr show -n "$ACR" --query loginServer -o tsv)
ACR_USER=$(az acr credential show -n "$ACR" --query username -o tsv)
ACR_PASS=$(az acr credential show -n "$ACR" --query 'passwords[0].value' -o tsv)
ADMIN_TOKEN=$(openssl rand -hex 24)

sed -e "s|__ENV_ID__|$ENV_ID|" -e "s|__ACR_SERVER__|$ACR_SERVER|g" \
    -e "s|__ACR_USER__|$ACR_USER|" -e "s|__ACR_PASSWORD__|$ACR_PASS|" \
    -e "s|__AZURE_OPENAI_KEY__|$AOAI_KEY|" \
    -e "s|__AZURE_OPENAI_ENDPOINT__|$AOAI_ENDPOINT|" \
    -e "s|__ADMIN_TOKEN__|$ADMIN_TOKEN|" -e "s|__TAG__|$TAG|g" \
    "$WORK/agora/src/infra/aca/containerapp.yaml.template" > "$WORK/app.yaml"
sed -i "1i location: $LOC" "$WORK/app.yaml"

az containerapp create -n "$APP" -g "$RG" --yaml "$WORK/app.yaml" -o none
FQDN=$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)

echo "=============================================="
echo "ADMIN_TOKEN: $ADMIN_TOKEN"
echo "PUBLIC URL:  https://$FQDN"
echo "=============================================="
