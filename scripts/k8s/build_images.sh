#!/usr/bin/env bash
set -euo pipefail

backend_image="${BACKEND_IMAGE:-langchain:latest}"
frontend_image="${FRONTEND_IMAGE:-langchain-ui:latest}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Building backend image: $backend_image"
docker build -t "$backend_image" "$repo_root"

echo "Building frontend image: $frontend_image"
docker build -t "$frontend_image" -f "$repo_root/langchainUI/Dockerfile" "$repo_root/langchainUI"

echo "OK: Built images"