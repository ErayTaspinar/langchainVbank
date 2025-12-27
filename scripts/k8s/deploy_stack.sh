#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

kubectl apply -f "$repo_root/k8s-deployment.yaml"

# Secrets are injected as env vars; pods must restart to pick up changes.
kubectl rollout restart deployment/langchain deployment/frontend >/dev/null 2>&1 || true

echo "OK: Applied k8s-deployment.yaml"