#!/usr/bin/env bash
set -euo pipefail

timeout="${ROLLOUT_TIMEOUT:-300s}"

echo "Waiting for Postgres rollout..."
kubectl rollout status deployment/postgres --timeout="$timeout"

echo "Waiting for backend rollout..."
kubectl rollout status deployment/langchain --timeout="$timeout"

echo "Waiting for frontend rollout..."
kubectl rollout status deployment/frontend --timeout="$timeout"

echo "OK: All deployments rolled out"