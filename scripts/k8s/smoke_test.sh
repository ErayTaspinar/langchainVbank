#!/usr/bin/env bash
set -euo pipefail

curl_image="${CURL_IMAGE:-curlimages/curl:8.6.0}"

# Run inside the cluster so we don't depend on LoadBalancer localhost behavior.
echo "Smoke test: backend /healthz"
kubectl run --rm -i --restart=Never smoke-curl-backend \
  --image="$curl_image" \
  --command -- sh -lc 'curl -fsS http://langchain-service:5001/healthz | cat'

echo "Smoke test: frontend (HTTP 200/3xx)"
kubectl run --rm -i --restart=Never smoke-curl-frontend \
  --image="$curl_image" \
  --command -- sh -lc 'code=$(curl -s -o /dev/null -w "%{http_code}" http://frontend-service:5162/); echo "status=$code"; test "$code" -ge 200 -a "$code" -lt 400'

echo "OK: Smoke tests passed"