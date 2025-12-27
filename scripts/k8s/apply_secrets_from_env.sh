#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Optional: load local .env (kept out of git) to make local CI/CD easier.
# Disable by setting DOTENV_LOAD=0.
if [[ "${DOTENV_LOAD:-1}" != "0" ]] && [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$repo_root/.env"
  set +a
fi

required_env=(DB_PASSWORD OPENROUTER_API_KEY JWT_SECRET PEPPER)
missing=()
for v in "${required_env[@]}"; do
  if [[ -z "${!v:-}" ]]; then
    missing+=("$v")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "ERROR: Missing required environment variables: ${missing[*]}" >&2
  echo "These are used to generate in-cluster Kubernetes Secrets without committing k8s-secrets.yaml." >&2
  echo "Tip: create a local .env (gitignored) or export them in your shell." >&2
  echo "Required: DB_PASSWORD, OPENROUTER_API_KEY, JWT_SECRET, PEPPER" >&2
  exit 1
fi

# Match the Blazor startup validation (see langchainUI/Services/Program.cs)
jwt_bytes="$(printf '%s' "$JWT_SECRET" | wc -c | tr -d ' ')"
if [[ "$jwt_bytes" -lt 16 ]]; then
  echo "ERROR: JWT_SECRET is too short ($jwt_bytes bytes). Must be at least 16 bytes for HS256 (32+ recommended)." >&2
  echo "Generate one with: openssl rand -base64 32" >&2
  exit 1
fi

if [[ "$JWT_SECRET" == "CHANGE_ME" || "$PEPPER" == "CHANGE_ME" || "$DB_PASSWORD" == "CHANGE_ME" || "$OPENROUTER_API_KEY" == "CHANGE_ME" ]]; then
  echo "ERROR: One or more required variables are still set to CHANGE_ME." >&2
  echo "Update your .env (or exported env vars) with real values." >&2
  exit 1
fi

# Optional keys
GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"
SEARCH_ENGINE_ID="${SEARCH_ENGINE_ID:-}"
STACK_EXCHANGE_API_KEY="${STACK_EXCHANGE_API_KEY:-}"

kubectl create secret generic backend-secrets \
  --from-literal=DB_PASSWORD="$DB_PASSWORD" \
  --from-literal=OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  --from-literal=GOOGLE_API_KEY="$GOOGLE_API_KEY" \
  --from-literal=SEARCH_ENGINE_ID="$SEARCH_ENGINE_ID" \
  --from-literal=STACK_EXCHANGE_API_KEY="$STACK_EXCHANGE_API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic frontend-secrets \
  --from-literal=JWT_SECRET="$JWT_SECRET" \
  --from-literal=PEPPER="$PEPPER" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "OK: Applied backend-secrets and frontend-secrets"