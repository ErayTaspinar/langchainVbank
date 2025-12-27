#!/usr/bin/env bash
set -euo pipefail

expected_context="${KUBE_CONTEXT:-docker-desktop}"
current_context="$(kubectl config current-context 2>/dev/null || true)"

if [[ -z "$current_context" ]]; then
  echo "ERROR: kubectl has no current context configured." >&2
  exit 1
fi

if [[ "$current_context" != "$expected_context" ]]; then
  echo "ERROR: Refusing to run against kube context '$current_context'. Expected '$expected_context'." >&2
  echo "Set KUBE_CONTEXT to override (at your own risk)." >&2
  exit 1
fi

echo "OK: kubectl context is '$current_context'"