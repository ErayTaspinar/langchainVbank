#!/usr/bin/env bash
set -euo pipefail

bash scripts/k8s/check_context.sh
bash scripts/k8s/build_images.sh
bash scripts/k8s/apply_secrets_from_env.sh
bash scripts/k8s/deploy_stack.sh
bash scripts/k8s/wait_rollout.sh
bash scripts/k8s/smoke_test.sh

echo "DONE: Local CI/CD pipeline succeeded"