.PHONY: k8s-check k8s-build k8s-secrets k8s-deploy k8s-wait k8s-smoke ci

k8s-check:
	bash scripts/k8s/check_context.sh

k8s-build:
	bash scripts/k8s/build_images.sh

k8s-secrets:
	bash scripts/k8s/apply_secrets_from_env.sh

k8s-deploy:
	bash scripts/k8s/deploy_stack.sh

k8s-wait:
	bash scripts/k8s/wait_rollout.sh

k8s-smoke:
	bash scripts/k8s/smoke_test.sh

ci: k8s-check k8s-build k8s-secrets k8s-deploy k8s-wait k8s-smoke
	@echo "DONE: make ci succeeded"