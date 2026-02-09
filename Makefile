PY ?= python3.11
DATA_CONFIG ?= configs/data.yaml
IFOREST_CONFIG ?= configs/iforest.yaml
PPO_CONFIG ?= configs/ppo.yaml
EVAL_CONFIG ?= configs/eval.yaml
MORL_CONFIG ?= configs/morl.yaml
META_CONFIG ?= configs/meta_controller_relaxed.yaml
JOIN_CONFIG ?= configs/join.yaml
POLICY_EVAL_CONFIG ?= configs/policy_eval.yaml
META_STABILITY_CONFIG ?= configs/meta_stability.yaml
OUT_ROOT ?= reports/runs

.PHONY: format lint test train_a train_b eval serve dev run run_quick robustness run_robust train_morl eval_morl run_morl_quick meta_select run_meta_quick join_diag policy_eval run_morl_policy_quick meta_stability run_meta_stability_quick

format:
	@$(PY) -m ruff format .

lint:
	@$(PY) -m ruff format --check .
	@$(PY) -m ruff check .

test: pytest

train_b:
	poetry run train_b --config configs/iforest.yaml --data-config configs/data.yaml

train_a:
	poetry run train_a --config configs/ppo.yaml --data-config configs/data.yaml

eval:
	poetry run eval_h2h --eval-config configs/eval.yaml --data-config configs/data.yaml \
		--iforest-config configs/iforest.yaml --ppo-config configs/ppo.yaml

serve:
	poetry run serve --config configs/serving.yaml --data-config configs/data.yaml

run:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --out-root $(OUT_ROOT)

run_quick:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --out-root $(OUT_ROOT) \
	  --quick

robustness:
	PYTHONPATH=. $(PY) -m evaluation.robustness \
	  --contract-dir $(CONTRACT_DIR) \
	  --iforest-model-dir $(IF_DIR) \
	  --ppo-model-dir $(PPO_DIR) \
	  --ppo-config $(PPO_CONFIG) \
	  --robustness-config configs/robustness.yaml \
	  --out-dir $(OUT_DIR)

run_robust:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --robustness-config configs/robustness.yaml \
	  --out-root $(OUT_ROOT)

train_morl:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --morl-config $(MORL_CONFIG) \
	  --out-root $(OUT_ROOT)

# MORL evaluation is integrated into runs.cli when --morl-config is provided.
eval_morl:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --morl-config $(MORL_CONFIG) \
	  --out-root $(OUT_ROOT) \
	  --quick

run_morl_quick:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --morl-config $(MORL_CONFIG) \
	  --out-root $(OUT_ROOT) \
	  --quick

meta_select:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --morl-config $(MORL_CONFIG) \
	  --meta-config $(META_CONFIG) \
	  --out-root $(OUT_ROOT)

run_meta_quick:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --morl-config $(MORL_CONFIG) \
	  --meta-config $(META_CONFIG) \
	  --out-root $(OUT_ROOT) \
	  --quick

join_diag:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --join-config $(JOIN_CONFIG) \
	  --out-root $(OUT_ROOT) \
	  --quick

policy_eval:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --policy-eval-config $(POLICY_EVAL_CONFIG) \
	  --out-root $(OUT_ROOT) \
	  --quick

run_morl_policy_quick:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --morl-config $(MORL_CONFIG) \
	  --meta-config $(META_CONFIG) \
	  --join-config $(JOIN_CONFIG) \
	  --policy-eval-config $(POLICY_EVAL_CONFIG) \
	  --out-root $(OUT_ROOT) \
	  --quick

meta_stability:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --morl-config $(MORL_CONFIG) \
	  --meta-config $(META_CONFIG) \
	  --meta-stability-config $(META_STABILITY_CONFIG) \
	  --out-root $(OUT_ROOT)

run_meta_stability_quick:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --morl-config $(MORL_CONFIG) \
	  --meta-config $(META_CONFIG) \
	  --meta-stability-config $(META_STABILITY_CONFIG) \
	  --out-root $(OUT_ROOT) \
	  --quick

dev:
	@$(PY) -m pip install -U pip
	@$(PY) -m pip install -U ruff

pytest:
	@PYTHONPATH="$(CURDIR)" $(PY) -m pytest -q
