PY ?= python3.11
DATA_CONFIG ?= configs/data.yaml
IFOREST_CONFIG ?= configs/iforest.yaml
PPO_CONFIG ?= configs/ppo.yaml
EVAL_CONFIG ?= configs/eval.yaml
MORL_CONFIG ?= configs/morl.yaml
META_CONFIG ?= configs/meta_controller_relaxed.yaml
ROBUSTNESS_CONFIG ?= configs/robustness.yaml
JOIN_CONFIG ?= configs/join.yaml
POLICY_EVAL_CONFIG ?= configs/policy_eval.yaml
META_STABILITY_CONFIG ?= configs/meta_stability.yaml
OUT_ROOT ?= reports/runs
CONTRACT_CACHE_ROOT ?= reports/contracts_cache
RUN_ID ?=
BQ_PROJECT ?= gatra-prd-c335
PUBLIC_OUT_ROOT ?= reports/public_exports
PAPER_INDEX ?= reports/paper_results/week1_run_index.csv
PAPER_INDEX_CSV ?= reports/paper_results/week1_run_index_csv.csv
PAPER_INDEX_BQ ?= reports/paper_results/week1_run_index_bq.csv
PAPER_RESULTS_OUT ?= reports/paper_results/paper_week1_results.csv
PAPER_CSV_SEEDS ?= 42,1337,2026
PAPER_BQ_SEEDS ?= 42

.PHONY: format lint test train_a train_b eval serve dev run run_quick robustness run_robust train_morl eval_morl run_morl_quick meta_select run_meta_quick join_diag policy_eval run_morl_policy_quick run_morl_policy_robust_quick meta_stability run_meta_stability_quick run_statistical_analysis paper_week1_csv paper_week1_bq paper_collect_week1 deploy_queue deploy_safe_view verify_queue export_sanitized_artifacts

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
	  --cache-root $(CONTRACT_CACHE_ROOT) \
	  --out-root $(OUT_ROOT)

run_quick:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --cache-root $(CONTRACT_CACHE_ROOT) \
	  --out-root $(OUT_ROOT) \
	  --quick

robustness:
	PYTHONPATH=. $(PY) -m evaluation.robustness \
	  --contract-dir $(CONTRACT_DIR) \
	  --iforest-model-dir $(IF_DIR) \
	  --ppo-model-dir $(PPO_DIR) \
	  --ppo-config $(PPO_CONFIG) \
	  --robustness-config $(ROBUSTNESS_CONFIG) \
	  --out-dir $(OUT_DIR)

run_robust:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --robustness-config $(ROBUSTNESS_CONFIG) \
	  --cache-root $(CONTRACT_CACHE_ROOT) \
	  --out-root $(OUT_ROOT)

train_morl:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --morl-config $(MORL_CONFIG) \
	  --cache-root $(CONTRACT_CACHE_ROOT) \
	  --out-root $(OUT_ROOT)

# MORL evaluation is integrated into runs.cli when --morl-config is provided.
eval_morl:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --morl-config $(MORL_CONFIG) \
	  --cache-root $(CONTRACT_CACHE_ROOT) \
	  --out-root $(OUT_ROOT) \
	  --quick

run_morl_quick:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --morl-config $(MORL_CONFIG) \
	  --cache-root $(CONTRACT_CACHE_ROOT) \
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
	  --cache-root $(CONTRACT_CACHE_ROOT) \
	  --out-root $(OUT_ROOT)

run_meta_quick:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --morl-config $(MORL_CONFIG) \
	  --meta-config $(META_CONFIG) \
	  --cache-root $(CONTRACT_CACHE_ROOT) \
	  --out-root $(OUT_ROOT) \
	  --quick

join_diag:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --join-config $(JOIN_CONFIG) \
	  --cache-root $(CONTRACT_CACHE_ROOT) \
	  --out-root $(OUT_ROOT) \
	  --quick

policy_eval:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --policy-eval-config $(POLICY_EVAL_CONFIG) \
	  --cache-root $(CONTRACT_CACHE_ROOT) \
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
	  --cache-root $(CONTRACT_CACHE_ROOT) \
	  --out-root $(OUT_ROOT) \
	  --quick

run_morl_policy_robust_quick:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config $(DATA_CONFIG) \
	  --iforest-config $(IFOREST_CONFIG) \
	  --ppo-config $(PPO_CONFIG) \
	  --eval-config $(EVAL_CONFIG) \
	  --morl-config $(MORL_CONFIG) \
	  --meta-config $(META_CONFIG) \
	  --join-config $(JOIN_CONFIG) \
	  --policy-eval-config $(POLICY_EVAL_CONFIG) \
	  --robustness-config $(ROBUSTNESS_CONFIG) \
	  --cache-root $(CONTRACT_CACHE_ROOT) \
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
	  --cache-root $(CONTRACT_CACHE_ROOT) \
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
	  --cache-root $(CONTRACT_CACHE_ROOT) \
	  --out-root $(OUT_ROOT) \
	  --quick

run_statistical_analysis:
	@test -n "$(RUN_ID)" || (echo "RUN_ID is required, e.g. make run_statistical_analysis RUN_ID=20260210T063154Z" && exit 1)
	PYTHONPATH=. $(PY) -m statistical_significance \
	  --morl-results reports/runs/$(RUN_ID)/morl_selected_test.json \
	  --classical-results reports/runs/$(RUN_ID)/classical_test.json \
	  --output reports/runs/$(RUN_ID)/statistical_analysis.json \
	  --alpha 0.05 \
	  --n-bootstrap 1000

paper_week1_csv:
	bash scripts/paper_matrix.sh --csv --seeds $(PAPER_CSV_SEEDS) --index-out $(PAPER_INDEX_CSV)

paper_week1_bq:
	bash scripts/paper_matrix.sh --bq --bq-seeds $(PAPER_BQ_SEEDS) --index-out $(PAPER_INDEX_BQ)

paper_collect_week1:
	PYTHONPATH=. $(PY) scripts/collect_paper_results.py \
	  --index $(PAPER_INDEX_CSV) \
	  --index $(PAPER_INDEX_BQ) \
	  --out $(PAPER_RESULTS_OUT)

deploy_queue:
	bq --project_id=$(BQ_PROJECT) query --use_legacy_sql=false < sql/10_deploy_prod_queue_top200.sql

deploy_safe_view:
	bq --project_id=$(BQ_PROJECT) query --use_legacy_sql=false < sql/30_deploy_safe_view.sql

verify_queue:
	bq --project_id=$(BQ_PROJECT) query --use_legacy_sql=false < sql/20_verify_prod_queue.sql

export_sanitized_artifacts:
	PROJECT_ID=$(BQ_PROJECT) OUT_ROOT=$(PUBLIC_OUT_ROOT) bash scripts/export_sanitized_artifacts.sh

dev:
	@$(PY) -m pip install -U pip
	@$(PY) -m pip install -U ruff

pytest:
	@PYTHONPATH="$(CURDIR)" $(PY) -m pytest -q
