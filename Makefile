PY ?= python3.11

.PHONY: format lint test train_a train_b eval serve dev run run_quick robustness run_robust

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
	  --data-config configs/data.yaml \
	  --iforest-config configs/iforest.yaml \
	  --ppo-config configs/ppo.yaml \
	  --eval-config configs/eval.yaml \
	  --out-root reports/runs

run_quick:
	PYTHONPATH=. $(PY) -m runs.cli \
	  --data-config configs/data.yaml \
	  --iforest-config configs/iforest.yaml \
	  --ppo-config configs/ppo.yaml \
	  --eval-config configs/eval.yaml \
	  --out-root reports/runs \
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
	  --data-config configs/data.yaml \
	  --iforest-config configs/iforest.yaml \
	  --ppo-config configs/ppo.yaml \
	  --eval-config configs/eval.yaml \
	  --robustness-config configs/robustness.yaml \
	  --out-root reports/runs

dev:
	@$(PY) -m pip install -U pip
	@$(PY) -m pip install -U ruff

pytest:
	@PYTHONPATH="$(CURDIR)" $(PY) -m pytest -q
