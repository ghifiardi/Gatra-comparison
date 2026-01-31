.PHONY: format lint test train_a train_b eval serve dev run run_quick

format:
	@python3.11 -m ruff format .

lint:
	@python3.11 -m ruff format --check .
	@python3.11 -m ruff check .

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
	PYTHONPATH=. python -m runs.cli \
	  --data-config configs/data.yaml \
	  --iforest-config configs/iforest.yaml \
	  --ppo-config configs/ppo.yaml \
	  --eval-config configs/eval.yaml \
	  --out-root reports/runs

run_quick:
	PYTHONPATH=. python -m runs.cli \
	  --data-config configs/data.yaml \
	  --iforest-config configs/iforest.yaml \
	  --ppo-config configs/ppo.yaml \
	  --eval-config configs/eval.yaml \
	  --out-root reports/runs \
	  --quick

dev:
	@python3.11 -m pip install -U pip
	@python3.11 -m pip install -U ruff

pytest:
	@PYTHONPATH="$(CURDIR)" python3.11 -m pytest -q
