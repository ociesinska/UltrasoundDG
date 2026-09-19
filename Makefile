.PHONY: check format lint format_and_lint sync test

sync:
	uv sync --extra dev

test:
	uv run --extra dev pytest

check:
	uv run --extra dev ruff check .
	uv run --extra dev ruff format --check .
	uv run --extra dev pytest

lint:
	uv run --extra dev ruff check . --fix

format:
	uv run --extra dev ruff format .

format_and_lint: lint format

.PHONY: mlflow
mlflow:
	mkdir -p artifacts/mlflow/artifacts
	uv run mlflow ui \
		--backend-store-uri sqlite:///artifacts/mlflow/mlflow.db \
		--artifacts-destination ./artifacts/mlflow/artifacts \
		--serve-artifacts \
		--port 8080