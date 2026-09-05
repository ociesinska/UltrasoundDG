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
