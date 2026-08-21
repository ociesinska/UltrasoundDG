.PHONY: format lint format_and_lint sync

sync:
	uv sync

lint:
	uv run ruff check . --fix

format:
	uv run ruff format .

format_and_lint: lint format
