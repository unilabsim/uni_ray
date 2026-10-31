.PHONY: sync lint typecheck test check

sync:
	uv sync --locked

lint:
	uv run ruff check .

typecheck:
	uv run mypy src/uniray

test:
	uv run pytest -q

check: lint typecheck test
