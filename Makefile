.PHONY: install test train lint serve serve-dev

install:
	uv sync

test:
	uv run python -m pytest

train:
	uv run python train_pipeline.py

serve:
	uv run python serve_api.py

serve-dev:
	uv run python serve_api.py --reload

lint:
	uv run ruff check .
	uv run black .
