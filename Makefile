.PHONY: lint format test

lint:
	ruff check src/ web/ experiments/ tests/

format:
	ruff format src/ web/ experiments/ tests/
	ruff check --fix src/ web/ experiments/ tests/

test:
	python -m pytest tests/ -q --tb=short
