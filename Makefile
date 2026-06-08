.PHONY: test lint lint-all format run clean coverage docker-build docker-run demo ablation baselines benchmark

PYTHON ?= python3

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

coverage:
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=src --cov=verarag --cov-report=term-missing --cov-report=html

lint:
	$(PYTHON) -m ruff check verarag src web experiments examples tests
	$(PYTHON) -m mypy src/ verarag/ --config-file mypy.ini

lint-all:
	$(PYTHON) -m ruff check verarag/ src/ web/ experiments/ examples/ tests/
	$(PYTHON) -m mypy src/ verarag/ --config-file mypy.ini

format:
	$(PYTHON) -m ruff format verarag/ src/ web/ experiments/ examples/ tests/
	$(PYTHON) -m ruff check --fix verarag/ src/ web/ experiments/ examples/ tests/

run:
	python -m uvicorn web.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

demo:
	python -m uvicorn web.app:create_app --factory --host 0.0.0.0 --port 8000

ablation:
	$(PYTHON) experiments/run_ablation.py --demo

baselines:
	$(PYTHON) experiments/run_baselines.py --demo

benchmark:
	$(PYTHON) experiments/run_verabench.py --demo

docker-build:
	docker build -t verarag .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/data:/app/data verarag

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov .coverage .mypy_cache
