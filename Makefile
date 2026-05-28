.PHONY: test lint format run clean coverage docker-build docker-run demo ablation baselines benchmark

test:
	python -m pytest tests/ -v --tb=short

coverage:
	python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing --cov-report=html

lint:
	ruff check src/ web/ experiments/ tests/
	mypy src/ --config-file mypy.ini

format:
	ruff format src/ web/ experiments/ tests/
	ruff check --fix src/ web/ experiments/ tests/

run:
	python -m uvicorn web.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

demo:
	python -m uvicorn web.app:create_app --factory --host 0.0.0.0 --port 8000

ablation:
	python experiments/run_ablation.py --demo

baselines:
	python experiments/run_baselines.py --demo

benchmark:
	python experiments/run_verabench.py --demo

docker-build:
	docker build -t verarag .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/data:/app/data verarag

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov .coverage .mypy_cache
