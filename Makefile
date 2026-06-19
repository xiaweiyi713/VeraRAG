.PHONY: test lint lint-all format security security-local version-check python-support-check doctor-check configs-check docs-check results-check examples-check deployment-check precommit-check deps-check metadata-check sbom-check release-checksums-check build package-check installed-wheel-check benchmark-check release-artifacts-check external-fixture-check release-check run clean coverage coverage-check docker-build docker-run demo ablation baselines benchmark gpu-sync gpu-status gpu-check

PYTHON ?= python3
COVERAGE_MIN ?= 80
RELEASE_HEALTH_DIR ?= build/release-health
RELEASE_CHECKSUMS ?= build/release-checksums.json

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

coverage:
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=src --cov=verarag --cov-report=term-missing --cov-report=html --cov-report=json:coverage.json

coverage-check:
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=src --cov=verarag --cov-report=term-missing --cov-report=json:coverage.json --cov-fail-under=$(COVERAGE_MIN)

lint:
	$(PYTHON) -m ruff check verarag src web experiments examples tests configs scripts demo.py demo_local.py run_web.py
	$(PYTHON) -m mypy src/ verarag/ --config-file mypy.ini
	$(PYTHON) experiments/scan_secrets.py

lint-all:
	$(PYTHON) -m ruff check verarag/ src/ web/ experiments/ examples/ tests/ configs/ scripts/ demo.py demo_local.py run_web.py
	$(PYTHON) -m mypy src/ verarag/ --config-file mypy.ini
	$(PYTHON) experiments/scan_secrets.py

security:
	$(PYTHON) experiments/scan_secrets.py

security-local:
	$(PYTHON) experiments/scan_secrets.py --include-ignored

version-check:
	$(PYTHON) experiments/validate_version_identity.py

python-support-check:
	$(PYTHON) experiments/validate_python_support.py

doctor-check:
	$(PYTHON) experiments/doctor.py

configs-check:
	$(PYTHON) experiments/validate_configs.py

docs-check:
	$(PYTHON) experiments/validate_docs.py

results-check:
	$(PYTHON) experiments/validate_results.py

examples-check:
	$(PYTHON) experiments/validate_examples.py

deployment-check:
	$(PYTHON) experiments/validate_deployment.py

precommit-check:
	$(PYTHON) experiments/validate_precommit.py

deps-check:
	$(PYTHON) experiments/validate_dependency_metadata.py

metadata-check:
	$(PYTHON) experiments/validate_project_metadata.py

sbom-check:
	$(PYTHON) experiments/generate_sbom.py --output build/sbom/verarag-sbom.cdx.json --check

release-checksums-check:
	$(PYTHON) experiments/generate_release_checksums.py --output $(RELEASE_CHECKSUMS) --check

build:
	$(PYTHON) -m build --sdist --wheel

package-check: build
	$(PYTHON) experiments/validate_package_contents.py --dist-dir dist
	$(PYTHON) experiments/validate_installed_wheel.py --dist-dir dist

installed-wheel-check: build
	$(PYTHON) experiments/validate_installed_wheel.py --dist-dir dist

benchmark-check:
	$(PYTHON) experiments/validate_release_health.py --output-dir $(RELEASE_HEALTH_DIR)

release-artifacts-check:
	$(PYTHON) experiments/validate_release_health.py --validate-manifest $(RELEASE_HEALTH_DIR)/release-artifacts-manifest.json --manifest-root $(RELEASE_HEALTH_DIR)

external-fixture-check:
	$(PYTHON) experiments/validate_external_conflict_set.py --data-dir data/external/conflict_mini_v1 --min-questions 6 --output /tmp/external-conflict-audit.json
	$(PYTHON) experiments/build_external_annotation_packet.py --data-dir data/external/conflict_mini_v1 --output-dir /tmp/external-conflict-packet --annotator ann_a --annotator ann_b --overwrite
	$(PYTHON) experiments/compile_external_annotations.py --help

release-check: lint version-check python-support-check doctor-check configs-check docs-check results-check examples-check deployment-check precommit-check deps-check metadata-check sbom-check coverage-check benchmark-check release-artifacts-check package-check release-checksums-check

format:
	$(PYTHON) -m ruff format verarag/ src/ web/ experiments/ examples/ tests/ configs/ scripts/ demo.py demo_local.py run_web.py
	$(PYTHON) -m ruff check --fix verarag/ src/ web/ experiments/ examples/ tests/ configs/ scripts/ demo.py demo_local.py run_web.py

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

gpu-sync:
	scripts/sync_windows_gpu.sh

gpu-check:
	scripts/check_windows_conflict_training_ready.sh

gpu-status:
	scripts/windows_gpu_status.sh

docker-build:
	docker build -t verarag .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/data:/app/data verarag

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov .coverage .mypy_cache
