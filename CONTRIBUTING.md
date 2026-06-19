# Contributing to VeraRAG

Thanks for helping improve VeraRAG. This project is both a research prototype and an engineering system, so contributions should preserve reproducibility, testability, and clear evaluation evidence.

## Development Setup

```bash
git clone https://github.com/xiaweiyi713/VeraRAG.git
cd VeraRAG
python -m pip install -r requirements.txt
python -m pip install -e .
verarag-doctor --json
```

Python 3.10+ is required. Optional dense retrieval, FAISS, NLI, and PDF features may need larger dependencies; BM25 and demo mode should work with the base install.
`verarag-doctor` reports the local Python version, required modules, optional feature dependencies, benchmark data files, and provider environment variable presence without printing secret values.

To mirror the fast local gates before each commit:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Before Opening a PR

Run the fast checks:

```bash
python -m pytest tests -q
make lint
make version-check
make python-support-check
make doctor-check
make configs-check
make docs-check
make results-check
make examples-check
make deployment-check
make precommit-check
make deps-check
```

If your change affects packaging, the Web UI, benchmark data, or default configs, also run:

```bash
make package-check
```

Before a release or any broad evaluation/packaging change, run the full gate:

```bash
make release-check
```

`make lint` runs the repository-wide Ruff gate, mypy over `src/` and
`verarag/`, and the high-confidence secret scanner. `make precommit-check`
validates that `.pre-commit-config.yaml`, CI, README, and this guide still
point at the same local quality hooks. `make release-check` also runs
documentation, no-key quickstart example validation, deployment config
validation, version identity validation, published results validation,
Python support validation, environment diagnosis, YAML config validation, pre-commit config validation, dependency metadata,
project metadata, coverage, benchmark health, package archive, and
installed-wheel smoke checks.

## Evaluation Changes

If a PR changes retrieval, conflict detection, confidence, answer behavior, benchmark scoring, or prompts, include at least one of:

- a new or updated unit test;
- a small `run_verabench.py --demo` or mocked evaluation result;
- a real VeraBench smoke result if the change depends on LLM behavior.

For expensive real runs, prefer a targeted command:

```bash
DEEPSEEK_API_KEY=<key> python experiments/run_verabench.py \
  --config configs/deepseek_run.yaml \
  --types conflict \
  --max 10 \
  --output results/conflict_diagnostic.json \
  --restart
```

Then inspect:

```bash
python experiments/analyze_verabench_results.py results/conflict_diagnostic.json
```

## Pull Request Expectations

- Keep changes scoped and explain the reason.
- Do not commit API keys, local database files, `.verarag_key`, generated caches, or private data.
- Update `README.md`, `DEV_PROGRESS.md`, or `docs/EVALUATION.md` when behavior, commands, metrics, or benchmark interpretation changes.
- Prefer deterministic IDs and reproducible outputs for benchmark work.
