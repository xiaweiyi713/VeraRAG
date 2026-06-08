# Releasing VeraRAG

This checklist keeps releases reproducible and avoids publishing a package whose
docs, benchmark summaries, and artifacts disagree.

## Version Policy

VeraRAG uses semantic versioning:

- `MAJOR`: breaking public API, output schema, CLI, config, or benchmark format
  changes.
- `MINOR`: new backward-compatible features, metrics, providers, docs, or
  benchmark tooling.
- `PATCH`: bug fixes, documentation corrections, packaging fixes, and test
  improvements.

The public compatibility surface is:

- imports under `verarag.*`;
- console scripts in `pyproject.toml`;
- `VeraRAGOutput.to_dict()` keys;
- VeraBench report top-level keys and `question_results` row fields;
- packaged VeraBench fallback loading via `load_verabench()`.

## Pre-Release Checklist

1. Update version in `pyproject.toml`.
2. Update `CHANGELOG.md`.
3. Regenerate result summaries if any real VeraBench run changed:

   ```bash
   python experiments/build_verabench_leaderboard.py \
     results/verabench_full.json \
     results/verabench_full_v2.json \
     results/verabench_full_v3.json \
     --output docs/RESULTS.md
   ```

4. Run quality gates:

   ```bash
   make lint
   python -m pytest tests -q
   python -m build --sdist --wheel --no-isolation
   ```

5. Inspect package contents:

   ```bash
   python - <<'PY'
   import tarfile, zipfile
   from pathlib import Path

   sdist = Path("dist/verarag-0.1.0.tar.gz")
   wheel = Path("dist/verarag-0.1.0-py3-none-any.whl")

   with tarfile.open(sdist) as tf:
       sdist_names = set(tf.getnames())
   assert "verarag-0.1.0/docs/API.md" in sdist_names
   assert "verarag-0.1.0/docs/RESULTS.md" in sdist_names
   assert "verarag-0.1.0/examples/quickstart.py" in sdist_names
   assert not any("docs/superpowers" in name for name in sdist_names)

   with zipfile.ZipFile(wheel) as zf:
       wheel_names = set(zf.namelist())
   assert "verarag/__init__.py" in wheel_names
   assert "verarag/py.typed" in wheel_names
   assert "src/benchmark/data/verabench/questions.jsonl" in wheel_names
   assert "web/templates/index.html" in wheel_names
   print("package contents ok")
   PY
   ```

6. Smoke test the wheel:

   ```bash
   python -m pip install --no-deps --upgrade --target /tmp/verarag-wheel-test \
     dist/verarag-0.1.0-py3-none-any.whl
   PYTHONPATH=/tmp/verarag-wheel-test python -c \
     "from verarag import VeraRAG, load_verabench; b=load_verabench(); print(VeraRAG.__name__, len(b.corpus), len(b.questions))"
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-web --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-benchmark --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-leaderboard --help
   ```

7. Confirm no whitespace errors:

   ```bash
   git diff --check
   ```

## Publishing Notes

Raw files under `results/` are ignored by default because real benchmark outputs
can be large and environment-specific. Publish raw JSON only for intentional
reproducibility releases. For normal releases, commit `docs/RESULTS.md` and the
generation command instead.

If a release changes benchmark scoring, mark the result as non-comparable with
older reports in `docs/RESULTS.md` and explain the metric change in
`CHANGELOG.md`.
