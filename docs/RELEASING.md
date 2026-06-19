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
- VeraBench question-level and shared-evidence dependency interval metadata,
  plus paired-comparison compatibility.

## Pre-Release Checklist

1. Update version in `pyproject.toml`, `src/__init__.py`, and `CITATION.cff`.
2. Update `CHANGELOG.md`.
3. Regenerate result summaries if any real VeraBench run changed:

   ```bash
   python experiments/build_verabench_leaderboard.py \
     results/verabench_v112_full.json \
     --output docs/RESULTS.md
   ```

   If answer scoring changed without a benchmark data change, rescore saved
   reports offline. The command rejects missing or mismatched benchmark
   fingerprints by default:

   ```bash
   verarag-validate-benchmark
   verarag-audit-contamination \
     --reference /path/to/local/training-or-public-corpus \
     --output results/verabench_contamination_audit.json
   python scripts/migrate_verabench_span_traceability_v112.py --check
   verarag-rescore results/verabench_full.json \
     --output results/verabench_full_rescored.json
   ```

   If benchmark annotations changed, rerun the benchmark. A cross-version
   rescore is allowed only for historical sensitivity analysis with
   `--allow-benchmark-mismatch`; keep the source identity and label the result
   non-comparable. Use `--allow-unverified` only for legacy reports that lack
   benchmark metadata.

   For partitioned runs, merge only reports with matching benchmark, code,
   config, model, and metric signatures:

   ```bash
   verarag-merge-reports results/part-a.json results/part-b.json \
     --require-complete --output results/verabench_full.json
   ```

   When claiming an improvement, compare complete aligned reports instead of
   point estimates alone:

   ```bash
   verarag-compare-reports \
     results/baseline.json \
     results/candidate.json \
     --output results/paired_comparison.md
   ```

   For surgical regressions, rerun exact question ids before a full expensive
   evaluation:

   ```bash
   verarag-benchmark --config configs/verabench_v112_canonical.yaml \
     --ids V036 V048 V084 \
     --output results/verabench_targeted_failures.json \
     --restart
   ```

4. Run quality gates:

   ```bash
   make release-check
   ```

   This runs lint, mypy, committed-secret scanning, release version identity
   validation, Python support metadata validation, local documentation link,
   anchor, and command-reference validation, no-key quickstart example
   validation, deployment config validation, pre-commit config validation,
   dependency metadata validation, open-source project metadata validation, CycloneDX SBOM
   generation/validation, the full local test suite with an 80% total coverage
   gate, VeraBench release health, offline release artifact manifest validation,
   sdist/wheel build, package content validation, installed-wheel smoke testing,
   and release checksum manifest validation.
   Project metadata validation also checks the PyPI/GitHub URL surface,
   governance/citation files, CODEOWNERS, issue-template security routing,
   Dependabot coverage, CodeQL scanning, OpenSSF Scorecard, GitHub Actions
   SHA pinning, GitHub Actions least-privilege/timeout settings, and the
   release workflow's tag-only publishing controls.
   Release health checks benchmark data sync and traceability, the external
   conflict fixture, blind annotation packet generation, demo metric plumbing,
   and demo paired comparison. It also writes
   `build/release-health/release-artifacts-manifest.json`, recording each
   generated audit/report artifact's path, SHA-256, byte size, generation
   command, and key metric summary, then validates the manifest before passing.
   Override the location with `RELEASE_HEALTH_DIR=/path make benchmark-check`
   when collecting artifacts elsewhere. Re-validate an existing manifest without
   rerunning the benchmark checks with `make release-artifacts-check` or
   `verarag-release-health --validate-manifest path/to/release-artifacts-manifest.json --manifest-root path/to/root`;
   manifest artifact paths must stay inside the manifest root. Package
   validation reads `pyproject.toml` and checks the exact current-version
   sdist/wheel, so older artifacts left in `dist/` do not make a release look
   valid. After package validation, `make release-checksums-check` writes
   `build/release-checksums.json` with SHA-256 and byte-size records for the
   sdist, wheel, CycloneDX SBOM, and release-health artifact manifest; verify a
   saved checksum manifest with
   `verarag-release-checksums --validate-manifest build/release-checksums.json`.

   The GitHub release workflow is intentionally conservative: manual
   `workflow_dispatch` runs build, provenance attestation, and auditable
   artifact upload only, while `v*.*.*` tags additionally enable PyPI OIDC
   trusted publishing and GitHub Release creation. The attested and uploaded
   artifact set must include the wheel, sdist, CycloneDX SBOM, release-health
   manifest, and checksum manifest.

5. Inspect package contents separately when debugging release failures:

   ```bash
   make package-check
   ```

   To debug the first-run developer experience without running the full release
   gate, use:

   ```bash
   make version-check
   verarag-validate-version --json
   make python-support-check
   verarag-validate-python --json
   make configs-check
   verarag-validate-configs --json
   make results-check
   verarag-validate-results --json
   make examples-check
   verarag-validate-examples --json
   make deployment-check
   verarag-validate-deployment --json
   make precommit-check
   verarag-validate-precommit --json
   ```

6. Smoke test the wheel:

   ```bash
   python -m pip install --no-deps --upgrade --target /tmp/verarag-wheel-test \
     dist/verarag-0.1.0-py3-none-any.whl
   PYTHONPATH=/tmp/verarag-wheel-test python -c \
     "from verarag import VeraRAG, load_verabench; b=load_verabench(); print(VeraRAG.__name__, len(b.corpus), len(b.questions))"
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-web --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-doctor --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-benchmark --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-leaderboard --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-compare-reports --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-build-conflict-data --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-train-conflict --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-audit-conflict-model --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-conflict-ablation --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-validate-benchmark --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-audit-contamination --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-scan-secrets --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-validate-version --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-validate-python --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-validate-configs --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-validate-docs --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-validate-results --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-validate-examples --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-validate-deployment --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-validate-precommit --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-validate-deps --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-validate-metadata --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-validate-package --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-validate-install --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-release-health --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-merge-reports --help
   PYTHONPATH=/tmp/verarag-wheel-test /tmp/verarag-wheel-test/bin/verarag-rescore --help
   ```

   For a local workstation audit before publishing, also run
   `verarag-doctor --json` and `verarag-scan-secrets --include-ignored` from
   the source checkout. The doctor reports Python, dependency, data-file, and
   provider environment readiness without printing API key values; the secret
   scan checks ignored `.env.local` style files while still redacting any
   detected values.
   CI systems that accept code-scanning reports can use
   `verarag-scan-secrets --sarif > secret-scan.sarif`; the SARIF output contains
   only rule ids, paths, line numbers, and redacted snippets.

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
