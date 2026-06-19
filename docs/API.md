# API Reference

This page documents the stable user-facing API. Internal imports under `src.*`
still work for repository development, but downstream users should prefer the
`verarag` package.

## Installation

```bash
pip install verarag
```

For local development:

```bash
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install "verarag[dense]"   # sentence-transformers + FAISS
pip install "verarag[eval]"    # matplotlib/seaborn reports
pip install "verarag[all]"     # all optional dependencies plus dev tools
```

## Public Imports

```python
from verarag import VeraRAG, VeraRAGOutput, create_verarag
from verarag import VeraBenchEvaluator, VeraBenchLoader, load_verabench
from verarag import build_conflict_pair_examples
```

Submodules are also stable:

```python
from verarag.pipeline import VeraRAG
from verarag.benchmark import BenchmarkQuestion, CorpusDocument, VeraBench
```

## Pipeline

### `VeraRAG(config: dict | None = None)`

Creates the end-to-end verifiable RAG pipeline.

Minimal OpenAI-compatible config:

```python
pipeline = VeraRAG({
    "llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": "${OPENAI_API_KEY}",
        "temperature": 0.0,
        "max_tokens": 800,
    },
    "pipeline": {
        "max_retrieval_rounds": 2,
        "enable_conflict_graph": True,
        "enable_uncertainty": True,
        "enable_verification": True,
        "enable_repair": True,
    },
})
```

Supported providers are `openai`, `anthropic`, `ollama`, `dashscope`,
`zhipuai`, and `deepseek`.

### `index_documents(documents)`

Builds the retrieval index.

Each document should be a dictionary with at least:

```python
{
    "id": "D001",
    "title": "Document title",
    "content": "Text used for retrieval",
    "metadata": {"source": "report"},
}
```

The retriever accepts common aliases such as `text`/`content`; prefer `content`
for new code.

### `query(question, max_rounds=None)`

Runs the complete pipeline and returns `VeraRAGOutput`.

```python
result = pipeline.query("这项政策的最新口径和旧口径是否冲突？")

print(result.answer)
print(result.confidence)
print(result.metadata["num_evidence"])
print(result.metadata["num_conflicts"])
```

### `query_stream(question, max_rounds=None, callback=None)`

Runs the same pipeline while emitting stage events.

```python
def on_event(event_type: str, data: dict) -> None:
    print(event_type, data)

result = pipeline.query_stream("问题", callback=on_event)
```

Common event types:

- `stage`: a pipeline stage started.
- `task_analysis`: task type, complexity, keywords, and retrieval need.
- `decomposition`: generated subquestions.
- `evidence`: retrieval and normalization progress.
- `conflict`: detected support/conflict edges.
- `uncertainty`: action chosen by the uncertainty controller.
- `reasoning`: draft answer, claims, and reasoning steps.
- `verification`: claim verification result.
- `complete`: elapsed time and final counts.

### `batch_query(questions, max_rounds=None)`

Runs `query()` for each question and returns a list of `VeraRAGOutput`.

### `create_verarag(config_path=None)`

Loads a YAML config and returns a `VeraRAG` instance.

```python
pipeline = create_verarag("configs/model.yaml")
```

## Output Schema

`VeraRAGOutput` is a dataclass with:

- `question`: original user question.
- `answer`: final repaired answer.
- `answer_claims`: atomic answer claims with verification metadata.
- `evidence`: normalized evidence items.
- `reasoning_chain`: structured reasoning steps.
- `conflict_report`: conflict graph summary and edges.
- `verification_report`: claim-level verification report, if enabled.
- `confidence`: calibrated final confidence.
- `uncertainty`: uncertainty breakdown.
- `metadata`: operational counts such as `num_evidence`, `num_conflicts`,
  `retrieval_rounds`, and `elapsed_time`.

Use `result.to_dict()` for JSON serialization.

## VeraBench

### `load_verabench(data_dir=None)`

Loads the benchmark. Without `data_dir`, VeraRAG first uses repository data and
then falls back to package data bundled inside the wheel.

```python
from verarag import load_verabench

bench = load_verabench()
print(bench.stats())
```

Expected bundled data in version `0.1.0`:

- 57 corpus documents.
- 152 benchmark questions.
- 6 question types: single evidence, multi evidence, conflict, temporal,
  unanswerable, and misleading.
- VeraBench ontology version 1.1.2 distribution: 27 / 26 / 11 / 25 / 26 / 37.

### `VeraBenchEvaluator`

Runs demo, baseline, or pipeline evaluations.

```python
from verarag import VeraBenchEvaluator, load_verabench

bench = load_verabench()
evaluator = VeraBenchEvaluator(benchmark=bench)
report = evaluator.evaluate(max_questions=5)
print(report.to_dict())
```

Without a `pipeline_factory`, evaluation runs in demo mode by scoring ground
truth against itself. This is useful for validating loader, metrics, and report
plumbing; it is not a model quality score. The demo identity check fills gold
answers, evidence, conflicts, behaviors, and confidence, so all supervised
metrics are internally consistent instead of looking like retrieval failures.

The `verarag-benchmark` CLI is stricter than direct evaluator construction:
exactly one of `--demo` or `--config` is required. A requested real run exits
non-zero if its configuration or pipeline cannot be loaded and never falls back
to demo mode.

Serialized reports include both question-type-stratified
`confidence_intervals` and shared-evidence-cluster
`dependency_robust_confidence_intervals`. Sparse conditional metrics also
report `effective_resamples_by_metric`, because bootstrap replicates without a
defined denominator are excluded for that metric.

For real evaluation:

```python
def make_pipeline():
    pipeline = create_verarag("configs/verabench_v112_canonical.yaml")
    # index documents before returning the pipeline
    return pipeline

evaluator = VeraBenchEvaluator(benchmark=bench, pipeline_factory=make_pipeline)
report = evaluator.evaluate(question_types=["conflict"], max_questions=10)
```

## Console Scripts

Installed packages expose:

```bash
verarag-web --port 8000
verarag-doctor --json
verarag-benchmark --demo
verarag-analyze results/verabench_full.json --risk-coverage-svg results/risk_coverage.svg --risk-coverage-csv results/risk_coverage.csv
verarag-rescore results/verabench_full.json --output results/verabench_rescored.json
verarag-merge-reports results/part-a.json results/part-b.json --output results/full.json
verarag-calibration --input results/verabench_full.json --correctness-field correct --output results/calibration_curve.svg
verarag-calibrate-report --input results/verabench_full.json --output results/verabench_full_calibrated.json --method platt --group-field actual_behavior
verarag-leaderboard results/verabench_full_v3.json --output docs/RESULTS.md
verarag-compare-reports results/baseline.json results/candidate.json --output results/comparison.md
verarag-audit-contamination --reference /path/to/local/corpus --containment-threshold 0.85 --output results/contamination_audit.json
verarag-scan-secrets
verarag-scan-secrets --include-ignored
verarag-scan-secrets --sarif > secret-scan.sarif
verarag-validate-version
verarag-validate-python
verarag-validate-configs --json
verarag-validate-docs
verarag-validate-results
verarag-validate-examples
verarag-validate-deployment
verarag-validate-precommit
verarag-validate-deps
verarag-validate-metadata
verarag-validate-package --dist-dir dist
verarag-validate-install --dist-dir dist
verarag-release-health
verarag-generate-sbom --output build/sbom/verarag-sbom.cdx.json --check
verarag-release-checksums --output build/release-checksums.json --check
verarag-build-conflict-data --output-dir outputs/conflict_pairs_v112_leakfree
verarag-train-conflict --train outputs/conflict_pairs_v112_leakfree/train.jsonl --val outputs/conflict_pairs_v112_leakfree/val.jsonl --dry-run
verarag-compare-conflicts --split test --learned-model-path outputs/conflict_cross_encoder_v112_leakfree_seed13
verarag-audit-conflict-model --help
verarag-build-external-annotation-packet --data-dir data/external/conflict_mini_v1 --output-dir outputs/conflict_mini_packet --annotator ann_a --annotator ann_b
verarag-compile-external-annotations --packet-dir outputs/conflict_mini_packet --output-dir outputs/conflict_mini_compiled
verarag-validate-external-conflicts --data-dir data/external/conflict_mini_v1 --min-questions 6
verarag-conflict-ablation --learned-model-path outputs/conflict_cross_encoder_expanded_balanced_seed13 --plan-only
```

`verarag-rescore` verifies the source benchmark version and corpus/question
fingerprints by default. Cross-version and metadata-free legacy inputs require
the explicit diagnostic overrides `--allow-benchmark-mismatch` and
`--allow-unverified`, respectively.

`verarag-analyze` computes offline diagnostics for a saved VeraBench report,
including behavior failures, conflict failures, confidence AUROC, AURC, and
coverage@accuracy. Use `--risk-coverage-svg` to write a publication-friendly
risk-coverage curve and `--risk-coverage-csv` to export the plotted points.

`verarag-calibration` defaults to the boolean `correct` row field, which tracks
VeraBench behavior correctness. Use `--correctness-field` to inspect another
boolean outcome such as `premise_refutation_correct`. The command rejects empty
reports, missing fields, non-boolean correctness values, and out-of-range
confidence scores instead of drawing a misleading reliability diagram.

`verarag-calibrate-report` fits held-out Platt scaling by default and writes a
new report JSON with calibrated confidence values. Use `--method temperature`
for temperature scaling, `--summary-output` for a separate before/after metric
summary, and `--seed` / `--calibration-fraction` to reproduce the deterministic
correctness-stratified split. Use `--group-field actual_behavior` to fit
separate behavior-level calibrators for answer, abstention, premise-correction,
and conflict-note rows. Groups with too few calibration rows or only one
correctness class fall back to a smoothed empirical constant by default; use
`--group-fallback global` to reuse the global model instead. Per-row diagnostics
record the group value, model scope, mode, and fallback reason when applicable.

`verarag-doctor` summarizes local readiness without exposing secrets. It checks
the supported Python floor, required runtime modules, optional feature
dependencies, core VeraBench files, and whether known LLM provider environment
variables are set. Missing optional feature dependencies and absent API keys are
warnings by default; use `--fail-on-warnings` for stricter workstation or CI
audits.

`verarag-validate-configs` validates the default YAML configuration surface. It
parses `configs/*.yaml`, checks runtime and dataset section shapes, enforces
probability thresholds in `[0, 1]`, requires positive integer budget fields,
verifies boolean feature flags, and rejects literal `llm.api_key` values that
are not environment-variable placeholders.

## Compatibility Policy

The `verarag.*` API is the preferred public surface. Changes that remove or
rename public imports, constructor arguments, output fields, console scripts, or
bundled VeraBench loading behavior should be treated as breaking changes and
called out in `CHANGELOG.md`.
