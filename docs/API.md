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
plumbing; it is not a model quality score.

For real evaluation:

```python
def make_pipeline():
    pipeline = create_verarag("configs/deepseek_run.yaml")
    # index documents before returning the pipeline
    return pipeline

evaluator = VeraBenchEvaluator(benchmark=bench, pipeline_factory=make_pipeline)
report = evaluator.evaluate(question_types=["conflict"], max_questions=10)
```

## Console Scripts

Installed packages expose:

```bash
verarag-web --port 8000
verarag-benchmark --demo
verarag-analyze results/verabench_full.json
verarag-calibration --input results/verabench_full.json --output results/calibration_curve.svg
verarag-leaderboard results/verabench_full_v3.json --output docs/RESULTS.md
```

## Compatibility Policy

The `verarag.*` API is the preferred public surface. Changes that remove or
rename public imports, constructor arguments, output fields, console scripts, or
bundled VeraBench loading behavior should be treated as breaking changes and
called out in `CHANGELOG.md`.
