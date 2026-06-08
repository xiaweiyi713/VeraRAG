# Evaluation Guide

This guide explains how to reproduce and interpret VeraBench results.

## Demo Evaluation

Demo mode scores ground truth against itself. It validates loader, grouping, metrics, and reporting plumbing; it is not a model quality result.

```bash
python experiments/run_verabench.py --demo
```

## Real Pipeline Evaluation

Use a fixed config and save output JSON:

```bash
DEEPSEEK_API_KEY=<key> python experiments/run_verabench.py \
  --config configs/deepseek_run.yaml \
  --output results/verabench_full.json \
  --restart
```

The runner writes an incremental checkpoint by default. If interrupted, rerun the same command without `--restart` to resume.

## Targeted Diagnostics

For conflict work:

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

Key fields:

- `conflict_summary.gold_conflicts`: expected conflict edges.
- `conflict_summary.predicted_conflicts`: predicted disagreement edges only; support edges are excluded.
- `conflict_summary.false_positives`: over-detected conflicts.
- `conflict_summary.false_negatives`: missed gold conflicts.
- `conflict_summary.dominant_failure`: `over_detection`, `under_detection`, `mixed`, or `none`.

## Calibration Diagnostics

Analyze confidence bins:

```bash
python experiments/analyze_verabench_results.py results/verabench_full.json
```

Generate a reliability diagram:

```bash
python experiments/calibration_curve.py \
  --input results/verabench_full.json \
  --output results/calibration_curve.svg
```

## Leaderboard / Result Summary

Generate a commit-friendly Markdown summary from one or more ignored raw result
JSON files:

```bash
python experiments/build_verabench_leaderboard.py \
  results/verabench_full.json \
  results/verabench_full_v2.json \
  results/verabench_full_v3.json \
  --output docs/RESULTS.md
```

Installed package equivalent:

```bash
verarag-leaderboard results/verabench_full_v3.json --output docs/RESULTS.md
```

Use `--format json` when another tool needs machine-readable output.

Key fields:

- `ece`: expected calibration error; lower is better.
- `brier_score`: squared error of confidence vs correctness; lower is better.
- `calibration_bins`: per-bin count, average confidence, actual accuracy, and gap.

## Reporting Standards

When reporting numbers, include:

- model provider and model name;
- temperature and max tokens;
- config path;
- git commit;
- data version or VeraBench commit;
- whether the run is demo or real pipeline mode;
- command used to produce the result.
