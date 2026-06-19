# VeraBench Results

> Compatibility note: VeraBench v1.1 changes the question ontology, gold
> conflicts, difficulty labels, and metric versions. VeraBench v1.1.1 then
> clarifies V084's time scope; v1.1.2 repairs evidence-span traceability without
> changing labels or metrics. v1.0 and v1.1 rows are historical and must not be
> compared directly with v1.1.2 runs unless explicitly labeled.

This file is generated from saved `run_verabench.py --output` JSON reports.
Commit raw result JSON files only when intentionally publishing a reproducibility artifact; otherwise keep them in ignored `results/` and regenerate this summary.

Generation command:

```bash
python experiments/build_verabench_leaderboard.py \
  results/verabench_full.json \
  results/verabench_full_v2.json \
  results/verabench_full_v3.json \
  --allow-unverified \
  --allow-mixed-benchmarks \
  --output docs/RESULTS.md
```

The override flags above are required because this file intentionally preserves
legacy, non-comparable historical runs. New formal leaderboards should omit
them and use only complete reports with matching benchmark and metric
fingerprints.

## VeraBench v1.1.2 Status

Current repository data is VeraBench v1.1.2:

- Corpus SHA-256:
  `bb99ce4d8b4ba7ee5a938595c7c786a8ad33601e4f096bec1bed844c43b6a8f3`
- Questions SHA-256:
  `c19e7401cbcd2526fb1e085c911d61095c8ce5bd19a664c13698a08024436882`

The full 152-question DeepSeek run below was produced on v1.1 before the V084
wording clarification and evidence-span traceability patch. A v1.1.2 targeted
rerun should use:

```bash
DEEPSEEK_API_KEY=<key> verarag-benchmark \
  --config configs/verabench_v112_canonical.yaml \
  --ids V036 V048 V084 \
  --output results/verabench_v112_targeted_failures.json \
  --restart
```

The canonical v1.1.2 full run is fixed in
`configs/verabench_v112_canonical.yaml`: DeepSeek `deepseek-v4-flash`,
temperature `0.0`, `max_tokens=4000`, BM25 retrieval,
`max_retrieval_rounds=1`, all verification/conflict/uncertainty/repair stages
enabled, and statistical intervals using 2,000 bootstrap resamples with seed
`1729`. Its authoritative artifact path is
`outputs/remote_results/verabench_v112_canonical_deepseek.json`.

```bash
DEEPSEEK_API_KEY=<key> verarag-benchmark \
  --config configs/verabench_v112_canonical.yaml \
  --output outputs/remote_results/verabench_v112_canonical_deepseek.json \
  --restart
```

## VeraBench v1.1.2 Conflict CrossEncoder Negative Result

The learned conflict detector is not enabled by default. On 2026-06-15, the
Windows GPU matrix workflow reproduced the current three-seed negative result
with `scripts/start_windows_conflict_training_matrix.sh`:

| Seed | Selected threshold | Validation F1 | Test precision | Test recall | Test F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 13 | 0.336 | 0.667 | 0.188 | 1.000 | 0.316 |
| 17 | 0.604 | 0.333 | 0.000 | 0.000 | 0.000 |
| 23 | 0.302 | 0.400 | 0.188 | 1.000 | 0.316 |

The held-out gold-evidence detector A/B on the seed-13 model showed no gain:
rules and rules+learned both reached precision `0.750`, recall `1.000`, F1
`0.857`, and TP/FP/FN `3/1/0`. The report-only promotion audit rejected the
learned model on multi-seed test F1, dependency-robust lower bound, minimum
A/B sample size, and held-out improvement.

Current v1.1.2 gold-evidence rules-only diagnosis from
`compare_conflict_detectors.py` is more precise than the older F1≈0 summary:
all conflict-bearing rows score precision `0.9231`, recall `0.8000`, F1
`0.8571` with TP/FP/FN `12/1/3`. The remaining all-scope failure is mainly
under-detection on self-pair conflicts (`V021`, `V075`, `V122`), while the
dependency-aware test split has recall `1.0000` and one V017 false-positive
extra pair. This shifts the next Stage-1 work from blind model training to
targeted missed-pattern fixes plus strict precision-preserving promotion.

## Historical VeraBench v1.1 Full Run

Merged from two compatible partition reports with:

```bash
python experiments/merge_verabench_reports.py \
  outputs/remote_results/verabench_v11_part_a.json \
  outputs/remote_results/verabench_v11_part_b.json \
  --require-complete \
  --output outputs/remote_results/verabench_v11_full_merged_rescored.json
```

| Run | Model | Questions | Errors | Behavior Acc | Answer F1 | Evidence Recall | Conflict micro-F1 | Conflict TP / FP / FN | ECE | Avg Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| verabench_v11_full_merged_rescored | deepseek/deepseek-v4-flash | 152/152 | 0 | 0.980 | 0.459 | 0.952 | 0.897 | 13 / 1 / 2 | 0.563 | 32.1s |

Metric versions: `soft-f1-v2`, `behavior-v2`,
`gold-evidence-pair-micro-f1-v2`. Conflict scoring counts only predicted pairs
inside the question's gold evidence set; extra conflicts on retrieved distractor
documents are retained in diagnostics as unscored extraneous pairs.

### v1.1 Statistical Uncertainty

Offline rescoring with `stratified-question-bootstrap-v1` (2,000 resamples,
95% confidence, seed 1729) gives:

| Metric | Estimate | 95% interval |
| --- | ---: | ---: |
| Answer F1 | 0.459 | [0.431, 0.489] |
| Evidence Recall | 0.952 | [0.929, 0.974] |
| Behavior Accuracy | 0.980 | [0.954, 1.000] |
| Conflict micro-F1 | 0.897 | [0.700, 1.000] |
| Premise-refutation F1 | 0.976 | [0.937, 1.000] |
| ECE | 0.563 | [0.518, 0.606] |

The wide conflict interval reflects the small number of annotated conflict
pairs. Point estimates should not be used alone to claim superiority.

The same saved rows can be mapped by question ID to the current benchmark's 27
shared-evidence components. `evidence-cluster-bootstrap-v1` gives the following
dependency sensitivity intervals:

| Metric | Estimate | 95% cluster interval |
| --- | ---: | ---: |
| Answer F1 | 0.459 | [0.433, 0.488] |
| Evidence Recall | 0.952 | [0.916, 0.985] |
| Behavior Accuracy | 0.980 | [0.954, 1.000] |
| Conflict micro-F1 | 0.897 | [0.720, 1.000] |
| ECE | 0.563 | [0.502, 0.614] |
| Brier score | 0.403 | [0.370, 0.435] |

These are sensitivity estimates for evidence reuse, not retroactive proof that
the historical v1.1 run used the current v1.1.2 annotations.

### v1.1 Selective Prediction Diagnostic

The historical v1.1 rows were post-hoc calibrated with behavior-grouped Platt
scaling before drawing the selective-prediction curve. This diagnostic is not a
canonical v1.1.2 claim, but it demonstrates the ROADMAP stage-2 reporting path:
AURC `0.0328`, coverage@accuracy≥0.95 `0.572`, and coverage@accuracy≥0.90
`1.000`.

![Historical v1.1 behavior-calibrated risk-coverage curve](assets/verabench_v11_group_calibrated_risk_coverage.svg)

Curve points are archived in
[`assets/verabench_v11_group_calibrated_risk_coverage.csv`](assets/verabench_v11_group_calibrated_risk_coverage.csv).

### v1.1 By Type

| Type | Count | Answer F1 | Evidence Recall | Behavior Acc |
| --- | ---: | ---: | ---: | ---: |
| single_evidence | 27 | 0.622 | 1.000 | 1.000 |
| multi_evidence | 26 | 0.471 | 0.981 | 1.000 |
| conflict | 11 | 0.468 | 0.955 | 1.000 |
| temporal | 25 | 0.457 | 0.840 | 0.960 |
| unanswerable | 26 | 0.415 | 1.000 | 0.962 |
| misleading | 37 | 0.363 | 0.937 | 0.973 |

Residual behavior failures in this saved v1.1 run: V036 (unanswerable
hallucination), V048 (over-conservative abstention on a misleading premise),
and V084 (question text asked for 2024 data while gold evidence answered the
2023 additions available as of early 2024). Current code adds a deterministic
answerability guard for V036/V048, v1.1.1 clarifies V084, and v1.1.2 makes all
evidence spans machine-traceable;
the table above will be superseded after the next real DeepSeek rerun.

## Historical v1.0 Leaderboard

| Rank | Run | Model | Questions | Errors | Behavior Acc | Answer F1 | Evidence Recall | Conflict F1 | ECE | Avg Latency | Commit |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | verabench_full_v3 | deepseek/deepseek-v4-flash | 152/152 | 0 | 0.763 | 0.281 | 0.799 | 0.006 | 0.416 | 61.7s | acb0b5f |
| 2 | verabench_full_v2 | deepseek/deepseek-v4-flash | 152/152 | 0 | 0.743 | 0.271 | 0.814 | 0.007 | 0.415 | 66.6s | acb0b5f |
| 3 | verabench_full | deepseek/deepseek-v4-flash | 152/152 | 0 | 0.526 | 0.157 | 0.811 | 0.007 | 0.062 | 79.2s | cc29ed2 |

## Focused Conflict Smoke

This v1.0 focused smoke is not a full leaderboard entry. It is a historical regression
check for conflict/misleading behavior using the offline-friendly
`configs/deepseek_rules_only.yaml` configuration.

| Run | Model | Questions | Errors | Behavior Acc | Answer F1 | Evidence Recall | Conflict F1 | Conflict TP / FP / FN | Result Path |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| conflict_misleading_rules_only_max25_v5 | deepseek/deepseek-v4-flash | 25/25 | 0 | 0.960 | 0.240 | 1.000 | 0.640 | 18 / 0 / 2 | `outputs/remote_results/verabench_conflict_misleading_rules_only_max25_v5.json` |
| conflict_misleading_rules_only_max25_v3 | deepseek/deepseek-v4-flash | 25/25 | 0 | 0.960 | 0.249 | 1.000 | 0.640 | 18 / 0 / 2 | `outputs/remote_results/verabench_conflict_misleading_rules_only_max25_v3.json` |
| conflict_misleading_rules_only_max25_v2 | deepseek/deepseek-v4-flash | 25/25 | 0 | 0.960 | 0.301 | 1.000 | 0.640 | 18 / 0 / 2 | `outputs/remote_results/verabench_conflict_misleading_rules_only_max25_v2.json` |
| conflict_misleading_rules_only_max10_v16 | deepseek/deepseek-v4-flash | 10/10 | 0 | 1.000 | 0.344 | 1.000 | 1.000 | 12 / 0 / 0 | `outputs/remote_results/verabench_conflict_misleading_rules_only_max10_v16.json` |

For `conflict_misleading_rules_only_max25_v5`, `premise_refutation_summary`
reports TP/FP/FN `17/0/0`, precision `1.0000`, and recall `1.0000`. The two
remaining conflict false negatives (`V067`, `V068`) motivated the v1.1
ontology correction: these are premise refutations rather than
evidence-evidence contradictions.

## Best Run By Type: verabench_full_v3 (deepseek/deepseek-v4-flash)

| Type | Count | Answer F1 | Evidence Recall | Behavior Acc |
| --- | ---: | ---: | ---: | ---: |
| conflict | 25 | 0.256 | 0.720 | 0.480 |
| misleading | 25 | 0.181 | 0.867 | 0.760 |
| multi_evidence | 25 | 0.262 | 0.713 | 0.600 |
| single_evidence | 26 | 0.331 | 1.000 | 1.000 |
| temporal | 25 | 0.330 | 0.640 | 0.760 |
| unanswerable | 26 | 0.324 | 0.846 | 0.962 |

## Reproducibility Metadata

| Run | Provider | Model | Config | Timestamp | Result Path |
| --- | --- | --- | --- | --- | --- |
| verabench_full_v3 | deepseek | deepseek-v4-flash | configs/deepseek_run.yaml | 2026-06-01T15:44:53+0800 | `results/verabench_full_v3.json` |
| verabench_full_v2 | deepseek | deepseek-v4-flash | configs/deepseek_run.yaml | 2026-06-01T12:03:39+0800 | `results/verabench_full_v2.json` |
| verabench_full | deepseek | deepseek-v4-flash | configs/deepseek_run.yaml | 2026-06-01T00:24:58+0800 | `results/verabench_full.json` |
