# Evaluation Guide

This guide explains how to reproduce and interpret VeraBench results.
Dataset scope, construction, provenance semantics, and limitations are
documented in [VERABENCH_CARD.md](VERABENCH_CARD.md).

## Validate Benchmark Data

VeraBench v1.1 separates evidence-evidence conflicts from evidence that jointly
refutes a misleading user premise. VeraBench v1.1.1 clarifies V084's time
scope, and v1.1.2 makes all evidence spans exactly or segment-wise traceable to
their corpus documents. Validate schema, ontology constraints, evidence-span
traceability, dependency groups, repository/package synchronization, and
SHA-256 fingerprints before a run:

```bash
python experiments/validate_verabench.py
# or: verarag-validate-benchmark
```

The ontology migration is reproducible and idempotent:

```bash
python scripts/migrate_verabench_ontology_v11.py --check
python scripts/migrate_verabench_span_traceability_v112.py --check
```

## Local Contamination Audit

Before publishing benchmark claims, check VeraBench text overlap against any
available local reference corpora that could have entered model training,
prompt development, or public benchmark leakage:

```bash
verarag-audit-contamination \
  --reference /path/to/local/training-or-public-corpus \
  --containment-threshold 0.85 \
  --output outputs/verabench_contamination_audit.json
```

The audit supports `.txt`, `.md`, `.json`, and `.jsonl` files or directories.
It reports exact normalized substring matches, character n-gram Jaccard near
duplicates, and item-containment n-gram matches that catch short benchmark
texts embedded in long reference files. The audit covers questions, gold
answers, evidence spans, document titles, and document contents. Each returned
match includes benchmark and reference excerpts; exact matches also include
reference character offsets so reviewers can inspect the local source file.
Use `--fail-on-high-risk-match` in release gates when a provided reference
corpus must not contain question or gold-answer text. A clean report only
proves no overlap with the supplied files; it is not proof that unknown
model-training corpora are uncontaminated.

## Demo Evaluation

Demo mode scores every supervised gold field against itself. It validates
loader, grouping, answer/evidence/conflict metrics, calibration, and reporting
plumbing; it is not a model quality result. A valid full demo reports 1.0 for
answer, evidence, conflict, behavior, and confidence metrics, with 0.0 ECE and
Brier score.

```bash
python experiments/run_verabench.py --demo
```

The CLI requires exactly one of `--demo` or `--config`. A requested real run
fails non-zero if its configuration or pipeline cannot load; it never falls
back to demo mode.

## Real Pipeline Evaluation

Use a fixed config and save output JSON:

```bash
DEEPSEEK_API_KEY=<key> python experiments/run_verabench.py \
  --config configs/verabench_v112_canonical.yaml \
  --output results/verabench_full.json \
  --restart
```

The runner writes an incremental checkpoint by default. If interrupted, rerun
the same command without `--restart` to resume. Checkpoint reuse is rejected
when benchmark, config, implementation, question filter, or maximum-question
signatures differ.

## Offline Rescoring

Reports record metric implementations in `metadata.metric_versions`. When an
answer metric changes, recompute answer F1, correctness, calibration, and group
aggregates without rerunning retrieval or calling an LLM:

```bash
python experiments/rescore_verabench.py results/verabench_full.json \
  --output results/verabench_full_rescored.json
# or: verarag-rescore ...
```

Rescoring is fail-closed: the source report must contain a benchmark version
and corpus/question fingerprints matching the selected benchmark data. Use
`--allow-benchmark-mismatch` only for explicitly labeled historical
sensitivity analysis, and `--allow-unverified` only for legacy reports whose
source benchmark metadata is absent. The output preserves the source benchmark
identity and records the target under `metadata.rescored_against_benchmark`;
an override never converts a historical run into a current-benchmark result.

`soft-f1-v2` handles concise Chinese fact answers with normalized containment
and uses numeric/English tokens plus Chinese bigrams for longer paraphrases.
Do not compare reports across benchmark or metric versions without explicitly
labeling the comparison.

## Statistical Inference

Every newly generated or offline-rescored VeraBench report contains two
complementary uncertainty analyses:

- `confidence_intervals`: deterministic question-type-stratified percentile
  bootstrap with 2,000 resamples, 95% confidence, and seed 1729;
- `dependency_robust_confidence_intervals`: bootstrap over the 27 connected
  components induced by shared gold document IDs.

The first preserves the six-type composition. The second treats questions that
reuse connected evidence documents as dependent observations. It is a
sensitivity analysis, not a guarantee of wider intervals.

## Calibration Diagnostics

VeraBench reports include `ece`, `brier_score`, and `calibration_bins` computed
from per-question confidence values and the boolean `correct` field. That field
tracks behavior correctness, not answer-token overlap. To render the same view
offline:

```bash
verarag-calibration \
  --input results/verabench_full.json \
  --correctness-field correct \
  --output results/calibration_curve.svg \
  --json-output results/calibration_curve.json
```

Use `--correctness-field premise_refutation_correct` for a focused diagnostic
on premise-refutation rows. The calibration CLI is fail-closed: it rejects empty
reports, missing correctness fields, non-boolean correctness values, non-finite
or out-of-range confidence scores, and invalid bin counts.

For ROADMAP stage-2 calibration diagnosis, use the offline analyzer. It reports
confidence distribution, correct-vs-incorrect separation, confidence AUROC,
diagnostic flags such as `underconfident`, `near_constant_confidence`, and
`weak_correctness_discrimination`, plus risk-coverage/AURC values:

```bash
verarag-analyze \
  outputs/remote_results/verabench_v11_full_merged_rescored.json \
  --risk-coverage-svg results/risk_coverage.svg \
  --risk-coverage-csv results/risk_coverage.csv \
  --json
```

The historical v1.1 full report currently shows mean confidence `0.352` versus
behavior correctness `0.914`, correct/incorrect mean confidence `0.354/0.330`,
confidence AUROC `0.555`, and flags `underconfident` plus
`weak_correctness_discrimination`. That narrows the stage-2 bug from generic
"bad ECE" to two concrete symptoms: confidence is too low overall and barely
ranks correct rows above incorrect rows.

The analyzer can also write the full selective-prediction curve. The SVG plots
risk against coverage after sorting rows by confidence; the CSV contains one
row per retained prefix with `kept`, `coverage`, `accuracy`, `risk`, and
`threshold`. The JSON summary reports AURC plus coverage@accuracy targets. On
the historical v1.1 report after behavior-grouped Platt calibration, AURC is
`0.0328`; retaining only rows above the threshold needed for at least 95%
accuracy keeps coverage `0.5724`, while 90% accuracy is available at full
coverage. This is useful evidence that selective prediction is measurable, but
the public claim still belongs to the canonical v1.1.2 run.

Runtime confidence now uses a behavior-level fused signal rather than only
`1 - overall_uncertainty`: claim verification status/confidence, evidence
quality and claim coverage, answer-claim/reasoning confidence, unresolved
conflict pressure, and abstention justification are combined before a bounded
uncertainty penalty is applied. This fixes the code-level root cause where a
five-dimensional uncertainty aggregate plus multiplicative calibration could
compress otherwise different outcomes into a narrow low-confidence band. The
change still requires a canonical v1.1.2 full run before claiming the stage-2
DoD targets for ECE, confidence AUROC, and risk-coverage.

For held-out post-hoc calibration, write a calibrated copy of a saved report:

```bash
verarag-calibrate-report \
  --input outputs/remote_results/verabench_v11_full_merged_rescored.json \
  --output outputs/remote_results/verabench_v11_full_merged_rescored_calibrated_platt.json \
  --summary-output outputs/remote_results/verabench_v11_full_merged_rescored_calibration_summary.json \
  --method platt \
  --json
```

The command also supports `--method temperature`, but Platt scaling is the
default because historical VeraBench confidence is mostly below 0.5 despite
high behavior accuracy. The split is deterministic, stratified by correctness,
and controlled by `--seed` plus `--calibration-fraction`. The calibrated report
preserves each row's original confidence under
`diagnostics.confidence_calibration.original_confidence` and writes the method,
split, seed, and before/after all/calibration/holdout ECE and Brier metrics to
`metadata.posthoc_confidence_calibration`. On the historical v1.1 full report,
the held-out Platt calibration run with seed 1729 changes ECE from `0.5523` to
`0.0110` and Brier from `0.3929` to `0.0836`; this is a historical diagnostic,
not the canonical v1.1.2 result.

Because VeraRAG confidence attaches to different behavior types, the same CLI
can calibrate each behavior family separately:

```bash
verarag-calibrate-report \
  --input outputs/remote_results/verabench_v11_full_merged_rescored.json \
  --output outputs/remote_results/verabench_v11_full_merged_rescored_calibrated_group_platt.json \
  --summary-output outputs/remote_results/verabench_v11_full_merged_rescored_group_calibration_summary.json \
  --method platt \
  --group-field actual_behavior \
  --min-group-rows 8 \
  --json
```

Rows are still split once by correctness, then grouped by `actual_behavior`.
Each group fits its own Platt/temperature model only when the calibration split
has enough rows and both correctness classes. Sparse or single-class groups fall
back to a smoothed empirical constant by default, with the reason recorded in
both `metadata.posthoc_confidence_calibration.groups` and
`diagnostics.confidence_calibration`. On the historical v1.1 report this
behavior-aware run changes holdout ECE from `0.5523` to `0.0666` and Brier from
`0.3929` to `0.0840`; `answer_with_citation` receives a group Platt model,
while `abstain`, `answer_with_conflict_note`, and `correct_premise` use
documented fallbacks because their calibration splits are sparse or single
class. These numbers are diagnostic only; the canonical v1.1.2 run must be
calibrated and validated before updating the public results table.

Intervals are reported for answer F1, evidence recall/precision,
citation/supporting-fact precision/recall/F1 when those fields are present,
behavior and correctness accuracy, conflict micro-F1, premise-refutation F1,
ECE, and Brier score when applicable. Each report includes
`effective_resamples_by_metric`; replicates where a sparse conditional metric
has no defined denominator are excluded for that metric instead of receiving
an imputed score. These intervals quantify sampling uncertainty on the fixed
VeraBench questions; they do not establish external validity on other domains.

For two systems evaluated on the same question IDs, use paired inference:

```bash
python experiments/compare_verabench_reports.py \
  results/baseline.json \
  results/candidate.json \
  --resamples 5000 \
  --output results/paired_comparison.md
# or: verarag-compare-reports ...
```

The command validates benchmark fingerprints, metric versions, complete
question coverage, and per-question alignment. It reports candidate-minus-
baseline delta intervals, bootstrap probability of improvement, per-question
wins/ties/losses, a two-sided exact McNemar test for behavior correctness, and
paired shared-evidence-cluster sensitivity intervals when dependency metadata
is available. Citation and supporting-fact metrics are included automatically
for reports produced by current VeraBench evaluators; older reports without
those fields are not backfilled as zero-valued comparisons.

## Partitioned Evaluation

Large runs can be split into disjoint question-type partitions and merged
without losing provenance. The merge command rejects duplicate question IDs
and mismatched benchmark, implementation, config, model, or metric signatures,
then recomputes all aggregate metrics offline:

```bash
python experiments/merge_verabench_reports.py \
  results/part-a.json results/part-b.json \
  --require-complete \
  --output results/verabench-full.json
# or: verarag-merge-reports ...
```

## Targeted Diagnostics

For a small set of known failures or regression examples, run by exact
question id. Question-id filters are recorded in the checkpoint signature, so
checkpoints from different id subsets cannot be silently reused:

```bash
DEEPSEEK_API_KEY=<key> python experiments/run_verabench.py \
  --config configs/verabench_v112_canonical.yaml \
  --ids V036 V048 V084 \
  --output results/verabench_targeted_failures.json \
  --restart
```

For conflict work:

```bash
DEEPSEEK_API_KEY=<key> python experiments/run_verabench.py \
  --config configs/verabench_v112_canonical.yaml \
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
- `conflict_summary.f1`: global micro-F1 over scored gold-evidence conflict
  pairs. Predicted conflicts involving retrieved non-gold evidence, or
  unannotated same-evidence premise-refutation pairs, are preserved under
  `question_results[].diagnostics.unscored_extraneous_conflict_pairs` but do
  not count as false positives.
- `conflict_summary.dominant_failure`: `over_detection`, `under_detection`, `mixed`, or `none`.
- `premise_refutation_summary.expected`: questions that ask the system to validate
  or reject a user premise or overgeneralization.
- `premise_refutation_summary.detected`: answers that explicitly reject or narrow
  the premise with wording such as "不意味着", "不代表", "不能说明",
  "不能简单地认为", or "不足以".
- `premise_refutation_summary.false_negatives`: cases where the answer should
  have rejected the premise but did not. This is intentionally separate from
  `conflict_summary`: two evidence items can jointly refute a user's premise
  without contradicting each other.
- `question_results[].diagnostics.evidence_ids`: retrieved evidence ids used by the pipeline.
- `question_results[].diagnostics.predicted_conflict_pairs`: conflict pairs after evaluator mapping.
- `question_results[].diagnostics.gold_conflict_pairs`: gold pairs for the question.
- `question_results[].diagnostics.conflict_edges`: up to 20 scored conflict edges with source/target claim ids, evidence ids, mapped pair, type, and rationale.
- `question_results[].diagnostics.premise_refutation_expected`: whether the
  evaluator classifies the question as requiring premise refutation.
- `question_results[].diagnostics.premise_refutation_detected`: whether the
  generated answer explicitly performed that refutation.
- `question_results[].diagnostics.citation_ids`: bracketed evidence IDs parsed
  from the generated answer, such as `E1` or pipeline chunk IDs like
  `D001_c0`.
- `question_results[].diagnostics.mapped_citation_ids`: answer citations after
  mapping pipeline chunk IDs back to question-local VeraBench evidence IDs.
- `question_results[].diagnostics.required_citation_ids`: gold evidence IDs that
  should be cited for the question.
- `question_results[].diagnostics.predicted_supporting_fact_ids`: claim-level
  supporting evidence IDs emitted by the pipeline after the same gold-ID
  mapping.
- `question_results[].diagnostics.gold_supporting_fact_ids`: gold supporting
  evidence IDs derived from VeraBench ground-truth claims.
- `question_results[].diagnostics.output_metadata.num_conflicts`: conflict count seen by the answer pipeline before scoring.

For example, if `conflict_summary` shows a false negative, inspect that
question's `diagnostics.evidence_ids` first. If the gold document is missing,
the failure is retrieval. If the gold document is present but
`conflict_report_edges` is zero, inspect claim extraction, evidence metadata, or
fact-slot gates. If `conflict_edges` contains the right edge but the mapped pair
does not match gold, inspect evaluator mapping or VeraBench annotations.

## Citation and Supporting-Fact Metrics

VeraBench reports citation and supporting-fact quality alongside evidence
retrieval quality:

- `citation_summary`: answer-level citation precision, recall, and F1. The
  parser accepts bracketed benchmark IDs (`[E1]`) and pipeline chunk IDs
  (`[D001_c0]`), then maps chunk IDs through the question's gold documents
  before scoring.
- `supporting_fact_summary`: HotpotQA-style supporting evidence precision,
  recall, and F1 over claim-level `answer_claims[].supporting_evidence`.
- `question_results[].citation_*` and
  `question_results[].supporting_fact_*`: per-question scores used for macro
  averages and type/difficulty breakdowns.

Both summaries include macro averages plus micro true positives, false
positives, and false negatives. Use `citation_summary` when answers cite the
wrong source despite retrieving the right document; use
`supporting_fact_summary` when the reasoning/verifier claim links fail even
though the final prose contains plausible citations.

## Retrieval Diagnostics

Run document-level retrieval evaluation without an LLM:

```bash
python experiments/evaluate_retrieval.py \
  --retriever bm25 \
  --top-k 10 \
  --sweep-top-k 1 2 3 4 5 6 8 10 \
  --output outputs/retrieval_eval_bm25_top10.json
```

The report scores retrieved document ids against VeraBench gold evidence ids and
groups precision, recall, F1, hit rate, all-gold-retrieved rate, MRR, and nDCG
by question type, difficulty, and multi-hop flag. By default, questions without
gold evidence are skipped so unanswerable rows do not distort retrieval quality;
pass `--include-no-gold` when auditing those rows explicitly.

Run a compact offline ablation matrix across retrievers, retrieval depths, and
selection policies:

```bash
python experiments/evaluate_retrieval.py \
  --matrix \
  --matrix-retrievers bm25 bm25_rerank hybrid dense \
  --matrix-dense-models BAAI/bge-base-en-v1.5 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --matrix-top-k 3 5 10 \
  --matrix-policies fixed precision_cap complexity_adaptive \
  --output outputs/retrieval_matrix_v112.json
```

The single-report mode supports `--retriever bm25`, `--retriever dense`,
`--retriever hybrid`, and the CrossEncoder reranked variants
`bm25_rerank`, `dense_rerank`, and `hybrid_rerank`. Matrix mode defaults to
BM25 only so a fresh checkout does not accidentally load neural models; pass
`--matrix-retrievers` explicitly for full Stage-3 ablations. Dense, hybrid, and
reranker variants use locally cached `sentence-transformers` files by default;
pass `--dense-allow-download` or `--reranker-allow-download` only when an
intentional online model download is acceptable. `--matrix-dense-models` adds a
checkpoint axis for dense-backed variants only, so BM25 is not duplicated while
Dense/Hybrid can compare English, Chinese, and multilingual checkpoints in one
report. In matrix mode unavailable model-backed variants are recorded as
`status: error` and the remaining variants continue unless `--fail-fast` is
set.

Reranked variants first retrieve a larger candidate pool, controlled by
`--reranker-candidate-k` and defaulting to 20, then use
`BAAI/bge-reranker-base` to reorder the final top-k results. The default is
designed for reproducibility audits: if the CrossEncoder is not already cached,
the variant records an error rather than silently downloading weights.
If Hugging Face downloads fail with `Invalid port: ':1'`, temporarily remove
bare IPv6 loopback entries from proxy bypass variables for that command, for
example `NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost ...`.

Current full offline matrix over BM25, Dense (`BAAI/bge-base-en-v1.5`,
local-files-only), and Hybrid at top-k `3/5/10` shows BM25 still dominates this
Chinese benchmark. The best variant is BM25 top-3 with
`complexity_adaptive`, scoring macro precision `0.4365`, recall `0.9138`, and
F1 `0.5771`. Hybrid top-3 with `complexity_adaptive` scores
`0.3889/0.8039/0.5113`; Dense top-3 with `complexity_adaptive` scores
`0.3141/0.6576/0.4147`. These numbers make dense retrieval a model-selection
problem rather than an automatic win; Chinese/multilingual dense checkpoints
should be tested before replacing BM25 in the canonical run.

A focused downloaded-model top-3 `complexity_adaptive` run over BM25,
BM25+Reranker (`--reranker-candidate-k 5`), Dense, and Hybrid with
`BAAI/bge-base-en-v1.5` plus
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` finds a stronger
offline frontier: `bm25_rerank` scores macro precision `0.4456`, recall
`0.9320`, and F1 `0.5893`. Multilingual Hybrid scores
`0.4376/0.9150/0.5785`, slightly above plain BM25 top-3 adaptive, while
multilingual Dense scores `0.4240/0.8912/0.5612`. English BGE remains behind on
this Chinese benchmark: Hybrid `0.3889/0.8039/0.5113` and Dense
`0.3141/0.6576/0.4147`. This makes BM25+Reranker the current best offline
retrieval candidate, but it still needs full end-to-end behavior validation
before changing the canonical run.

A three-question end-to-end smoke A/B over `V001`, `V017`, and `V041`
validates the reranked candidate in the real DeepSeek pipeline before full-run
spend. Compared with canonical BM25 fixed-depth retrieval on the same questions,
`bm25_rerank` top-3 adaptive kept Evidence Recall at `1.0000`, improved
Evidence Precision from `0.1667` to `0.8333`, improved Answer F1 from `0.5382`
to `0.6282`, improved Conflict micro-F1 from `0.8000` to `1.0000`, preserved
Behavior Accuracy at `1.0000`, and reduced mean per-question latency from
`57.58s` to `11.66s`. The paired smoke report is generated under
`outputs/remote_results/verabench_v112_retrieval_rerank_top3_smoke3_comparison.md`
and is intentionally not a publication claim; it is the launch gate for the
full v1.1.2 A/B.

The full candidate run has also completed on all 152 VeraBench v1.1.2
questions with zero errors:
`outputs/remote_results/verabench_v112_retrieval_rerank_top3_deepseek.json`.
As a standalone candidate report, BM25+Reranker top-3 adaptive scores Evidence
Precision `0.4934`, Evidence Recall `0.8827`, Answer F1 `0.4052`, Conflict
micro-F1 `0.5641`, Behavior Accuracy `0.9934`, and mean latency `17.57s`.
By type, Behavior Accuracy is `1.0000` for single-evidence, multi-evidence,
conflict, temporal, and unanswerable rows, and `0.9730` for misleading rows.
This satisfies the Stage-3 precision/recall target for the candidate itself,
but it is not yet a paired A/B claim; run the canonical BM25 fixed-depth full
baseline and then `compare_verabench_reports.py` before changing the canonical
configuration.

Run the full candidate with:

```bash
python experiments/run_verabench.py \
  --config configs/verabench_v112_retrieval_rerank_top3.yaml \
  --output outputs/remote_results/verabench_v112_retrieval_rerank_top3_deepseek.json \
  --checkpoint outputs/remote_results/verabench_v112_retrieval_rerank_top3_deepseek.json.ckpt.jsonl \
  --restart
```

Reproduce the smoke A/B with:

```bash
python experiments/run_verabench.py \
  --config configs/verabench_v112_canonical.yaml \
  --ids V001 V017 V041 \
  --output outputs/remote_results/verabench_v112_canonical_smoke3.json \
  --checkpoint outputs/remote_results/verabench_v112_canonical_smoke3.ckpt.jsonl \
  --restart

python experiments/run_verabench.py \
  --config configs/verabench_v112_retrieval_rerank_top3.yaml \
  --ids V001 V017 V041 \
  --output outputs/remote_results/verabench_v112_retrieval_rerank_top3_smoke3.json \
  --checkpoint outputs/remote_results/verabench_v112_retrieval_rerank_top3_smoke3.ckpt.jsonl \
  --restart

python experiments/compare_verabench_reports.py \
  outputs/remote_results/verabench_v112_canonical_smoke3.json \
  outputs/remote_results/verabench_v112_retrieval_rerank_top3_smoke3.json \
  --baseline-label bm25:fixed:smoke3 \
  --candidate-label bm25_rerank:adaptive_top3:smoke3 \
  --output outputs/remote_results/verabench_v112_retrieval_rerank_top3_smoke3_comparison.md
```

The focused run command was:

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python experiments/evaluate_retrieval.py \
  --matrix \
  --matrix-retrievers bm25 bm25_rerank dense hybrid \
  --matrix-dense-models BAAI/bge-base-en-v1.5 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --matrix-top-k 3 \
  --matrix-policies complexity_adaptive \
  --dense-allow-download \
  --reranker-allow-download \
  --reranker-candidate-k 5 \
  --output outputs/retrieval_matrix_v112_top3_dense_rerank.json
```

Current bundled VeraBench v1.1.2 BM25 top-10 baseline evaluates 147 rows and
scores macro precision `0.1293`, macro recall `0.9830`, macro F1 `0.2244`,
hit rate `1.0000`, all-gold-retrieved rate `0.9660`, MRR `0.9427`, and nDCG
`0.9325`. This makes the Stage-3 failure mode concrete: recall is already high,
while precision is low enough that reranking, dynamic top-k, and evidence
deduplication should be measured before any end-to-end LLM spend.

Two offline selection policies are available for that precision work:

```bash
python experiments/evaluate_retrieval.py \
  --retriever bm25 \
  --top-k 10 \
  --top-k-policy precision_cap

python experiments/evaluate_retrieval.py \
  --retriever bm25 \
  --top-k 10 \
  --top-k-policy complexity_adaptive
```

On current v1.1.2 data, `precision_cap` caps the retained set at four documents
and scores macro precision `0.3044`, recall `0.9546`, and F1 `0.4492`.
`complexity_adaptive` keeps two documents for simple rows, four for
temporal/misleading rows, and five for multi-hop/conflict rows; it scores macro
precision `0.3500`, recall `0.9456`, and F1 `0.4977`. These are offline
retrieval-selection diagnostics, not end-to-end behavior claims.

The same reranked retrieval-depth and retained-evidence policy can be enabled
in the pipeline through the `retriever` config section:

```yaml
retriever:
  type: bm25_rerank
  retrieval_top_k: 3
  top_k_policy: complexity_adaptive
  adaptive_simple_top_k: 2
  adaptive_medium_top_k: 4
  adaptive_complex_top_k: 5
  reranker_model_name: BAAI/bge-reranker-base
  reranker_candidate_k: 5
  reranker_batch_size: 16
  reranker_local_files_only: false
```

Pipeline defaults and the canonical v1.1.2 run keep `top_k_policy: fixed` and
the default retrieval depth of 10 until an end-to-end ablation proves that the
reranked smaller evidence set improves Evidence Precision without causing
over-abstention or behavior regressions. For that A/B, use
`configs/verabench_v112_retrieval_rerank_top3.yaml`; it mirrors the canonical
v1.1.2 DeepSeek run except for `retriever.type`,
`retriever.retrieval_top_k`, `retriever.top_k_policy`, adaptive policy limits,
reranker settings, and its run/output identity.
The older `configs/verabench_v112_retrieval_adaptive.yaml` remains available as
a policy-only depth-10 sensitivity check, and
`configs/verabench_v112_retrieval_adaptive_top3.yaml` remains available as a
non-reranked top-3 adaptive sensitivity check.

Generate the fixed-vs-adaptive run plan before launching remote jobs:

```bash
python experiments/plan_retrieval_ablation.py \
  --restart \
  --output outputs/remote_results/verabench_v112_retrieval_ablation_plan.json
```

The plan validates that benchmark version, LLM settings, and pipeline settings
match, then records the intentional `retriever.type`,
`retriever.retrieval_top_k`, and `retriever.top_k_policy` differences. It emits
the baseline run command, candidate run command, and paired comparison command.

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

Formal leaderboard generation rejects demo reports, incomplete or errored
runs, missing reproducibility metadata, and mixed benchmark fingerprints or
metric versions. Diagnostic or historical tables must opt in explicitly with
`--allow-demo`, `--allow-incomplete`, `--allow-unverified`, or
`--allow-mixed-benchmarks`, as applicable.

## Conflict Detector Training Data

Build pairwise conflict examples from VeraBench annotations:

```bash
python experiments/build_conflict_training_data.py \
  --output-dir outputs/conflict_pairs_v112_leakfree
```

The default builder includes:

- gold conflict pairs, including self-conflicts where a single evidence span
  contains both sides of a contradiction;
- conservative weak positives from supporting/conflicting pairs with mismatched
  dates or numeric values;
- topical cross-question hard negatives;
- deterministic split assignment over shared-evidence connected components.

Questions connected through any gold document remain in one split. Hard
negatives may only draw their second side from the same split. Dataset metadata
reports dependency-group and exact-text overlap, and the training CLI rejects
cross-split dependency-group, question, or exact-text leakage by default. The
manifest also records the VeraBench version, corpus/question fingerprints, and
SHA-256 for every split; trained-model metadata copies that manifest and hashes
the exact files consumed.

Validate the training script without GPU work:

```bash
python experiments/train_conflict_cross_encoder.py \
  --train outputs/conflict_pairs_v112_leakfree/train.jsonl \
  --val outputs/conflict_pairs_v112_leakfree/val.jsonl \
  --test outputs/conflict_pairs_v112_leakfree/test.jsonl \
  --dry-run
```

Full training writes `training_metadata.json` and `training_metrics.json` under
the output directory. The metrics file records the threshold-selection
objective, default-threshold metrics, and validation-selected precision,
recall, and F1 for validation/test splits. The default objective is validation
F1; use `--threshold-objective precision` with
`--min-threshold-precision` for research runs that treat learned-conflict false
positives as more damaging than missed learned edges. It also hashes
`val_predictions.jsonl` and `test_predictions.jsonl`, which contain row-level
labels, probabilities, predictions, and dependency-component identities. Pair
metrics include a dependency-component bootstrap interval; hard-negative links
are included when constructing those components.
Training oversamples positives in the loader by default; the raw validation and
test splits remain unchanged for evaluation. Use
`compare_conflict_detectors.py --split test` for a held-out gold-evidence A/B;
an all-question comparison is diagnostic, not evidence of generalization.

Full Windows GPU instructions are in `docs/GPU_TRAINING.md`.

## Conflict Detector A/B

Before a full LLM pipeline run, compare conflict graph variants on VeraBench
gold evidence only:

```bash
python experiments/compare_conflict_detectors.py \
  --learned-model-path outputs/conflict_cross_encoder_expanded_balanced_seed13 \
  --learned-threshold 0.5 \
  --output outputs/conflict_detector_ablation_seed13.json
```

Installed package equivalent:

```bash
verarag-compare-conflicts \
  --learned-model-path outputs/conflict_cross_encoder_expanded_balanced_seed13 \
  --learned-threshold 0.5
```

This check does not call an LLM or retrieval. It converts VeraBench gold
evidence into structured claims, runs `ConflictGraphBuilder`, and reports
precision/recall/F1 for rules-only and rules+learned variants.
The report also includes `diagnosis`: per-variant dominant failure
(`under_detection`, `over_detection`, `mixed`, or `none`), by-type TP/FP/FN,
top false-negative/false-positive questions, and learned-vs-rules deltas when a
learned model is supplied. Use that field to decide whether the next change
should increase recall, tighten precision, or reject a learned layer despite a
headline F1 gain.

On current bundled VeraBench v1.1.2 gold evidence, rules-only conflict graph
diagnosis is no longer the old F1≈0 failure mode. The first structured
diagnosis isolated missed self-pair conflicts in `V021`, `V075`, and `V122`,
plus a V017 extra pair. After adding high-precision same-evidence numeric
contrast handling, ITER first-plasma schedule self-refutation, and corrected
reported-claim cross-evidence deduplication, the all-scope gold-evidence check
scores precision `1.0000`, recall `1.0000`, F1 `1.0000`, with 15/0/0 TP/FP/FN;
the dependency-aware test split also scores 3/0/0. This proves the deterministic
gold-evidence edge layer is closed on current annotations. It does not yet prove
end-to-end conflict behavior on retrieved distractors or with LLM generation;
that remains part of the canonical v1.1.2 run and downstream ablations.

For an independently maintained VeraBench-compatible test set, record an
immutable evaluation id and file fingerprints:

```bash
python experiments/compare_conflict_detectors.py \
  --data-dir data/external/conflict_test_v1 \
  --independent-test \
  --evaluation-id conflict-test-v1 \
  --learned-model-path outputs/conflict_cross_encoder_seed13 \
  --learned-threshold 0.5 \
  --output outputs/conflict_detector_external_test.json
```

The promotion audit fails closed unless all training runs share one verified
dataset manifest, prediction artifacts match their hashes, multi-seed pair
metrics clear the configured thresholds, the external questions fingerprint
differs from the training benchmark, and rules+learned improves held-out F1
without recall loss or additional false positives:

```bash
verarag-audit-conflict-model \
  --runs outputs/conflict_cross_encoder_seed13 \
         outputs/conflict_cross_encoder_seed17 \
         outputs/conflict_cross_encoder_seed23 \
  --ablation outputs/conflict_detector_external_test.json \
  --output outputs/conflict_model_promotion_audit.json
```

`--allow-internal-heldout` is a development-only override. It does not turn an
internal VeraBench split into evidence of external validity.

## External Conflict-Set Annotation Audit

External conflict claims should include the benchmark files plus three
annotation artifacts:

- `manifest.json`: dataset id, source, license, and annotation protocol;
- `annotations.jsonl`: at least two independent annotator labels per question;
- `adjudications.jsonl`: one resolved gold label per question.

Generate a blind annotation packet from a VeraBench-compatible candidate set:

```bash
verarag-build-external-annotation-packet \
  --data-dir data/external/conflict_test_v1 \
  --output-dir outputs/conflict_test_v1_annotation_packet \
  --annotator ann_a \
  --annotator ann_b
```

The packet writes one JSONL template per annotator plus an adjudication
template and a `packet_manifest.json`. Templates intentionally omit gold
answers, expected conflicts, expected behaviors, question types, and evidence
categories so annotators do not see the benchmark labels while judging
conflict presence.

After annotators and the adjudicator fill the templates, compile the packet
into the audit schema:

```bash
verarag-compile-external-annotations \
  --packet-dir outputs/conflict_test_v1_annotation_packet \
  --output-dir outputs/conflict_test_v1_compiled_annotations
```

The compiler writes a complete audit-ready directory containing
`corpus.jsonl`, `questions.jsonl`, `manifest.json`, `annotations.jsonl`,
`adjudications.jsonl`, and `compiled_manifest.json`.

Run the protocol audit before using an external set for model promotion:

```bash
verarag-validate-external-conflicts \
  --data-dir outputs/conflict_test_v1_compiled_annotations \
  --internal-data-dir data/verabench \
  --min-questions 50 \
  --min-annotators-per-question 2 \
  --min-conflict-kappa 0.6 \
  --output outputs/external_conflict_test_v1_audit.json
```

The repository includes `data/external/conflict_mini_v1` as a CI fixture. It
validates the schema, fingerprints, annotator coverage, binary conflict
Cohen's kappa, conflict-type agreement, and adjudication overrides. It is not
a model-promotion benchmark; formal promotion still requires a separately
maintained external set with enough conflict-bearing questions.

## Conflict Pipeline A/B

After the offline detector A/B looks healthy, run a paired VeraBench pipeline
comparison with the same base config. First generate the plan without LLM calls:

```bash
python experiments/run_conflict_ablation.py \
  --config configs/verabench_v112_canonical.yaml \
  --learned-model-path outputs/conflict_cross_encoder_expanded_balanced_seed13 \
  --learned-threshold 0.5 \
  --types conflict misleading \
  --max 10 \
  --plan-only
```

When the API key is available, remove `--plan-only`:

```bash
DEEPSEEK_API_KEY=<key> python experiments/run_conflict_ablation.py \
  --config configs/verabench_v112_canonical.yaml \
  --learned-model-path outputs/conflict_cross_encoder_expanded_balanced_seed13 \
  --learned-threshold 0.5 \
  --types conflict misleading \
  --max 10 \
  --restart
```

Installed package equivalent:

```bash
verarag-conflict-ablation \
  --config configs/verabench_v112_canonical.yaml \
  --learned-model-path outputs/conflict_cross_encoder_expanded_balanced_seed13 \
  --learned-threshold 0.5 \
  --types conflict misleading \
  --max 10 \
  --plan-only
```

The output directory contains generated configs, the two VeraBench reports, and
`summary.json` with `delta_rules_plus_learned_minus_rules` for answer,
evidence, behavior, calibration, and conflict metrics.

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
- benchmark corpus/questions SHA-256 fingerprints;
- answer metric version from `metadata.metric_versions`;
- whether the run is demo or real pipeline mode;
- command used to produce the result.
