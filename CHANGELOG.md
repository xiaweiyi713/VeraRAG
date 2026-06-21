# Changelog

All notable changes to VeraRAG are recorded here.

## Unreleased

### Evaluation integrity

- Require an explicit, mutually exclusive `--demo` or `--config` mode for the
  VeraBench CLI.
- Fail closed when a requested real pipeline configuration cannot be loaded;
  real experiments no longer fall back silently to demo mode.
- Make demo reports a complete gold-identity plumbing check, including gold
  evidence, conflicts, behaviors, claim counts, and calibrated confidence.
- Add CI assertions for VeraBench validation and all demo identity metrics.
- Make leaderboard generation reject demo, incomplete, unverified, or
  benchmark/metric-incompatible reports by default, with explicit diagnostic
  override flags for historical analysis.
- Add a published-results validator that checks `docs/RESULTS.md` generation
  commands, current VeraBench v1.1.2 fingerprints, historical/non-comparable
  labels, and formal leaderboard integrity documentation.
- Add deterministic question-type-stratified bootstrap confidence intervals to
  every VeraBench report and offline rescore.
- Add shared-evidence-cluster bootstrap sensitivity intervals based on the 27
  connected components induced by reused gold documents.
- Report effective bootstrap replicates per metric and exclude replicates where
  sparse conditional metrics have no defined denominator.
- Make offline rescoring reject missing or mismatched benchmark provenance by
  default; explicit overrides preserve both source and target identities for
  legacy or historical sensitivity analysis.
- Make the installed `verarag-validate-benchmark` command discover bundled
  VeraBench data when it is run outside a source checkout.
- Add `verarag-compare-reports` for strict paired report comparison, including
  delta intervals, probability of improvement, wins/ties/losses, and an exact
  McNemar behavior test, plus paired evidence-cluster sensitivity intervals.
- Include citation and supporting-fact precision/recall/F1 in paired report
  comparisons when both reports expose those current VeraBench fields.
- Include per-question `latency_seconds` in paired report comparisons as a
  lower-is-better metric, so retrieval and routing A/B reports capture cost
  changes alongside behavior and evidence quality.
- Fix ECE binning so confidence values exactly equal to 1.0 are included in the
  final calibration bin.
- Make VeraBench checkpoint resume validate saved rows against the current
  question text, type, ground truth, and expected behavior, skipping stale or
  out-of-scope rows so filtered or edited runs cannot reuse polluted results.
- Add a checkpoint repair helper for long VeraBench runs that backs up JSONL
  checkpoints and removes transient errored rows so resume reruns only failed
  questions, exposed as `verarag-repair-checkpoint`.
- Add `verarag-benchmark-status`, a read-only checkpoint/report progress
  summary CLI for long VeraBench runs.
- Normalize pipeline and loaded report confidence values into finite `[0, 1]`
  probabilities before calibration and aggregate reporting.
- Rebuild runtime pipeline confidence as a behavior-level fusion of verifier
  support, evidence quality/coverage, claim and reasoning confidence, conflict
  pressure, abstention justification, and bounded uncertainty pressure instead
  of relying on a single multiplicatively discounted uncertainty aggregate.
- Add VeraBench citation and supporting-fact scoring to per-question rows and
  report summaries, including pipeline chunk-ID to gold-evidence-ID mapping for
  answer citations such as `[D001_c0]`.
- Add an offline retrieval ablation matrix to `verarag-evaluate-retrieval`,
  covering retriever variants, retrieval depths, and top-k selection policies
  without running an LLM.
- Make offline dense/hybrid retrieval evaluation use locally cached
  SentenceTransformer files by default, with `--dense-allow-download` as an
  explicit opt-in for model downloads.
- Add configurable pipeline retrieval depth via `retriever.retrieval_top_k`,
  plus a VeraBench v1.1.2 BM25 top-3 complexity-adaptive candidate config and
  ablation plan metadata so the strongest offline retrieval matrix point can be
  tested end-to-end.
- Add CrossEncoder reranked retrieval variants (`bm25_rerank`,
  `dense_rerank`, `hybrid_rerank`) to offline VeraBench retrieval evaluation,
  using locally cached reranker models by default and recording unavailable
  variants as matrix errors.
- Add `--matrix-dense-models` so offline retrieval matrices can compare
  English, Chinese, and multilingual dense checkpoints in one report without
  duplicating BM25-only variants.
- Make offline retrieval evaluation fail closed when Hybrid degrades to BM25
  fallback, and record a downloaded-model top-3 adaptive result where
  `bm25_rerank` becomes the current offline retrieval frontier.
- Add precision/recall-aware threshold selection to Conflict CrossEncoder
  training and wire the Windows GPU launchers to pass those objectives for
  reproducible false-positive-sensitive experiments.
- Run a precision-first Windows GPU conflict-training matrix; it improved only
  seed 13 and was still rejected by the promotion audit, confirming that the
  learned detector needs stronger data or objectives before pipeline use.
- Add end-to-end pipeline support for reranked retriever types
  (`bm25_rerank`, `dense_rerank`, `hybrid_rerank`) and a VeraBench v1.1.2
  BM25+Reranker top-3 adaptive candidate config for full behavior A/B runs.
- Run the first real DeepSeek Stage-3 smoke A/B on `V001`, `V017`, and `V041`:
  the BM25+Reranker top-3 adaptive candidate preserved recall and behavior,
  improved evidence precision and conflict F1, and cut mean latency versus the
  canonical BM25 fixed-depth smoke.
- Complete the full 152-question DeepSeek Stage-3 candidate run for
  BM25+Reranker top-3 adaptive retrieval, with zero errors, Evidence Precision
  `0.4934`, Evidence Recall `0.8827`, Conflict F1 `0.5641`, Behavior Accuracy
  `0.9934`, and mean latency `17.57s`.
- Complete the canonical VeraBench v1.1.2 DeepSeek full run with 152/152
  questions, zero errors, Behavior Accuracy `0.9934`, Answer F1 `0.4031`,
  Evidence Precision/Recall `0.1244/0.9485`, Conflict F1 `0.5385`,
  stratified and evidence-cluster bootstrap intervals, and report-level
  benchmark/config/implementation signatures.
- Publish the full Stage-3 paired comparison: BM25+Reranker top-3 adaptive
  improves Evidence Precision by `+0.3690` and mean latency by `-150.90s`, but
  significantly reduces Evidence Recall (`-0.0658`) and Citation F1
  (`-0.0425`) while worsening Brier score, so the canonical default remains
  BM25 fixed-depth pending recall/citation safeguards.
- Add a recall-guarded reranking candidate: `reranker_preserve_base_top_k`
  preserves top base-retriever candidates as anchors after CrossEncoder
  reranking, exposed in pipeline configs and offline retrieval evaluation; the
  guarded BM25+Reranker top-3 adaptive offline run improves macro P/R/F1 to
  `0.4478/0.9342/0.5916`.
- Add a deterministic answer citation guard:
  `reasoning.enforce_answer_citations` appends missing in-pool
  `answer_claims[].supporting_evidence` IDs to the final answer so answer text
  citations and claim-level supporting facts use the same evidence chain.
- Record guarded rerank and citation-guard switches in VeraBench report
  metadata (`retriever_reranker_preserve_base_top_k` and
  `reasoning_enforce_answer_citations`) so Stage-3 reports remain auditable.
- Run the guarded+citation DeepSeek smoke on `V001`, `V017`, and `V041`:
  Evidence Precision improves from `0.1667` to `0.8333`, Citation F1 from
  `0.0000` to `1.0000`, Answer F1 from `0.5382` to `0.6914`, and
  Recall/Behavior/Supporting-Fact F1 stay at `1.0000` versus canonical smoke;
  the result clears the three-question guarded gate but remains too small for
  full-run promotion.
- Add question-aware NLI conflict pruning for law-status claims: ordinary
  status questions suppress NLI false positives between same-polarity
  "passed/approved" claims, while premise-validation questions retain
  cross-evidence disagreements. The guarded smoke now reaches Conflict F1
  `1.0000` with zero conflict FP/FN.
- Add a point-in-time value guard for exact current-value questions: when an
  answer drifts into early-version evidence despite no true conflict, the
  pipeline filters historical-version chunks and emits a single direct
  claim/citation for the current value.
- Add `run_verabench.py --ids-file` and `verarag-filter-report` so broader
  paired gates can reuse fixed question-id lists and slice completed full
  reports into statistically comparable subsets.
- Add `configs/verabench_v112_guarded_gate18_ids.txt`, an 18-question
  Stage-3 guarded gate with three rows per VeraBench type. Top-3 guarded
  BM25+Reranker improves Evidence Precision (`0.1760` to `0.5463`) and
  Citation F1 (`0.2111` to `0.6593`) versus canonical on this gate, but
  Evidence Recall (`0.9722` to `0.6944`) and Supporting-Fact F1
  (`0.7856` to `0.6278`) reject a full-run promotion.
- Add and gate-test `configs/verabench_v112_retrieval_rerank_expanded_guarded.yaml`.
  The naive expanded-depth variant is rejected because Behavior Accuracy falls
  to `0.9444`, Answer F1 falls to `0.3336`, and Evidence Recall does not
  improve versus top-3 guarded.
- Add targeted second-pass retrieval for under-covered medium/complex
  subquestions, disabled by default and enabled in
  `configs/verabench_v112_retrieval_rerank_targeted_guarded.yaml`. On gate18 it
  improves over top-3 guarded on Evidence Recall (`0.6944` to `0.7407`) but
  initially regresses Answer F1 to `0.3109`.
- Add optional answer-side claim-slot selection for the reasoning prompt. The
  targeted guarded config now compresses answer evidence slots while preserving
  the full evidence pool for verification/evaluation; gate18 mitigates the
  targeted Answer F1 regression to `0.3786`, keeps Behavior Accuracy at
  `1.0000`, and improves Evidence Precision to `0.5583`, but still remains
  below top-3 guarded Answer F1 (`0.4102`), so it is not promoted to a full run.
- Add narrow answer guards for simple numeric answers, abstention answers with
  unrelated conflict preambles, and premise-verification answers that repeat
  unreliable reports without correction. On gate18, the targeted guarded
  candidate now improves over top-3 guarded on Answer F1 (`0.4102` to
  `0.4362`), Evidence Recall (`0.6944` to `0.7407`), Evidence Precision
  (`0.5463` to `0.5583`), and correctness accuracy (`0.7222` to `0.8889`) while
  keeping Behavior Accuracy at `1.0000`; it still regresses Citation F1,
  Supporting-Fact F1, latency, and calibration, so full-run promotion remains
  blocked pending citation/support synchronization and recalibration.
- Add post-guard citation/support synchronization in the VeraRAG pipeline:
  after repair and deterministic answer guards, in-pool answer citations are
  reflected in `answer_claims[].supporting_evidence`, missing claim support IDs
  are appended back to the answer when citation enforcement is enabled, and
  out-of-pool support IDs are dropped before final confidence is estimated.

### Added

- VeraBench v1.1.2 ontology/data migration, structural validator, packaged-data
  synchronization check, and benchmark SHA-256 fingerprints.
- Evidence-span traceability migration and validation: all 208 references are
  now exact or ordered ellipsis-separated corpus substrings.
- Benchmark dependency audit with shared-document components, document reuse,
  component sizes, and near-duplicate question candidates.
- VeraBench dataset card covering construction, provenance semantics, intended
  uses, evaluation protocol, licensing, and known limitations.
- Evaluation run signatures and checkpoint sidecars that reject stale
  benchmark/config/implementation/question-filter resumptions.
- Versioned Chinese answer metric `soft-f1-v2` and offline report rescoring via
  `experiments/rescore_verabench.py` / `verarag-rescore`.
- Provenance-preserving partition report merging via
  `experiments/merge_verabench_reports.py` / `verarag-merge-reports`, with
  duplicate coverage and run-signature compatibility checks.
- Exact VeraBench question-id filtering via `run_verabench.py --ids` /
  `verarag-benchmark --ids`, useful for rerunning surgical regressions such as
  V036/V048/V084 without a full benchmark pass.
- Versioned behavior and conflict metrics: `behavior-v2` and
  `gold-evidence-pair-micro-f1-v2`, including offline rescoring of saved
  conflict diagnostics.
- Process-local optional model cache for semantic deduplication and NLI,
  including cached failures to prevent repeated network/model loads.
- VeraBench report diagnostics: behavior confusion, failure summaries, calibration bins, and conflict TP/FP/FN summaries.
- Offline result analysis CLI: `experiments/analyze_verabench_results.py`.
- VeraBench leaderboard generator: `experiments/build_verabench_leaderboard.py` / `verarag-leaderboard`.
- VeraBench conflict-pair dataset builder and CrossEncoder training CLI for GPU fine-tuning.
- Optional learned conflict detector layer in `ConflictGraphBuilder`, configurable via `conflict_graph.learned_model_path` or `VERARAG_CONFLICT_MODEL`.
- Learned conflict candidates are scored in a single batch per evidence graph
  and cached for the existing layered dispatcher.
- Conflict training now writes hash-verified row-level validation/test
  predictions and dependency-component bootstrap intervals.
- Add `verarag-audit-conflict-model`, a fail-closed multi-seed promotion policy
  that requires independent-test fingerprints and measurable rules+learned
  improvement before the learned detector can be considered for default use.
- Add `verarag-validate-external-conflicts`, an external conflict-set annotation
  audit covering dataset manifest, independent annotator coverage,
  adjudicated gold labels, SHA-256 fingerprints, Cohen's kappa, and
  adjudication overrides.
- Add `verarag-build-external-annotation-packet`, which creates blind JSONL
  templates for independent annotators and adjudicators while omitting gold
  answers, expected conflicts, question types, and evidence categories.
- Add `verarag-compile-external-annotations`, which converts completed blind
  annotation packets into audit-ready `annotations.jsonl` and
  `adjudications.jsonl` directories.
- Add `verarag-audit-contamination`, a local exact/near-duplicate overlap audit
  for checking VeraBench text against caller-supplied training, prompt
  development, or public dump corpora without overclaiming unknown-model
  contamination status. Near-duplicate auditing now combines Jaccard with
  item-containment n-gram recall so short benchmark texts embedded in long
  reference files are not diluted away by document length, and match reports
  now include local reference excerpts plus exact-match character offsets for
  reviewer inspection. The audit now has regression coverage for parameter
  validation, reference path failures, recursive directory reads, JSON/JSONL
  string-leaf extraction, invalid JSON fallback, match-basis classification,
  excerpt selection, and truncation accounting.
- Add `verarag-scan-secrets`, a standard-library high-confidence secret scanner
  wired into local and CI quality gates to catch accidental API-key, `.env.*`,
  AWS access-key, private-key header, or token commits without printing the full
  secret value. Local workstation audits can opt into ignored-file scanning with
  `--include-ignored` / `make security-local` for `.env.local` style files, and
  `--sarif` emits SARIF 2.1.0 for code-scanning integrations without exposing
  raw secrets. CI preserves the SARIF artifact even when the hard secret gate
  fails so maintainers still get redacted file/line diagnostics.
- Add `verarag-validate-package`, a release-package audit that checks built
  sdist/wheel archives for required VeraBench data, docs, Web assets, typed
  package markers, console script names and targets, and forbidden
  cache/planning directories. The audit now selects the exact sdist/wheel for
  the current `pyproject.toml` project name and version so stale `dist/`
  artifacts cannot satisfy release checks, and verifies every console script
  target module is present in both sdist and wheel so installed commands do not
  fail from missing packaged modules. CI now reuses `make package-check` so
  archive validation and installed-wheel smoke testing stay aligned with the
  local pre-release gate.
- Add `verarag-validate-install`, an installed-wheel smoke check that expands
  the built wheel into a temporary import root outside the source checkout,
  verifies the public API, packaged VeraBench data, and Web app creation, and
  loads every declared console script with `--help`.
- Add `verarag-validate-docs` / `make docs-check`, a standard-library
  documentation gate for local Markdown file links, same-page/cross-page
  anchors, documented `make` targets, and public console-script references.
- Add `verarag-validate-results` / `make results-check`, a fast release gate
  for VeraBench result-page provenance and leaderboard publication contracts.
- Add `verarag-validate-version` / `make version-check`, a release identity
  gate that keeps `pyproject.toml`, source-checkout `src.__version__`,
  `CITATION.cff`, public package version fallback behavior, and release
  checklist instructions aligned.
- Add `verarag-validate-python` / `make python-support-check`, a support-matrix
  gate that keeps `requires-python`, PyPI classifiers, GitHub Actions Python
  matrix, Ruff/mypy targets, `environment.yml`, and public docs aligned.
- Add `verarag-doctor`, a local environment diagnostic CLI that reports Python,
  required modules, optional feature dependencies, VeraBench data files, and LLM
  provider environment readiness without printing secret values.
- Wire the environment doctor into `make doctor-check`, pre-commit, GitHub
  Actions, and `make release-check` so local workstation readiness cannot drift
  away from the published quality gate.
- Add `verarag-validate-configs` / `make configs-check`, a YAML configuration
  gate for default runtime and dataset configs that checks parseability, section
  shapes, threshold ranges, boolean/integer fields, and API-key environment
  placeholders.
- Harden `configs.load_config()` and `merge_configs()` against path traversal,
  non-YAML names, non-mapping YAML roots, and nested mutation side effects.
- Add `scripts/windows_gpu_status.sh` plus `make gpu-status`, a read-only
  Windows GPU operations helper that reports tmux sessions, attach commands,
  GPU utilization, disk space, and recent training/evaluation artifacts.
- Add `scripts/check_windows_conflict_training_ready.sh` plus `make gpu-check`,
  and make both conflict-training launchers run this preflight by default so
  SSH, tmux, conda, train dependencies, CUDA visibility, and offline base-model
  availability fail before a detached tmux job is started.
- Add `configs/verabench_v112_canonical.yaml`, the canonical VeraBench v1.1.2
  DeepSeek full-run configuration, and make the remote VeraBench launcher,
  RESULTS page, and validation contract point at the same reproducible run
  identity.
- Extend `verarag-analyze` with confidence diagnostics and risk-coverage
  summaries so calibration failures can be diagnosed offline before spending
  LLM budget.
- Add `verarag-analyze --risk-coverage-svg/--risk-coverage-csv`, producing
  publication-ready selective-prediction curves and reusable curve-point
  exports with AURC and coverage@accuracy summaries.
- Extend `verarag-compare-conflicts` with machine-readable detector diagnosis,
  including dominant failure mode, by-type TP/FP/FN, top missed/extra pairs,
  and learned-vs-rules deltas for precision-preserving promotion decisions.
- Close current v1.1.2 gold-evidence rules-only conflict edge gaps by handling
  same-evidence numeric contrasts, ITER schedule self-refutations, and corrected
  reported-claim cross-evidence deduplication.
- Add `verarag-evaluate-retrieval` / `experiments/evaluate_retrieval.py`, an
  offline VeraBench retrieval diagnostic that scores BM25/Hybrid document
  retrieval against gold evidence with grouped precision, recall, F1, hit rate,
  all-gold-retrieved rate, MRR, and nDCG.
- Extend `verarag-evaluate-retrieval` with top-k sweeps plus `precision_cap`
  and `complexity_adaptive` selection policies, making the retrieval
  precision/recall frontier reproducible before changing pipeline defaults.
- Add optional pipeline support for `retriever.top_k_policy`, allowing
  `precision_cap` and `complexity_adaptive` evidence selection to be tested
  end-to-end while canonical runs remain on the historical `fixed` policy.
- Record retriever type and top-k policy fields in VeraBench report metadata so
  future fixed-vs-adaptive runs can be audited without manually opening config
  files.
- Add `configs/verabench_v112_retrieval_adaptive.yaml`, a fixed-identity
  end-to-end A/B config for the Stage-3 retrieval top-k policy experiment.
- Add `verarag-plan-retrieval-ablation`, a no-key planning CLI that validates
  the fixed-vs-adaptive retrieval configs and emits the exact VeraBench run and
  paired comparison commands.
- Add `verarag-calibrate-report`, a held-out post-hoc confidence calibration
  CLI that fits Platt or temperature scaling, preserves original row
  confidences, and writes before/after calibration and holdout metrics into
  report metadata.
- Extend `verarag-calibrate-report` with behavior-grouped calibration via
  `--group-field actual_behavior`, including documented fallbacks for sparse or
  single-class behavior groups and per-row calibration diagnostics.
- Add `verarag-validate-examples` / `make examples-check`, a no-key quickstart
  gate that runs `examples/quickstart.py`, checks the README exposes the same
  first-run command, and is wired into release checks.
- Add `verarag-validate-deployment` / `make deployment-check`, a Docker/Web
  deployment config gate covering Dockerfile entrypoint, non-root runtime,
  healthcheck, `.dockerignore`, Makefile targets, README commands, and the Web
  `/api/status` route.
- Add `.pre-commit-config.yaml` plus `verarag-validate-precommit` /
  `make precommit-check`, using local hooks for Ruff, mypy, secret scanning,
  documentation, examples, deployment, dependency metadata, and project metadata
  gates while validating CI, README, and CONTRIBUTING do not drift away from the
  same hook surface.
- Add a pinned GitHub release workflow that runs `make release-check`, uploads
  wheel/sdist/SBOM/release-health/checksum artifacts, generates GitHub Artifact
  Attestations for the same release subjects, publishes to PyPI via OIDC trusted
  publishing only from `v*.*.*` tags, and creates GitHub Releases only for
  version tags.
- Add `verarag-validate-deps` / `make deps-check`, a dependency metadata gate
  that keeps core `pyproject.toml` dependencies mirrored in `requirements.txt`,
  validates release-tool dev dependencies, verifies `environment.yml` delegates
  to `requirements.txt`, and checks README install-path coverage.
- Add `verarag-validate-metadata` / `make metadata-check`, an open-source
  metadata gate that validates PyPI project URLs, governance files, GitHub
  issue/PR templates, README governance/citation links, `CITATION.cff`
  consistency, CONTRIBUTING quality-gate commands, Dependabot coverage, and
  GitHub Actions least-privilege/timeout settings, including release workflow
  trusted-publishing and provenance-attestation controls.
- Add CODEOWNERS and GitHub issue-template routing, and make the metadata and
  package gates verify maintainer ownership, disabled blank issues, and private
  security reporting links.
- Add a least-privilege CodeQL workflow for scheduled and PR Python security
  analysis, and make metadata/package gates reject missing or unbounded CodeQL
  scanning.
- Pin GitHub Actions workflow dependencies to full commit SHAs and make the
  metadata gate reject floating action tags such as `@v4`.
- Add an OpenSSF Scorecard workflow with published results and JSON artifacts,
  and make metadata/package gates require the Scorecard workflow.
- Add `verarag-generate-sbom` / `make sbom-check`, a standard-library
  CycloneDX 1.5 JSON SBOM generator and validator wired into release checks.
- Add `make release-check`, a one-command pre-release gate that chains lint,
  documentation validation, dependency metadata validation, project metadata
  validation, SBOM generation/validation, an 80% total coverage gate,
  VeraBench release-health validation, sdist/wheel build, and package content
  validation.
- Add `verarag-release-health`, a unified pre-release health check covering
  VeraBench v1.1.2 data traceability/sync, the external conflict fixture, blind
  annotation packet generation, demo metric plumbing, and demo paired
  comparison.
- `verarag-release-health` now writes and validates a
  `release-artifacts-manifest.json` with release-critical artifact paths,
  SHA-256 hashes, byte sizes, generation commands, and key metric summaries.
- Add an offline `verarag-release-health --validate-manifest` mode and
  `make release-artifacts-check` target for re-validating existing release
  artifact manifests without rerunning benchmark checks; manifest paths are
  constrained to the selected artifact root.
- Add `verarag-release-checksums` / `make release-checksums-check`, a release
  checksum manifest gate that records and validates SHA-256 hashes and byte
  sizes for the current sdist, wheel, CycloneDX SBOM, and release-health
  artifact manifest.
- VeraBench report export now records a `config_metadata_warning` when a
  pipeline config cannot be reopened for model/provider provenance instead of
  silently dropping those fields.
- Add `scripts/start_windows_verabench_eval.sh`, a detached Windows GPU
  VeraBench launcher that prompts for the DeepSeek key locally and injects it
  into the remote tmux job through an SSH/FIFO handoff instead of requiring
  inline shell-history-prone API-key commands.
- Harden Windows GPU helper scripts so remote project/model paths are shell
  quoted, `~` is resolved on the remote host before rsync, and single-seed
  training refuses to overwrite an existing tmux session.
- Add `scripts/start_windows_conflict_training_matrix.sh`, a detached
  Windows GPU multi-seed launcher that trains seeds 13/17/23, writes row-level
  artifacts, runs held-out detector A/B, and emits a report-only promotion
  audit for reproducible negative or positive evidence.
- Add `data/external/conflict_mini_v1` as a small CI fixture for the external
  conflict annotation protocol. It validates the workflow but is not a
  promotion benchmark.
- Conflict CrossEncoder training now reuses NLI encoder bodies while
  reinitializing the classifier head for VeraBench binary conflict scores.
- Conflict CrossEncoder training now writes validation/test metrics and a
  validation-selected threshold to `training_metrics.json`.
- Conflict CrossEncoder training now accepts `--seed` and fixes DataLoader
  shuffle/model initialization for reproducible GPU runs.
- Conflict CrossEncoder saving no longer waits on remote model-card metadata
  when the GPU host is offline, and the small-data baseline consistently uses
  10 warmup steps across the CLI, launcher, and documentation.
- Conflict CrossEncoder training now oversamples positives in the training
  loader by default, with `--no-balance-train` available for ablations.
- Conflict-pair data generation now includes gold self-conflicts, conservative
  weak positives, topical hard negatives, and `by_sample_source` metadata.
- Conflict-pair train/validation/test assignment now operates on connected
  components of shared gold documents, and hard negatives cannot cross split
  boundaries.
- Conflict training fails closed on dependency-group, question, exact-text, or
  declared-file split leakage; custom train paths derive the default test path
  from the same directory instead of silently reading an unrelated dataset.
- Conflict dataset and model metadata now chain benchmark fingerprints,
  per-split SHA-256 hashes, and the copied dataset manifest for end-to-end
  training provenance.
- Offline conflict detector A/B CLI:
  `experiments/compare_conflict_detectors.py` / `verarag-compare-conflicts`,
  including held-out `--split train|val|test` evaluation.
- Paired conflict pipeline A/B CLI:
  `experiments/run_conflict_ablation.py` / `verarag-conflict-ablation`.
- Calibration curve script compatibility with full VeraBench report JSON.
- Calibration curve CLI now validates confidence/correctness rows fail-closed,
  supports explicit boolean correctness fields, and can emit a JSON summary next
  to the SVG reliability diagram.
- Contribution, security, citation, issue, and PR governance files.
- Public `verarag` API package, typed package markers, API/architecture docs, and quickstart example.
- Embedded counter-claim extraction for reported false/misleading assertions
  such as "X claimed Y, but Y is false; actually Z".
- Question-focused conflict graph filtering in the main pipeline, including
  entity/year focus and reported-claim conflict deduplication.
- Deterministic reasoning fallback that explicitly acknowledges detected
  conflicts when an LLM answer otherwise omits the conflict note.
- Offline-friendly DeepSeek rules-only evaluation config
  (`configs/deepseek_rules_only.yaml`) that avoids Hugging Face model loading
  by disabling verifier, NLI, learned detector, and semantic dedup.
- Comparative numeric claim extraction for compact evidence such as "ECS is
  estimated at 2.5°C, lower than the IPCC AR6 estimate of 3.0°C".
- Deterministic answerability guard in the main pipeline for two high-risk RAG
  failures: exact-value questions answered from approximate evidence, and
  premise-validation questions turned into generic abstentions.

### Changed

- VeraBench v1.1 separates evidence-evidence conflicts from premise
  refutation, with 27/26/11/25/26/37 questions across the six types.
- VeraBench v1.1.1 clarifies V084's time scope so the question asks for 2023
  renewable-energy additions as reported by the early-2024 source, matching its
  gold evidence and answer.
- VeraBench v1.1.2 repairs 25 evidence-span annotations without changing
  questions, answers, evidence IDs, conflict labels, behavior labels, or metric
  implementations.
- Conflict filtering now conditions both sides of an edge on the question fact
  slot, distinguishes numeric semantic units, and suppresses expected
  historical version evolution for point-in-time questions.
- Verifier NLI now uses evidence as premise, claim as hypothesis, model
  `id2label`, and numerically stable three-class softmax.
- Conflict scoring now ignores non-conflict graph edges such as `support` and `partial_support`.
- Conflict graph detection now uses stricter fact-slot gates: shared entities
  alone are insufficient, date/month/period numbers are ignored, revenue facts
  are separated by year/quarter/annual period and business line, and same
  evidence-object claims are not compared by default.
- Chinese rule-based claim extraction now splits on Chinese punctuation and
  preserves number units so conflict rules can distinguish dates from values.
- Chinese entity extraction now trims predicate tails such as "欧盟AI法案已于2024"
  back to stable entity anchors such as "欧盟AI法案".
- Source-reliability, scope, and granularity conflict heuristics are now opt-in
  for the main dispatcher because they were too noisy in full RAG pipeline runs.
- Learned conflict detection now only scores pairs that pass fact-slot gating;
  the broad shared-entity-plus-number fallback was removed.
- The main pipeline now honors `retriever.type`; DeepSeek evaluation uses BM25
  explicitly, and hybrid retrieval falls back to BM25 if dense model loading
  fails in offline environments.
- Evidence semantic deduplication can now be disabled with
  `evidence.semantic_dedup: false`, allowing exact deduplication in offline
  or network-restricted runs.
- RetrievalAgent now preserves retriever metadata such as `date` and `author`
  when converting results to `Evidence`, allowing conflict deduplication to
  prefer newer official evidence.
- The pipeline now preserves the original user question as a retrieval anchor
  after LLM decomposition, reducing recall loss from query drift on conflict
  and misleading questions.
- Conflict fact-slot gating now distinguishes climate sensitivity estimates
  from policy temperature targets and total sales from export-only sales.
- Temporal conflict detection now binds years to their specific status
  attribute, so a 2025 effective date no longer conflicts with a 2024 passage
  date for the same law.
- Behavior classification now treats misleading-question premise corrections
  as `correct_premise` even when the answer also mentions evidence conflicts.
- Same-evidence claim comparison remains disabled by default, but explicit
  `reported_claim` versus `corrective_claim` pairs are compared to capture
  fact-check passages.
- Semantic contradiction detection now requires same fact-slot context, and
  contrastive mentions such as "unlike the EU AI Act..." no longer create
  conflicts for the contrasted entity.
- Reasoning conflict context now includes evidence ids, titles, and claim text
  instead of opaque claim ids.
- `pyproject.toml` development dependencies now match the Ruff-based Makefile workflow.
- Core package dependencies now include `cryptography` and
  `python-multipart`, which are required by API-key storage and the Web upload
  route.
- SQLite write transactions now always close their connections after commit or
  rollback, eliminating Web test resource leaks.
- Web corpus/upload indexing now normalizes benchmark document fields, avoids
  first-upload lock recursion, and attaches the shared mutable BM25 index to
  real SSE and WebSocket query pipelines.
- Web uploads now enforce raw-file and extracted-text size limits before
  indexing.
- IngestionPipeline now rejects empty indexes and invalid pre-parsed documents
  instead of silently creating empty-id or empty-text chunks, while preserving
  metadata from valid pre-parsed inputs.
- TextChunker now preserves short leading sections, emits contiguous chunk
  ids/indexes after small-fragment merges, and splits oversized Markdown
  sections, paragraphs, or single sentences instead of indexing unbounded
  chunks.
- DenseRetriever now consumes its config dictionary, validates empty documents
  and malformed embedding shapes, rejects query/index dimension mismatches, and
  makes unindexed saves explicit errors; FAISSRetriever reloads now rebuild the
  FAISS search structure instead of returning empty results after load.
- DecompositionPlanner now normalizes LLM, fallback, and uncertainty-refined
  subquestion plans so ids are contiguous, dependencies only point backward to
  valid subquestions, empty LLM plans fall back safely, and invalid
  `max_subquestions` values do not escape as runtime errors.
- DynamicRetrievalAgent now clamps direct retrieval and counter-evidence
  top-k values to at least one, keeps challenge, alternative, and temporal
  counter-evidence query paths inside the bounded query set, and writes
  refined low-coverage subquestions back into the active retrieval plan so
  later rounds use the improved query.
- ConfidenceCalibrator now validates temperature-scaling inputs, clips
  boundary probabilities before logit conversion, rejects non-finite labels or
  confidences, and makes calibration metric vector mismatches explicit instead
  of producing NaN/Inf values. UncertaintyController now validates threshold
  ordering and public probability/round inputs, and honors the legacy
  `high_conflict_threshold` config key as the high-decision threshold when
  `high_threshold` is absent.
- VerifierAgent now normalizes every verifier result before report aggregation,
  accepts enum/status-name/wire-value status inputs, clamps confidence into
  `[0, 1]`, treats unknown status or invalid confidence as conservative
  `NOT_ENOUGH_INFO`, and covers NLI unavailable/invalid/exception paths.
- LLMClient now normalizes provider names, preserves explicit zero generation
  settings, uses Ollama's prompt `generate` API, supports JSON mode for Ollama,
  and lets OpenAI-compatible providers such as DeepSeek override `base_url` for
  proxy or private gateway deployments.
- External annotation packet compilation now rejects unsafe annotator ids,
  absolute or escaping template paths, symlink escapes, malformed packet
  manifests, and no-conflict labels whose `conflict_type` is not explicitly
  `none`.
- ConflictGraphBuilder learned-detector score normalization now fails closed on
  invalid, NaN, infinite, empty, or extreme CrossEncoder outputs; LLM fallback
  adjudication now accepts only known relationships and finite confidence
  values instead of silently producing SUPPORT edges for malformed responses.
- ConflictGraphBuilder incremental graph updates now compare registered
  existing claims against newly added evidence, avoid duplicate undirected
  edges, detect new self-refuting claims, and batch-prime learned scores for
  the update window. NLI label mapping now fails closed on non-finite scores or
  ambiguous partial `id2label` mappings instead of trusting the wrong column.
- EvidenceScorer now treats conflict/support graph edges as undirected for
  evidence scoring and accepts both evidence-id and claim-id endpoints, so
  evidence quality is not biased by arbitrary edge direction. Hallucination
  metrics now normalize claim text for unsupported-claim matching and reject
  mismatched or non-finite overclaiming inputs instead of silently truncating
  evaluation rows.
- DocumentLoader now closes PDF handles even when page extraction fails, marks
  empty-PDF fallbacks with `source="file"`, and covers no-heading Markdown,
  preface sections, empty Markdown split sections, blank JSONL lines, fake
  PyMuPDF pages, empty PDFs, page errors, and missing PyMuPDF with regressions.
- AnswerMetrics now scores token F1 with multiset overlap instead of de-duping
  repeated tokens, prevents empty keyword features from giving unrelated
  one-character Chinese answers a perfect soft F1 score, and exposes
  `soft_f1_score` through `compute_all` and batch helpers.
- TaskAnalyzer now keeps simple rule-based questions at low complexity instead
  of treating the default task type as a multi-hop signal, escapes literal
  dollar-sign numeric cues correctly, falls back to rules when no LLM is
  configured, and normalizes LLM task-analysis aliases, booleans, hop counts,
  and keywords before they reach downstream planning.
- HallucinationMetrics now preserves signs when extracting percentages and
  numeric values, so decreases such as `-5%` are no longer treated as matching
  `+5%`; numerical hallucination scoring now rejects negative tolerance and
  non-finite answer or evidence numbers.
- BaseRetriever now centralizes query and `top_k` validation for batch
  retrieval, BM25, Dense, FAISS, and Hybrid retrievers; negative retrieval
  limits fail explicitly instead of producing Python slice artifacts.
- EvidenceExtractor now recognizes Chinese numerical, temporal, causal,
  comparative, definitional, and uncertainty cues when assigning claim types;
  LLM claim extraction now accepts object or list JSON payloads, enforces
  `max_claims`, skips malformed rows, and normalizes claim type, list, boolean,
  and support-type fields instead of letting one bad field discard the whole
  extraction.
- ConflictGraphBuilder numeric conflict detection now preserves parsed-number
  alignment with the original raw token, so malformed upstream number tokens do
  not shift units/context onto the wrong value; non-string number tokens are
  coerced safely before year/date/unit checks.
- ConflictGraphBuilder learned CrossEncoder score normalization now handles
  two-class `[negative, positive]` probability or logit outputs by using the
  positive conflict class instead of accidentally reading the negative class.
- ConflictGraphBuilder now normalizes probability-like threshold config values
  from strings, rejects booleans and non-finite values, and bounds out-of-range
  values before learned, NLI, support, and LLM-adjudication comparisons.
- The public `verarag` package now resolves `__version__` through installed
  package metadata and falls back to the single source-tree version for checkout
  usage, with regression coverage for both paths.
- VeraBench evaluator checkpoint resume now logs and skips stale,
  out-of-scope, or malformed checkpoint rows; pipeline and baseline exceptions
  stay isolated to the affected question; non-finite or out-of-range
  confidence values are clamped before calibration.
- RepairAgent now normalizes verifier statuses from enums, enum names, and
  wire values, clamps verifier confidence into `[0, 1]`, preserves
  `VerificationStatus` enum values on repaired claims, and keeps repair caveats
  idempotent.
- Exact numeric questions that retrieve only approximate/incomplete evidence now
  abstain with the available approximate value cited instead of presenting it as
  a precise answer.
- Assertion and premise-check questions that the LLM answers with a generic
  abstention are reframed as premise corrections when the answer lacks explicit
  correction language.
- CI now checks Python 3.10 through 3.13, the public `verarag` package, examples, package build, and full tests.
- Package build instructions and CI now use isolated PEP 517 builds so the
  declared `setuptools>=77` requirement is honored consistently.

### Verified

- Full local suite: `791 passed, 3 skipped`; Ruff, mypy, secret scan,
  version identity, Python support, documentation/results/example/dependency/project metadata validation,
  release-check/package audit, installed-wheel smoke, release checksum
  validation, sdist, and wheel build pass locally.
- Answerability guard regression tests cover the V036-style exact/approximate
  value gap and the V048-style premise-abstention correction; direct evaluator
  classification maps them to `abstain` and `correct_premise` respectively.
- Historical VeraBench v1.1 full DeepSeek run completed 152/152 with zero
  errors after partition merge and offline rescoring: Behavior Acc `0.9803`,
  Answer F1 `0.4593`, Evidence Recall `0.9518`, Conflict micro-F1 `0.8966`,
  and conflict TP/FP/FN `13/1/2`. A v1.1.2 rerun is required after the V084
  wording clarification and evidence-span traceability migration.
- DeepSeek full-config v1.1 single-evidence smoke completed 3/3 with zero
  errors, Evidence Recall `1.0000`, Behavior Acc `1.0000`, and zero predicted
  conflicts after question-conditioned conflict fixes.
- DeepSeek conflict pipeline smoke (`--types conflict --max 3`) now reaches
  Answer F1 `0.5223`, Evidence Recall `1.0000`, Conflict F1 `1.0000`, and
  Behavior Acc `1.0000`, with conflict TP/FP/FN `3/0/0`.
- DeepSeek rules-only conflict+misleading smoke (`--types conflict misleading
  --max 10`) now reaches Answer F1 `0.3437`, Evidence Recall `1.0000`,
  Conflict F1 `1.0000`, and Behavior Acc `1.0000`, with conflict TP/FP/FN
  `12/0/0`.
- Windows GPU multi-seed conflict-training matrix reproduced the v1.1.2
  negative result on 2026-06-15: seeds 13/17/23 reached held-out pair test F1
  `0.316/0/0.316`, rules and rules+learned detector A/B both reached F1
  `0.857`, and the report-only promotion audit rejected the learned model.
- The latest reproducible run is
  `outputs/remote_results/verabench_conflict_misleading_rules_only_max25_v5.json`.
  It verifies the real DynamicRetrievalAgent path after preserving evidence
  entity metadata and keeping `sq_original` at full top-10 retrieval depth.
- A larger DeepSeek max25 run exposed conflict over-detection at TP/FP/FN
  `13/17/7`; after the current structural fixes, the remote max25 v5 rerun
  reaches conflict TP/FP/FN `18/0/2`, Conflict F1 `0.6400`, Behavior Acc
  `0.9600`, and zero conflict false positives.
- `premise_refutation_summary` now tracks premise/overgeneralization correction
  separately from evidence-evidence conflicts. On max25 v5 it reaches TP/FP/FN
  `17/0/0`, precision `1.0000`, and recall `1.0000`.
- Earlier DeepSeek conflict smoke after fact-slot gating reduced rules-only
  over-detection from `108/3` predicted/gold conflicts to `0/3`, which exposed
  recall as the next bottleneck; learned detection remains experimental because
  it still produced `3/3` predicted conflicts with zero true positives in that
  smoke.
- Focused benchmark regression suite: `33 passed`; full conflict-focused suite
  remains covered by the full local test run.
- Full local test suite: `304 passed, 3 skipped` on Python 3.13.9.
- `make lint` passes Ruff, mypy, and secret scan; `python -m build` produces both sdist and wheel.
