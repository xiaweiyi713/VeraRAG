# GPU Training

This guide documents the reproducible Windows GPU workflow for training
VeraRAG components. The first supported target is a VeraBench conflict
CrossEncoder, because current full-run diagnostics show conflict detection as
the clearest quality bottleneck.

## Target

The training flow uses VeraBench evidence-pair annotations:

1. `experiments/build_conflict_training_data.py` converts benchmark
   `expected_conflicts` into pairwise JSONL.
2. The builder includes gold self-conflicts, conservative weak positives, and
   topical cross-question hard negatives.
3. `experiments/train_conflict_cross_encoder.py` trains a binary CrossEncoder
   classifier on those pairs.
4. The trained artifact is written under `outputs/conflict_cross_encoder/`,
   alongside `training_metadata.json`, `training_metrics.json`, and hashed
   validation/test row-level prediction files.
5. `training_metrics.json` reports validation/test precision, recall, F1, the
   threshold-selection objective, the selected threshold, and
   dependency-component bootstrap intervals. Validation F1 remains the default
   selection objective; precision-first selection is available for supplemental
   learned layers where false positives are more damaging than missed learned
   edges.
6. Positive examples are oversampled in the training loader by default so hard
   negatives do not dominate; pass `--no-balance-train` to disable this.
7. Set `VERARAG_CONFLICT_MODEL=outputs/conflict_cross_encoder` or configure
   `conflict_graph.learned_model_path` to enable the learned layer.

Default commands are safe to run locally in dry-run mode:

```bash
python experiments/build_conflict_training_data.py --output-dir outputs/conflict_pairs
python experiments/train_conflict_cross_encoder.py \
  --train outputs/conflict_pairs/train.jsonl \
  --val outputs/conflict_pairs/val.jsonl \
  --test outputs/conflict_pairs/test.jsonl \
  --dry-run
```

## Current v1.1.2 Leakage-Resistant Baseline

VeraBench v1.1.2 training data uses
`shared-evidence-component-stratified-v1`. Questions connected through any gold
document stay in one split, hard negatives may only draw from that same split,
and training fails closed on dependency-group, question, or exact-text overlap.
The current dataset contains:

- 181 pairs: 29 positive and 152 negative;
- train/validation/test: 129/26/26 pairs;
- positive counts: 23/3/3;
- dependency-group overlap: 0;
- exact-text cross-split overlap: 0.

`metadata.json` records the VeraBench version and corpus/question SHA-256 plus
the exact train/validation/test file hashes. Every trained model copies this
manifest into `training_metadata.json` and records the files it consumed.

Generation and verification:

```bash
python experiments/build_conflict_training_data.py \
  --output-dir outputs/conflict_pairs_v112_leakfree
python experiments/train_conflict_cross_encoder.py \
  --train outputs/conflict_pairs_v112_leakfree/train.jsonl \
  --val outputs/conflict_pairs_v112_leakfree/val.jsonl \
  --test outputs/conflict_pairs_v112_leakfree/test.jsonl \
  --warmup-steps 10 \
  --dry-run
```

Balanced 3-epoch GPU runs with seeds 13, 17, and 23 show that the current
learned detector is not release quality. Thresholds are selected on validation
F1 only:

| Seed | Selected threshold | Validation F1 | Test precision | Test recall | Test F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 13 | 0.336 | 0.667 | 0.188 | 1.000 | 0.316 |
| 17 | 0.604 | 0.333 | 0.000 | 0.000 | 0.000 |
| 23 | 0.302 | 0.400 | 0.188 | 1.000 | 0.316 |

The three-seed test F1 mean is 0.211 with range [0.000, 0.316]. Disabling
positive oversampling at seed 13 also produces test F1 0.000. This instability
is reported as a negative result; the learned layer remains disabled by
default. The `scripts/start_windows_conflict_training_matrix.sh` workflow
reproduced these numbers on `windows-gpu` on 2026-06-15 using the local
`~/models/verarag/cross-encoder_nli-distilroberta-base` checkpoint and RTX 4060
Laptop GPU.

The row-level v2 artifacts expose only three dependency components in the test
split. Dependency-component bootstrap 95% F1 intervals are `[0.000, 0.353]`
for seeds 13 and 23 and `[0.000, 0.000]` for seed 17. The promotion audit
therefore rejects the model on multi-seed F1, robust lower bound, independent
test provenance, minimum A/B sample size, and lack of held-out improvement.
The matrix workflow's internal held-out report-only audit rejected on
`multi_seed_test_f1`, `dependency_robust_f1_lower_bound`,
`minimum_ablation_sample`, and `heldout_ablation_improvement`.
It does pass split verification, shared-manifest, prediction-hash,
prediction/metric consistency, and distinct-seed checks.

A follow-up precision-first matrix on 2026-06-20 used:

```bash
VERARAG_GPU_TMUX_SESSION=verarag-conflict-train-precision \
VERARAG_GPU_OUTPUT_PREFIX=outputs/conflict_cross_encoder_v112_precision \
VERARAG_GPU_DATASET_DIR=outputs/conflict_pairs_v112_precision \
VERARAG_GPU_ABLATION_OUTPUT=outputs/conflict_detector_v112_precision_matrix_test.json \
VERARAG_GPU_AUDIT_OUTPUT=outputs/conflict_model_promotion_audit_precision.json \
VERARAG_GPU_THRESHOLD_OBJECTIVE=precision \
VERARAG_GPU_MIN_THRESHOLD_PRECISION=1.0 \
scripts/start_windows_conflict_training_matrix.sh
```

It improved only seed 13 and remained unstable across seeds:

| Seed | Selected threshold | Validation P/R/F1 | Test P/R/F1 | Test TP / FP / FN |
| ---: | ---: | ---: | ---: | ---: |
| 13 | 0.769 | 1.000 / 0.333 / 0.500 | 0.333 / 0.667 / 0.444 | 2 / 4 / 1 |
| 17 | 0.604 | 0.333 / 0.333 / 0.333 | 0.000 / 0.000 / 0.000 | 0 / 3 / 3 |
| 23 | 0.302 | 0.286 / 0.667 / 0.400 | 0.188 / 1.000 / 0.316 | 3 / 13 / 0 |

The report-only promotion audit again rejected the learned model. The
precision-first run confirms that the bottleneck is not just threshold tuning;
stage-1 work should add stronger hard negatives, more independently labeled
conflict pairs, or a different training target before enabling a learned
detector in the pipeline.

On the held-out dependency-aware test split, only three conflict-bearing
questions and three gold pairs are available. Gold-evidence pipeline A/B gives
the same result with and without the seed-13 model:

| Variant | Precision | Recall | F1 | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: |
| rules | 0.750 | 1.000 | 0.857 | 3 / 1 / 0 |
| rules + learned | 0.750 | 1.000 | 0.857 | 3 / 1 / 0 |

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python experiments/compare_conflict_detectors.py \
  --split test \
  --learned-model-path outputs/conflict_cross_encoder_v112_leakfree_seed13 \
  --learned-threshold 0.336213 \
  --output outputs/conflict_detector_v112_leakfree_seed13_test.json
```

Do not spend LLM API budget on a full learned-detector pipeline A/B until the
offline held-out detector improves beyond the rules baseline.

## Promotion Gate

Learned conflict detection remains disabled until
`verarag-audit-conflict-model` returns `decision: promote`. The default policy
requires:

- at least three distinct random seeds with identical verified dataset
  manifests and hash-verified row-level predictions;
- mean test F1 at least 0.70, worst-seed F1 at least 0.50, and worst dependency
  bootstrap F1 lower bound at least 0.30;
- a separately maintained test set whose questions fingerprint differs from
  the training benchmark;
- a passing `verarag-validate-external-conflicts` audit for that test set,
  including independent labels, adjudication, and acceptable binary conflict
  Cohen's kappa;
- annotation packets generated by `verarag-build-external-annotation-packet`
  or an equivalent blind process that does not expose gold answers or expected
  conflicts to annotators;
- compiled annotation outputs generated by `verarag-compile-external-annotations`
  or an equivalent process that produces the same audit schema;
- at least 10 conflict-bearing questions and 10 gold conflicts in the A/B;
- rules+learned F1 improvement of at least 0.02, no recall regression, and no
  additional false positives.

The thresholds are CLI-configurable for research sensitivity analysis, but
changing them must be reported. `--allow-internal-heldout` is explicitly
development-only and is not acceptable evidence for a release claim.

## Historical Pipeline Smokes

The following DeepSeek pipeline smoke on the first three conflict questions completed
without runtime errors. The fact-slot gates fixed the original over-detection
failure, and embedded counter-claim extraction plus question-focused graph
filtering now recover the three gold conflicts without false positives. A
reasoning-stage fallback also ensures detected conflicts are explicitly
acknowledged in the final answer.

| Variant | Answer F1 | Evidence Recall | Conflict F1 | Conflict Precision | Conflict Recall | Predicted / Gold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rules, counterclaim + focused graph + conflict answer note | 0.5223 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3 / 3 |
| rules | 0.2802 | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0 / 3 |
| rules + learned, threshold 0.5 | 0.2103 | 0.8333 | 0.0000 | 0.0000 | 0.0000 | 3 / 3 |
| rules + learned, threshold 0.8 | 0.5549 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 3 / 3 |

The historical broader Windows GPU smoke below uses `configs/deepseek_rules_only.yaml`,
which disables verifier/NLI/semantic-dedup Hugging Face model loading for
network-restricted WSL environments while keeping the rules conflict graph,
repair, and uncertainty stages enabled.

For targeted reruns after fixing specific failures, use exact VeraBench ids.
This is the preferred check before spending time on a full 152-question run:

```bash
DEEPSEEK_API_KEY=<key> python experiments/run_verabench.py \
  --config configs/verabench_v112_canonical.yaml \
  --ids V036 V048 V084 \
  --output results/verabench_targeted_failures.json \
  --restart
```

```bash
DEEPSEEK_API_KEY=<key> python experiments/run_verabench.py \
  --config configs/deepseek_rules_only.yaml \
  --output results/verabench_conflict_misleading_rules_only_max10_v16/rules.json \
  --types conflict misleading \
  --max 10 \
  --restart
```

| Run | Answer F1 | Evidence Recall | Conflict F1 | Conflict Precision | Conflict Recall | Behavior Acc | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rules-only max25 v5 | 0.2399 | 1.0000 | 0.6400 | 1.0000 | 0.9000 | 0.9600 | 18 / 0 / 2 |
| rules-only max25 v3 | 0.2487 | 1.0000 | 0.6400 | 1.0000 | 0.9000 | 0.9600 | 18 / 0 / 2 |
| rules-only max25 v2 | 0.3008 | 1.0000 | 0.6400 | 1.0000 | 0.9000 | 0.9600 | 18 / 0 / 2 |
| rules-only max10 v16 | 0.3437 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 12 / 0 / 0 |
| rules-only max10 v8 | 0.3586 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 12 / 0 / 0 |
| rules-only max10 v6 | 0.3264 | 1.0000 | 0.9333 | 0.8571 | 1.0000 | 1.0000 | 12 / 2 / 0 |
| rules-only max10 v4 | 0.3846 | 1.0000 | 0.7667 | 0.8182 | 0.7500 | 0.8000 | 9 / 2 / 3 |
| previous rules max10 | 0.4149 | 0.9500 | 0.4000 | 1.0000 | 0.3333 | 0.5000 | 4 / 0 / 8 |

The max10 v16 smoke clears the previous V021/V024/V042 failures and the later
V017/V041 over-detection. A larger max25 run then exposed a different pattern:
the first real max25 pipeline reported TP/FP/FN `13/17/7`, dominated by
over-detection from weak `AI` entity matches, process-node numeric comparisons,
and broad self-refutation edges. After the structural fixes, `windows-gpu`
max25 v2/v3/v5 verifies the repair end to end at TP/FP/FN `18/0/2`: precision is
now `1.0000`, recall is `0.9000`, and the remaining misses are
overgeneralization cases where the gold pair represents "evidence jointly
refutes the user premise" rather than two evidence items contradicting each
other. The v5 evaluator reports this separately as premise refutation:
TP/FP/FN `17/0/0`, precision `1.0000`, recall `1.0000`.

Use a trained model in the pipeline:

```yaml
conflict_graph:
  enable_learned_detector: true
  learned_model_path: outputs/conflict_cross_encoder
  learned_threshold: 0.8
```

## Windows GPU Workflow

Start Windows, log in to the desktop, and wait for Tailscale and the WSL keepalive
task to start. From the Mac:

```bash
ssh windows-gpu
```

Sync this repository to Windows:

```bash
scripts/sync_windows_gpu.sh
# or
make gpu-sync
```

The script defaults to:

- host: `windows-gpu`
- remote path: `~/projects/VeraRAG`

Override when needed:

```bash
VERARAG_GPU_HOST=windows-gpu \
VERARAG_GPU_PROJECT='~/projects/VeraRAG' \
scripts/sync_windows_gpu.sh
```

Quote `VERARAG_GPU_PROJECT` when setting it inline so the Mac shell does not
expand `~` into a local path before `rsync` runs.
The sync helper resolves `~` on the remote host before calling rsync, so paths
with spaces remain supported after shell quoting.

Start a quick single-seed conflict training smoke in a detached tmux session:

```bash
scripts/start_windows_conflict_training.sh
```

The launcher refuses to start if the target tmux session already exists. Attach
to the existing session instead of overwriting it.

Attach later:

```bash
ssh windows-gpu
tmux attach -t verarag-conflict-train
```

Detach without stopping training: press `Ctrl+B`, then `D`.

Check remote training state without attaching to tmux:

```bash
scripts/windows_gpu_status.sh
# or
make gpu-status
```

The status helper is read-only. It reports the resolved remote project path,
known tmux sessions, attach commands, current GPU utilization and memory, disk
space, and the newest training/evaluation artifacts under `outputs/`.

Watch GPU continuously:

```bash
scripts/windows_gpu_status.sh gpu
```

Before starting a training run, the launchers run a remote preflight by
default. You can also run it directly:

```bash
scripts/check_windows_conflict_training_ready.sh
# or
make gpu-check
```

The preflight checks SSH reachability, the remote project path, tmux, the
`train` conda environment, train dependencies, CUDA visibility, and the
offline base model path when `VERARAG_GPU_OFFLINE=1`. If you intentionally want
to bypass this gate during a manual repair, set `VERARAG_GPU_SKIP_PREFLIGHT=1`
for the launcher command.

For formal reproducibility evidence, use the multi-seed matrix launcher:

```bash
scripts/start_windows_conflict_training_matrix.sh
```

It defaults to seeds `13 17 23`, writes separate model directories under
`outputs/conflict_cross_encoder_v112_leakfree_seed*`, runs a held-out
gold-evidence detector A/B on the first seed, and writes a report-only
promotion audit to `outputs/conflict_model_promotion_audit_matrix.json`.
Attach to the matrix session with:

```bash
ssh windows-gpu
tmux attach -t verarag-conflict-train-matrix
```

Customize formal runs with environment variables:

```bash
VERARAG_GPU_SEEDS="13 17 23 29 31" \
VERARAG_GPU_EPOCHS=3 \
VERARAG_GPU_OUTPUT_PREFIX=outputs/conflict_cross_encoder_v112_expanded \
scripts/start_windows_conflict_training_matrix.sh
```

The promotion audit is intentionally `--report-only` inside the detached job:
a rejected model is still a useful, reproducible negative result and should not
erase the training artifacts or stop the evidence collection script early.

## Secure Remote VeraBench Evaluation

Use the detached VeraBench launcher for real DeepSeek runs on the Windows GPU
host. It reads the DeepSeek key silently on the Mac, transfers it through an SSH
stdin pipe into a one-shot remote FIFO, and exports it only inside the tmux job.
The key is not written to repository files, output JSON, or shell history by the
launcher.

```bash
scripts/start_windows_verabench_eval.sh
```

Defaults:

- config: `configs/verabench_v112_canonical.yaml`
- output: `outputs/remote_results/verabench_v112_canonical_deepseek.json`
- session: `verarag-verabench-eval`
- restart mode: enabled

Attach later:

```bash
ssh windows-gpu
tmux attach -t verarag-verabench-eval
```

For targeted repair checks before a full 152-question spend:

```bash
VERARAG_EVAL_IDS="V036 V048 V084" \
VERARAG_EVAL_OUTPUT=outputs/remote_results/verabench_v112_targeted_failures.json \
scripts/start_windows_verabench_eval.sh
```

For a bounded rules-only smoke:

```bash
VERARAG_EVAL_CONFIG=configs/deepseek_rules_only.yaml \
VERARAG_EVAL_TYPES="conflict misleading" \
VERARAG_EVAL_MAX=25 \
VERARAG_EVAL_OUTPUT=outputs/remote_results/verabench_v112_conflict_misleading_max25.json \
scripts/start_windows_verabench_eval.sh
```

If your current terminal already has a fresh key exported, the launcher can
read `VERARAG_DEEPSEEK_API_KEY`; prefer that variable over putting
`DEEPSEEK_API_KEY=<key>` directly in a command line.

For one-off manual GPU inspection after SSH, `watch -n 1
/usr/lib/wsl/lib/nvidia-smi` is still useful. Prefer
`scripts/windows_gpu_status.sh gpu` from the Mac when you do not need an
interactive shell.

## Manual Remote Commands

If you prefer to run step by step:

```bash
ssh windows-gpu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate train
cd ~/projects/VeraRAG
python experiments/build_conflict_training_data.py --output-dir outputs/conflict_pairs
tmux new -s verarag-conflict-train
python experiments/train_conflict_cross_encoder.py \
  --train outputs/conflict_pairs/train.jsonl \
  --val outputs/conflict_pairs/val.jsonl \
  --test outputs/conflict_pairs/test.jsonl \
  --output-dir outputs/conflict_cross_encoder \
  --device cuda \
  --warmup-steps 10 \
  --seed 13
```

## Operational Notes

- Windows shutdown, reboot, sleep, or hibernate stops active training.
- It is fine to shut down Windows when no training is running.
- During training, keep Windows powered on, online, and with automatic sleep
  disabled.
- For real VeraBench runs, prefer `scripts/start_windows_verabench_eval.sh` over
  inline `DEEPSEEK_API_KEY=<key> ...` commands so the key does not enter shell
  history.
- The first real training run downloads the base CrossEncoder from Hugging Face.
  If the Windows GPU host cannot reach Hugging Face, pre-cache the model or pass
  a local model directory with `--model /path/to/model`.
- NLI CrossEncoder checkpoints may contain a 3-way classifier head. VeraRAG
  intentionally reinitializes that head as a single conflict-score output while
  keeping the encoder body.
- The default `--warmup-steps 10` is part of the baseline definition. On this
  small dataset, using 100 warmup steps changes the optimization regime because
  a three-epoch balanced run has only 42 optimizer steps.
- Model saving disables automatic Hugging Face model-card generation so an
  offline Windows host cannot stall while resolving remote model metadata.
- Use `training_metrics.json` to choose `conflict_graph.learned_threshold`; the
  default threshold remains 0.7 unless the validation-selected value is stronger.
- Use `--seed` (default: 13) for reproducible shuffle and model initialization
  when comparing training runs.
- `training_metadata.json` records both the raw train split and the balanced
  `train_loader` distribution used for optimization.
- After a Windows reboot, log in to the user account and wait one or two minutes
  before SSH so the keepalive task and Tailscale are ready.
- Raw model outputs under `outputs/` are ignored by git. Publish only model cards,
  metrics, and small reproducibility metadata unless intentionally releasing a
  trained artifact.
