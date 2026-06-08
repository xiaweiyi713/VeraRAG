# VeraBench Results

This file is generated from saved `run_verabench.py --output` JSON reports.
Commit raw result JSON files only when intentionally publishing a reproducibility artifact; otherwise keep them in ignored `results/` and regenerate this summary.

Generation command:

```bash
python experiments/build_verabench_leaderboard.py results/verabench_full.json results/verabench_full_v2.json results/verabench_full_v3.json --output docs/RESULTS.md
```

## Leaderboard

| Rank | Run | Model | Questions | Errors | Behavior Acc | Answer F1 | Evidence Recall | Conflict F1 | ECE | Avg Latency | Commit |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | verabench_full_v3 | deepseek/deepseek-v4-flash | 152/152 | 0 | 0.763 | 0.281 | 0.799 | 0.006 | 0.416 | 61.7s | acb0b5f |
| 2 | verabench_full_v2 | deepseek/deepseek-v4-flash | 152/152 | 0 | 0.743 | 0.271 | 0.814 | 0.007 | 0.415 | 66.6s | acb0b5f |
| 3 | verabench_full | deepseek/deepseek-v4-flash | 152/152 | 0 | 0.526 | 0.157 | 0.811 | 0.007 | 0.062 | 79.2s | cc29ed2 |

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
