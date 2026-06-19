"""Smoke tests for GPU training orchestration scripts."""

import os
import subprocess
from pathlib import Path


def test_windows_conflict_training_scripts_have_valid_bash_syntax():
    for script in (
        "scripts/check_windows_conflict_training_ready.sh",
        "scripts/start_windows_conflict_training.sh",
        "scripts/start_windows_conflict_training_matrix.sh",
        "scripts/start_windows_verabench_eval.sh",
        "scripts/windows_gpu_status.sh",
    ):
        subprocess.run(["bash", "-n", script], check=True)


def test_matrix_training_script_matches_promotion_audit_workflow():
    script = Path("scripts/start_windows_conflict_training_matrix.sh").read_text(
        encoding="utf-8"
    )

    assert "VERARAG_GPU_SEEDS:-13 17 23" in script
    assert "check_windows_conflict_training_ready.sh" in script
    assert "VERARAG_GPU_SKIP_PREFLIGHT" in script
    assert "quote_remote()" in script
    assert "tmux has-session" in script
    assert "VERARAG_REMOTE_PROJECT=$(quote_remote \"${REMOTE_PROJECT}\")" in script
    assert "printf 'VERARAG_REMOTE_PROJECT=%q\\n'" in script
    assert "printf 'VERARAG_REMOTE_AUDIT_OUTPUT=%q\\n'" in script
    assert "VERARAG_REMOTE_PROJECT='${VERARAG_REMOTE_PROJECT}'" not in script
    assert "experiments/compare_conflict_detectors.py" in script
    assert "experiments/audit_conflict_model.py" in script
    assert "--report-only" in script
    assert "conflict_model_promotion_audit_matrix.json" in script


def test_sync_script_resolves_remote_project_before_rsync():
    script = Path("scripts/sync_windows_gpu.sh").read_text(encoding="utf-8")

    assert "quote_remote()" in script
    assert "REMOTE_PROJECT_RESOLVED" in script
    assert 'project="${VERARAG_REMOTE_PROJECT/#\\~/$HOME}"' in script
    assert "mkdir -p \"${project}\"" in script
    assert "${REMOTE_HOST}:${REMOTE_PROJECT_RSYNC}/" in script
    assert "mkdir -p ${REMOTE_PROJECT}" not in script


def test_windows_gpu_status_reports_read_only_remote_state():
    script = Path("scripts/windows_gpu_status.sh").read_text(encoding="utf-8")

    assert "Usage: scripts/windows_gpu_status.sh [status|gpu]" in script
    assert "VERARAG_GPU_SESSIONS" in script
    assert "VERARAG_REMOTE_PROJECT=$(quote_remote \"${REMOTE_PROJECT}\")" in script
    assert 'project="${VERARAG_REMOTE_PROJECT/#\\~/$HOME}"' in script
    assert "tmux has-session" in script
    assert "tmux attach -t" in script
    assert "/usr/lib/wsl/lib/nvidia-smi --query-gpu" in script
    assert "training_metrics.json" in script
    assert "training_metadata.json" in script
    assert "train.log" in script
    assert "tmux kill-session" not in script
    assert "rm -rf" not in script


def test_single_seed_training_script_quotes_remote_values():
    script = Path("scripts/start_windows_conflict_training.sh").read_text(
        encoding="utf-8"
    )

    assert "check_windows_conflict_training_ready.sh" in script
    assert "VERARAG_GPU_SKIP_PREFLIGHT" in script
    assert "quote_remote()" in script
    assert "tmux has-session" in script
    assert "VERARAG_REMOTE_PROJECT=$(quote_remote \"${REMOTE_PROJECT}\")" in script
    assert 'project="${VERARAG_REMOTE_PROJECT/#\\~/$HOME}"' in script
    assert 'base_model="${VERARAG_REMOTE_BASE_MODEL/#\\~/$HOME}"' in script
    assert 'cd "${project}"' in script
    assert "cd ${REMOTE_PROJECT}" not in script
    assert "--model \"${base_model}\"" in script


def test_conflict_training_preflight_checks_remote_cuda_and_model():
    script = Path("scripts/check_windows_conflict_training_ready.sh").read_text(
        encoding="utf-8"
    )

    assert "VERARAG_GPU_SSH_CONNECT_TIMEOUT" in script
    assert "ConnectTimeout=${SSH_CONNECT_TIMEOUT}" in script
    assert "conda env exists" in script
    assert "torch.cuda.is_available()" in script
    assert "sentence_transformers" in script
    assert "offline base model exists" in script
    assert "Remote conflict-training preflight failed" in script
    assert "Remote conflict-training preflight could not complete" in script
    assert "VERARAG_GPU_SKIP_PREFLIGHT=1" in script


def test_verabench_eval_launcher_uses_fifo_secret_injection():
    script = Path("scripts/start_windows_verabench_eval.sh").read_text(
        encoding="utf-8"
    )

    assert "read -r -s -p" in script
    assert "VERARAG_DEEPSEEK_API_KEY" in script
    assert "configs/verabench_v112_canonical.yaml" in script
    assert "verabench_v112_canonical_deepseek.json" in script
    assert "mkfifo" in script
    assert "printf 'VERARAG_REMOTE_PROJECT=%q\\n'" in script
    assert "printf 'VERARAG_REMOTE_KEY_FIFO=%q\\n'" in script
    assert "printf 'VERARAG_REMOTE_EXTRA_ARGS=%q\\n'" in script
    assert "VERARAG_REMOTE_PROJECT='${VERARAG_REMOTE_PROJECT}'" not in script
    assert 'timeout 30s bash -c' in script
    assert 'IFS= read -r key < "$1"; printf "%s" "${key}"' in script
    assert "timeout 30s cat > $(quote_remote \"${KEY_FIFO}\")" in script
    assert "DEEPSEEK_API_KEY=<" not in script
    assert "--config \"${VERARAG_REMOTE_CONFIG}\"" in script
    assert "--output \"${VERARAG_REMOTE_OUTPUT}\"" in script
    assert "Set either VERARAG_EVAL_TYPES or VERARAG_EVAL_IDS, not both." in script


def test_verabench_eval_launcher_does_not_expose_key_in_ssh_args(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ssh = fake_bin / "ssh"
    log_path = tmp_path / "ssh.log"
    key = "unit-test-deepseek-key"
    fake_ssh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
log="${VERARAG_FAKE_SSH_LOG}"
printf 'ARGS:' >> "${log}"
printf ' %q' "$@" >> "${log}"
printf '\\n' >> "${log}"
payload="$(cat)"
case "$*" in
  *"cat >"*)
    printf 'STDIN_LEN:%s\\n' "${#payload}" >> "${log}"
    ;;
  *)
    case "${payload}" in
      *"${VERARAG_DEEPSEEK_API_KEY}"*)
        printf 'REMOTE_STDIN_HAS_KEY:yes\\n' >> "${log}"
        exit 9
        ;;
      *)
        printf 'REMOTE_STDIN_HAS_KEY:no\\n' >> "${log}"
        ;;
    esac
    printf 'verarag-verabench-eval: 1 windows\\n'
    ;;
esac
""",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "VERARAG_FAKE_SSH_LOG": str(log_path),
        "VERARAG_DEEPSEEK_API_KEY": key,
    }
    result = subprocess.run(
        ["bash", "scripts/start_windows_verabench_eval.sh"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    log = log_path.read_text(encoding="utf-8")
    assert key not in log
    assert key not in result.stdout
    assert key not in result.stderr
    assert "REMOTE_STDIN_HAS_KEY:no" in log
    assert f"STDIN_LEN:{len(key)}" in log
