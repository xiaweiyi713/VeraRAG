"""Tests for the repository secret scanner."""

import subprocess
import sys

from experiments.scan_secrets import findings_to_sarif, scan_paths


def _fake_secret() -> str:
    return "sk-" + ("r" * 32)


def test_secret_scan_detects_realistic_api_key(tmp_path):
    leaked = tmp_path / "leaked.py"
    token = _fake_secret()
    leaked.write_text(
        f'DEEPSEEK_API_KEY = "{token}"\n',
        encoding="utf-8",
    )

    findings = scan_paths([tmp_path])

    assert len(findings) == 2
    assert {finding.rule for finding in findings} == {
        "generic_secret_assignment",
        "openai_or_deepseek_style_key",
    }
    assert all(token not in finding.redacted for finding in findings)


def test_secret_scan_sarif_output_is_machine_readable_and_redacted(tmp_path):
    leaked = tmp_path / "leaked.py"
    token = _fake_secret()
    leaked.write_text(f'DEEPSEEK_API_KEY = "{token}"\n', encoding="utf-8")

    findings = scan_paths([tmp_path])
    sarif = findings_to_sarif(findings, root=tmp_path)

    assert sarif["version"] == "2.1.0"
    run = sarif["runs"][0]
    assert run["tool"]["driver"]["name"] == "verarag-scan-secrets"
    assert {rule["id"] for rule in run["tool"]["driver"]["rules"]} == {
        "generic_secret_assignment",
        "openai_or_deepseek_style_key",
    }
    assert {result["ruleId"] for result in run["results"]} == {
        "generic_secret_assignment",
        "openai_or_deepseek_style_key",
    }
    for result in run["results"]:
        assert token not in result["message"]["text"]
        location = result["locations"][0]["physicalLocation"]
        assert location["artifactLocation"]["uri"] == "leaked.py"
        assert location["region"]["startLine"] == 1


def test_secret_scan_allows_placeholders_and_test_keys(tmp_path):
    docs = tmp_path / "README.md"
    docs.write_text(
        "\n".join(
            [
                'OPENAI_API_KEY="sk-xxx"',
                'DEEPSEEK_API_KEY="<key>"',
                'api_key = "sk-test-secret-key-12345"',
                'api_key = "${DEEPSEEK_API_KEY}"',
                'api_key = "your-api-key"',
            ]
        ),
        encoding="utf-8",
    )

    assert scan_paths([tmp_path]) == []


def test_secret_scan_skips_build_and_environment_directories(tmp_path):
    openai_token = "sk-" + ("a" * 32)
    github_token = "github_pat_" + ("a" * 48)
    leaked_dir = tmp_path / ".venv"
    leaked_dir.mkdir()
    (leaked_dir / "activate").write_text(
        f'OPENAI_API_KEY="{openai_token}"\n',
        encoding="utf-8",
    )
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "metadata.txt").write_text(
        f'token="{github_token}"\n',
        encoding="utf-8",
    )

    assert scan_paths([tmp_path]) == []


def test_secret_scan_covers_env_local_unquoted_assignment(tmp_path):
    token = _fake_secret()
    env_file = tmp_path / ".env.local"
    env_file.write_text(f"DEEPSEEK_API_KEY={token}\n", encoding="utf-8")

    findings = scan_paths([tmp_path])

    assert {finding.rule for finding in findings} == {
        "env_secret_assignment",
        "openai_or_deepseek_style_key",
    }
    assert all(".env.local" in finding.path for finding in findings)


def test_secret_scan_can_include_ignored_local_env_files(tmp_path):
    token = _fake_secret()
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / ".gitignore").write_text(".env.local\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text(
        f"DEEPSEEK_API_KEY={token}\n",
        encoding="utf-8",
    )

    assert scan_paths([tmp_path]) == []

    findings = scan_paths([tmp_path], include_ignored=True)

    assert {finding.rule for finding in findings} == {
        "env_secret_assignment",
        "openai_or_deepseek_style_key",
    }
    assert all(".env.local" in finding.path for finding in findings)


def test_secret_scan_detects_aws_keys_and_private_key_headers(tmp_path):
    aws_key = "AKIA" + ("A" * 16)
    private_key_header = "-----BEGIN " + "PRIVATE KEY-----"
    credentials = tmp_path / "credentials.txt"
    credentials.write_text(
        f"aws_access_key_id={aws_key}\n{private_key_header}\n",
        encoding="utf-8",
    )

    findings = scan_paths([tmp_path])

    assert {finding.rule for finding in findings} == {
        "aws_access_key_id",
        "env_secret_assignment",
        "private_key_header",
    }


def test_secret_scan_ignores_non_literal_frontend_api_key_assignment(tmp_path):
    frontend = tmp_path / "settings.js"
    frontend.write_text(
        "const api_key = document.getElementById('cfg-apikey').value;\n",
        encoding="utf-8",
    )

    assert scan_paths([tmp_path]) == []


def test_secret_scan_cli_emits_sarif_and_rejects_dual_formats(tmp_path):
    leaked = tmp_path / "leaked.env"
    token = _fake_secret()
    leaked.write_text(f"DEEPSEEK_API_KEY={token}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "experiments/scan_secrets.py",
            "--sarif",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert '"version": "2.1.0"' in result.stdout
    assert "verarag-scan-secrets" in result.stdout
    assert token not in result.stdout

    result = subprocess.run(
        [
            sys.executable,
            "experiments/scan_secrets.py",
            "--json",
            "--sarif",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "mutually exclusive" in result.stderr
