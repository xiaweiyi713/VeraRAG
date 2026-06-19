"""Tests for local Markdown documentation link validation."""

import subprocess
import sys

from experiments.validate_docs import validate_docs


def test_validate_docs_accepts_local_files_and_anchors(tmp_path):
    readme = tmp_path / "README.md"
    docs = tmp_path / "docs"
    docs.mkdir()
    guide = docs / "GUIDE.md"
    readme.write_text(
        "\n".join(
            [
                "# Project",
                "[Guide](docs/GUIDE.md)",
                "[Section](docs/GUIDE.md#details)",
                "[External](https://example.com/path)",
            ]
        ),
        encoding="utf-8",
    )
    guide.write_text("# Guide\n\n## Details\n", encoding="utf-8")

    report = validate_docs(["README.md", "docs"], root=tmp_path)

    assert report["valid"]
    assert report["checked_links"] == 2
    assert report["issues"] == []


def test_validate_docs_reports_missing_file_and_anchor(tmp_path):
    readme = tmp_path / "README.md"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "GUIDE.md").write_text("# Guide\n", encoding="utf-8")
    readme.write_text(
        "\n".join(
            [
                "# Project",
                "[Missing](docs/MISSING.md)",
                "[Missing anchor](docs/GUIDE.md#missing-section)",
            ]
        ),
        encoding="utf-8",
    )

    report = validate_docs(["README.md", "docs"], root=tmp_path)

    assert not report["valid"]
    assert [issue["message"] for issue in report["issues"]] == [
        "local link target does not exist",
        "Markdown anchor does not exist",
    ]


def test_validate_docs_cli_emits_json(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Project\n[Self](#project)\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "experiments/validate_docs.py",
            "--root",
            str(tmp_path),
            "--json",
            "README.md",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"valid": true' in result.stdout
    assert '"checked_links": 1' in result.stdout


def test_validate_docs_default_scan_excludes_internal_superpowers(tmp_path):
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")
    docs = tmp_path / "docs"
    internal = docs / "superpowers" / "plans"
    internal.mkdir(parents=True)
    (internal / "plan.md").write_text("[Broken](missing.md)\n", encoding="utf-8")

    report = validate_docs(["README.md", "docs"], root=tmp_path)

    assert report["valid"]
    assert "docs/superpowers/plans/plan.md" not in report["documents"]


def test_validate_docs_accepts_documented_commands(tmp_path):
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "# Project",
                "Run `make docs-check` and `verarag-validate-docs`.",
                "```bash",
                "make examples-check",
                "verarag-validate-docs --json",
                "verarag-validate-examples --json",
                "```",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "Makefile").write_text(
        "docs-check:\n\tpython validate.py\nexamples-check:\n\tpython examples.py\n",
        encoding="utf-8",
    )
    _write_pyproject_scripts(
        tmp_path,
        {
            "verarag-validate-docs": "experiments.validate_docs:main",
            "verarag-validate-examples": "experiments.validate_examples:main",
        },
    )

    report = validate_docs(["README.md"], root=tmp_path)

    assert report["valid"]
    assert report["checked_command_references"] == 4
    assert report["issues"] == []


def test_validate_docs_reports_command_drift(tmp_path):
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "# Project",
                "```bash",
                "make missing-target",
                "verarag-missing-command",
                "```",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "Makefile").write_text("docs-check:\n\tpython validate.py\n", encoding="utf-8")
    _write_pyproject_scripts(
        tmp_path,
        {
            "verarag-validate-docs": "experiments.validate_docs:main",
        },
    )

    report = validate_docs(["README.md"], root=tmp_path)

    assert not report["valid"]
    assert [issue["message"] for issue in report["issues"]] == [
        "documented make target is not declared in Makefile",
        "documented console script is not declared in pyproject.toml",
        "console script is not documented in Markdown docs",
    ]


def _write_pyproject_scripts(root, scripts):
    script_lines = [f'{name} = "{target}"' for name, target in scripts.items()]
    (root / "pyproject.toml").write_text(
        "\n".join(["[project]", 'name = "demo"', 'version = "0.1.0"', "[project.scripts]", *script_lines]),
        encoding="utf-8",
    )
