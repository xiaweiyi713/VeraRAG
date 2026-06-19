"""Tests for open-source project metadata validation."""

from pathlib import Path

from experiments.validate_project_metadata import validate_project_metadata

CHECKOUT_SHA = "11bd71901bbe5b1630ceea73d27597364c9af683"
SETUP_PYTHON_SHA = "a26af69be951a213d495a4c3e4e4022e16d87065"
CODEQL_SHA = "3e6af16ff035267728e2ebc35df5d4c4cf249f81"
SCORECARD_SHA = "4eaacf0543bb3f2c246792bd56e8cdeffafb205a"
UPLOAD_ARTIFACT_SHA = "ea165f8d65b6e75b540449e92b4886f43607fa02"
DOWNLOAD_ARTIFACT_SHA = "d3f86a106a0bac45b974a628896c90dbdf5c8093"
PYPI_PUBLISH_SHA = "7f25271a4aa483500f742f9492b2ab5648d61011"
ATTEST_BUILD_PROVENANCE_SHA = "a2bbfa25375fe432b6a289bc6b6cd05ecd0c4c32"


def test_project_metadata_accepts_consistent_fixture(tmp_path):
    _write_metadata_fixture(tmp_path)

    audit = validate_project_metadata(tmp_path)

    assert audit.valid
    assert audit.errors == []
    assert audit.project_name == "verarag"
    assert audit.project_version == "0.1.0"
    assert audit.project_urls["Repository"] == "https://github.com/example/verarag"


def test_project_metadata_reports_missing_project_url(tmp_path):
    _write_metadata_fixture(tmp_path, include_security_url=False)

    audit = validate_project_metadata(tmp_path)

    assert not audit.valid
    assert "pyproject [project.urls] missing Security" in audit.errors


def test_project_metadata_reports_missing_readme_governance_link(tmp_path):
    _write_metadata_fixture(tmp_path, readme_links=("CONTRIBUTING.md", "SECURITY.md"))

    audit = validate_project_metadata(tmp_path)

    assert not audit.valid
    assert "README missing link or reference to CODE_OF_CONDUCT.md" in audit.errors
    assert "README missing link or reference to CITATION.cff" in audit.errors


def test_project_metadata_reports_stale_contributing_gate(tmp_path):
    _write_metadata_fixture(tmp_path, contributing_commands=("make lint", "make docs-check"))

    audit = validate_project_metadata(tmp_path)

    assert not audit.valid
    assert "CONTRIBUTING missing quality gate command: make version-check" in audit.errors
    assert "CONTRIBUTING missing quality gate command: make python-support-check" in audit.errors
    assert "CONTRIBUTING missing quality gate command: make results-check" in audit.errors
    assert "CONTRIBUTING missing quality gate command: make examples-check" in audit.errors
    assert "CONTRIBUTING missing quality gate command: make deployment-check" in audit.errors
    assert "CONTRIBUTING missing quality gate command: make precommit-check" in audit.errors
    assert "CONTRIBUTING missing quality gate command: make release-check" in audit.errors
    assert "CONTRIBUTING missing quality gate command: make package-check" in audit.errors


def test_project_metadata_reports_citation_version_mismatch(tmp_path):
    _write_metadata_fixture(tmp_path, citation_version="9.9.9")

    audit = validate_project_metadata(tmp_path)

    assert not audit.valid
    assert "CITATION.cff version must match pyproject project.version" in audit.errors


def test_project_metadata_rejects_deprecated_license_classifier(tmp_path):
    _write_metadata_fixture(tmp_path, include_license_classifier=True)

    audit = validate_project_metadata(tmp_path)

    assert not audit.valid
    assert "pyproject classifiers should not include deprecated license classifiers" in audit.errors


def test_project_metadata_reports_incomplete_dependabot_config(tmp_path):
    _write_metadata_fixture(tmp_path, dependabot=_dependabot_content(include_actions=False))

    audit = validate_project_metadata(tmp_path)

    assert not audit.valid
    assert "Dependabot should monitor GitHub Actions" in audit.errors
    assert "Dependabot pip and GitHub Actions updates should run weekly" in audit.errors


def test_project_metadata_reports_unbounded_workflow_permissions(tmp_path):
    _write_metadata_fixture(
        tmp_path,
        workflow="name: Tests\njobs:\n  test:\n    runs-on: ubuntu-latest\n",
    )

    audit = validate_project_metadata(tmp_path)

    assert not audit.valid
    assert "GitHub Actions workflow should set top-level contents: read permission" in audit.errors
    assert "GitHub Actions test job should define timeout-minutes" in audit.errors
    assert "GitHub Actions workflow should run project metadata validation" in audit.errors
    assert "GitHub Actions workflow should run version identity validation" in audit.errors
    assert "GitHub Actions workflow should run Python support validation" in audit.errors
    assert "GitHub Actions workflow should run published results validation" in audit.errors


def test_project_metadata_reports_unowned_or_unrouted_github_intake(tmp_path):
    _write_metadata_fixture(
        tmp_path,
        codeowners="docs/ README.md\n",
        issue_template_config="blank_issues_enabled: true\n",
    )

    audit = validate_project_metadata(tmp_path)

    assert not audit.valid
    assert "CODEOWNERS should define a repository-wide owner" in audit.errors
    assert "GitHub issue templates should disable blank issues" in audit.errors
    assert "GitHub issue templates should route vulnerabilities to SECURITY" in audit.errors


def test_project_metadata_reports_disabled_codeql_security_scanning(tmp_path):
    _write_metadata_fixture(
        tmp_path,
        codeql_workflow=(
            "name: CodeQL\n"
            "permissions:\n"
            "  contents: read\n"
            "jobs:\n"
            "  analyze:\n"
            "    runs-on: ubuntu-latest\n"
        ),
    )

    audit = validate_project_metadata(tmp_path)

    assert not audit.valid
    assert "CodeQL workflow should initialize github/codeql-action" in audit.errors
    assert "CodeQL workflow should run github/codeql-action/analyze" in audit.errors
    assert "CodeQL workflow should grant security-events: write" in audit.errors
    assert "CodeQL workflow should define timeout-minutes" in audit.errors
    assert "CodeQL workflow should run on a recurring schedule" in audit.errors
    assert "CodeQL workflow should analyze Python" in audit.errors


def test_project_metadata_reports_unpinned_github_actions(tmp_path):
    _write_metadata_fixture(
        tmp_path,
        workflow=_workflow_content(checkout_ref="v4"),
        codeql_workflow=_codeql_workflow_content(codeql_ref="v3"),
        release_workflow=_release_workflow_content(
            attest_ref="v4.1.0",
            pypi_publish_ref="v1.12.4",
        ),
        scorecard_workflow=_scorecard_workflow_content(scorecard_ref="v2"),
    )

    audit = validate_project_metadata(tmp_path)

    assert not audit.valid
    assert ".github/workflows/test.yml action should be pinned to a full commit SHA: actions/checkout@v4" in audit.errors
    assert (
        ".github/workflows/codeql.yml action should be pinned to a full commit SHA: "
        "github/codeql-action/init@v3"
    ) in audit.errors
    assert (
        ".github/workflows/codeql.yml action should be pinned to a full commit SHA: "
        "github/codeql-action/analyze@v3"
    ) in audit.errors
    assert (
        ".github/workflows/scorecard.yml action should be pinned to a full commit SHA: "
        "ossf/scorecard-action@v2"
    ) in audit.errors
    assert (
        ".github/workflows/release.yml action should be pinned to a full commit SHA: "
        "pypa/gh-action-pypi-publish@v1.12.4"
    ) in audit.errors
    assert (
        ".github/workflows/release.yml action should be pinned to a full commit SHA: "
        "actions/attest-build-provenance@v4.1.0"
    ) in audit.errors


def test_project_metadata_reports_disabled_scorecard_security_analysis(tmp_path):
    _write_metadata_fixture(
        tmp_path,
        scorecard_workflow=(
            "name: Scorecard\n"
            "permissions:\n"
            "  contents: read\n"
            "jobs:\n"
            "  scorecard:\n"
            "    runs-on: ubuntu-latest\n"
        ),
    )

    audit = validate_project_metadata(tmp_path)

    assert not audit.valid
    assert "Scorecard workflow should run ossf/scorecard-action" in audit.errors
    assert "Scorecard workflow should grant security-events: write" in audit.errors
    assert "Scorecard workflow should define timeout-minutes" in audit.errors
    assert "Scorecard workflow should run on a recurring schedule" in audit.errors
    assert "Scorecard workflow should publish results" in audit.errors
    assert "Scorecard workflow should preserve a JSON artifact" in audit.errors


def test_project_metadata_reports_incomplete_release_workflow(tmp_path):
    _write_metadata_fixture(
        tmp_path,
        release_workflow=(
            "name: Release\n"
            "on:\n"
            "  push:\n"
            "    branches: [main]\n"
            "permissions:\n"
            "  contents: read\n"
            "jobs:\n"
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            f"      - uses: actions/checkout@{CHECKOUT_SHA}\n"
        ),
    )

    audit = validate_project_metadata(tmp_path)

    assert not audit.valid
    assert "Release workflow should support workflow_dispatch dry-runs" in audit.errors
    assert "Release workflow should run on version tags" in audit.errors
    assert "Release workflow should run make release-check" in audit.errors
    assert "Release workflow should upload dist/*.whl" in audit.errors
    assert "Release workflow should use PyPI trusted publishing action" in audit.errors
    assert "Release workflow should attest release artifact provenance" in audit.errors
    assert "Release workflow should grant id-token: write for PyPI publishing and attestation" in audit.errors
    assert "Release workflow should grant attestations: write for provenance" in audit.errors
    assert "Release workflow should attest the uploaded artifact subject paths" in audit.errors
    assert "Release workflow should use a protected pypi environment" in audit.errors
    assert "Release workflow should publish only from version tags" in audit.errors
    assert "Release workflow should create GitHub releases for version tags" in audit.errors


def _write_metadata_fixture(
    root: Path,
    *,
    include_security_url: bool = True,
    readme_links: tuple[str, ...] = (
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "CITATION.cff",
        "CHANGELOG.md",
    ),
    contributing_commands: tuple[str, ...] = (
        "make lint",
        "make version-check",
        "make python-support-check",
        "make docs-check",
        "make results-check",
        "make examples-check",
        "make deployment-check",
        "make precommit-check",
        "make deps-check",
        "make package-check",
        "make release-check",
    ),
    citation_version: str = "0.1.0",
    include_license_classifier: bool = False,
    dependabot: str | None = None,
    codeowners: str | None = None,
    issue_template_config: str | None = None,
    codeql_workflow: str | None = None,
    release_workflow: str | None = None,
    scorecard_workflow: str | None = None,
    workflow: str | None = None,
) -> None:
    (root / ".github/ISSUE_TEMPLATE").mkdir(parents=True)
    for relative_path in (
        "CHANGELOG.md",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "LICENSE",
        ".github/CODEOWNERS",
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/workflows/codeql.yml",
        ".github/workflows/release.yml",
        ".github/workflows/scorecard.yml",
        ".github/workflows/test.yml",
        ".github/dependabot.yml",
    ):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_content_for(relative_path), encoding="utf-8")

    urls = [
        'Homepage = "https://github.com/example/verarag"',
        'Repository = "https://github.com/example/verarag"',
        'Documentation = "https://github.com/example/verarag#readme"',
        'Issues = "https://github.com/example/verarag/issues"',
        'Changelog = "https://github.com/example/verarag/blob/main/CHANGELOG.md"',
        'Citation = "https://github.com/example/verarag/blob/main/CITATION.cff"',
    ]
    if include_security_url:
        urls.append('Security = "https://github.com/example/verarag/security/policy"')
    classifiers = [
        '    "Operating System :: OS Independent",',
    ]
    if include_license_classifier:
        classifiers.insert(0, '    "License :: OSI Approved :: MIT License",')
    (root / "pyproject.toml").write_text(
        "\n".join([
            "[project]",
            'name = "verarag"',
            'version = "0.1.0"',
            'description = "Verifiable RAG"',
            'readme = "README.md"',
            'requires-python = ">=3.10"',
            'license = "MIT"',
            'keywords = ["rag", "verification", "reasoning", "benchmark"]',
            "classifiers = [",
            *classifiers,
            "]",
            "",
            "[project.urls]",
            *urls,
        ])
        + "\n",
        encoding="utf-8",
    )
    (root / ".github/dependabot.yml").write_text(
        dependabot or _dependabot_content(include_actions=True),
        encoding="utf-8",
    )
    (root / ".github/CODEOWNERS").write_text(
        codeowners or "* @example\n",
        encoding="utf-8",
    )
    (root / ".github/ISSUE_TEMPLATE/config.yml").write_text(
        issue_template_config or _issue_template_config_content(),
        encoding="utf-8",
    )
    (root / ".github/workflows/codeql.yml").write_text(
        codeql_workflow or _codeql_workflow_content(),
        encoding="utf-8",
    )
    (root / ".github/workflows/scorecard.yml").write_text(
        scorecard_workflow or _scorecard_workflow_content(),
        encoding="utf-8",
    )
    (root / ".github/workflows/release.yml").write_text(
        release_workflow or _release_workflow_content(),
        encoding="utf-8",
    )
    (root / ".github/workflows/test.yml").write_text(
        workflow or _workflow_content(),
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text(
        "\n".join([
            "cff-version: 1.2.0",
            'message: "Please cite VeraRAG."',
            'title: "VeraRAG: Verifiable Agentic Retrieval-Augmented Reasoning"',
            f"version: {citation_version}",
            "date-released: 2026-06-06",
            "license: MIT",
            "authors:",
            '  - name: "VeraRAG Team"',
            'repository-code: "https://github.com/example/verarag"',
        ])
        + "\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("\n".join(readme_links), encoding="utf-8")
    (root / "CONTRIBUTING.md").write_text(
        "\n".join([*contributing_commands, "real VeraBench evidence"]),
        encoding="utf-8",
    )


def _content_for(relative_path: str) -> str:
    if relative_path == "SECURITY.md":
        return "Report via private security advisory. If a key leaks, revoke or rotate it.\n"
    if relative_path == "CODE_OF_CONDUCT.md":
        return "Unacceptable behavior includes harassment.\n"
    if relative_path == "LICENSE":
        return "MIT License\n"
    return f"{relative_path}\n"


def _dependabot_content(*, include_actions: bool) -> str:
    entries = [
        '  - package-ecosystem: "pip"',
        '    directory: "/"',
        "    schedule:",
        '      interval: "weekly"',
        "    open-pull-requests-limit: 5",
    ]
    if include_actions:
        entries.extend([
            '  - package-ecosystem: "github-actions"',
            '    directory: "/"',
            "    schedule:",
            '      interval: "weekly"',
            "    open-pull-requests-limit: 5",
        ])
    return "\n".join(["version: 2", "updates:", *entries]) + "\n"


def _issue_template_config_content() -> str:
    return "\n".join([
        "blank_issues_enabled: false",
        "contact_links:",
        "  - name: Security vulnerability",
        "    url: https://github.com/example/verarag/security/policy",
        "    about: Report suspected vulnerabilities privately.",
    ]) + "\n"


def _codeql_workflow_content(*, codeql_ref: str = CODEQL_SHA) -> str:
    return "\n".join([
        "name: CodeQL",
        "on:",
        "  push:",
        "    branches: [main]",
        "  schedule:",
        '    - cron: "37 3 * * 1"',
        "permissions:",
        "  actions: read",
        "  contents: read",
        "  security-events: write",
        "jobs:",
        "  analyze:",
        "    runs-on: ubuntu-latest",
        "    timeout-minutes: 30",
        "    strategy:",
        "      matrix:",
        '        language: ["python"]',
        "    steps:",
        f"      - uses: actions/checkout@{CHECKOUT_SHA}",
        f"      - uses: github/codeql-action/init@{codeql_ref}",
        f"      - uses: github/codeql-action/analyze@{codeql_ref}",
    ]) + "\n"


def _scorecard_workflow_content(*, scorecard_ref: str = SCORECARD_SHA) -> str:
    return "\n".join([
        "name: Scorecard",
        "on:",
        "  push:",
        "    branches: [main]",
        "  schedule:",
        '    - cron: "19 4 * * 2"',
        "permissions:",
        "  contents: read",
        "  security-events: write",
        "jobs:",
        "  scorecard:",
        "    runs-on: ubuntu-latest",
        "    timeout-minutes: 30",
        "    steps:",
        f"      - uses: ossf/scorecard-action@{scorecard_ref}",
        "        with:",
        "          results_file: scorecard-results.json",
        "          results_format: json",
        "          publish_results: true",
        f"      - uses: actions/upload-artifact@{UPLOAD_ARTIFACT_SHA}",
    ]) + "\n"


def _release_workflow_content(
    *,
    attest_ref: str = ATTEST_BUILD_PROVENANCE_SHA,
    pypi_publish_ref: str = PYPI_PUBLISH_SHA,
) -> str:
    return "\n".join([
        "name: Release",
        "on:",
        "  workflow_dispatch:",
        "  push:",
        "    tags:",
        '      - "v*.*.*"',
        "permissions:",
        "  contents: read",
        "jobs:",
        "  build:",
        "    runs-on: ubuntu-latest",
        "    timeout-minutes: 60",
        "    permissions:",
        "      contents: read",
        "      attestations: write",
        "      id-token: write",
        "    steps:",
        f"      - uses: actions/checkout@{CHECKOUT_SHA}",
        f"      - uses: actions/setup-python@{SETUP_PYTHON_SHA}",
        "      - run: make release-check PYTHON=python",
        f"      - uses: actions/attest-build-provenance@{attest_ref}",
        "        with:",
        "          subject-path: |",
        "            dist/*.tar.gz",
        "            dist/*.whl",
        "            build/sbom/verarag-sbom.cdx.json",
        "            build/release-health/release-artifacts-manifest.json",
        "            build/release-checksums.json",
        f"      - uses: actions/upload-artifact@{UPLOAD_ARTIFACT_SHA}",
        "        with:",
        "          path: |",
        "            dist/*.tar.gz",
        "            dist/*.whl",
        "            build/sbom/verarag-sbom.cdx.json",
        "            build/release-health/release-artifacts-manifest.json",
        "            build/release-checksums.json",
        "  publish-pypi:",
        "    if: startsWith(github.ref, 'refs/tags/v')",
        "    environment: pypi",
        "    permissions:",
        "      contents: read",
        "      id-token: write",
        "    steps:",
        f"      - uses: actions/download-artifact@{DOWNLOAD_ARTIFACT_SHA}",
        f"      - uses: pypa/gh-action-pypi-publish@{pypi_publish_ref}",
        "  github-release:",
        "    if: startsWith(github.ref, 'refs/tags/v')",
        "    permissions:",
        "      contents: write",
        "    steps:",
        f"      - uses: actions/download-artifact@{DOWNLOAD_ARTIFACT_SHA}",
        "      - run: gh release create ${GITHUB_REF_NAME}",
    ]) + "\n"


def _workflow_content(*, checkout_ref: str = CHECKOUT_SHA) -> str:
    return "\n".join([
        "name: Tests",
        "on:",
        "  push:",
        "    branches: [main]",
        "permissions:",
        "  contents: read",
        "jobs:",
        "  test:",
        "    runs-on: ubuntu-latest",
        "    timeout-minutes: 45",
        "    steps:",
        f"      - uses: actions/checkout@{checkout_ref}",
        f"      - uses: actions/setup-python@{SETUP_PYTHON_SHA}",
        "      - run: python experiments/validate_version_identity.py",
        "      - run: python experiments/validate_python_support.py",
        "      - run: python experiments/validate_results.py",
        "      - run: python experiments/validate_project_metadata.py",
        "      - run: make package-check PYTHON=python",
    ]) + "\n"
