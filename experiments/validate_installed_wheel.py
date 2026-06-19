#!/usr/bin/env python3
"""Install the built VeraRAG wheel into a temporary environment and smoke test it."""

from __future__ import annotations

import argparse
import json
import os
import site
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.validate_package_contents import (  # noqa: E402
    _distribution_filename_name,
    _parse_pyproject_project_metadata,
    _parse_pyproject_script_entries,
)


@dataclass(frozen=True)
class InstalledWheelAudit:
    wheel: str
    valid: bool
    checks: list[str]
    errors: list[str]
    console_scripts: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "wheel": self.wheel,
            "valid": self.valid,
            "checks": self.checks,
            "errors": self.errors,
            "console_scripts": self.console_scripts,
        }


def validate_installed_wheel(
    dist_dir: str | Path = "dist",
    project_root: str | Path = ".",
    *,
    timeout_seconds: int = 20,
    include_current_site_packages: bool = True,
) -> InstalledWheelAudit:
    root = Path(project_root).resolve()
    dist_path = Path(dist_dir).resolve()
    metadata = _parse_pyproject_project_metadata(root / "pyproject.toml")
    wheel = dist_path / (
        f"{_distribution_filename_name(metadata['name'])}-"
        f"{metadata['version']}-py3-none-any.whl"
    )
    if not wheel.is_file():
        raise FileNotFoundError(f"Expected built wheel {wheel}")

    console_scripts = sorted(_parse_pyproject_script_entries(root / "pyproject.toml"))
    checks: list[str] = []
    errors: list[str] = []

    with tempfile.TemporaryDirectory(prefix="verarag-wheel-smoke-") as tmp:
        tmp_path = Path(tmp)
        install_dir = tmp_path / "site-packages"
        work_dir = tmp_path / "work"
        install_dir.mkdir()
        work_dir.mkdir()
        env = _smoke_env(
            root,
            install_dir=install_dir,
            include_current_site_packages=include_current_site_packages,
        )
        install = _extract_wheel(wheel, install_dir)
        _record(install, checks, errors)

        if install.valid:
            import_check = _run(
                "import installed public API and packaged data",
                [
                    sys.executable,
                    "-c",
                    _IMPORT_SMOKE,
                    str(install_dir),
                ],
                cwd=work_dir,
                env=env,
                timeout_seconds=timeout_seconds,
            )
            _record(import_check, checks, errors)

            entry_points = _parse_pyproject_script_entries(root / "pyproject.toml")
            for script in console_scripts:
                script_check = _run(
                    f"{script} --help",
                    [
                        sys.executable,
                        "-c",
                        _ENTRY_POINT_SMOKE,
                        script,
                        entry_points[script],
                        "--help",
                    ],
                    cwd=work_dir,
                    env=env,
                    timeout_seconds=timeout_seconds,
                )
                _record(script_check, checks, errors)

    return InstalledWheelAudit(
        wheel=str(wheel),
        valid=not errors,
        checks=checks,
        errors=errors,
        console_scripts=console_scripts,
    )


@dataclass(frozen=True)
class _CommandResult:
    label: str
    valid: bool
    error: str | None = None


_IMPORT_SMOKE = r"""
import pathlib
import sys

install_dir = pathlib.Path(sys.argv[1]).resolve()

import verarag
from verarag import VeraRAG
from src.benchmark import load_verabench
from web.app import create_app

origin = pathlib.Path(verarag.__file__).resolve()
if install_dir not in origin.parents:
    raise SystemExit(f"verarag imported from {origin}, expected under {install_dir}")

benchmark = load_verabench()
if len(benchmark.questions) != 152:
    raise SystemExit(f"expected 152 VeraBench questions, found {len(benchmark.questions)}")

app = create_app(db_path="smoke.db")
if app.title != "VeraRAG":
    raise SystemExit(f"unexpected app title {app.title!r}")

print(verarag.__version__, VeraRAG.__name__, len(benchmark.questions), app.title)
"""


_ENTRY_POINT_SMOKE = r"""
import importlib
import inspect
import sys

script_name = sys.argv[1]
target = sys.argv[2]
script_args = sys.argv[3:]
module_name, attr_name = target.split(":", 1)
attr_name = attr_name.split("[", 1)[0]
module = importlib.import_module(module_name)
func = module
for part in attr_name.split("."):
    func = getattr(func, part)

sys.argv = [script_name, *script_args]
try:
    if len(inspect.signature(func).parameters) == 0:
        func()
    else:
        func(script_args)
except SystemExit as exc:
    if exc.code not in (0, None):
        raise
"""


def _record(result: _CommandResult, checks: list[str], errors: list[str]) -> None:
    checks.append(result.label)
    if result.error:
        errors.append(result.error)


def _extract_wheel(wheel: Path, install_dir: Path) -> _CommandResult:
    try:
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                destination = (install_dir / member.filename).resolve()
                if install_dir.resolve() not in destination.parents and destination != install_dir:
                    return _CommandResult(
                        "extract wheel",
                        False,
                        f"wheel member escapes install directory: {member.filename}",
                    )
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
    except zipfile.BadZipFile as exc:
        return _CommandResult("extract wheel", False, f"invalid wheel zip: {exc}")
    return _CommandResult("extract wheel", True)


def _run(
    label: str,
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int,
) -> _CommandResult:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        return _CommandResult(label, False, f"{label} failed: {exc}")
    except subprocess.TimeoutExpired as exc:
        return _CommandResult(label, False, f"{label} timed out after {exc.timeout}s")

    if completed.returncode == 0:
        return _CommandResult(label, True)
    output = _tail("\n".join([completed.stdout, completed.stderr]).strip())
    return _CommandResult(
        label,
        False,
        f"{label} exited {completed.returncode}: {output}",
    )


def _smoke_env(
    project_root: Path,
    *,
    install_dir: Path,
    include_current_site_packages: bool,
) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    paths = [install_dir.resolve()]
    if include_current_site_packages:
        paths.extend(_current_dependency_paths(project_root))
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in paths)
    return env


def _current_dependency_paths(project_root: Path) -> list[Path]:
    candidates = [Path(path).resolve() for path in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(Path(user_site).resolve())
    paths = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate == project_root or project_root in candidate.parents:
            continue
        if candidate not in paths:
            paths.append(candidate)
    return paths


def _tail(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return "..." + text[-limit:]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        default="dist",
        help="Directory containing the built VeraRAG wheel.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Source checkout root containing pyproject.toml.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Per-command timeout in seconds.",
    )
    parser.add_argument(
        "--no-current-site-packages",
        action="store_true",
        help="Do not expose the current interpreter's installed dependencies.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON audit output.")
    args = parser.parse_args(argv)

    audit = validate_installed_wheel(
        args.dist_dir,
        args.project_root,
        timeout_seconds=args.timeout,
        include_current_site_packages=not args.no_current_site_packages,
    )
    if args.json:
        print(json.dumps(audit.to_dict(), indent=2))
    elif audit.valid:
        print(
            "Installed wheel smoke validated: "
            f"{len(audit.console_scripts)} console scripts and packaged data."
        )
    else:
        print("Installed wheel smoke validation failed:")
        for error in audit.errors:
            print(f"- {error}")

    if not audit.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
