#!/usr/bin/env python3
"""Generate and validate a CycloneDX dependency SBOM for VeraRAG."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

CYCLONEDX_SPEC_VERSION = "1.5"
DEFAULT_OUTPUT = "build/sbom/verarag-sbom.cdx.json"


@dataclass(frozen=True)
class SbomComponent:
    name: str
    requirement: str
    scope: str
    groups: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        properties = [
            {"name": "verarag:requirement", "value": self.requirement},
            {"name": "verarag:dependency-groups", "value": ",".join(self.groups)},
        ]
        return {
            "type": "library",
            "bom-ref": f"pkg:pypi/{self.name}",
            "name": self.name,
            "scope": self.scope,
            "purl": f"pkg:pypi/{self.name}",
            "properties": properties,
        }


@dataclass(frozen=True)
class SbomAudit:
    valid: bool
    errors: list[str]
    output_path: str
    component_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "output_path": self.output_path,
            "component_count": self.component_count,
        }


def build_sbom(project_root: str | Path = ".") -> dict[str, object]:
    root = Path(project_root)
    metadata = _parse_pyproject(root / "pyproject.toml")
    project = metadata["project"]
    project_name = str(project["name"])
    project_version = str(project["version"])
    components = _dependency_components(
        project.get("dependencies", []),
        metadata.get("project.optional-dependencies", {}),
    )
    serial = uuid5(
        NAMESPACE_URL,
        f"{metadata['repository-url']}#{project_name}@{project_version}",
    )
    return {
        "bomFormat": "CycloneDX",
        "specVersion": CYCLONEDX_SPEC_VERSION,
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "bom-ref": f"pkg:pypi/{project_name}@{project_version}",
                "name": project_name,
                "version": project_version,
                "purl": f"pkg:pypi/{project_name}@{project_version}",
            },
            "properties": [
                {"name": "verarag:repository", "value": metadata["repository-url"]},
                {"name": "verarag:sbom-generator", "value": "experiments.generate_sbom"},
            ],
        },
        "components": [component.to_dict() for component in components],
    }


def write_sbom(sbom: dict[str, object], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_sbom(
    output_path: str | Path = DEFAULT_OUTPUT,
    project_root: str | Path = ".",
) -> SbomAudit:
    path = Path(output_path)
    errors: list[str] = []
    try:
        sbom = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return SbomAudit(False, [f"SBOM file does not exist: {path}"], str(path), 0)
    except json.JSONDecodeError as exc:
        return SbomAudit(False, [f"SBOM file is not valid JSON: {exc}"], str(path), 0)

    expected = build_sbom(project_root)
    components = sbom.get("components", [])
    expected_components = expected.get("components", [])
    if sbom.get("bomFormat") != "CycloneDX":
        errors.append("SBOM bomFormat must be CycloneDX")
    if sbom.get("specVersion") != CYCLONEDX_SPEC_VERSION:
        errors.append(f"SBOM specVersion must be {CYCLONEDX_SPEC_VERSION}")
    if sbom.get("metadata", {}).get("component") != expected.get("metadata", {}).get("component"):
        errors.append("SBOM metadata component does not match pyproject project identity")
    if not isinstance(components, list) or not components:
        errors.append("SBOM must include dependency components")
    elif components != expected_components:
        errors.append("SBOM dependency components are stale or incomplete")
    component_refs = [
        str(component.get("bom-ref", ""))
        for component in components
        if isinstance(component, dict)
    ]
    if len(component_refs) != len(set(component_refs)):
        errors.append("SBOM dependency components must have unique bom-ref values")
    return SbomAudit(
        valid=not errors,
        errors=errors,
        output_path=str(path),
        component_count=len(components) if isinstance(components, list) else 0,
    )


def _dependency_components(
    project_dependencies: object,
    optional_dependencies: object,
) -> list[SbomComponent]:
    component_data: dict[str, tuple[str, set[str], str]] = {}
    if isinstance(project_dependencies, list):
        for requirement in project_dependencies:
            _add_requirement(component_data, str(requirement), "required", "default")
    if isinstance(optional_dependencies, dict):
        for group, requirements in optional_dependencies.items():
            if group == "all" or not isinstance(requirements, list):
                continue
            for requirement in requirements:
                _add_requirement(component_data, str(requirement), "optional", str(group))

    components = []
    for name, (requirement, groups, scope) in sorted(component_data.items()):
        components.append(
            SbomComponent(
                name=name,
                requirement=requirement,
                scope=scope,
                groups=tuple(sorted(groups)),
            )
        )
    return components


def _add_requirement(
    component_data: dict[str, tuple[str, set[str], str]],
    requirement: str,
    scope: str,
    group: str,
) -> None:
    name = _requirement_name(requirement)
    if not name:
        return
    if name not in component_data:
        component_data[name] = (requirement, {group}, scope)
        return
    existing_requirement, groups, existing_scope = component_data[name]
    groups.add(group)
    merged_scope = "required" if "required" in {scope, existing_scope} else "optional"
    component_data[name] = (existing_requirement, groups, merged_scope)


def _parse_pyproject(path: Path) -> dict[str, object]:
    metadata: dict[str, object] = {
        "project": {},
        "project.optional-dependencies": {},
        "repository-url": "",
    }
    section = ""
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]")
            i += 1
            continue
        if section == "project" and "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            project = metadata["project"]
            if not isinstance(project, dict):
                raise ValueError("invalid project metadata shape")
            if key in {"name", "version"}:
                project[key] = _strip_toml_string(value)
            elif key == "dependencies":
                array, i = _collect_toml_array(lines, i)
                project[key] = _literal_string_list(array)
                continue
        elif section == "project.optional-dependencies" and "=" in line:
            key = line.split("=", 1)[0].strip()
            array, i = _collect_toml_array(lines, i)
            optional = metadata["project.optional-dependencies"]
            if not isinstance(optional, dict):
                raise ValueError("invalid optional dependency metadata shape")
            optional[key] = _literal_string_list(array)
            continue
        elif section == "project.urls" and line.startswith("Repository"):
            metadata["repository-url"] = _strip_toml_string(line.split("=", 1)[1].strip())
        i += 1

    project = metadata["project"]
    if not isinstance(project, dict) or not project.get("name") or not project.get("version"):
        raise ValueError("pyproject [project] missing name or version")
    if not metadata["repository-url"]:
        raise ValueError("pyproject [project.urls] missing Repository")
    return metadata


def _collect_toml_array(lines: list[str], start: int) -> tuple[str, int]:
    collected = [lines[start].split("=", 1)[1].strip()]
    i = start + 1
    while not _toml_array_complete("\n".join(collected)) and i < len(lines):
        collected.append(lines[i].strip())
        i += 1
    return "\n".join(collected), i


def _toml_array_complete(value: str) -> bool:
    depth = 0
    quote: str | None = None
    escaped = False
    saw_array = False
    for char in value:
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char == "[":
            saw_array = True
            depth += 1
        elif char == "]":
            depth -= 1
    return saw_array and depth == 0 and quote is None


def _literal_string_list(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected TOML array, found {value}")
    return [str(item) for item in parsed]


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    if not match:
        return ""
    return match.group(1).lower().replace("_", "-")


def _strip_toml_string(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=".",
        help="Source checkout root containing pyproject.toml.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="CycloneDX JSON output path.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the generated SBOM after writing it.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON audit output.")
    args = parser.parse_args(argv)

    path = write_sbom(build_sbom(args.project_root), args.output)
    audit = validate_sbom(path, args.project_root)
    if args.json:
        print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
    elif audit.valid:
        print(f"SBOM validated: {audit.component_count} components at {audit.output_path}.")
    else:
        print("SBOM validation failed:")
        for error in audit.errors:
            print(f"- {error}")
    if args.check and not audit.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
