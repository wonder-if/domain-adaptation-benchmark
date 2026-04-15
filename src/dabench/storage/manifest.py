"""Manifest loading for storage preparation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _manifest_dir() -> Path:
    configured = os.environ.get("DABENCH_MANIFEST_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "manifests"


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return {}
    if value in {"null", "None"}:
        return None
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def _parse_yaml_subset(text: str, *, source: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2:
            raise ValueError(f"{source}:{line_number}: indentation must use multiples of two spaces.")
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"{source}:{line_number}: expected `key: value`.")
        key, value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{source}:{line_number}: empty key is not allowed.")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"{source}:{line_number}: invalid indentation.")
        parent = stack[-1][1]
        parsed_value = _parse_scalar(value)
        parent[key] = parsed_value
        if isinstance(parsed_value, dict):
            stack.append((indent, parsed_value))
    return root


def _validate_manifest(payload: dict[str, Any], *, source: str) -> dict[str, Any]:
    dataset_id = payload.get("id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError(f"Manifest {source} must define a non-empty string `id`.")
    aliases = payload.get("aliases", [])
    if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
        raise ValueError(f"Manifest {source} field `aliases` must be a list of strings.")
    storage = payload.get("storage")
    if not isinstance(storage, dict):
        raise ValueError(f"Manifest {source} must define a mapping `storage`.")
    backend = storage.get("backend")
    if backend not in {"hf", "ms", "other"}:
        raise ValueError(f"Manifest {source} storage.backend must be one of: hf, ms, other.")
    prepared = payload.get("prepared", {})
    if not isinstance(prepared, dict):
        raise ValueError(f"Manifest {source} field `prepared` must be a mapping.")
    payload["aliases"] = list(dict.fromkeys([dataset_id, *aliases]))
    return payload


def load_manifest_file(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    payload = _parse_yaml_subset(manifest_path.read_text(encoding="utf-8"), source=str(manifest_path))
    return _validate_manifest(payload, source=str(manifest_path))


def list_manifests(manifest_dir: str | Path | None = None) -> list[dict[str, Any]]:
    root = Path(manifest_dir).expanduser().resolve() if manifest_dir else _manifest_dir()
    if not root.is_dir():
        raise FileNotFoundError(f"Manifest directory not found: {root}")
    manifests = [load_manifest_file(path) for path in sorted(root.glob("*.yaml"))]
    if not manifests:
        raise FileNotFoundError(f"No dataset manifests found under: {root}")
    return manifests


def get_manifest(name: str, manifest_dir: str | Path | None = None) -> dict[str, Any]:
    normalized = name.strip().lower().replace("_", "-")
    for manifest in list_manifests(manifest_dir):
        aliases = {alias.strip().lower().replace("_", "-") for alias in manifest["aliases"]}
        if normalized in aliases:
            return manifest
    available = ", ".join(manifest["id"] for manifest in list_manifests(manifest_dir))
    raise ValueError(f"Unsupported dataset manifest: {name}. Available datasets: {available}")
