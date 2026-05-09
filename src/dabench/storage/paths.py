"""Local dataset path configuration."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

from dabench.utils.paths import ensure_dir, expand_path


def _config_dir() -> Path:
    configured = os.environ.get("DABENCH_CONFIG_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "config"


def _paths_file() -> Path:
    configured = os.environ.get("DABENCH_PATHS_FILE")
    if configured:
        return Path(configured).expanduser().resolve()
    return _config_dir() / "paths.json"


def _read_payload() -> dict[str, Any]:
    path = _paths_file()
    if not path.is_file():
        return {"datasets": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid dataset path config: {path}")
    return payload


def _write_payload(payload: dict[str, Any]) -> Path:
    path = _paths_file()
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _dataset_map(payload: dict[str, Any]) -> dict[str, Any]:
    datasets = payload.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError("Invalid dataset path config: `datasets` must be a mapping.")
    return datasets


def get_dataset_entry(name: str) -> Any:
    payload = _read_payload()
    entry = _dataset_map(payload).get(_normalize_name(name))
    return entry


def _entry_path(entry: Any) -> Path | None:
    if entry is None:
        return None
    if isinstance(entry, str):
        return expand_path(entry)
    if isinstance(entry, dict):
        value = entry.get("path")
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Invalid dataset path config: `path` must be a string.")
        return expand_path(value)
    raise ValueError("Invalid dataset path config entry.")


def get_dataset_path(name: str) -> Path | None:
    entry = get_dataset_entry(name)
    return _entry_path(entry)


def get_dataset_field(name: str, field: str) -> Any:
    entry = get_dataset_entry(name)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get(field)
    return None


def get_dataset_field_path(name: str, field: str) -> Path | None:
    value = get_dataset_field(name, field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Invalid dataset path config: field {field!r} for dataset {name!r} must be a string.")
    return expand_path(value)


def set_dataset_path(name: str, path: str | Path) -> Path:
    payload = _read_payload()
    resolved = expand_path(path)
    datasets = _dataset_map(payload)
    datasets[_normalize_name(name)] = {"path": str(resolved)}
    payload["datasets"] = datasets
    _write_payload(payload)
    return resolved


def list_dataset_paths() -> dict[str, Path]:
    payload = _read_payload()
    return {
        name: path
        for name, path in (
            (dataset_name, _entry_path(entry))
            for dataset_name, entry in _dataset_map(payload).items()
        )
        if path is not None
    }


def resolve_dataset_path(name: str, path: str | Path | None = None) -> Path:
    if path is not None:
        return expand_path(path)
    resolved = get_dataset_path(name)
    if resolved is not None:
        return resolved
    raise FileNotFoundError(
        f"No local path configured for dataset {name!r}. "
        f"Set one in {_paths_file()} or pass `path` explicitly."
    )


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")
