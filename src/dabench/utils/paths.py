"""Filesystem path helpers."""

from __future__ import annotations

from pathlib import Path


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def ensure_dir(path: str | Path) -> Path:
    target = expand_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
