"""Small filesystem helpers shared across the package."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
