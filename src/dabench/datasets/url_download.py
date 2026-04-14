"""Shared helpers for datasets distributed as direct URLs."""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any

from dabench.datasets.hf_download import proxy_env
from dabench.io import ensure_dir


def require_requests():
    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "requests is required for direct URL dataset download. Install with `pip install -e .[data]`."
        ) from exc
    return requests


def download_file(
    *,
    url: str,
    dest: str | Path,
    proxy: str = "disable",
    chunk_size: int = 1024 * 1024,
    force: bool = False,
) -> dict[str, Any]:
    target = Path(dest).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        return {"url": url, "path": str(target), "size": target.stat().st_size, "skipped": True}

    requests = require_requests()
    with proxy_env(proxy):
        with requests.get(url, stream=True, timeout=(30, 600)) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        handle.write(chunk)

    return {"url": url, "path": str(target), "size": target.stat().st_size, "skipped": False}


def extract_tar(
    *,
    archive_path: str | Path,
    dest: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    archive = Path(archive_path).expanduser().resolve()
    output_dir = ensure_dir(dest)
    with tarfile.open(archive, "r") as handle:
        members = handle.getmembers()
        top_level_names = {
            Path(member.name).parts[0]
            for member in members
            if member.name and Path(member.name).parts
        }
        if top_level_names and not force:
            extracted_targets = [output_dir / name for name in top_level_names]
            if all(target.exists() and (not target.is_dir() or any(target.iterdir())) for target in extracted_targets):
                return {
                    "archive": str(archive),
                    "dest": str(output_dir),
                    "members": len(members),
                    "skipped": True,
                }
        handle.extractall(output_dir)
    return {
        "archive": str(archive),
        "dest": str(output_dir),
        "members": len(members),
        "skipped": False,
    }
