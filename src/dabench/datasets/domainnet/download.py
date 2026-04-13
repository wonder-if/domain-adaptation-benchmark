"""Download helpers for the Hugging Face DomainNet dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DATASET_NAME = "wltjr1007/DomainNet"
from dabench.datasets.hf_download import download_prepared_dataset, inspect_prepared_dataset


def download_dataset(
    *,
    dest: str | Path,
    source: str = "mirror",
    config: str | None = None,
    proxy: str = "disable",
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    num_proc: int | None = None,
    file_format: str = "arrow",
    verification_mode: str | None = None,
    force_redownload: bool = False,
) -> dict[str, Any]:
    return download_prepared_dataset(
        dataset_name=DATASET_NAME,
        dest=dest,
        source=source,
        proxy=proxy,
        config=config,
        cache_dir=cache_dir,
        revision=revision,
        token=token,
        num_proc=num_proc,
        file_format=file_format,
        verification_mode=verification_mode,
        force_redownload=force_redownload,
    )


def inspect_dataset(
    *,
    path: str | Path,
) -> dict[str, Any]:
    return inspect_prepared_dataset(path)
