"""Download helpers for the Hugging Face iWildCam dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dabench.datasets.hf_download import download_prepared_dataset

DATASET_NAME = "anngrosha/iWildCam2020"


def download_dataset(
    *,
    dest: str | Path,
    source: str = "mirror",
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
        cache_dir=cache_dir,
        revision=revision,
        token=token,
        num_proc=num_proc,
        file_format=file_format,
        verification_mode=verification_mode,
        force_redownload=force_redownload,
    )
