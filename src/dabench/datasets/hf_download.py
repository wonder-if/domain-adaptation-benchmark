"""Shared helpers for downloading Hugging Face datasets via `datasets`."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dabench.io import ensure_dir

HF_ENDPOINTS = {
    "hf": "https://huggingface.co",
    "mirror": "https://hf-mirror.com",
}

PROXY_KEYS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
)


def require_datasets():
    try:
        from datasets import DownloadConfig, DownloadMode, load_dataset_builder  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for Hugging Face dataset download. Install with `pip install -e .[data]`."
        ) from exc
    return load_dataset_builder, DownloadConfig, DownloadMode


@contextmanager
def proxy_env(mode: str):
    if mode not in {"keep", "disable"}:
        raise ValueError("proxy must be one of: keep, disable")

    if mode == "keep":
        yield
        return

    original = {key: os.environ.get(key) for key in PROXY_KEYS}
    for key in PROXY_KEYS:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def hf_source_env(source: str):
    if source not in HF_ENDPOINTS:
        raise ValueError("source must be one of: hf, mirror")

    original = os.environ.get("HF_ENDPOINT")
    os.environ["HF_ENDPOINT"] = HF_ENDPOINTS[source]
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = original


def builder_kwargs(
    *,
    config: str | None,
    cache_dir: str | Path | None,
    revision: str | None,
    token: str | bool | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config:
        kwargs["name"] = config
    if cache_dir:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
    if revision:
        kwargs["revision"] = revision
    if token is not None:
        kwargs["token"] = token
    return kwargs


def download_prepared_dataset(
    *,
    dataset_name: str,
    dest: str | Path,
    source: str = "mirror",
    proxy: str = "disable",
    config: str | None = None,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    num_proc: int | None = None,
    file_format: str = "arrow",
    verification_mode: str | None = None,
    force_redownload: bool = False,
) -> dict[str, Any]:
    destination = ensure_dir(dest)
    resolved_cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None

    with proxy_env(proxy):
        with hf_source_env(source):
            load_dataset_builder, DownloadConfig, DownloadMode = require_datasets()
            builder = load_dataset_builder(
                dataset_name,
                download_config=DownloadConfig(),
                **builder_kwargs(
                    config=config,
                    cache_dir=resolved_cache_dir,
                    revision=revision,
                    token=token,
                ),
            )
            builder.download_and_prepare(
                output_dir=str(destination),
                file_format=file_format,
                num_proc=num_proc,
                verification_mode=verification_mode,
                download_mode=DownloadMode.FORCE_REDOWNLOAD if force_redownload else None,
            )

    split_names = sorted(builder.info.splits.keys()) if builder.info.splits else []
    return {
        "dataset": dataset_name,
        "source": source,
        "config": builder.config.name,
        "dest": str(destination),
        "cache_dir": str(resolved_cache_dir) if resolved_cache_dir else None,
        "splits": split_names,
        "file_format": file_format,
    }


def inspect_prepared_dataset(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    dataset_info = target / "dataset_info.json"
    state = {
        "path": str(target),
        "exists": target.exists(),
        "dataset_info": str(dataset_info),
        "has_dataset_info": dataset_info.is_file(),
        "splits": None,
        "features": None,
    }
    if dataset_info.is_file():
        with dataset_info.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        splits = payload.get("splits") or {}
        state["splits"] = sorted(splits.keys()) if isinstance(splits, dict) else splits
        state["features"] = payload.get("features")
    return state
