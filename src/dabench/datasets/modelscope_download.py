"""Shared helpers for downloading datasets from ModelScope."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dabench.datasets.hf_download import proxy_env
from dabench.io import ensure_dir


def require_modelscope():
    try:
        from modelscope.msdatasets import MsDataset  # type: ignore
        from modelscope.utils.constant import Hubs  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "modelscope is required for ModelScope dataset download. Install with `pip install -e .[data]`."
        ) from exc
    return MsDataset, Hubs


def download_modelscope_dataset(
    *,
    dataset_name: str,
    dest: str | Path,
    namespace: str | None = None,
    version: str | None = "master",
    subset_name: str | None = None,
    split: str | None = None,
    data_dir: str | None = None,
    data_files: Any = None,
    cache_dir: str | Path | None = None,
    token: str | None = None,
    use_streaming: bool = False,
    dataset_info_only: bool = False,
    trust_remote_code: bool = False,
    proxy: str = "disable",
    download_mode: str = "reuse_dataset_if_exists",
    **config_kwargs: Any,
) -> dict[str, Any]:
    destination = ensure_dir(dest)
    resolved_cache_dir = (
        Path(cache_dir).expanduser().resolve() if cache_dir else destination
    )

    with proxy_env(proxy):
        MsDataset, Hubs = require_modelscope()
        dataset = MsDataset.load(
            dataset_name,
            namespace=namespace or "modelscope",
            version=version,
            hub=Hubs.modelscope,
            subset_name=subset_name,
            split=split,
            data_dir=data_dir,
            data_files=data_files,
            download_mode=download_mode,
            cache_dir=str(resolved_cache_dir),
            use_streaming=use_streaming,
            token=token,
            dataset_info_only=dataset_info_only,
            trust_remote_code=trust_remote_code,
            **config_kwargs,
        )

    result: dict[str, Any] = {
        "dataset": dataset_name,
        "namespace": namespace or "modelscope",
        "dest": str(destination),
        "cache_dir": str(resolved_cache_dir),
        "version": version,
        "subset_name": subset_name,
        "split": split,
        "download_mode": download_mode,
    }
    if isinstance(dataset, dict):
        result["splits"] = sorted(dataset.keys())
    else:
        result["dataset_type"] = type(dataset).__name__
    return result


def inspect_modelscope_dataset(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    hub_dir = target / "hub" / "datasets"
    snapshots = sorted(str(p.relative_to(target)) for p in hub_dir.glob("*/*"))
    return {
        "path": str(target),
        "exists": target.exists(),
        "hub_dir": str(hub_dir),
        "has_hub_dir": hub_dir.is_dir(),
        "snapshots": snapshots,
    }
