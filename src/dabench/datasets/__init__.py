"""Lightweight dataset entrypoints for dabench."""

from __future__ import annotations

from typing import Any

from dabench.datasets import camelyon17, domainnet, iwildcam, office31, office_home, visda2017
from dabench.datasets.common import UDABundle, build_torch_dataset, default_collate, make_paired_loader
from dabench.datasets.transforms import get_train_transform, get_val_transform

_DATASETS = {
    "camelyon17": camelyon17,
    "domainnet": domainnet,
    "iwildcam": iwildcam,
    "office-home": office_home,
    "office31": office31,
    "office-31": office31,
    "visda2017": visda2017,
    "visda-2017": visda2017,
}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _module(name: str):
    dataset_name = _normalize_name(name)
    if dataset_name not in _DATASETS:
        raise ValueError(f"Unsupported dataset: {name}")
    return _DATASETS[dataset_name]


def load_hf_dataset(name: str, **kwargs: Any):
    module = _module(name)
    if not hasattr(module, "load_hf_dataset"):
        raise ValueError(f"{name} does not expose load_hf_dataset")
    return module.load_hf_dataset(**kwargs)


def load_dataset(name: str, **kwargs: Any):
    return _module(name).load_dataset(**kwargs)


def load_uda(name: str, **kwargs: Any) -> UDABundle:
    return _module(name).load_uda(**kwargs)


def download_dataset(name: str, **kwargs: Any):
    return _module(name).download_dataset(**kwargs)


def inspect_dataset(name: str, **kwargs: Any) -> dict[str, Any]:
    return _module(name).inspect_dataset(**kwargs)


def list_domains(name: str, **kwargs: Any) -> list[str]:
    module = _module(name)
    if not hasattr(module, "list_domains"):
        return []
    return module.list_domains(**kwargs)


def list_classes(name: str, **kwargs: Any) -> list[str]:
    module = _module(name)
    if not hasattr(module, "list_classes"):
        return []
    return module.list_classes(**kwargs)


__all__ = [
    "UDABundle",
    "build_torch_dataset",
    "camelyon17",
    "default_collate",
    "domainnet",
    "download_dataset",
    "get_train_transform",
    "get_val_transform",
    "inspect_dataset",
    "iwildcam",
    "list_classes",
    "list_domains",
    "load_dataset",
    "load_hf_dataset",
    "load_uda",
    "make_paired_loader",
    "office31",
    "office_home",
    "visda2017",
]
