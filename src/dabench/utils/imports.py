"""Optional dependency checks."""

from __future__ import annotations


def require_datasets_for_loading():
    try:
        from datasets import ClassLabel, Dataset, DatasetDict, Image, concatenate_datasets  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for dataset loading. Install with `pip install -e .[data]`."
        ) from exc
    return Dataset, DatasetDict, Image, ClassLabel, concatenate_datasets


def require_torch_for_loading():
    try:
        import torch  # type: ignore
        from torch.utils.data import DataLoader, IterableDataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for dataset loading. Install it in the active environment."
        ) from exc
    return torch, DataLoader, IterableDataset


def require_requests():
    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "requests is required for direct URL dataset download. Install with `pip install -e .[data]`."
        ) from exc
    return requests
