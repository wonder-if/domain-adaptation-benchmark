"""UniDA setting helpers."""

from __future__ import annotations

from pathlib import Path

from dabench.data import load_unida as _load_unida


def load_unida(
    *,
    dataset: str,
    task: str,
    shared: int,
    source_private: int,
    target_private: int,
    format: str = "hf",
    feature_model: str | None = None,
    feature_cache_path: str | Path | None = None,
    source_train_batch_size: int,
    target_train_batch_size: int | None = None,
    test_batch_size: int | None = None,
    source_train_transform=None,
    target_train_transform=None,
    test_transform=None,
    num_workers: int = 4,
    pin_memory: bool | None = None,
    decode: bool = True,
):
    if format not in {"hf", "feature"}:
        raise ValueError("UniDA setting currently supports format='hf' or format='feature'.")
    return _load_unida(
        dataset=dataset,
        task=task,
        shared=shared,
        source_private=source_private,
        target_private=target_private,
        format=format,
        feature_model=feature_model,
        feature_cache_path=feature_cache_path,
        source_batch_size=source_train_batch_size,
        target_batch_size=target_train_batch_size,
        test_batch_size=test_batch_size,
        source_transform=source_train_transform,
        target_transform=target_train_transform,
        test_transform=test_transform,
        num_workers=num_workers,
        pin_memory=pin_memory,
        decode=decode,
    )
