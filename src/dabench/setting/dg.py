"""Domain generalization setting helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from dabench.data import build_loader, load_view
from dabench.data.dataset import build_torch_dataset
from dabench.setting.uda import _role_views as _uda_role_views
from dabench.utils.imports import require_datasets_for_loading


def _source_view_spec(dataset: str, domain: str | int) -> dict[str, str | int | None]:
    views = _uda_role_views(dataset, source_domain=domain, target_domain=domain)
    return views["source_train"]


def _target_eval_specs(dataset: str, domain: str | int) -> tuple[dict[str, str | int | None], dict[str, str | int | None]]:
    views = _uda_role_views(dataset, source_domain=domain, target_domain=domain)
    return views["val"], views["test"]


def _load_hf_view(dataset: str, domain: str | int, split: str | None, *, decode: bool):
    return load_view(dataset, domain=domain, split=split, format="hf", decode=decode)


def _maybe_to_torch(dataset, *, format: Literal["hf", "torch"]):
    if format == "hf":
        return dataset
    if format == "torch":
        return build_torch_dataset(dataset, domain_column="domain", path_column="image_path")
    raise ValueError("format must be one of: hf, torch")


def load_dg(
    *,
    dataset: str,
    source_domains: Sequence[str | int],
    target_domain: str | int,
    format: Literal["hf", "torch"] = "hf",
    source_train_batch_size: int,
    val_batch_size: int | None = None,
    test_batch_size: int | None = None,
    source_train_transform=None,
    val_transform=None,
    test_transform=None,
    num_workers: int = 4,
    pin_memory: bool | None = None,
    decode: bool = True,
) -> tuple[Any, Any, Any]:
    if isinstance(source_domains, (str, bytes)):
        source_domains = (source_domains,)
    else:
        source_domains = tuple(source_domains)
    if not source_domains:
        raise ValueError("load_dg requires at least one source domain.")

    _, _, _, _, concatenate_datasets = require_datasets_for_loading()

    source_views = [
        _load_hf_view(dataset, domain, _source_view_spec(dataset, domain)["split"], decode=decode)
        for domain in source_domains
    ]
    source_dataset = source_views[0] if len(source_views) == 1 else concatenate_datasets(source_views)

    val_spec, test_spec = _target_eval_specs(dataset, target_domain)
    val_dataset = _load_hf_view(dataset, target_domain, val_spec["split"], decode=decode)
    test_dataset = _load_hf_view(dataset, target_domain, test_spec["split"], decode=decode)

    source_dataset = _maybe_to_torch(source_dataset, format=format)
    val_dataset = _maybe_to_torch(val_dataset, format=format)
    test_dataset = _maybe_to_torch(test_dataset, format=format)

    train_loader = build_loader(
        source_dataset,
        batch_size=source_train_batch_size,
        mode="train",
        transform=source_train_transform,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=val_batch_size or source_train_batch_size,
        mode="test",
        transform=val_transform,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = build_loader(
        test_dataset,
        batch_size=test_batch_size or val_batch_size or source_train_batch_size,
        mode="test",
        transform=test_transform,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


__all__ = ["load_dg"]
