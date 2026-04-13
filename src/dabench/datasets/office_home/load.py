"""Load Office-Home datasets for training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from dabench.datasets.common import (
    UDABundle,
    build_torch_dataset,
    filter_dataset,
    infer_class_names,
    infer_domain_names,
    load_prepared_dataset_dict,
)
from dabench.datasets.hf_download import inspect_prepared_dataset

DOMAIN_ALIASES = {
    "ar": "Art",
    "cl": "Clipart",
    "pr": "Product",
    "rw": "Real World",
}
DOMAINS = ["Art", "Clipart", "Product", "Real World"]


def _normalize_domains(domains: Iterable[str] | None) -> list[str] | None:
    if domains is None:
        return None
    return [DOMAIN_ALIASES.get(domain.strip().lower(), domain) for domain in domains]


def _dataset_dict(path: str | Path, *, decode: bool = True):
    return load_prepared_dataset_dict(path, decode=decode)


def load_hf_dataset(
    *,
    path: str | Path,
    split: str = "train",
    domains: Iterable[str] | None = None,
    labels: Iterable[int] | None = None,
    decode: bool = True,
) -> Any:
    dataset = _dataset_dict(path, decode=decode)[split]
    return filter_dataset(dataset, domains=_normalize_domains(domains), labels=labels, domain_column="domain")


def load_dataset(
    *,
    path: str | Path,
    split: str = "train",
    domains: Iterable[str] | None = None,
    labels: Iterable[int] | None = None,
    transform=None,
    decode: bool = True,
) -> Any:
    dataset = load_hf_dataset(path=path, split=split, domains=domains, labels=labels, decode=decode)
    return build_torch_dataset(dataset, transform=transform, domain_column="domain", path_column=None)


def load_uda(
    *,
    path: str | Path,
    source: Iterable[str],
    target: Iterable[str],
    train_transform=None,
    val_transform=None,
    decode: bool = True,
) -> UDABundle:
    dataset = _dataset_dict(path, decode=decode)["train"]
    class_names = infer_class_names(dataset, "label")
    train_source = build_torch_dataset(
        filter_dataset(dataset, domains=_normalize_domains(source), domain_column="domain"),
        transform=train_transform,
        domain_column="domain",
        path_column=None,
    )
    target_dataset = filter_dataset(dataset, domains=_normalize_domains(target), domain_column="domain")
    train_target = build_torch_dataset(
        target_dataset,
        transform=train_transform,
        domain_column="domain",
        path_column=None,
    )
    eval_target = build_torch_dataset(
        target_dataset,
        transform=val_transform,
        domain_column="domain",
        path_column=None,
    )
    return UDABundle(
        train_source=train_source,
        train_target=train_target,
        val=eval_target,
        test=eval_target,
        num_classes=len(class_names),
        class_names=class_names,
    )


def inspect_dataset(*, path: str | Path) -> dict[str, Any]:
    state = inspect_prepared_dataset(path)
    if state.get("has_dataset_info"):
        dataset = _dataset_dict(path, decode=False)["train"]
        state["domains"] = infer_domain_names(dataset, "domain")
        state["num_classes"] = len(infer_class_names(dataset, "label"))
    return state


def list_domains(*, path: str | Path | None = None, split: str = "train") -> list[str]:
    if path is None:
        return DOMAINS.copy()
    return infer_domain_names(_dataset_dict(path, decode=False)[split], "domain")


def list_classes(*, path: str | Path, split: str = "train") -> list[str]:
    return infer_class_names(_dataset_dict(path, decode=False)[split], "label")
