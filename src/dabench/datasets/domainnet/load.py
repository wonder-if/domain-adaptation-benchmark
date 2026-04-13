"""Load DomainNet datasets for training and evaluation."""

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
    "c": "clipart",
    "i": "infograph",
    "p": "painting",
    "q": "quickdraw",
    "r": "real",
    "s": "sketch",
}
DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]


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
    return build_torch_dataset(dataset, transform=transform, domain_column="domain", path_column="image_path")


def load_uda(
    *,
    path: str | Path,
    source: Iterable[str],
    target: Iterable[str],
    train_transform=None,
    val_transform=None,
    decode: bool = True,
) -> UDABundle:
    dataset_dict = _dataset_dict(path, decode=decode)
    class_names = infer_class_names(dataset_dict["train"], "label")
    train_source = build_torch_dataset(
        filter_dataset(dataset_dict["train"], domains=_normalize_domains(source), domain_column="domain"),
        transform=train_transform,
        domain_column="domain",
        path_column="image_path",
    )
    train_target = build_torch_dataset(
        filter_dataset(dataset_dict["train"], domains=_normalize_domains(target), domain_column="domain"),
        transform=train_transform,
        domain_column="domain",
        path_column="image_path",
    )
    test_target = build_torch_dataset(
        filter_dataset(dataset_dict["test"], domains=_normalize_domains(target), domain_column="domain"),
        transform=val_transform,
        domain_column="domain",
        path_column="image_path",
    )
    return UDABundle(
        train_source=train_source,
        train_target=train_target,
        val=test_target,
        test=test_target,
        num_classes=len(class_names),
        class_names=class_names,
    )


def inspect_dataset(*, path: str | Path) -> dict[str, Any]:
    state = inspect_prepared_dataset(path)
    if state.get("has_dataset_info"):
        dataset_dict = _dataset_dict(path, decode=False)
        state["domains"] = infer_domain_names(dataset_dict["train"], "domain")
        state["num_classes"] = len(infer_class_names(dataset_dict["train"], "label"))
    return state


def list_domains(*, path: str | Path | None = None, split: str = "train") -> list[str]:
    if path is None:
        return DOMAINS.copy()
    return infer_domain_names(_dataset_dict(path, decode=False)[split], "domain")


def list_classes(*, path: str | Path, split: str = "train") -> list[str]:
    return infer_class_names(_dataset_dict(path, decode=False)[split], "label")
