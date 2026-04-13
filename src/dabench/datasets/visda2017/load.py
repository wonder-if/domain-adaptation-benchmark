"""Load VisDA-2017 datasets for training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dabench.datasets.common import (
    UDABundle,
    build_torch_dataset,
    infer_class_names,
    infer_domain_names,
    load_visda_dataset_dict,
)

SPLIT_ALIASES = {
    "source": "train",
    "target": "validation",
    "val": "validation",
    "eval": "validation",
    "test": "test",
}
DOMAINS = ["train", "validation", "test"]


def _resolve_split(split: str) -> str:
    return SPLIT_ALIASES.get(split, split)


def _dataset_dict(path: str | Path, *, decode: bool = True):
    return load_visda_dataset_dict(path, decode=decode)


def load_hf_dataset(
    *,
    path: str | Path,
    split: str = "train",
    decode: bool = True,
) -> Any:
    return _dataset_dict(path, decode=decode)[_resolve_split(split)]


def load_dataset(
    *,
    path: str | Path,
    split: str = "train",
    transform=None,
    decode: bool = True,
) -> Any:
    dataset = load_hf_dataset(path=path, split=split, decode=decode)
    return build_torch_dataset(dataset, transform=transform, domain_column="domain", path_column="image_path")


def load_uda(
    *,
    path: str | Path,
    source=None,
    target=None,
    train_transform=None,
    val_transform=None,
    decode: bool = True,
) -> UDABundle:
    dataset_dict = _dataset_dict(path, decode=decode)
    class_names = infer_class_names(dataset_dict["train"], "label")
    val_dataset = build_torch_dataset(dataset_dict["validation"], transform=val_transform, domain_column="domain", path_column="image_path")
    return UDABundle(
        train_source=build_torch_dataset(dataset_dict["train"], transform=train_transform, domain_column="domain", path_column="image_path"),
        train_target=build_torch_dataset(dataset_dict["validation"], transform=train_transform, domain_column="domain", path_column="image_path"),
        val=val_dataset,
        test=val_dataset,
        num_classes=len(class_names),
        class_names=class_names,
    )


def inspect_dataset(*, path: str | Path) -> dict[str, Any]:
    dataset_dict = _dataset_dict(path, decode=False)
    return {
        "path": str(Path(path).expanduser().resolve()),
        "exists": True,
        "splits": list(dataset_dict.keys()),
        "domains": {split: infer_domain_names(dataset, "domain") for split, dataset in dataset_dict.items()},
        "num_classes": len(infer_class_names(dataset_dict["train"], "label")) if "train" in dataset_dict else 0,
    }


def list_domains(*, path: str | Path | None = None, split: str = "validation") -> list[str]:
    if path is None:
        return DOMAINS.copy()
    return infer_domain_names(_dataset_dict(path, decode=False)[_resolve_split(split)], "domain")


def list_classes(*, path: str | Path, split: str = "train") -> list[str]:
    return infer_class_names(_dataset_dict(path, decode=False)[_resolve_split(split)], "label")
