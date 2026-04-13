"""Load Camelyon17 datasets for training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dabench.datasets.common import (
    UDABundle,
    build_torch_dataset,
    infer_class_names,
    load_prepared_dataset_dict,
)
from dabench.datasets.hf_download import inspect_prepared_dataset

SPLIT_ALIASES = {
    "train": "id_train",
    "id_train": "id_train",
    "id_val": "id_val",
    "val": "id_val",
    "validation": "id_val",
    "unlabeled": "unlabeled_train",
    "unlabeled_train": "unlabeled_train",
    "target_train": "unlabeled_train",
    "ood_val": "ood_val",
    "eval": "ood_val",
    "ood_test": "ood_test",
    "test": "ood_test",
}


def _resolve_split(split: str) -> str:
    return SPLIT_ALIASES.get(split, split)


def _dataset_dict(path: str | Path, *, decode: bool = True):
    return load_prepared_dataset_dict(path, decode=decode)


def load_hf_dataset(
    *,
    path: str | Path,
    split: str = "id_train",
    decode: bool = True,
) -> Any:
    return _dataset_dict(path, decode=decode)[_resolve_split(split)]


def load_dataset(
    *,
    path: str | Path,
    split: str = "id_train",
    transform=None,
    decode: bool = True,
) -> Any:
    dataset = load_hf_dataset(path=path, split=split, decode=decode)
    return build_torch_dataset(dataset, transform=transform, domain_column=None, path_column=None)


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
    class_names = infer_class_names(dataset_dict["id_train"], "label")
    return UDABundle(
        train_source=build_torch_dataset(dataset_dict["id_train"], transform=train_transform, domain_column=None, path_column=None),
        train_target=build_torch_dataset(dataset_dict["unlabeled_train"], transform=train_transform, domain_column=None, path_column=None),
        val=build_torch_dataset(dataset_dict["ood_val"], transform=val_transform, domain_column=None, path_column=None),
        test=build_torch_dataset(dataset_dict["ood_test"], transform=val_transform, domain_column=None, path_column=None),
        num_classes=len(class_names),
        class_names=class_names,
    )


def inspect_dataset(*, path: str | Path) -> dict[str, Any]:
    state = inspect_prepared_dataset(path)
    if state.get("has_dataset_info"):
        dataset = _dataset_dict(path, decode=False)["id_train"]
        state["num_classes"] = len(infer_class_names(dataset, "label"))
    return state


def list_domains(*, path: str | Path | None = None, split: str = "ood_test") -> list[str]:
    return []


def list_classes(*, path: str | Path, split: str = "id_train") -> list[str]:
    return infer_class_names(_dataset_dict(path, decode=False)[_resolve_split(split)], "label")
