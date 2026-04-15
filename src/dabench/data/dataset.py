"""Generic dataset loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dabench.data.base import DomainDatasetView
from dabench.data.common import build_torch_dataset, load_folder_domain_dataset, load_prepared_dataset_dict, load_visda_dataset_dict
from dabench.storage.manifest import get_manifest
from dabench.storage.paths import resolve_dataset_path


def _dataset_dict(name: str, *, path: str | Path | None, decode: bool):
    manifest = get_manifest(name)
    actual_path = resolve_dataset_path(name, path)
    layout = manifest.get("prepared", {}).get("layout")
    if layout == "hf_prepared":
        return load_prepared_dataset_dict(actual_path, decode=decode)
    if layout == "office31_images":
        return load_folder_domain_dataset(actual_path, decode=decode)
    if layout == "visda2017_official":
        return load_visda_dataset_dict(actual_path, decode=decode)
    raise ValueError(
        f"Unsupported prepared layout for dataset {name!r}: {layout!r}. "
        "The manifest must define a known prepared.layout."
    )


def _resolve_domain_value(dataset, domain: str | int):
    feature = dataset.features.get("domain")
    if feature is None:
        raise ValueError("This dataset does not expose a domain column.")
    if isinstance(domain, str):
        names = getattr(feature, "names", None)
        if names is not None:
            if domain not in names:
                available = ", ".join(str(item) for item in names)
                raise ValueError(f"Unsupported domain {domain!r}. Available domains: {available}")
            return names.index(domain)
    return domain


def _select_split(dataset_dict, split: str):
    if split not in dataset_dict:
        available = ", ".join(dataset_dict.keys())
        raise ValueError(f"Unsupported split {split!r}. Available splits: {available}")
    return dataset_dict[split]


def load_hf_dataset(
    name: str,
    *,
    path: str | Path | None = None,
    split: str = "train",
    domain: str | int | None = None,
    decode: bool = True,
) -> Any:
    dataset_dict = _dataset_dict(name, path=path, decode=decode)
    dataset = _select_split(dataset_dict, split)
    if domain is None:
        return dataset
    if "domain" not in dataset.column_names:
        raise ValueError(f"Dataset {name!r} split {split!r} does not expose a domain column.")
    resolved_domain = _resolve_domain_value(dataset, domain)
    return dataset.filter(lambda row, target=resolved_domain: row["domain"] == target)


def load_dataset(
    name: str,
    *,
    path: str | Path | None = None,
    split: str = "train",
    domain: str | int | None = None,
    transform=None,
    decode: bool = True,
) -> Any:
    dataset = load_hf_dataset(name, path=path, split=split, domain=domain, decode=decode)
    return build_torch_dataset(dataset, transform=transform, domain_column="domain", path_column="image_path")


def load_view(
    name: str,
    *,
    path: str | Path | None = None,
    domain: str | int | None = None,
    split: str = "train",
    decode: bool = True,
    format: str = "hf",
    transform=None,
):
    if domain is None:
        raise ValueError("load_view requires an explicit `domain`.")
    if format == "hf":
        return load_hf_dataset(name, path=path, split=split, domain=domain, decode=decode)
    if format == "torch":
        return load_dataset(name, path=path, split=split, domain=domain, transform=transform, decode=decode)
    raise ValueError("format must be one of: hf, torch")


__all__ = [
    "DomainDatasetView",
    "build_torch_dataset",
    "load_dataset",
    "load_hf_dataset",
    "load_view",
]
