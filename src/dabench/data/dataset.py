"""Generic dataset loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dabench.data.base import DomainDatasetView
from dabench.data.common import (
    build_torch_dataset,
    load_folder_domain_dataset,
    load_prepared_dataset_dict,
    load_visda_dataset_dict,
)
from dabench.data.minidomainnet import load_mini_domainnet_dataset_dict
from dabench.storage.manifest import get_manifest
from dabench.storage.paths import resolve_dataset_path

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


def _select_domain(dataset, domain: str | int):
    resolved_domain = _resolve_domain_value(dataset, domain)
    domain_values = dataset["domain"]
    indices = [index for index, value in enumerate(domain_values) if value == resolved_domain]
    if not indices:
        raise ValueError(f"Domain {domain!r} produced an empty subset.")
    return dataset.select(indices)


def _select_split(dataset_dict, split: str | None):
    if split is None:
        splits = list(dataset_dict.keys())
        if len(splits) == 1:
            return dataset_dict[splits[0]]
        available = ", ".join(splits)
        raise ValueError(f"This dataset requires an explicit split. Available splits: {available}")
    if split not in dataset_dict:
        available = ", ".join(dataset_dict.keys())
        raise ValueError(f"Unsupported split {split!r}. Available splits: {available}")
    return dataset_dict[split]


def load_hf_dataset(path: str | Path, *, decode: bool = True):
    return load_prepared_dataset_dict(path, decode=decode)


def load_view(
    name: str,
    *,
    domain: str | int | None = None,
    split: str | None = None,
    decode: bool = True,
    format: str = "hf",
):
    if domain is None:
        raise ValueError("load_view requires an explicit `domain`.")

    manifest = get_manifest(name)
    actual_path = resolve_dataset_path(name)
    layout = manifest.get("prepared", {}).get("layout")
    if layout == "hf_prepared":
        dataset_dict = load_hf_dataset(actual_path, decode=decode)
        dataset = _select_split(dataset_dict, split)
    elif layout == "domainnet_split_files":
        dataset_dict = load_mini_domainnet_dataset_dict(actual_path, decode=decode, dataset_name=manifest["id"])
        dataset = _select_split(dataset_dict, split)
    elif layout == "office31_images":
        dataset_dict = load_folder_domain_dataset(actual_path, decode=decode)
        dataset = dataset_dict["all"]
    elif layout == "visda2017_official":
        dataset_dict = load_visda_dataset_dict(actual_path, decode=decode)
        dataset = _select_split(dataset_dict, split)
    else:
        raise ValueError(
            f"Unsupported prepared layout for dataset {name!r}: {layout!r}. "
            "The manifest must define a known prepared.layout."
        )

    if domain is not None:
        if "domain" not in dataset.column_names:
            raise ValueError(f"Dataset {name!r} split {split!r} does not expose a domain column.")
        dataset = _select_domain(dataset, domain)

    if format == "hf":
        return dataset
    if format == "torch":
        return build_torch_dataset(dataset, domain_column="domain", path_column="image_path")
    raise ValueError("format must be one of: hf, torch")


__all__ = [
    "DomainDatasetView",
    "build_torch_dataset",
    "load_hf_dataset",
    "load_view",
]
