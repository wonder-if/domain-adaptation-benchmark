"""Dataset class-name helpers."""

from __future__ import annotations

from dabench.data.iwildcam import get_iwildcam_class_names
from dabench.data.minidomainnet import get_mini_domainnet_class_names
from dabench.storage.manifest import get_manifest

CAMELYON17_CLASS_NAMES = ("normal tissue", "tumor tissue")


def get_class_names(dataset: str) -> tuple[str, ...]:
    """Return class names suitable for CLIP text-feature extraction."""

    dataset_id = str(get_manifest(dataset)["id"])
    if dataset_id == "camelyon17":
        return CAMELYON17_CLASS_NAMES
    if dataset_id == "iwildcam":
        return get_iwildcam_class_names()
    if dataset_id == "minidomainnet":
        return get_mini_domainnet_class_names(dataset_name=dataset_id)
    raise ValueError(f"No class-name helper is registered for dataset {dataset!r}.")


__all__ = [
    "CAMELYON17_CLASS_NAMES",
    "get_class_names",
]
