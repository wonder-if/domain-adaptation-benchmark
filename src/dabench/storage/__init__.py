"""Local dataset storage preparation APIs."""

from dabench.storage.manifest import get_manifest, list_manifests
from dabench.storage.prepare import prepare_dataset
from dabench.storage.paths import (
    get_dataset_entry,
    get_dataset_field,
    get_dataset_field_path,
    get_dataset_path,
    list_dataset_paths,
    resolve_dataset_path,
    set_dataset_path,
)


def download_dataset(name: str, **kwargs):
    return prepare_dataset(name, **kwargs)

__all__ = [
    "download_dataset",
    "get_dataset_entry",
    "get_dataset_field",
    "get_dataset_field_path",
    "get_dataset_path",
    "get_manifest",
    "list_manifests",
    "list_dataset_paths",
    "prepare_dataset",
    "resolve_dataset_path",
    "set_dataset_path",
]
