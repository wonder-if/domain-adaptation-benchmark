"""iWildCam metadata-aware loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from dabench.data.common import load_prepared_dataset_dict
from dabench.storage.paths import get_dataset_field_path
from dabench.utils.imports import require_datasets_for_loading

IWILDCAM_SPLITS = ("train", "test")

_TRAIN_ANNOTATIONS_FIELD = "train_annotations_path"
_TEST_INFORMATION_FIELD = "test_information_path"


def load_iwildcam_dataset_dict(path: str | Path, *, decode: bool = True, splits: Iterable[str] | None = None):
    """Load iWildCam image splits and join metadata from configured JSON files."""

    _Dataset, DatasetDict, Image, _ClassLabel, _concat = require_datasets_for_loading()
    raw = load_prepared_dataset_dict(path, decode=False)
    selected_splits = tuple(raw.keys()) if splits is None else tuple(splits)
    splits = {}
    for split in selected_splits:
        if split not in raw:
            available = ", ".join(raw.keys())
            raise ValueError(f"Unsupported iWildCam split {split!r}. Available splits: {available}")
        dataset = raw[split]
        splits[split] = enrich_iwildcam_split(dataset, split=split, decode=decode)
    return DatasetDict(splits)


def enrich_iwildcam_split(dataset, *, split: str, decode: bool = True):
    """Join iWildCam JSON metadata to one image-only split."""

    _Dataset, _DatasetDict, Image, _ClassLabel, _concat = require_datasets_for_loading()
    metadata = _load_split_metadata(split)
    categories = _categories_by_id(metadata)
    category_ids = sorted(categories)
    category_id_to_label = {category_id: index for index, category_id in enumerate(category_ids)}
    image_info_by_file = _image_info_by_file(metadata)
    annotation_by_image_id = _annotation_by_image_id(metadata) if split == "train" else {}

    columns: dict[str, list[Any]] = {
        "source_index": [],
        "image_id": [],
        "location": [],
        "seq_id": [],
        "frame_num": [],
        "seq_num_frames": [],
        "datetime": [],
        "width": [],
        "height": [],
    }
    if split == "train":
        columns["label"] = []
        columns["category_id"] = []
        columns["category_name"] = []

    for image_path in _image_paths(dataset):
        file_name = Path(str(image_path or "")).name
        if not file_name:
            raise ValueError("iWildCam image rows must expose `image.path` for metadata join.")
        info = image_info_by_file.get(file_name)
        if info is None:
            raise ValueError(f"Missing iWildCam metadata for image file {file_name!r}.")
        image_id = str(info["id"])
        columns["source_index"].append(int(info["_source_index"]))
        columns["image_id"].append(image_id)
        columns["location"].append(int(info["location"]))
        columns["seq_id"].append(str(info["seq_id"]))
        columns["frame_num"].append(int(info["frame_num"]))
        columns["seq_num_frames"].append(int(info["seq_num_frames"]))
        columns["datetime"].append(str(info["datetime"]))
        columns["width"].append(int(info["width"]))
        columns["height"].append(int(info["height"]))

        if split == "train":
            annotation = annotation_by_image_id.get(image_id)
            if annotation is None:
                raise ValueError(f"Missing iWildCam train annotation for image id {image_id!r}.")
            category_id = int(annotation["category_id"])
            columns["category_id"].append(category_id)
            columns["label"].append(category_id_to_label[category_id])
            columns["category_name"].append(str(categories[category_id]["name"]))

    enriched = dataset
    for column_name, values in columns.items():
        if column_name in enriched.column_names:
            enriched = enriched.remove_columns(column_name)
        enriched = enriched.add_column(column_name, values)
    if decode:
        enriched = enriched.cast_column("image", Image(decode=decode))
    return enriched


def _image_paths(dataset) -> list[str | None]:
    image_column = dataset.data.column("image")
    paths: list[str | None] = []
    for chunk in image_column.chunks:
        paths.extend(chunk.field("path").to_pylist())
    return paths


def get_iwildcam_class_names() -> tuple[str, ...]:
    """Return iWildCam category names ordered by contiguous label index."""

    metadata = _load_split_metadata("train")
    categories = _categories_by_id(metadata)
    return tuple(categories[category_id]["name"] for category_id in sorted(categories))


def _load_split_metadata(split: str) -> dict[str, Any]:
    if split == "train":
        path = get_dataset_field_path("iwildcam", _TRAIN_ANNOTATIONS_FIELD)
    elif split == "test":
        path = get_dataset_field_path("iwildcam", _TEST_INFORMATION_FIELD)
    else:
        raise ValueError(f"Unsupported iWildCam split {split!r}. Available splits: {', '.join(IWILDCAM_SPLITS)}")

    if path is None:
        field = _TRAIN_ANNOTATIONS_FIELD if split == "train" else _TEST_INFORMATION_FIELD
        raise FileNotFoundError(f"No iWildCam metadata path configured for field {field!r}.")
    if not path.is_file():
        raise FileNotFoundError(f"Missing iWildCam metadata file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid iWildCam metadata JSON object: {path}")
    return payload


def _categories_by_id(metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    categories = metadata.get("categories")
    if not isinstance(categories, list):
        raise ValueError("iWildCam metadata must contain a `categories` list.")
    by_id = {}
    for category in categories:
        if not isinstance(category, dict) or "id" not in category or "name" not in category:
            raise ValueError("iWildCam category records must contain `id` and `name`.")
        by_id[int(category["id"])] = category
    if not by_id:
        raise ValueError("iWildCam metadata contains no categories.")
    return by_id


def _image_info_by_file(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    images = metadata.get("images")
    if not isinstance(images, list):
        raise ValueError("iWildCam metadata must contain an `images` list.")
    by_file = {}
    for index, info in enumerate(images):
        if not isinstance(info, dict) or "file_name" not in info:
            raise ValueError("iWildCam image records must contain `file_name`.")
        record = dict(info)
        record["_source_index"] = index
        by_file[str(record["file_name"])] = record
    return by_file


def _annotation_by_image_id(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    annotations = metadata.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("iWildCam train metadata must contain an `annotations` list.")
    by_image_id = {}
    for annotation in annotations:
        if not isinstance(annotation, dict) or "image_id" not in annotation or "category_id" not in annotation:
            raise ValueError("iWildCam annotations must contain `image_id` and `category_id`.")
        by_image_id[str(annotation["image_id"])] = annotation
    return by_image_id


__all__ = [
    "IWILDCAM_SPLITS",
    "enrich_iwildcam_split",
    "get_iwildcam_class_names",
    "load_iwildcam_dataset_dict",
]
