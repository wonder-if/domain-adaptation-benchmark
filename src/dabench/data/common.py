"""Lightweight dataset helpers for loading prepared datasets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from dabench.utils.imports import require_datasets_for_loading


def _require_datasets():
    return require_datasets_for_loading()


class _TorchDatasetView:
    def __init__(
        self,
        dataset,
        *,
        image_column: str = "image",
        label_column: str | None = "label",
        domain_column: str | None = "domain",
        path_column: str | None = "image_path",
        transform=None,
    ) -> None:
        self.dataset = dataset
        self.image_column = image_column
        self.label_column = label_column
        self.domain_column = domain_column
        self.path_column = path_column
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataset[index]
        image = row[self.image_column]
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        output: dict[str, Any] = {"index": index, "image": image}
        if self.transform is not None:
            output["pixel_values"] = self.transform(image)
        if self.label_column and self.label_column in row:
            output["labels"] = int(row[self.label_column])
        if self.domain_column and self.domain_column in row:
            output["domain"] = row[self.domain_column]
        if self.path_column and self.path_column in row:
            output["image_path"] = row[self.path_column]
        return output

    def _load_image(self, path: str):
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required for format='torch' image loading. Install it in the active environment."
            ) from exc
        with Image.open(path) as image:
            return image.convert("RGB")


def build_torch_dataset(
    dataset,
    *,
    image_column: str = "image",
    label_column: str | None = "label",
    domain_column: str | None = "domain",
    path_column: str | None = "image_path",
    transform=None,
):
    return _TorchDatasetView(
        dataset,
        image_column=image_column,
        label_column=label_column,
        domain_column=domain_column,
        path_column=path_column,
        transform=transform,
    )


def load_prepared_dataset_dict(path: str | Path, *, decode: bool = True):
    Dataset, DatasetDict, Image, _ClassLabel, concatenate_datasets = _require_datasets()
    root = Path(path).expanduser().resolve()
    info_path = root / "dataset_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset_info.json under {root}")

    import json

    info = json.loads(info_path.read_text(encoding="utf-8"))
    splits = list((info.get("splits") or {}).keys())
    datasets_by_split = {}

    for split_name in splits:
        pattern = re.compile(rf"-(?:{re.escape(split_name)})(?:-\d{{5}}-of-\d{{5}})?\.arrow$")
        files = sorted(file_path for file_path in root.glob("*.arrow") if pattern.search(file_path.name))
        if not files:
            raise FileNotFoundError(f"No Arrow shards found for split {split_name} under {root}")
        shards = [Dataset.from_file(str(file_path)) for file_path in files]
        split_dataset = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
        if not decode:
            for column_name, feature in split_dataset.features.items():
                if isinstance(feature, Image):
                    split_dataset = split_dataset.cast_column(column_name, Image(decode=False))
        datasets_by_split[split_name] = split_dataset
    return DatasetDict(datasets_by_split)


def load_folder_domain_dataset(path: str | Path, *, decode: bool = True):
    Dataset, DatasetDict, Image, ClassLabel, _concat = _require_datasets()
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing dataset root: {root}")

    domain_dirs = sorted(child for child in root.iterdir() if child.is_dir())
    class_names = sorted(
        {
            class_dir.name
            for domain_dir in domain_dirs
            for class_dir in (domain_dir / "images").iterdir()
            if class_dir.is_dir()
        }
    )
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    records = []
    for domain_dir in domain_dirs:
        image_root = domain_dir / "images"
        if not image_root.is_dir():
            raise FileNotFoundError(f"Missing Office-31 image directory: {image_root}")
        for class_dir in sorted(child for child in image_root.iterdir() if child.is_dir()):
            label = class_to_id[class_dir.name]
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    records.append(
                        {
                            "image": str(image_path),
                            "label": label,
                            "domain": domain_dir.name,
                            "image_path": str(image_path),
                        }
                    )
    dataset = Dataset.from_list(records)
    dataset = dataset.cast_column("image", Image(decode=decode))
    dataset = dataset.cast_column("label", ClassLabel(names=class_names))
    return DatasetDict({"all": dataset})


def load_visda_dataset_dict(path: str | Path, *, decode: bool = True):
    Dataset, DatasetDict, Image, ClassLabel, _concat = _require_datasets()
    root = Path(path).expanduser().resolve() / "data"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing VisDA data directory: {root}")

    split_dirs = {
        split_name: root / split_name
        for split_name in ("train", "validation", "test")
        if (root / split_name).is_dir()
    }
    class_names = sorted(
        {
            class_dir.name
            for split_dir in split_dirs.values()
            if split_dir.name != "test"
            for class_dir in split_dir.iterdir()
            if class_dir.is_dir()
        }
    )
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    datasets_by_split = {}
    for split_name, split_dir in split_dirs.items():
        records = []
        class_dirs = sorted(child for child in split_dir.iterdir() if child.is_dir())
        if class_dirs and split_name != "test":
            for class_dir in class_dirs:
                label = class_to_id[class_dir.name]
                for image_path in sorted(class_dir.rglob("*")):
                    if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                        records.append(
                            {
                                "image": str(image_path),
                                "label": label,
                                "domain": "synthetic" if split_name == "train" else "real",
                                "image_path": str(image_path),
                                "split": split_name,
                            }
                        )
        else:
            for image_path in sorted(split_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    records.append(
                        {
                            "image": str(image_path),
                            "label": -1,
                            "domain": "real",
                            "image_path": str(image_path),
                            "split": split_name,
                        }
                    )
        dataset = Dataset.from_list(records)
        dataset = dataset.cast_column("image", Image(decode=decode))
        if class_names and "label" in dataset.features and all(item["label"] >= 0 for item in records):
            dataset = dataset.cast_column("label", ClassLabel(names=class_names))
        datasets_by_split[split_name] = dataset
    return DatasetDict(datasets_by_split)
