"""Lightweight dataset helpers for training and evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _require_datasets():
    try:
        from datasets import ClassLabel, Dataset, DatasetDict, Image, concatenate_datasets  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for dataset loading. Install with `pip install -e .[data]`."
        ) from exc
    return Dataset, DatasetDict, Image, ClassLabel, concatenate_datasets


def _require_torch():
    try:
        import torch  # type: ignore
        from torch.utils.data import DataLoader, IterableDataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for dataset loading. Install it in the active environment."
        ) from exc
    return torch, DataLoader, IterableDataset


@dataclass
class UDABundle:
    train_source: Any
    train_target: Any
    val: Any
    test: Any
    num_classes: int
    class_names: list[str]


class TorchImageDataset:
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
        self.hf_dataset = dataset
        self.image_column = image_column
        self.label_column = label_column
        self.domain_column = domain_column
        self.path_column = path_column
        self.transform = transform
        self.class_names = infer_class_names(dataset, label_column)
        self.domain_names = infer_domain_names(dataset, domain_column)
        self.classes = self.class_names

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.hf_dataset[index]
        image = row[self.image_column]
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        output: dict[str, Any] = {
            "index": index,
            "image": image,
        }
        if self.transform is not None:
            output["pixel_values"] = self.transform(image)

        if self.label_column and self.label_column in row:
            label = int(row[self.label_column])
            output["labels"] = label
            if 0 <= label < len(self.class_names):
                output["class_name"] = self.class_names[label]

        if self.domain_column and self.domain_column in row:
            domain_value = row[self.domain_column]
            if isinstance(domain_value, int):
                output["domain_ids"] = domain_value
                if 0 <= domain_value < len(self.domain_names):
                    output["domain_name"] = self.domain_names[domain_value]
            else:
                domain_name = str(domain_value)
                output["domain_name"] = domain_name
                if domain_name in self.domain_names:
                    output["domain_ids"] = self.domain_names.index(domain_name)

        if self.path_column and self.path_column in row:
            output["image_path"] = row[self.path_column]
        return output


def default_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch, _DataLoader, _IterableDataset = _require_torch()
    first = batch[0]
    collated: dict[str, Any] = {}

    if "pixel_values" in first:
        collated["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])
    else:
        collated["images"] = [item["image"] for item in batch]

    for key in ("labels", "domain_ids", "index"):
        if key in first:
            collated[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)

    for key in first:
        if key in collated or key in {"pixel_values", "image", "labels", "domain_ids", "index"}:
            continue
        collated[key] = [item.get(key) for item in batch]
    return collated


class _ForeverDataIterator:
    def __init__(self, loader) -> None:
        self.loader = loader
        self.iterator = iter(loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


class _PairedForeverIterable:
    def __init__(self, source_loader, target_loader) -> None:
        self.source_loader = source_loader
        self.target_loader = target_loader

    def __iter__(self):
        source_iter = _ForeverDataIterator(self.source_loader)
        target_iter = _ForeverDataIterator(self.target_loader)
        while True:
            yield {"source": next(source_iter), "target": next(target_iter)}


def make_paired_loader(
    source_dataset,
    target_dataset,
    *,
    source_batch_size: int,
    target_batch_size: int | None = None,
    num_workers: int = 4,
    collate_fn=default_collate,
    pin_memory: bool | None = None,
    drop_last: bool = True,
):
    torch, DataLoader, _IterableDataset = _require_torch()
    resolved_pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    target_batch_size = target_batch_size or source_batch_size
    source_loader = DataLoader(
        source_dataset,
        batch_size=source_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=resolved_pin_memory,
        drop_last=drop_last,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=target_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=resolved_pin_memory,
        drop_last=drop_last,
    )
    return _PairedForeverIterable(source_loader, target_loader)


def infer_class_names(dataset, label_column: str | None = "label") -> list[str]:
    if label_column is None or label_column not in dataset.features:
        return []
    _Dataset, _DatasetDict, _Image, ClassLabel, _concat = _require_datasets()
    feature = dataset.features[label_column]
    if isinstance(feature, ClassLabel):
        return list(feature.names)
    values = sorted(dataset.unique(label_column))
    return [str(value) for value in values]


def infer_domain_names(dataset, domain_column: str | None = "domain") -> list[str]:
    if domain_column is None or domain_column not in dataset.features:
        return []
    _Dataset, _DatasetDict, _Image, ClassLabel, _concat = _require_datasets()
    feature = dataset.features[domain_column]
    if isinstance(feature, ClassLabel):
        return list(feature.names)
    values = sorted(dataset.unique(domain_column))
    return [str(value) for value in values]


def build_torch_dataset(
    dataset,
    *,
    image_column: str = "image",
    label_column: str | None = "label",
    domain_column: str | None = "domain",
    path_column: str | None = "image_path",
    transform=None,
) -> TorchImageDataset:
    return TorchImageDataset(
        dataset,
        image_column=image_column,
        label_column=label_column,
        domain_column=domain_column,
        path_column=path_column,
        transform=transform,
    )


def filter_dataset(
    dataset,
    *,
    domains: Iterable[Any] | None = None,
    labels: Iterable[int] | None = None,
    domain_column: str | None = "domain",
    label_column: str | None = "label",
) -> Any:
    if domains and domain_column and domain_column in dataset.features:
        _Dataset, _DatasetDict, _Image, ClassLabel, _concat = _require_datasets()
        feature = dataset.features[domain_column]
        if isinstance(feature, ClassLabel):
            accepted_domains = {feature.str2int(str(value)) for value in domains}
        else:
            accepted_domains = set(domains)
        dataset = dataset.filter(
            lambda value: value in accepted_domains,
            input_columns=[domain_column],
        )
    if labels and label_column and label_column in dataset.features:
        accepted_labels = set(labels)
        dataset = dataset.filter(
            lambda value: value in accepted_labels,
            input_columns=[label_column],
        )
    return dataset


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
            for class_dir in domain_dir.iterdir()
            if class_dir.is_dir()
        }
    )
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    records = []
    for domain_dir in domain_dirs:
        for class_dir in sorted(child for child in domain_dir.iterdir() if child.is_dir()):
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
                                "domain": split_name,
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
                            "domain": split_name,
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
