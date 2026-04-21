"""DataLoader helpers for loaded dataset views."""

from __future__ import annotations

from typing import Any

from dabench.data.transforms import build_test_transform, build_train_transform
from dabench.utils.imports import require_torch_for_loading


def _as_dict(sample: Any) -> dict[str, Any]:
    if isinstance(sample, dict):
        return dict(sample)
    if isinstance(sample, tuple):
        output: dict[str, Any] = {}
        if len(sample) > 0:
            output["image"] = sample[0]
        if len(sample) > 1:
            output["label"] = sample[1]
        if len(sample) > 2:
            output["domain"] = sample[2]
        return output
    raise TypeError(f"Unsupported sample type: {type(sample)!r}")


class _TransformedDataset:
    def __init__(self, dataset, transform=None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = _as_dict(self.dataset[index])
        image = sample.get("image")
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        if image is not None and self.transform is not None:
            sample["pixel_values"] = self.transform(image)
        return sample


def default_collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch, _, _ = require_torch_for_loading()
    first = batch[0]
    images = [item["pixel_values"] if "pixel_values" in item else item["image"] for item in batch]
    output: dict[str, Any] = {"pixel_values": torch.stack(images)}

    label_key = "labels" if "labels" in first else "label" if "label" in first else None
    if label_key is not None:
        output["labels"] = torch.tensor([item[label_key] for item in batch], dtype=torch.long)
    return output


def build_loader(
    dataset,
    *,
    batch_size: int,
    mode: str = "train",
    transform=None,
    collator=default_collator,
    num_workers: int = 4,
    pin_memory: bool | None = None,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
):
    torch, DataLoader, _IterableDataset = require_torch_for_loading()
    if mode not in {"train", "test"}:
        raise ValueError("mode must be one of: train, test")
    resolved_transform = transform
    if resolved_transform is None:
        resolved_transform = build_train_transform() if mode == "train" else build_test_transform()
    wrapped = _TransformedDataset(dataset, transform=resolved_transform)
    resolved_shuffle = (mode == "train") if shuffle is None else shuffle
    resolved_drop_last = (mode == "train") if drop_last is None else drop_last
    resolved_pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    return DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=resolved_shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=resolved_pin_memory,
        drop_last=resolved_drop_last,
    )


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


def make_paired_forever_loader(
    source_dataset,
    target_dataset,
    *,
    source_batch_size: int,
    target_batch_size: int | None = None,
    source_transform=None,
    target_transform=None,
    source_mode: str = "train",
    target_mode: str = "train",
    num_workers: int = 4,
    collator=default_collator,
    pin_memory: bool | None = None,
    drop_last: bool = True,
):
    _torch, _DataLoader, IterableDataset = require_torch_for_loading()
    source_loader = build_loader(
        source_dataset,
        batch_size=source_batch_size,
        mode=source_mode,
        transform=source_transform,
        collator=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=drop_last,
    )
    target_loader = build_loader(
        target_dataset,
        batch_size=target_batch_size or source_batch_size,
        mode=target_mode,
        transform=target_transform,
        collator=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=drop_last,
    )

    class _PairedForeverIterable(IterableDataset):
        def __iter__(self):
            source_iter = _ForeverDataIterator(source_loader)
            target_iter = _ForeverDataIterator(target_loader)
            while True:
                yield {"source": next(source_iter), "target": next(target_iter)}

        def __len__(self):
            raise TypeError("This iterable is infinite; use max_steps to control training length.")

    return _PairedForeverIterable()


__all__ = [
    "build_loader",
    "default_collator",
    "make_paired_forever_loader",
]
