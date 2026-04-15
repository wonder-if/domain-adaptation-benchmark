"""Shared dataset types and helpers."""

from __future__ import annotations

from typing import Any, Literal, Sequence, TypedDict

DatasetFormat = Literal["hf", "torch"]


class IndexItem(TypedDict, total=False):
    path: str
    label: int
    domain: str
    class_name: str
    split: str
    image_path: str
    image: Any


class DomainDatasetView:
    """PyTorch-style dataset for one domain."""

    def __init__(self, items: Sequence[IndexItem], domain: str, transform=None) -> None:
        self.items = list(items)
        self.domain = domain
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image = self._load_image(item["path"])
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": item["label"],
            "class_name": item.get("class_name"),
            "domain": self.domain,
            "index": index,
            "path": item["path"],
            "image_path": item["path"],
        }

    def _load_image(self, path: str):
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required for format='torch' image loading. Install it in the active environment."
            ) from exc
        with Image.open(path) as image:
            return image.convert("RGB")
