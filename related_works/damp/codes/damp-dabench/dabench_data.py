from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from torch.utils.data import DataLoader, Dataset

from dabench.data import load_view


DATASET_NAME_MAP = {
    "OfficeHome": "office-home",
    "Office31": "office-31",
    "VisDA17": "visda-2017",
    "VisDA2017": "visda-2017",
    "miniDomainNet": "minidomainnet",
    "DomainNet": "domainnet",
}

DOMAIN_NAME_MAP = {
    "office-home": {
        "art": "Art",
        "clipart": "Clipart",
        "product": "Product",
        "real_world": "Real World",
        "real-world": "Real World",
        "real world": "Real World",
    },
    "domainnet": {
        "clipart": "clipart",
        "infograph": "infograph",
        "painting": "painting",
        "quickdraw": "quickdraw",
        "real": "real",
        "sketch": "sketch",
    },
    "visda-2017": {
        "synthetic": "synthetic",
        "real": "real",
    },
    "office-31": {
        "amazon": "amazon",
        "dslr": "dslr",
        "webcam": "webcam",
    },
}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _dataset_key(cfg: Any) -> str:
    dabench_cfg = _cfg_get(cfg, "DABENCH", None)
    if dabench_cfg is not None:
        explicit_name = _cfg_get(dabench_cfg, "DATASET_NAME", "")
        if explicit_name:
            return explicit_name
    dataset_name = _cfg_get(_cfg_get(cfg, "DATASET", None), "NAME")
    return DATASET_NAME_MAP.get(dataset_name, str(dataset_name).strip().lower().replace("_", "-"))


def _normalize_domain(dataset_name: str, domain: str) -> str:
    mapping = DOMAIN_NAME_MAP.get(dataset_name, {})
    key = str(domain).strip().lower()
    return mapping.get(key, domain)


def _role_splits(dataset_name: str) -> dict[str, str | None]:
    normalized = dataset_name.strip().lower().replace("_", "-")
    if normalized in {"office-31", "office-home"}:
        return {
            "source_train": None,
            "target_train": None,
            "val": None,
            "test": None,
        }
    if normalized in {"domainnet", "minidomainnet"}:
        return {
            "source_train": "train",
            "target_train": "train",
            "val": "test",
            "test": "test",
        }
    if normalized == "visda-2017":
        return {
            "source_train": "train",
            "target_train": "validation",
            "val": "validation",
            "test": "validation",
        }
    raise ValueError(f"Unsupported dabench dataset for DAMP: {dataset_name!r}")


def _classnames(dataset) -> list[str]:
    label_feature = dataset.features.get("label")
    names = getattr(label_feature, "names", None)
    if names:
        return list(names)
    labels = sorted({int(row["label"]) for row in dataset})
    return [str(label) for label in labels]


class _TwoViewDataset(Dataset):
    def __init__(self, dataset, *, weak_transform, strong_transform) -> None:
        self.dataset = dataset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        image = sample["image"]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        return {
            "img": self.weak_transform(image),
            "img2": self.strong_transform(image),
            "label": int(sample["label"]),
            "index": index,
        }


class _EvalDataset(Dataset):
    def __init__(self, dataset, *, transform) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        image = sample["image"]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        return {
            "img": self.transform(image),
            "label": int(sample["label"]),
            "index": index,
        }


@dataclass
class DabenchDataBundle:
    train_loader_x: Any
    train_loader_u: Any
    val_loader: Any
    test_loader: Any
    num_classes: int
    lab2cname: dict[int, str]
    dataset: Any


def build_dabench_data_bundle(
    cfg,
    *,
    weak_transform,
    strong_transform,
    eval_transform,
) -> DabenchDataBundle:
    dataset_name = _dataset_key(cfg)
    split_map = _role_splits(dataset_name)

    source_domain = _normalize_domain(dataset_name, cfg.DATASET.SOURCE_DOMAINS[0])
    target_domain = _normalize_domain(dataset_name, cfg.DATASET.TARGET_DOMAINS[0])

    source_train = load_view(
        dataset_name,
        domain=source_domain,
        split=split_map["source_train"],
        format="hf",
        decode=True,
    )
    target_train = load_view(
        dataset_name,
        domain=target_domain,
        split=split_map["target_train"],
        format="hf",
        decode=True,
    )
    val_dataset = load_view(
        dataset_name,
        domain=target_domain,
        split=split_map["val"],
        format="hf",
        decode=True,
    )
    test_dataset = load_view(
        dataset_name,
        domain=target_domain,
        split=split_map["test"],
        format="hf",
        decode=True,
    )

    classnames = _classnames(source_train)
    lab2cname = {idx: name for idx, name in enumerate(classnames)}
    num_workers = int(cfg.DATALOADER.NUM_WORKERS)
    pin_memory = bool(cfg.USE_CUDA)

    train_loader_x = DataLoader(
        _TwoViewDataset(source_train, weak_transform=weak_transform, strong_transform=strong_transform),
        batch_size=int(cfg.DATALOADER.TRAIN_X.BATCH_SIZE),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    train_loader_u = DataLoader(
        _TwoViewDataset(target_train, weak_transform=weak_transform, strong_transform=strong_transform),
        batch_size=int(cfg.DATALOADER.TRAIN_U.BATCH_SIZE),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        _EvalDataset(val_dataset, transform=eval_transform),
        batch_size=int(cfg.DATALOADER.TEST.BATCH_SIZE),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        _EvalDataset(test_dataset, transform=eval_transform),
        batch_size=int(cfg.DATALOADER.TEST.BATCH_SIZE),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    dataset_meta = SimpleNamespace(classnames=classnames)
    return DabenchDataBundle(
        train_loader_x=train_loader_x,
        train_loader_u=train_loader_u,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=len(classnames),
        lab2cname=lab2cname,
        dataset=dataset_meta,
    )


def describe_dataset_source(cfg) -> str:
    dataset_name = _dataset_key(cfg)
    source = _cfg_get(_cfg_get(cfg, "DABENCH", None), "DATASET_NAME", "")
    if cfg.DATASET.NAME == "miniDomainNet" and dataset_name == "minidomainnet":
        return "miniDomainNet config routed to dabench minidomainnet"
    if source:
        return f"dabench dataset={source}"
    return f"dabench dataset={dataset_name}"
