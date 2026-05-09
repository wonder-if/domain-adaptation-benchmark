"""UDA setting helpers."""

from __future__ import annotations

from typing import Any, Literal

from dabench.data import build_loader, load_view, make_paired_forever_loader


def _role_views(dataset: str, *, source_domain: str | int, target_domain: str | int) -> dict[str, dict[str, str | int | None]]:
    normalized = dataset.strip().lower().replace("_", "-")
    if normalized in {"office-31", "office31"}:
        return {
            "source_train": {"domain": source_domain, "split": None},
            "target_train": {"domain": target_domain, "split": None},
            "val": {"domain": target_domain, "split": None},
            "test": {"domain": target_domain, "split": None},
        }
    if normalized in {"office-home", "officehome"}:
        return {
            "source_train": {"domain": source_domain, "split": None},
            "target_train": {"domain": target_domain, "split": None},
            "val": {"domain": target_domain, "split": None},
            "test": {"domain": target_domain, "split": None},
        }
    if normalized == "domainnet":
        return {
            "source_train": {"domain": source_domain, "split": "train"},
            "target_train": {"domain": target_domain, "split": "train"},
            "val": {"domain": target_domain, "split": "test"},
            "test": {"domain": target_domain, "split": "test"},
        }
    if normalized in {"minidomainnet", "mini-domainnet"}:
        return {
            "source_train": {"domain": source_domain, "split": "train"},
            "target_train": {"domain": target_domain, "split": "train"},
            "val": {"domain": target_domain, "split": "test"},
            "test": {"domain": target_domain, "split": "test"},
        }
    if normalized == "visda-2017":
        return {
            "source_train": {"domain": source_domain, "split": "train"},
            "target_train": {"domain": target_domain, "split": "validation"},
            "val": {"domain": target_domain, "split": "validation"},
            # VisDA UDA typically uses train(synthetic) -> validation(real); test is not part of the usual split.
            "test": {"domain": target_domain, "split": "validation"},
        }
    raise ValueError(f"Unsupported dataset for UDA split routing: {dataset!r}")


def load_uda(
    *,
    dataset: str,
    source_domain: str | int,
    target_domain: str | int,
    format: Literal["hf", "torch"] = "hf",
    source_train_batch_size: int,
    target_train_batch_size: int | None = None,
    val_batch_size: int | None = None,
    test_batch_size: int | None = None,
    source_train_transform=None,
    target_train_transform=None,
    val_transform=None,
    test_transform=None,
    num_workers: int = 4,
    pin_memory: bool | None = None,
    decode: bool = True,
) -> tuple[Any, Any, Any]:
    views = _role_views(dataset, source_domain=source_domain, target_domain=target_domain)
    source_train = load_view(
        dataset,
        domain=views["source_train"]["domain"],
        split=views["source_train"]["split"],
        format=format,
        decode=decode,
    )
    target_train = load_view(
        dataset,
        domain=views["target_train"]["domain"],
        split=views["target_train"]["split"],
        format=format,
        decode=decode,
    )
    val_dataset = load_view(
        dataset,
        domain=views["val"]["domain"],
        split=views["val"]["split"],
        format=format,
        decode=decode,
    )
    test_dataset = load_view(
        dataset,
        domain=views["test"]["domain"],
        split=views["test"]["split"],
        format=format,
        decode=decode,
    )

    train_loader = make_paired_forever_loader(
        source_train,
        target_train,
        source_batch_size=source_train_batch_size,
        target_batch_size=target_train_batch_size,
        source_transform=source_train_transform,
        target_transform=target_train_transform,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=val_batch_size or source_train_batch_size,
        mode="test",
        transform=val_transform,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = build_loader(
        test_dataset,
        batch_size=test_batch_size or val_batch_size or source_train_batch_size,
        mode="test",
        transform=test_transform,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
