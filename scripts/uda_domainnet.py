#!/usr/bin/env python3
"""Minimal UDA loading example for DomainNet."""

from __future__ import annotations

from dabench.suite import build_suites, load_suite_item


def _shape(value) -> str:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return str(tuple(shape))
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}={_shape(item)}" for key, item in value.items()) + "}"
    return type(value).__name__


def main() -> None:
    suites = build_suites(datasets="domainnet", setting="uda", format="hf")[0]
    settings = suites["settings"]
    print(f"suite: {suites['suite_id']} ({len(settings)} items)")
    item = settings[0]
    print(f"item: {item['name']}")

    train_loader, val_loader, test_loader = load_suite_item(item)
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))

    print("train:", _shape(train_batch))
    print("val:", _shape(val_batch))
    print("test:", _shape(test_batch))


if __name__ == "__main__":
    main()
