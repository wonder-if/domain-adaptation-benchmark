"""Shared helpers for benchmark suite builders."""

from __future__ import annotations

from typing import Any

from dabench.setting import load_uda


def _merge_suite_config(*configs):
    cfg = {}
    for c in configs:
        if c:
            cfg.update(c)
    return cfg


def build_uda_suite_items(
    *,
    items,
    dataset_defaults=None,
    setting_defaults=None,
    format: str,
):
    dataset_defaults = {} if dataset_defaults is None else dataset_defaults
    setting_defaults = {} if setting_defaults is None else setting_defaults

    suite_items = []
    for item in items:
        cfg = _merge_suite_config(
            dataset_defaults,
            setting_defaults,
            item,
            {"format": format},
        )
        suite_items.append(cfg)
    return suite_items


def load_suite_item(item: dict[str, Any]):
    item = dict(item)
    item.pop("name", None)
    return load_uda(**item)
