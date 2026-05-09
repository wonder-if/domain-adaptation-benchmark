"""Office-31 UDA suite helpers."""

from __future__ import annotations

from itertools import permutations

from dabench.suite._common import build_uda_suite_items

OFFICE31_DOMAINS = ("amazon", "dslr", "webcam")

OFFICE31_UDA_DATASET_DEFAULTS = {
    "dataset": "office-31",
    "decode": True,
}

DG_LOADER_DEFAULTS = {
    "source_train_batch_size": 32,
    "val_batch_size": None,
    "test_batch_size": None,
    "source_train_transform": None,
    "val_transform": None,
    "test_transform": None,
    "num_workers": 4,
    "pin_memory": None,
}

UDA_LOADER_DEFAULTS = {
    "source_train_batch_size": 32,
    "target_train_batch_size": None,
    "val_batch_size": None,
    "test_batch_size": None,
    "source_train_transform": None,
    "target_train_transform": None,
    "val_transform": None,
    "test_transform": None,
    "num_workers": 4,
    "pin_memory": None,
}

UNIDA_LOADER_DEFAULTS = {
    "source_train_batch_size": 32,
    "target_train_batch_size": None,
    "test_batch_size": None,
    "source_train_transform": None,
    "target_train_transform": None,
    "test_transform": None,
    "num_workers": 4,
    "pin_memory": None,
}


def build_office31_uda_suite(
    *,
    format: str,
    dataset_defaults=None,
    setting_defaults=None,
):
    items = (
        {
            "name": f"{source}_to_{target}",
            "source_domain": source,
            "target_domain": target,
        }
        for source, target in permutations(OFFICE31_DOMAINS, 2)
    )
    return build_uda_suite_items(
        items=items,
        dataset_defaults=OFFICE31_UDA_DATASET_DEFAULTS if dataset_defaults is None else dataset_defaults,
        setting_defaults=UDA_LOADER_DEFAULTS if setting_defaults is None else setting_defaults,
        format=format,
        setting_name="uda",
    )


def build_office31_dg_suite(
    *,
    format: str,
    dataset_defaults=None,
    setting_defaults=None,
):
    items = (
        {
            "name": f"all_except_{target}_to_{target}",
            "source_domains": tuple(domain for domain in OFFICE31_DOMAINS if domain != target),
            "target_domain": target,
        }
        for target in OFFICE31_DOMAINS
    )
    return build_uda_suite_items(
        items=items,
        dataset_defaults=OFFICE31_UDA_DATASET_DEFAULTS if dataset_defaults is None else dataset_defaults,
        setting_defaults=DG_LOADER_DEFAULTS if setting_defaults is None else setting_defaults,
        format=format,
        setting_name="dg",
    )


def build_office31_unida_suite(
    *,
    format: str,
    dataset_defaults=None,
    setting_defaults=None,
):
    scenarios = (
        ("cda", 31, 0, 0),
        ("pda", 10, 21, 0),
        ("oda", 10, 0, 11),
        ("opda", 10, 10, 11),
    )
    code_by_domain = {"amazon": "a", "dslr": "d", "webcam": "w"}
    items = []
    for source, target in permutations(OFFICE31_DOMAINS, 2):
        task = f"{code_by_domain[source]}{code_by_domain[target]}"
        for scenario, shared, source_private, target_private in scenarios:
            items.append(
                {
                    "name": f"{scenario}_{task}_{shared}-{source_private}-{target_private}",
                    "task": task,
                    "shared": shared,
                    "source_private": source_private,
                    "target_private": target_private,
                }
            )
    return build_uda_suite_items(
        items=items,
        dataset_defaults={"dataset": "office"} if dataset_defaults is None else dataset_defaults,
        setting_defaults=UNIDA_LOADER_DEFAULTS if setting_defaults is None else setting_defaults,
        format=format,
        setting_name="unida",
    )
