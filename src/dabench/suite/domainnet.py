"""DomainNet UDA suite helpers."""

from __future__ import annotations

from itertools import permutations

from dabench.suite._common import build_uda_suite_items

DOMAINNET_DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")

DOMAINNET_UDA_DATASET_DEFAULTS = {
    "dataset": "domainnet",
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


def build_domainnet_uda_suite(
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
        for source, target in permutations(DOMAINNET_DOMAINS, 2)
    )
    return build_uda_suite_items(
        items=items,
        dataset_defaults=DOMAINNET_UDA_DATASET_DEFAULTS if dataset_defaults is None else dataset_defaults,
        setting_defaults=UDA_LOADER_DEFAULTS if setting_defaults is None else setting_defaults,
        format=format,
        setting_name="uda",
    )


def build_domainnet_dg_suite(
    *,
    format: str,
    dataset_defaults=None,
    setting_defaults=None,
):
    items = (
        {
            "name": f"all_except_{target}_to_{target}",
            "source_domains": tuple(domain for domain in DOMAINNET_DOMAINS if domain != target),
            "target_domain": target,
        }
        for target in DOMAINNET_DOMAINS
    )
    return build_uda_suite_items(
        items=items,
        dataset_defaults=DOMAINNET_UDA_DATASET_DEFAULTS if dataset_defaults is None else dataset_defaults,
        setting_defaults=DG_LOADER_DEFAULTS if setting_defaults is None else setting_defaults,
        format=format,
        setting_name="dg",
    )


def build_domainnet_unida_suite(
    *,
    format: str,
    dataset_defaults=None,
    setting_defaults=None,
):
    code_by_domain = {"painting": "p", "real": "r", "sketch": "s"}
    items = []
    for source, target in permutations(("painting", "real", "sketch"), 2):
        task = f"{code_by_domain[source]}{code_by_domain[target]}"
        items.append(
            {
                "name": f"opda_{task}_150-50-145",
                "task": task,
                "shared": 150,
                "source_private": 50,
                "target_private": 145,
            }
        )
    return build_uda_suite_items(
        items=items,
        dataset_defaults={"dataset": "domainnet"} if dataset_defaults is None else dataset_defaults,
        setting_defaults=UNIDA_LOADER_DEFAULTS if setting_defaults is None else setting_defaults,
        format=format,
        setting_name="unida",
    )
