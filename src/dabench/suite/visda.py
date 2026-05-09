"""VisDA-2017 UDA suite helpers."""

from __future__ import annotations

from dabench.suite._common import build_uda_suite_items

VISDA_DOMAINS = ("synthetic", "real")

VISDA_UDA_DATASET_DEFAULTS = {
    "dataset": "visda-2017",
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


def build_visda_uda_suite(
    *,
    format: str,
    dataset_defaults=None,
    setting_defaults=None,
):
    return build_uda_suite_items(
        items=(
            {
                "name": "synthetic_to_real",
                "source_domain": "synthetic",
                "target_domain": "real",
            },
        ),
        dataset_defaults=VISDA_UDA_DATASET_DEFAULTS if dataset_defaults is None else dataset_defaults,
        setting_defaults=UDA_LOADER_DEFAULTS if setting_defaults is None else setting_defaults,
        format=format,
        setting_name="uda",
    )


def build_visda_dg_suite(
    *,
    format: str,
    dataset_defaults=None,
    setting_defaults=None,
):
    items = (
        {
            "name": f"all_except_{target}_to_{target}",
            "source_domains": tuple(domain for domain in VISDA_DOMAINS if domain != target),
            "target_domain": target,
        }
        for target in VISDA_DOMAINS
    )
    return build_uda_suite_items(
        items=items,
        dataset_defaults=VISDA_UDA_DATASET_DEFAULTS if dataset_defaults is None else dataset_defaults,
        setting_defaults=DG_LOADER_DEFAULTS if setting_defaults is None else setting_defaults,
        format=format,
        setting_name="dg",
    )


def build_visda_unida_suite(
    *,
    format: str,
    dataset_defaults=None,
    setting_defaults=None,
):
    return build_uda_suite_items(
        items=(
            {
                "name": "opda_SR_6-3-3",
                "task": "SR",
                "shared": 6,
                "source_private": 3,
                "target_private": 3,
            },
        ),
        dataset_defaults={"dataset": "visda"} if dataset_defaults is None else dataset_defaults,
        setting_defaults=UNIDA_LOADER_DEFAULTS if setting_defaults is None else setting_defaults,
        format=format,
        setting_name="unida",
    )
