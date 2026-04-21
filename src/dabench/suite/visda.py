"""VisDA-2017 UDA suite helpers."""

from __future__ import annotations

from dabench.suite._common import build_uda_suite_items

VISDA_DOMAINS = ("synthetic", "real")

VISDA_UDA_DATASET_DEFAULTS = {
    "dataset": "visda-2017",
    "decode": True,
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
    )
