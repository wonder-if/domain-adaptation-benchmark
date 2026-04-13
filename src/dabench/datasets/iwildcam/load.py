"""Loading and inspection helpers for iWildCam."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dabench.datasets.common import build_torch_dataset, load_prepared_dataset_dict
from dabench.datasets.hf_download import inspect_prepared_dataset


def load_hf_dataset(
    *,
    path: str | Path,
    split: str | None = None,
    decode: bool = True,
    **kwargs: Any,
):
    dest = Path(path).expanduser().resolve()
    dataset_dict = load_prepared_dataset_dict(dest, decode=decode)
    resolved_split = split or "train"
    return dataset_dict[resolved_split]


def load_dataset(
    *,
    path: str | Path,
    split: str | None = None,
    transform=None,
    decode: bool = True,
    **kwargs: Any,
):
    dataset = load_hf_dataset(
        path=path,
        split=split,
        decode=decode,
        **kwargs,
    )
    return build_torch_dataset(
        dataset,
        transform=transform,
        label_column=None,
        domain_column=None,
        path_column=None,
    )


def load_uda(*args: Any, **kwargs: Any):
    raise NotImplementedError(
        "iWildCam UDA loading is not implemented yet. "
        "This dataset still needs metadata alignment before exposing source/target splits."
    )


def inspect_dataset(
    *,
    path: str | Path,
    split: str | None = None,
) -> dict[str, Any]:
    dest = Path(path).expanduser().resolve()
    state = inspect_prepared_dataset(dest)
    state["backend"] = "hf_prepared"
    return state
