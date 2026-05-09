"""miniDomainNet loading helpers built on top of prepared DomainNet data."""

from __future__ import annotations

from pathlib import Path

from dabench.data.common import load_prepared_dataset_dict
from dabench.storage.paths import get_dataset_field_path
from dabench.utils.imports import require_datasets_for_loading

MINIDOMAINNET_DOMAINS = ("clipart", "painting", "real", "sketch")
MINIDOMAINNET_SPLITS = ("train", "test")
_CACHE: dict[tuple[str, str, bool, str], object] = {}


def _normalize_image_path(value: str) -> str:
    return value.strip().replace("\\", "/").lstrip("./")


def _split_dir_for_dataset(name: str, *, dataset_root: Path) -> Path:
    configured = get_dataset_field_path(name, "split_dir")
    if configured is not None:
        return configured
    fallback = dataset_root / "splits_mini"
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError(
        f"miniDomainNet split directory is not configured for dataset {name!r}. "
        f"Set `split_dir` in dabench config or place `splits_mini/` under {dataset_root}."
    )


def _load_split_paths(split_dir: Path, *, split: str) -> set[str]:
    selected: set[str] = set()
    missing: list[str] = []
    for domain in MINIDOMAINNET_DOMAINS:
        split_file = split_dir / f"{domain}_{split}.txt"
        if not split_file.is_file():
            missing.append(str(split_file))
            continue
        for line in split_file.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            rel_path = raw.split()[0]
            selected.add(_normalize_image_path(rel_path))
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"miniDomainNet split files are missing: {joined}")
    if not selected:
        raise ValueError(f"miniDomainNet split {split!r} under {split_dir} is empty.")
    return selected


def _filter_by_paths(dataset, *, selected_paths: set[str]):
    normalized_paths = [_normalize_image_path(path) for path in dataset["image_path"]]
    indices = [index for index, path in enumerate(normalized_paths) if path in selected_paths]
    if not indices:
        raise ValueError("miniDomainNet split selection produced an empty dataset.")
    return dataset.select(indices)


def load_mini_domainnet_dataset_dict(path: str | Path, *, decode: bool = True, dataset_name: str = "minidomainnet"):
    _Dataset, DatasetDict, _Image, _ClassLabel, concatenate_datasets = require_datasets_for_loading()
    root = Path(path).expanduser().resolve()
    split_dir = _split_dir_for_dataset(dataset_name, dataset_root=root)
    cache_key = (str(root), str(split_dir), bool(decode), dataset_name)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    prepared = load_prepared_dataset_dict(root, decode=decode)
    combined = concatenate_datasets([prepared["train"], prepared["test"]])
    datasets_by_split = {}
    for split_name in MINIDOMAINNET_SPLITS:
        selected_paths = _load_split_paths(split_dir, split=split_name)
        datasets_by_split[split_name] = _filter_by_paths(combined, selected_paths=selected_paths)
    dataset_dict = DatasetDict(datasets_by_split)
    _CACHE[cache_key] = dataset_dict
    return dataset_dict


__all__ = ["MINIDOMAINNET_DOMAINS", "load_mini_domainnet_dataset_dict"]
