"""Precomputed feature cache loading helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dabench.storage.manifest import get_manifest
from dabench.storage.paths import get_dataset_field_path
from dabench.utils.imports import require_datasets_for_loading, require_torch_for_loading

_FEATURE_CACHE_ENV = "DABENCH_FEATURE_CACHE_PATH"
_DEFAULT_FEATURE_CACHE_PATH = Path("/data/wyh/datasets/adaptation-of-clip_feature_cache")
_PREPARED_LABEL_NAMES_CACHE: dict[str, tuple[str, ...]] = {}


def _expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _normalize_component(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def _feature_cache_root(dataset: str, feature_cache_path: str | Path | None = None) -> Path:
    if feature_cache_path is not None:
        return _expand_path(feature_cache_path)

    configured_env = os.environ.get(_FEATURE_CACHE_ENV)
    if configured_env:
        return _expand_path(configured_env)

    configured = get_dataset_field_path(dataset, "feature_cache_path")
    if configured is not None:
        return configured

    manifest = get_manifest(dataset)
    base_dataset = manifest.get("prepared", {}).get("base_dataset")
    if isinstance(base_dataset, str):
        configured = get_dataset_field_path(base_dataset, "feature_cache_path")
        if configured is not None:
            return configured

    if _DEFAULT_FEATURE_CACHE_PATH.is_dir():
        return _DEFAULT_FEATURE_CACHE_PATH

    raise FileNotFoundError(
        "No feature cache path configured. Set DABENCH_FEATURE_CACHE_PATH, pass "
        "`feature_cache_path`, or add `feature_cache_path` to the dataset path config."
    )


def resolve_feature_cache_root(dataset: str, feature_cache_path: str | Path | None = None) -> Path:
    """Resolve the feature cache root for a dataset."""

    return _feature_cache_root(dataset, feature_cache_path)


def _available_dataset_ids(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _cache_dataset_id(manifest: dict[str, Any]) -> str:
    return str(manifest["id"])


def _dataset_cache_dir(dataset: str, feature_cache_path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    manifest = get_manifest(dataset)
    dataset_id = str(manifest["id"])
    cache_dataset_id = _cache_dataset_id(manifest)

    root = _feature_cache_root(dataset_id, feature_cache_path)
    dataset_root = root / cache_dataset_id
    if dataset_root.is_dir():
        return manifest, dataset_root

    available = ", ".join(_available_dataset_ids(root)) or "<none>"
    raise FileNotFoundError(
        f"Missing feature cache dataset directory: {dataset_root}. Available cached datasets: {available}"
    )


def _resolve_domain_name(manifest: dict[str, Any], domain: str | int) -> str:
    if isinstance(domain, int):
        domains = manifest.get("prepared", {}).get("domains", [])
        if not isinstance(domains, list) or not (0 <= domain < len(domains)):
            raise ValueError(f"Domain index {domain!r} is not valid for dataset {manifest['id']!r}.")
        return str(domains[domain])
    return str(domain)


def _domain_cache_dir(dataset_root: Path, manifest: dict[str, Any], domain: str | int) -> tuple[str, Path]:
    requested = _resolve_domain_name(manifest, domain)
    exact = dataset_root / requested
    if exact.is_dir():
        return requested, exact

    available_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    normalized = {_normalize_component(path.name): path for path in available_dirs}
    matched = normalized.get(_normalize_component(requested))
    if matched is not None:
        return matched.name, matched

    available = ", ".join(path.name for path in available_dirs) or "<none>"
    raise FileNotFoundError(
        f"Missing feature cache domain {requested!r} under {dataset_root}. Available domains: {available}"
    )


def _cache_axis(manifest: dict[str, Any]) -> str:
    prepared = manifest.get("prepared", {})
    domains = prepared.get("domains")
    if isinstance(domains, list) and domains:
        return "domain"
    return "split"


def _select_split(dataset, split: str | None):
    _Dataset, DatasetDict, _Image, _ClassLabel, _concat = require_datasets_for_loading()
    if isinstance(dataset, DatasetDict):
        if split is None:
            splits = list(dataset.keys())
            if len(splits) == 1:
                return dataset[splits[0]]
            available = ", ".join(splits)
            raise ValueError(f"This feature cache requires an explicit split. Available splits: {available}")
        if split not in dataset:
            available = ", ".join(dataset.keys())
            raise ValueError(f"Unsupported feature cache split {split!r}. Available splits: {available}")
        return dataset[split]
    return dataset


def _ensure_domain_column(dataset, domain: str):
    if "domain" in dataset.column_names:
        return dataset
    return dataset.add_column("domain", [domain] * len(dataset))


def _configured_dataset_root(dataset: str) -> Path:
    configured = get_dataset_field_path(dataset, "path")
    if configured is not None:
        return configured

    manifest = get_manifest(dataset)
    base_dataset = manifest.get("prepared", {}).get("base_dataset")
    if isinstance(base_dataset, str):
        configured = get_dataset_field_path(base_dataset, "path")
        if configured is not None:
            return configured

    raise FileNotFoundError(
        f"No local path configured for dataset {dataset!r}. Set `path` in the dabench path config."
    )


def _feature_source_paths(feature_root: Path) -> list[str] | None:
    info_path = feature_root / "dataset_info.json"
    if not info_path.is_file():
        return None
    payload = json.loads(info_path.read_text(encoding="utf-8"))
    checksums = payload.get("download_checksums")
    if not isinstance(checksums, dict):
        return None

    from dabench.data.minidomainnet import normalize_domainnet_image_path

    return [normalize_domainnet_image_path(path) for path in checksums]


def _domain_value_matches(dataset, value, domain_name: str) -> bool:
    feature = dataset.features.get("domain")
    names = getattr(feature, "names", None)
    if names is not None and domain_name in names:
        return value == names.index(domain_name)
    return str(value) == domain_name


def _minidomainnet_prepared_records(
    *,
    dataset_root: Path,
    domain_name: str,
    split: str,
) -> tuple[tuple[str, int | None], ...]:
    from dabench.data.minidomainnet import (
        load_mini_domainnet_dataset_dict,
        load_mini_domainnet_split_records,
        normalize_domainnet_image_path,
    )

    split_records = load_mini_domainnet_split_records(
        dataset_root,
        split=split,
        dataset_name="minidomainnet",
        domains=(domain_name,),
    )
    if all(label is not None for _path, label in split_records):
        return split_records

    try:
        dataset_dict = load_mini_domainnet_dataset_dict(dataset_root, decode=False, dataset_name="minidomainnet")
    except FileNotFoundError:
        return split_records

    if split not in dataset_dict:
        available = ", ".join(dataset_dict.keys())
        raise ValueError(f"Unsupported miniDomainNet split {split!r}. Available splits: {available}")
    dataset = dataset_dict[split]
    if "image_path" not in dataset.column_names:
        raise ValueError("Prepared miniDomainNet data must expose an `image_path` column.")

    labels = dataset["label"] if "label" in dataset.column_names else [None] * len(dataset)
    domain_values = dataset["domain"] if "domain" in dataset.column_names else [domain_name] * len(dataset)
    records = []
    for path, label, domain_value in zip(dataset["image_path"], labels, domain_values, strict=True):
        if _domain_value_matches(dataset, domain_value, domain_name):
            records.append((normalize_domainnet_image_path(path), int(label) if label is not None else None))
    if not records:
        raise ValueError(
            f"Prepared miniDomainNet split {split!r} produced an empty view for domain {domain_name!r}."
        )
    return tuple(records)


def _filter_minidomainnet_feature_view(loaded, *, feature_root: Path, domain_name: str, split: str | None):
    if split is None:
        raise ValueError("miniDomainNet feature loading requires an explicit `split`.")

    from dabench.data.minidomainnet import normalize_domainnet_image_path

    dataset_root = _configured_dataset_root("minidomainnet")
    selected_records = _minidomainnet_prepared_records(
        dataset_root=dataset_root,
        domain_name=domain_name,
        split=split,
    )
    selected_paths = [path for path, _label in selected_records]
    if "image_path" in loaded.column_names:
        source_paths = [normalize_domainnet_image_path(path) for path in loaded["image_path"]]
    else:
        source_paths = _feature_source_paths(feature_root)
        if source_paths is None:
            raise ValueError(
                f"miniDomainNet feature loading needs source image paths, but {feature_root} "
                "does not expose an `image_path` column or dataset_info.json download_checksums."
            )
    if len(source_paths) != len(loaded):
        raise ValueError(
            f"Source path metadata length mismatch for {feature_root}: "
            f"{len(source_paths)} paths for {len(loaded)} feature rows."
        )

    source_set = set(source_paths)
    missing_paths = sorted(set(selected_paths) - source_set)
    if missing_paths:
        examples = ", ".join(missing_paths[:5])
        raise ValueError(
            f"miniDomainNet split {split!r} contains {len(missing_paths)} paths missing from "
            f"the feature cache for domain {domain_name!r}. Examples: {examples}"
        )

    path_to_index = {path: index for index, path in enumerate(source_paths)}
    indices = [path_to_index[path] for path in selected_paths]
    if not indices:
        raise ValueError(
            f"miniDomainNet split {split!r} produced an empty feature view for domain {domain_name!r}."
        )

    selected = loaded.select(indices)
    selected_labels = [label for _path, label in selected_records]
    if all(label is not None for label in selected_labels):
        if "label" in selected.column_names:
            selected = selected.remove_columns("label")
        selected = selected.add_column("label", selected_labels)
    if "image_path" not in selected.column_names:
        selected = selected.add_column("image_path", [source_paths[index] for index in indices])
    return selected


def _cache_label_names(domain_root: Path, feature_model: str) -> tuple[str, ...] | None:
    info_path = domain_root / "image_features" / feature_model / "dataset_info.json"
    if not info_path.is_file():
        return None
    payload = json.loads(info_path.read_text(encoding="utf-8"))
    label_feature = payload.get("features", {}).get("label", {})
    names = label_feature.get("names")
    if not isinstance(names, list):
        return None
    return tuple(str(name) for name in names)


def _prepared_label_names(dataset: str) -> tuple[str, ...] | None:
    cached = _PREPARED_LABEL_NAMES_CACHE.get(dataset)
    if cached is not None:
        return cached

    if str(get_manifest(dataset)["id"]) == "minidomainnet":
        from dabench.data.minidomainnet import get_mini_domainnet_class_names

        resolved = get_mini_domainnet_class_names(dataset_name="minidomainnet")
        _PREPARED_LABEL_NAMES_CACHE[dataset] = resolved
        return resolved

    dataset_root = _configured_dataset_root(dataset)
    info_path = dataset_root / "dataset_info.json"
    if info_path.is_file():
        payload = json.loads(info_path.read_text(encoding="utf-8"))
        label_feature = payload.get("features", {}).get("label", {})
        names = label_feature.get("names")
        if isinstance(names, list):
            resolved = tuple(str(name) for name in names)
            _PREPARED_LABEL_NAMES_CACHE[dataset] = resolved
            return resolved

    from dabench.data.common import load_prepared_dataset_dict

    try:
        prepared = load_prepared_dataset_dict(dataset_root, decode=False)
    except FileNotFoundError:
        return None
    for split_dataset in prepared.values():
        feature = split_dataset.features.get("label")
        names = getattr(feature, "names", None)
        if names is not None:
            resolved = tuple(str(name) for name in names)
            _PREPARED_LABEL_NAMES_CACHE[dataset] = resolved
            return resolved
    return None


def _remap_minidomainnet_text_features(payload, *, domain_root: Path, feature_model: str):
    cache_names = _cache_label_names(domain_root, feature_model)
    if cache_names is None:
        return payload
    target_names = _prepared_label_names("minidomainnet")
    if target_names is None or cache_names == target_names:
        return payload

    cache_index = {name: index for index, name in enumerate(cache_names)}
    if any(name not in cache_index for name in target_names):
        return payload
    indices = [cache_index[name] for name in target_names]
    if isinstance(payload, dict):
        text_features = payload.get("text_features")
        if text_features is None or len(text_features) != len(cache_names):
            return payload
        remapped = dict(payload)
        remapped["text_features"] = text_features[indices]
        return remapped
    if len(payload) != len(cache_names):
        return payload
    return payload[indices]


def load_feature_view(
    dataset: str,
    *,
    domain: str | int,
    feature_model: str,
    split: str | None = None,
    feature_cache_path: str | Path | None = None,
):
    """Load cached image features for one dataset/domain/model view."""

    if not feature_model:
        raise ValueError("load_feature_view requires a non-empty `feature_model`.")

    _Dataset, _DatasetDict, _Image, _ClassLabel, _concat = require_datasets_for_loading()
    from datasets import load_from_disk  # type: ignore

    manifest, dataset_root = _dataset_cache_dir(dataset, feature_cache_path)
    domain_name, domain_root = _domain_cache_dir(dataset_root, manifest, domain)
    feature_root = domain_root / "image_features" / feature_model
    if not feature_root.is_dir():
        available_root = domain_root / "image_features"
        available = (
            ", ".join(sorted(path.name for path in available_root.iterdir() if path.is_dir()))
            if available_root.is_dir()
            else "<none>"
        )
        raise FileNotFoundError(
            f"Missing image feature cache for model {feature_model!r} under {domain_root}. "
            f"Available models: {available}"
        )

    loaded = load_from_disk(str(feature_root))
    loaded = _select_split(loaded, split)
    if "image_features" not in loaded.column_names:
        available = ", ".join(loaded.column_names)
        raise ValueError(
            f"Feature cache {feature_root} does not expose an `image_features` column. "
            f"Available columns: {available}"
        )
    if str(manifest["id"]) == "minidomainnet":
        loaded = _filter_minidomainnet_feature_view(
            loaded,
            feature_root=feature_root,
            domain_name=domain_name,
            split=split,
        )
    return _ensure_domain_column(loaded, domain_name)


def _available_image_models(domain_root: Path) -> list[str]:
    image_root = domain_root / "image_features"
    if not image_root.is_dir():
        return []
    return sorted(path.name for path in image_root.iterdir() if path.is_dir())


def _available_text_prompts(domain_root: Path, feature_model: str) -> list[str]:
    text_root = domain_root / "text_features"
    if not text_root.is_dir():
        return []
    prefix = f"{feature_model}-"
    prompts = []
    for path in sorted(text_root.glob("*.pt")):
        stem = path.stem
        if stem.startswith(prefix):
            prompts.append(stem[len(prefix) :])
    return prompts


def _text_feature_candidates(feature_model: str, prompt: str) -> list[str]:
    prompts = [prompt]
    if prompt.endswith("."):
        prompts.append(prompt.rstrip("."))
    else:
        prompts.append(f"{prompt}.")

    filenames = []
    for candidate in dict.fromkeys(prompts):
        filename = f"{feature_model}-{candidate}"
        if not filename.endswith(".pt"):
            filename = f"{filename}.pt"
        filenames.append(filename)
    return filenames


def load_text_features(
    dataset: str,
    *,
    domain: str | int | None = None,
    split: str | None = None,
    feature_model: str,
    prompt: str,
    feature_cache_path: str | Path | None = None,
    map_location: str = "cpu",
):
    """Load cached text features for one dataset/domain/model/prompt."""

    if not feature_model:
        raise ValueError("load_text_features requires a non-empty `feature_model`.")
    if not prompt:
        raise ValueError("load_text_features requires a non-empty `prompt`.")
    cache_domain = domain if domain is not None else split
    if cache_domain is None:
        raise ValueError("load_text_features requires `domain`, or `split` for split-only datasets.")

    torch, _DataLoader, _IterableDataset = require_torch_for_loading()
    manifest, dataset_root = _dataset_cache_dir(dataset, feature_cache_path)
    _domain_name, domain_root = _domain_cache_dir(dataset_root, manifest, cache_domain)
    text_root = domain_root / "text_features"
    text_path = next(
        (text_root / name for name in _text_feature_candidates(feature_model, prompt) if (text_root / name).is_file()),
        None,
    )
    if text_path is None:
        available = ", ".join(_available_text_prompts(domain_root, feature_model)) or "<none>"
        raise FileNotFoundError(
            f"Missing text feature cache for model {feature_model!r} and prompt {prompt!r} under {text_root}. "
            f"Available prompts for this model: {available}"
        )
    payload = torch.load(text_path, map_location=map_location)
    if str(manifest["id"]) == "minidomainnet":
        payload = _remap_minidomainnet_text_features(payload, domain_root=domain_root, feature_model=feature_model)
    return payload


def load_feature_decode_errors(
    dataset: str,
    *,
    domain: str | int | None = None,
    split: str | None = None,
    feature_model: str,
    feature_cache_path: str | Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Load image decode error records saved beside a feature cache."""

    if not feature_model:
        raise ValueError("load_feature_decode_errors requires a non-empty `feature_model`.")
    cache_domain = domain if domain is not None else split
    if cache_domain is None:
        raise ValueError("load_feature_decode_errors requires `domain`, or `split` for split-only datasets.")

    manifest, dataset_root = _dataset_cache_dir(dataset, feature_cache_path)
    _domain_name, domain_root = _domain_cache_dir(dataset_root, manifest, cache_domain)
    return _read_decode_errors(domain_root / "image_features" / feature_model)


def _read_decode_errors(feature_root: Path) -> tuple[dict[str, Any], ...]:
    path = feature_root / "decode_errors.json"
    if not path.is_file():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Invalid decode error payload: {path}")
    records = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid decode error record in {path}")
        records.append(item)
    return tuple(records)


def _feature_cache_stats(feature_root: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {"decode_errors": len(_read_decode_errors(feature_root))}
    try:
        from datasets import load_from_disk  # type: ignore
    except ImportError:
        return stats

    loaded = load_from_disk(str(feature_root))
    loaded = _select_split(loaded, None)
    stats["rows"] = len(loaded)
    if len(loaded) and "image_features" in loaded.column_names:
        stats["feature_dim"] = len(loaded[0]["image_features"])
    return stats


def list_feature_caches(
    dataset: str | None = None,
    *,
    feature_cache_path: str | Path | None = None,
    include_stats: bool = False,
) -> list[dict[str, Any]]:
    """List cached feature datasets, domains/splits, image models, and text prompts."""

    root = _feature_cache_root(dataset or "domainnet", feature_cache_path)
    dataset_dirs: list[tuple[Path, str, str, dict[str, Any] | None]] = []
    manifests_by_id: dict[str, dict[str, Any] | None] = {}
    if dataset is None:
        dataset_dirs = [
            (root / dataset_id, dataset_id, dataset_id, None) for dataset_id in _available_dataset_ids(root)
        ]
    else:
        manifest = get_manifest(dataset)
        dataset_id = str(manifest["id"])
        cache_dataset_id = _cache_dataset_id(manifest)
        manifests_by_id[dataset_id] = manifest
        dataset_dirs = [(root / cache_dataset_id, dataset_id, cache_dataset_id, manifest)]

    records: list[dict[str, Any]] = []
    for dataset_dir, record_dataset_id, cache_dataset_id, requested_manifest in dataset_dirs:
        if not dataset_dir.is_dir():
            continue
        manifest = requested_manifest or manifests_by_id.get(record_dataset_id)
        if manifest is None:
            try:
                manifest = get_manifest(dataset_dir.name)
            except ValueError:
                manifest = None
        axis = _cache_axis(manifest) if manifest is not None else "domain"
        allowed_domains = None
        if manifest is not None:
            domains = manifest.get("prepared", {}).get("domains")
            if isinstance(domains, list) and domains:
                allowed_domains = {_normalize_component(str(domain)) for domain in domains}
        for domain_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            if allowed_domains is not None and _normalize_component(domain_dir.name) not in allowed_domains:
                continue
            image_models = _available_image_models(domain_dir)
            text_prompts = {}
            image_model_stats = {}
            for model in image_models:
                prompts = _available_text_prompts(domain_dir, model)
                if prompts:
                    text_prompts[model] = prompts
                if include_stats:
                    image_model_stats[model] = _feature_cache_stats(domain_dir / "image_features" / model)
            record = {
                "dataset": record_dataset_id,
                "axis": axis,
                "cache_key": domain_dir.name,
                "domain": domain_dir.name,
                "split": domain_dir.name if axis == "split" else None,
                "image_models": tuple(image_models),
                "text_prompts": text_prompts,
            }
            if cache_dataset_id != record_dataset_id:
                record["cache_dataset"] = cache_dataset_id
            if include_stats:
                record["image_model_stats"] = image_model_stats
            records.append(record)
    return records


__all__ = [
    "list_feature_caches",
    "load_feature_decode_errors",
    "load_feature_view",
    "load_text_features",
    "resolve_feature_cache_root",
]
