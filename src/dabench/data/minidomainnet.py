"""miniDomainNet loading helpers built from DomainNet images and mini split files."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from dabench.data.common import load_prepared_dataset_dict
from dabench.storage.paths import get_dataset_field_path
from dabench.utils.imports import require_datasets_for_loading

MINIDOMAINNET_DOMAINS = ("clipart", "painting", "real", "sketch")
MINIDOMAINNET_SPLITS = ("train", "test")
_CACHE: dict[tuple[str, str, str | None, bool, str], object] = {}
_CLASS_NAMES_CACHE: dict[tuple[str, str], tuple[str, ...]] = {}


def normalize_domainnet_image_path(value: str) -> str:
    """Normalize DomainNet paths from split files or imagefolder metadata."""

    normalized = value.strip().replace("\\", "/").lstrip("./")
    parts = [part for part in normalized.split("/") if part and part != "."]
    if "domainnet" in parts:
        parts = parts[parts.index("domainnet") + 1 :]
    if len(parts) >= 4 and parts[1] in MINIDOMAINNET_SPLITS:
        parts = [parts[0], *parts[2:]]
    return "/".join(parts)


def _normalize_image_path(value: str) -> str:
    return normalize_domainnet_image_path(value)


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


def _image_root_for_dataset(name: str) -> Path | None:
    configured = get_dataset_field_path(name, "image_root")
    if configured is not None:
        return configured
    return None


def _resolve_split_domains(domains: Iterable[str] | None) -> tuple[str, ...]:
    if domains is None:
        return MINIDOMAINNET_DOMAINS
    resolved = tuple(str(domain) for domain in domains)
    invalid = [domain for domain in resolved if domain not in MINIDOMAINNET_DOMAINS]
    if invalid:
        available = ", ".join(MINIDOMAINNET_DOMAINS)
        raise ValueError(f"Unsupported miniDomainNet domain(s): {invalid}. Available domains: {available}")
    return resolved


def _load_split_path_sequence(split_dir: Path, *, split: str, domains: Iterable[str] | None = None) -> tuple[str, ...]:
    selected: list[str] = []
    missing: list[str] = []
    for domain in _resolve_split_domains(domains):
        split_file = split_dir / f"{domain}_{split}.txt"
        if not split_file.is_file():
            missing.append(str(split_file))
            continue
        for line in split_file.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            rel_path = raw.split()[0]
            selected.append(_normalize_image_path(rel_path))
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"miniDomainNet split files are missing: {joined}")
    if not selected:
        raise ValueError(f"miniDomainNet split {split!r} under {split_dir} is empty.")
    return tuple(selected)


def _load_split_records(split_dir: Path, *, split: str, domains: Iterable[str] | None = None) -> tuple[tuple[str, int | None], ...]:
    records: list[tuple[str, int | None]] = []
    missing: list[str] = []
    for domain in _resolve_split_domains(domains):
        split_file = split_dir / f"{domain}_{split}.txt"
        if not split_file.is_file():
            missing.append(str(split_file))
            continue
        for line in split_file.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split()
            label = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else None
            records.append((_normalize_image_path(parts[0]), label))
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"miniDomainNet split files are missing: {joined}")
    if not records:
        raise ValueError(f"miniDomainNet split {split!r} under {split_dir} is empty.")
    if all(label is not None for _path, label in records):
        records.sort(key=lambda item: (int(item[1]), item[0]))
    return tuple(records)


def _class_name_from_path(path: str) -> str:
    parts = _normalize_image_path(path).split("/")
    if len(parts) < 3:
        raise ValueError(f"miniDomainNet split path must have domain/class/file form: {path!r}")
    return parts[1]


def _class_names_from_split_dir(split_dir: Path) -> tuple[str, ...]:
    cache_key = (str(split_dir), "class_names")
    cached = _CLASS_NAMES_CACHE.get(cache_key)
    if cached is not None:
        return cached

    label_to_name: dict[int, str] = {}
    for split in MINIDOMAINNET_SPLITS:
        for path, label in _load_split_records(split_dir, split=split):
            if label is None:
                raise ValueError("miniDomainNet split files must include integer labels.")
            class_name = _class_name_from_path(path)
            existing = label_to_name.get(label)
            if existing is not None and existing != class_name:
                raise ValueError(
                    f"miniDomainNet label {label} maps to both {existing!r} and {class_name!r}."
                )
            label_to_name[label] = class_name
    if not label_to_name:
        raise ValueError(f"miniDomainNet split directory {split_dir} contains no labeled records.")
    labels = sorted(label_to_name)
    expected = list(range(labels[-1] + 1))
    if labels != expected:
        raise ValueError(f"miniDomainNet labels must be contiguous from 0. Found labels: {labels[:5]}...{labels[-5:]}")
    class_names = tuple(label_to_name[index] for index in expected)
    _CLASS_NAMES_CACHE[cache_key] = class_names
    return class_names


def get_mini_domainnet_class_names(
    path: str | Path | None = None,
    *,
    dataset_name: str = "minidomainnet",
) -> tuple[str, ...]:
    """Return miniDomainNet class names ordered by split-file label id."""

    root = Path(path).expanduser().resolve() if path is not None else Path(".").resolve()
    split_dir = _split_dir_for_dataset(dataset_name, dataset_root=root)
    return _class_names_from_split_dir(split_dir)


def _load_split_paths(split_dir: Path, *, split: str, domains: Iterable[str] | None = None) -> set[str]:
    return set(_load_split_path_sequence(split_dir, split=split, domains=domains))


def load_mini_domainnet_split_paths(
    path: str | Path,
    *,
    split: str,
    dataset_name: str = "minidomainnet",
    domains: Iterable[str] | None = None,
) -> set[str]:
    """Load normalized miniDomainNet image paths for a split/domain selection."""

    root = Path(path).expanduser().resolve()
    split_dir = _split_dir_for_dataset(dataset_name, dataset_root=root)
    return _load_split_paths(split_dir, split=split, domains=domains)


def load_mini_domainnet_split_path_sequence(
    path: str | Path,
    *,
    split: str,
    dataset_name: str = "minidomainnet",
    domains: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Load normalized miniDomainNet image paths while preserving split-file order."""

    root = Path(path).expanduser().resolve()
    split_dir = _split_dir_for_dataset(dataset_name, dataset_root=root)
    return _load_split_path_sequence(split_dir, split=split, domains=domains)


def load_mini_domainnet_split_records(
    path: str | Path,
    *,
    split: str,
    dataset_name: str = "minidomainnet",
    domains: Iterable[str] | None = None,
) -> tuple[tuple[str, int | None], ...]:
    """Load normalized miniDomainNet split records as ``(image_path, label)`` pairs."""

    root = Path(path).expanduser().resolve()
    split_dir = _split_dir_for_dataset(dataset_name, dataset_root=root)
    return _load_split_records(split_dir, split=split, domains=domains)


def _filter_by_paths(dataset, *, selected_paths: set[str]):
    normalized_paths = [_normalize_image_path(path) for path in dataset["image_path"]]
    indices = [index for index, path in enumerate(normalized_paths) if path in selected_paths]
    if not indices:
        raise ValueError("miniDomainNet split selection produced an empty dataset.")
    return dataset.select(indices)


def _load_from_image_root(
    image_root: Path,
    split_dir: Path,
    *,
    decode: bool,
):
    Dataset, DatasetDict, Image, ClassLabel, _concat = require_datasets_for_loading()
    class_names = _class_names_from_split_dir(split_dir)
    datasets_by_split = {}
    for split_name in MINIDOMAINNET_SPLITS:
        records = []
        missing = []
        for rel_path, label in _load_split_records(split_dir, split=split_name):
            if label is None:
                raise ValueError("miniDomainNet split files must include integer labels.")
            image_path = image_root / rel_path
            if not image_path.is_file():
                missing.append(str(image_path))
                continue
            records.append(
                {
                    "image": str(image_path),
                    "label": int(label),
                    "domain": rel_path.split("/", 1)[0],
                    "image_path": rel_path,
                }
            )
        if missing:
            examples = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"miniDomainNet split {split_name!r} has {len(missing)} missing image files under "
                f"{image_root}. Examples: {examples}"
            )
        dataset = Dataset.from_list(records)
        dataset = dataset.cast_column("image", Image(decode=decode))
        dataset = dataset.cast_column("label", ClassLabel(names=list(class_names)))
        datasets_by_split[split_name] = dataset
    return DatasetDict(datasets_by_split)


def load_mini_domainnet_dataset_dict(path: str | Path, *, decode: bool = True, dataset_name: str = "minidomainnet"):
    _Dataset, DatasetDict, _Image, _ClassLabel, concatenate_datasets = require_datasets_for_loading()
    root = Path(path).expanduser().resolve()
    split_dir = _split_dir_for_dataset(dataset_name, dataset_root=root)
    image_root = _image_root_for_dataset(dataset_name)
    cache_key = (str(root), str(split_dir), str(image_root) if image_root is not None else None, bool(decode), dataset_name)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    if image_root is not None:
        dataset_dict = _load_from_image_root(image_root, split_dir, decode=decode)
    else:
        prepared = load_prepared_dataset_dict(root, decode=decode)
        combined = concatenate_datasets([prepared["train"], prepared["test"]])
        datasets_by_split = {}
        for split_name in MINIDOMAINNET_SPLITS:
            selected_paths = _load_split_paths(split_dir, split=split_name)
            datasets_by_split[split_name] = _filter_by_paths(combined, selected_paths=selected_paths)
        dataset_dict = DatasetDict(datasets_by_split)
    _CACHE[cache_key] = dataset_dict
    return dataset_dict


__all__ = [
    "MINIDOMAINNET_DOMAINS",
    "get_mini_domainnet_class_names",
    "load_mini_domainnet_dataset_dict",
    "load_mini_domainnet_split_path_sequence",
    "load_mini_domainnet_split_paths",
    "load_mini_domainnet_split_records",
    "normalize_domainnet_image_path",
]
