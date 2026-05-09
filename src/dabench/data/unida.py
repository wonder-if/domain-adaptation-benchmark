"""Universal domain adaptation dataset helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dabench.data.dataset import load_view
from dabench.data.loader import build_loader


_DATASET_NAME_ALIASES = {
    "office": "office-31",
    "office-31": "office-31",
    "office31": "office-31",
    "officehome": "office-home",
    "office-home": "office-home",
    "office_home": "office-home",
    "domainnet": "domainnet",
    "visda": "visda-2017",
    "visda-2017": "visda-2017",
    "visda2017": "visda-2017",
}

_TASK_MAP = {
    "office": {
        "a": "amazon",
        "d": "dslr",
        "w": "webcam",
    },
    "officehome": {
        "A": "Art",
        "C": "Clipart",
        "P": "Product",
        "R": "Real",
    },
    "visda": {
        "S": "train",
        "R": "validation",
    },
    "domainnet": {
        "c": "clipart",
        "i": "infograph",
        "p": "painting",
        "q": "quickdraw",
        "r": "real",
        "s": "sketch",
    },
}

_DATASET_CLASS_COUNTS = {
    "office": 31,
    "officehome": 65,
    "domainnet": 345,
    "visda": 12,
}


@dataclass(frozen=True)
class UniDATask:
    dataset: str
    source_domain: str
    target_domain: str
    source_view: dict[str, str | None]
    target_view: dict[str, str | None]
    eval_view: dict[str, str | None]
    class_split: dict[str, list[int]]


def _normalize_dataset_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-")
    try:
        return _DATASET_NAME_ALIASES[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_DATASET_NAME_ALIASES))
        raise ValueError(f"Unsupported UniDA dataset {name!r}. Available aliases: {available}") from exc


def _normalize_task_dataset_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "")
    if normalized in {"office", "office31", "office-31".replace("-", "")}:
        return "office"
    if normalized in {"officehome", "office-home".replace("-", "")}:
        return "officehome"
    if normalized in {"domainnet"}:
        return "domainnet"
    if normalized in {"visda", "visda2017", "visda-2017".replace("-", "")}:
        return "visda"
    raise ValueError(f"Unsupported UniDA task dataset {name!r}.")


def get_task(conf_dataset_name: str, conf_dataset_task: str) -> tuple[str, str]:
    dataset_name = _normalize_task_dataset_name(conf_dataset_name)
    if len(conf_dataset_task) != 2:
        raise ValueError(f"UniDA task must be a 2-character code, got {conf_dataset_task!r}.")
    source = conf_dataset_task[0]
    target = conf_dataset_task[1]
    try:
        task_source = _TASK_MAP[dataset_name][source]
        task_target = _TASK_MAP[dataset_name][target]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported task code {conf_dataset_task!r} for dataset {conf_dataset_name!r}."
        ) from exc
    return task_source, task_target


def make_class_split(dataset_name: str, shared: int, source_private: int, target_private: int) -> dict[str, list[int]]:
    normalized = _normalize_task_dataset_name(dataset_name)
    num_class = _DATASET_CLASS_COUNTS[normalized]
    total = shared + source_private + target_private
    if num_class < total:
        raise RuntimeError(
            f"shared/source_private/target_private = {shared}/{source_private}/{target_private}. "
            f"Total number of splits exceeds class count of dataset {dataset_name!r} ({num_class})."
        )

    class_split = {
        "shared": list(range(shared)),
        "source_private": list(range(shared, shared + source_private)),
        "target_private": list(range(shared + source_private, shared + source_private + target_private)),
    }
    if normalized == "office" and source_private == 0 and target_private != 0:
        class_split["target_private"] = list(range(num_class - target_private, num_class))
    return class_split


def _label_names(dataset) -> list[str]:
    feature = dataset.features.get("label")
    names = getattr(feature, "names", None)
    if names is None:
        labels = sorted({int(label) for label in dataset["label"] if int(label) >= 0})
        return [str(label) for label in labels]
    return list(names)


def _filter_labels(dataset, labels: list[int]):
    allowed = set(labels)
    indices = [index for index, label in enumerate(dataset["label"]) if int(label) in allowed]
    if not indices:
        raise ValueError("Requested UniDA class filter produced an empty subset.")
    return dataset.select(indices)


def _count_labels(labels: list[int], class_split: dict[str, list[int]]) -> tuple[int, int, int]:
    shared_set = set(class_split["shared"])
    source_private_set = set(class_split["source_private"])
    target_private_set = set(class_split["target_private"])
    num_shared = 0
    num_source_private = 0
    num_target_private = 0
    for label in labels:
        if label in shared_set:
            num_shared += 1
        elif label in source_private_set:
            num_source_private += 1
        elif label in target_private_set:
            num_target_private += 1
    return num_shared, num_source_private, num_target_private


def _classnames_split(lab2cname: dict[str, str], class_split: dict[str, list[int]], *, dataset_name: str) -> dict[str, list[str]]:
    output = {
        "shared": [lab2cname[str(label)] for label in class_split["shared"] if str(label) in lab2cname],
        "source_private": [lab2cname[str(label)] for label in class_split["source_private"] if str(label) in lab2cname],
        "source": [
            lab2cname[str(label)]
            for label in class_split["shared"] + class_split["source_private"]
            if str(label) in lab2cname
        ],
    }
    if dataset_name != "domainnet":
        output["target_private"] = [
            lab2cname[str(label)] for label in class_split["target_private"] if str(label) in lab2cname
        ]
        output["target"] = [
            lab2cname[str(label)]
            for label in class_split["shared"] + class_split["target_private"]
            if str(label) in lab2cname
        ]
    return output


def _build_task(
    *,
    dataset: str,
    source_domain: str,
    target_domain: str,
    class_split: dict[str, list[int]],
) -> UniDATask:
    normalized = _normalize_dataset_name(dataset)
    if normalized == "office-31":
        source_view = {"domain": source_domain, "split": None}
        target_view = {"domain": target_domain, "split": None}
        eval_view = {"domain": target_domain, "split": None}
    elif normalized == "office-home":
        source_view = {"domain": "Real World" if source_domain == "Real" else source_domain, "split": None}
        target_view = {"domain": "Real World" if target_domain == "Real" else target_domain, "split": None}
        eval_view = dict(target_view)
    elif normalized == "domainnet":
        source_view = {"domain": source_domain, "split": "train"}
        target_view = {"domain": target_domain, "split": "train"}
        eval_view = {"domain": target_domain, "split": "train"}
    elif normalized == "visda-2017":
        source_view = {"domain": "synthetic", "split": "train"}
        target_view = {"domain": "real", "split": "validation"}
        eval_view = {"domain": "real", "split": "validation"}
    else:
        raise ValueError(f"Unsupported UniDA dataset routing: {dataset!r}")
    return UniDATask(
        dataset=normalized,
        source_domain=source_domain,
        target_domain=target_domain,
        source_view=source_view,
        target_view=target_view,
        eval_view=eval_view,
        class_split=class_split,
    )


def load_unida_views(
    *,
    dataset: str,
    task: str,
    shared: int,
    source_private: int,
    target_private: int,
    decode: bool = True,
):
    task_dataset = _normalize_task_dataset_name(dataset)
    task_source, task_target = get_task(task_dataset, task)
    class_split = make_class_split(task_dataset, shared, source_private, target_private)
    task_spec = _build_task(
        dataset=dataset,
        source_domain=task_source,
        target_domain=task_target,
        class_split=class_split,
    )

    source_full = load_view(
        task_spec.dataset,
        domain=task_spec.source_view["domain"],
        split=task_spec.source_view["split"],
        format="hf",
        decode=decode,
    )
    target_full = load_view(
        task_spec.dataset,
        domain=task_spec.target_view["domain"],
        split=task_spec.target_view["split"],
        format="hf",
        decode=decode,
    )
    eval_full = load_view(
        task_spec.dataset,
        domain=task_spec.eval_view["domain"],
        split=task_spec.eval_view["split"],
        format="hf",
        decode=decode,
    )

    source_labels = class_split["shared"] + class_split["source_private"]
    target_labels = class_split["shared"] + class_split["target_private"]
    source_dataset = _filter_labels(source_full, source_labels)
    target_dataset = _filter_labels(target_full, target_labels)
    eval_dataset = _filter_labels(eval_full, target_labels)

    label_names = _label_names(source_full if len(source_full) >= len(target_full) else target_full)
    lab2cname = {str(index): name.replace("_", " ") for index, name in enumerate(label_names)}
    target_counts = _count_labels([int(label) for label in target_dataset["label"]], class_split)
    source_counts = _count_labels([int(label) for label in source_dataset["label"]], class_split)
    metadata = {
        "dataset": task_spec.dataset,
        "task": task,
        "task_source": task_source,
        "task_target": task_target,
        "class_split": class_split,
        "lab2cname": lab2cname,
        "classnames_all": [lab2cname[str(index)] for index in range(len(label_names)) if str(index) in lab2cname],
        "classnames_split": _classnames_split(lab2cname, class_split, dataset_name=task_spec.dataset),
        "num_class": shared + source_private + target_private,
        "num_classes": shared + source_private + target_private,
        "unknown": shared + source_private,
        "source_counts": {
            "shared": source_counts[0],
            "source_private": source_counts[1],
            "target_private": source_counts[2],
        },
        "target_counts": {
            "shared": target_counts[0],
            "source_private": target_counts[1],
            "target_private": target_counts[2],
        },
    }
    return source_dataset, target_dataset, eval_dataset, metadata


def load_unida(
    *,
    dataset: str,
    task: str,
    shared: int,
    source_private: int,
    target_private: int,
    source_batch_size: int,
    target_batch_size: int | None = None,
    test_batch_size: int | None = None,
    source_transform=None,
    target_transform=None,
    test_transform=None,
    num_workers: int = 4,
    pin_memory: bool | None = None,
    decode: bool = True,
) -> dict[str, Any]:
    source_dataset, target_dataset, eval_dataset, metadata = load_unida_views(
        dataset=dataset,
        task=task,
        shared=shared,
        source_private=source_private,
        target_private=target_private,
        decode=decode,
    )
    return {
        "source_train_dataset": source_dataset,
        "target_train_dataset": target_dataset,
        "test_dataset": eval_dataset,
        "source_train_loader": build_loader(
            source_dataset,
            batch_size=source_batch_size,
            mode="train",
            transform=source_transform,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "target_train_loader": build_loader(
            target_dataset,
            batch_size=target_batch_size or source_batch_size,
            mode="train",
            transform=target_transform,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test_loader": build_loader(
            eval_dataset,
            batch_size=test_batch_size or target_batch_size or source_batch_size,
            mode="test",
            transform=test_transform,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "metadata": metadata,
    }
