"""Shared helpers for result recording and rendering."""

from __future__ import annotations

from datetime import datetime, timezone

DOMAIN_ALIASES = {
    "office-31": "office-31",
    "office31": "office-31",
    "office-home": "office-home",
    "office_home": "office-home",
    "officehome": "office-home",
    "domainnet": "domainnet",
    "visda-2017": "visda-2017",
    "visda17": "visda-2017",
    "visda2017": "visda-2017",
}

TABLE_LAYOUTS = {
    "office-31": "transfer_pairs",
    "office-home": "transfer_pairs",
    "domainnet": "transfer_matrix",
    "visda-2017": "per_class",
}

DOMAIN_LABELS = {
    "office-31": {
        "amazon": "A",
        "dslr": "D",
        "webcam": "W",
    },
    "office-home": {
        "art": "Ar",
        "clipart": "Cl",
        "product": "Pr",
        "real_world": "Rw",
        "real world": "Rw",
    },
    "domainnet": {
        "clipart": "Clp",
        "infograph": "Inf",
        "painting": "Pnt",
        "quickdraw": "Qdr",
        "real": "Rel",
        "sketch": "Skt",
    },
    "visda-2017": {
        "synthetic": "Syn",
        "real": "Rel",
    },
}

VISDA_CLASS_ORDER = (
    ("aeroplane", "plane"),
    ("bicycle", "bicycle"),
    ("bus", "bus"),
    ("car", "car"),
    ("horse", "horse"),
    ("knife", "knife"),
    ("motorcycle", "mcycl"),
    ("person", "person"),
    ("plant", "plant"),
    ("skateboard", "sktbrd"),
    ("train", "train"),
    ("truck", "truck"),
)


def normalize_dataset_name(dataset: str) -> str:
    """Return the canonical dataset id used by dabench result payloads."""

    normalized = dataset.strip().lower().replace("_", "-")
    try:
        return DOMAIN_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(TABLE_LAYOUTS))
        raise ValueError(f"Unsupported dataset: {dataset!r}. Supported datasets: {supported}") from exc


def normalize_setting_name(setting: str) -> str:
    """Return the canonical setting id."""

    return setting.strip().lower().replace("_", "-")


def domain_code(dataset: str, domain: str) -> str:
    """Return a short display code for a dataset domain."""

    mapping = DOMAIN_LABELS[dataset]
    key = domain.strip().lower().replace("-", "_")
    try:
        return mapping[key]
    except KeyError:
        return domain


def iso_timestamp_now() -> str:
    """Return the current UTC timestamp in a stable JSON-friendly format."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sanitize_component(value: object) -> str:
    """Return a filesystem-friendly run id component."""

    text = str(value).strip()
    chars = []
    for char in text:
        if char.isalnum():
            chars.append(char)
        else:
            chars.append("_")
    sanitized = "".join(chars).strip("_")
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized or "unknown"
