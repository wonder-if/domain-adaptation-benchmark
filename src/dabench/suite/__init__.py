"""Built-in benchmark suite accessors."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from dabench.suite import domainnet as _domainnet
from dabench.suite import office31 as _office31
from dabench.suite import officehome as _officehome
from dabench.suite import visda as _visda

__all__ = ["build_suites", "get_suite", "list_suites", "load_suite_item"]


_SUITE_SPECS = {
    "office-31": {
        "suite_id": "office31_uda",
        "name": "Office-31 UDA",
        "builder": _office31.build_office31_uda_suite,
        "domains": _office31.OFFICE31_DOMAINS,
    },
    "office-home": {
        "suite_id": "office_home_uda",
        "name": "Office-Home UDA",
        "builder": _officehome.build_office_home_uda_suite,
        "domains": _officehome.OFFICE_HOME_DOMAINS,
    },
    "domainnet": {
        "suite_id": "domainnet_uda",
        "name": "DomainNet UDA",
        "builder": _domainnet.build_domainnet_uda_suite,
        "domains": _domainnet.DOMAINNET_DOMAINS,
    },
    "visda-2017": {
        "suite_id": "visda_uda",
        "name": "VisDA-2017 UDA",
        "builder": _visda.build_visda_uda_suite,
        "domains": _visda.VISDA_DOMAINS,
    },
}

_SUITE_ALIASES = {
    "office31": "office-31",
    "office-31": "office-31",
    "officehome": "office-home",
    "office-home": "office-home",
    "office_home": "office-home",
    "domainnet": "domainnet",
    "visda2017": "visda-2017",
    "visda-2017": "visda-2017",
    "visda_2017": "visda-2017",
}


def _normalize_suite_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-")
    return _SUITE_ALIASES.get(normalized, normalized)


def _suite_descriptor(
    *,
    suite_id: str,
    name: str,
    settings,
    domains,
) -> dict[str, Any]:
    return {
        "suite_id": suite_id,
        "name": name,
        "settings": tuple(settings),
        "metadata": {"domains": domains},
    }


def _resolve_datasets(datasets: str | Iterable[str] | None) -> tuple[str, ...]:
    if datasets is None:
        return tuple(_SUITE_SPECS.keys())
    if isinstance(datasets, str):
        return (_normalize_suite_name(datasets),)
    return tuple(_normalize_suite_name(item) for item in datasets)


def build_suites(
    *,
    datasets: str | Iterable[str] | None = None,
    setting: str = "uda",
    format: str = "hf",
) -> list[dict[str, Any]]:
    if setting != "uda":
        raise ValueError("Only setting='uda' is currently supported.")

    suite_ids = _resolve_datasets(datasets)
    suites: list[dict[str, Any]] = []
    for dataset_name in suite_ids:
        try:
            spec = _SUITE_SPECS[dataset_name]
        except KeyError as exc:
            available = ", ".join(sorted(_SUITE_SPECS))
            raise ValueError(f"Unsupported dataset suite: {dataset_name!r}. Available datasets: {available}") from exc
        suites.append(
            _suite_descriptor(
                suite_id=spec["suite_id"],
                name=spec["name"],
                settings=spec["builder"](format=format),
                domains=spec["domains"],
            )
        )
    return suites


def load_suite_item(item: dict[str, Any]):
    item = dict(item)
    item.pop("name", None)
    from dabench.setting import load_uda

    return load_uda(**item)


def list_suites(*, datasets: str | Iterable[str] | None = None, setting: str = "uda", format: str = "hf") -> list[dict[str, Any]]:
    return build_suites(datasets=datasets, setting=setting, format=format)


def get_suite(suite_id: str, *, setting: str = "uda", format: str = "hf") -> dict[str, Any]:
    suites = {suite["suite_id"]: suite for suite in build_suites(setting=setting, format=format)}
    try:
        return suites[suite_id]
    except KeyError as exc:
        available = ", ".join(sorted(suites))
        raise ValueError(f"Unsupported benchmark suite: {suite_id}. Available suites: {available}") from exc
