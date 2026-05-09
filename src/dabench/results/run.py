"""Run-record schema helpers for benchmark experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from dabench.results._common import normalize_dataset_name, normalize_setting_name, sanitize_component

RUN_RECORD_SCHEMA_VERSION = "1.0"
RUN_RECORD_TYPE = "run_record"
RUN_STATUSES = frozenset({"running", "completed", "failed", "interrupted"})
METRIC_SOURCES = frozenset({"final", "best", "selected"})


def _validate_numeric_mapping(name: str, payload: object) -> dict[str, float]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a mapping of metric names to numbers.")
    normalized: dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} must use string keys.")
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name}.{key} must be numeric.")
        normalized[key] = float(value)
    return normalized


def _validate_optional_mapping(name: str, payload: object) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a mapping when provided.")
    return dict(payload)


def _validate_history(history: object) -> list[dict[str, Any]]:
    if history is None:
        return []
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise ValueError("eval_history must be a list of evaluation events.")
    normalized_events: list[dict[str, Any]] = []
    for event in history:
        if not isinstance(event, Mapping):
            raise ValueError("Each eval_history event must be a mapping.")
        split = event.get("split")
        if not isinstance(split, str) or not split:
            raise ValueError("Each eval_history event must include a non-empty split.")
        normalized_event = dict(event)
        normalized_event["metrics"] = _validate_numeric_mapping("eval_history.metrics", event.get("metrics"))
        if not normalized_event["metrics"]:
            raise ValueError("Each eval_history event must include at least one metric.")
        class_metrics = _validate_numeric_mapping("eval_history.class_metrics", event.get("class_metrics"))
        if class_metrics:
            normalized_event["class_metrics"] = class_metrics
        elif "class_metrics" in normalized_event:
            normalized_event.pop("class_metrics")
        normalized_events.append(normalized_event)
    return normalized_events


def build_run_id(
    *,
    dataset: str,
    setting: str,
    method: str,
    source_domain: str | None,
    target_domain: str | None,
    seed: int,
    backbone: str | None = None,
) -> str:
    """Build a stable run id from explicit experiment identity fields."""

    parts = [
        normalize_dataset_name(dataset),
        normalize_setting_name(setting),
        method,
        backbone or "unknown-backbone",
        f"{source_domain or 'na'}_to_{target_domain or 'na'}",
        f"seed{seed}",
    ]
    return "__".join(sanitize_component(part) for part in parts)


def make_run_record(
    *,
    dataset: str,
    setting: str,
    method: str,
    seed: int,
    source_domain: str | None = None,
    target_domain: str | None = None,
    backbone: str | None = None,
    algorithm_name: str | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
    status: str = "running",
    selected_checkpoint: str = "last",
    selection_metric: str = "accuracy",
    selection_mode: str = "max",
    start_time: str | None = None,
    end_time: str | None = None,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    final_metrics: Mapping[str, float] | None = None,
    best_metrics: Mapping[str, float] | None = None,
    final_class_metrics: Mapping[str, float] | None = None,
    best_class_metrics: Mapping[str, float] | None = None,
    eval_history: Sequence[Mapping[str, Any]] | None = None,
    failure_reason: str | None = None,
    metrics_schema_version: str = RUN_RECORD_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Construct a run-record payload with stable defaults."""

    normalized_dataset = normalize_dataset_name(dataset)
    normalized_setting = normalize_setting_name(setting)
    return {
        "schema_version": RUN_RECORD_SCHEMA_VERSION,
        "record_type": RUN_RECORD_TYPE,
        "metrics_schema_version": metrics_schema_version,
        "run_id": run_id
        or build_run_id(
            dataset=normalized_dataset,
            setting=normalized_setting,
            method=method,
            source_domain=source_domain,
            target_domain=target_domain,
            seed=seed,
            backbone=backbone,
        ),
        "dataset": normalized_dataset,
        "setting": normalized_setting,
        "method": method,
        "algorithm_name": algorithm_name,
        "backbone": backbone,
        "source_domain": source_domain,
        "target_domain": target_domain,
        "seed": seed,
        "status": status,
        "output_dir": output_dir,
        "selected_checkpoint": selected_checkpoint,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "start_time": start_time,
        "end_time": end_time,
        "config": dict(config or {}),
        "extra": dict(extra or {}),
        "final_metrics": dict(final_metrics or {}),
        "best_metrics": dict(best_metrics or {}),
        "final_class_metrics": dict(final_class_metrics or {}),
        "best_class_metrics": dict(best_class_metrics or {}),
        "eval_history": [dict(event) for event in (eval_history or [])],
        "failure_reason": failure_reason,
    }


def validate_run_record(record: Mapping[str, object]) -> dict[str, Any]:
    """Validate and lightly normalize a run-record payload."""

    normalized = dict(record)
    if normalized.get("record_type") != RUN_RECORD_TYPE:
        raise ValueError(f"Run record must set record_type={RUN_RECORD_TYPE!r}.")
    normalized["dataset"] = normalize_dataset_name(str(normalized["dataset"]))
    normalized["setting"] = normalize_setting_name(str(normalized["setting"]))
    method = normalized.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError("Run record must include a non-empty method.")
    seed = normalized.get("seed")
    if not isinstance(seed, int):
        raise ValueError("Run record must include an integer seed.")
    status = normalized.get("status")
    if status not in RUN_STATUSES:
        supported = ", ".join(sorted(RUN_STATUSES))
        raise ValueError(f"Unsupported run status {status!r}. Supported values: {supported}")
    selected_checkpoint = normalized.get("selected_checkpoint")
    if selected_checkpoint not in {"last", "best"}:
        raise ValueError("selected_checkpoint must be either 'last' or 'best'.")
    selection_mode = normalized.get("selection_mode")
    if selection_mode not in {"max", "min"}:
        raise ValueError("selection_mode must be either 'max' or 'min'.")

    normalized["config"] = _validate_optional_mapping("config", normalized.get("config"))
    normalized["extra"] = _validate_optional_mapping("extra", normalized.get("extra"))
    normalized["final_metrics"] = _validate_numeric_mapping("final_metrics", normalized.get("final_metrics"))
    normalized["best_metrics"] = _validate_numeric_mapping("best_metrics", normalized.get("best_metrics"))
    normalized["final_class_metrics"] = _validate_numeric_mapping(
        "final_class_metrics",
        normalized.get("final_class_metrics"),
    )
    normalized["best_class_metrics"] = _validate_numeric_mapping(
        "best_class_metrics",
        normalized.get("best_class_metrics"),
    )
    normalized["eval_history"] = _validate_history(normalized.get("eval_history"))

    if status == "completed" and not normalized["final_metrics"]:
        raise ValueError("Completed run records must include final_metrics.")
    if status in {"failed", "interrupted"} and normalized.get("failure_reason") is not None:
        if not isinstance(normalized["failure_reason"], str):
            raise ValueError("failure_reason must be a string when provided.")
    run_id = normalized.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        normalized["run_id"] = build_run_id(
            dataset=normalized["dataset"],
            setting=normalized["setting"],
            method=method,
            source_domain=normalized.get("source_domain"),
            target_domain=normalized.get("target_domain"),
            seed=seed,
            backbone=normalized.get("backbone"),
        )
    return normalized


def run_record_output_path(record: Mapping[str, object], *, records_root: str | Path) -> Path:
    """Return the canonical file path for a run-record payload."""

    normalized = validate_run_record(record)
    root = Path(records_root)
    return (
        root
        / str(normalized["dataset"])
        / str(normalized["setting"])
        / str(normalized["method"])
        / f"{normalized['run_id']}.json"
    )


def write_run_record(
    record: Mapping[str, object],
    *,
    path: str | Path | None = None,
    records_root: str | Path | None = None,
) -> Path:
    """Write a validated run record to disk."""

    normalized = validate_run_record(record)
    if path is None:
        if records_root is None:
            raise ValueError("write_run_record requires either path or records_root.")
        path_obj = run_record_output_path(normalized, records_root=records_root)
    else:
        path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(normalized, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path_obj


def load_run_record(path: str | Path) -> dict[str, Any]:
    """Load and validate a run record from disk."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_run_record(payload)


def collect_run_records(
    root: str | Path,
    *,
    dataset: str | None = None,
    setting: str | None = None,
    method: str | None = None,
    status: str | None = "completed",
) -> list[dict[str, Any]]:
    """Collect validated run records from a directory tree."""

    root_path = Path(root)
    normalized_dataset = normalize_dataset_name(dataset) if dataset is not None else None
    normalized_setting = normalize_setting_name(setting) if setting is not None else None
    if status is not None and status not in RUN_STATUSES:
        supported = ", ".join(sorted(RUN_STATUSES))
        raise ValueError(f"Unsupported run status filter {status!r}. Supported values: {supported}")

    records: list[dict[str, Any]] = []
    for path in sorted(root_path.rglob("*.json")):
        try:
            record = load_run_record(path)
        except (json.JSONDecodeError, OSError, ValueError):
            continue
        if normalized_dataset is not None and record["dataset"] != normalized_dataset:
            continue
        if normalized_setting is not None and record["setting"] != normalized_setting:
            continue
        if method is not None and record["method"] != method:
            continue
        if status is not None and record["status"] != status:
            continue
        records.append(record)
    return records


def clone_run_record(record: Mapping[str, object]) -> dict[str, Any]:
    """Return a deep copy of a validated run record."""

    return deepcopy(validate_run_record(record))
