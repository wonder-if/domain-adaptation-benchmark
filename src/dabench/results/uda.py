"""Minimal UDA result schema and Markdown table rendering."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from dabench.results._common import TABLE_LAYOUTS, VISDA_CLASS_ORDER, domain_code, normalize_dataset_name
from dabench.results.run import validate_run_record


def _normalize_dataset(dataset: str) -> str:
    return normalize_dataset_name(dataset)


def uda_table_layout(dataset: str) -> str:
    """Return the default table layout for a UDA dataset."""

    return TABLE_LAYOUTS[_normalize_dataset(dataset)]


def validate_uda_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate and lightly normalize a minimal UDA result payload."""

    if payload.get("setting") != "uda":
        raise ValueError("UDA result payload must set setting='uda'.")
    dataset = payload.get("dataset")
    if not isinstance(dataset, str):
        raise ValueError("UDA result payload must include a string dataset field.")
    normalized_dataset = _normalize_dataset(dataset)
    layout = payload.get("table_layout", uda_table_layout(normalized_dataset))
    if layout != uda_table_layout(normalized_dataset):
        raise ValueError(
            f"Dataset {normalized_dataset!r} expects table_layout={uda_table_layout(normalized_dataset)!r}, "
            f"got {layout!r}."
        )
    results = payload.get("results")
    if not isinstance(results, Sequence):
        raise ValueError("UDA result payload must include a results list.")
    return {
        **payload,
        "dataset": normalized_dataset,
        "table_layout": layout,
    }


def _domain_code(dataset: str, domain: str) -> str:
    return domain_code(dataset, domain)


def _metric_value(result: Mapping[str, object], metric_name: str) -> float:
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Each result must contain a metrics mapping.")
    value = metrics.get(metric_name)
    if not isinstance(value, (int, float)):
        source = result.get("source_domain", "?")
        target = result.get("target_domain", "?")
        raise ValueError(f"Missing numeric metric {metric_name!r} for {source}->{target}.")
    return float(value)


def _format_score(value: float) -> str:
    return f"{value:.1f}"


def _render_transfer_pairs(payload: Mapping[str, object]) -> str:
    dataset = str(payload["dataset"])
    metric_name = str(payload.get("primary_metric", "accuracy"))
    results = payload["results"]
    assert isinstance(results, Sequence)
    ordered = sorted(
        results,
        key=lambda item: (
            _domain_code(dataset, str(item["source_domain"])),
            _domain_code(dataset, str(item["target_domain"])),
        ),
    )
    headers = [
        f"{_domain_code(dataset, str(item['source_domain']))}->{_domain_code(dataset, str(item['target_domain']))}"
        for item in ordered
    ]
    values = [_format_score(_metric_value(item, metric_name)) for item in ordered]
    return "\n".join(
        [
            f"| {' | '.join(headers)} | Avg |",
            f"| {' | '.join(['---'] * (len(headers) + 1))} |",
            f"| {' | '.join(values)} | {_format_score(sum(float(v) for v in values) / len(values))} |",
        ]
    )


def _render_transfer_matrix(payload: Mapping[str, object]) -> str:
    dataset = str(payload["dataset"])
    metric_name = str(payload.get("primary_metric", "accuracy"))
    results = payload["results"]
    assert isinstance(results, Sequence)
    pairs = {
        (str(item["source_domain"]), str(item["target_domain"])): _metric_value(item, metric_name)
        for item in results
    }
    domains = sorted({*{src for src, _ in pairs}, *{tgt for _, tgt in pairs}}, key=lambda x: _domain_code(dataset, x))
    header = ["Tgt\\Src", *(_domain_code(dataset, domain) for domain in domains)]
    lines = [
        f"| {' | '.join(header)} |",
        f"| {' | '.join(['---'] * len(header))} |",
    ]
    for target in domains:
        row = [_domain_code(dataset, target)]
        for source in domains:
            if source == target:
                row.append("-")
                continue
            value = pairs.get((source, target))
            row.append(_format_score(value) if value is not None else "")
        lines.append(f"| {' | '.join(row)} |")
    return "\n".join(lines)


def _render_per_class(payload: Mapping[str, object]) -> str:
    results = payload["results"]
    assert isinstance(results, Sequence)
    if len(results) != 1:
        raise ValueError("Per-class UDA table currently expects exactly one transfer result.")
    result = results[0]
    class_metrics = result.get("class_metrics")
    if not isinstance(class_metrics, Mapping):
        raise ValueError("Per-class UDA result must include class_metrics.")
    headers = [label for _, label in VISDA_CLASS_ORDER] + ["Avg"]
    values = []
    for class_name, _ in VISDA_CLASS_ORDER:
        value = class_metrics.get(class_name)
        if not isinstance(value, (int, float)):
            raise ValueError(f"Missing class metric for {class_name!r}.")
        values.append(_format_score(float(value)))
    avg = result.get("metrics", {}).get("average_class_accuracy")
    if not isinstance(avg, (int, float)):
        raise ValueError("Per-class UDA result must include metrics.average_class_accuracy.")
    values.append(_format_score(float(avg)))
    return "\n".join(
        [
            f"| {' | '.join(headers)} |",
            f"| {' | '.join(['---'] * len(headers))} |",
            f"| {' | '.join(values)} |",
        ]
    )


def render_uda_markdown_table(payload: Mapping[str, object]) -> str:
    """Render a Markdown table from the minimal UDA result payload."""

    normalized = validate_uda_payload(payload)
    layout = str(normalized["table_layout"])
    if layout == "transfer_pairs":
        return _render_transfer_pairs(normalized)
    if layout == "transfer_matrix":
        return _render_transfer_matrix(normalized)
    if layout == "per_class":
        return _render_per_class(normalized)
    raise ValueError(f"Unsupported UDA table layout: {layout!r}")


def _metric_payload_from_run(record: Mapping[str, Any], metric_source: str) -> tuple[dict[str, float], dict[str, float]]:
    if metric_source == "final":
        metrics = record.get("final_metrics", {})
        class_metrics = record.get("final_class_metrics", {})
    elif metric_source == "best":
        metrics = record.get("best_metrics", {})
        class_metrics = record.get("best_class_metrics", {})
    elif metric_source == "selected":
        selected_checkpoint = record.get("selected_checkpoint", "last")
        if selected_checkpoint == "best":
            metrics = record.get("best_metrics", {})
            class_metrics = record.get("best_class_metrics", {})
        else:
            metrics = record.get("final_metrics", {})
            class_metrics = record.get("final_class_metrics", {})
    else:
        raise ValueError("metric_source must be one of: final, best, selected.")
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError(f"Run {record.get('run_id', '?')} does not have {metric_source} metrics.")
    if not isinstance(class_metrics, Mapping):
        raise ValueError(f"Run {record.get('run_id', '?')} has invalid class metrics.")
    return {key: float(value) for key, value in metrics.items()}, {key: float(value) for key, value in class_metrics.items()}


def _aggregate_numeric_mappings(mappings: Sequence[Mapping[str, float]], *, reduction: str) -> tuple[dict[str, float], dict[str, float]]:
    if not mappings:
        return {}, {}
    keys = sorted({key for mapping in mappings for key in mapping})
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for key in keys:
        values = [float(mapping[key]) for mapping in mappings if key in mapping]
        means[key] = sum(values) / len(values)
        if reduction == "mean_std":
            variance = sum((value - means[key]) ** 2 for value in values) / len(values)
            stds[key] = math.sqrt(variance)
    return means, stds


def build_uda_result_view(
    records: Sequence[Mapping[str, object]],
    *,
    metric_source: str = "final",
    reduction: str = "none",
    primary_metric: str = "accuracy",
    include_failed: bool = False,
) -> dict[str, Any]:
    """Build a table-ready UDA result payload from one or more run records."""

    if reduction not in {"none", "mean", "mean_std"}:
        raise ValueError("reduction must be one of: none, mean, mean_std.")
    if metric_source not in {"final", "best", "selected"}:
        raise ValueError("metric_source must be one of: final, best, selected.")

    validated = [validate_run_record(record) for record in records]
    if not include_failed:
        validated = [record for record in validated if record["status"] == "completed"]
    if not validated:
        raise ValueError("No run records available to build a UDA result view.")

    datasets = {str(record["dataset"]) for record in validated}
    settings = {str(record["setting"]) for record in validated}
    methods = {str(record["method"]) for record in validated}
    if len(datasets) != 1 or len(settings) != 1 or len(methods) != 1:
        raise ValueError("build_uda_result_view expects records from exactly one dataset, setting, and method.")
    dataset = next(iter(datasets))
    setting = next(iter(settings))
    method = next(iter(methods))
    if setting != "uda":
        raise ValueError("build_uda_result_view only supports UDA run records.")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in validated:
        source_domain = record.get("source_domain")
        target_domain = record.get("target_domain")
        if not isinstance(source_domain, str) or not isinstance(target_domain, str):
            raise ValueError("UDA run records must include source_domain and target_domain.")
        grouped[(source_domain, target_domain)].append(record)

    results: list[dict[str, Any]] = []
    for (source_domain, target_domain), items in sorted(
        grouped.items(),
        key=lambda item: (_domain_code(dataset, item[0][0]), _domain_code(dataset, item[0][1])),
    ):
        payloads = [_metric_payload_from_run(item, metric_source) for item in items]
        metric_maps = [metrics for metrics, _ in payloads]
        class_maps = [class_metrics for _, class_metrics in payloads]
        result: dict[str, Any] = {
            "source_domain": source_domain,
            "target_domain": target_domain,
            "run_count": len(items),
        }
        if reduction == "none":
            if len(items) != 1:
                raise ValueError(
                    f"Found {len(items)} runs for {source_domain}->{target_domain}; "
                    "use reduction='mean' or reduction='mean_std' to aggregate multiple seeds."
                )
            result["metrics"] = dict(metric_maps[0])
            if class_maps[0]:
                result["class_metrics"] = dict(class_maps[0])
            result["run"] = {
                "seed": items[0]["seed"],
                "run_id": items[0]["run_id"],
                "output_dir": items[0].get("output_dir"),
            }
        else:
            metrics_mean, metrics_std = _aggregate_numeric_mappings(metric_maps, reduction=reduction)
            result["metrics"] = metrics_mean
            if reduction == "mean_std":
                result["metrics_std"] = metrics_std
            if any(class_maps):
                class_mean, class_std = _aggregate_numeric_mappings(class_maps, reduction=reduction)
                result["class_metrics"] = class_mean
                if reduction == "mean_std":
                    result["class_metrics_std"] = class_std
            result["seeds"] = [int(item["seed"]) for item in items]
        results.append(result)

    return {
        "schema_version": "1.0",
        "view_type": "benchmark_result_view",
        "setting": "uda",
        "dataset": dataset,
        "method": method,
        "table_layout": uda_table_layout(dataset),
        "primary_metric": primary_metric,
        "aggregation": {
            "metric_source": metric_source,
            "reduction": reduction,
            "num_runs": len(validated),
        },
        "results": results,
    }
