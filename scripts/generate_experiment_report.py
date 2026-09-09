#!/usr/bin/env python3
"""Generate a compact report from extracted benchmark result views."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
DAMP_RESULT_ROOT = REPO_ROOT / "related_works" / "damp" / "results" / "uda"
TASC_RESULT_ROOT = REPO_ROOT / "related_works" / "tasc" / "results" / "unida"
REPORT_ROOT = REPO_ROOT / "reports" / "experiment_results"
FIGURE_ROOT = REPORT_ROOT / "figures"
TABLE_ROOT = REPORT_ROOT / "tables"

DAMP_FILES = (
    "office_home_damp.json",
    "minidomainnet_damp.json",
    "minidomainnet_damp_vit_b_16.json",
    "visda17_damp.json",
)

SETTING_ORDER = ("CDA", "PDA", "ODA", "OPDA")
DATASET_ORDER = ("office", "office-home", "domainnet", "visda-2017")
OKABE_ITO = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "-"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}"
    return str(value)


def slug(value: str) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        else:
            chars.append("_")
    text = "".join(chars).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return text or "item"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered_rows = [[fmt(item) for item in row] for row in rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rendered_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def damp_display_average(payload: dict[str, Any]) -> float:
    if payload.get("table_layout") == "per_class":
        result = payload["results"][0]
        return float(result["metrics"]["average_class_accuracy"])
    metric = payload.get("primary_metric", "accuracy")
    values = [float(result["metrics"][metric]) for result in payload["results"]]
    return sum(values) / len(values)


def damp_payload_backbone(payload: dict[str, Any]) -> str:
    backbone = payload.get("backbone")
    if isinstance(backbone, str) and backbone:
        return backbone
    first_result = payload.get("results", [{}])[0]
    run_id = first_result.get("run", {}).get("run_id", "") if isinstance(first_result, dict) else ""
    if "ViT_B_16" in run_id:
        return "ViT-B/16"
    if "RN50" in run_id:
        return "RN50"
    return "unknown"


def damp_payload_label(payload: dict[str, Any]) -> str:
    return f"{payload['dataset']} ({damp_payload_backbone(payload)})"


def damp_payload_slug(payload: dict[str, Any]) -> str:
    return slug(damp_payload_label(payload))


def damp_overview_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for payload in payloads:
        row = {
            "method": payload["method"],
            "setting": payload["setting"].upper(),
            "dataset": payload["dataset"],
            "backbone": damp_payload_backbone(payload),
            "result_view": damp_payload_label(payload),
            "metric": payload["primary_metric"],
            "metric_source": payload["aggregation"]["metric_source"],
            "num_runs": payload["aggregation"]["num_runs"],
            "reported_avg": round(damp_display_average(payload), 4),
        }
        if payload.get("table_layout") == "per_class":
            row["overall_accuracy"] = payload["results"][0]["metrics"].get("accuracy")
            row["metric"] = "average_class_accuracy"
        rows.append(row)
    return rows


def damp_detailed_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for result in payload["results"]:
            base = {
                "method": payload["method"],
                "setting": payload["setting"].upper(),
                "dataset": payload["dataset"],
                "backbone": damp_payload_backbone(payload),
                "source_domain": result["source_domain"],
                "target_domain": result["target_domain"],
                "run_count": result["run_count"],
                "seed": result.get("run", {}).get("seed"),
                "run_id": result.get("run", {}).get("run_id"),
            }
            for key, value in result.get("metrics", {}).items():
                base[key] = value
            rows.append(base)
            for class_name, value in result.get("class_metrics", {}).items():
                rows.append(
                    {
                        "method": payload["method"],
                        "setting": payload["setting"].upper(),
                        "dataset": payload["dataset"],
                        "backbone": damp_payload_backbone(payload),
                        "source_domain": result["source_domain"],
                        "target_domain": result["target_domain"],
                        "class_name": class_name,
                        "class_accuracy": value,
                        "run_count": result["run_count"],
                        "seed": result.get("run", {}).get("seed"),
                        "run_id": result.get("run", {}).get("run_id"),
                    }
                )
    return rows


def transfer_matrix(results: list[dict[str, Any]], value_key: str = "primary_value") -> pd.DataFrame:
    sources = sorted({item["source_domain"] for item in results})
    targets = sorted({item["target_domain"] for item in results})
    data = pd.DataFrame(np.nan, index=targets, columns=sources, dtype=float)
    for item in results:
        if value_key == "primary_value":
            value = item["primary_value"]
        else:
            value = item["metrics"][value_key]
        data.loc[item["target_domain"], item["source_domain"]] = float(value)
    return data


def plot_heatmap(data: pd.DataFrame, title: str, path: Path, *, label: str = "score") -> None:
    width = max(4.2, 0.72 * len(data.columns) + 1.5)
    height = max(3.4, 0.62 * len(data.index) + 1.2)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        data,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        cbar_kws={"label": label},
        linewidths=0.5,
        linecolor="white",
        mask=data.isna(),
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Source")
    ax.set_ylabel("Target")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    savefig(fig, path)


def plot_bar(rows: list[dict[str, Any]], x: str, y: str, title: str, path: Path, *, hue: str | None = None) -> None:
    data = pd.DataFrame(rows)
    width = max(5.5, 0.7 * len(data[x].unique()) + (1.8 if hue else 0.0))
    fig, ax = plt.subplots(figsize=(width, 3.4))
    if hue is None:
        sns.barplot(data=data, x=x, y=y, color=OKABE_ITO[0], ax=ax)
    else:
        sns.barplot(data=data, x=x, y=y, hue=hue, palette=OKABE_ITO, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)
    if hue:
        ax.legend(title=hue.replace("_", " ").title(), frameon=False, ncol=2)
    savefig(fig, path)


def plot_damp_figures(payloads: list[dict[str, Any]]) -> list[tuple[str, str]]:
    figures: list[tuple[str, str]] = []
    overview = damp_overview_rows(payloads)
    overview_path = FIGURE_ROOT / "damp_uda_overview.png"
    plot_bar(overview, "result_view", "reported_avg", "DAMP UDA reported average", overview_path)
    figures.append(("DAMP UDA overview", overview_path.relative_to(REPORT_ROOT).as_posix()))

    for payload in payloads:
        dataset = payload["dataset"]
        label = damp_payload_label(payload)
        artifact_slug = damp_payload_slug(payload)
        if payload.get("table_layout") == "per_class":
            result = payload["results"][0]
            rows = [
                {"class": class_name, "accuracy": value}
                for class_name, value in result.get("class_metrics", {}).items()
            ]
            path = FIGURE_ROOT / f"damp_{artifact_slug}_class_accuracy.png"
            fig, ax = plt.subplots(figsize=(7.2, 3.2))
            sns.barplot(data=pd.DataFrame(rows), x="class", y="accuracy", color=OKABE_ITO[0], ax=ax)
            ax.axhline(result["metrics"]["average_class_accuracy"], color=OKABE_ITO[1], linestyle="--", linewidth=1)
            ax.set_title(f"DAMP {label}: per-class accuracy")
            ax.set_xlabel("")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 105)
            ax.tick_params(axis="x", rotation=35)
            savefig(fig, path)
            figures.append((f"DAMP {label} class accuracy", path.relative_to(REPORT_ROOT).as_posix()))
            continue

        path = FIGURE_ROOT / f"damp_{artifact_slug}_transfer_heatmap.png"
        matrix = transfer_matrix(payload["results"], str(payload.get("primary_metric", "accuracy")))
        plot_heatmap(matrix, f"DAMP {label}: transfer accuracy", path, label="accuracy")
        figures.append((f"DAMP {label} transfer heatmap", path.relative_to(REPORT_ROOT).as_posix()))
    return figures


def tasc_overview_rows(collection: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "method": collection["method"],
            "setting": collection["setting"].upper(),
            "dataset": row["dataset"],
            "category_shift": row["category_shift"],
            "metric": row["primary_metric"],
            "num_runs": row["num_runs"],
            "reported_avg": row["average"],
        }
        for row in collection["overview"]
    ]


def tasc_detailed_rows(collection: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_name, payload in collection["datasets"].items():
        for result in payload["results"]:
            base = {
                "method": payload["method"],
                "setting": payload["setting"].upper(),
                "dataset": dataset_name,
                "category_shift": result["category_shift"],
                "task": result["task"],
                "source_domain": result["source_domain"],
                "target_domain": result["target_domain"],
                "shared_classes": result["shared_classes"],
                "source_private_classes": result["source_private_classes"],
                "target_private_classes": result["target_private_classes"],
                "primary_metric": result["primary_metric"],
                "primary_value": result["primary_value"],
                "run_count": result["run_count"],
                "seed": result.get("run", {}).get("seed"),
                "final_iteration": result.get("run", {}).get("final_iteration"),
                "run_id": result.get("run", {}).get("run_id"),
            }
            for key, value in result.get("metrics", {}).items():
                base[key] = value
            rows.append(base)
    return rows


def plot_tasc_figures(collection: dict[str, Any]) -> list[tuple[str, str]]:
    figures: list[tuple[str, str]] = []
    overview = tasc_overview_rows(collection)
    overview_path = FIGURE_ROOT / "tasc_unida_overview.png"
    plot_bar(
        overview,
        "dataset",
        "reported_avg",
        "TASC UniDA average by category shift",
        overview_path,
        hue="category_shift",
    )
    figures.append(("TASC UniDA overview", overview_path.relative_to(REPORT_ROOT).as_posix()))

    for dataset_name in DATASET_ORDER:
        payload = collection["datasets"].get(dataset_name)
        if not payload:
            continue
        by_shift: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in payload["results"]:
            by_shift[result["category_shift"]].append(result)
        for shift in SETTING_ORDER:
            results = by_shift.get(shift)
            if not results:
                continue
            path = FIGURE_ROOT / f"tasc_{slug(dataset_name)}_{slug(shift)}_transfer_heatmap.png"
            matrix = transfer_matrix(results, "primary_value")
            plot_heatmap(matrix, f"TASC {dataset_name} {shift}: primary metric", path, label="score")
            figures.append((f"TASC {dataset_name} {shift}", path.relative_to(REPORT_ROOT).as_posix()))
    return figures


def result_table_from_payload(payload: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    if payload.get("table_layout") == "per_class":
        result = payload["results"][0]
        class_metrics = result["class_metrics"]
        headers = list(class_metrics) + ["Avg", "Overall Acc"]
        row = list(class_metrics.values()) + [
            result["metrics"].get("average_class_accuracy"),
            result["metrics"].get("accuracy"),
        ]
        return headers, [row]

    metric = str(payload.get("primary_metric", "accuracy"))
    if payload.get("table_layout") == "transfer_matrix":
        matrix = transfer_matrix(payload["results"], metric)
        headers = ["Tgt\\Src", *matrix.columns.tolist(), "Avg"]
        rows: list[list[Any]] = []
        all_values: list[float] = []
        for target, values in matrix.iterrows():
            valid = [float(item) for item in values.tolist() if not pd.isna(item)]
            all_values.extend(valid)
            rows.append([target, *values.tolist(), sum(valid) / len(valid) if valid else None])
        avg_row: list[Any] = ["Avg"]
        for source in matrix.columns:
            valid = [float(item) for item in matrix[source].tolist() if not pd.isna(item)]
            avg_row.append(sum(valid) / len(valid) if valid else None)
        avg_row.append(sum(all_values) / len(all_values) if all_values else None)
        rows.append(avg_row)
        return headers, rows

    ordered = sorted(payload["results"], key=lambda item: (item["source_domain"], item["target_domain"]))
    headers = [f"{item['source_domain']}->{item['target_domain']}" for item in ordered] + ["Avg"]
    values = [float(item["metrics"][metric]) for item in ordered]
    return headers, [values + [sum(values) / len(values)]]


def tasc_primary_table(payload: dict[str, Any], shift: str) -> tuple[list[str], list[list[Any]]]:
    results = [item for item in payload["results"] if item["category_shift"] == shift]
    ordered = sorted(results, key=lambda item: item["task"])
    headers = [item["task"] for item in ordered] + ["Avg"]
    values = [float(item["primary_value"]) for item in ordered]
    return headers, [values + [sum(values) / len(values)]]


def write_report(
    damp_payloads: list[dict[str, Any]],
    tasc_collection: dict[str, Any],
    damp_figures: list[tuple[str, str]],
    tasc_figures: list[tuple[str, str]],
) -> None:
    damp_overview = damp_overview_rows(damp_payloads)
    tasc_overview = tasc_overview_rows(tasc_collection)
    lines: list[str] = [
        "# Experiment Results Report",
        "",
        "This report is generated from extracted result views under `related_works/*/results`.",
        "",
        "## Sources",
        "",
        "- DAMP UDA: `related_works/damp/results/uda/*.json`",
        "- TASC UniDA: `related_works/tasc/results/unida/tasc_unida_results.json`",
        "",
        "## Summary",
        "",
        f"- DAMP UDA: {sum(row['num_runs'] for row in damp_overview)} runs across {len(damp_overview)} result views.",
        f"- TASC UniDA: {tasc_collection['aggregation']['num_runs']} selected runs across {len(tasc_collection['overview'])} dataset/shift summaries.",
        "- DAMP VisDA reports per-class average in the class table; the run-level overall accuracy is shown separately.",
        "",
        "### Topline",
        "",
        markdown_table(
            ["Method", "Setting", "Dataset", "Backbone", "Shift", "Metric", "Source", "Num runs", "Avg"],
            [
                [
                    row["method"],
                    row["setting"],
                    row["dataset"],
                    row["backbone"],
                    "-",
                    row["metric"],
                    row["metric_source"],
                    row["num_runs"],
                    row["reported_avg"],
                ]
                for row in damp_overview
            ]
            + [
                [
                    row["method"],
                    row["setting"],
                    row["dataset"],
                    "-",
                    row["category_shift"],
                    row["metric"],
                    "selected",
                    row["num_runs"],
                    row["reported_avg"],
                ]
                for row in tasc_overview
            ],
        ),
        "",
        "![DAMP UDA overview](figures/damp_uda_overview.png)",
        "",
        "![TASC UniDA overview](figures/tasc_unida_overview.png)",
        "",
        "## DAMP UDA",
        "",
    ]

    for payload in damp_payloads:
        dataset = payload["dataset"]
        label = damp_payload_label(payload)
        headers, rows = result_table_from_payload(payload)
        lines.extend(
            [
                f"### {label}",
                "",
                f"- method: {payload['method']}",
                f"- backbone: {damp_payload_backbone(payload)}",
                f"- primary metric: {payload['primary_metric']}",
                f"- metric source: {payload['aggregation']['metric_source']}",
                f"- num runs: {payload['aggregation']['num_runs']}",
                "",
                markdown_table(headers, rows),
                "",
            ]
        )
        figure_token = f"damp_{damp_payload_slug(payload)}"
        for title, rel_path in damp_figures:
            if Path(rel_path).stem.startswith(figure_token):
                lines.extend([f"![{title}]({rel_path})", ""])

    lines.extend(["## TASC UniDA", ""])
    lines.extend(
        [
            "- method: TASC",
            "- backbone: CLIP ViT-B/16",
            "- primary metric: accuracy for CDA/PDA; H-score for ODA/OPDA",
            f"- selection: {tasc_collection['aggregation']['selection']}",
            "",
            "### Overview",
            "",
            markdown_table(
                ["Dataset", "Shift", "Metric", "Num runs", "Avg"],
                [
                    [row["dataset"], row["category_shift"], row["metric"], row["num_runs"], row["reported_avg"]]
                    for row in tasc_overview
                ],
            ),
            "",
        ]
    )

    for dataset_name in DATASET_ORDER:
        payload = tasc_collection["datasets"].get(dataset_name)
        if not payload:
            continue
        lines.extend([f"### {dataset_name}", ""])
        for shift in SETTING_ORDER:
            if not any(item["category_shift"] == shift for item in payload["results"]):
                continue
            headers, rows = tasc_primary_table(payload, shift)
            metric = payload["primary_metric"][shift]
            lines.extend(
                [
                    f"#### {shift}",
                    "",
                    f"- primary metric: {metric}",
                    "",
                    markdown_table(headers, rows),
                    "",
                ]
            )
            stem_token = f"tasc_{slug(dataset_name)}_{slug(shift)}"
            for title, rel_path in tasc_figures:
                if Path(rel_path).stem.startswith(stem_token):
                    lines.extend([f"![{title}]({rel_path})", ""])
                    break

    lines.extend(
        [
            "## Generated Artifacts",
            "",
            "- `tables/damp_uda_overview.csv`",
            "- `tables/damp_uda_detailed.csv`",
            "- `tables/tasc_unida_overview.csv`",
            "- `tables/tasc_unida_detailed.csv`",
            "- `figures/*.png`",
            "",
        ]
    )
    (REPORT_ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_plot_style()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    for path in FIGURE_ROOT.glob("damp_*.png"):
        path.unlink()

    damp_payloads = [load_json(DAMP_RESULT_ROOT / name) for name in DAMP_FILES]
    tasc_collection = load_json(TASC_RESULT_ROOT / "tasc_unida_results.json")

    write_csv(TABLE_ROOT / "damp_uda_overview.csv", damp_overview_rows(damp_payloads))
    write_csv(TABLE_ROOT / "damp_uda_detailed.csv", damp_detailed_rows(damp_payloads))
    write_csv(TABLE_ROOT / "tasc_unida_overview.csv", tasc_overview_rows(tasc_collection))
    write_csv(TABLE_ROOT / "tasc_unida_detailed.csv", tasc_detailed_rows(tasc_collection))

    damp_figures = plot_damp_figures(damp_payloads)
    tasc_figures = plot_tasc_figures(tasc_collection)
    write_report(damp_payloads, tasc_collection, damp_figures, tasc_figures)

    print(f"wrote={REPORT_ROOT.relative_to(REPO_ROOT)}")
    print(f"figures={len(damp_figures) + len(tasc_figures)}")
    print("tables=4")


if __name__ == "__main__":
    main()
