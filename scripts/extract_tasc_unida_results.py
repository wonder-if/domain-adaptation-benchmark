#!/usr/bin/env python3
"""Extract TASC UniDA experiment results into Markdown and JSON artifacts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "related_works/tasc/codes/tasc-dabench/output"
RESULT_ROOT = REPO_ROOT / "related_works/tasc/results/unida"
SOURCE_ROOT_DISPLAY = "related_works/tasc/codes/tasc-dabench/output"

RUN_RE = re.compile(
    r"^runs_(?P<stamp>\d{4}_\d{4}_\d{6})_"
    r"(?P<dataset>domainnet|officehome|office|visda)_"
    r"(?P<task>[A-Za-z]{2})_"
    r"(?P<shared>\d+)-(?P<source_private>\d+)-(?P<target_private>\d+)_"
    r"TASC_(?P<exp_name>.+)$"
)

DATASET_NAMES = {
    "office": "office",
    "officehome": "office-home",
    "domainnet": "domainnet",
    "visda": "visda-2017",
}

DATASET_FILES = {
    "office": "office_tasc",
    "officehome": "office_home_tasc",
    "domainnet": "domainnet_tasc",
    "visda": "visda17_tasc",
}

DOMAIN_NAMES = {
    "office": {
        "a": ("A", "Amazon"),
        "d": ("D", "DSLR"),
        "w": ("W", "Webcam"),
    },
    "officehome": {
        "A": ("A", "Art"),
        "C": ("C", "Clipart"),
        "P": ("P", "Product"),
        "R": ("R", "Real World"),
    },
    "domainnet": {
        "p": ("Pnt", "Painting"),
        "r": ("Rel", "Real"),
        "s": ("Skt", "Sketch"),
    },
    "visda": {
        "S": ("S", "Synthetic"),
        "R": ("R", "Real"),
    },
}

TASK_ORDER = {
    "office": ["ad", "aw", "da", "dw", "wa", "wd"],
    "officehome": [
        "AC",
        "AP",
        "AR",
        "CA",
        "CP",
        "CR",
        "PA",
        "PC",
        "PR",
        "RA",
        "RC",
        "RP",
    ],
    "domainnet": ["pr", "ps", "rp", "rs", "sp", "sr"],
    "visda": ["SR"],
}

SETTING_ORDER = ["CDA", "PDA", "ODA", "OPDA"]


@dataclass(frozen=True)
class RunRecord:
    dataset: str
    setting: str
    task: str
    shared: int
    source_private: int
    target_private: int
    exp_name: str
    stamp: str
    final_iteration: int
    run_dir: Path
    metric_file: Path
    seed: int | None
    metrics: dict[str, float | None]

    @property
    def task_label(self) -> str:
        return task_label(self.dataset, self.task)

    @property
    def source_domain(self) -> str:
        return domain_pair(self.dataset, self.task)[0]

    @property
    def target_domain(self) -> str:
        return domain_pair(self.dataset, self.task)[1]


def infer_setting(source_private: int, target_private: int) -> str:
    if source_private == 0 and target_private == 0:
        return "CDA"
    if source_private == 0 and target_private > 0:
        return "ODA"
    if source_private > 0 and target_private == 0:
        return "PDA"
    return "OPDA"


def task_label(dataset: str, task: str) -> str:
    source_key, target_key = task[0], task[1]
    source = DOMAIN_NAMES[dataset][source_key][0]
    target = DOMAIN_NAMES[dataset][target_key][0]
    return f"{source}->{target}"


def domain_pair(dataset: str, task: str) -> tuple[str, str]:
    source_key, target_key = task[0], task[1]
    return (
        DOMAIN_NAMES[dataset][source_key][1],
        DOMAIN_NAMES[dataset][target_key][1],
    )


def primary_metric(setting: str) -> str:
    if setting in {"CDA", "PDA"}:
        return "accuracy"
    return "h_score"


def primary_metric_label(setting: str) -> str:
    if setting in {"CDA", "PDA"}:
        return "accuracy"
    return "H-score"


def parse_iteration(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def parse_seed(run_dir: Path) -> int | None:
    conf_file = run_dir / "config/conf.yaml"
    if not conf_file.exists():
        return None
    text = conf_file.read_text(encoding="utf-8")
    match = re.search(r"^\s*seed:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    if not match:
        return None
    return int(match.group(1))


def nested_get(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def clean_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return round(number, 4)
    return None


def extract_metrics(metric_file: Path) -> dict[str, float | None]:
    with metric_file.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    metrics = {
        "accuracy": clean_number(nested_get(raw, "Closed-set", "OA")),
        "closed_set_oa": clean_number(nested_get(raw, "Closed-set", "OA")),
        "closed_set_recall": clean_number(nested_get(raw, "Closed-set", "Recall")),
        "h_score": clean_number(nested_get(raw, "UniDA", "H-score")),
        "h3_score": clean_number(nested_get(raw, "UniDA", "H3-score")),
        "ucr": clean_number(nested_get(raw, "UniDA", "UCR")),
        "open_set_oa": clean_number(nested_get(raw, "Open-set", "OA")),
        "os_star": clean_number(nested_get(raw, "Open-set", "OS*")),
        "unknown": clean_number(nested_get(raw, "Open-set", "UNK")),
        "os": clean_number(nested_get(raw, "Open-set", "OS")),
        "open_set_nmi": clean_number(nested_get(raw, "Open-set", "NMI")),
        "auroc": clean_number(nested_get(raw, "Unknown_binary", "AUROC")),
        "aupr": clean_number(nested_get(raw, "Unknown_binary", "AUPR")),
        "aupr_neg": clean_number(nested_get(raw, "Unknown_binary", "AUPR-neg")),
        "ms_s": clean_number(nested_get(raw, "Unknown_binary", "MS-s")),
        "ms_t": clean_number(nested_get(raw, "Unknown_binary", "MS-t")),
        "ms_s_with_ent": clean_number(nested_get(raw, "Unknown_binary", "MS-s-w/ent")),
        "ms_t_with_ent": clean_number(nested_get(raw, "Unknown_binary", "MS-t-w/ent")),
        "unims": clean_number(nested_get(raw, "Unknown_binary", "UniMS")),
    }
    if metrics["accuracy"] is None:
        metrics["accuracy"] = metrics["ucr"]
    return metrics


def discover_candidates() -> list[RunRecord]:
    candidates: list[RunRecord] = []
    for run_dir in OUTPUT_ROOT.glob("*/*"):
        if not run_dir.is_dir() or not run_dir.name.startswith("runs_"):
            continue
        if "_smoke_" in run_dir.name:
            continue

        match = RUN_RE.match(run_dir.name)
        if not match:
            continue

        metric_files = sorted(
            (run_dir / "metrics").glob("metric_*.json"),
            key=parse_iteration,
        )
        if not metric_files:
            continue

        info = match.groupdict()
        dataset = info["dataset"]
        source_private = int(info["source_private"])
        target_private = int(info["target_private"])
        setting = infer_setting(source_private, target_private)
        metric_file = metric_files[-1]
        metrics = extract_metrics(metric_file)
        primary = primary_metric(setting)
        if metrics.get(primary) is None:
            continue

        candidates.append(
            RunRecord(
                dataset=dataset,
                setting=setting,
                task=info["task"],
                shared=int(info["shared"]),
                source_private=source_private,
                target_private=target_private,
                exp_name=info["exp_name"],
                stamp=info["stamp"],
                final_iteration=parse_iteration(metric_file),
                run_dir=run_dir,
                metric_file=metric_file,
                seed=parse_seed(run_dir),
                metrics=metrics,
            )
        )
    return candidates


def select_completed_runs(candidates: list[RunRecord]) -> list[RunRecord]:
    selected: dict[tuple[str, str, str], RunRecord] = {}
    for record in candidates:
        key = (record.dataset, record.setting, record.task)
        current = selected.get(key)
        if current is None:
            selected[key] = record
            continue
        if (record.final_iteration, record.stamp) > (
            current.final_iteration,
            current.stamp,
        ):
            selected[key] = record
    return sorted(
        selected.values(),
        key=lambda item: (
            list(DATASET_NAMES).index(item.dataset),
            SETTING_ORDER.index(item.setting),
            TASK_ORDER[item.dataset].index(item.task),
        ),
    )


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def avg(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return round(mean(valid), 4)


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def record_to_json(record: RunRecord) -> dict[str, Any]:
    setting_metric = primary_metric(record.setting)
    return {
        "category_shift": record.setting,
        "task": record.task_label,
        "source_domain": record.source_domain,
        "target_domain": record.target_domain,
        "shared_classes": record.shared,
        "source_private_classes": record.source_private,
        "target_private_classes": record.target_private,
        "run_count": 1,
        "primary_metric": primary_metric_label(record.setting),
        "primary_value": record.metrics[setting_metric],
        "metrics": record.metrics,
        "run": {
            "seed": record.seed,
            "run_id": record.run_dir.name,
            "exp_name": record.exp_name,
            "final_iteration": record.final_iteration,
            "metric_file": rel(record.metric_file),
            "output_dir": rel(record.run_dir),
        },
    }


def group_by_dataset(records: list[RunRecord]) -> dict[str, list[RunRecord]]:
    grouped = {dataset: [] for dataset in DATASET_NAMES}
    for record in records:
        grouped[record.dataset].append(record)
    return grouped


def grouped_by_setting(records: list[RunRecord]) -> dict[str, list[RunRecord]]:
    grouped = {setting: [] for setting in SETTING_ORDER}
    for record in records:
        grouped[record.setting].append(record)
    return {setting: values for setting, values in grouped.items() if values}


def dataset_notes(dataset: str, records: list[RunRecord]) -> list[str]:
    notes: list[str] = []
    if dataset == "officehome":
        cda_exp_names = {
            record.exp_name
            for record in records
            if record.setting == "CDA" and record.exp_name != "exp-tasc20_final-0811_CDA_full"
        }
        if cda_exp_names:
            notes.append(
                "Office-Home CDA includes runs from "
                f"{', '.join(sorted(cda_exp_names))}; their configs still use method=TASC."
            )
    return notes


def dataset_json(dataset: str, records: list[RunRecord]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "view_type": "benchmark_result_view",
        "setting": "unida",
        "dataset": DATASET_NAMES[dataset],
        "method": "TASC",
        "backbone": "CLIP ViT-B/16",
        "table_layout": "transfer_pairs_by_category_shift",
        "primary_metric": {
            "CDA": "accuracy",
            "PDA": "accuracy",
            "ODA": "H-score",
            "OPDA": "H-score",
        },
        "aggregation": {
            "metric_source": "final",
            "reduction": "none",
            "num_runs": len(records),
        },
        "notes": dataset_notes(dataset, records),
        "results": [record_to_json(record) for record in records],
    }


def markdown_primary_table(dataset: str, records: list[RunRecord], setting: str) -> str:
    by_task = {record.task: record for record in records if record.setting == setting}
    ordered_tasks = [task for task in TASK_ORDER[dataset] if task in by_task]
    metric_key = primary_metric(setting)
    values = [by_task[task].metrics[metric_key] for task in ordered_tasks]
    header = [task_label(dataset, task) for task in ordered_tasks] + ["Avg"]
    row = [fmt(value) for value in values] + [fmt(avg(values))]
    return "\n".join(
        [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
            "| " + " | ".join(row) + " |",
        ]
    )


def markdown_aux_table(dataset: str, records: list[RunRecord], setting: str) -> str:
    by_task = {record.task: record for record in records if record.setting == setting}
    ordered_tasks = [task for task in TASK_ORDER[dataset] if task in by_task]
    rows = [
        "| Task | OS* | UNK | OS | AUROC | UCR |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for task in ordered_tasks:
        record = by_task[task]
        rows.append(
            "| "
            + " | ".join(
                [
                    record.task_label,
                    fmt(record.metrics["os_star"]),
                    fmt(record.metrics["unknown"]),
                    fmt(record.metrics["os"]),
                    fmt(record.metrics["auroc"]),
                    fmt(record.metrics["ucr"]),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def markdown_setting_section(dataset: str, records: list[RunRecord], setting: str) -> str:
    setting_records = [record for record in records if record.setting == setting]
    lines = [
        f"## {setting}",
        "",
        f"- primary_metric: {primary_metric_label(setting)}",
        f"- num_runs: {len(setting_records)}",
        "",
        markdown_primary_table(dataset, records, setting),
    ]
    if setting in {"ODA", "OPDA"}:
        lines.extend(["", "Auxiliary open-set metrics:", "", markdown_aux_table(dataset, records, setting)])
    return "\n".join(lines)


def markdown_dataset(dataset: str, records: list[RunRecord]) -> str:
    lines = [
        f"# TASC {DATASET_NAMES[dataset]} UniDA Results",
        "",
        "- method: TASC",
        "- backbone: CLIP ViT-B/16",
        "- primary_metric: H-score for ODA/OPDA; accuracy for CDA/PDA",
        "- aggregation: final metric snapshot per selected run",
        f"- num_runs: {len(records)}",
        f"- source: {SOURCE_ROOT_DISPLAY}",
    ]
    for note in dataset_notes(dataset, records):
        lines.append(f"- note: {note}")
    lines.append("")

    for setting in SETTING_ORDER:
        if any(record.setting == setting for record in records):
            lines.append(markdown_setting_section(dataset, records, setting))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def overview_rows(records: list[RunRecord]) -> list[list[str]]:
    rows: list[list[str]] = []
    for dataset, dataset_records in group_by_dataset(records).items():
        for setting in SETTING_ORDER:
            setting_records = [
                record for record in dataset_records if record.setting == setting
            ]
            if not setting_records:
                continue
            metric_key = primary_metric(setting)
            values = [record.metrics[metric_key] for record in setting_records]
            rows.append(
                [
                    DATASET_NAMES[dataset],
                    setting,
                    primary_metric_label(setting),
                    str(len(setting_records)),
                    fmt(avg(values)),
                ]
            )
    return rows


def markdown_overview_table(records: list[RunRecord]) -> str:
    lines = [
        "| Dataset | Setting | Primary metric | Num runs | Avg |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in overview_rows(records):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def markdown_summary(records: list[RunRecord]) -> str:
    lines = [
        "# TASC UniDA Results",
        "",
        "- method: TASC",
        "- backbone: CLIP ViT-B/16",
        "- primary_metric: H-score for ODA/OPDA; accuracy for CDA/PDA",
        "- aggregation: final metric snapshot per selected run",
        f"- num_runs: {len(records)}",
        f"- source: {SOURCE_ROOT_DISPLAY}",
        "- selection: group by dataset, category-shift setting, and transfer task; keep the run with the largest final iteration",
        "",
        "## Overview",
        "",
        markdown_overview_table(records),
        "",
    ]

    for dataset, dataset_records in group_by_dataset(records).items():
        if not dataset_records:
            continue
        lines.extend([f"## {DATASET_NAMES[dataset]}", ""])
        for setting in SETTING_ORDER:
            if any(record.setting == setting for record in dataset_records):
                lines.extend(
                    [
                        f"### {setting}",
                        "",
                        f"- primary_metric: {primary_metric_label(setting)}",
                        f"- num_runs: {sum(record.setting == setting for record in dataset_records)}",
                        "",
                        markdown_primary_table(dataset, dataset_records, setting),
                        "",
                    ]
                )
    return "\n".join(lines).rstrip() + "\n"


def summary_json(records: list[RunRecord]) -> dict[str, Any]:
    grouped = group_by_dataset(records)
    return {
        "schema_version": "1.0",
        "view_type": "benchmark_result_collection",
        "setting": "unida",
        "method": "TASC",
        "backbone": "CLIP ViT-B/16",
        "source": SOURCE_ROOT_DISPLAY,
        "aggregation": {
            "metric_source": "final",
            "reduction": "none",
            "num_runs": len(records),
            "selection": (
                "group by dataset, category-shift setting, and transfer task; "
                "keep the run with the largest final iteration"
            ),
        },
        "overview": [
            {
                "dataset": row[0],
                "category_shift": row[1],
                "primary_metric": row[2],
                "num_runs": int(row[3]),
                "average": None if row[4] == "-" else float(row[4]),
            }
            for row in overview_rows(records)
        ],
        "datasets": {
            DATASET_NAMES[dataset]: dataset_json(dataset, dataset_records)
            for dataset, dataset_records in grouped.items()
            if dataset_records
        },
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    candidates = discover_candidates()
    selected = select_completed_runs(candidates)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    grouped = group_by_dataset(selected)
    for dataset, records in grouped.items():
        if not records:
            continue
        stem = DATASET_FILES[dataset]
        (RESULT_ROOT / f"{stem}.md").write_text(
            markdown_dataset(dataset, records),
            encoding="utf-8",
        )
        write_json(RESULT_ROOT / f"{stem}.json", dataset_json(dataset, records))

    (RESULT_ROOT / "tasc_unida_results.md").write_text(
        markdown_summary(selected),
        encoding="utf-8",
    )
    write_json(RESULT_ROOT / "tasc_unida_results.json", summary_json(selected))

    print(f"selected_runs={len(selected)}")
    print(f"candidate_runs={len(candidates)}")
    for dataset, records in grouped.items():
        if records:
            by_setting = grouped_by_setting(records)
            counts = ", ".join(
                f"{setting}:{len(setting_records)}"
                for setting, setting_records in by_setting.items()
            )
            print(f"{DATASET_NAMES[dataset]}={len(records)} ({counts})")
    print(f"wrote={rel(RESULT_ROOT)}")


if __name__ == "__main__":
    main()
