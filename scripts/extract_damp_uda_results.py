#!/usr/bin/env python3
"""Extract minimal UDA result JSON and Markdown tables from DAMP logs."""

from __future__ import annotations

import json
import re
from pathlib import Path

from dabench.results import (
    build_uda_result_view,
    make_run_record,
    render_uda_markdown_table,
    write_run_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "related_works" / "damp" / "codes" / "damp-dabench" / "output"
REPORT_ROOT = REPO_ROOT / "reports" / "results" / "uda"
RUN_RECORD_ROOT = REPO_ROOT / "reports" / "results" / "run_records"

RESULT_HEADER_RE = re.compile(r"^=> result$")
METRIC_RE = re.compile(r"^\* ([a-z0-9_]+): ([0-9.]+)%$")
CLASS_RE = re.compile(r"^\* class: \d+ \(([^)]+)\)\s+total: [0-9,]+\s+correct: [0-9,]+\s+acc: ([0-9.]+)%$")

OFFICE_HOME_DOMAIN_DISPLAY = {
    "art": "Art",
    "clipart": "Clipart",
    "product": "Product",
    "real_world": "Real World",
}

VISDA_DOMAIN_DISPLAY = {
    "synthetic": "synthetic",
    "real": "real",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _select_log_with_result(seed_dir: Path) -> Path:
    candidates = sorted(seed_dir.glob("log.txt*"))
    for path in reversed(candidates):
        text = _read_text(path)
        if "=> result" in text and "Deploy the last-epoch model" in text:
            return path
    for path in reversed(candidates):
        text = _read_text(path)
        if "=> result" in text:
            return path
    raise ValueError(f"No completed result log found under {seed_dir}.")


def _parse_last_result_block(log_text: str) -> dict[str, object]:
    lines = log_text.splitlines()
    blocks: list[dict[str, object]] = []
    i = 0
    while i < len(lines):
        if not RESULT_HEADER_RE.match(lines[i].strip()):
            i += 1
            continue
        metrics: dict[str, float] = {}
        class_metrics: dict[str, float] = {}
        i += 1
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("epoch [") or line.startswith("Checkpoint saved") or line.startswith("Finish training") or line.startswith("Elapsed:"):
                break
            if RESULT_HEADER_RE.match(line):
                break
            metric_match = METRIC_RE.match(line)
            if metric_match:
                metrics[metric_match.group(1)] = float(metric_match.group(2))
                i += 1
                continue
            class_match = CLASS_RE.match(line)
            if class_match:
                class_metrics[class_match.group(1)] = float(class_match.group(2))
                i += 1
                continue
            i += 1
        if metrics:
            block: dict[str, object] = {"metrics": metrics}
            if class_metrics:
                block["class_metrics"] = class_metrics
            blocks.append(block)
        continue
    if not blocks:
        raise ValueError("No evaluation result block found in log.")
    return blocks[-1]


def _office_home_record(run_dir: Path) -> dict[str, object]:
    task_name = run_dir.name.split("_", 2)[2]
    source, target = task_name.split("_to_")
    seed_dir = run_dir / "seed_1"
    log_path = _select_log_with_result(seed_dir)
    parsed = _parse_last_result_block(_read_text(log_path))
    metrics = dict(parsed["metrics"])
    return make_run_record(
        dataset="office-home",
        setting="uda",
        method="DAMP",
        backbone="RN50",
        source_domain=OFFICE_HOME_DOMAIN_DISPLAY[source],
        target_domain=OFFICE_HOME_DOMAIN_DISPLAY[target],
        seed=1,
        status="completed",
        output_dir=str(seed_dir.relative_to(REPO_ROOT)),
        config={"log_path": str(log_path.relative_to(REPO_ROOT))},
        final_metrics=metrics,
    )


def build_office_home_payload() -> tuple[list[dict[str, object]], dict[str, object]]:
    run_root = OUTPUT_ROOT / "office_home" / "DAMP" / "damp"
    records = [_office_home_record(run_dir) for run_dir in sorted(run_root.iterdir()) if run_dir.is_dir()]
    return records, build_uda_result_view(records)


def build_visda_payload() -> tuple[list[dict[str, object]], dict[str, object]]:
    run_dir = OUTPUT_ROOT / "visda17" / "DAMP" / "damp" / "0.5_2.0_synthetic_to_real"
    seed_dir = run_dir / "seed_1"
    log_path = _select_log_with_result(seed_dir)
    parsed = _parse_last_result_block(_read_text(log_path))
    metrics = dict(parsed["metrics"])
    class_metrics = dict(parsed.get("class_metrics", {}))
    metrics["average_class_accuracy"] = metrics.pop("average")
    record = make_run_record(
        dataset="visda-2017",
        setting="uda",
        method="DAMP",
        backbone="ViT-B/16",
        source_domain=VISDA_DOMAIN_DISPLAY["synthetic"],
        target_domain=VISDA_DOMAIN_DISPLAY["real"],
        seed=1,
        status="completed",
        output_dir=str(seed_dir.relative_to(REPO_ROOT)),
        config={"log_path": str(log_path.relative_to(REPO_ROOT))},
        final_metrics=metrics,
        final_class_metrics=class_metrics,
    )
    return [record], build_uda_result_view([record])


def _write_run_records(records: list[dict[str, object]]) -> None:
    for record in records:
        write_run_record(record, records_root=RUN_RECORD_ROOT)


def _write_payload(name: str, payload: dict[str, object]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_ROOT / f"{name}.json"
    md_path = REPORT_ROOT / f"{name}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    table = render_uda_markdown_table(payload)
    md_lines = [
        f"# {payload['dataset']} UDA Results",
        "",
        f"- method: {payload['method']}",
        f"- primary_metric: {payload['primary_metric']}",
        "",
        table,
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    office_home_records, office_home_payload = build_office_home_payload()
    visda_records, visda_payload = build_visda_payload()
    _write_run_records(office_home_records)
    _write_run_records(visda_records)
    _write_payload("office_home_damp", office_home_payload)
    _write_payload("visda17_damp", visda_payload)


if __name__ == "__main__":
    main()
