#!/usr/bin/env python3
"""Extract minimal UDA result JSON and Markdown tables from DAMP logs."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from dabench.results import (
    build_uda_result_view,
    make_run_record,
    render_uda_markdown_table,
    write_run_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "related_works" / "damp" / "codes" / "damp-dabench" / "output"
REPORT_ROOT = REPO_ROOT / "related_works" / "damp" / "results" / "uda"
RUN_RECORD_ROOT = REPORT_ROOT / "run_records"

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

MINIDOMAINNET_DOMAIN_DISPLAY = {
    "clipart": "clipart",
    "painting": "painting",
    "real": "real",
    "sketch": "sketch",
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


def _parse_result_blocks(log_text: str) -> list[dict[str, object]]:
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
            if (
                line.startswith("epoch [")
                or line.startswith("Checkpoint saved")
                or line.startswith("Finish training")
                or line.startswith("Elapsed:")
                or line.startswith("Deploy the last-epoch model")
            ):
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
            block: dict[str, object] = {"metrics": metrics, "eval_index": len(blocks) + 1}
            if class_metrics:
                block["class_metrics"] = class_metrics
            blocks.append(block)
        continue
    if not blocks:
        raise ValueError("No evaluation result block found in log.")
    return blocks


def _best_result_block(blocks: list[dict[str, object]], metric_name: str) -> dict[str, object]:
    def metric_value(block: dict[str, object]) -> float:
        metrics = block["metrics"]
        assert isinstance(metrics, dict)
        value = metrics.get(metric_name)
        if not isinstance(value, (int, float)):
            raise ValueError(f"Missing best-selection metric {metric_name!r}.")
        return float(value)

    return max(blocks, key=metric_value)


def _metrics(block: dict[str, object]) -> dict[str, float]:
    return dict(block["metrics"])


def _class_metrics(block: dict[str, object]) -> dict[str, float]:
    return dict(block.get("class_metrics", {}))


def _eval_history(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for block in blocks:
        event: dict[str, object] = {
            "split": "test",
            "step": int(block["eval_index"]),
            "metrics": _metrics(block),
        }
        class_metrics = _class_metrics(block)
        if class_metrics:
            event["class_metrics"] = class_metrics
        events.append(event)
    return events


def _add_payload_metadata(payload: dict[str, object], *, backbone: str) -> dict[str, object]:
    payload["backbone"] = backbone
    payload["display_name"] = f"{payload['dataset']} ({backbone})"
    return payload


def _office_home_record(run_dir: Path) -> dict[str, object]:
    task_name = run_dir.name.split("_", 2)[2]
    source, target = task_name.split("_to_")
    seed_dir = run_dir / "seed_1"
    log_path = _select_log_with_result(seed_dir)
    blocks = _parse_result_blocks(_read_text(log_path))
    best = _best_result_block(blocks, "accuracy")
    return make_run_record(
        dataset="office-home",
        setting="uda",
        method="DAMP",
        backbone="RN50",
        source_domain=OFFICE_HOME_DOMAIN_DISPLAY[source],
        target_domain=OFFICE_HOME_DOMAIN_DISPLAY[target],
        seed=1,
        status="completed",
        selected_checkpoint="best",
        selection_metric="accuracy",
        output_dir=str(seed_dir.relative_to(REPO_ROOT)),
        config={"log_path": str(log_path.relative_to(REPO_ROOT))},
        final_metrics=_metrics(blocks[-1]),
        best_metrics=_metrics(best),
        eval_history=_eval_history(blocks),
    )


def _minidomainnet_record(run_dir: Path, *, backbone: str) -> dict[str, object]:
    task_name = run_dir.name.split("_", 2)[2]
    source, target = task_name.split("_to_")
    seed_dir = run_dir / "seed_1"
    log_path = _select_log_with_result(seed_dir)
    blocks = _parse_result_blocks(_read_text(log_path))
    best = _best_result_block(blocks, "accuracy")
    return make_run_record(
        dataset="minidomainnet",
        setting="uda",
        method="DAMP",
        backbone=backbone,
        source_domain=MINIDOMAINNET_DOMAIN_DISPLAY[source],
        target_domain=MINIDOMAINNET_DOMAIN_DISPLAY[target],
        seed=1,
        status="completed",
        selected_checkpoint="best",
        selection_metric="accuracy",
        output_dir=str(seed_dir.relative_to(REPO_ROOT)),
        config={"log_path": str(log_path.relative_to(REPO_ROOT))},
        final_metrics=_metrics(blocks[-1]),
        best_metrics=_metrics(best),
        eval_history=_eval_history(blocks),
    )


def build_office_home_payload() -> tuple[list[dict[str, object]], dict[str, object]]:
    run_root = OUTPUT_ROOT / "office_home" / "DAMP" / "damp"
    records = [_office_home_record(run_dir) for run_dir in sorted(run_root.iterdir()) if run_dir.is_dir()]
    payload = build_uda_result_view(records, metric_source="best")
    return records, _add_payload_metadata(payload, backbone="RN50")


def build_minidomainnet_payload(run_name: str, *, backbone: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    run_root = OUTPUT_ROOT / "miniDomainNet" / "DAMP" / run_name
    records = [
        _minidomainnet_record(run_dir, backbone=backbone)
        for run_dir in sorted(run_root.iterdir())
        if run_dir.is_dir()
    ]
    payload = build_uda_result_view(records, metric_source="best")
    return records, _add_payload_metadata(payload, backbone=backbone)


def _visda_metrics(block: dict[str, object]) -> dict[str, float]:
    metrics = _metrics(block)
    if "average" in metrics:
        metrics["average_class_accuracy"] = metrics.pop("average")
    return metrics


def _visda_eval_history(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for block in blocks:
        event: dict[str, object] = {
            "split": "test",
            "step": int(block["eval_index"]),
            "metrics": _visda_metrics(block),
            "class_metrics": _class_metrics(block),
        }
        events.append(event)
    return events


def build_visda_payload() -> tuple[list[dict[str, object]], dict[str, object]]:
    run_dir = OUTPUT_ROOT / "visda17" / "DAMP" / "damp" / "0.5_2.0_synthetic_to_real"
    seed_dir = run_dir / "seed_1"
    log_path = _select_log_with_result(seed_dir)
    blocks = _parse_result_blocks(_read_text(log_path))
    for block in blocks:
        metrics = block["metrics"]
        assert isinstance(metrics, dict)
        metrics["average_class_accuracy"] = metrics.pop("average")
    best = _best_result_block(blocks, "average_class_accuracy")
    record = make_run_record(
        dataset="visda-2017",
        setting="uda",
        method="DAMP",
        backbone="ViT-B/16",
        source_domain=VISDA_DOMAIN_DISPLAY["synthetic"],
        target_domain=VISDA_DOMAIN_DISPLAY["real"],
        seed=1,
        status="completed",
        selected_checkpoint="best",
        selection_metric="average_class_accuracy",
        output_dir=str(seed_dir.relative_to(REPO_ROOT)),
        config={"log_path": str(log_path.relative_to(REPO_ROOT))},
        final_metrics=_visda_metrics(blocks[-1]),
        best_metrics=_visda_metrics(best),
        final_class_metrics=_class_metrics(blocks[-1]),
        best_class_metrics=_class_metrics(best),
        eval_history=_visda_eval_history(blocks),
    )
    payload = build_uda_result_view([record], metric_source="best", primary_metric="average_class_accuracy")
    return [record], _add_payload_metadata(payload, backbone="ViT-B/16")


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
        f"# {payload['dataset']} UDA Results ({payload.get('backbone', 'unknown backbone')})",
        "",
        f"- method: {payload['method']}",
        f"- backbone: {payload.get('backbone', 'unknown')}",
        f"- primary_metric: {payload['primary_metric']}",
        f"- metric_source: {payload['aggregation']['metric_source']}",
        "",
        table,
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")


def _write_summary(payloads: list[tuple[str, dict[str, object]]]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = ["# DAMP UDA Results", ""]
    for _, payload in payloads:
        lines.extend(
            [
                f"## {payload.get('display_name', payload['dataset'])}",
                "",
                f"- method: {payload['method']}",
                f"- backbone: {payload.get('backbone', 'unknown')}",
                f"- primary_metric: {payload['primary_metric']}",
                f"- metric_source: {payload['aggregation']['metric_source']}",
                f"- num_runs: {payload['aggregation']['num_runs']}",
                "",
                render_uda_markdown_table(payload),
                "",
            ]
        )
    (REPORT_ROOT / "damp_uda_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if RUN_RECORD_ROOT.exists():
        shutil.rmtree(RUN_RECORD_ROOT)
    office_home_records, office_home_payload = build_office_home_payload()
    minidomainnet_rn50_records, minidomainnet_rn50_payload = build_minidomainnet_payload("damp", backbone="RN50")
    minidomainnet_vit_records, minidomainnet_vit_payload = build_minidomainnet_payload(
        "damp_vit_b_16",
        backbone="ViT-B/16",
    )
    visda_records, visda_payload = build_visda_payload()
    _write_run_records(office_home_records)
    _write_run_records(minidomainnet_rn50_records)
    _write_run_records(minidomainnet_vit_records)
    _write_run_records(visda_records)
    payloads = [
        ("office_home_damp", office_home_payload),
        ("minidomainnet_damp", minidomainnet_rn50_payload),
        ("minidomainnet_damp_vit_b_16", minidomainnet_vit_payload),
        ("visda17_damp", visda_payload),
    ]
    for name, payload in payloads:
        _write_payload(name, payload)
    _write_summary(payloads)


if __name__ == "__main__":
    main()
