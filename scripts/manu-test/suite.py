"""Minimal manual test for the built-in UDA suites."""

from __future__ import annotations

from dabench.suite import build_suites, load_suite_item


def _table(rows: list[dict[str, object]], headers: tuple[str, ...]) -> str:
    widths = {
        key: max(len(key), max(len(str(row.get(key, ""))) for row in rows))
        for key in headers
    }
    lines = []
    lines.append("| " + " | ".join(key.ljust(widths[key]) for key in headers) + " |")
    lines.append("| " + " | ".join("-" * widths[key] for key in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")).ljust(widths[key]) for key in headers) + " |")
    return "\n".join(lines)


def _first_batch(loader):
    return next(iter(loader))


def _shape(value) -> str:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return str(tuple(shape))
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            parts.append(f"{key}={_shape(item)}")
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_shape(item) for item in value) + "]"
    return type(value).__name__


def _count_and_shape(batch) -> tuple[str, str]:
    if isinstance(batch, dict) and "pixel_values" in batch:
        pixel_values = batch["pixel_values"]
        parts = [f"pixel_values={_shape(pixel_values)}"]
        if "labels" in batch:
            parts.append(f"labels={_shape(batch['labels'])}")
        return str(int(pixel_values.shape[0])), "; ".join(parts)
    if isinstance(batch, dict) and "source" in batch and "target" in batch:
        source = batch["source"]
        target = batch["target"]
        source_count, source_shape = _count_and_shape(source)
        target_count, target_shape = _count_and_shape(target)
        return (
            f"source={source_count}, target={target_count}",
            f"source:{source_shape}; target:{target_shape}",
        )
    return "?", _shape(batch)


def _dataset_specs() -> list[dict[str, object]]:
    return [
        {
            "dataset": "office-31",
            "format": "hf",
        },
        {
            "dataset": "office-home",
            "format": "hf",
        },
        {
            "dataset": "domainnet",
            "format": "hf",
        },
        {
            "dataset": "visda-2017",
            "format": "hf",
        },
    ]


def main() -> None:
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for spec in _dataset_specs():
        dataset_name = str(spec["dataset"])
        format_name = str(spec["format"])

        print(f"[build] {dataset_name} format={format_name}", flush=True)
        suite = build_suites(datasets=dataset_name, setting="uda", format=format_name)[0]
        settings = suite["settings"]
        summary_rows.append(
            {
                "dataset": dataset_name,
                "format": format_name,
                "num_items": len(settings),
            }
        )

        item = settings[0]
        print(f"[load] {dataset_name} format={format_name} item={item['name']}", flush=True)
        try:
            train_loader, val_loader, test_loader = load_suite_item(item)
            train_batch = _first_batch(train_loader)
            val_batch = _first_batch(val_loader)
            test_batch = _first_batch(test_loader)

            train_count, train_shape = _count_and_shape(train_batch)
            val_count, val_shape = _count_and_shape(val_batch)
            test_count, test_shape = _count_and_shape(test_batch)

            detail_rows.append(
                {
                    "dataset": dataset_name,
                    "format": format_name,
                    "name": item["name"],
                    "train": train_count,
                    "train_shape": train_shape,
                    "val": val_count,
                    "val_shape": val_shape,
                    "test": test_count,
                    "test_shape": test_shape,
                }
            )
        except Exception as exc:
            detail_rows.append(
                {
                    "dataset": dataset_name,
                    "format": format_name,
                    "name": item["name"],
                    "train": "ERR",
                    "train_shape": f"{type(exc).__name__}: {exc}",
                    "val": "",
                    "val_shape": "",
                    "test": "",
                    "test_shape": "",
                }
            )

    print("suite counts")
    print(_table(summary_rows, ("dataset", "format", "num_items")))
    print()
    print("loader first batch")
    print(_table(detail_rows, ("dataset", "format", "name", "train", "train_shape", "val", "val_shape", "test", "test_shape")))


if __name__ == "__main__":
    main()
