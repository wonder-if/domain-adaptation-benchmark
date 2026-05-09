# Integration

This page is for the actual research project that trains or evaluates methods.

The recommended pattern is:

1. initialize one recorder for one run
2. log validation/test results explicitly
3. mark or auto-track the best checkpoint
4. finalize once and write one `RunRecord` JSON

## Minimal runtime example

```python
from dabench.results import ExperimentRecorder

recorder = ExperimentRecorder(
    dataset="office-home",
    setting="uda",
    method="DAMP",
    backbone="RN50",
    source_domain="Art",
    target_domain="Clipart",
    seed=1,
    output_dir="experiments/office_home/art_to_clipart/seed_1",
    records_root="experiments/records",
    config={
        "lr": 0.003,
        "batch_size": 32,
    },
)

for epoch in range(1, 31):
    val_metrics = evaluate_on_val(...)
    recorder.log_eval(
        step=epoch,
        split="val",
        metrics=val_metrics,
        checkpoint=f"epoch-{epoch}",
    )

test_metrics = evaluate_on_test(...)
recorder.log_eval(
    step=30,
    split="test",
    metrics=test_metrics,
    checkpoint="epoch-30",
)

path = recorder.finalize()
print(path)
```

Notes:

- `val` events are used to auto-track `best_metrics` when `selection_metric` is present.
- `finalize()` uses the last `test` event as `final_metrics` by default.
- if there is no `test` event, it falls back to the last logged event.

## VisDA example with class metrics

```python
from dabench.results import ExperimentRecorder

recorder = ExperimentRecorder(
    dataset="visda-2017",
    setting="uda",
    method="DAMP",
    backbone="ViT-B/16",
    source_domain="synthetic",
    target_domain="real",
    seed=1,
    records_root="experiments/records",
)

metrics, class_metrics = evaluate_visda(...)
recorder.log_eval(
    step=30,
    split="test",
    metrics=metrics,
    class_metrics=class_metrics,
    checkpoint="epoch-30",
)
recorder.finalize()
```

## Aggregation and table generation

Once multiple runs exist, aggregate them in a separate step:

```python
from dabench.results import (
    build_uda_result_view,
    collect_run_records,
    render_uda_markdown_table,
)

runs = collect_run_records(
    "experiments/records",
    dataset="office-home",
    setting="uda",
    method="DAMP",
)

view = build_uda_result_view(runs, metric_source="final", reduction="mean")
table = render_uda_markdown_table(view)
print(table)
```

## Required caller-owned fields

The research project must provide:

- `dataset`
- `setting`
- `method`
- `seed`
- `source_domain` and `target_domain` for UDA
- actual evaluated metrics

The research project should usually also provide:

- `backbone`
- `output_dir`
- key hyperparameters in `config`

## Compatibility path

If an older project already writes method-specific logs, use a separate conversion script to create `RunRecord` JSON. That compatibility path should end at the same schema as the runtime recorder.
