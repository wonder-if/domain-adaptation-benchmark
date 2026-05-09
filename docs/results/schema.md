# Results Schema

`dabench` currently defines two result payloads:

- `RunRecord`: the canonical record written by a research run
- `BenchmarkResultView`: a derived view used for table generation

## RunRecord

Required identity fields:

- `record_type`: always `run_record`
- `schema_version`
- `dataset`
- `setting`
- `method`
- `seed`
- `run_id`
- `status`

Common metadata fields:

- `backbone`
- `algorithm_name`
- `source_domain`
- `target_domain`
- `output_dir`
- `start_time`
- `end_time`
- `selected_checkpoint`: `last` or `best`
- `selection_metric`
- `selection_mode`: `max` or `min`
- `config`
- `extra`

Result fields:

- `final_metrics`
- `best_metrics`
- `final_class_metrics`
- `best_class_metrics`
- `eval_history`

Failure/debug fields:

- `failure_reason`

### Minimal Office-Home example

```json
{
  "schema_version": "1.0",
  "record_type": "run_record",
  "metrics_schema_version": "1.0",
  "run_id": "office_home__uda__DAMP__RN50__Art_to_Clipart__seed1",
  "dataset": "office-home",
  "setting": "uda",
  "method": "DAMP",
  "backbone": "RN50",
  "source_domain": "Art",
  "target_domain": "Clipart",
  "seed": 1,
  "status": "completed",
  "output_dir": "experiments/office_home/art_to_clipart/seed_1",
  "selected_checkpoint": "last",
  "selection_metric": "accuracy",
  "selection_mode": "max",
  "start_time": "2026-05-09T12:00:00Z",
  "end_time": "2026-05-09T12:50:00Z",
  "config": {
    "lr": 0.003,
    "batch_size": 32
  },
  "extra": {},
  "final_metrics": {
    "accuracy": 59.1,
    "error": 40.9,
    "macro_f1": 56.8
  },
  "best_metrics": {
    "accuracy": 60.5,
    "error": 39.5,
    "macro_f1": 58.0
  },
  "final_class_metrics": {},
  "best_class_metrics": {},
  "eval_history": []
}
```

### VisDA example with class metrics

```json
{
  "schema_version": "1.0",
  "record_type": "run_record",
  "metrics_schema_version": "1.0",
  "run_id": "visda_2017__uda__DAMP__ViT_B_16__synthetic_to_real__seed1",
  "dataset": "visda-2017",
  "setting": "uda",
  "method": "DAMP",
  "backbone": "ViT-B/16",
  "source_domain": "synthetic",
  "target_domain": "real",
  "seed": 1,
  "status": "completed",
  "selected_checkpoint": "last",
  "selection_metric": "accuracy",
  "selection_mode": "max",
  "config": {},
  "extra": {},
  "final_metrics": {
    "accuracy": 87.7,
    "error": 12.3,
    "macro_f1": 88.5,
    "average_class_accuracy": 90.0
  },
  "best_metrics": {},
  "final_class_metrics": {
    "aeroplane": 98.7,
    "bicycle": 92.5,
    "bus": 86.0,
    "car": 75.0,
    "horse": 98.1,
    "knife": 97.5,
    "motorcycle": 94.5,
    "person": 78.0,
    "plant": 92.5,
    "skateboard": 95.9,
    "train": 94.6,
    "truck": 77.1
  },
  "best_class_metrics": {},
  "eval_history": []
}
```

## BenchmarkResultView

This object is not written by the algorithm project directly. It is derived from one or more `RunRecord` objects.

Common fields:

- `view_type`: currently `benchmark_result_view`
- `schema_version`
- `dataset`
- `setting`
- `method`
- `table_layout`
- `primary_metric`
- `aggregation`
- `results`

`results` is benchmark-shaped:

- transfer-pair datasets use one item per source-target pair
- matrix datasets use one item per source-target pair and are rendered as a matrix
- VisDA keeps one item with `class_metrics`

When multiple seeds are aggregated, each result item keeps:

- `run_count`
- `seeds`
- aggregated `metrics`
- optional `metrics_std`
- aggregated `class_metrics`

## Canonical path

Recommended layout for research projects:

```text
<experiment_root>/records/<dataset>/<setting>/<method>/<run_id>.json
```
