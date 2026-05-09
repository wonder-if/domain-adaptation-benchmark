# Results

`dabench.results` standardizes how research projects record one experiment run and how benchmark-facing tables are derived from those runs.

## Why this layer exists

Training code is method-owned, but result structure should not be method-owned. If each project writes logs in its own shape, later comparison, aggregation, and table generation become ad hoc.

The results layer fixes that by separating:

- runtime experiment recording
- canonical single-run JSON
- benchmark-specific result views
- final Markdown table rendering

## Core objects

```text
research project
    -> ExperimentRecorder
        -> RunRecord JSON
            -> build_uda_result_view(...)
                -> render_uda_markdown_table(...)
```

- `RunRecord`: one actual run, usually one seed and one transfer pair.
- `BenchmarkResultView`: one benchmark-facing aggregated view derived from multiple run records.

## Current defaults

- one run writes one JSON file
- UDA is the first supported result setting
- paper-style tables are dataset-specific:
  - `office-31`: flattened transfer pairs
  - `office-home`: flattened transfer pairs
  - `domainnet`: source-to-target matrix
  - `visda-2017`: per-class table

## Next pages

- [Schema](schema.md): exact fields and JSON examples
- [Integration](integration.md): how another research project should call the recorder API
