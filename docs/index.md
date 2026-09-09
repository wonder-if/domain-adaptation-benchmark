# dabench

`dabench` is a lightweight dataset utility package for domain adaptation research.

![dabench dataset overview](assets/dataset_matrix_overview.png)

## Overview

The project is organized as a layered loading flow:

```text
storage  ->  data  ->  setting  ->  suite
manifest     load_view  load_uda   build_suites
prepare      load_hf_dataset       load_suite_item

models
load_model
```

- `storage` prepares local dataset directories from manifests.
- `data` loads one concrete image or cached-feature dataset view from local config.
- `models` loads local pretrained backbones from the path config without adaptation algorithms.
- `setting` assembles task-level loaders such as UDA.
- `suite` builds batchable experiment configurations on top of settings.
- `results` records one run and aggregates benchmark-facing tables.

Start with [Datasets](data/index.md) and [Models](model/index.md), then
[Settings](setting/index.md), [Suites](suite/index.md), and [Results](results/index.md).
