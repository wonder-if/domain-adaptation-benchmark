# dabench

`dabench` is a lightweight dataset utility package for domain adaptation research.

## Overview

The project is organized as a layered loading flow:

```text
storage  ->  data  ->  setting  ->  suite
manifest     load_view  load_uda   build_suites
prepare      load_hf_dataset       load_suite_item
```

- `storage` prepares local dataset directories from manifests.
- `data` loads one concrete dataset view from a local path or prepared config.
- `setting` assembles task-level loaders such as UDA.
- `suite` builds batchable experiment configurations on top of settings.

Start with [Datasets](data/index.md), then [Settings](setting/index.md), then [Suites](suite/index.md).
