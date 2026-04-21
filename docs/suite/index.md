# Suites

`dabench.suite` provides a thin configuration layer on top of `setting`. It builds reusable experiment items from dataset defaults, setting defaults, and dataset/domain pairs.

## Overview

```text
build_suites(datasets=..., setting=..., format=...)
    -> suite descriptor
        -> settings
            -> load_suite_item(item)
                -> load_uda(...)
```

## Build suites

Use the unified builder:

```python
from dabench.suite import build_suites

suite = build_suites(datasets="domainnet", setting="uda", format="hf")[0]
print(suite["suite_id"])
print(len(suite["settings"]))
```

`datasets` can be a single dataset name or a list of names. Current built-in UDA suites are:

- `office-31`
- `office-home`
- `domainnet`
- `visda-2017`

## Execute one item

`load_suite_item(...)` removes the human-readable `name` field and dispatches to the setting loader:

```python
from dabench.suite import build_suites, load_suite_item

suite = build_suites(datasets="domainnet", setting="uda", format="hf")[0]
item = suite["settings"][0]
train_loader, val_loader, test_loader = load_suite_item(item)
```

## Typical use

The suite layer is the easiest way to run repeatable UDA experiments:

- choose dataset(s)
- choose `setting="uda"`
- choose `format="hf"` or `format="torch"`
- iterate over `suite["settings"]`
