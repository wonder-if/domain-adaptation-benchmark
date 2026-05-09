# Suites

`dabench.suite` provides a thin configuration layer on top of `setting`. It builds reusable experiment items from dataset defaults, setting defaults, and dataset/domain pairs.

## Overview

```text
build_suites(datasets=..., setting=..., format=...)
    -> suite descriptor
        -> settings
            -> load_suite_item(item)
                -> load_uda(...) / load_dg(...) / load_unida(...)
```

## Build suites

Use the unified builder:

```python
from dabench.suite import build_suites

suite = build_suites(datasets="domainnet", setting="uda", format="hf")[0]
print(suite["suite_id"])
print(len(suite["settings"]))
```

`datasets` can be a single dataset name or a list of names. Current built-in suites are:

- `office-31`
- `office-home`
- `domainnet`
- `minidomainnet`
- `visda-2017`

Current built-in settings are:

- `uda`
- `dg`
- `unida`

## Execute one item

`load_suite_item(...)` removes the human-readable `name` field and dispatches to the setting loader based on `item["setting"]:`

```python
from dabench.suite import build_suites, load_suite_item

suite = build_suites(datasets="domainnet", setting="uda", format="hf")[0]
item = suite["settings"][0]
train_loader, val_loader, test_loader = load_suite_item(item)
```

For `unida`, the returned object is the `load_unida(...)` payload dictionary instead of a 3-loader tuple.

## Built-in suite shapes

The suite builder intentionally keeps task generation deterministic and explicit:

| Setting | Dataset family | Built-in item shape |
| --- | --- | --- |
| `uda` | all supported datasets | all ordered source-to-target pairs |
| `dg` | all supported datasets | one item per target domain, using all other domains as source domains |
| `unida` | `office-31`, `office-home` | all ordered source-target pairs across built-in scenario presets |
| `unida` | `domainnet` | only `painting`, `real`, `sketch` task pairs, `opda` preset |
| `unida` | `visda-2017` | one built-in `SR` task, `opda` preset |

Concrete examples:

- `office-home` UDA: 12 transfer pairs
- `minidomainnet` UDA: 12 transfer pairs
- `office-home` DG: 4 target-domain items
- `domainnet` DG: 6 target-domain items
- `office-home` UniDA: domain-pair tasks crossed with `cda/pda/oda/opda`

## Inspect suite metadata

Each suite descriptor contains:

- `suite_id`
- `name`
- `setting`
- `settings`
- `metadata.domains`

You can inspect it directly:

```python
from dabench.suite import build_suites

suite = build_suites(datasets="minidomainnet", setting="dg", format="hf")[0]
print(suite["suite_id"])
print(suite["metadata"]["domains"])
print(suite["settings"][0]["name"])
```

## Typical use

The suite layer is the easiest way to run repeatable benchmark experiments:

- choose dataset(s)
- choose `setting="uda"`, `setting="dg"`, or `setting="unida"`
- choose `format="hf"` or `format="torch"`
- iterate over `suite["settings"]`
