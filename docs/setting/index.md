# Settings

`dabench.setting` contains task-level entrypoints. The current UDA path does not ask users to manage split details directly. The setting layer selects the right split for each dataset internally and returns three loaders: train, val, and test.

## UDA flow

The current flow is:

```text
suite item -> load_suite_item -> load_uda -> load_view -> prepared local data
```

Example:

```python
from dabench.setting import load_uda

train_loader, val_loader, test_loader = load_uda(
    dataset="domainnet",
    source_domain="clipart",
    target_domain="real",
    source_train_batch_size=32,
)
```

`load_uda` now routes dataset-specific split behavior internally:

- `office-31`: split is ignored
- `office-home`: single `train` split is filled internally
- `domainnet`: source/target train use `train`, validation/test use `test`
- `visda-2017`: source uses `synthetic/train`, target uses `real/validation`, and test currently reuses validation

`load_uda` returns loaders only. Model code, optimizers, logging, and evaluation loops remain user-owned.

## Suite helpers

For batchable experiment configuration, use the suite layer:

```python
from dabench.suite import build_suites, load_suite_item

suite = build_suites(datasets="domainnet", setting="uda", format="hf")[0]
item = suite["settings"][0]
train_loader, val_loader, test_loader = load_suite_item(item)
```
