# Datasets

`dabench` separates dataset access into two layers:

- `load_view(...)` is the user-facing entrypoint. It resolves the local dataset path from config and applies dataset-specific split/domain rules.
- `load_hf_dataset(path)` is the lower-level primitive for already-prepared Hugging Face datasets.

## Common usage

Use `load_view` in experiments and scripts:

```python
from dabench.data import load_view

domainnet = load_view("domainnet", domain="clipart", split="train", format="hf")
office_home = load_view("office-home", domain="Art", format="hf")
office31 = load_view("office-31", domain="amazon", format="torch")
visda = load_view("visda-2017", domain="synthetic", split="train", format="hf")
```

The returned object is either a native `datasets.Dataset` or a torch-style wrapper, depending on `format`.

## Dataset rules

The four UDA datasets currently supported by the suite layer have different layout rules:

| Dataset | View rule |
| --- | --- |
| Office-31 | domain only, split is ignored |
| Office-Home | domain only, single `train` split is filled internally |
| DomainNet | domain + explicit `train` / `test` split |
| VisDA-2017 | `synthetic -> train`, `real -> validation` |

These rules are applied by `load_view`, not by user code.

## Lower-level loader

If you already have a prepared Hugging Face dataset directory, use the lower-level loader directly:

```python
from dabench.data import load_hf_dataset

dataset_dict = load_hf_dataset("/path/to/domainnet_prepared")
```

This function expects a prepared local directory containing `dataset_info.json` and Arrow shards.

## Download

Loading is local-only. Use `download_dataset` to prepare data explicitly before training:

```python
from dabench.storage import download_dataset

download_dataset(
    "domainnet",
    dest="/path/to/domainnet_prepared",
    source="mirror",
    proxy="disable",
)
```

Office-31 uses the ModelScope Git LFS repository and prepares a local image layout:

```bash
dabench download office-31 --dest /path/to/office31 --proxy disable
```

## Suite layer

For UDA experiments, prefer the suite layer on top of `load_view`:

```python
from dabench.suite import build_suites, load_suite_item

suite = build_suites(datasets="domainnet", setting="uda", format="hf")[0]
item = suite["settings"][0]
train_loader, val_loader, test_loader = load_suite_item(item)
```

`build_suites(...)` builds the dataset suite, `load_suite_item(...)` executes one item, and the setting loader decides which split to use internally.
