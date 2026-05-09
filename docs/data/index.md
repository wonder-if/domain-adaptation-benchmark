# Datasets

`dabench` separates dataset access into two layers:

- `load_view(...)` is the user-facing entrypoint. It resolves the local dataset path from config and applies dataset-specific split/domain rules.
- `load_hf_dataset(path)` is the lower-level primitive for already-prepared Hugging Face datasets.

## Supported datasets

Current dataset loaders cover:

| Dataset | Canonical id | Domains / task axis | Notes |
| --- | --- | --- | --- |
| Office-31 | `office-31` | `amazon`, `dslr`, `webcam` | image-folder style, split is ignored |
| Office-Home | `office-home` | `Art`, `Clipart`, `Product`, `Real World` | single prepared split, domain-only access |
| DomainNet | `domainnet` | `clipart`, `infograph`, `painting`, `quickdraw`, `real`, `sketch` | explicit `train` / `test` split |
| miniDomainNet | `minidomainnet` | `clipart`, `painting`, `real`, `sketch` | filtered from prepared DomainNet using `splits_mini/*.txt` |
| VisDA-2017 | `visda-2017` | `synthetic`, `real` | loader routes to `train` / `validation` internally |

## Common usage

Use `load_view` in experiments and scripts:

```python
from dabench.data import load_view

domainnet = load_view("domainnet", domain="clipart", split="train", format="hf")
office_home = load_view("office-home", domain="Art", format="hf")
office31 = load_view("office-31", domain="amazon", format="torch")
visda = load_view("visda-2017", domain="synthetic", split="train", format="hf")
minidomainnet = load_view("minidomainnet", domain="clipart", split="train", format="hf")
```

The returned object is either a native `datasets.Dataset` or a torch-style wrapper, depending on `format`.

## Dataset rules

The supported benchmark datasets do not share one universal split rule:

| Dataset | View rule |
| --- | --- |
| Office-31 | domain only, split is ignored |
| Office-Home | domain only, single `train` split is filled internally |
| DomainNet | domain + explicit `train` / `test` split |
| miniDomainNet | same public API as DomainNet, but filtered from prepared DomainNet by split files |
| VisDA-2017 | `synthetic -> train`, `real -> validation` |

These rules are applied by `load_view`, not by user code.

## miniDomainNet

`minidomainnet` is not downloaded as a separate raw dataset. It is derived from prepared DomainNet data:

- `path` should point to prepared DomainNet data
- `split_dir` should point to the directory containing `clipart_train.txt`, `clipart_test.txt`, and the other mini split files
- if `split_dir` is omitted, `dabench` falls back to `splits_mini/` under the prepared dataset root

This keeps the external API simple:

```python
from dabench.data import load_view

train = load_view("minidomainnet", domain="real", split="train", format="hf")
test = load_view("minidomainnet", domain="real", split="test", format="hf")
```

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

`minidomainnet` does not currently define a separate downloader. Prepare DomainNet first, then point `minidomainnet` to the prepared DomainNet path plus the mini split files.

## Suite layer

For benchmark experiments, prefer the suite layer on top of `load_view`:

```python
from dabench.suite import build_suites, load_suite_item

suite = build_suites(datasets="domainnet", setting="uda", format="hf")[0]
item = suite["settings"][0]
train_loader, val_loader, test_loader = load_suite_item(item)
```

`build_suites(...)` builds the dataset suite, `load_suite_item(...)` executes one item, and the setting loader decides which split to use internally.
