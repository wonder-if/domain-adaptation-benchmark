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
| Camelyon17 | `camelyon17` | `id_train`, `id_val`, `unlabeled_train`, `ood_val`, `ood_test` | WILDS split-only dataset |
| iWildCam | `iwildcam` | `train`, `test`; metadata includes `location` ids | WILDS split-only dataset with category metadata joined from configured JSON files |

## Common usage

Use `load_view` in experiments and scripts:

```python
from dabench.data import load_view

domainnet = load_view("domainnet", domain="clipart", split="train", format="hf")
office_home = load_view("office-home", domain="Art", format="hf")
office31 = load_view("office-31", domain="amazon", format="torch")
visda = load_view("visda-2017", domain="synthetic", split="train", format="hf")
minidomainnet = load_view("minidomainnet", domain="clipart", split="train", format="hf")
camelyon17 = load_view("camelyon17", split="ood_val", format="hf")
iwildcam = load_view("iwildcam", split="train", format="hf")
```

The returned object is either a native `datasets.Dataset` or a torch-style wrapper, depending on `format`.

Use `format="feature"` to load cached CLIP image features instead of images:

```python
from dabench.data import list_feature_caches, load_feature_decode_errors, load_text_features, load_view

features = load_view(
    "office-home",
    domain="Clipart",
    format="feature",
    feature_model="clip-vit-base-patch16",
)
wilds_features = load_view(
    "iwildcam",
    split="train",
    format="feature",
    feature_model="clip-vit-base-patch16",
)
mini_features = load_view(
    "minidomainnet",
    domain="clipart",
    split="train",
    format="feature",
    feature_model="clip-vit-base-patch16",
)
wilds_text_features = load_text_features(
    "iwildcam",
    split="train",
    feature_model="clip-vit-base-patch16",
    prompt="a photo of a {CLASS}.",
)
wilds_cache_records = list_feature_caches("iwildcam", include_stats=True)
wilds_decode_errors = load_feature_decode_errors(
    "iwildcam",
    split="train",
    feature_model="clip-vit-base-patch16",
)
text_features = load_text_features(
    "office-home",
    domain="Clipart",
    feature_model="clip-vit-base-patch16",
    prompt="a photo of a {CLASS}.",
)
```

Feature cache loading uses `feature_cache_path` from the dataset path config, the
`DABENCH_FEATURE_CACHE_PATH` environment variable, or an explicit `feature_cache_path`
argument. The current cache layout is:

```text
{feature_cache_path}/{dataset}/{domain-or-split}/image_features/{feature_model}/
{feature_cache_path}/{dataset}/{domain-or-split}/text_features/{feature_model}-{prompt}.pt
```

The current cache directory covers `office-31`, `office-home`, `domainnet`,
`minidomainnet`, `camelyon17`, and `iwildcam`. `minidomainnet` does not need a
separate cache directory: `dabench` reads the full `domainnet` cache, then
filters image feature rows with the configured mini split files. This filtering
uses an `image_path` column when present; for the existing DomainNet cache it
uses the source image paths retained in `dataset_info.json`. Labels and loaded
text features are remapped to the prepared DomainNet class order exposed by
`load_view("minidomainnet", ...)`.

For WILDS datasets, `camelyon17` and `iwildcam` do not expose adaptation domains
through `load_view`; their public axis is the split. Cached feature loading uses
that split name in the `{domain-or-split}` path component.

`list_feature_caches(..., include_stats=True)` keeps the legacy `domain` field
for compatibility and also exposes `axis`, `cache_key`, and `split` so callers
can distinguish domain-based caches from split-based WILDS caches.

## Dataset rules

The supported benchmark datasets do not share one universal split rule:

| Dataset | View rule |
| --- | --- |
| Office-31 | domain only, split is ignored |
| Office-Home | domain only, single `train` split is filled internally |
| DomainNet | domain + explicit `train` / `test` split |
| miniDomainNet | same public API as DomainNet, but filtered from prepared DomainNet by split files |
| VisDA-2017 | `synthetic -> train`, `real -> validation` |
| Camelyon17 | explicit split only; no domain argument required |
| iWildCam | explicit split only; train rows receive `label`, `category_id`, and `category_name` from configured WILDS metadata |

These rules are applied by `load_view`, not by user code.

## WILDS Metadata

`camelyon17` and `iwildcam` use local dataset paths from
`src/dabench/config/paths.json`, or an overridden dabench path config. iWildCam's
prepared Hugging Face directory only stores images, so `dabench` joins the
official WILDS JSON metadata configured as `train_annotations_path` and
`test_information_path`.

This is why iWildCam has category counts in the dataset documentation even when
the prepared Arrow dataset itself does not expose a label column. `dabench`
recovers train labels and class names from the configured metadata files:

```python
from dabench.data import get_class_names, load_view

train = load_view("iwildcam", split="train", decode=False)
class_names = get_class_names("iwildcam")
```

The iWildCam test split has image and location metadata but no labels in the
public metadata file.

## WILDS Feature Extraction

Use the WILDS CLIP extraction script when you need to generate local cached
features through the dabench data loader:

```bash
PYTHONPATH=src python scripts/extract_wilds_clip_features.py \
  --datasets camelyon17 iwildcam \
  --models clip-vit-base-patch16 clip-vit-l-14-datacomp.xl-s13b-b90k
```

The script calls `load_view(...)` for every split, uses `dabench.models` to load
local pretrained CLIP checkpoints, writes normalized image features under the
feature cache layout above, and writes text features from the dataset class names.

The generated WILDS caches currently include:

| Dataset | Split | `clip-vit-base-patch16` | `clip-vit-l-14-datacomp.xl-s13b-b90k` | Decode errors |
| --- | ---: | ---: | ---: | ---: |
| `camelyon17` | `id_train` | 302436 x 512 | 302436 x 768 | 0 |
| `camelyon17` | `id_val` | 33560 x 512 | 33560 x 768 | 0 |
| `camelyon17` | `unlabeled_train` | 600030 x 512 | 600030 x 768 | 0 |
| `camelyon17` | `ood_val` | 34904 x 512 | 34904 x 768 | 0 |
| `camelyon17` | `ood_test` | 85054 x 512 | 85054 x 768 | 0 |
| `iwildcam` | `train` | 217940 x 512 | 217940 x 768 | 19 |
| `iwildcam` | `test` | 62870 x 512 | 62870 x 768 | 24 |

iWildCam decode failures are skipped during feature extraction and recorded as
`decode_errors.json` under the corresponding `image_features/{feature_model}/`
directory. Use `load_feature_decode_errors(...)` to inspect those records.

## miniDomainNet

`minidomainnet` is not downloaded as a separate raw dataset. It is derived from prepared DomainNet data:

- `path` should point to prepared DomainNet data
- `split_dir` should point to the directory containing `clipart_train.txt`, `clipart_test.txt`, and the other mini split files
- if `split_dir` is omitted, `dabench` falls back to `splits_mini/` under the prepared dataset root
- feature loading reuses `{feature_cache_path}/domainnet/{domain}/...` and applies the same split files to the cached rows

This keeps the external API simple:

```python
from dabench.data import load_view

train = load_view("minidomainnet", domain="real", split="train", format="hf")
test = load_view("minidomainnet", domain="real", split="test", format="hf")
train_features = load_view(
    "minidomainnet",
    domain="real",
    split="train",
    format="feature",
    feature_model="clip-vit-base-patch16",
)
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

Cached feature suites are explicit:

```python
suite = build_suites(
    datasets="office-31",
    setting="uda",
    format="feature",
    dataset_defaults={"feature_model": "clip-vit-base-patch16"},
)[0]
```

`build_suites(...)` builds the dataset suite, `load_suite_item(...)` executes one item, and the setting loader decides which split to use internally.
