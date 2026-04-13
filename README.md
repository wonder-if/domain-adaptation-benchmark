# Domain Adaptation Benchmark

`dabench` is a lightweight Python package for domain adaptation benchmark workflows.
It focuses on the data side: downloading, inspecting, and loading datasets in a way
that stays friendly to `datasets`, PyTorch, and `transformers`.

The package is intentionally small. It is not a training framework, model zoo, or
experiment manager.

## Dataset Overview

![dabench dataset overview](docs/assets/dataset_matrix_overview.png)

Current dataset support includes:

| Dataset | Source | Main domains / splits |
| --- | --- | --- |
| DomainNet | Hugging Face | `clipart`, `infograph`, `painting`, `quickdraw`, `real`, `sketch` |
| Office-Home | Hugging Face | `Art`, `Clipart`, `Product`, `Real World` |
| Office-31 | ModelScope | `amazon`, `dslr`, `webcam` |
| Camelyon17 | Hugging Face | `id_train`, `id_val`, `unlabeled_train`, `ood_val`, `ood_test` |
| VisDA-2017 | Official GitHub / URLs | `train`, `validation`, `test` |
| iWildCam | Hugging Face | prepared image dataset loading |

## Install

For local development:

```bash
git clone <your-repo-url>
cd domain-adaptation-benchmark
pip install -e .[data]
```

The base package has no heavy runtime dependencies. The `data` extra installs the
dataset-loading dependencies used by the current loaders.

## Load Data

The main Python entrypoint is `load_hf_dataset`. Loaders read local prepared data
and return a native Hugging Face `datasets.Dataset` whenever possible.

```python
from dabench.datasets import load_hf_dataset, list_domains

print(list_domains("office-31"))

dataset = load_hf_dataset(
    "office-31",
    path="/path/to/office31",
    domains=["A"],
)

print(dataset)
```

This keeps the data easy to adapt for different research stacks:

```python
dataset.set_transform(transform)
```

or:

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="./outputs",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=data_collator,
)
```

## Download Data

Downloads are explicit. Loading functions do not silently download data.
For Hugging Face datasets, `dabench` supports both the official Hub and
`hf-mirror.com`:

| `source` | Endpoint |
| --- | --- |
| `mirror` | `https://hf-mirror.com` |
| `hf` | `https://huggingface.co` |

The default is `source="mirror"` for the HF-backed download helpers, and
`proxy="disable"` can be used to clear proxy environment variables during the
download.

```python
from dabench.datasets import download_dataset

download_dataset(
    "office-home",
    dest="/path/to/office_home_prepared",
    source="mirror",
    proxy="disable",
)
```

CLI examples:

```bash
dabench download office-31 --dest /path/to/office31 --proxy disable
dabench inspect office-31 --path /path/to/office31
```

The old iWildCam shell entrypoint is kept as a thin compatibility wrapper:

```bash
bash scripts/download_iwildcam.sh --dest /path/to/iwildcam_prepared --source mirror --proxy disable
```

## Documentation

The user-facing docs live in [`docs/`](docs/). They are served with MkDocs and include
Chinese / English navigation.

Preview locally:

```bash
mkdocs serve -a 0.0.0.0:10004
```

Build locally:

```bash
mkdocs build --strict
```

## GitHub Pages

This repo is already structured for MkDocs Pages deployment. The simplest manual
workflow is:

```bash
pip install mkdocs mkdocs-material
mkdocs gh-deploy
```

`mkdocs gh-deploy` builds the docs and pushes the generated site to a `gh-pages`
branch. In GitHub, open:

```text
Settings -> Pages -> Build and deployment -> Source
```

Then select:

```text
Deploy from a branch
Branch: gh-pages
Folder: / (root)
```

For this repository, the expected public URL after GitHub Pages is enabled is:

```text
https://wonder-if.github.io/domain-adaptation-benchmark/
```

If you prefer automated deployment, add a GitHub Actions workflow later to build
MkDocs on every push to `main`.
