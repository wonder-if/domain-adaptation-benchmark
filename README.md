# dabench

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://wonder-if.github.io/domain-adaptation-benchmark/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

Domain adaptation dataset utilities for download, inspection, loading, and evaluation-side workflows.

`dabench` keeps domain adaptation benchmarks easy to prepare and easy to reuse across `datasets`, PyTorch, and `transformers`. It focuses on the data layer: explicit downloads, local dataset checks, Hugging Face `Dataset` loading where possible, and small helpers for UDA-style evaluation pipelines.

This library is intentionally small. It is not a training framework, a model zoo, or an experiment manager.

![dabench dataset overview](docs/assets/dataset_matrix_overview.png)

## Installation

`dabench` works with Python 3.10+.

Install for local development:

```bash
git clone <your-repo-url>
cd domain-adaptation-benchmark
pip install -e .[data]
```

The base package has no heavy runtime dependencies. The `data` extra installs the dataset loading and download dependencies used by the current loaders.

Install the documentation dependencies only when you want to preview or build the docs:

```bash
pip install -e .[docs]
```

## Quickstart

Load a prepared local dataset with the unified dataset entrypoints:

```python
from dabench.datasets import list_domains, load_hf_dataset

print(list_domains("office-31"))

dataset = load_hf_dataset(
    "office-31",
    path="/path/to/office31",
    domains=["amazon"],
)

print(dataset)
```

Use the returned dataset with your own transforms, dataloaders, or `transformers` training and evaluation code:

```python
from transformers import Trainer, TrainingArguments

dataset.set_transform(transform)

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

## Supported Datasets

Current dataset support includes:

| Dataset | Source | Main domains / splits |
| --- | --- | --- |
| DomainNet | Hugging Face | `clipart`, `infograph`, `painting`, `quickdraw`, `real`, `sketch` |
| Office-Home | Hugging Face | `Art`, `Clipart`, `Product`, `Real World` |
| Office-31 | ModelScope | `amazon`, `dslr`, `webcam` |
| Camelyon17 | Hugging Face | `id_train`, `id_val`, `unlabeled_train`, `ood_val`, `ood_test` |
| VisDA-2017 | Official GitHub / URLs | `train`, `validation`, `test` |
| iWildCam | Hugging Face | prepared image dataset loading |

## Download Data

Downloads are explicit. Loading functions do not silently download data.

For Hugging Face-backed datasets, `dabench` can use either the official Hub or `hf-mirror.com`:

| `source` | Endpoint |
| --- | --- |
| `mirror` | `https://hf-mirror.com` |
| `hf` | `https://huggingface.co` |

The default is `source="mirror"` for the HF-backed download helpers. Use `proxy="disable"` to clear proxy environment variables during the download.

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
dabench download domainnet --dest /path/to/domainnet --proxy disable
dabench download office-31 --dest /path/to/office31 --proxy disable
dabench inspect office-31 --path /path/to/office31
```

The old iWildCam shell entrypoint remains available as a thin compatibility wrapper:

```bash
bash scripts/download_iwildcam.sh --dest /path/to/iwildcam_prepared --source mirror --proxy disable
```

## Why Use dabench?

1. Local-first benchmark handling:
   - Downloads are separate from loading.
   - Prepared dataset paths are explicit.
   - Inspection commands make local dataset state easier to verify.

2. Small interoperability layer:
   - Loaders return Hugging Face `Dataset` objects where possible.
   - PyTorch-oriented helper utilities live alongside the dataset entrypoints.
   - Dataset-specific behavior stays in dataset-specific modules.

3. Practical benchmark coverage:
   - Includes common domain adaptation datasets such as DomainNet, Office-Home, Office-31, Camelyon17, VisDA-2017, and iWildCam.
   - Supports multiple download sources where useful for reproducible local preparation.

## Why Not Use dabench?

- It does not provide domain adaptation method implementations.
- It does not manage model zoos or training recipes.
- It does not hide machine-specific environment setup in repository docs.
- It is designed as benchmark support code, not a full experiment platform.

## Documentation

The user-facing docs are published as a MkDocs site:

```text
https://wonder-if.github.io/domain-adaptation-benchmark/
```

The source files live in [`docs/`](docs/), but GitHub renders those files as plain Markdown pages rather than as the full MkDocs website.

Preview locally:

```bash
mkdocs serve -a 0.0.0.0:10004
```

Build locally:

```bash
mkdocs build --strict
```

## GitHub Pages

This repo includes a GitHub Actions workflow for publishing the MkDocs site. In GitHub, open:

```text
Settings -> Pages -> Build and deployment -> Source
```

Then select:

```text
GitHub Actions
```

After that, every push to `main` builds and deploys the docs. The expected public URL is:

```text
https://wonder-if.github.io/domain-adaptation-benchmark/
```

Manual deploy:

```bash
pip install -e .[docs]
mkdocs gh-deploy
```
