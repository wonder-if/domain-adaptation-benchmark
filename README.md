# dabench

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://wonder-if.github.io/domain-adaptation-benchmark/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

Lightweight dataset utilities for domain adaptation benchmarks.

`dabench` focuses on dataset download, local inspection, and experiment-friendly loading. It keeps data preparation explicit and returns Hugging Face `datasets.Dataset` objects where possible, so the same datasets can be used with PyTorch, `transformers`, or custom research code.

![dabench dataset overview](docs/assets/dataset_matrix_overview.png)

## Getting Started

```bash
git clone <your-repo-url>
cd domain-adaptation-benchmark
pip install -e .[data]
```

Load a prepared local dataset:

```python
from dabench.datasets import list_domains, load_hf_dataset

print(list_domains("office-31"))

dataset = load_hf_dataset(
    "office-31",
    path="/path/to/office31",
    domains=["amazon"],
)
```

Download data explicitly when needed:

```python
from dabench.datasets import download_dataset

download_dataset(
    "office-home",
    dest="/path/to/office_home_prepared",
    source="mirror",
    proxy="disable",
)
```

For Hugging Face-backed datasets, `source="mirror"` uses `https://hf-mirror.com`; `source="hf"` uses the official Hugging Face endpoint.

## Supported Datasets

| Dataset | Source | Domains / splits |
| --- | --- | --- |
| DomainNet | Hugging Face | `clipart`, `infograph`, `painting`, `quickdraw`, `real`, `sketch` |
| Office-Home | Hugging Face | `Art`, `Clipart`, `Product`, `Real World` |
| Office-31 | ModelScope | `amazon`, `dslr`, `webcam` |
| Camelyon17 | Hugging Face | `id_train`, `id_val`, `unlabeled_train`, `ood_val`, `ood_test` |
| VisDA-2017 | Official GitHub / URLs | `train`, `validation`, `test` |
| iWildCam | Hugging Face | domains are camera traps / `location` ids; local HF metadata: 325 train locations, 91 test locations |

## Documentation

Full documentation is published as a MkDocs site:

<https://wonder-if.github.io/domain-adaptation-benchmark/>

Local preview:

```bash
pip install -e .[docs]
mkdocs serve -a 0.0.0.0:10004
```

If the Pages site is not live yet, enable GitHub Pages with `Settings -> Pages -> Source -> GitHub Actions`.
