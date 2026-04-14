# dabench

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://wonder-if.github.io/domain-adaptation-benchmark/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

Lightweight dataset utilities for domain adaptation benchmarks.

`dabench` focuses on dataset download, local inspection, and experiment-friendly loading. It keeps data preparation explicit and returns Hugging Face `datasets.Dataset` objects where possible, so the same datasets can be used with PyTorch, `transformers`, or custom research code.

![dabench dataset overview](docs/assets/dataset_matrix_overview.png)

## Getting Started

```bash
git clone https://github.com/wonder-if/domain-adaptation-benchmark.git
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

| Dataset | Homepage | Domains / splits |
| --- | --- | --- |
| DomainNet | 🤗 [wltjr1007/DomainNet](https://huggingface.co/datasets/wltjr1007/DomainNet) | `clipart`, `infograph`, `painting`, `quickdraw`, `real`, `sketch` |
| Office-Home | 🤗 [flwrlabs/office-home](https://huggingface.co/datasets/flwrlabs/office-home) | `Art`, `Clipart`, `Product`, `Real World` |
| Office-31 | Ⓜ️ [OmniData/Office-31](https://www.modelscope.cn/datasets/OmniData/Office-31) | `amazon`, `dslr`, `webcam` |
| Camelyon17 | 🤗 [jxie/camelyon17](https://huggingface.co/datasets/jxie/camelyon17) | `id_train`, `id_val`, `unlabeled_train`, `ood_val`, `ood_test` |
| VisDA-2017 | 🐙 [taskcv-2017-public](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) | `train`, `validation`, `test` |
| iWildCam | 🤗 [anngrosha/iWildCam2020](https://huggingface.co/datasets/anngrosha/iWildCam2020) | camera traps / `location` ids; 325 train, 91 test |