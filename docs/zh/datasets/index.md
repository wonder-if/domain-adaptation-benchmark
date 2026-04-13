# 数据集

`dabench` 提供面向领域自适应数据集的轻量加载接口。加载函数接收本地数据路径，并尽量返回原生 🤗 Hugging Face [`datasets.Dataset`](https://huggingface.co/docs/datasets)，因此可以继续配合 `map`、`filter`、`set_transform`、`transformers.Trainer` 或自定义 PyTorch 代码使用。

## 支持的数据集

![dabench 支持的数据集概览](../../assets/dataset_matrix_overview.png)

| 数据集 | 主页 | 域 / 划分 |
| --- | --- | --- |
| DomainNet | 🤗 [wltjr1007/DomainNet](https://huggingface.co/datasets/wltjr1007/DomainNet) | `clipart`, `infograph`, `painting`, `quickdraw`, `real`, `sketch` |
| Office-Home | 🤗 [flwrlabs/office-home](https://huggingface.co/datasets/flwrlabs/office-home) | `Art`, `Clipart`, `Product`, `Real World` |
| Office-31 | <span class="twemoji">Ⓜ️</span> [OmniData/Office-31](https://www.modelscope.cn/datasets/OmniData/Office-31) | `amazon`, `dslr`, `webcam` |
| Camelyon17 | 🤗 [jxie/camelyon17](https://huggingface.co/datasets/jxie/camelyon17) | `id_train`, `id_val`, `unlabeled_train`, `ood_val`, `ood_test` |
| VisDA-2017 | <span class="twemoji">🐙</span> [taskcv-2017-public](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) | `train`, `validation`, `test` |
| iWildCam | 🤗 [anngrosha/iWildCam2020](https://huggingface.co/datasets/anngrosha/iWildCam2020) | 已支持图像加载；任务相关的 split/domain 处理还未定稿 |

## 加载数据集

```python
from dabench.datasets import load_hf_dataset

dataset = load_hf_dataset(
    "office-31",
    path="/path/to/office31",
    domains=["A"],
)
```

返回值是 `datasets.Dataset`。对于领域自适应数据集，如果对应信息存在，加载器会尽量使用下面这组常规列名：

```text
image
label
domain
image_path
```

不同数据集的可用列不完全相同。例如，当前 prepared iWildCam 数据只提供 `image`。

## 下载

加载函数只读取本地数据。训练前应显式使用 `download_dataset` 或命令行准备数据：

对于 Hugging Face 来源的数据集，`source="mirror"` 会通过 `HF_ENDPOINT` 使用 `https://hf-mirror.com`，`source="hf"` 则使用官方 `https://huggingface.co`。默认使用 `mirror`。

```python
from dabench.datasets import download_dataset

download_dataset(
    "office-home",
    dest="/path/to/office_home_prepared",
    source="mirror",
    proxy="disable",
)
```

Office-31 使用 ModelScope Git LFS 仓库，并会整理成本地图像目录：

```bash
dabench download office-31 --dest /path/to/office31 --proxy disable
```

所有数据集都遵循同样的边界：先显式下载，再在实验中从 prepared 本地路径加载。

## 域和划分

域过滤支持完整名称，也支持常用缩写：

```python
office31 = load_hf_dataset("office-31", path="/path/to/office31", domains=["A"])
office_home = load_hf_dataset("office-home", path="/path/to/office_home", domains=["Ar"])
domainnet = load_hf_dataset("domainnet", path="/path/to/domainnet", split="train", domains=["c"])
```

带固定划分的数据集使用其原始 split 名称：

```python
camelyon17 = load_hf_dataset(
    "camelyon17",
    path="/path/to/camelyon17",
    split="id_train",
)

visda = load_hf_dataset(
    "visda-2017",
    path="/path/to/visda2017",
    split="validation",
)
```

## 图像预处理

可以用 `set_transform` 在读取样本时附加模型相关预处理，而不改写磁盘上的数据：

```python
dataset = load_hf_dataset("office-31", path="/path/to/office31", domains=["A"])

def transform(example):
    inputs = processor(example["image"], return_tensors="pt")
    example["pixel_values"] = inputs["pixel_values"][0]
    return example

dataset.set_transform(transform)
```

如果训练循环需要 PyTorch 风格的 dataset，可以显式包装：

```python
from dabench.datasets import build_torch_dataset, get_train_transform

torch_dataset = build_torch_dataset(
    dataset,
    transform=get_train_transform(),
    domain_column="domain",
    path_column="image_path",
)
```

## Transformers Trainer

`transformers.Trainer` 可以直接接收原生 dataset：

```python
from transformers import Trainer, TrainingArguments
from dabench.datasets import load_hf_dataset

train_dataset = load_hf_dataset("office-31", path="/path/to/office31", domains=["A"])
train_dataset.set_transform(transform)

args = TrainingArguments(
    output_dir="./outputs",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)
```

视觉数据集通常更适合显式设置 `remove_unused_columns=False`，尤其当预处理发生在 transform 或 collator 中时。否则 Trainer 可能在数据进入输入管线前移除 `image`、`domain`、`image_path` 等字段。

## 本地数据布局

加载器只读取本地路径，不会隐式发起下载。先准备或下载数据集，再把 prepared 路径传给 `load_hf_dataset`。

```python
load_hf_dataset("domainnet", path="/path/to/domainnet_prepared", split="train")
load_hf_dataset("office-home", path="/path/to/office_home_prepared", split="train")
load_hf_dataset("office-31", path="/path/to/office31", domains=["A"])
load_hf_dataset("camelyon17", path="/path/to/camelyon17_prepared", split="id_train")
load_hf_dataset("visda-2017", path="/path/to/visda2017_official", split="train")
load_hf_dataset("iwildcam", path="/path/to/iwildcam_prepared", split="train")
```

Hugging Face prepared dataset 的路径下应包含 `dataset_info.json` 和 Arrow 分片。Office-31 使用包含 `amazon/`、`dslr/`、`webcam/` 的本地图像目录。VisDA-2017 使用官方压缩包解压后的 `data/` 目录。
