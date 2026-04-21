# 数据集

`dabench` 将数据访问分成两层：

- `load_view(...)` 是用户层入口。它会从本地配置解析路径，并按照数据集规则处理 split / domain。
- `load_hf_dataset(path)` 是更底层的原语，用于已经准备好的 Hugging Face 数据目录。

## 常用方式

实验和脚本里优先使用 `load_view`：

```python
from dabench.data import load_view

domainnet = load_view("domainnet", domain="clipart", split="train", format="hf")
office_home = load_view("office-home", domain="Art", format="hf")
office31 = load_view("office-31", domain="amazon", format="torch")
visda = load_view("visda-2017", domain="synthetic", split="train", format="hf")
```

返回值会根据 `format` 变成原生 `datasets.Dataset` 或 PyTorch 风格封装。

## 数据集规则

当前 suite 层支持的四个 UDA 数据集，加载规则并不相同：

| 数据集 | 规则 |
| --- | --- |
| Office-31 | 只关心 domain，split 会被忽略 |
| Office-Home | 只关心 domain，内部自动使用单一 `train` split |
| DomainNet | 需要 domain + 显式 `train` / `test` split |
| VisDA-2017 | `synthetic -> train`，`real -> validation` |

这些规则由 `load_view` 处理，不需要用户手工判断。

## 底层加载

如果你已经有准备好的 Hugging Face 数据目录，可以直接用底层加载器：

```python
from dabench.data import load_hf_dataset

dataset_dict = load_hf_dataset("/path/to/domainnet_prepared")
```

它要求目录下有 `dataset_info.json` 和 Arrow 分片。

## 下载

加载器不会隐式下载数据。训练前请显式使用 `download_dataset` 准备数据：

```python
from dabench.storage import download_dataset

download_dataset(
    "domainnet",
    dest="/path/to/domainnet_prepared",
    source="mirror",
    proxy="disable",
)
```

Office-31 使用 ModelScope Git LFS 仓库，准备后会形成本地图像目录：

```bash
dabench download office-31 --dest /path/to/office31 --proxy disable
```

## 套件层

做 UDA 实验时，优先使用套件层：

```python
from dabench.suite import build_suites, load_suite_item

suite = build_suites(datasets="domainnet", setting="uda", format="hf")[0]
item = suite["settings"][0]
train_loader, val_loader, test_loader = load_suite_item(item)
```

`build_suites(...)` 用于构造套件，`load_suite_item(...)` 用于执行单个 item，内部由 setting 层决定 split。
