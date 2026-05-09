# 数据集

`dabench` 将数据访问分成两层：

- `load_view(...)` 是用户层入口。它会从本地配置解析路径，并按照数据集规则处理 split / domain。
- `load_hf_dataset(path)` 是更底层的原语，用于已经准备好的 Hugging Face 数据目录。

## 当前支持的数据集

| 数据集 | 标准 id | 域 / 任务轴 | 说明 |
| --- | --- | --- | --- |
| Office-31 | `office-31` | `amazon`, `dslr`, `webcam` | 图像目录式数据，split 会被忽略 |
| Office-Home | `office-home` | `Art`, `Clipart`, `Product`, `Real World` | 单一 prepared split，按 domain 访问 |
| DomainNet | `domainnet` | `clipart`, `infograph`, `painting`, `quickdraw`, `real`, `sketch` | 显式 `train` / `test` split |
| miniDomainNet | `minidomainnet` | `clipart`, `painting`, `real`, `sketch` | 基于 prepared DomainNet 和 `splits_mini/*.txt` 过滤得到 |
| VisDA-2017 | `visda-2017` | `synthetic`, `real` | 内部会自动路由到 `train` / `validation` |

## 常用方式

实验和脚本里优先使用 `load_view`：

```python
from dabench.data import load_view

domainnet = load_view("domainnet", domain="clipart", split="train", format="hf")
office_home = load_view("office-home", domain="Art", format="hf")
office31 = load_view("office-31", domain="amazon", format="torch")
visda = load_view("visda-2017", domain="synthetic", split="train", format="hf")
minidomainnet = load_view("minidomainnet", domain="clipart", split="train", format="hf")
```

返回值会根据 `format` 变成原生 `datasets.Dataset` 或 PyTorch 风格封装。

## 数据集规则

当前 benchmark 数据集的加载规则并不完全相同：

| 数据集 | 规则 |
| --- | --- |
| Office-31 | 只关心 domain，split 会被忽略 |
| Office-Home | 只关心 domain，内部自动使用单一 `train` split |
| DomainNet | 需要 domain + 显式 `train` / `test` split |
| miniDomainNet | API 和 DomainNet 一致，但底层是从 prepared DomainNet 过滤出来的 |
| VisDA-2017 | `synthetic -> train`，`real -> validation` |

这些规则由 `load_view` 处理，不需要用户手工判断。

## miniDomainNet

`minidomainnet` 不是单独下载的一套原始数据，而是建立在 prepared DomainNet 之上：

- `path` 应该指向 prepared DomainNet
- `split_dir` 应该指向包含 `clipart_train.txt`、`clipart_test.txt` 等 mini split 文件的目录
- 如果不显式配置 `split_dir`，`dabench` 会尝试使用数据根目录下的 `splits_mini/`

对外使用方式仍然保持简单：

```python
from dabench.data import load_view

train = load_view("minidomainnet", domain="real", split="train", format="hf")
test = load_view("minidomainnet", domain="real", split="test", format="hf")
```

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

`minidomainnet` 目前没有独立 downloader。应先准备 DomainNet，再把 `minidomainnet` 指向该 prepared 数据和 mini split 文件。

## 套件层

做 benchmark 实验时，优先使用套件层：

```python
from dabench.suite import build_suites, load_suite_item

suite = build_suites(datasets="domainnet", setting="uda", format="hf")[0]
item = suite["settings"][0]
train_loader, val_loader, test_loader = load_suite_item(item)
```

`build_suites(...)` 用于构造套件，`load_suite_item(...)` 用于执行单个 item，内部由 setting 层决定 split。
