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
| Camelyon17 | `camelyon17` | `id_train`, `id_val`, `unlabeled_train`, `ood_val`, `ood_test` | WILDS split-only 数据集 |
| iWildCam | `iwildcam` | `train`, `test`；metadata 里包含 `location` id | WILDS split-only 数据集，类别信息来自配置的 JSON metadata |

## 常用方式

实验和脚本里优先使用 `load_view`：

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

返回值会根据 `format` 变成原生 `datasets.Dataset`、PyTorch 风格封装，或缓存好的特征数据。

如果想跳过重复的 CLIP 图像推理，可以显式使用 `format="feature"` 读取缓存特征：

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

feature cache 的路径优先从数据集路径配置里的 `feature_cache_path` 读取；也可以用
`DABENCH_FEATURE_CACHE_PATH` 环境变量或 `feature_cache_path` 参数覆盖。当前约定的目录结构是：

```text
{feature_cache_path}/{dataset}/{domain-or-split}/image_features/{feature_model}/
{feature_cache_path}/{dataset}/{domain-or-split}/text_features/{feature_model}-{prompt}.pt
```

当前这个缓存目录覆盖 `office-31`、`office-home`、`domainnet`、`minidomainnet`、
`camelyon17` 和 `iwildcam`。`minidomainnet` 不需要单独的 cache 目录：
`dabench` 会读取完整的 `domainnet` cache，再根据配置的 split 文件过滤 image
feature 行。过滤时优先使用 `image_path` 列；对现有 DomainNet cache，则使用
`dataset_info.json` 中保留的源图片路径。返回的 label 和读取的 text features
会被重映射到 `load_view("minidomainnet", ...)` 使用的 prepared DomainNet 类别顺序。

对 WILDS 数据集来说，`camelyon17` 和 `iwildcam` 在 `load_view` 里不暴露自适应
domain；它们的公开轴是 split。读取缓存特征时会把 split 名称放在
`{domain-or-split}` 这一层目录里。

`list_feature_caches(..., include_stats=True)` 会保留兼容旧代码的 `domain` 字段，
同时新增 `axis`、`cache_key` 和 `split` 字段，方便区分 domain-based cache 和
WILDS 的 split-based cache。

## 数据集规则

当前 benchmark 数据集的加载规则并不完全相同：

| 数据集 | 规则 |
| --- | --- |
| Office-31 | 只关心 domain，split 会被忽略 |
| Office-Home | 只关心 domain，内部自动使用单一 `train` split |
| DomainNet | 需要 domain + 显式 `train` / `test` split |
| miniDomainNet | API 和 DomainNet 一致，但底层是从 prepared DomainNet 过滤出来的 |
| VisDA-2017 | `synthetic -> train`，`real -> validation` |
| Camelyon17 | 只需要显式 split，不需要传 domain |
| iWildCam | 只需要显式 split；train 行会从配置的 WILDS metadata 补出 `label`、`category_id` 和 `category_name` |

这些规则由 `load_view` 处理，不需要用户手工判断。

## WILDS Metadata

`camelyon17` 和 `iwildcam` 都通过 `src/dabench/config/paths.json` 里的本地路径读取，
也可以使用覆盖后的 dabench path config。iWildCam 的 prepared Hugging Face 目录本身
只有图像列，所以 `dabench` 会读取配置中的 `train_annotations_path` 和
`test_information_path`，把官方 WILDS JSON metadata 拼回数据行。

这解释了为什么 iWildCam 在文档图表里有类别数量，但当前 Arrow 数据目录里看不到类别列：
类别数量和类别名来自 WILDS metadata，而不是原始 prepared Arrow 列。使用方式如下：

```python
from dabench.data import get_class_names, load_view

train = load_view("iwildcam", split="train", decode=False)
class_names = get_class_names("iwildcam")
```

iWildCam 的 test split 有图像和 location 等 metadata，但公开 metadata 文件里没有 label。

## WILDS 特征抽取

如果要通过 dabench 数据加载器生成本地 CLIP 缓存特征，使用 WILDS 抽取脚本：

```bash
PYTHONPATH=src python scripts/extract_wilds_clip_features.py \
  --datasets camelyon17 iwildcam \
  --models clip-vit-base-patch16 clip-vit-l-14-datacomp.xl-s13b-b90k
```

脚本会对每个 split 调用 `load_view(...)`，用 `dabench.models` 加载本地预训练 CLIP，
按上面的 feature cache 布局写入归一化后的 image features，并用数据集类别名写入 text features。

当前已经生成的 WILDS cache 包括：

| 数据集 | Split | `clip-vit-base-patch16` | `clip-vit-l-14-datacomp.xl-s13b-b90k` | 解码失败 |
| --- | ---: | ---: | ---: | ---: |
| `camelyon17` | `id_train` | 302436 x 512 | 302436 x 768 | 0 |
| `camelyon17` | `id_val` | 33560 x 512 | 33560 x 768 | 0 |
| `camelyon17` | `unlabeled_train` | 600030 x 512 | 600030 x 768 | 0 |
| `camelyon17` | `ood_val` | 34904 x 512 | 34904 x 768 | 0 |
| `camelyon17` | `ood_test` | 85054 x 512 | 85054 x 768 | 0 |
| `iwildcam` | `train` | 217940 x 512 | 217940 x 768 | 19 |
| `iwildcam` | `test` | 62870 x 512 | 62870 x 768 | 24 |

iWildCam 中无法解码的 JPEG 会在特征抽取时被跳过，并记录到对应
`image_features/{feature_model}/decode_errors.json`。可以用
`load_feature_decode_errors(...)` 查看这些记录。

## miniDomainNet

`minidomainnet` 不是单独下载的一套原始数据，而是建立在 prepared DomainNet 之上：

- `path` 应该指向 prepared DomainNet
- `split_dir` 应该指向包含 `clipart_train.txt`、`clipart_test.txt` 等 mini split 文件的目录
- 如果不显式配置 `split_dir`，`dabench` 会尝试使用数据根目录下的 `splits_mini/`
- feature 加载会复用 `{feature_cache_path}/domainnet/{domain}/...`，并用同一套 split 文件过滤缓存行

对外使用方式仍然保持简单：

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

如果要用缓存特征跑 suite，需要显式选择 `format="feature"`，并通过 `dataset_defaults` 指定模型：

```python
suite = build_suites(
    datasets="office-31",
    setting="uda",
    format="feature",
    dataset_defaults={"feature_model": "clip-vit-base-patch16"},
)[0]
```

`build_suites(...)` 用于构造套件，`load_suite_item(...)` 用于执行单个 item，内部由 setting 层决定 split。
