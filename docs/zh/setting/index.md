# 场景

`dabench.setting` 放的是任务级入口。当前公开的 setting 有：

- `uda`：无监督领域自适应
- `dg`：领域泛化
- `unida`：通用领域自适应

这些 setting 都建立在底层 dataset view 之上。用户只需要传 benchmark 级别的任务身份，例如 source/target domain 或 UniDA task code，setting 层会负责内部的 split 路由和 loader 组织。

## UDA

UDA 的入口是：

```python
from dabench.setting import load_uda

train_loader, val_loader, test_loader = load_uda(
    dataset="domainnet",
    source_domain="clipart",
    target_domain="real",
    source_train_batch_size=32,
)
```

UDA 会返回三个 loader：

- source / target 配对训练 loader
- target validation loader
- target test loader

当前数据集路由规则：

| 数据集 | Source train | Target train | Val/Test |
| --- | --- | --- | --- |
| `office-31` | source domain，忽略 split | target domain，忽略 split | target domain，忽略 split |
| `office-home` | source domain，忽略 split | target domain，忽略 split | target domain，忽略 split |
| `domainnet` | source `train` | target `train` | target `test` |
| `minidomainnet` | source `train` | target `train` | target `test` |
| `visda-2017` | `synthetic/train` | `real/validation` | `real/validation` |

## DG

DG 的入口是：

```python
from dabench.setting import load_dg

train_loader, val_loader, test_loader = load_dg(
    dataset="domainnet",
    source_domains=("clipart", "painting", "sketch"),
    target_domain="real",
    source_train_batch_size=32,
)
```

DG 会把多个 source domains 合并成一个训练集，并保持 target domain 在训练中不可见：

- source domains 拼接后作为训练集
- validation / test 都在 target domain 上评估
- split 规则复用对应数据集的 UDA 路由

## UniDA

UniDA 的入口是：

```python
from dabench.setting import load_unida

payload = load_unida(
    dataset="office-home",
    task="AR",
    shared=10,
    source_private=5,
    target_private=50,
    source_train_batch_size=32,
)
```

UniDA 目前只支持 `format="hf"`，返回值也不是 3-tuple，而是一个字典：

- `source_train_dataset`
- `target_train_dataset`
- `test_dataset`
- `source_train_loader`
- `target_train_loader`
- `test_loader`
- `metadata`

其中 `metadata` 会包含：

- 解析后的 source / target task 域
- class split 定义
- label 到 classname 的映射
- 各 split 对应的 classnames
- source / target 侧的标签计数摘要

### 当前内置 UniDA 任务族

| 数据集 | 支持的 task code | suite 内置 scenario |
| --- | --- | --- |
| `office-31` | 两字符 domain code，例如 `aw`、`da` | `cda`、`pda`、`oda`、`opda` |
| `office-home` | 两字符 domain code，例如 `AR`、`CP` | `cda`、`pda`、`oda`、`opda` |
| `domainnet` | 仅 `painting/real/sketch` 之间的任务 | 仅 `opda` |
| `visda-2017` | `SR` | 仅 `opda` |

## 关于 split 路由的统一说明

三种 setting 下，用户都不需要手工把 benchmark 任务翻译成底层 split 名称。这部分逻辑统一放在 `dabench.setting` 里，并按数据集实现。

## 套件辅助

如果要做批量实验配置，可以直接用 suite 层：

```python
from dabench.suite import build_suites, load_suite_item

suite = build_suites(datasets="domainnet", setting="uda", format="hf")[0]
item = suite["settings"][0]
train_loader, val_loader, test_loader = load_suite_item(item)
```
