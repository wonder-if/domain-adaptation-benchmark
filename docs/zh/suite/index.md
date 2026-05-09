# 套件

`dabench.suite` 是放在 setting 之上的轻量配置层。它根据数据集默认值、setting 默认值以及数据集 / 域对，批量生成实验配置。

## 概览

```text
build_suites(datasets=..., setting=..., format=...)
    -> suite descriptor
        -> settings
            -> load_suite_item(item)
                -> load_uda(...) / load_dg(...) / load_unida(...)
```

## 构建套件

使用统一入口：

```python
from dabench.suite import build_suites

suite = build_suites(datasets="domainnet", setting="uda", format="hf")[0]
print(suite["suite_id"])
print(len(suite["settings"]))
```

`datasets` 可以是单个数据集名字，也可以是列表。当前内置的套件包括：

- `office-31`
- `office-home`
- `domainnet`
- `minidomainnet`
- `visda-2017`

当前内置的 setting 包括：

- `uda`
- `dg`
- `unida`

## 执行单个 item

`load_suite_item(...)` 会去掉 human-readable 的 `name` 字段，然后按 `item["setting"]` 转交给 setting 层：

```python
from dabench.suite import build_suites, load_suite_item

suite = build_suites(datasets="domainnet", setting="uda", format="hf")[0]
item = suite["settings"][0]
train_loader, val_loader, test_loader = load_suite_item(item)
```

如果 `setting="unida"`，返回值会是 `load_unida(...)` 的结果字典，而不是 3 个 loader 的 tuple。

## 内置套件形状

suite builder 会显式、确定性地生成任务：

| Setting | 数据集族 | 内置 item 形状 |
| --- | --- | --- |
| `uda` | 所有已支持数据集 | 所有有序 source-to-target 迁移对 |
| `dg` | 所有已支持数据集 | 每个 target domain 一个 item，其余 domain 全部作为 source domains |
| `unida` | `office-31`、`office-home` | 所有有序 source-target 对，与内置 scenario 预设交叉 |
| `unida` | `domainnet` | 只覆盖 `painting`、`real`、`sketch` 之间的任务，内置 `opda` |
| `unida` | `visda-2017` | 只内置一个 `SR` 任务，scenario 为 `opda` |

具体数量示例：

- `office-home` UDA：12 个迁移对
- `minidomainnet` UDA：12 个迁移对
- `office-home` DG：4 个 target-domain item
- `domainnet` DG：6 个 target-domain item
- `office-home` UniDA：domain pair 和 `cda/pda/oda/opda` 的笛卡尔组合

## 查看 suite metadata

每个 suite descriptor 都包含：

- `suite_id`
- `name`
- `setting`
- `settings`
- `metadata.domains`

可以直接检查：

```python
from dabench.suite import build_suites

suite = build_suites(datasets="minidomainnet", setting="dg", format="hf")[0]
print(suite["suite_id"])
print(suite["metadata"]["domains"])
print(suite["settings"][0]["name"])
```

## 典型用法

suite 层适合做可重复的 benchmark 实验：

- 选择数据集
- 选择 `setting="uda"`、`setting="dg"` 或 `setting="unida"`
- 选择 `format="hf"` 或 `format="torch"`
- 遍历 `suite["settings"]`
