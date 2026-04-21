# 套件

`dabench.suite` 是放在 setting 之上的轻量配置层。它根据数据集默认值、setting 默认值以及数据集 / 域对，批量生成实验配置。

## 概览

```text
build_suites(datasets=..., setting=..., format=...)
    -> suite descriptor
        -> settings
            -> load_suite_item(item)
                -> load_uda(...)
```

## 构建套件

使用统一入口：

```python
from dabench.suite import build_suites

suite = build_suites(datasets="domainnet", setting="uda", format="hf")[0]
print(suite["suite_id"])
print(len(suite["settings"]))
```

`datasets` 可以是单个数据集名字，也可以是列表。当前内置的 UDA 套件包括：

- `office-31`
- `office-home`
- `domainnet`
- `visda-2017`

## 执行单个 item

`load_suite_item(...)` 会去掉 human-readable 的 `name` 字段，然后转交给 setting 层：

```python
from dabench.suite import build_suites, load_suite_item

suite = build_suites(datasets="domainnet", setting="uda", format="hf")[0]
item = suite["settings"][0]
train_loader, val_loader, test_loader = load_suite_item(item)
```

## 典型用法

suite 层适合做可重复的 UDA 实验：

- 选择数据集
- 选择 `setting="uda"`
- 选择 `format="hf"` 或 `format="torch"`
- 遍历 `suite["settings"]`
