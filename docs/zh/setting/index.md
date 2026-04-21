# 场景

`dabench.setting` 放的是任务级入口。当前 UDA 路径不再要求用户管理 split 细节，setting 层会针对数据集自动选择合适的 split，并返回三个 loader：train、val、test。

## UDA 流程

当前流程是：

```text
suite item -> load_suite_item -> load_uda -> load_view -> prepared local data
```

示例：

```python
from dabench.setting import load_uda

train_loader, val_loader, test_loader = load_uda(
    dataset="domainnet",
    source_domain="clipart",
    target_domain="real",
    source_train_batch_size=32,
)
```

`load_uda` 现在会在内部处理不同数据集的 split 规则：

- `office-31`：忽略 split
- `office-home`：内部自动使用单一 `train` split
- `domainnet`：source / target train 使用 `train`，validation / test 使用 `test`
- `visda-2017`：source 使用 `synthetic/train`，target 使用 `real/validation`，test 目前复用 validation

`load_uda` 只返回 loader。模型、优化器、日志和训练循环仍由用户自己负责。

## 套件辅助

如果要做批量实验配置，可以直接用 suite 层：

```python
from dabench.suite import build_suites, load_suite_item

suite = build_suites(datasets="domainnet", setting="uda", format="hf")[0]
item = suite["settings"][0]
train_loader, val_loader, test_loader = load_suite_item(item)
```
