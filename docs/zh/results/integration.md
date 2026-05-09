# 接入方式

面向真正训练算法的研究项目，推荐流程是：

1. 一个 run 初始化一个 recorder
2. 每次验证或测试时显式记录指标
3. 训练结束时 finalize，写出一个 `RunRecord`

## 最小示例

```python
from dabench.results import ExperimentRecorder

recorder = ExperimentRecorder(
    dataset="office-home",
    setting="uda",
    method="DAMP",
    backbone="RN50",
    source_domain="Art",
    target_domain="Clipart",
    seed=1,
    output_dir="experiments/office_home/art_to_clipart/seed_1",
    records_root="experiments/records",
    config={"lr": 0.003, "batch_size": 32},
)

for epoch in range(1, 31):
    recorder.log_eval(
        step=epoch,
        split="val",
        metrics=evaluate_on_val(...),
        checkpoint=f"epoch-{epoch}",
    )

recorder.log_eval(
    step=30,
    split="test",
    metrics=evaluate_on_test(...),
    checkpoint="epoch-30",
)

recorder.finalize()
```

## 聚合成表格

```python
from dabench.results import build_uda_result_view, collect_run_records, render_uda_markdown_table

runs = collect_run_records("experiments/records", dataset="office-home", setting="uda", method="DAMP")
view = build_uda_result_view(runs, metric_source="final", reduction="mean")
print(render_uda_markdown_table(view))
```

## 调用方必须提供的内容

- `dataset`
- `setting`
- `method`
- `seed`
- UDA 下的 `source_domain` / `target_domain`
- 实际评测得到的指标

旧项目如果已经有自己的日志格式，可以单独写转换脚本，把旧结果转换成同样的 `RunRecord` JSON。
