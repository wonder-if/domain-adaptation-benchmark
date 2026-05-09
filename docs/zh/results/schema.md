# 结果 Schema

当前有两个结果对象：

- `RunRecord`：研究项目直接写出的标准 run 记录
- `BenchmarkResultView`：由多个 `RunRecord` 聚合出的展示视图

## RunRecord

核心身份字段：

- `record_type`：固定为 `run_record`
- `schema_version`
- `dataset`
- `setting`
- `method`
- `seed`
- `run_id`
- `status`

常见元数据字段：

- `backbone`
- `algorithm_name`
- `source_domain`
- `target_domain`
- `output_dir`
- `start_time`
- `end_time`
- `selected_checkpoint`
- `selection_metric`
- `selection_mode`
- `config`
- `extra`

结果字段：

- `final_metrics`
- `best_metrics`
- `final_class_metrics`
- `best_class_metrics`
- `eval_history`

推荐目录结构：

```text
<experiment_root>/records/<dataset>/<setting>/<method>/<run_id>.json
```

English 页面里给了完整 JSON 示例，这里不重复粘贴大段示例。实现时按同一字段集即可。
