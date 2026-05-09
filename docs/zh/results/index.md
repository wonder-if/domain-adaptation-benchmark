# 结果

`dabench.results` 用来统一研究项目的实验记录格式，以及后续 benchmark 表格的生成方式。

## 这一层解决什么问题

训练代码可以由各个方法自己管理，但结果格式不应该各写各的。否则后续做多方法对比、多 seed 聚合、论文表格生成时就会变成一次性脚本。

结果层把这几件事拆开：

- 运行时记录实验
- 写出标准单次 run JSON
- 从多个 run 聚合出 benchmark 结果视图
- 渲染成最终表格

## 核心对象

```text
research project
    -> ExperimentRecorder
        -> RunRecord JSON
            -> build_uda_result_view(...)
                -> render_uda_markdown_table(...)
```

- `RunRecord`：一次真实实验运行，通常对应一个 seed 和一个迁移对。
- `BenchmarkResultView`：由多个 `RunRecord` 聚合出的展示视图。

## 当前默认约定

- 一个 run 写一个 JSON 文件
- 先支持 `uda`
- 表格形状按数据集固定：
  - `office-31`：平铺迁移对
  - `office-home`：平铺迁移对
  - `domainnet`：矩阵
  - `visda-2017`：按类表格

继续看：

- [Schema](schema.md)
- [接入方式](integration.md)
