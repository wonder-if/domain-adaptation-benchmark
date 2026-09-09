# dabench

`dabench` 是一个面向领域自适应研究的轻量数据集工具包。

![dabench dataset overview](../assets/dataset_matrix_overview.png)

## 概览

项目当前按下面的层次组织：

```text
storage  ->  data  ->  setting  ->  suite
manifest     load_view  load_uda   build_suites
prepare      load_hf_dataset       load_suite_item

models
load_model
```

- `storage` 负责根据 manifest 准备本地数据目录。
- `data` 负责从本地配置加载一个具体的图像视图或缓存特征视图。
- `models` 负责从路径配置加载本地预训练 backbone，不包含自适应算法。
- `setting` 负责组装任务级 loader，例如 UDA。
- `suite` 负责在 setting 之上批量生成实验配置。
- `results` 负责记录单次实验并聚合 benchmark 表格。

从[数据集](data/index.md)和[模型](model/index.md)开始，再看[场景](setting/index.md)、[套件](suite/index.md)，最后看[结果](results/index.md)。
