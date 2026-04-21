# dabench

`dabench` 是一个面向领域自适应研究的轻量数据集工具包。

## 概览

项目当前按下面的层次组织：

```text
storage  ->  data  ->  setting  ->  suite
manifest     load_view  load_uda   build_suites
prepare      load_hf_dataset       load_suite_item
```

- `storage` 负责根据 manifest 准备本地数据目录。
- `data` 负责从本地路径加载一个具体的数据视图。
- `setting` 负责组装任务级 loader，例如 UDA。
- `suite` 负责在 setting 之上批量生成实验配置。

从[数据集](data/index.md)开始，再看[场景](setting/index.md)，最后看[套件](suite/index.md)。
