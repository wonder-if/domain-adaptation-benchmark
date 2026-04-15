# 任务

Task 表示 benchmark protocol 的实验单元，但不接管用户训练代码。

第一版内置 closed-set UDA suite：

```bash
dabench tasks list
dabench tasks show office31_closed_set_uda
```

Python API 可以枚举任务并加载对应 dataset view：

```python
from dabench.suite import get_suite
from dabench.task import load_task

suite = get_suite("office31_closed_set_uda")
task = suite.tasks[0]
data = load_task(task, path="/path/to/office31")

source_train = data.source_train
target_train = data.target_train
target_eval = data.target_eval
```

`load_task` 返回现有 dataset loader 产生的对象。模型、trainer、loss、optimizer 和日志仍由用户代码负责。
