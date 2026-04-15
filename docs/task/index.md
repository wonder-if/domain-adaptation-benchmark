# Tasks

Tasks describe benchmark protocol units without taking over user training code.

The first built-in suites cover closed-set unsupervised domain adaptation:

```bash
dabench tasks list
dabench tasks show office31_closed_set_uda
```

Use the Python API to enumerate tasks and load dataset views:

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

`load_task` returns the existing dataset objects from the dataset loaders. User code still owns models, trainers, losses, optimizers, and logging.
