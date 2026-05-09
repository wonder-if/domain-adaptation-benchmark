# Settings

`dabench.setting` contains task-level entrypoints. The current public settings are:

- `uda`: unsupervised domain adaptation
- `dg`: domain generalization
- `unida`: universal domain adaptation

Each setting wraps lower-level dataset views and exposes a benchmark-shaped interface. The user passes explicit task identity such as source/target domains or UniDA task code, while the setting layer handles dataset-specific split routing internally.

## UDA

The UDA entrypoint is:

```python
from dabench.setting import load_uda

train_loader, val_loader, test_loader = load_uda(
    dataset="domainnet",
    source_domain="clipart",
    target_domain="real",
    source_train_batch_size=32,
)
```

UDA returns three loaders:

- paired source/target training loader
- validation loader on target
- test loader on target

Current dataset routing:

| Dataset | Source train | Target train | Val/Test |
| --- | --- | --- | --- |
| `office-31` | source domain, split ignored | target domain, split ignored | target domain, split ignored |
| `office-home` | source domain, split ignored | target domain, split ignored | target domain, split ignored |
| `domainnet` | source `train` | target `train` | target `test` |
| `minidomainnet` | source `train` | target `train` | target `test` |
| `visda-2017` | `synthetic/train` | `real/validation` | `real/validation` |

## DG

The DG entrypoint is:

```text
load_dg(dataset, source_domains, target_domain, ...)
```

Example:

```python
from dabench.setting import load_dg

train_loader, val_loader, test_loader = load_dg(
    dataset="domainnet",
    source_domains=("clipart", "painting", "sketch"),
    target_domain="real",
    source_train_batch_size=32,
)
```

DG keeps the target domain fully unseen during training:

- source domains are concatenated into one training dataset
- validation and test are both evaluated on the target domain
- split routing is reused from the corresponding UDA dataset rules

## UniDA

The UniDA entrypoint is:

```python
from dabench.setting import load_unida

payload = load_unida(
    dataset="office-home",
    task="AR",
    shared=10,
    source_private=5,
    target_private=50,
    source_train_batch_size=32,
)
```

UniDA currently supports `format="hf"` only and returns a dictionary instead of a 3-tuple:

- `source_train_dataset`
- `target_train_dataset`
- `test_dataset`
- `source_train_loader`
- `target_train_loader`
- `test_loader`
- `metadata`

The `metadata` field includes:

- resolved source/target task domains
- class split definition
- label-to-classname mapping
- per-split classnames
- source/target label count summaries

### Supported UniDA task families

| Dataset | Supported task code space | Scenarios in built-in suite |
| --- | --- | --- |
| `office-31` | 2-letter domain code such as `aw`, `da` | `cda`, `pda`, `oda`, `opda` |
| `office-home` | 2-letter domain code such as `AR`, `CP` | `cda`, `pda`, `oda`, `opda` |
| `domainnet` | only `painting/real/sketch` task pairs | `opda` only |
| `visda-2017` | `SR` | `opda` only |

## Common note on split routing

In all settings, users do not need to manually translate benchmark tasks into low-level split names. `dabench.setting` keeps that logic centralized and dataset-specific.

## Suite helpers

For batchable experiment configuration, use the suite layer:

```python
from dabench.suite import build_suites, load_suite_item

suite = build_suites(datasets="domainnet", setting="uda", format="hf")[0]
item = suite["settings"][0]
train_loader, val_loader, test_loader = load_suite_item(item)
```
