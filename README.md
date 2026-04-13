# Domain Adaptation Benchmark

`dabench` is a small, installable benchmark-side package for domain adaptation research.

Developer-facing usage documentation lives under `docs/`. The current docs focus on dataset loading and are structured so they can be served with MkDocs later.

Right now it focuses on dataset download/loading utilities and the surrounding workflow:

- explicit dataset downloads
- local artifact inspection
- loading prepared datasets through `datasets`
- a package layout that is ready for `pip install -e .`

## Usage

```bash
git clone <your-repo-url>
cd domain-adaptation-benchmark
pip install -e .
```

Download a dataset explicitly when needed:

```bash
dabench download office-31 --dest /path/to/office31 --proxy disable
```

Inspect local artifacts:

```bash
dabench inspect office-31 --path /path/to/office31
```

The iWildCam shell wrapper is still available:

```bash
bash scripts/download_iwildcam.sh --help
```

## Python

```python
from dabench.datasets import inspect_dataset, load_hf_dataset

info = inspect_dataset("office-31", path="/path/to/office31")
ds = load_hf_dataset("office-31", path="/path/to/office31", domains=["A"])
```

See `docs/` for dataset loading details.

If you want the loading helpers, install with:

```bash
pip install -e .[data]
```
