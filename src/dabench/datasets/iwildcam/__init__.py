"""iWildCam dataset utilities."""

from dabench.datasets.iwildcam.download import download_dataset
from dabench.datasets.iwildcam.load import inspect_dataset, load_dataset, load_hf_dataset, load_uda

__all__ = ["download_dataset", "inspect_dataset", "load_dataset", "load_hf_dataset", "load_uda"]
