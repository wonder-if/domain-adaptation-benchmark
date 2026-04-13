"""DomainNet dataset utilities."""

from dabench.datasets.domainnet.download import download_dataset
from dabench.datasets.domainnet.load import (
    inspect_dataset,
    list_classes,
    list_domains,
    load_dataset,
    load_hf_dataset,
    load_uda,
)

__all__ = ["download_dataset", "inspect_dataset", "list_classes", "list_domains", "load_dataset", "load_hf_dataset", "load_uda"]
