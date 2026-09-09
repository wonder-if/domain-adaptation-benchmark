"""Dataset loading entrypoints for dabench."""

from dabench.data.dataset import (
    DomainDatasetView,
    load_feature_view,
    load_hf_dataset,
    load_view,
)
from dabench.data.feature_cache import (
    list_feature_caches,
    load_feature_decode_errors,
    load_text_features,
    resolve_feature_cache_root,
)
from dabench.data.classnames import (
    CAMELYON17_CLASS_NAMES,
    get_class_names,
)
from dabench.data.iwildcam import (
    IWILDCAM_SPLITS,
    enrich_iwildcam_split,
    get_iwildcam_class_names,
    load_iwildcam_dataset_dict,
)
from dabench.data.unida import (
    get_task,
    load_unida,
    load_unida_views,
    make_class_split,
)
from dabench.data.loader import (
    build_loader,
    default_feature_collator,
    default_collator,
    make_paired_forever_loader,
)
from dabench.data.transforms import (
    ResizeImage,
    build_test_transform,
    build_train_transform,
)

__all__ = [
    "DomainDatasetView",
    "CAMELYON17_CLASS_NAMES",
    "IWILDCAM_SPLITS",
    "ResizeImage",
    "build_loader",
    "build_test_transform",
    "build_train_transform",
    "default_feature_collator",
    "default_collator",
    "enrich_iwildcam_split",
    "get_class_names",
    "get_iwildcam_class_names",
    "list_feature_caches",
    "load_feature_decode_errors",
    "load_feature_view",
    "load_hf_dataset",
    "load_iwildcam_dataset_dict",
    "load_text_features",
    "load_unida",
    "load_unida_views",
    "load_view",
    "make_paired_forever_loader",
    "resolve_feature_cache_root",
    "get_task",
    "make_class_split",
]
