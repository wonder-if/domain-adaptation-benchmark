from __future__ import annotations

import warnings
from collections.abc import Sequence

from PIL import Image
from torch.utils.data import DataLoader, Dataset

import torch.multiprocessing

from dabench.data import get_task, make_class_split
from dabench.data.dataset import load_view

torch.multiprocessing.set_sharing_strategy("file_system")

try:
    from mmcls.datasets.pipelines import Compose as MMCompose
except ImportError:  # pragma: no cover - optional dependency
    MMCompose = None


def rgb_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def _extract_image_paths(dataset):
    if "image_path" in dataset.column_names:
        return list(dataset["image_path"])
    if "image" in dataset.column_names:
        # Avoid decoding every image during dataset construction. Paths can be
        # recovered lazily in __getitem__ when needed.
        return [""] * len(dataset)
    raise ValueError("Dataset does not expose `image_path` or `image` columns.")


class UniversalDataset(Dataset):
    """
    dabench-backed implementation of the dataset interface expected by TASC.
    """

    def __init__(self, data_root, dataset_name, domain, source=True, data_transforms=None, class_split=None):
        del data_root
        if class_split is None:
            raise ValueError("'class_split' is None!")

        self.dataset_name = dataset_name
        self.domain = domain
        self.source = source
        if data_transforms is None:
            self.transforms = []
        elif isinstance(data_transforms, Sequence) and not isinstance(data_transforms, (str, bytes)):
            self.transforms = list(data_transforms)
        else:
            self.transforms = [data_transforms]

        self.class_shared = class_split["shared"]
        self.class_source_private = class_split["source_private"]
        self.class_target_private = class_split["target_private"]
        self.num_class = len(self.class_shared + self.class_source_private + self.class_target_private)
        self.loader = rgb_loader

        base_dataset_name = {
            "office": "office-31",
            "officehome": "office-home",
            "domainnet": "domainnet",
            "visda": "visda-2017",
        }[dataset_name]
        actual_domain = "Real World" if dataset_name == "officehome" and domain == "Real" else domain
        split = None
        if dataset_name == "visda":
            split = "train" if domain == "train" else "validation"
        elif dataset_name == "domainnet":
            split = "train"

        view_domain = actual_domain
        if dataset_name == "visda":
            view_domain = "synthetic" if domain == "train" else "real"
        full_dataset = load_view(base_dataset_name, domain=view_domain, split=split, decode=True)

        allowed_labels = self.class_shared + (self.class_source_private if source else self.class_target_private)
        allowed_set = set(allowed_labels)
        indices = [index for index, label in enumerate(full_dataset["label"]) if int(label) in allowed_set]
        dataset = full_dataset.select(indices)
        self.dataset = dataset

        label_feature = full_dataset.features.get("label")
        label_names = list(getattr(label_feature, "names", []) or [])
        self.lab2cname = {str(index): name.replace("_", " ") for index, name in enumerate(label_names)}
        self.classnames_all = [self.lab2cname[str(index)] for index in range(len(label_names))]
        self.classnames_split = {
            "shared": [self.lab2cname[str(label)] for label in self.class_shared if str(label) in self.lab2cname],
            "source_private": [
                self.lab2cname[str(label)] for label in self.class_source_private if str(label) in self.lab2cname
            ],
            "source": [
                self.lab2cname[str(label)]
                for label in self.class_shared + self.class_source_private
                if str(label) in self.lab2cname
            ],
        }
        if dataset_name != "domainnet":
            self.classnames_split["target_private"] = [
                self.lab2cname[str(label)] for label in self.class_target_private if str(label) in self.lab2cname
            ]
            self.classnames_split["target"] = [
                self.lab2cname[str(label)]
                for label in self.class_shared + self.class_target_private
                if str(label) in self.lab2cname
            ]
        self.imgs = _extract_image_paths(dataset)
        self.labels = [int(label) for label in dataset["label"]]
        self.num_imgs_shared, self.num_imgs_src_pri, self.num_imgs_tar_pri = self.split_count(self.labels, class_split)

        if dataset_name == "domainnet" and ("target" not in self.classnames_split or "target_private" not in self.classnames_split):
            warnings.warn(
                "DomainNet target-private classnames are incomplete because some prepared subsets miss categories."
            )

    def load_image(self, index, path, transforms):
        mmcv_flag = MMCompose is not None and isinstance(transforms, MMCompose)
        if mmcv_flag:
            data = dict(img_info=dict(filename=path), gt_label=0, img_prefix=None)
            img = transforms(data)["img"]
        else:
            image_value = self.dataset[index]["image"]
            if isinstance(image_value, dict):
                path_in_image = image_value.get("path") or image_value.get("filename")
                if path_in_image is None:
                    raise ValueError("Image feature does not expose a usable local path.")
                img = rgb_loader(path_in_image)
            else:
                img = image_value.convert("RGB") if hasattr(image_value, "convert") else image_value
            if transforms is not None:
                img = transforms(img)
        return img

    def __getitem__(self, index):
        path = self.imgs[index]
        gt_label = self.labels[index]
        if not path:
            image_value = self.dataset[index]["image"]
            if isinstance(image_value, dict):
                path = image_value.get("path") or image_value.get("filename") or ""
            else:
                path = getattr(image_value, "filename", "") or ""
        data_info = {"filename": path, "image_ind": index}

        transform0 = self.transforms[0] if self.transforms else None
        img = self.load_image(index, path, transform0)

        if len(self.transforms) > 1:
            aug = [self.load_image(index, path, transforms) for transforms in self.transforms[1:]]
            return {"img": img, "aug": aug, "gt_label": gt_label, "data_info": data_info}

        return {"img": img, "gt_label": gt_label, "data_info": data_info}

    def __len__(self):
        return len(self.imgs)

    def split_count(self, labels, class_split):
        shared = 0
        src_pri = 0
        tar_pri = 0
        for label in labels:
            if label in self.class_shared:
                shared += 1
            elif label in self.class_source_private:
                src_pri += 1
            elif label in self.class_target_private:
                tar_pri += 1
        return [shared, src_pri, tar_pri]


class DATALoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._length = super().__len__()
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return self._length

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
