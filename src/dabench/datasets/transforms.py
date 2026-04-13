"""Lightweight image transforms with a clipadapt-style interface."""

from __future__ import annotations

from typing import Tuple


class ResizeImage:
    def __init__(self, size: int | tuple[int, int]):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image):
        return image.resize(self.size)


def _require_torchvision():
    try:
        from torchvision import transforms as T  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required for dataset transforms. Install it in the active environment."
        ) from exc
    return T


def get_train_transform(
    resizing: str = "default",
    scale: Tuple[float, float] = (0.08, 1.0),
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    random_horizontal_flip: bool = True,
    random_color_jitter: bool = False,
    resize_size: int = 224,
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    auto_augment: str | None = None,
):
    T = _require_torchvision()
    if resizing == "default":
        transform = T.Compose([ResizeImage(256), T.RandomResizedCrop(224, scale=scale, ratio=ratio)])
        transformed_img_size = 224
    elif resizing == "cen.crop":
        transform = T.Compose([ResizeImage(256), T.CenterCrop(224)])
        transformed_img_size = 224
    elif resizing == "ran.crop":
        transform = T.Compose([ResizeImage(256), T.RandomCrop(224)])
        transformed_img_size = 224
    elif resizing == "res.":
        transform = ResizeImage(resize_size)
        transformed_img_size = resize_size
    else:
        raise NotImplementedError(resizing)

    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if auto_augment:
        try:
            from timm.data.auto_augment import auto_augment_transform, rand_augment_transform  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "timm is required when auto_augment is enabled."
            ) from exc
        aa_params = dict(
            translate_const=int(transformed_img_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in norm_mean]),
            interpolation=3,
        )
        if auto_augment.startswith("rand"):
            transforms.append(rand_augment_transform(auto_augment, aa_params))
        else:
            transforms.append(auto_augment_transform(auto_augment, aa_params))
    elif random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))

    transforms.extend([T.ToTensor(), T.Normalize(mean=norm_mean, std=norm_std)])
    return T.Compose(transforms)


def get_val_transform(
    resizing: str = "default",
    resize_size: int = 224,
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
):
    T = _require_torchvision()
    if resizing == "default":
        transform = T.Compose([ResizeImage(256), T.CenterCrop(224)])
    elif resizing == "res.":
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([transform, T.ToTensor(), T.Normalize(mean=norm_mean, std=norm_std)])
