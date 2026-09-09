"""Registry and path resolution for local pretrained models."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ROOT = Path("/data/wyh/models")
MODEL_ROOT_ENV = "DABENCH_MODEL_ROOT"


@dataclass(frozen=True)
class ModelSpec:
    """Description of one locally mirrored pretrained model."""

    name: str
    relative_path: str
    loader: str
    kind: str
    architecture: str | None = None
    description: str = ""

    def path(self, model_root: str | Path | None = None) -> Path:
        return get_model_root(model_root) / self.relative_path


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="clip-vit-base-patch16",
        relative_path="modelscope/hub/openai-mirror/clip-vit-base-patch16",
        loader="transformers",
        kind="clip",
        description="OpenAI CLIP ViT-B/16 mirrored from ModelScope.",
    ),
    ModelSpec(
        name="clip-vit-base-patch32",
        relative_path="modelscope/hub/thomas/clip-vit-base-patch32",
        loader="transformers",
        kind="clip",
        description="OpenAI CLIP ViT-B/32 mirrored from ModelScope.",
    ),
    ModelSpec(
        name="clip-vit-large-patch14",
        relative_path="modelscope/hub/openai-mirror/clip-vit-large-patch14",
        loader="transformers",
        kind="clip",
        description="OpenAI CLIP ViT-L/14 mirrored from ModelScope.",
    ),
    ModelSpec(
        name="clip-vit-large-patch14-336",
        relative_path="modelscope/hub/openai-mirror/clip-vit-large-patch14-336",
        loader="transformers",
        kind="clip",
        description="OpenAI CLIP ViT-L/14@336px mirrored from ModelScope.",
    ),
    ModelSpec(
        name="clip-vit-b-16-datacomp.l-s1b-b8k",
        relative_path="modelscope/hub/laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K",
        loader="transformers",
        kind="clip",
        description="LAION DataComp CLIP ViT-B/16 in Hugging Face Transformers format.",
    ),
    ModelSpec(
        name="clip-vit-l-14-datacomp.xl-s13b-b90k",
        relative_path="modelscope/hub/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
        loader="transformers",
        kind="clip",
        description="LAION DataComp CLIP ViT-L/14 in Hugging Face Transformers format.",
    ),
    ModelSpec(
        name="chinese-clip-vit-base-patch16",
        relative_path="modelscope/hub/Xenova/chinese-clip-vit-base-patch16",
        loader="transformers",
        kind="clip",
        description="Chinese CLIP ViT-B/16 in Hugging Face Transformers format.",
    ),
    ModelSpec(
        name="vit-base-patch16",
        relative_path="modelscope/hub/google/vit-base-patch16-224-in21k",
        loader="transformers",
        kind="vision",
        description="Google ViT-B/16 pretrained on ImageNet-21k.",
    ),
    ModelSpec(
        name="vit-base-patch16-384",
        relative_path="modelscope/hub/google/vit-base-patch16-384",
        loader="transformers",
        kind="vision",
        description="Google ViT-B/16 fine-tuned at 384px in Hugging Face Transformers format.",
    ),
    ModelSpec(
        name="vit-large-patch16-384",
        relative_path="modelscope/hub/google/vit-large-patch16-384",
        loader="transformers",
        kind="vision",
        description="Google ViT-L/16 fine-tuned at 384px in Hugging Face Transformers format.",
    ),
    ModelSpec(
        name="resnet-50",
        relative_path="modelscope/hub/microsoft/resnet-50",
        loader="transformers",
        kind="vision",
        description="Microsoft ResNet-50 in Hugging Face Transformers format.",
    ),
    ModelSpec(
        name="resnet-101",
        relative_path="modelscope/hub/microsoft/resnet-101",
        loader="transformers",
        kind="vision",
        description="Microsoft ResNet-101 in Hugging Face Transformers format.",
    ),
    ModelSpec(
        name="resnet-152",
        relative_path="modelscope/hub/microsoft/resnet-152",
        loader="transformers",
        kind="vision",
        description="Microsoft ResNet-152 in Hugging Face Transformers format.",
    ),
    ModelSpec(
        name="dinov2-base",
        relative_path="modelscope/hub/facebook/dinov2-base",
        loader="transformers",
        kind="vision",
        description="Facebook DINOv2 base vision backbone.",
    ),
    ModelSpec(
        name="eva02-base-patch14",
        relative_path="modelscope/hub/timm/eva02_base_patch14_224.mim_in22k",
        loader="timm",
        kind="vision",
        architecture="eva02_base_patch14_224",
        description="EVA-02 base patch-14 model in timm format.",
    ),
    ModelSpec(
        name="vit-eva02-base-patch16",
        relative_path="modelscope/hub/nomic-ai/vit_eva02_base_patch16_224.mim_in22k",
        loader="timm",
        kind="vision",
        architecture="eva02_base_patch14_224",
        description="Nomic AI EVA-02 base model in timm format.",
    ),
)

_MODEL_SPECS_BY_NAME = {spec.name: spec for spec in MODEL_SPECS}


def get_model_root(model_root: str | Path | None = None) -> Path:
    """Return the model root from an explicit value, environment, or config."""

    root = model_root
    if root is None:
        root = os.environ.get(MODEL_ROOT_ENV)
    if root is None:
        root = _configured_model_root()
    if root is None:
        root = DEFAULT_MODEL_ROOT
    return Path(root).expanduser().resolve()


def _configured_model_root() -> str | None:
    payload = _read_paths_payload()
    models = payload.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("Invalid path config: `models` must be a mapping.")
    root = models.get("root")
    if root is None:
        return None
    if not isinstance(root, str):
        raise ValueError("Invalid path config: `models.root` must be a string.")
    return root


def _read_paths_payload() -> dict[str, Any]:
    path = _paths_file()
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid path config: {path}")
    return payload


def _paths_file() -> Path:
    configured = os.environ.get("DABENCH_PATHS_FILE")
    if configured:
        return Path(configured).expanduser().resolve()

    configured_dir = os.environ.get("DABENCH_CONFIG_DIR")
    if configured_dir:
        return Path(configured_dir).expanduser().resolve() / "paths.json"

    return Path(__file__).resolve().parents[1] / "config" / "paths.json"


def list_model_specs(
    *,
    model_root: str | Path | None = None,
    check_exists: bool = False,
) -> tuple[ModelSpec, ...]:
    """List registered local model specs.

    Set ``check_exists=True`` to keep only models that are present under the
    selected root.
    """

    specs = tuple(sorted(MODEL_SPECS, key=lambda spec: spec.name))
    if not check_exists:
        return specs
    return tuple(spec for spec in specs if spec.path(model_root).exists())


def list_model_names(
    *,
    model_root: str | Path | None = None,
    check_exists: bool = False,
) -> tuple[str, ...]:
    """List registered model aliases."""

    specs = list_model_specs(model_root=model_root, check_exists=check_exists)
    return tuple(spec.name for spec in specs)


def get_model_spec(name: str) -> ModelSpec:
    """Return the registered spec for ``name``."""

    try:
        return _MODEL_SPECS_BY_NAME[name]
    except KeyError as exc:
        available = ", ".join(list_model_names())
        raise ValueError(f"Unknown model {name!r}. Available models: {available}") from exc


def resolve_model_path(
    name_or_path: str | Path,
    *,
    model_root: str | Path | None = None,
    check_exists: bool = True,
) -> Path:
    """Resolve a registered model alias or explicit filesystem path."""

    value = str(name_or_path)
    spec = _MODEL_SPECS_BY_NAME.get(value)
    if spec is not None:
        path = spec.path(model_root)
        if check_exists and not path.exists():
            raise FileNotFoundError(f"Model {value!r} is registered but missing at {path}")
        return path

    candidate = Path(name_or_path).expanduser()
    is_path_like = (
        candidate.exists()
        or candidate.is_absolute()
        or candidate.parts[:1] in ((".",), ("..",))
        or "/" in value
        or "\\" in value
    )
    if is_path_like:
        path = candidate.resolve()
        if check_exists and not path.exists():
            raise FileNotFoundError(f"Model path does not exist: {path}")
        return path

    available = ", ".join(list_model_names())
    raise ValueError(f"Unknown model {value!r}. Available models: {available}")


__all__ = [
    "DEFAULT_MODEL_ROOT",
    "MODEL_ROOT_ENV",
    "MODEL_SPECS",
    "ModelSpec",
    "get_model_root",
    "get_model_spec",
    "list_model_names",
    "list_model_specs",
    "resolve_model_path",
]
