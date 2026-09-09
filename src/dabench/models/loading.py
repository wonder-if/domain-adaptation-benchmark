"""Pretrained model loading utilities.

This module intentionally does not define classifiers, prompts, adapters, or
training heads. It only resolves local pretrained checkpoints and loads the
corresponding backbone plus preprocessing objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dabench.models.registry import (
    ModelSpec,
    get_model_spec,
    resolve_model_path,
)


@dataclass
class LoadedPretrainedModel:
    """Container returned by :func:`load_model`."""

    name: str
    path: Path
    loader: str
    model: Any
    processor: Any | None = None
    tokenizer: Any | None = None
    image_processor: Any | None = None
    preprocess_train: Any | None = None
    preprocess_eval: Any | None = None


def load_model(
    name_or_path: str | Path,
    *,
    model_root: str | Path | None = None,
    loader: str | None = None,
    device: str | Any | None = None,
    eval_mode: bool = True,
    local_files_only: bool = True,
    **model_kwargs: Any,
) -> LoadedPretrainedModel:
    """Load a local pretrained model by registry alias or path.

    Parameters are deliberately limited to model-loading concerns. Algorithmic
    components such as prompt learning, LoRA, classifier heads, or losses should
    live in experiment code, not in dabench.
    """

    spec = _maybe_get_model_spec(str(name_or_path))
    path = resolve_model_path(name_or_path, model_root=model_root)
    resolved_loader = loader or (spec.loader if spec is not None else infer_loader(path))
    name = spec.name if spec is not None else path.name

    if resolved_loader == "transformers":
        return _load_transformers_model(
            name=name,
            path=path,
            device=device,
            eval_mode=eval_mode,
            local_files_only=local_files_only,
            **model_kwargs,
        )
    if resolved_loader == "timm":
        return _load_timm_model(
            name=name,
            path=path,
            spec=spec,
            device=device,
            eval_mode=eval_mode,
            **model_kwargs,
        )

    raise ValueError("loader must be one of: transformers, timm")


def infer_loader(path: str | Path) -> str:
    """Infer the loader backend for an explicit local model path."""

    model_path = Path(path)
    config = _read_json(model_path / "config.json")
    if isinstance(config.get("pretrained_cfg"), dict) and "architecture" in config:
        return "timm"
    if config.get("model_type") is not None or config.get("architectures") is not None:
        return "transformers"
    return "transformers"


def _load_transformers_model(
    *,
    name: str,
    path: Path,
    device: str | Any | None,
    eval_mode: bool,
    local_files_only: bool,
    **model_kwargs: Any,
) -> LoadedPretrainedModel:
    try:
        from transformers import (  # type: ignore
            AutoImageProcessor,
            AutoModel,
            AutoProcessor,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for model loading. Install with `pip install -e .[models]`."
        ) from exc

    model = AutoModel.from_pretrained(
        str(path),
        local_files_only=local_files_only,
        **model_kwargs,
    )
    _move_and_set_mode(model, device=device, eval_mode=eval_mode)

    processor = _try_from_pretrained(AutoProcessor, path, local_files_only=local_files_only)
    tokenizer = _try_from_pretrained(AutoTokenizer, path, local_files_only=local_files_only)
    image_processor = _try_from_pretrained(
        AutoImageProcessor,
        path,
        local_files_only=local_files_only,
    )

    return LoadedPretrainedModel(
        name=name,
        path=path,
        loader="transformers",
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )


def _load_timm_model(
    *,
    name: str,
    path: Path,
    spec: ModelSpec | None,
    device: str | Any | None,
    eval_mode: bool,
    **model_kwargs: Any,
) -> LoadedPretrainedModel:
    try:
        import timm  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "torch and timm are required for timm model loading. Install with `pip install -e .[models]`."
        ) from exc

    config = _read_json(path / "config.json")
    architecture = spec.architecture if spec is not None else config.get("architecture")
    if not architecture:
        raise ValueError(f"Cannot infer timm architecture from {path / 'config.json'}")

    create_kwargs: dict[str, Any] = {"pretrained": False}
    if "num_classes" in config:
        create_kwargs["num_classes"] = config["num_classes"]
    if "global_pool" in config:
        create_kwargs["global_pool"] = config["global_pool"]
    create_kwargs.update(model_kwargs)

    model = timm.create_model(architecture, **create_kwargs)
    checkpoint = _first_existing(path, ("model.safetensors", "pytorch_model.bin"))
    if checkpoint is None:
        raise FileNotFoundError(f"Missing timm checkpoint under {path}")

    state_dict = _load_state_dict(checkpoint, torch=torch)
    incompatible = model.load_state_dict(state_dict, strict=False)
    model.load_report = incompatible
    _move_and_set_mode(model, device=device, eval_mode=eval_mode)

    return LoadedPretrainedModel(name=name, path=path, loader="timm", model=model)


def _maybe_get_model_spec(name: str) -> ModelSpec | None:
    try:
        return get_model_spec(name)
    except ValueError:
        return None


def _move_and_set_mode(model: Any, *, device: str | Any | None, eval_mode: bool) -> None:
    if device is not None and hasattr(model, "to"):
        model.to(device)
    if eval_mode and hasattr(model, "eval"):
        model.eval()


def _try_from_pretrained(factory: Any, path: Path, *, local_files_only: bool) -> Any | None:
    try:
        return factory.from_pretrained(str(path), local_files_only=local_files_only)
    except (OSError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _first_existing(path: Path, filenames: tuple[str, ...]) -> Path | None:
    for filename in filenames:
        candidate = path / filename
        if candidate.exists():
            return candidate
    return None


def _load_state_dict(path: Path, *, torch: Any) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "safetensors is required for this checkpoint. Install with `pip install -e .[models]`."
            ) from exc
        state = load_file(str(path))
    else:
        state = torch.load(str(path), map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Expected a state dict in {path}")

    return {key.removeprefix("module."): value for key, value in state.items()}


__all__ = [
    "LoadedPretrainedModel",
    "infer_loader",
    "load_model",
]
