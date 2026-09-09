#!/usr/bin/env python3
"""Extract CLIP image/text features for WILDS datasets through dabench loaders."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from dabench.data import get_class_names, load_view, resolve_feature_cache_root
from dabench.models import load_model
from dabench.storage.manifest import get_manifest

WILDS_SPLITS = {
    "camelyon17": ("id_train", "id_val", "unlabeled_train", "ood_val", "ood_test"),
    "iwildcam": ("train", "test"),
}
DEFAULT_MODELS = (
    "clip-vit-base-patch16",
    "clip-vit-l-14-datacomp.xl-s13b-b90k",
)
DEFAULT_PROMPT = "a photo of a {CLASS}."
PRESERVED_COLUMNS = (
    "label",
    "source_index",
    "category_id",
    "category_name",
    "image_id",
    "location",
    "seq_id",
    "frame_num",
    "seq_num_frames",
    "datetime",
    "width",
    "height",
)


class ImageRows:
    def __init__(self, dataset, *, limit: int | None = None) -> None:
        self.dataset = dataset
        self.length = min(len(dataset), limit) if limit is not None else len(dataset)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.dataset[index])
        row["_row_index"] = index
        image = row["image"]
        if isinstance(image, dict):
            row["_image_path"] = image.get("path")
            row["_image_bytes"] = len(image.get("bytes") or b"")
        try:
            if isinstance(image, dict):
                image = _open_image_dict(image)
            if hasattr(image, "convert"):
                image = image.convert("RGB")
        except Exception as exc:
            row["image"] = None
            row["_decode_error"] = f"{type(exc).__name__}: {exc}"
            return row
        row["image"] = image
        row["source_index"] = int(row.get("source_index", index))
        return row


def _open_image_dict(image: dict[str, Any]):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to decode image bytes.") from exc

    if image.get("bytes") is not None:
        return Image.open(io.BytesIO(image["bytes"]))
    if image.get("path"):
        return Image.open(image["path"])
    raise ValueError("Image dictionaries must contain `bytes` or `path`.")


def _decode_error_record(row: dict[str, Any]) -> dict[str, Any]:
    record = {
        "row_index": row.get("_row_index"),
        "error": row.get("_decode_error"),
        "image_path": row.get("_image_path"),
        "image_bytes": row.get("_image_bytes"),
    }
    for column in PRESERVED_COLUMNS:
        if column in row:
            record[column] = row[column]
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(WILDS_SPLITS), default=sorted(WILDS_SPLITS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--splits", nargs="+", default=None, help="Only valid when exactly one dataset is selected.")
    parser.add_argument("--feature-cache-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.splits is not None and len(args.datasets) != 1:
        raise SystemExit("--splits can only be used with exactly one dataset.")

    for model_name in args.models:
        bundle = load_model(model_name, device=args.device)
        for dataset_name in args.datasets:
            splits = tuple(args.splits) if args.splits is not None else WILDS_SPLITS[dataset_name]
            class_names = get_class_names(dataset_name)
            for split in splits:
                extract_one_split(
                    dataset_name=dataset_name,
                    split=split,
                    model_name=model_name,
                    bundle=bundle,
                    args=args,
                )
                if not args.skip_text:
                    save_text_features(
                        dataset_name=dataset_name,
                        split=split,
                        model_name=model_name,
                        bundle=bundle,
                        class_names=class_names,
                        args=args,
                    )


def extract_one_split(
    *,
    dataset_name: str,
    split: str,
    model_name: str,
    bundle,
    args: argparse.Namespace,
) -> None:
    import torch
    from datasets import Dataset
    from torch.utils.data import DataLoader

    output_dir = _image_feature_dir(dataset_name, split, model_name, args.feature_cache_path)
    if output_dir.exists() and not args.overwrite:
        print(f"[skip] {dataset_name}/{split}/{model_name} already exists at {output_dir}", flush=True)
        return

    dataset = load_view(dataset_name, split=split, decode=False, format="hf")
    rows = ImageRows(dataset, limit=args.limit)
    collate_fn = _build_collator(bundle)
    loader = DataLoader(
        rows,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        collate_fn=collate_fn,
    )

    print(
        f"[start] image features dataset={dataset_name} split={split} model={model_name} "
        f"rows={len(rows)} batch_size={args.batch_size} device={args.device}",
        flush=True,
    )
    start = time.time()
    feature_chunks: list[torch.Tensor] = []
    preserved: dict[str, list[Any]] = {column: [] for column in PRESERVED_COLUMNS if column in dataset.column_names}
    decode_errors: list[dict[str, Any]] = []

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            decode_errors.extend(batch.get("decode_errors", []))
            pixel_values = batch.get("pixel_values")
            if pixel_values is None:
                continue
            pixel_values = pixel_values.to(args.device, non_blocking=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(not args.no_amp) and str(args.device).startswith("cuda"),
            ):
                image_features = _encode_images(bundle, pixel_values)
            image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
            feature_chunks.append(image_features.cpu())
            for column in preserved:
                preserved[column].extend(batch[column])
            if batch_index == 1 or batch_index % 50 == 0:
                processed = min(batch_index * args.batch_size, len(rows))
                print(
                    f"[progress] {dataset_name}/{split}/{model_name} "
                    f"{processed}/{len(rows)} ({processed / max(len(rows), 1):.1%})",
                    flush=True,
                )

    if not feature_chunks:
        raise RuntimeError(f"No valid images were decoded for {dataset_name}/{split}/{model_name}.")

    features = torch.cat(feature_chunks, dim=0).numpy().astype(np.float32, copy=False)
    payload: dict[str, Any] = {"image_features": features}
    for column, values in preserved.items():
        payload[column] = values

    output = Dataset.from_dict(payload)
    _save_dataset_atomic(output, output_dir, overwrite=args.overwrite, decode_errors=decode_errors)
    elapsed = time.time() - start
    print(
        f"[done] image features dataset={dataset_name} split={split} model={model_name} "
        f"rows={len(output)} skipped={len(decode_errors)} dim={features.shape[1]} seconds={elapsed:.1f}",
        flush=True,
    )


def save_text_features(
    *,
    dataset_name: str,
    split: str,
    model_name: str,
    bundle,
    class_names: tuple[str, ...],
    args: argparse.Namespace,
) -> None:
    import torch

    output_path = _text_feature_path(dataset_name, split, model_name, args.prompt, args.feature_cache_path)
    if output_path.exists() and not args.overwrite:
        print(f"[skip] text features already exist at {output_path}", flush=True)
        return

    prompts = [args.prompt.replace("{CLASS}", class_name) for class_name in class_names]
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(not args.no_amp) and str(args.device).startswith("cuda"),
        ):
            text_features = _encode_texts(bundle, prompts, args.device)
        text_features = torch.nn.functional.normalize(text_features.float(), dim=-1).cpu()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + f".tmp-{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()
    torch.save(
        {
            "text_features": text_features,
            "classnames": tuple(class_names),
            "prompt": args.prompt,
            "model": model_name,
            "dataset": dataset_name,
            "split": split,
        },
        tmp_path,
    )
    tmp_path.replace(output_path)
    print(
        f"[done] text features dataset={dataset_name} split={split} model={model_name} "
        f"classes={len(class_names)} dim={text_features.shape[1]}",
        flush=True,
    )


def _build_collator(bundle):
    if bundle.loader == "transformers":
        processor = bundle.processor or bundle.image_processor
        if processor is None:
            raise RuntimeError(f"Model {bundle.name!r} does not expose an image processor.")

        def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
            valid, decode_errors = _split_valid_rows(batch)
            if not valid:
                return {"pixel_values": None, "decode_errors": decode_errors}
            images = [item["image"] for item in valid]
            encoded = processor(images=images, return_tensors="pt")
            output = {"pixel_values": encoded["pixel_values"], "decode_errors": decode_errors}
            _copy_preserved_columns(valid, output)
            return output

        return collate

    raise ValueError(f"Unsupported CLIP loader for feature extraction: {bundle.loader}")


def _split_valid_rows(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid = []
    decode_errors = []
    for item in batch:
        if item.get("_decode_error"):
            decode_errors.append(_decode_error_record(item))
        else:
            valid.append(item)
    return valid, decode_errors


def _copy_preserved_columns(batch: list[dict[str, Any]], output: dict[str, Any]) -> None:
    for column in PRESERVED_COLUMNS:
        if column in batch[0]:
            output[column] = [item[column] for item in batch]


def _encode_images(bundle, pixel_values):
    if bundle.loader == "transformers":
        return _feature_tensor(bundle.model.get_image_features(pixel_values=pixel_values))
    raise ValueError(f"Unsupported CLIP loader: {bundle.loader}")


def _encode_texts(bundle, prompts: list[str], device: str):
    if bundle.loader == "transformers":
        tokenizer = bundle.tokenizer
        if tokenizer is None:
            raise RuntimeError(f"Model {bundle.name!r} does not expose a tokenizer.")
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        return _feature_tensor(bundle.model.get_text_features(**inputs))
    raise ValueError(f"Unsupported CLIP loader: {bundle.loader}")


def _feature_tensor(output):
    if hasattr(output, "float"):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, attr, None)
        if value is not None:
            return value
    if isinstance(output, (tuple, list)):
        for value in output:
            if hasattr(value, "float"):
                return value
    raise TypeError(f"Cannot extract a feature tensor from output type {type(output).__name__}.")


def _image_feature_dir(
    dataset_name: str,
    split: str,
    model_name: str,
    feature_cache_path: str | None,
) -> Path:
    root = resolve_feature_cache_root(dataset_name, feature_cache_path)
    dataset_id = str(get_manifest(dataset_name)["id"])
    return root / dataset_id / split / "image_features" / model_name


def _text_feature_path(
    dataset_name: str,
    split: str,
    model_name: str,
    prompt: str,
    feature_cache_path: str | None,
) -> Path:
    root = resolve_feature_cache_root(dataset_name, feature_cache_path)
    dataset_id = str(get_manifest(dataset_name)["id"])
    return root / dataset_id / split / "text_features" / f"{model_name}-{prompt}.pt"


def _save_dataset_atomic(
    dataset,
    output_dir: Path,
    *,
    overwrite: bool,
    decode_errors: list[dict[str, Any]] | None = None,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing feature cache: {output_dir}")
    dataset.save_to_disk(str(tmp_dir))
    if decode_errors:
        (tmp_dir / "decode_errors.json").write_text(
            json.dumps(decode_errors, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    tmp_dir.replace(output_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
