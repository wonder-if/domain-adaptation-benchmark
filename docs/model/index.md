# Models

`dabench.models` is a thin local pretrained-model loading layer. It is separate
from data loading and intentionally does not include adaptation algorithms,
trainable prompts, LoRA modules, classifier heads, losses, or training loops.

Install the optional model dependencies when you need to instantiate backbones:

```bash
pip install -e ".[models]"
```

## Model Root

The local model root is read from `models.root` in the dabench path config.
Override it with `DABENCH_MODEL_ROOT` or the `model_root` argument:

```python
from dabench.models import load_model

bundle = load_model("clip-vit-base-patch16")
bundle = load_model("clip-vit-base-patch16", model_root="/path/to/models")
```

`dabench` does not download model weights implicitly. The alias must resolve to a
local directory, or you can pass an explicit local path.

## Usage

```python
from dabench.models import list_model_names, load_model, resolve_model_path

print(list_model_names(check_exists=True))
print(resolve_model_path("clip-vit-base-patch16"))

bundle = load_model("clip-vit-base-patch16", device="cuda")
model = bundle.model
processor = bundle.processor
tokenizer = bundle.tokenizer
```

The returned `LoadedPretrainedModel` records the resolved path, loader backend,
model object, and any available preprocessing object:

- Transformers models expose `processor`, `tokenizer`, and `image_processor` when
  the local directory provides them.
- timm models expose the loaded backbone and keep the timm load report on
  `bundle.model.load_report`.

## WILDS CLIP Features

For WILDS feature caching, use the extraction script rather than reading image
files directly:

```bash
PYTHONPATH=src python scripts/extract_wilds_clip_features.py \
  --datasets camelyon17 iwildcam \
  --models clip-vit-base-patch16 clip-vit-l-14-datacomp.xl-s13b-b90k
```

The script loads every split through `dabench.data.load_view`, loads local
pretrained CLIP checkpoints through `dabench.models.load_model`, and stores
normalized image features plus class-name text features. It does not create
trainable prompts or adaptation modules. DataComp CLIP models are loaded through
Transformers, matching the local model layout used by `adaptation-of-clip`.

iWildCam contains a small number of undecodable JPEG rows. Feature extraction
skips those rows, writes valid features for the remaining rows, and stores the
skipped metadata in `decode_errors.json` beside the image feature cache.

## Registered Models

| Alias | Loader | Relative path under the configured model root |
| --- | --- | --- |
| `clip-vit-base-patch16` | `transformers` | `modelscope/hub/openai-mirror/clip-vit-base-patch16` |
| `clip-vit-base-patch32` | `transformers` | `modelscope/hub/thomas/clip-vit-base-patch32` |
| `clip-vit-b-16-datacomp.l-s1b-b8k` | `transformers` | `modelscope/hub/laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K` |
| `clip-vit-l-14-datacomp.xl-s13b-b90k` | `transformers` | `modelscope/hub/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K` |
| `chinese-clip-vit-base-patch16` | `transformers` | `modelscope/hub/Xenova/chinese-clip-vit-base-patch16` |
| `vit-base-patch16` | `transformers` | `modelscope/hub/google/vit-base-patch16-224-in21k` |
| `resnet-50` | `transformers` | `modelscope/hub/microsoft/resnet-50` |
| `dinov2-base` | `transformers` | `modelscope/hub/facebook/dinov2-base` |
| `eva02-base-patch14` | `timm` | `modelscope/hub/timm/eva02_base_patch14_224.mim_in22k` |
| `vit-eva02-base-patch16` | `timm` | `modelscope/hub/nomic-ai/vit_eva02_base_patch16_224.mim_in22k` |

Use `loader="transformers"` or `loader="timm"` only when you need to override
the registry default for an explicit local path.
