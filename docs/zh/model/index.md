# 模型

`dabench.models` 是一个很薄的本地预训练模型加载层。它和数据加载分离，并且刻意不包含
领域自适应算法、可训练 prompt、LoRA、分类头、loss 或训练循环。

需要真正实例化 backbone 时，先安装可选模型依赖：

```bash
pip install -e ".[models]"
```

## 模型根目录

本地模型根目录从 dabench 路径配置里的 `models.root` 读取。
可以用 `DABENCH_MODEL_ROOT` 或 `model_root` 参数覆盖：

```python
from dabench.models import load_model

bundle = load_model("clip-vit-base-patch16")
bundle = load_model("clip-vit-base-patch16", model_root="/path/to/models")
```

`dabench` 不会隐式下载模型权重。模型别名必须能解析到本地目录，也可以直接传入本地路径。

## 使用方式

```python
from dabench.models import list_model_names, load_model, resolve_model_path

print(list_model_names(check_exists=True))
print(resolve_model_path("clip-vit-base-patch16"))

bundle = load_model("clip-vit-base-patch16", device="cuda")
model = bundle.model
processor = bundle.processor
tokenizer = bundle.tokenizer
```

返回的 `LoadedPretrainedModel` 会记录解析后的路径、加载后端、模型对象，以及本地目录中可用的预处理对象：

- Transformers 模型会尽量提供 `processor`、`tokenizer` 和 `image_processor`。
- timm 模型会提供已加载的 backbone，并把 timm 加载报告保存在 `bundle.model.load_report`。

## WILDS CLIP 特征

生成 WILDS 缓存特征时，应使用抽取脚本，而不是在脚本里重新直接读取图片文件：

```bash
PYTHONPATH=src python scripts/extract_wilds_clip_features.py \
  --datasets camelyon17 iwildcam \
  --models clip-vit-base-patch16 clip-vit-l-14-datacomp.xl-s13b-b90k
```

脚本会通过 `dabench.data.load_view` 读取每个 split，通过 `dabench.models.load_model`
加载本地预训练 CLIP，并写入归一化后的 image features 和基于类别名的 text features。
它不包含可训练 prompt 或领域自适应模块。DataComp CLIP 模型通过 Transformers 加载，
和 `adaptation-of-clip` 中的本地模型布局保持一致。

iWildCam 中有少量 JPEG 无法解码。特征抽取会跳过这些行，为其余样本写入特征，
并把跳过样本的 metadata 记录到对应 image feature cache 旁边的 `decode_errors.json`。

## 已注册模型

| 别名 | 加载器 | 已配置模型根目录下的相对路径 |
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

只有在对显式本地路径需要覆盖默认判断时，才需要手动传入 `loader="transformers"`
或 `loader="timm"`。
