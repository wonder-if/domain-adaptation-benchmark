# 代码结构与运行链路

本节按一次训练实际发生的顺序读代码。主实现建议看 [`../codes/TASC`](../codes/TASC)，`codes/tasc-dabench` 基本是另一份拷贝。

## 1. 入口：train_sapphire.py

运行命令：

```bash
python train_sapphire.py --cfg configs/unida.yaml
```

入口文件：[`train_sapphire.py`](../codes/TASC/train_sapphire.py)

主要步骤：

1. 读取 YAML 配置。
2. 设置 `CUDA_VISIBLE_DEVICES`。
3. 创建 TensorBoard logdir，并复制代码和配置到输出目录。
4. 构建 `TASCTrainer`。
5. 依次调用：

```python
trainer.build_model()
trainer.build_data_loaders()
trainer.build_optimizer()
trainer.build_lr_schedule()
trainer.train()
```

`METHOD = {'TASC': TASCTrainer}`，所以核心 trainer 来自 [`train_sapphire_TASC.py`](../codes/TASC/train_sapphire_TASC.py)。

## 2. 配置文件关键项

默认配置：[`configs/unida.yaml`](../codes/TASC/configs/unida.yaml)

关键超参：

```yaml
method_args:
  ce_prob_temp: 0.02
  text_temp: 0.02
  ent_temp: 0.01
  metric_temp: 0.01
  ent_thr: 0.3
  loss_s_ce: 1.0
  loss_t_im_ent: 1.0
  loss_t_im_div: 0.6
  templates_type: ensemble
  TASC:
    num_clusters_dict:
      office: 100
      officehome: 100
      visda: 100
      domainnet: 400
    n_inner: 20
    K_s: 300
    K_beam: 1
```

这里 `n_inner` 对应论文中的 outer iteration 次数，代码命名为 inner；`K_s` 是每个位置采样的 candidate nouns 数量；`K_beam` 预留 beam search，但默认是 1，实际接近单路径贪心。

## 3. 数据集与类别划分

数据集逻辑在 [`sapphire/datasets/universalDA.py`](../codes/TASC/sapphire/datasets/universalDA.py)。

配置里：

```yaml
dataset:
  name: office
  task: dw
  shared: 10
  source_private: 10
  target_private: 11
```

`task: dw` 表示 source 是 DSLR，target 是 Webcam。`make_class_split()` 会按连续 label 划分类别：

```python
class_split['shared'] = list(range(shared))
class_split['source_private'] = list(range(shared, shared+source_private))
class_split['target_private'] = list(range(shared+source_private, ...))
```

构建 source dataset 时只包含 shared + source_private；构建 target dataset 时只包含 shared + target_private。

注意：目标域训练时虽然没有标签，但代码中的 dataset 仍保存 `gt_label`，主要用于测试和搜索阶段打印 clustering metrics。论文方法本身不使用 target labels 优化。

## 4. CLIP + LoRA 构建

模型基类在 [`train_sapphire_CLIPLoRABase.py`](../codes/TASC/train_sapphire_CLIPLoRABase.py)。

构建流程：

```python
self.clip_model = build_backbone_local(self.model_name)
self.backbone = self.clip_model.visual
self.text_encoder = LoRA_ViT(TextEncoder(self.clip_model), lora_r=self.lora_r_t)
self.backbone = LoRA_ViT(self.backbone, lora_r=self.lora_r)
```

LoRA 实现在 [`sapphire/models/LoRA_layer.py`](../codes/TASC/sapphire/models/LoRA_layer.py)。它替换 `nn.MultiheadAttention`，只对 query 和 value projection 加 LoRA：

```python
self.q_proj = Linear(..., lora_r=lora_r)
self.k_proj = nn.Linear(...)
self.v_proj = Linear(..., lora_r=lora_r)
```

优化器只训练名字里包含 `lora_` 的参数：

```python
for name, param in self.model.named_parameters():
    param.requires_grad = False
...
if "lora_" in name:
    param.requires_grad = True
```

所以 TASC 实验不是 full fine-tune，而是 CLIP image/text encoders 的 LoRA adaptation。

## 5. 训练前：before_training 触发离散搜索

`trainer.train()` 中，在进入迭代训练前调用：

```python
self.model.before_training()
```

对 TASC 来说，这会执行：

```python
self.discovered_classnames, self.num_clusters, self.estimated_shared = self.discrete_optimization()
```

也就是先搜索 target nouns，再开始 LoRA 训练。

`discrete_optimization()` 的顺序：

1. `init_memory()` 抽取全部 target features。
2. `embed_classnames(source)` 编码源类名。
3. `get_all_nouns_features()` 编码或加载 WordNet noun embeddings。
4. 调用 `greedy_search()`。
5. 保存搜索出的 `target_classnames.txt`。

## 6. WordNet noun embeddings

[`train_sapphire_TASCBase.py`](../codes/TASC/train_sapphire_TASCBase.py) 中：

```python
self.nouns_np = get_nouns_from_wordnet()
path_extracted = "data/WordNet/nouns_feat_rm_redundancy_ensemble.pth"
```

如果预先下载的 `.pth` 存在，就直接加载。否则会遍历 WordNet 所有 nouns，用 CLIP text encoder 和 ensemble templates 编码：

```python
prompts = [template.format(noun) for template in templates]
text_features = self.text_encoder(embeddings, tokenized_prompts)
text_features = text_features.mean(dim=0)
text_features = F.normalize(text_features)
```

这一步非常耗时，所以 README 建议下载作者预提取的 noun features。

## 7. 搜索函数 greedy_search

核心文件：[`train_sapphire_TASC.py`](../codes/TASC/train_sapphire_TASC.py)

核心状态：

```python
S_iter      # shape: K_beam x K_0 x feat_dim
r_iter      # shape: K_beam x K_0, bool
nouns_iter  # shape: K_beam x K_0, string
```

为了加速，每次评估只随机采样最多 5000 个 target features：

```python
rand_indices = torch.randperm(num_samples)
rand_indices = rand_indices[:min(5000, num_samples)]
image_features_p = image_features[rand_indices]
```

每一轮对每个位置 `i` 做候选替换或丢弃评估。实际 objective 通过 `Metric()` 计算，它复用 `shot_im_loss()`，因此搜索阶段和训练阶段目标一致。

搜索结束后：

```python
estimated_shared = torch.nonzero(r_iter[:num_src_names]).size(0)
```

这个值后面用于 GMM 阈值的 mixture weights。

## 8. LoRA 训练阶段 forward

搜索结束后进入常规迭代训练。`TASC.forward()` 同时拿 source batch 和 target batch：

```python
source_images = batched_inputs[0]['aug'][0]
target_images = batched_inputs[1]['img']
```

源域：

```python
text_features = self.text_forward(self.source_classnames)
source_logits = (source_feat @ text_features.T) / self.ce_prob_temp
loss_s_ce = F.cross_entropy(source_logits, source_labels, label_smoothing=0.1)
```

目标域：

```python
text_features_target = self.text_forward(self.discovered_classnames)
cluster_logits = (target_feat @ text_features_target.T) / self.ce_prob_temp
loss_t_im_ent, loss_t_im_div = shot_im_loss(cluster_logits)
```

每个 iteration 会随机选择 prompt template：

```python
text_indexs = torch.randint(self.n_templates, (len(classnames),))
prompts = [template.format(name) for name, template in zip(classnames, text_templates)]
```

所以训练中的文本中心带有 prompt augmentation。

## 9. 测试阶段

`TASCTrainer.test()` 逐批调用 `self.model.predict()`。

`predict()` 做三件事：

1. 用源类名中心预测 known class label。
2. 用 target noun centers 得到 cluster logits。
3. 计算多种 unknown scores，包括 UniMS。

之后 trainer 归一化 known score，用 GMM 得阈值，并把低于阈值的样本标签改成 unknown：

```python
known_mask = known_scores > thr
predict_labels[~known_mask] = self.info.num_classes
```

最后调用 `evaluate()` 统计 H-score、AUROC、NMI 等指标。

## 10. 代码和论文之间的几个命名差异

| 论文概念 | 代码变量 / 函数 | 说明 |
| --- | --- | --- |
| `K_0` | `num_clusters_dict[dataset]` | cluster 数上界 |
| `N_outer` | `n_inner` | 外层遍历次数，代码命名容易误解 |
| `n_c` | `K_s` | 每个位置采样的 WordNet noun 候选数 |
| `L_clu` / `L_TASC` | `shot_im_loss` + `lambda_dict` | 信息最大化损失 |
| `T^r` | `discovered_classnames` | 搜索后保留的 target nouns |
| `ent^s` | `s2t_ent` | source centers 到 target centers 的熵 |
| `ent^t` | `t2s_ent` | target centers 到 source centers 的熵 |
| source semantic centers | `text_features` | 源类名编码 |
| target semantic centers | `text_features_target` | 搜索 noun 编码 |

