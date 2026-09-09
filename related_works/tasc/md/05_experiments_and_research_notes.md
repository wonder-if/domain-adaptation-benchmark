# 实验结论、局限与研究切入点

## 1. 实验设置

论文在四个域适应基准上评估：

- Office
- Office-Home
- VisDA
- DomainNet

类别偏移场景包括：

- CDA：无类别偏移。
- PDA：目标类是源类子集。
- ODA：目标域有未知类。
- OPDA：源域和目标域都有私有类。

实现细节：

- Backbone：CLIP ViT-B/16。
- 训练方式：image encoder 和 text encoder 都用 LoRA，rank = 8。
- 学习率：`0.0001`，SGD，momentum `0.9`，weight decay `0.0005`。
- scheduler：`eta = eta0 * (1 + 10p)^(-0.75)`。
- 默认 `lambda_div = 0.6`。
- `tau = 0.02`。
- 离散搜索：Office / Office-Home / VisDA 的 `K0=100`，DomainNet 的 `K0=400`，`K_s=300`，`gamma_ent=0.3`，`N_outer=20`。

代码中这些参数主要在 [`configs/unida.yaml`](../codes/TASC/configs/unida.yaml) 和 `configs-reproduce` 下的任务配置中。

## 2. 主结果怎么读

论文 Table 1 报告：

- OPDA / ODA：H-score。
- PDA / CDA：classification accuracy。

核心结论：

1. TASC 在四个 benchmark 的平均表现都最好。
2. 在更大规模的 DomainNet 和 VisDA 上优势尤其明显。
3. 相比同样使用 CLIP + LoRA 的 baselines，TASC 仍然明显更强，说明提升不只是来自 CLIP backbone。

这支持论文主张：把 target semantic centers 限制在离散文本空间，确实提升了 UniDA 对不同 category shift 的鲁棒性。

## 3. 消融实验怎么读

Table 2 消融：

| 组件 | 作用 |
| --- | --- |
| `L_CE` | 源域监督，维持源分类器 |
| `L_TASC` | 目标域信息最大化，完成 target clustering 和 common alignment |
| `UniMS` | 推理时 unknown detection |

结果显示：

- 单独源域训练已经是很强 baseline，因为 CLIP 本身迁移性强。
- 加 UniMS 后，unknown detection 更好，H-score 上升。
- 再加 `L_TASC` 后，target alignment 和 clustering 改善，整体最好。

Table 3 说明使用 TASC 估计出的 target-private 比例作为 GMM weights 更有效，尤其在类别极不平衡场景。

Table 4 对 UniMS 分项消融：

- `MS-s` 是很强 baseline。
- 加 `ent^s` 后更好。
- `MS-t` 单独较弱，但加 `ent^t` 后明显提升。
- 两项相减的 UniMS 最好，说明 target semantic centers 提供了互补信息。

## 4. 论文方法的主要优点

### 4.1 目标函数统一

TASC 没有专门设计复杂的 common class matching loss。它把目标域问题统一成信息最大化：

$$
L_{TASC}
=
\mathbb{E}_{x^t} \operatorname{Entropy}(p(x^t))
- \lambda_{div}\operatorname{Entropy}(\bar p)
$$

这种简单性来自搜索空间约束：如果 target centers 足够语义化、少域偏置，那么对 target 做 clustering 本身就能起到 alignment 作用。

### 4.2 能估计 cluster 数量

隐藏状态 `r` 让模型可以从上界 `K0` 中自动保留或丢弃中心：

$$
K = \sum_i r_i
$$

这比手动设定 target clusters 更适合 UniDA。

### 4.3 类别偏移被显式建模

通过 `W_s` 和 `S_t` 互相分类的 entropy，TASC 为每个 source class 和 target center 估计“是否 common”的倾向，再把它用于 UniMS。

## 5. 局限与可能风险

### 5.1 搜索依赖 WordNet noun 空间

如果目标类别不容易被 WordNet nouns 表达，或者类别名需要多词短语、属性组合、动作描述，搜索空间可能不够好。

可以考虑扩展到：

- dataset-specific candidate phrases
- LLM 生成的 class-name candidates
- caption noun phrases
- hierarchical WordNet synsets
- CLIP retrieval 得到的 web-scale text candidates

### 5.2 贪心搜索不是全局最优

默认 `K_beam=1`，本质是单路径贪心。每步只随机采样 `K_s` 个 nouns，且每次只看最多 5000 个 target features。

这带来两个问题：

- 搜索结果可能对随机种子敏感。
- 一些 candidate noun 的组合效应无法被单步评估发现。

可研究方向：

- 增大 beam width。
- 用 evolutionary search / MCTS / simulated annealing。
- 先做 candidate pruning，再做更精细组合优化。
- 用 differentiable relaxation 近似 `r` 和 noun selection。

### 5.3 信息最大化可能鼓励错误均衡

`L_div` 鼓励平均预测分布更均匀，但真实 target 类分布可能长尾。论文把 `lambda_div` 从 SHOT-IM 常用的 1.0 降到 0.6，说明这点确实敏感。

可研究方向：

- 类先验自适应估计。
- 用 entropy regularization 替代强均衡。
- 对 `L_div` 做 schedule，早期强、后期弱，或根据 predicted cluster size 调整。

### 5.4 UniMS 仍然依赖阈值模型

GMM 阈值比手调阈值更稳，但仍假设 known score 能被两个高斯成分合理建模。在 target-private 极少、score 分布重叠或存在多模态 unknown 时可能失效。

可研究方向：

- non-parametric thresholding
- energy-based unknown modeling
- conformal prediction
- per-class threshold
- 用 target center entropy 和 image uncertainty 联合建模

## 6. 适合继续研究的具体问题

### 6.1 更强的候选文本空间

当前候选来自 WordNet nouns，语义覆盖广但不一定贴合数据集。可以研究：

```text
WordNet nouns
  + source class names
  + generated class candidates
  + dataset captions
  + noun phrases from vision-language captioner
  + synonyms / hypernyms / hyponyms
```

重点问题：如何避免候选空间变大后搜索更不稳定。

### 6.2 更稳的 K 估计

`K = sum r_i` 是 TASC 的亮点，也可能是后续改进点。可以单独评估：

- `K0` 对不同数据集的敏感性。
- `gamma_ent` 对 source-private discard 的影响。
- 搜索过程中的 `K` 收敛曲线。
- 估计 `K` 与真实 target 类数的偏差如何影响 H-score。

### 6.3 把 TASC 框架换成别的 clustering objective

论文也说 `L_clu` 的形式本质上可以替换。可尝试：

- contrastive clustering
- neighborhood consistency
- Sinkhorn balanced assignment
- prototype contrastive learning
- mutual information with class prior correction

评估重点是：目标函数变复杂后，是否仍然保持 TASC 的鲁棒性。

### 6.4 更细的 unknown 处理

当前 UniDA 评估把所有 target-private 合成一个 unknown 类。但 TASC 搜索出的 target nouns 本身可能已经提供 private class clustering。可以进一步研究：

- unknown 样本内部命名。
- target-private 类别发现。
- open-world incremental adaptation。
- 对 discovered nouns 的 human interpretability 评估。

## 7. 复现实验时的注意事项

1. 代码要求 CUDA，入口会直接检查 `torch.cuda.is_available()`。
2. README 提到 `faiss.kmeans` 在某些 GPU 上有问题，作者建议 RTX3090 或 RTX3080ti。
3. 需要下载 CLIP checkpoint、txt split 文件、WordNet noun features。
4. 搜索阶段会把结果写到 logdir 下的 `target_classnames.txt`，这是分析 TASC 行为最重要的文件之一。
5. 如果要 debug 方法，建议先用 Office / Office-Home 小任务，不要直接跑 DomainNet。

## 8. 建议的研究日志模板

每次改动 TASC 时，建议记录：

```text
Dataset / task / split:
Seed:
K0:
K_s:
gamma_ent:
lambda_div:
estimated K:
estimated shared:
target_classnames examples:
H-score / OS* / UNK:
AUROC:
NMI:
Failure cases:
```

特别要看 `target_classnames.txt`，因为 TASC 的成败通常能从搜索出的 nouns 质量中直接看出来。

