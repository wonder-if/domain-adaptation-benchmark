# TASC 方法主体

本节是整篇论文最重要的部分。TASC 把 UniDA 写成一个混合整数非线性优化问题，然后用“两阶段”近似求解：

1. 固定 CLIP 编码器，在离散文本空间中搜索 target semantic centers。
2. 固定搜索结果，只优化 LoRA 参数，做源域监督和目标域信息最大化。

## 1. 目标域聚类的数学形式

目标域没有标签，也没有类别名。TASC 希望找到一组文本 nouns：

$$
T = (t_1, t_2, \ldots, t_K)
$$

这些 nouns 的 CLIP text embeddings：

$$
S_t = [s_1^t, s_2^t, \ldots, s_K^t],
\quad s_j^t = g(t_j)
$$

作为目标域 semantic centers。每个 target image `x^t` 经 image encoder 得到：

$$
z^t = f(x^t)
$$

再对所有 target semantic centers 做 softmax：

$$
p(x^t) = h(f(x^t); S_t, \tau)
$$

论文采用信息最大化作为目标域聚类损失：

$$
L_{clu}
= L_{ent} + \lambda_{div} L_{div}
=
\mathbb{E}_{x^t \in D_t} \operatorname{Entropy}(p(x^t))
- \lambda_{div} \operatorname{Entropy}(\bar p)
$$

其中：

$$
\bar p = \mathbb{E}_{x^t \in D_t}p(x^t)
$$

直观解释：

- `L_ent` 小：每个样本的预测要自信，避免模糊归类。
- `-Entropy(\bar p)` 小，也就是 `Entropy(\bar p)` 大：整个 batch / dataset 的平均预测要分散，避免所有样本塌缩到一个类。

代码对应：

```python
def shot_im_loss(logits, normalized=False):
    probs = F.softmax(logits, dim=-1)
    mprobs = probs.mean(dim=-2, keepdim=True)
    entropy = -(probs * (probs + 1e-5).log()).sum(dim=-1)
    mentropy = -(mprobs * (mprobs + 1e-5).log()).sum(dim=-1)
    return entropy.mean(dim=-1), -mentropy.squeeze(-1)
```

在 [`train_sapphire_TASC.py`](../codes/TASC/train_sapphire_TASC.py) 中，`shot_im_loss` 返回：

- `loss_t_im_ent = L_ent`
- `loss_t_im_div = -Entropy(\bar p)`

配置中默认：

```yaml
loss_t_im_ent: 1.0
loss_t_im_div: 0.6
```

所以：

$$
L_{TASC} = 1.0 \cdot L_{ent} + 0.6 \cdot L_{div}
$$

## 2. 为什么需要隐藏状态 r

如果直接搜索 `K` 个 nouns，就必须预先知道目标域类别数。但 UniDA 中这正是未知的。

TASC 设一个上界 `K_0`，再引入二值隐藏状态：

$$
r = [r_1, r_2, \ldots, r_{K_0}] \in \{0, 1\}^{K_0}
$$

其中：

- `r_i = 1`：第 `i` 个 noun 被保留，参与 target clustering。
- `r_i = 0`：第 `i` 个 noun 被丢弃。

于是实际 cluster 数量是：

$$
K = \sum_{i=1}^{K_0}r_i
$$

保留下来的 noun 序列记为：

$$
T^r = (t_i \mid r_i = 1)
$$

完整优化问题变成：

$$
\min_{T,r,\theta}
L_{TASC}(T, r, \theta; D_t)
=
L_{clu}(T^r, \theta; D_t)
$$

约束为：

$$
t_i \in T^s \cup T^{nouns},\quad
r_i \in \{0,1\}
$$

这是一个混合整数非线性问题。`T` 和 `r` 是离散变量，CLIP/LoRA 参数 `theta` 是连续变量。

## 3. 第一阶段：Greedy Search-based Discrete Optimization

论文第一阶段固定模型参数，只搜索 `T` 和 `r`。

初始化：

$$
T = (t_1^s, t_2^s, \ldots, t_{|C_s|}^s, t_{|C_s|+1}, \ldots, t_{K_0})
$$

前 `|C_s|` 个位置放源类名，后面随机放 WordNet nouns。`r` 初始全为 1。

代码对应 [`greedy_search`](../codes/TASC/train_sapphire_TASC.py)：

```python
S_iter[b] = torch.cat([W_src, nouns_features[rand_index]])
r_iter[b] = torch.ones(K_0) > 0
nouns_iter[b] = np.concatenate([W_src_classnames, nouns[rand_index.cpu().numpy()]])
```

单步更新第 `i` 个位置时，其它位置固定：

1. 如果 `i` 是源类名位置，候选只允许是原源类名。
2. 如果 `i` 是 WordNet noun 位置，从所有 nouns 中随机采样 `K_s` 个候选，并额外保留当前位置的旧 noun。
3. 计算每个候选被保留时的聚类目标。
4. 计算当前位置被丢弃时的聚类目标。
5. 选择更优状态，更新 `T_i` 和 `r_i`。

论文中的两个核心量：

$$
L_{min}
=
\min_{t' \in T^c}
L_{TASC}(T_{i|t'}, r_{i|1}, \theta; D_t)
$$

$$
L_{dis}
=
L_{TASC}(T, r_{i|0}, \theta; D_t)
$$

如果 `L_min < L_dis`，说明保留某个候选 noun 更好；否则丢弃该位置更好。

代码实现中为了统一成“越大越好”的 score，`Metric` 返回的是 `-loss`：

```python
def Metric(S_iter, r_iter, image_features, lambda_dict, ce_prob_temp=0.01):
    logits = (image_features @ S_iter[:, r_iter, :].mT) / ce_prob_temp
    loss_t_im_ent, loss_t_im_div = shot_im_loss(logits, normalized=True)
    score = 0
    score += lambda_ent * loss_t_im_ent
    score += lambda_div * loss_t_im_div
    return -score
```

因此代码里是选 `Metric` 最大的 candidate。

## 4. 搜索阶段的两个关键工程设计

### 4.1 源类名位置不替换

前 `|C_s|` 个位置代表源类名。为了让 common 类在源域和目标域拥有相同或极相似的文本中心，TASC 不替换这些 nouns：

```python
if i < num_src_names:
    results_features = W_src[i:i+1]
    results_nouns = W_src_classnames[i:i+1]
```

这很关键：如果某个源类是 common，它应该在 target semantic centers 中保留下来；如果是 source-private，它应该被 `r_i=0` 丢弃，而不是被替换成另一个 noun。

### 4.2 用 entropy 约束保留 common 源类

论文提出：对源类名位置，计算该源类 embedding 相对于当前 target prototypes 的熵。如果熵低于阈值 `gamma_ent`，说明它能明确匹配某个 target prototype，更可能是 common 类，应强制保留。

代码：

```python
if ent_thr is not None and i < num_src_names:
    prototypes = get_prototypes(image_features_p, S_iter[b][r_iter[b]].unsqueeze(0)).squeeze(0)
    ent_best = get_entropy(S_iter[b, i:i+1, :], prototypes, temp=ent_temp).squeeze()
    if ent_best < ent_thr:
        M_beam_dis[b] -= 1e5
```

这里 `M_beam_dis` 是丢弃该源类的 score。若 `ent_best < ent_thr`，代码把丢弃分数减去一个很大的数，等价于禁止丢弃。

配置默认：

```yaml
ent_thr: 0.3
ent_temp: 0.01
```

## 5. 第二阶段：Model Refinement

离散搜索结束后得到：

$$
T^*,\quad r^*
$$

此时固定搜索结果，只训练模型参数 `theta`。总损失是：

$$
\min_\theta L_{all}
= L_{CE} + L_{TASC}
$$

其中源域交叉熵：

$$
L_{CE}
=
\operatorname{CE}
\left(
\operatorname{softmax}
\left(
\frac{f(x^s)^T W_s}{\tau}
\right),
y^s
\right)
$$

目标域仍然用搜索出的 target centers：

$$
L_{TASC}
=
L_{clu}(T^{*r}, \theta; D_t)
$$

代码对应 `TASC.forward()`：

```python
source_feat = self.backbone(source_images)
source_logits = (source_feat @ text_features.T) / self.ce_prob_temp
loss_s_ce = F.cross_entropy(source_logits, source_labels, label_smoothing=0.1)

target_feat = self.backbone(target_images)
cluster_logits = (target_feat @ text_features_target.T) / self.ce_prob_temp
loss_t_im_ent, loss_t_im_div = shot_im_loss(cluster_logits)
```

其中：

- `text_features = text_forward(source_classnames)`
- `text_features_target = text_forward(discovered_classnames)`

也就是说训练时不是直接使用第一阶段保存的固定 embedding，而是固定 noun 名称，再用当前 text encoder 重新编码。这使 LoRA 微调能同时影响 image encoder 和 text encoder。

## 6. 这一步为什么能同时做 alignment 和 private clustering

`L_CE` 让源类名对应的分类器保持源域监督能力。`L_TASC` 让目标样本靠近搜索出的 target semantic centers。

由于搜索阶段专门让 common 类的 target center 尽量保留源类名，所以：

- 对 common target 样本，`L_TASC` 会把它们推向与源分类器共享的语义中心，实现 domain alignment。
- 对 target-private 样本，搜索出的 WordNet nouns 成为额外 target centers，`L_TASC` 会把它们聚起来，但不会强行对齐到源类。

这也是论文声称“用一个信息最大化目标同时完成 common alignment 和 private clustering”的原因。

## 7. 伪代码

```text
Input:
  source class names Ts
  target images Dt
  WordNet nouns Tnouns
  CLIP image encoder f, text encoder g

Stage 1: discrete optimization
  freeze f and g
  extract target features Zt = f(Dt)
  encode source class names Ws = g(Ts)
  encode WordNet nouns Wn = g(Tnouns)
  initialize T = [Ts, random nouns], r = all ones

  repeat N_outer times:
    for i in 1..K0:
      build candidates for T_i
      evaluate retaining candidate with information maximization
      evaluate discarding position i
      apply source-class entropy constraint if i <= |Cs|
      update T_i and r_i

  output discovered target nouns T*r

Stage 2: model refinement
  fix discovered noun names
  train LoRA parameters with:
    Lall = LCE(source images, source labels, source names)
         + Lent(target images, discovered names)
         + lambda_div Ldiv(target images, discovered names)
```

