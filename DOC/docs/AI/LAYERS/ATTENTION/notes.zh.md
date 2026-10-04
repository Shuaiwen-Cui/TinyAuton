# TinyAI · 层 · 注意力 — 设计说明 {#_1}

!!! info "实现依据与记录"
    本节接口以 `CODE/AIoTNode-TinyAuton-AI/middleware/` 为依据。源码摘录与串口输出包含历史记录；是否运行某项测试，请核对工程入口和启用开关。

!!! note "说明"
    `Attention` 实现多头自注意力（Multi-Head Self-Attention），是 Transformer 架构的核心模块。该实现支持任意 `embed_dim` / `num_heads`（要求能整除），并提供完整的反向传播，便于在端侧微调。

!!! abstract "Attention — 多头自注意力：从多个角度看输入"
    每个位置"关注"其他位置，提取上下文信息。QKV 机制的核心直觉。

## 算法直觉 {#_2}

### Q、K、V 的角色（图书馆借书比喻） {#qkv}

| 角色 | 符号 | 什么意思 | 图书馆比喻 |
|------|------|----------|------------|
| **Query** | Q | "我在找什么" | 你想借的书名 |
| **Key** | K | "我有什么" | 书架上每本书的标签 |
| **Value** | V | "我有什么内容" | 书里面的内容 |

**过程**：

1. 你的 Query 和每个 Key 做匹配（点积）→ 得到每个 Key 的匹配分数
2. Softmax 把匹配分数变成注意力权重（和为 1 的概率）
3. 用注意力权重对 Value 做加权求和 → 得到关注后的结果

### 多头注意力 {#_3}

不是只从一种角度"关注"，而是分成多个头，每个头学习不同的关注角度：

- 头 1：关注局部上下文（当前词附近）
- 头 2：关注全局结构（句子开头的相关信息）
- 头 3：关注特定模式（否定词、转折等）


!!! tip "自注意力 = 每个位置和其他所有位置交互"
    "自"的意思是 Q、K、V 都来自同一个输入。每个点看看其他点都做了什么，然后决定自己该输出什么。

### 形状变化 {#_4}

```
输入 [B, seq_len, embed_dim]  →  输出 [B, seq_len, embed_dim]
```

形状不变，但每个位置的表达融入了上下文信息。

---

## 数学定义 {#_5}

输入 `x` 形状 `[batch, seq_len, embed_dim]`，记 `B / S / E`。每个头的维度 `D = E / H`。

### Q / K / V 投影 {#q-k-v}

\[
Q = x W_q^\top + b_q,\quad
K = x W_k^\top + b_k,\quad
V = x W_v^\top + b_v
\]

形状均为 `[B, S, E]`。

### 多头切分 {#_6}

按头索引 `h = 0..H-1`，对 Q / K / V 沿最后一维切片成 `[B, S, D]`。

### 缩放点积注意力 {#_7}

对每个头：

\[
A_{b, s_1, s_2} = \frac{Q_{b, s_1, :}\,K_{b, s_2, :}^\top}{\sqrt{D}}
\]

\[
A = \mathrm{softmax}_{s_2}(A)
\]

\[
\mathrm{ctx}_{b, s_1, :} = \sum_{s_2} A_{b, s_1, s_2}\,V_{b, s_2, :}
\]

### 拼接 + 输出投影 {#_8}

把每个头的 `ctx_h` 写回 `ctx[:, :, h*D : (h+1)*D]`，然后过最后一次线性变换：

\[
y = \mathrm{ctx}\,W_o^\top + b_o
\]

输出形状 `[B, S, E]`，与输入一致。

## 类定义 {#_9}

```cpp
class Attention : public Layer
{
public:
    Tensor Wq, Wk, Wv, Wo;     // [E, E] 投影权重
    Tensor bq, bk, bv, bo;     // [E]    偏置（可关）
#if TINY_AI_TRAINING_ENABLED
    Tensor dWq, dWk, dWv, dWo;
    Tensor dbq, dbk, dbv, dbo;
#endif

    Attention(int embed_dim, int num_heads, bool use_bias = true);

    Tensor forward (const Tensor &x) override;
    Tensor backward(const Tensor &grad_out) override;
    void   collect_params(std::vector<ParamGroup> &groups) override;
};
```

要求 `embed_dim % num_heads == 0`，`head_dim = embed_dim / num_heads`。

## 训练缓存 {#_10}

- `x_cache_`：原始输入，反向时用于计算 `dWq/k/v` 与 `dx`。
- `Q_cache_ / K_cache_ / V_cache_`：完整投影结果，反向中按头切分。
- `A_cache_`：attention 权重，形状 `[B*H, S, S]`，softmax 反向需要它。
- `ctx_cache_`：拼接后的上下文，反向 `Wo` 时使用。

## 反向传播步骤 {#_11}

1. **Wo 反传**
    - `dWo += ctx^T @ grad_out`，按 batch×seq 累加。
    - `dctx  = grad_out @ Wo`。
2. **多头注意力反传**（逐头）
    - `dV  = A^T @ dctx_h`
    - `dA  = dctx_h @ V^T`
    - 对 `softmax`：`dS = A * (dA - rowsum(dA*A)) * scale`
    - `dQ  = dS @ K`,  `dK = dS^T @ Q`，并把头 `h` 的梯度累加回全 `[B, S, E]`。
3. **Wq / Wk / Wv 反传**
    - 每个权重做 `dW += x^T @ dProj`、`dx += dProj @ W`。

## 使用示例 {#_12}

### Iris 上的小型 Transformer {#iris-transformer}

```cpp
const int SEQ_LEN = 4;       // 把 4 个特征当作 4 个 token
const int EMB_DIM = 8;
const int N_HEADS = 2;

Dense           embed_proj(IRIS_N_FEATURES, SEQ_LEN * EMB_DIM, true);
ActivationLayer embed_act(ActType::RELU);
Attention       attn(EMB_DIM, N_HEADS, true);
GlobalAvgPool   gap;
Dense           classifier(EMB_DIM, IRIS_N_CLASSES, true);

// forward
Tensor e0 = embed_proj.forward(X_batch);     // [B, 32]
Tensor e1 = embed_act.forward(e0);           // [B, 32]
e1.reshape_3d(B, SEQ_LEN, EMB_DIM);          // [B, 4, 8]
Tensor a0     = attn.forward(e1);            // [B, 4, 8]
Tensor p0     = gap.forward(a0);             // [B, 8]
Tensor logits = classifier.forward(p0);      // [B, 3]
```

完整代码见 [EXAMPLES/ATTENTION](../../EXAMPLES/ATTENTION/notes.md)。

## 资源消耗 {#_13}

- **参数量**：`4 * E^2 + 4 * E`（含偏置）。
- **激活内存**：`A_cache_` 大小 `B * H * S^2`，对长序列敏感（注意力的二次方复杂度）。
- **复杂度**：`O(B * H * S^2 * D + B * S * E^2)`。

!!! warning "ESP32-S3 实战建议"
    在 PSRAM 容量有限时，把 `seq_len` 控制在 64 以下，或拆分成多个小 attention 块。`embed_dim ≤ 64` + `num_heads = 2~4` 在 ESP32-S3 上可以跑得动。
