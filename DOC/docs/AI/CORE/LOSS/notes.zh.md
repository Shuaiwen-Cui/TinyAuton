# 说明

!!! note "说明"
    `tiny_loss` 提供 4 种常见损失函数：均方误差 MSE、平均绝对误差 MAE、Softmax + Cross-Entropy、Binary Cross-Entropy。每个损失同时提供前向标量值与反向梯度张量。

## LossType 枚举

```cpp
enum class LossType
{
    MSE = 0,           // 均方误差
    MAE,               // 平均绝对误差
    CROSS_ENTROPY,     // Softmax + 交叉熵（输入为 logits）
    BINARY_CE          // 二分类交叉熵（输入为 sigmoid 概率）
};
```

## 数学定义

### MSE — 均方误差

\[
L = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2,\quad
\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{N}(\hat{y}_i - y_i)
\]

### MAE — 平均绝对误差

\[
L = \frac{1}{N} \sum_i |\hat{y}_i - y_i|,\quad
\frac{\partial L}{\partial \hat{y}_i} = \frac{1}{N}\,\mathrm{sign}(\hat{y}_i - y_i)
\]

### Cross-Entropy — 交叉熵（带数值稳定 Softmax）

`cross_entropy_forward` 期待原始 logits，内部用 log-sum-exp 技巧计算每行的负对数似然：

\[
L_b = -\big(\mathrm{logits}_{b, y_b} - m_b\big) + \log\!\Big(\sum_j e^{\mathrm{logits}_{b,j} - m_b}\Big),\;
m_b = \max_j \mathrm{logits}_{b,j}
\]

\[
L = \frac{1}{B} \sum_b L_b
\]

反向梯度恰好等于 `softmax(logits) - one_hot(labels)`，再除以 batch 大小：

\[
\frac{\partial L}{\partial \mathrm{logits}_{b,j}} = \frac{1}{B}\big(\mathrm{softmax}(\mathrm{logits})_{b,j} - \mathbb{1}[j = y_b]\big)
\]

!!! warning "标签格式"
    `cross_entropy_*` 的 labels 是 `int*`，长度等于 batch，每个元素是类别下标 `[0, num_classes)`，不是 one-hot 张量。

### Binary CE — 二分类交叉熵

输入是已经过 sigmoid 的概率 `pred ∈ (0, 1)`，target 是 0/1：

\[
L = -\frac{1}{N} \sum_i \big[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\big]
\]

数值稳定性：在 `log` 内对分子加 `TINY_MATH_MIN_POSITIVE_INPUT_F32`，避免 `log(0)`。

## API 概览

```cpp
float  mse_forward          (const Tensor &pred, const Tensor &target);
Tensor mse_backward         (const Tensor &pred, const Tensor &target);

float  mae_forward          (const Tensor &pred, const Tensor &target);
Tensor mae_backward         (const Tensor &pred, const Tensor &target);

float  cross_entropy_forward (const Tensor &logits, const int *labels);
Tensor cross_entropy_backward(const Tensor &logits, const int *labels);

float  binary_ce_forward    (const Tensor &pred, const Tensor &target);
Tensor binary_ce_backward   (const Tensor &pred, const Tensor &target);
```

### Dispatch 助手

```cpp
float  loss_forward (const Tensor &pred, const Tensor &target,
                     LossType type, const int *labels = nullptr);

Tensor loss_backward(const Tensor &pred, const Tensor &target,
                     LossType type, const int *labels = nullptr);
```

`Trainer` 内部就是通过 `loss_forward / loss_backward` 加 `LossType` 配置项实现损失可插拔。

## 使用建议

| 场景 | 推荐损失 | 模型最后一层 |
| --- | --- | --- |
| 多类分类 | `CROSS_ENTROPY` | Dense 输出 raw logits（通常省略 Softmax，因为损失函数内部已计算） |
| 二分类 | `BINARY_CE` | Dense + Sigmoid |
| 回归 | `MSE` | Dense |
| 对异常值鲁棒的回归 | `MAE` | Dense |

!!! tip "Softmax 与 Cross-Entropy 的关系"
    `cross_entropy_forward` 已经包含 Softmax，所以模型最后一层用 `ActType::LINEAR`（或干脆不加激活）即可。`MLP` / `CNN1D` 默认开启 `use_softmax`，主要是为了 `predict()` / `accuracy()` 中读取概率，可视情况关掉。
