# TinyAI · 核心 · 损失函数 — 设计说明 {#_1}

!!! info "当前输入约定"
    当前 CrossEntropy 接口接受 logits，内部归一化为概率。下方概率公式用于解释损失含义，不表示调用时应传入已做 Softmax 的预测。

!!! info "实现依据与记录"
    本节接口以 `CODE/AIoTNode-TinyAuton-AI/middleware/` 为依据。源码摘录与串口输出包含历史记录；是否运行某项测试，请核对工程入口和启用开关。

!!! note "说明"
    `tiny_loss` 提供 4 种常见损失函数：均方误差 MSE、平均绝对误差 MAE、Softmax + Cross-Entropy、Binary Cross-Entropy。每个损失同时提供前向标量值与反向梯度张量。

!!! abstract "Loss — 衡量预测和真实的差距"
    损失函数告诉你模型当前做得有多差。目标就是**最小化这个值**。

## 算法直觉 {#_2}

### 损失 = 惩罚 {#_3}

模型输出 \(\hat{y}\)，真实值 \(y\)。损失函数 \(L(\hat{y}, y)\) 给出一个数字：

- **0** = 完美预测
- **越大** = 错得越离谱

### 常见损失函数 {#_4}

| 损失 | 公式 | 适用场景 | 为什么这么设计 |
|------|------|----------|----------------|
| MSE | \((\hat{y} - y)^2\) 的均值 | **回归**（数值预测） | 平方惩罚大误差——偏离 2 倍代价是 4 倍 |
| MAE | \(|\hat{y} - y|\) 的均值 | 回归（对异常值鲁棒） | 绝对值惩罚比平方温和 |
| CrossEntropy | \(-\sum y \cdot \log(\hat{y})\) | **分类**（概率输出） | 预测概率偏离真实标签越多惩罚越大 |

!!! tip "直觉理解 CrossEntropy"
    假设真实标签是猫（`[1,0,0]`），模型预测 `[0.8, 0.1, 0.1]` → 损失小；预测 `[0.3, 0.6, 0.1]` → 损失很大。**模型很确定但错了，惩罚最重**。

### 梯度 = 下山的方向 {#_5}

损失函数的梯度 \(\partial L / \partial \text{params}\) 告诉优化器：

- **方向**：参数该往哪个方向调才能降低损失
- **大小**：多快会变化（步长的参考）

这就是反向传播和梯度下降的核心。

---

## LossType 枚举 {#losstype}

```cpp
enum class LossType
{
    MSE = 0,           // 均方误差
    MAE,               // 平均绝对误差
    CROSS_ENTROPY,     // Softmax + 交叉熵（输入为 logits）
    BINARY_CE          // 二分类交叉熵（输入为 sigmoid 概率）
};
```

## 数学定义 {#_6}

### MSE — 均方误差 {#mse}

\[
L = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2,\quad
\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{N}(\hat{y}_i - y_i)
\]

### MAE — 平均绝对误差 {#mae}

\[
L = \frac{1}{N} \sum_i |\hat{y}_i - y_i|,\quad
\frac{\partial L}{\partial \hat{y}_i} = \frac{1}{N}\,\mathrm{sign}(\hat{y}_i - y_i)
\]

### Cross-Entropy — 交叉熵（带数值稳定 Softmax） {#cross-entropy-softmax}

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

### Binary CE — 二分类交叉熵 {#binary-ce}

输入是已经过 sigmoid 的概率 `pred ∈ (0, 1)`，target 是 0/1：

\[
L = -\frac{1}{N} \sum_i \big[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\big]
\]

数值稳定性：在 `log` 内对分子加 `TINY_MATH_MIN_POSITIVE_INPUT_F32`，避免 `log(0)`。

## API 概览 {#api}

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

### Dispatch 助手 {#dispatch}

```cpp
float  loss_forward (const Tensor &pred, const Tensor &target,
                     LossType type, const int *labels = nullptr);

Tensor loss_backward(const Tensor &pred, const Tensor &target,
                     LossType type, const int *labels = nullptr);
```

`Trainer` 内部就是通过 `loss_forward / loss_backward` 加 `LossType` 配置项实现损失可插拔。

## 使用建议 {#_7}

| 场景 | 推荐损失 | 模型最后一层 |
| --- | --- | --- |
| 多类分类 | `CROSS_ENTROPY` | Dense 输出 raw logits（通常省略 Softmax，因为损失函数内部已计算） |
| 二分类 | `BINARY_CE` | Dense + Sigmoid |
| 回归 | `MSE` | Dense |
| 对异常值鲁棒的回归 | `MAE` | Dense |

!!! tip "Softmax 与 Cross-Entropy 的关系"
    `cross_entropy_forward` 已经包含 Softmax，所以模型最后一层用 `ActType::LINEAR`（或干脆不加激活）即可。`MLP` / `CNN1D` 默认开启 `use_softmax`，主要是为了 `predict()` / `accuracy()` 中读取概率，可视情况关掉。
