# 说明

!!! note "说明"
    `tiny_activation` 提供 7 种常用激活函数的前向 / 反向 / 原地实现，全部在 `tiny::Tensor` 上工作。Softmax 沿最后一维做数值稳定的归一化，可直接用于分类网络的输出层。

## ActType 枚举

```cpp
enum class ActType
{
    RELU = 0,        // max(0, x)
    LEAKY_RELU,      // x > 0 ? x : alpha*x
    SIGMOID,         // 1 / (1 + exp(-x))
    TANH,            // tanh(x)
    SOFTMAX,         // exp(xi) / sum(exp(xj))，沿最后一维
    GELU,            // 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    LINEAR           // 恒等
};
```

## 数学定义

| 激活 | 前向 | 反向 |
| --- | --- | --- |
| ReLU | \( y = \max(0, x) \) | \( \frac{dL}{dx} = \frac{dL}{dy} \cdot \mathbb{1}[x > 0] \) |
| Leaky ReLU | \( y = x \cdot (x > 0) + \alpha x \cdot (x \le 0) \) | \( \frac{dL}{dx} = \frac{dL}{dy} \cdot (\mathbb{1}[x>0] + \alpha \mathbb{1}[x \le 0]) \) |
| Sigmoid | \( y = \frac{1}{1 + e^{-x}} \) | \( \frac{dL}{dx} = \frac{dL}{dy} \cdot y(1-y) \) |
| Tanh | \( y = \tanh(x) \) | \( \frac{dL}{dx} = \frac{dL}{dy} \cdot (1 - y^2) \) |
| Softmax | \( y_i = \frac{e^{x_i - \max}}{\sum_j e^{x_j - \max}} \) | \( \frac{dL}{dx_i} = y_i \left(\frac{dL}{dy_i} - \sum_j \frac{dL}{dy_j} y_j\right) \) |
| GELU | \( y = 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715 x^3)\right)\right) \) | 数值微分（实现见 `gelu_backward`） |

!!! tip "Softmax 数值稳定性"
    实现先对每行求最大值并相减，再做 `exp` 与归一化，等价于 \(\operatorname{softmax}(x)\) 但避免溢出。归一化时分母加 `TINY_MATH_MIN_DENOMINATOR` 防止除零。

## API 概览

### 前向（返回新张量）

```cpp
Tensor relu_forward       (const Tensor &x);
Tensor leaky_relu_forward (const Tensor &x, float alpha = 0.01f);
Tensor sigmoid_forward    (const Tensor &x);
Tensor tanh_forward       (const Tensor &x);
Tensor softmax_forward    (const Tensor &x);
Tensor gelu_forward       (const Tensor &x);
```

每个 `*_forward` 内部都 `clone()` 输入再调用对应的 `*_inplace`。

### 原地版本（直接修改 x）

```cpp
void relu_inplace       (Tensor &x);
void leaky_relu_inplace (Tensor &x, float alpha = 0.01f);
void sigmoid_inplace    (Tensor &x);
void tanh_inplace       (Tensor &x);
void softmax_inplace    (Tensor &x);
void gelu_inplace       (Tensor &x);
```

适合显存敏感、不需要保留输入的场景（推理流水线常用）。

### 反向（仅在 `TINY_AI_TRAINING_ENABLED` 时编译）

```cpp
Tensor relu_backward       (const Tensor &x, const Tensor &grad_out);
Tensor leaky_relu_backward (const Tensor &x, const Tensor &grad_out, float alpha = 0.01f);
Tensor sigmoid_backward    (const Tensor &y, const Tensor &grad_out);   // 传 forward 的输出
Tensor tanh_backward       (const Tensor &y, const Tensor &grad_out);   // 传 forward 的输出
Tensor softmax_backward    (const Tensor &y, const Tensor &grad_out);   // 传 forward 的输出
Tensor gelu_backward       (const Tensor &x, const Tensor &grad_out);
```

!!! warning "反向函数的 cache"
    - **ReLU / LeakyReLU / GELU**：传入 `x`（forward 的输入）。
    - **Sigmoid / Tanh / Softmax**：传入 `y`（forward 的输出），避免重新求 sigmoid。
    `ActivationLayer::forward()` 会自动按这个规则 `cache_` 正确的张量。

### Dispatch 助手

```cpp
Tensor act_forward (const Tensor &x, ActType type, float alpha = 0.01f);
void   act_inplace (Tensor &x,       ActType type, float alpha = 0.01f);
Tensor act_backward(const Tensor &cache, const Tensor &grad_out,
                    ActType type, float alpha = 0.01f);
```

按枚举值 dispatch，便于从配置文件 / 模型构造参数中传入。

## 常见用法

```cpp
// 直接使用函数式 API
Tensor h = tanh_forward(x);

// 用 ActType + dispatch 实现可插拔
ActType act = ActType::GELU;
Tensor y = act_forward(x, act);

// 与 ActivationLayer 配合（Sequential 网络中）
Sequential m;
m.add(new Dense(in, hid));
m.add(new ActivationLayer(ActType::RELU));
```

## 适用场景

- **隐藏层激活**：ReLU / LeakyReLU / GELU。
- **概率输出**：Sigmoid（二分类）、Softmax（多分类）。
- **Transformer 风格 MLP**：GELU。
- **饱和归一**：Tanh。
- **跳过激活**：LINEAR（恒等），常用于 `LossType::CROSS_ENTROPY` 接收 raw logits 的回归 / 分类。
