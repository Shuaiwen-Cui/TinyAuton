# TinyAI · 层 · 基类 — 设计说明 {#_1}

!!! info "实现依据与记录"
    本节接口以 `CODE/AIoTNode-TinyAuton-AI/middleware/` 为依据。源码摘录与串口输出包含历史记录；是否运行某项测试，请核对工程入口和启用开关。

!!! note "说明"
    `tiny_layer` 定义了所有神经网络层的抽象基类 `Layer`，并提供三个无参数的工具层：`ActivationLayer`、`Flatten`、`GlobalAvgPool`。这些工具层不需要可学习权重，但保持与 `Layer` 接口一致，便于在 `Sequential` 中直接堆叠。

!!! abstract "Layer — 所有神经网络层的基石"
    `tiny::Layer` 是抽象基类。理解它，你就理解了整个网络是如何连接的。

## 算法直觉 {#_2}

### 前向传播 = 数据流过网络 {#_3}

`forward()` 接收输入张量，经过层内的计算，输出下一个层需要的张量。

想象生产线：传感器读数 → 归一化 → 特征提取 → 分类器 → 输出结果。每个站就是一层。

### 反向传播（训练时） {#_4}

`backward()` 接收损失对输出的梯度，计算损失对输入的梯度（传到上一层）和对参数的梯度（更新权重）。

!!! tip "梯度沿着网络反向流动"
    最后一层从损失函数拿到梯度，逐层往前传。每层先算自己对参数的梯度（更新自己），再传梯度给前一层。

### 内置层组件 {#_5}

除了自定义层外，库提供三个无需训练参数的实用层：

| 层 | 功能 | 形状变化 |
|----|------|----------|
| **ActivationLayer** | 包装激活函数 | 形状不变 |
| **Flatten** | 将多维展平成 1D | `[B,C,L]→[B, C*L]` |
| **GlobalAvgPool** | 每个通道取平均 | `[B,C,L]→[B,C]` |

---

## Layer 抽象基类 {#layer}

```cpp
class Layer
{
public:
    const char *name;        // 用于 summary() 打印
    bool        trainable;   // 是否包含可学习参数

    explicit Layer(const char *name = "layer", bool trainable = false)
        : name(name), trainable(trainable) {}

    virtual ~Layer() {}

    virtual Tensor forward(const Tensor &x) = 0;

#if TINY_AI_TRAINING_ENABLED
    virtual Tensor backward(const Tensor &grad_out) = 0;
    virtual void   collect_params(std::vector<ParamGroup> &groups) {}
#endif
};
```

约定：

- **`forward(x)`**：必须返回新张量。
- **`backward(grad_out)`**：必须返回 `dL/dx`，并更新本层的 `dweight`、`dbias` 等成员（梯度累加）。
- **`collect_params()`**：仅可训练层需要重写，把 `(param, grad)` 压入 `groups`。
- **`trainable`**：用于 `Sequential::collect_params()` 提前剪掉不带参数的层（如 `Flatten`）。

## ActivationLayer {#activationlayer}

把无状态激活函数包成 `Layer`，可以直接在 `Sequential` 里堆。

```cpp
class ActivationLayer : public Layer
{
public:
    explicit ActivationLayer(ActType type, float alpha = 0.01f);
    Tensor forward (const Tensor &x) override;
    Tensor backward(const Tensor &grad_out) override;
};
```

实现细节：

- **缓存策略**：`forward()` 内部根据激活类型决定缓存输入还是输出
    - Sigmoid / Tanh / Softmax：缓存输出 `y`，反向时直接用。
    - ReLU / LeakyReLU / GELU：缓存输入 `x`。
- **alpha 默认 0.01**：仅对 LeakyReLU 生效。

## Flatten {#flatten}

把 `[batch, ...]` 形状压成 `[batch, flat]`：

```cpp
class Flatten : public Layer
{
    Tensor forward (const Tensor &x) override;   // [B, ...] → [B, B/size]
    Tensor backward(const Tensor &grad_out) override;
};
```

- 计算 `flat = size / batch`，调用 `reshape_2d`。
- 反向时把梯度还原回原始 `(in_ndim, in_shape)`。
- `Flatten` 不保留权重，`trainable = false`，因此不参与 `collect_params`。

CNN1D 的标准用法：

```txt
Conv1D + ReLU + MaxPool1D × N → Flatten → Dense → Softmax
```

`tiny_cnn.cpp` 在构造时就把 `Flatten` 接在卷积块之后。

## GlobalAvgPool {#globalavgpool}

跨序列维度求均值，常用于 Transformer 风格输出（把 `[B, S, F]` 聚合成 `[B, F]`）：

```cpp
class GlobalAvgPool : public Layer
{
    Tensor forward (const Tensor &x) override;   // [B, S, F] → [B, F]
    Tensor backward(const Tensor &grad_out) override;
};
```

- 前向：对 `s` 维求平均。
- 反向：把 `grad_out` 平均分配到 `S` 个序列位置。

`example_attention.cpp` 中的串接：

```txt
Attention([B, S, E]) → GlobalAvgPool([B, E]) → Dense([B, num_classes])
```

## 与 Sequential 的关系 {#sequential}

```cpp
Sequential m;
m.add(new Dense(F, H));
m.add(new ActivationLayer(ActType::RELU));
m.add(new Dense(H, C));
m.add(new ActivationLayer(ActType::SOFTMAX));
```

- `Sequential` 拥有所有用 `add()` 注册的 `Layer*`，析构时统一 `delete`。
- `forward(x)` 顺序调用每层的 `forward`；`backward(grad_out)` 倒序调用每层的 `backward`。
- 仅 `trainable == true` 的层会被 `Sequential::collect_params()` 调用。

## 自定义层 {#_6}

继承 `Layer`，实现 `forward / backward / collect_params` 即可。例：

```cpp
class Scale : public Layer
{
public:
    Tensor scale;   // [feat]
#if TINY_AI_TRAINING_ENABLED
    Tensor dscale;
#endif

    Scale(int feat) : Layer("scale", true), scale(feat)
    {
        scale.fill(1.0f);
#if TINY_AI_TRAINING_ENABLED
        dscale = Tensor::zeros_like(scale);
#endif
    }

    Tensor forward(const Tensor &x) override
    {
        Tensor out = x.clone();
        // ...逐元素乘 scale...
        return out;
    }

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override { /* ... */ return grad_out; }
    void collect_params(std::vector<ParamGroup> &g) override
    {
        g.push_back({&scale, &dscale});
    }
#endif
};
```
