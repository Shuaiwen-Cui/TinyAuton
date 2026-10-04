# TinyAI · 层 · 卷积层 — 设计说明 {#_1}

!!! info "实现依据与记录"
    本节接口以 `CODE/AIoTNode-TinyAuton-AI/middleware/` 为依据。源码摘录与串口输出包含历史记录；是否运行某项测试，请核对工程入口和启用开关。

!!! note "说明"
    `tiny_conv` 提供 1-D 与 2-D 卷积层。`Conv1D` 适用于时序 / 信号数据，`Conv2D` 适用于图像 / 特征图。两者都使用 He（Kaiming）正态初始化以匹配 ReLU-类激活，并在训练阶段缓存填充后的输入用于反向传播。

!!! abstract "Conv — 卷积：用模板扫描信号"
    一个卷积核 = 一个特征检测器，在信号/图像上滑动扫描，在每处报告匹配强度。

## 算法直觉 {#_2}

### Conv1D = 滑动窗口模板匹配 {#conv1d}

想象你有一个模板（心电图的 QRS 波），你在信号上从左到右滑动它，在每个位置计算点积。点积高 → 这里和模板很像。

**权重共享**：同一个卷积核扫遍整个信号，所以不管波形出现在信号的哪个位置，都能被同一种方式检测到。

### 关键参数 {#_3}

| 参数 | 含义 | 直觉 |
|------|------|------|
| `in_channels` | 输入有几路信号 | 3 通道加速度计 = 3 |
| `out_channels` | 想学几种特征 | 32 = 32 种不同波形检测器 |
| `kernel_size` | 模板有多宽 | 9 = 一次看 9 个采样点 |
| `stride` | 每次滑几步 | 2 = 每隔一个点看一次 |
| `padding` | 边界补零 | 1 = 让输出长度和输入一致 |

!!! tip "Conv1D 适用于 SHM"
    加速度信号是一维时序，Conv1D 是自然的选择。多个卷积核可以学习不同频段/模态的振动模式。

### 形状变化 {#_4}

```
输入 [B, in_ch, L]  →  输出 [B, out_ch, (L + 2*pad - kernel)/stride + 1]
```

---

## 形状约定 {#_5}

| 层 | 输入 | 输出 |
| --- | --- | --- |
| Conv1D | `[batch, in_channels, length]` | `[batch, out_channels, out_length]` |
| Conv2D | `[batch, in_channels, height, width]` | `[batch, out_channels, out_height, out_width]` |

输出长度公式：

\[
L_\text{out} = \left\lfloor\frac{L_\text{in} + 2P - K}{S}\right\rfloor + 1
\]

二维同时对 H 和 W 应用上述公式。

## Conv1D {#conv1d_1}

### 类定义 {#_6}

```cpp
class Conv1D : public Layer
{
public:
    Tensor weight;   // [out_ch, in_ch, kernel]
    Tensor bias;     // [out_ch]
#if TINY_AI_TRAINING_ENABLED
    Tensor dweight;
    Tensor dbias;
#endif

    Conv1D(int in_channels, int out_channels, int kernel_size,
           int stride = 1, int padding = 0, bool use_bias = true);

    Tensor forward (const Tensor &x) override;
    Tensor backward(const Tensor &grad_out) override;
    void   collect_params(std::vector<ParamGroup> &groups) override;
};
```

### 前向 {#_7}

\[
y_{b, oc, t} = \sum_{ic=0}^{C_\text{in}-1}\sum_{k=0}^{K-1}
W_{oc, ic, k}\cdot x_\text{pad}\!\big[b, ic, t S + k\big] + b_{oc}
\]

实现首先构造 `xp`（在两端各 padding 0），保存在 `x_cache_` 中以备反向。

### 反向 {#_8}

- `dweight[oc, ic, k] += Σ_{b,t} grad_out[b, oc, t] · x_pad[b, ic, t·s + k]`
- `dbias[oc]         += Σ_{b,t} grad_out[b, oc, t]`
- `g_xp[b, ic, t·s + k] += Σ_{oc} grad_out[b, oc, t] · W[oc, ic, k]`，最后剥离 padding 得 `g_x`。

### He 初始化 {#he}

\[
W \sim \mathcal{N}\!\left(0,\;\sqrt{\frac{2}{C_\text{in}\,K}}\right)
\]

实现使用 Box-Muller 变换从均匀分布生成正态噪声，bias 初始化为 0。

## Conv2D {#conv2d}

### 类定义 {#_9}

```cpp
class Conv2D : public Layer
{
public:
    Tensor weight;   // [out_ch, in_ch, kH, kW]
    Tensor bias;     // [out_ch]
#if TINY_AI_TRAINING_ENABLED
    Tensor dweight;
    Tensor dbias;
#endif

    Conv2D(int in_channels, int out_channels, int kH, int kW,
           int sH = 1, int sW = 1, int pH = 0, int pW = 0,
           bool use_bias = true);

    Tensor forward (const Tensor &x) override;
    Tensor backward(const Tensor &grad_out) override;
    void   collect_params(std::vector<ParamGroup> &groups) override;
};
```

参数命名约定：`kH/kW` 为核高 / 核宽，`sH/sW` 为步幅，`pH/pW` 为零填充。

### 前向 {#_10}

\[
y_{b, oc, h, w} = \sum_{ic, kh, kw}
W_{oc, ic, kh, kw}\cdot x_\text{pad}\!\big[b, ic,\,h s_H + kh,\,w s_W + kw\big] + b_{oc}
\]

### 反向 {#_11}

公式与 Conv1D 类似，逐 `(oh, ow)` 累加。He 初始化方差为 `2 / (in_ch · kH · kW)`。

## 内存与性能 {#_12}

- **参数量**：`out_ch · in_ch · kH · kW (+ out_ch bias)`。
- **复杂度**：`O(B · out_ch · OH · OW · in_ch · kH · kW)`。
- **运行内存**：训练时需要存 `x_cache_`（填充后的输入）+ `dweight` 与 `dbias`。
- **PSRAM 建议**：当 `out_ch · in_ch · K^d ≥ 32 KiB` 时，建议把 `weight` / `dweight` 放到 PSRAM。

## 使用示例 {#_13}

### 时序信号分类 {#_14}

```cpp
Sequential m;
m.add(new Conv1D(1, 8, 5));                      // [B,1,L] → [B,8,L-4]
m.add(new ActivationLayer(ActType::RELU));
m.add(new MaxPool1D(2));                         // [B,8,(L-4)/2]
m.add(new Conv1D(8, 16, 3));                     // [B,16,...]
m.add(new ActivationLayer(ActType::RELU));
m.add(new MaxPool1D(2));
m.add(new Flatten());
m.add(new Dense(flat_dim, 32));
m.add(new ActivationLayer(ActType::RELU));
m.add(new Dense(32, num_classes));
m.add(new ActivationLayer(ActType::SOFTMAX));
```

或直接用 `CNN1D` 便捷封装（详见 [MODELS/CNN](../../MODELS/CNN/notes.md)）。

### 图像分类 {#_15}

```cpp
Conv2D c1(1, 16, 3, 3, 1, 1, 1, 1);   // 3×3 conv with same-padding
```

## 局限与注意事项 {#_16}

- **无 dilation / groups**：当前实现是经典的密集卷积；如需 depthwise / grouped 卷积，可继承 `Layer` 自行扩展。
- **Padding 模式**：仅支持零填充，且 `H / W` 两侧对称填充。
- **stride > kernel**：理论可设，但会跳过部分输入；典型用法仍为 `stride ≤ kernel`。
- **训练显存**：`x_cache_` 是填充后的整个输入张量，对 ESP32-S3 而言要确保 `B · in_ch · (H+2pH) · (W+2pW)` 在预算内。
