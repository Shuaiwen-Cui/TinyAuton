# 说明

!!! note "说明"
    `tiny_conv` 提供 1-D 与 2-D 卷积层。`Conv1D` 适用于时序 / 信号数据，`Conv2D` 适用于图像 / 特征图。两者都使用 He（Kaiming）正态初始化以匹配 ReLU-类激活，并在训练阶段缓存填充后的输入用于反向传播。

## 形状约定

| 层 | 输入 | 输出 |
| --- | --- | --- |
| Conv1D | `[batch, in_channels, length]` | `[batch, out_channels, out_length]` |
| Conv2D | `[batch, in_channels, height, width]` | `[batch, out_channels, out_height, out_width]` |

输出长度公式：

\[
L_\text{out} = \left\lfloor\frac{L_\text{in} + 2P - K}{S}\right\rfloor + 1
\]

二维同时对 H 和 W 应用上述公式。

## Conv1D

### 类定义

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

### 前向

\[
y_{b, oc, t} = \sum_{ic=0}^{C_\text{in}-1}\sum_{k=0}^{K-1}
W_{oc, ic, k}\cdot x_\text{pad}\!\big[b, ic, t S + k\big] + b_{oc}
\]

实现首先构造 `xp`（在两端各 padding 0），保存在 `x_cache_` 中以备反向。

### 反向

- `dweight[oc, ic, k] += Σ_{b,t} grad_out[b, oc, t] · x_pad[b, ic, t·s + k]`
- `dbias[oc]         += Σ_{b,t} grad_out[b, oc, t]`
- `g_xp[b, ic, t·s + k] += Σ_{oc} grad_out[b, oc, t] · W[oc, ic, k]`，最后剥离 padding 得 `g_x`。

### He 初始化

\[
W \sim \mathcal{N}\!\left(0,\;\sqrt{\frac{2}{C_\text{in}\,K}}\right)
\]

实现使用 Box-Muller 变换从均匀分布生成正态噪声，bias 初始化为 0。

## Conv2D

### 类定义

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

### 前向

\[
y_{b, oc, h, w} = \sum_{ic, kh, kw}
W_{oc, ic, kh, kw}\cdot x_\text{pad}\!\big[b, ic,\,h s_H + kh,\,w s_W + kw\big] + b_{oc}
\]

### 反向

公式与 Conv1D 类似，逐 `(oh, ow)` 累加。He 初始化方差为 `2 / (in_ch · kH · kW)`。

## 内存与性能

- **参数量**：`out_ch · in_ch · kH · kW (+ out_ch bias)`。
- **复杂度**：`O(B · out_ch · OH · OW · in_ch · kH · kW)`。
- **运行内存**：训练时需要存 `x_cache_`（填充后的输入）+ `dweight` 与 `dbias`。
- **PSRAM 建议**：当 `out_ch · in_ch · K^d ≥ 32 KiB` 时，建议把 `weight` / `dweight` 放到 PSRAM。

## 使用示例

### 时序信号分类

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

### 图像分类

```cpp
Conv2D c1(1, 16, 3, 3, 1, 1, 1, 1);   // 3×3 conv with same-padding
```

## 局限与注意事项

- **无 dilation / groups**：当前实现是经典的密集卷积；如需 depthwise / grouped 卷积，可继承 `Layer` 自行扩展。
- **Padding 模式**：仅支持零填充，且 `H / W` 两侧对称填充。
- **stride > kernel**：理论可设，但会跳过部分输入；典型用法仍为 `stride ≤ kernel`。
- **训练显存**：`x_cache_` 是填充后的整个输入张量，对 ESP32-S3 而言要确保 `B · in_ch · (H+2pH) · (W+2pW)` 在预算内。
