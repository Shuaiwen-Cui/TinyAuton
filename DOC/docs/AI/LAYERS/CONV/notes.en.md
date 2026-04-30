# Notes

!!! note "Notes"
    `tiny_conv` provides 1-D and 2-D convolutional layers. `Conv1D` is meant for time-series / signal data; `Conv2D` for images / feature maps. Both use He (Kaiming) normal initialisation (matched to ReLU-like activations) and cache the padded input for backward.

## SHAPE CONVENTIONS

| Layer | Input | Output |
| --- | --- | --- |
| Conv1D | `[batch, in_channels, length]` | `[batch, out_channels, out_length]` |
| Conv2D | `[batch, in_channels, height, width]` | `[batch, out_channels, out_height, out_width]` |

Output length:

\[
L_\text{out} = \left\lfloor\frac{L_\text{in} + 2P - K}{S}\right\rfloor + 1
\]

In 2-D the same formula applies independently to H and W.

## Conv1D

### Class definition

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

### Forward

\[
y_{b, oc, t} = \sum_{ic=0}^{C_\text{in}-1}\sum_{k=0}^{K-1}
W_{oc, ic, k}\cdot x_\text{pad}\!\big[b, ic, t S + k\big] + b_{oc}
\]

The implementation builds `xp` (zero-padded on both ends) and caches it in `x_cache_` for backward.

### Backward

- `dweight[oc, ic, k] += Σ_{b,t} grad_out[b, oc, t] · x_pad[b, ic, t·s + k]`
- `dbias[oc]         += Σ_{b,t} grad_out[b, oc, t]`
- `g_xp[b, ic, t·s + k] += Σ_{oc} grad_out[b, oc, t] · W[oc, ic, k]`, then strip the padding to get `g_x`.

### He init

\[
W \sim \mathcal{N}\!\left(0,\;\sqrt{\frac{2}{C_\text{in}\,K}}\right)
\]

Box–Muller transforms uniform samples into normal noise; bias is zero-initialised.

## Conv2D

### Class definition

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

Parameter naming: `kH/kW` kernel height/width, `sH/sW` strides, `pH/pW` zero padding.

### Forward

\[
y_{b, oc, h, w} = \sum_{ic, kh, kw}
W_{oc, ic, kh, kw}\cdot x_\text{pad}\!\big[b, ic,\,h s_H + kh,\,w s_W + kw\big] + b_{oc}
\]

### Backward

Mirrors Conv1D, looped over `(oh, ow)`. He init variance is `2 / (in_ch · kH · kW)`.

## MEMORY & PERFORMANCE

- **Param count**: `out_ch · in_ch · kH · kW (+ out_ch bias)`.
- **Complexity**: `O(B · out_ch · OH · OW · in_ch · kH · kW)`.
- **Runtime memory**: training requires `x_cache_` (padded input) plus `dweight` & `dbias`.
- **PSRAM**: when `out_ch · in_ch · K^d ≥ 32 KiB`, place `weight` / `dweight` in PSRAM.

## EXAMPLES

### 1-D signal classification

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

Or use the `CNN1D` wrapper (see [MODELS/CNN](../../MODELS/CNN/notes.md)).

### Image classification

```cpp
Conv2D c1(1, 16, 3, 3, 1, 1, 1, 1);   // 3×3 conv with same-padding
```

## LIMITATIONS

- **No dilation / groups**: classic dense convolution only. For depthwise / grouped variants, subclass `Layer`.
- **Padding**: zero-padding only, symmetric on H and W.
- **stride > kernel**: technically allowed but skips parts of the input; the typical case is `stride ≤ kernel`.
- **Training memory**: `x_cache_` stores the entire padded input — keep `B · in_ch · (H+2pH) · (W+2pW)` within budget on ESP32-S3.
