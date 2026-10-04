# TinyAI · Layers · Conv — Design notes {#notes}

<!-- Original section links retained for compatibility. -->
<span id="1-d-signal-classification"></span>

!!! info "Implementation and records"
    APIs in this section are based on `CODE/AIoTNode-TinyAuton-AI/middleware/`. Source excerpts and serial output include historical records; check the project entry and enabled selectors before reproducing a test.

!!! note "Notes"
    `tiny_conv` provides 1-D and 2-D convolutional layers. `Conv1D` is meant for time-series / signal data; `Conv2D` for images / feature maps. Both use He (Kaiming) normal initialisation (matched to ReLU-like activations) and cache the padded input for backward.

!!! abstract "Conv — Scanning the Signal with a Template"
    One convolution kernel = one feature detector, sliding across the signal/image.

## Intuition {#intuition}

### Conv1D = Sliding Template Matching {#conv1d-sliding-template-matching}

Imagine you have a template (e.g., an ECG QRS complex). You slide it across the signal, computing a dot product at each position. High dot product → template matches here.

**Weight sharing**: the same kernel scans the entire signal. No matter where a pattern appears, it's detected the same way.

### Key Parameters {#key-parameters}

| Parameter | Meaning | Intuition |
|-----------|---------|-----------|
| `in_channels` | Input signal channels | 3-ch accelerometer = 3 |
| `out_channels` | Features to learn | 32 = 32 pattern detectors |
| `kernel_size` | Template width | 9 = looks at 9 points at once |
| `stride` | Step size | 2 = every other point |
| `padding` | Zero padding at edges | 1 = keep output length same |

!!! tip "Conv1D is natural for SHM"
    Acceleration signals are 1-D time series. Multiple kernels learn different frequency bands/modes.

### Shape Change {#shape-change}

```
Input [B, in_ch, L]  →  Output [B, out_ch, (L + 2*pad - kernel)/stride + 1]
```

---

## SHAPE CONVENTIONS {#shape-conventions}

| Layer | Input | Output |
| --- | --- | --- |
| Conv1D | `[batch, in_channels, length]` | `[batch, out_channels, out_length]` |
| Conv2D | `[batch, in_channels, height, width]` | `[batch, out_channels, out_height, out_width]` |

Output length:

\[
L_\text{out} = \left\lfloor\frac{L_\text{in} + 2P - K}{S}\right\rfloor + 1
\]

In 2-D the same formula applies independently to H and W.

## Conv1D {#conv1d}

### Class definition {#class-definition}

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

### Forward {#forward}

\[
y_{b, oc, t} = \sum_{ic=0}^{C_\text{in}-1}\sum_{k=0}^{K-1}
W_{oc, ic, k}\cdot x_\text{pad}\!\big[b, ic, t S + k\big] + b_{oc}
\]

The implementation builds `xp` (zero-padded on both ends) and caches it in `x_cache_` for backward.

### Backward {#backward}

- `dweight[oc, ic, k] += Σ_{b,t} grad_out[b, oc, t] · x_pad[b, ic, t·s + k]`
- `dbias[oc]         += Σ_{b,t} grad_out[b, oc, t]`
- `g_xp[b, ic, t·s + k] += Σ_{oc} grad_out[b, oc, t] · W[oc, ic, k]`, then strip the padding to get `g_x`.

### He init {#he-init}

\[
W \sim \mathcal{N}\!\left(0,\;\sqrt{\frac{2}{C_\text{in}\,K}}\right)
\]

Box–Muller transforms uniform samples into normal noise; bias is zero-initialised.

## Conv2D {#conv2d}

### Class definition {#class-definition_1}

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

### Forward {#forward_1}

\[
y_{b, oc, h, w} = \sum_{ic, kh, kw}
W_{oc, ic, kh, kw}\cdot x_\text{pad}\!\big[b, ic,\,h s_H + kh,\,w s_W + kw\big] + b_{oc}
\]

### Backward {#backward_1}

Mirrors Conv1D, looped over `(oh, ow)`. He init variance is `2 / (in_ch · kH · kW)`.

## MEMORY & PERFORMANCE {#memory-performance}

- **Param count**: `out_ch · in_ch · kH · kW (+ out_ch bias)`.
- **Complexity**: `O(B · out_ch · OH · OW · in_ch · kH · kW)`.
- **Runtime memory**: training requires `x_cache_` (padded input) plus `dweight` & `dbias`.
- **PSRAM**: when `out_ch · in_ch · K^d ≥ 32 KiB`, place `weight` / `dweight` in PSRAM.

## EXAMPLES {#examples}

### 1-D signal classification {#1-d-signal-classification}

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

### Image classification {#image-classification}

```cpp
Conv2D c1(1, 16, 3, 3, 1, 1, 1, 1);   // 3×3 conv with same-padding
```

## LIMITATIONS {#limitations}

- **No dilation / groups**: classic dense convolution only. For depthwise / grouped variants, subclass `Layer`.
- **Padding**: zero-padding only, symmetric on H and W.
- **stride > kernel**: technically allowed but skips parts of the input; the typical case is `stride ≤ kernel`.
- **Training memory**: `x_cache_` stores the entire padded input — keep `B · in_ch · (H+2pH) · (W+2pW)` within budget on ESP32-S3.
