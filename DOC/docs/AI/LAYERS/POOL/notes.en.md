# TinyAI · Layers · Pool — Design notes {#notes}

!!! info "Implementation and records"
    APIs in this section are based on `CODE/AIoTNode-TinyAuton-AI/middleware/`. Source excerpts and serial output include historical records; check the project entry and enabled selectors before reproducing a test.

!!! note "Notes"
    `tiny_pool` ships 1-D and 2-D max-/avg-pool layers. Pooling is independent across channels and does not introduce learnable parameters; during training, only MaxPool stores argmax positions for backward.

!!! abstract "Pool — Local Downsampling: Keep Key Info, Reduce Data"
    Take max or average within a local window, shrinking the feature map.

## Intuition {#intuition}

| Pool type | Value | Meaning | Effect |
|-----------|-------|---------|--------|
| **MaxPool** | Max in window | "Strongest response here?" | Retains strongest features, shift invariance |
| **AvgPool** | Average in window | "Overall response here?" | Retains trend, smooths features |

### Why Pool? {#why-pool}

1. **Less compute**: smaller feature maps → fewer params
2. **Larger receptive field**: later convs see wider context
3. **Shift invariance**: slight shift may still pick same max

### Shape Change {#shape-change}

```
Input [B, C, L]  →  Output [B, C, (L - kernel)/stride + 1]
```

---

## SHAPE CONVENTIONS {#shape-conventions}

| Layer | Input | Output |
| --- | --- | --- |
| MaxPool1D / AvgPool1D | `[batch, channels, length]` | `[batch, channels, (L - K) / S + 1]` |
| MaxPool2D / AvgPool2D | `[batch, channels, height, width]` | `[batch, channels, (H - kH) / sH + 1, (W - kW) / sW + 1]` |

The constructor's `stride` defaults to `kernel` (i.e. non-overlapping pooling).

## MaxPool1D {#maxpool1d}

```cpp
class MaxPool1D : public Layer
{
    explicit MaxPool1D(int kernel_size, int stride = -1);
    Tensor forward (const Tensor &x) override;
    Tensor backward(const Tensor &grad_out) override;
};
```

### Forward {#forward}

```cpp
out[b, c, l] = max(x[b, c, l*S + 0..K-1])
```

The absolute argmax index is recorded in `mask_[b, c, l]` (training only).

### Backward {#backward}

`grad_out[b, c, l]` is written to `g[b, c, mask_[b, c, l]]` (gradient flows only into the max position).

## AvgPool1D {#avgpool1d}

```cpp
class AvgPool1D : public Layer { ... };
```

### Forward {#forward_1}

```cpp
out[b, c, l] = (1/K) * Σ_k x[b, c, l*S + k]
```

### Backward {#backward_1}

Distribute `grad_out / K` evenly to the K positions in the receptive field.

## MaxPool2D {#maxpool2d}

```cpp
class MaxPool2D : public Layer
{
    MaxPool2D(int kH, int kW, int sH = -1, int sW = -1);
};
```

`mask_` has shape `[B, C, OH, OW * 2]`, packing `(ih, iw)` as two consecutive floats (saves a reshape). Backward unpacks the same layout and writes `grad_out` to the original max position.

## AvgPool2D {#avgpool2d}

Mirrors AvgPool1D: forward averages `kH * kW` elements; backward broadcasts `grad_out / (kH * kW)` back.

## NOTES {#notes_1}

- **`stride` defaults to `kernel`** → non-overlapping pooling (most common).
- **MaxPool gradients are sparse**: only the max position receives gradient (winner-take-all), often desirable for feature selection.
- **AvgPool gradients are uniform**: more stable but less selective; commonly used for global average pooling (see `GlobalAvgPool` under [LAYERS/BASE](../BASE/notes.md)).
- **No padding**: the implementation assumes `(L - K)` and `(H/W - kH/kW)` are divisible by `stride`. Callers must align shapes; otherwise edge elements are silently dropped.

## CONV + POOL RECIPE {#conv-pool-recipe}

A standard CNN1D block looks like:

```txt
Conv1D + ReLU + MaxPool1D + (repeat) → Flatten → Dense → Softmax
```

Each `MaxPool1D(2)` halves the sequence length, expanding the receptive field while shrinking the downstream Dense parameter count.

## COMPUTE / MEMORY {#compute-memory}

- **Complexity**: `O(B · C · OH · OW · kH · kW)` (no matmul).
- **Training memory** (MaxPool only): `mask_` matches output size in 1-D, 2× in 2-D.
- **PSRAM**: pool layers themselves carry no weights; activation memory depends on batch / channels / length.
