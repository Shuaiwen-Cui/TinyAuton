# Notes

!!! note "Notes"
    `tiny_pool` ships 1-D and 2-D max-/avg-pool layers. Pooling is independent across channels and does not introduce learnable parameters; during training, only MaxPool stores argmax positions for backward.

## SHAPE CONVENTIONS

| Layer | Input | Output |
| --- | --- | --- |
| MaxPool1D / AvgPool1D | `[batch, channels, length]` | `[batch, channels, (L - K) / S + 1]` |
| MaxPool2D / AvgPool2D | `[batch, channels, height, width]` | `[batch, channels, (H - kH) / sH + 1, (W - kW) / sW + 1]` |

The constructor's `stride` defaults to `kernel` (i.e. non-overlapping pooling).

## MaxPool1D

```cpp
class MaxPool1D : public Layer
{
    explicit MaxPool1D(int kernel_size, int stride = -1);
    Tensor forward (const Tensor &x) override;
    Tensor backward(const Tensor &grad_out) override;
};
```

### Forward

```cpp
out[b, c, l] = max(x[b, c, l*S + 0..K-1])
```

The absolute argmax index is recorded in `mask_[b, c, l]` (training only).

### Backward

`grad_out[b, c, l]` is written to `g[b, c, mask_[b, c, l]]` (gradient flows only into the max position).

## AvgPool1D

```cpp
class AvgPool1D : public Layer { ... };
```

### Forward

```cpp
out[b, c, l] = (1/K) * Σ_k x[b, c, l*S + k]
```

### Backward

Distribute `grad_out / K` evenly to the K positions in the receptive field.

## MaxPool2D

```cpp
class MaxPool2D : public Layer
{
    MaxPool2D(int kH, int kW, int sH = -1, int sW = -1);
};
```

`mask_` has shape `[B, C, OH, OW * 2]`, packing `(ih, iw)` as two consecutive floats (saves a reshape). Backward unpacks the same layout and writes `grad_out` to the original max position.

## AvgPool2D

Mirrors AvgPool1D: forward averages `kH * kW` elements; backward broadcasts `grad_out / (kH * kW)` back.

## NOTES

- **`stride` defaults to `kernel`** → non-overlapping pooling (most common).
- **MaxPool gradients are sparse**: only the max position receives gradient (winner-take-all), often desirable for feature selection.
- **AvgPool gradients are uniform**: more stable but less selective; commonly used for global average pooling (see `GlobalAvgPool` under [LAYERS/BASE](../BASE/notes.md)).
- **No padding**: the implementation assumes `(L - K)` and `(H/W - kH/kW)` are divisible by `stride`. Callers must align shapes; otherwise edge elements are silently dropped.

## CONV + POOL RECIPE

A standard CNN1D block looks like:

```txt
Conv1D + ReLU + MaxPool1D + (repeat) → Flatten → Dense → Softmax
```

Each `MaxPool1D(2)` halves the sequence length, expanding the receptive field while shrinking the downstream Dense parameter count.

## COMPUTE / MEMORY

- **Complexity**: `O(B · C · OH · OW · kH · kW)` (no matmul).
- **Training memory** (MaxPool only): `mask_` matches output size in 1-D, 2× in 2-D.
- **PSRAM**: pool layers themselves carry no weights; activation memory depends on batch / channels / length.
