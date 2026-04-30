# Notes

!!! note "Notes"
    `tiny_layer` defines the abstract base class `Layer` shared by every neural-network layer, plus three parameter-free utility layers: `ActivationLayer`, `Flatten`, and `GlobalAvgPool`. These utilities expose the same `Layer` interface so they can be stacked directly inside a `Sequential` model.

## Layer ABSTRACT BASE

```cpp
class Layer
{
public:
    const char *name;        // shown in summary()
    bool        trainable;   // does this layer carry learnable parameters?

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

Contract:

- **`forward(x)`**: must return a new tensor.
- **`backward(grad_out)`**: must return `dL/dx`, while accumulating gradients into `dweight`, `dbias`, … members.
- **`collect_params()`**: only trainable layers override it to push `(param, grad)` pairs.
- **`trainable`**: lets `Sequential::collect_params()` skip parameter-free layers (e.g. `Flatten`).

## ActivationLayer

Wraps a stateless activation as a `Layer` so it stacks naturally inside `Sequential`.

```cpp
class ActivationLayer : public Layer
{
public:
    explicit ActivationLayer(ActType type, float alpha = 0.01f);
    Tensor forward (const Tensor &x) override;
    Tensor backward(const Tensor &grad_out) override;
};
```

Implementation notes:

- **Cache strategy**: `forward()` decides what to cache based on the activation type
    - Sigmoid / Tanh / Softmax → cache the output `y` (used directly by backward).
    - ReLU / LeakyReLU / GELU → cache the input `x`.
- **alpha** defaults to 0.01, applies only to LeakyReLU.

## Flatten

Reshapes `[batch, ...]` into `[batch, flat]`:

```cpp
class Flatten : public Layer
{
    Tensor forward (const Tensor &x) override;   // [B, ...] → [B, size/B]
    Tensor backward(const Tensor &grad_out) override;
};
```

- `flat = size / batch`, dispatched via `reshape_2d`.
- Backward restores the gradient's original `(in_ndim, in_shape)`.
- No parameters, `trainable = false` → ignored by `collect_params`.

Standard CNN1D recipe:

```txt
Conv1D + ReLU + MaxPool1D × N → Flatten → Dense → Softmax
```

`tiny_cnn.cpp` plugs `Flatten` in right after the conv blocks.

## GlobalAvgPool

Mean over the sequence axis, common for Transformer-style outputs (`[B, S, F]` → `[B, F]`):

```cpp
class GlobalAvgPool : public Layer
{
    Tensor forward (const Tensor &x) override;   // [B, S, F] → [B, F]
    Tensor backward(const Tensor &grad_out) override;
};
```

- Forward averages over `s`.
- Backward distributes `grad_out` evenly across the `S` positions.

The Attention example chains it:

```txt
Attention([B, S, E]) → GlobalAvgPool([B, E]) → Dense([B, num_classes])
```

## Interaction with Sequential

```cpp
Sequential m;
m.add(new Dense(F, H));
m.add(new ActivationLayer(ActType::RELU));
m.add(new Dense(H, C));
m.add(new ActivationLayer(ActType::SOFTMAX));
```

- `Sequential` owns every `Layer*` registered via `add()` and `delete`s them in its destructor.
- `forward(x)` calls each `forward` in order; `backward(grad_out)` calls each `backward` in reverse.
- Only `trainable == true` layers are visited by `Sequential::collect_params()`.

## Custom Layer

Subclass `Layer`, implement `forward / backward / collect_params`. Example:

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

    Tensor forward(const Tensor &x) override { /* x * scale */ return x.clone(); }

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override { return grad_out; }
    void collect_params(std::vector<ParamGroup> &g) override
    {
        g.push_back({&scale, &dscale});
    }
#endif
};
```
