# Notes

!!! note "Notes"
    `tiny_activation` provides forward / backward / in-place implementations of seven common activation functions, all operating on `tiny::Tensor`. Softmax is computed numerically stably along the last dimension, ready to be plugged into the output layer of a classifier.

## ActType ENUM

```cpp
enum class ActType
{
    RELU = 0,        // max(0, x)
    LEAKY_RELU,      // x > 0 ? x : alpha*x
    SIGMOID,         // 1 / (1 + exp(-x))
    TANH,            // tanh(x)
    SOFTMAX,         // exp(xi) / sum(exp(xj)), along the last dim
    GELU,            // 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    LINEAR           // identity
};
```

## MATH

| Activation | Forward | Backward |
| --- | --- | --- |
| ReLU | \( y = \max(0, x) \) | \( \frac{dL}{dx} = \frac{dL}{dy} \cdot \mathbb{1}[x > 0] \) |
| Leaky ReLU | \( y = x \cdot (x > 0) + \alpha x \cdot (x \le 0) \) | \( \frac{dL}{dx} = \frac{dL}{dy} \cdot (\mathbb{1}[x>0] + \alpha \mathbb{1}[x \le 0]) \) |
| Sigmoid | \( y = \frac{1}{1 + e^{-x}} \) | \( \frac{dL}{dx} = \frac{dL}{dy} \cdot y(1-y) \) |
| Tanh | \( y = \tanh(x) \) | \( \frac{dL}{dx} = \frac{dL}{dy} \cdot (1 - y^2) \) |
| Softmax | \( y_i = \frac{e^{x_i - \max}}{\sum_j e^{x_j - \max}} \) | \( \frac{dL}{dx_i} = y_i \left(\frac{dL}{dy_i} - \sum_j \frac{dL}{dy_j} y_j\right) \) |
| GELU | \( y = 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715 x^3)\right)\right) \) | numeric (see `gelu_backward`) |

!!! tip "Softmax stability"
    The implementation first subtracts the row max, then `exp` and normalises — equivalent to \(\operatorname{softmax}(x)\) but free of overflow. The denominator gets `TINY_MATH_MIN_DENOMINATOR` added to avoid divide-by-zero.

## API OVERVIEW

### Forward (returns a new Tensor)

```cpp
Tensor relu_forward       (const Tensor &x);
Tensor leaky_relu_forward (const Tensor &x, float alpha = 0.01f);
Tensor sigmoid_forward    (const Tensor &x);
Tensor tanh_forward       (const Tensor &x);
Tensor softmax_forward    (const Tensor &x);
Tensor gelu_forward       (const Tensor &x);
```

Each `*_forward` clones the input then dispatches to the matching `*_inplace`.

### In-place (mutates x)

```cpp
void relu_inplace       (Tensor &x);
void leaky_relu_inplace (Tensor &x, float alpha = 0.01f);
void sigmoid_inplace    (Tensor &x);
void tanh_inplace       (Tensor &x);
void softmax_inplace    (Tensor &x);
void gelu_inplace       (Tensor &x);
```

Useful when you do not need to keep the input (typical inference pipeline).

### Backward (compiled only when `TINY_AI_TRAINING_ENABLED`)

```cpp
Tensor relu_backward       (const Tensor &x, const Tensor &grad_out);
Tensor leaky_relu_backward (const Tensor &x, const Tensor &grad_out, float alpha = 0.01f);
Tensor sigmoid_backward    (const Tensor &y, const Tensor &grad_out);   // pass forward's output
Tensor tanh_backward       (const Tensor &y, const Tensor &grad_out);   // pass forward's output
Tensor softmax_backward    (const Tensor &y, const Tensor &grad_out);   // pass forward's output
Tensor gelu_backward       (const Tensor &x, const Tensor &grad_out);
```

!!! warning "Backward cache"
    - **ReLU / LeakyReLU / GELU**: pass `x` (the forward input).
    - **Sigmoid / Tanh / Softmax**: pass `y` (the forward output) so we don't recompute the activation.
    `ActivationLayer::forward()` automatically caches the right tensor following this rule.

### Dispatch helpers

```cpp
Tensor act_forward (const Tensor &x, ActType type, float alpha = 0.01f);
void   act_inplace (Tensor &x,       ActType type, float alpha = 0.01f);
Tensor act_backward(const Tensor &cache, const Tensor &grad_out,
                    ActType type, float alpha = 0.01f);
```

Switch on the enum value, useful when the activation type is configured at runtime.

## TYPICAL USAGE

```cpp
// Functional API
Tensor h = tanh_forward(x);

// Pluggable via ActType + dispatch
ActType act = ActType::GELU;
Tensor y = act_forward(x, act);

// Inside a Sequential model
Sequential m;
m.add(new Dense(in, hid));
m.add(new ActivationLayer(ActType::RELU));
```

## WHEN TO USE WHAT

- **Hidden activation**: ReLU / LeakyReLU / GELU.
- **Probability output**: Sigmoid (binary), Softmax (multi-class).
- **Transformer-style MLP**: GELU.
- **Saturating normaliser**: Tanh.
- **No activation**: LINEAR (identity), commonly used when `LossType::CROSS_ENTROPY` consumes raw logits.
