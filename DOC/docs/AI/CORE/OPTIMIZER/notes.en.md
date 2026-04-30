# Notes

!!! note "Notes"
    `tiny_optimizer` exposes two gradient-descent optimisers tuned for the ESP32-S3 memory budget: SGD (momentum + L2) and Adam (lite). All optimisers consume a `std::vector<ParamGroup>` populated by the model layers.

## ParamGroup

```cpp
struct ParamGroup
{
    Tensor *param;  // weight / bias tensor
    Tensor *grad;   // matching gradient tensor
};
```

Each trainable layer (`Dense`, `Conv1D`, `Conv2D`, `LayerNorm`, `Attention`) overrides `Layer::collect_params()` and pushes its `(weight, dweight)`, `(bias, dbias)` pairs onto a `std::vector<ParamGroup>`. `Sequential::collect_params()` collects from the whole network.

## Optimizer Abstract Base

```cpp
class Optimizer
{
public:
    virtual void init(const std::vector<ParamGroup> &groups) = 0;
    virtual void step(std::vector<ParamGroup> &groups)       = 0;
    virtual void zero_grad(std::vector<ParamGroup> &groups);
};
```

Required call order:

1. **Construct**: `SGD opt(lr, mom)` or `Adam opt(lr, β1, β2, ε)`.
2. **Collect params**: `model.collect_params(params)`.
3. **Init**: `opt.init(params)` — only here are momentum / Adam moment buffers allocated to match each parameter's shape.
4. **Training loop**: per batch run `opt.zero_grad(params)` → forward → backward → `opt.step(params)`.

## SGD with momentum & L2

```cpp
SGD(float lr = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f);
```

Update:

\[
g \leftarrow \nabla_\theta + \lambda\,\theta\quad(\text{if}~\lambda > 0)
\]

\[
v \leftarrow \mu\,v + g\quad(\text{if}~\mu > 0)
\]

\[
\theta \leftarrow \theta - \eta \cdot v
\]

Params:

- `lr`: learning rate \(\eta\).
- `momentum`: \(\mu\); 0 falls back to vanilla SGD.
- `weight_decay`: L2 coefficient \(\lambda\).

`init()` allocates one velocity tensor per parameter; `zero_grad()` is provided by the base class.

## Adam (lite)

```cpp
Adam(float lr     = 1e-3f,
     float beta1  = 0.9f,
     float beta2  = 0.999f,
     float epsilon = 1e-8f,
     float weight_decay = 0.0f);
```

Per step:

\[
g \leftarrow \nabla_\theta + \lambda\,\theta
\]

\[
m \leftarrow \beta_1 m + (1-\beta_1) g,\quad
v \leftarrow \beta_2 v + (1-\beta_2) g^2
\]

Bias correction is applied to the LR (cheaper than per-element):

\[
\eta_t = \eta\;\frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
\]

\[
\theta \leftarrow \theta - \eta_t\;\frac{m}{\sqrt{v} + \varepsilon}
\]

`init()` allocates `m` and `v` per parameter; `step()` increments the internal time step `t_`.

!!! tip "Practical defaults"
    - For SHM, biomedical signals or other small/unstable datasets: Adam with defaults works.
    - For sparse / large-batch training: SGD with `lr` ~0.1 and `momentum=0.9`.
    - `weight_decay > 0` matches PyTorch L2 regularisation; do not over-decay biases (the implementation does decay them but their magnitude is small).

## Memory / PSRAM impact

- **SGD**: +1 velocity tensor per parameter → ~2× memory.
- **Adam**: +2 moment tensors per parameter → ~3× memory.

If you place model weights in PSRAM, you typically want optimiser buffers in PSRAM too. `Tensor` defaults to `TINY_AI_MALLOC`; replace weight tensors with `Tensor::from_data(psram_buf, ...)` views when the budget is tight.

## Trainer Integration

`Trainer::ensure_params_collected()` runs lazily on the first `fit()` call:

```cpp
model_->collect_params(params_);
optimizer_->init(params_);
params_collected_ = true;
```

Per batch:

```cpp
optimizer_->zero_grad(params_);
auto logits = model_->forward(X_batch);
auto grad   = loss_backward(logits, ..., loss_type_, y_batch);
model_->backward(grad);
optimizer_->step(params_);
```

Implementing a custom optimiser is just a matter of subclassing `Optimizer` and overriding `init / step` — no changes to layers or Trainer required.
