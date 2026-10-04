# TinyAI · Layers · Norm — Design notes {#notes}

!!! info "Implementation and records"
    APIs in this section are based on `CODE/AIoTNode-TinyAuton-AI/middleware/`. Source excerpts and serial output include historical records; check the project entry and enabled selectors before reproducing a test.

!!! note "Notes"
    `tiny_norm` now provides three normalisation layers: `LayerNorm`, `BatchNorm1D`, and `BatchNorm2D`. BatchNorm layers default to **inference mode**, using `running_mean/running_var` so MCU deployment can directly match PC-trained checkpoints.

!!! abstract "Norm — LayerNorm: Stabilizing Activations Layer by Layer"
    During training, the distribution of each layer's input keeps changing (Internal Covariate Shift). Normalization pulls it back to a stable range.

## Intuition {#intuition}

### LayerNorm {#layernorm}

**LayerNorm** normalizes **each sample independently**: compute mean and variance across all features of that sample, then normalize.

\[
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \cdot \hat{x} + \beta
\]

- \(\mu, \sigma\): mean and variance of the current sample (across feature dims)
- \(\gamma, \beta\): learnable scale and shift (network decides the best distribution post-norm)

### Why Before Activation? {#why-before-activation}

Norm is typically applied before Activation (**Pre-Norm**), constraining the numerical range before entering the activation's saturation zone.

!!! tip "When to use LayerNorm?"
    LayerNorm works regardless of batch size—even batch=1. Ideal for Attention layers and sequence models.

---

## LayerNorm {#layernorm_1}

`LayerNorm` normalises along the last dimension (`feat`) without running statistics:

\[
\mu = \frac{1}{F}\sum_f x_f,\quad
\sigma^2 = \frac{1}{F}\sum_f (x_f-\mu)^2,\quad
\hat{x}_f = \frac{x_f-\mu}{\sqrt{\sigma^2+\varepsilon}}
\]

\[
y_f = \gamma_f \hat{x}_f + \beta_f
\]

- Learnable params: `gamma/beta` of shape `[feat]`.
- Default `epsilon=1e-5`.
- Works with any rank as long as the last dim equals `feat`.

## BatchNorm1D (Dense / MLP) {#batchnorm1d-dense-mlp}

- Input/output: `[batch, feat]`.
- Constructor: `BatchNorm1D(int feat, float momentum=0.1f, float epsilon=1e-5f)`.
- Training mode: computes per-feature batch `mu/var` and updates running stats:
  - `running_mean = (1-m) * running_mean + m * mu`
  - `running_var  = (1-m) * running_var  + m * var`
- Inference mode: fuses each feature to `scale+shift` constants.

## BatchNorm2D (Conv outputs) {#batchnorm2d-conv-outputs}

- Input/output: `[N,C,L]` (Conv1D) or `[N,C,H,W]` (Conv2D), same shape out.
- Constructor: `BatchNorm2D(int num_channels, float momentum=0.1f, float epsilon=1e-5f)`.
- Statistics are computed per channel over all non-channel axes (`N * spatial`).
- Inference mode also uses fused `running_mean/running_var`.

## Training / Inference switch {#training-inference-switch}

Both `BatchNorm1D` and `BatchNorm2D` expose:

```cpp
bn->set_training(true);   // batch stats + running-stat update
bn->set_training(false);  // running stats only
```

At model level:

```cpp
model.set_training_mode(true);   // training
model.set_training_mode(false);  // inference
```

## Practical guidance {#practical-guidance}

- **Deployment inference**: load `gamma/beta/running_mean/running_var` from PC training and keep `training_mode=false`.
- **Very small batches**: prefer `LayerNorm` for more stable behavior.
- **Updated demos**: `example_mlp` now includes a `BatchNorm1D` demo; `example_cnn` includes a `BatchNorm2D` demo with explicit mode switching and running-stat inspection.
