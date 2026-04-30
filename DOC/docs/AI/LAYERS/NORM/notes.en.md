# Notes

!!! note "Notes"
    `tiny_norm` now provides three normalisation layers: `LayerNorm`, `BatchNorm1D`, and `BatchNorm2D`. BatchNorm layers default to **inference mode**, using `running_mean/running_var` so MCU deployment can directly match PC-trained checkpoints.

## LayerNorm

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

## BatchNorm1D (Dense / MLP)

- Input/output: `[batch, feat]`.
- Constructor: `BatchNorm1D(int feat, float momentum=0.1f, float epsilon=1e-5f)`.
- Training mode: computes per-feature batch `mu/var` and updates running stats:
  - `running_mean = (1-m) * running_mean + m * mu`
  - `running_var  = (1-m) * running_var  + m * var`
- Inference mode: fuses each feature to `scale+shift` constants.

## BatchNorm2D (Conv outputs)

- Input/output: `[N,C,L]` (Conv1D) or `[N,C,H,W]` (Conv2D), same shape out.
- Constructor: `BatchNorm2D(int num_channels, float momentum=0.1f, float epsilon=1e-5f)`.
- Statistics are computed per channel over all non-channel axes (`N * spatial`).
- Inference mode also uses fused `running_mean/running_var`.

## Training / Inference switch

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

## Practical guidance

- **Deployment inference**: load `gamma/beta/running_mean/running_var` from PC training and keep `training_mode=false`.
- **Very small batches**: prefer `LayerNorm` for more stable behavior.
- **Updated demos**: `example_mlp` now includes a `BatchNorm1D` demo; `example_cnn` includes a `BatchNorm2D` demo with explicit mode switching and running-stat inspection.
