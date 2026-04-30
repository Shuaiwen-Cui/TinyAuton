# Notes

!!! note "Notes"
    `tiny_loss` ships four common loss functions: Mean Squared Error (MSE), Mean Absolute Error (MAE), Softmax + Cross-Entropy, and Binary Cross-Entropy. Every loss exposes both a scalar forward value and a gradient tensor backward.

## LossType ENUM

```cpp
enum class LossType
{
    MSE = 0,           // Mean Squared Error
    MAE,               // Mean Absolute Error
    CROSS_ENTROPY,     // Softmax + Cross-Entropy (input = raw logits)
    BINARY_CE          // Binary CE (input = sigmoid probabilities)
};
```

## MATH

### MSE

\[
L = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2,\quad
\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{N}(\hat{y}_i - y_i)
\]

### MAE

\[
L = \frac{1}{N} \sum_i |\hat{y}_i - y_i|,\quad
\frac{\partial L}{\partial \hat{y}_i} = \frac{1}{N}\,\mathrm{sign}(\hat{y}_i - y_i)
\]

### Cross-Entropy (numerically stable, expects logits)

`cross_entropy_forward` consumes raw logits and uses the log-sum-exp trick:

\[
L_b = -\big(\mathrm{logits}_{b, y_b} - m_b\big) + \log\!\Big(\sum_j e^{\mathrm{logits}_{b,j} - m_b}\Big),\;
m_b = \max_j \mathrm{logits}_{b,j}
\]

\[
L = \frac{1}{B} \sum_b L_b
\]

Its gradient is `softmax(logits) - one_hot(labels)` divided by the batch size:

\[
\frac{\partial L}{\partial \mathrm{logits}_{b,j}} = \frac{1}{B}\big(\mathrm{softmax}(\mathrm{logits})_{b,j} - \mathbb{1}[j = y_b]\big)
\]

!!! warning "Label format"
    `cross_entropy_*` takes `int*` labels (length = batch), each entry is a class index in `[0, num_classes)`, not a one-hot tensor.

### Binary CE

Inputs are sigmoid probabilities `pred ∈ (0, 1)`; targets are 0/1:

\[
L = -\frac{1}{N} \sum_i \big[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\big]
\]

For numerical stability, `TINY_MATH_MIN_POSITIVE_INPUT_F32` is added inside the `log` to avoid `log(0)`.

## API OVERVIEW

```cpp
float  mse_forward          (const Tensor &pred, const Tensor &target);
Tensor mse_backward         (const Tensor &pred, const Tensor &target);

float  mae_forward          (const Tensor &pred, const Tensor &target);
Tensor mae_backward         (const Tensor &pred, const Tensor &target);

float  cross_entropy_forward (const Tensor &logits, const int *labels);
Tensor cross_entropy_backward(const Tensor &logits, const int *labels);

float  binary_ce_forward    (const Tensor &pred, const Tensor &target);
Tensor binary_ce_backward   (const Tensor &pred, const Tensor &target);
```

### Dispatch helpers

```cpp
float  loss_forward (const Tensor &pred, const Tensor &target,
                     LossType type, const int *labels = nullptr);

Tensor loss_backward(const Tensor &pred, const Tensor &target,
                     LossType type, const int *labels = nullptr);
```

`Trainer` plugs the loss in via `loss_forward / loss_backward + LossType`, so the loss is fully swappable.

## RECOMMENDATIONS

| Scenario | Loss | Final layer |
| --- | --- | --- |
| Multi-class classification | `CROSS_ENTROPY` | Dense (raw logits — softmax is built into the loss) |
| Binary classification | `BINARY_CE` | Dense + Sigmoid |
| Regression | `MSE` | Dense |
| Robust regression | `MAE` | Dense |

!!! tip "Softmax + Cross-Entropy"
    `cross_entropy_forward` already contains softmax, so the model's last activation can be `ActType::LINEAR` (or omitted). `MLP` / `CNN1D` default to `use_softmax = true` mostly because `predict()` / `accuracy()` need probabilities downstream — feel free to disable it if you don't.
