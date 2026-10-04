# TinyAI · Core · Loss — Design notes {#notes}

!!! info "Current input contract"
    CrossEntropy accepts logits and normalizes them internally. The probability formula below explains the loss; it does not require already-softmaxed predictions at the API boundary.

!!! info "Implementation and records"
    APIs in this section are based on `CODE/AIoTNode-TinyAuton-AI/middleware/`. Source excerpts and serial output include historical records; check the project entry and enabled selectors before reproducing a test.

!!! note "Notes"
    `tiny_loss` ships four common loss functions: Mean Squared Error (MSE), Mean Absolute Error (MAE), Softmax + Cross-Entropy, and Binary Cross-Entropy. Every loss exposes both a scalar forward value and a gradient tensor backward.

!!! abstract "Loss — Measuring the Gap Between Prediction and Truth"
    The loss function tells you how wrong the model currently is. Training = **minimizing this value**.

## Intuition {#intuition}

### Loss = Penalty {#loss-penalty}

Model predicts \(\hat{y}\), ground truth is \(y\). The loss \(L(\hat{y}, y)\) outputs:

- **0** = perfect prediction
- **Larger** = more wrong

### Common Losses {#common-losses}

| Loss | Formula | Use case | Why this design |
|------|---------|----------|-----------------|
| MSE | mean of \((\hat{y} - y)^2\) | **Regression** (numeric prediction) | Squares penalize large errors—2× off = 4× cost |
| MAE | mean of \(|\hat{y} - y|\) | Regression (robust to outliers) | Absolute penalty is gentler |
| CrossEntropy | \(-\sum y \cdot \log(\hat{y})\) | **Classification** (probabilities) | Worst when model is confidently wrong |

!!! tip "CrossEntropy intuition"
    True label is cat (`[1,0,0]`). Model predicts `[0.8, 0.1, 0.1]` → small loss. Predicts `[0.3, 0.6, 0.1]` → large loss. **Confidently wrong = heaviest penalty**.

### Gradient = Direction Downhill {#gradient-direction-downhill}

The loss gradient \(\partial L / \partial \text{params}\) tells the optimizer:

- **Direction**: which way to adjust parameters to reduce loss
- **Magnitude**: how sensitive the loss is to each parameter

This is the essence of backpropagation and gradient descent.

---

## LossType ENUM {#losstype-enum}

```cpp
enum class LossType
{
    MSE = 0,           // Mean Squared Error
    MAE,               // Mean Absolute Error
    CROSS_ENTROPY,     // Softmax + Cross-Entropy (input = raw logits)
    BINARY_CE          // Binary CE (input = sigmoid probabilities)
};
```

## MATH {#math}

### MSE {#mse}

\[
L = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2,\quad
\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{N}(\hat{y}_i - y_i)
\]

### MAE {#mae}

\[
L = \frac{1}{N} \sum_i |\hat{y}_i - y_i|,\quad
\frac{\partial L}{\partial \hat{y}_i} = \frac{1}{N}\,\mathrm{sign}(\hat{y}_i - y_i)
\]

### Cross-Entropy (numerically stable, expects logits) {#cross-entropy-numerically-stable-expects-logits}

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

### Binary CE {#binary-ce}

Inputs are sigmoid probabilities `pred ∈ (0, 1)`; targets are 0/1:

\[
L = -\frac{1}{N} \sum_i \big[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\big]
\]

For numerical stability, `TINY_MATH_MIN_POSITIVE_INPUT_F32` is added inside the `log` to avoid `log(0)`.

## API OVERVIEW {#api-overview}

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

### Dispatch helpers {#dispatch-helpers}

```cpp
float  loss_forward (const Tensor &pred, const Tensor &target,
                     LossType type, const int *labels = nullptr);

Tensor loss_backward(const Tensor &pred, const Tensor &target,
                     LossType type, const int *labels = nullptr);
```

`Trainer` plugs the loss in via `loss_forward / loss_backward + LossType`, so the loss is fully swappable.

## RECOMMENDATIONS {#recommendations}

| Scenario | Loss | Final layer |
| --- | --- | --- |
| Multi-class classification | `CROSS_ENTROPY` | Dense (raw logits — softmax is built into the loss) |
| Binary classification | `BINARY_CE` | Dense + Sigmoid |
| Regression | `MSE` | Dense |
| Robust regression | `MAE` | Dense |

!!! tip "Softmax + Cross-Entropy"
    `cross_entropy_forward` already contains softmax, so the model's last activation can be `ActType::LINEAR` (or omitted). `MLP` / `CNN1D` default to `use_softmax = true` mostly because `predict()` / `accuracy()` need probabilities downstream — feel free to disable it if you don't.
