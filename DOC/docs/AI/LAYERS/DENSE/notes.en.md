# TinyAI · Layers · Dense — Design notes {#notes}

!!! info "Implementation and records"
    APIs in this section are based on `CODE/AIoTNode-TinyAuton-AI/middleware/`. Source excerpts and serial output include historical records; check the project entry and enabled selectors before reproducing a test.

!!! note "Notes"
    `Dense` is the fully-connected (linear) layer: \( y = x W^\top + b \). It powers MLPs and classification heads. Weights are initialised with Xavier-uniform to keep activations well-scaled in deep stacks.

!!! abstract "Dense — Fully-Connected: Every Output Sees Every Input"
    \(\mathbf{y} = \mathbf{x} \cdot W^T + \mathbf{b}\) — a weighted sum at its core.

## Intuition {#intuition}

### Each Neuron = One Weighted Vote {#each-neuron-one-weighted-vote}

One output neuron's job:

1. Look at **all** input features
2. Assign a **weight** to each (important features → higher weight)
3. Weighted sum + bias
4. Pass through activation to decide whether to "fire"

### What the Weight Matrix Means {#what-the-weight-matrix-means}

\(W\) has shape `[out_features, in_features]` — each row contains all the weights for one output neuron.

### Why Xavier Initialization {#why-xavier-initialization}

If weights are too small → signal vanishes through layers. Too large → signal explodes.

**Xavier uniform**: \(W \sim U\left[-\sqrt{6/(n_{in}+n_{out})},\; \sqrt{6/(n_{in}+n_{out})}\right]\)
Keeps variance consistent, enabling stable signal flow through deep networks.

!!! tip "Dense = One Matrix Multiply"
    ```cpp
    Tensor output = matmul(input, weight.T()) + bias;  // [B, out_features]
    ```

---

## MATH {#math}

Input `x` has shape `[batch, in_features]`, output `y` has shape `[batch, out_features]`:

\[
y_{b, o} = \sum_{i=0}^{F-1} W_{o, i}\, x_{b, i} + b_o
\]

Weights `[out_features, in_features]` (rows = output dim), bias `[out_features]`.

### Xavier-uniform init {#xavier-uniform-init}

\[
W_{o, i} \sim \mathcal{U}(-L, L),\quad
L = \sqrt{\frac{6}{F_\text{in} + F_\text{out}}}
\]

Bias is zero-initialised.

## CLASS DEFINITION {#class-definition}

```cpp
class Dense : public Layer
{
public:
    Tensor weight;   // [out_features, in_features]
    Tensor bias;     // [out_features]   (empty when use_bias=false)

#if TINY_AI_TRAINING_ENABLED
    Tensor dweight;
    Tensor dbias;
#endif

    Dense(int in_features, int out_features, bool use_bias = true);

    Tensor forward(const Tensor &x) override;     // [B, in_feat] → [B, out_feat]
    Tensor backward(const Tensor &grad_out) override;
    void   collect_params(std::vector<ParamGroup> &groups) override;

    int in_features()  const;
    int out_features() const;
};
```

## BACKWARD {#backward}

Input is cached in `x_cache_` (a `clone()` of the forward input). The backward equations:

\[
\frac{\partial L}{\partial W_{o, i}} \mathrel{+}= \sum_b \mathrm{grad\_out}_{b, o}\,x_{b, i}
\]

\[
\frac{\partial L}{\partial b_o} \mathrel{+}= \sum_b \mathrm{grad\_out}_{b, o}
\]

\[
\frac{\partial L}{\partial x_{b, i}} = \sum_o \mathrm{grad\_out}_{b, o}\,W_{o, i}
\]

`dweight` and `dbias` are **accumulated**; the optimiser's `zero_grad()` clears them at the start of every mini-batch.

## PARAMETER COLLECTION {#parameter-collection}

```cpp
void Dense::collect_params(std::vector<ParamGroup> &groups)
{
    groups.push_back({&weight, &dweight});
    if (use_bias_) groups.push_back({&bias, &dbias});
}
```

When `use_bias = false`, the `bias` tensor is empty and is not registered.

## USAGE {#usage}

```cpp
Dense fc1(F, 128);                  // [B, F] → [B, 128]
Dense fc2(128, num_classes);        // [B, 128] → [B, num_classes]

Sequential m;
m.add(new Dense(F, 128));
m.add(new ActivationLayer(ActType::RELU));
m.add(new Dense(128, num_classes));
m.add(new ActivationLayer(ActType::SOFTMAX));
```

Or via the `MLP` convenience wrapper:

```cpp
MLP m({F, 128, 64, num_classes}, ActType::RELU);
```

which auto-inserts ReLU between hidden Dense layers and a final Softmax.

## PERFORMANCE & MEMORY {#performance-memory}

- **Param count**: `F_in * F_out + F_out` (with bias).
- **Complexity**: forward `O(B * F_in * F_out)`; backward of the same order.
- **Memory**: training adds another ~2× weight (`dweight`) and ~1× bias (`dbias`).
- **PSRAM**: when `F_in * F_out ≥ 64 KiB`, store `weight` in PSRAM via `Tensor::from_data`.

## QUANTISATION HOOKS {#quantisation-hooks}

- INT8 PTQ: `quantize_weights(weight, qp)` produces an `int8_t*`, then call `tiny_quant_dense_forward_int8` for fully-integer inference.
- FP8: `calibrate(weight, TINY_DTYPE_FP8_E4M3)` + `quantize(weight, buf, qp)` saves 4× storage; dequantise back to float at runtime.

See [QUANT/INT](../../QUANT/INT/notes.md) and [QUANT/FP8](../../QUANT/FP8/notes.md).
