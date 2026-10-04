# TinyAI · Models · MLP — Design notes {#notes}

!!! info "Implementation and records"
    APIs in this section are based on `CODE/AIoTNode-TinyAuton-AI/middleware/`. Source excerpts and serial output include historical records; check the project entry and enabled selectors before reproducing a test.

!!! note "Notes"
    `MLP` is a convenience wrapper around `Sequential` for Multi-Layer Perceptrons. Pass a list of dimensions `{in, h1, h2, ..., out}`; it auto-inserts Dense + activation layers and optionally a final Softmax.

!!! abstract "MLP — Multi-Layer Perceptron: Universal Function Approximator"
    MLP = stacked Dense + Activation layers. With enough neurons, can approximate any function.

## Intuition {#intuition}

### Universal Function Approximator {#universal-function-approximator}

**Structure**: `Input → [Dense → ReLU] × N → Dense (output)`

### vs Hand-Crafted Features {#vs-hand-crafted-features}

MLP automatically learns what features matter, instead of requiring manual feature engineering.

### IRIS Example {#iris-example}

```
Dense(4→64) → ReLU → Dense(64→3) → Softmax
```
Input: 4 flower dimensions → Output: 3 Iris species.

---

## CLASS DEFINITION {#class-definition}

```cpp
class MLP : public Sequential
{
public:
    MLP(std::initializer_list<int> dims,
        ActType hidden_act = ActType::RELU,
        bool    use_softmax = true,
        bool    use_bias    = true);

    int in_features()  const;
    int out_features() const;
};
```

## CONSTRUCTION LOGIC {#construction-logic}

For `dims = {d0, d1, ..., d_{N-1}}`, loop `i = 0..N-2`:

1. `add(new Dense(d_i, d_{i+1}, use_bias))`.
2. If `i < N-2`: `add(new ActivationLayer(hidden_act))` (hidden activation).
3. If `i == N-2` and `use_softmax == true`: `add(new ActivationLayer(SOFTMAX))` (output softmax).

## EXAMPLE {#example}

```cpp
// Iris: 4 → 16 → 8 → 3
MLP model({4, 16, 8, 3}, ActType::RELU, /*use_softmax=*/true);

model.summary();
/*  Sequential model  (6 layers)
    --------------------
      [ 0] dense
      [ 1] activation        // ReLU
      [ 2] dense
      [ 3] activation        // ReLU
      [ 4] dense
      [ 5] activation        // SOFTMAX
    --------------------
*/

Adam opt(1e-3f);
Trainer trainer(&model, &opt, LossType::CROSS_ENTROPY);
trainer.fit(train_data, cfg);
```

## SHOULD I DISABLE SOFTMAX? {#should-i-disable-softmax}

`cross_entropy_forward` already contains softmax, so adding or removing the final softmax does not change the loss value. However:

- Need probabilities from `model.predict()` / `accuracy()` → keep `use_softmax = true`.
- Need raw logits (for temperature scaling, distillation, …) → `use_softmax = false`.

## PARAMETER COUNT {#parameter-count}

For `dims = {F, h1, h2, ..., C}`:

\[
\text{params} = \sum_{i=0}^{N-2} (d_i \cdot d_{i+1} + d_{i+1})
\]

E.g. `{4, 16, 8, 3}` → `4*16 + 16 + 16*8 + 8 + 8*3 + 3 = 251` floats ≈ 1 KB. Even with training (extra m/v buffers) the budget stays below ~3 KB — easily fits in ESP32-S3 internal SRAM.

## RELATION TO Sequential {#relation-to-sequential}

`MLP` only overrides the constructor; `forward / backward / summary / predict / accuracy` come straight from `Sequential`. For richer topologies (residuals, branches, skip connections), use `Sequential` directly or subclass it.
