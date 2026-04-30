# Notes

!!! info "Demo overview"
    `example_attention` is a minimal Transformer-style classifier: 4 Iris features are projected into "4 tokens × 8 dims", then fed to a Multi-Head Self-Attention block, followed by global average pooling and a linear classifier. It validates that `Attention`, `GlobalAvgPool`, and `Dense` can be trained on-device on a small dataset.

## DATA SOURCE

```cpp
Dataset dataset(
    &IRIS_X[0][0], IRIS_Y,
    IRIS_N_SAMPLES, IRIS_N_FEATURES, IRIS_N_CLASSES);

Dataset train_ds(dataset), test_ds(dataset);
dataset.split(0.2f, train_ds, test_ds, 42);
```

Same Iris data (`iris_data.hpp`) used by the MLP example: 150 samples × 4 features × 3 classes.

## MODEL

```txt
Input          : [B, 4]
embed_proj     : Dense(4, 32) + ReLU       -> [B, 32]
reshape        : [B, 32] -> [B, 4, 8]      (4 tokens × embed_dim 8)
attn           : Attention(embed_dim=8, heads=2)  -> [B, 4, 8]
gap            : GlobalAvgPool             -> [B, 8]
classifier     : Dense(8, 3)               -> [B, 3]   (raw logits)
```

Key points:

- `embed_proj` expands 4-D features into a `4 * 8 = 32`-D vector, then reshapes into 4 tokens of dim 8.
- `Attention(embed_dim=8, heads=2, head_dim=4)` is a standard multi-head self-attention block.
- Because `cross_entropy_forward(logits, y)` consumes raw logits, the classifier **does not** end with a softmax layer.
- Since the data flow toggles between `[B, 32]` and `[B, 4, 8]`, the example wires layers up directly instead of using `Sequential`.

## TRAINING LOOP

Reshape-aware training loop:

```cpp
std::vector<ParamGroup> params;
embed_proj.collect_params(params);
attn.collect_params(params);
classifier.collect_params(params);
opt.init(params);

for (int epoch = 0; epoch < 100; epoch++)
{
    train_ds.shuffle(epoch + 1);
    while (true)
    {
        int actual = train_ds.next_batch(X_batch, y_batch, 16);
        if (actual == 0) break;

        // forward
        Tensor e0 = embed_proj.forward(X_batch);
        Tensor e1 = embed_act.forward(e0);
        e1.reshape_3d(actual, 4, 8);
        Tensor a0     = attn.forward(e1);
        Tensor p0     = gap.forward(a0);
        Tensor logits = classifier.forward(p0);

        float loss = cross_entropy_forward(logits, y_batch);

        // backward
        opt.zero_grad(params);
        Tensor dlogits = cross_entropy_backward(logits, y_batch);
        Tensor dp0     = classifier.backward(dlogits);
        Tensor da0     = gap.backward(dp0);
        Tensor de1     = attn.backward(da0);
        de1.reshape_2d(actual, 32);
        Tensor de0     = embed_act.backward(de1);
        embed_proj.backward(de0);
        opt.step(params);
    }
}
```

Notes:

- All trainable layers must be passed through `collect_params` before `opt.init(params)` — otherwise SGD/Adam has nothing to update.
- The backward chain mirrors forward exactly, including the `reshape_3d / reshape_2d` pair.
- `Attention` already caches the intermediate `Q/K/V/A` tensors internally; no manual bookkeeping is needed.

## EVALUATION

```cpp
auto eval_accuracy = [&](Dataset &ds, const char *tag) {
    ds.reset();
    while (true)
    {
        int actual = ds.next_batch(Xb, yb, batch_size);
        if (actual == 0) break;
        Tensor e0     = embed_proj.forward(Xb);
        Tensor e1     = embed_act.forward(e0);
        e1.reshape_3d(actual, 4, 8);
        Tensor a0     = attn.forward(e1);
        Tensor p0     = gap.forward(a0);
        Tensor logits = classifier.forward(p0);
        // argmax over logits...
    }
};
```

Because `Sequential::predict` cannot express the reshape, the example does an inline argmax instead.

## Measured output (2026-04-30)

```txt
========================================
  tiny_ai  |  Attention Example (Iris)
========================================
Dataset split: 120 train / 30 test
Model summary:
  Dense(4, 32) + ReLU
  reshape [B, 32] -> [B, 4, 8]  (tokens, embed_dim)
  Attention(embed_dim=8, heads=2, head_dim=4)
  GlobalAvgPool
  Dense(8, 3)  [raw logits]

Training...
Epoch [ 20/100]  loss: 0.319672
Epoch [ 40/100]  loss: 0.110597
Epoch [ 60/100]  loss: 0.046880
Epoch [ 80/100]  loss: 0.033336
Epoch [100/100]  loss: 0.027159

--- Float32 Results ---
  Train accuracy: 99.17%
  Test  accuracy: 90.00%
example_attention  DONE
```

## Interpretation

- **Convergence is strong**: loss decreases from `0.319` to `0.027`, showing stable learning across the attention pipeline.
- **There is a generalization gap**: `Train 99.17%` vs `Test 90.00%`; this is common on small datasets and can be further checked with early stopping, regularization, or multi-seed runs.
- **Compared with MLP**: both current Attention and MLP runs reach about `90%` test accuracy, so Attention is not clearly superior under this setup, but its full training path is validated.

## RESOURCE COST

- The attention block caches `[B, S, F]`, `[B, H, S, dh]`, `[B, H, S, S]` per batch; with batch=16, S=4, F=8 these are only a few KB.
- Total training-time memory is well under 100 KB and runs comfortably in ESP32-S3 internal RAM. For larger configurations, redirect `embed_proj` and `attn` intermediate tensors to PSRAM.

## ENTRY POINT

```cpp
extern "C" void example_attention(void);
example_attention();
```
