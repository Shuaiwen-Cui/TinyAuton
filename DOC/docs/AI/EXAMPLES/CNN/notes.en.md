# Notes

!!! info "Demo overview"
    `example_cnn` now has three parts: 1-D CNN training, FP8 E4M3FN compression demo, and a new `BatchNorm2D` demo that illustrates BN behavior in Conv pipelines.

## DATA SOURCE

```cpp
namespace tiny_data
{
constexpr int SIG_N_CLASSES         = 3;
constexpr int SIG_SAMPLES_PER_CLASS = 50;
constexpr int SIG_N_SAMPLES         = SIG_N_CLASSES * SIG_SAMPLES_PER_CLASS;  // 150
constexpr int SIG_SIGNAL_LEN        = 64;

void generate_signal_dataset(float *X, int *Y);
}
```

`generate_signal_dataset` synthesises 150 × 64-point signals (sines, squares, noisy variants) into a PSRAM buffer that is then wrapped into a `Dataset`.

## MODEL

```cpp
CNN1DConfig cfg;
cfg.in_channels   = 1;
cfg.signal_length = 64;
cfg.num_classes   = 3;
cfg.filters       = {8, 16};
cfg.kernels       = {5, 3};
cfg.pool_size     = 2;
cfg.fc_units      = 32;

CNN1D model(cfg);
```

Expands to:

```txt
Conv1D(1→8,  k=5)  + ReLU + MaxPool1D(2)   -> [B, 8, (64-5+1)/2 = 30]
Conv1D(8→16, k=3)  + ReLU + MaxPool1D(2)   -> [B,16, (30-3+1)/2 = 14]
Flatten                                     -> [B, 16*14 = 224]
Dense(224 → 32) + ReLU
Dense(32  → 3)  + Softmax
```

`model.flat_features()` returns the flattened dimension (224 here) for verification.

## CUSTOM TRAINING LOOP

`Dataset` returns `[B, L]`, but `Conv1D` expects `[B, 1, L]`, so the example bypasses `Trainer::fit` and writes the loop manually:

```cpp
std::vector<ParamGroup> params;
model.collect_params(params);
opt.init(params);

for (int epoch = 0; epoch < 50; epoch++)
{
    train_ds.shuffle(epoch + 1);
    while (next_batch returns > 0)
    {
        Tensor X3(actual, 1, SIG_SIGNAL_LEN);  // [B, L] → [B, 1, L]
        ... copy ...

        Tensor logits = model.forward(X3);
        float  loss   = cross_entropy_forward(logits, y_batch);
        opt.zero_grad(params);
        Tensor grad   = cross_entropy_backward(logits, y_batch);
        model.backward(grad);
        opt.step(params);
    }
}
```

The hand-rolled loop also illustrates how `Optimizer + Sequential + Dataset` can be used independently.

## EVALUATION

```cpp
auto eval_accuracy = [&](Dataset &ds, const char *tag) {
    ds.reset();
    ...
    Tensor X3(actual, 1, SIG_SIGNAL_LEN);
    model.predict(X3, yp);
    ...
};
eval_accuracy(train_ds, "Train");
eval_accuracy(test_ds,  "Test ");
```

Same `[B, L] → [B, 1, L]` reshape per batch, then `Sequential::predict` produces argmax labels.

## FP8 COMPRESSION DEMO

```cpp
Tensor w(16, 8, 3);      // mock Conv1D weight
QuantParams qp = calibrate(w, TINY_DTYPE_FP8_E4M3);

uint8_t *fp8_buf = (uint8_t *)TINY_AI_MALLOC(w.size);
quantize(w, fp8_buf, qp);

Tensor w_recon = Tensor::zeros_like(w);
dequantize(fp8_buf, w_recon, qp);
```

Prints:

- 4:1 compression ratio (`size * 4` bytes → `size` bytes).
- Max absolute error and MSE.
- The first 4 elements: original, FP8 byte, reconstructed.

Useful as a benchmark against INT8 compression.

## New: BatchNorm2D demo

At the end, `example_cnn` runs `batchnorm2d_demo(train_ds, test_ds)` (training builds only):

```txt
Conv1D(1,8,5)  + BN2D(8)  + ReLU + MaxPool1D(2)
Conv1D(8,16,3) + BN2D(16) + ReLU + MaxPool1D(2)
Flatten -> Dense(224,32)+ReLU -> Dense(32,3)+Softmax
```

Highlights:

- BN2D is applied on Conv1D outputs `[B, C, L]`, with per-channel stats over `N*L`.
- `training_mode=ON` updates `running_mean/running_var`.
- `training_mode=OFF` uses running stats for inference.
- The demo prints train/test accuracy in both modes and dumps leading channel stats from the first BN layer.

## Measured output (2026-04-30)

```txt
========================================
  tiny_ai  |  CNN Example (Signal)
========================================
Generated 150 signals (3 classes × 50 samples, 64 pts each)
Split: 120 train / 30 test
Sequential model  (11 layers)
--------------------
  [ 0] conv1d
  [ 1] activation
  [ 2] max_pool1d
  [ 3] conv1d
  [ 4] activation
  [ 5] max_pool1d
  [ 6] flatten
  [ 7] dense
  [ 8] activation
  [ 9] dense
  [10] activation
--------------------
  Flat features after conv blocks: 224
Training...
Epoch [ 10/ 50]  loss: 0.558837
Epoch [ 20/ 50]  loss: 0.552400
Epoch [ 30/ 50]  loss: 0.551829
Epoch [ 40/ 50]  loss: 0.551640
Epoch [ 50/ 50]  loss: 0.551561

--- Float32 Results ---
  Train accuracy: 100.00%
  Test  accuracy: 100.00%

--- FP8 E4M3FN Compression Demo ---
  E4M3 scale = 0.008571
  Size: FP32=1536 bytes  FP8=384 bytes  (4:1 compression)
  Max abs error = 0.990000   MSE = 0.16210581
  w[0] orig=-3.8400  fp8=0xC4  recon=-3.0000
  ...

[BN2D] training_mode = ON
  [train mode] Train acc: 100.00%
  [train mode] Test  acc: 100.00%
[BN2D] training_mode = OFF  (running stats active)
  [infer mode] Train acc: 100.00%
  [infer mode] Test  acc: 100.00%
BN2D[0] running_mean (first 4 ch): 0.0000 -0.0023 -0.0046 0.0108
BN2D[0] running_var  (first 4 ch): 0.2838 1.0601 0.6727 2.8801
example_cnn  DONE
```

## Interpretation

- **Task is fully learned**: both train and test are `100%`, indicating this synthetic dataset is easy for the current model.
- **Higher loss plateau with perfect accuracy is possible**: accuracy tracks class decisions; loss also depends on confidence, so they do not have to move together.
- **FP8 distortion is significant here**: `max_err=0.99`, `MSE=0.162` indicate strong clipping/quantization under E4M3 for this value range; this is a compression-error demonstration, not end-to-end deployment accuracy.
- **BN2D mode switch is stable**: matching ON/OFF accuracies suggest running stats are well aligned with the data distribution.

## ENTRY POINT

```cpp
extern "C" void example_cnn(void);
example_cnn();
```
