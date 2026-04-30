# Notes

!!! info "Demo overview"
    `example_mlp` now has two parts: the main Iris MLP flow (training + INT8 PTQ demo) and a new `BatchNorm1D` demo that explicitly shows training/inference mode switching.

## DATA SOURCE

Data comes from `example/data/iris_data.hpp`:

```cpp
namespace tiny_data
{
constexpr int IRIS_N_SAMPLES  = 150;
constexpr int IRIS_N_FEATURES = 4;
constexpr int IRIS_N_CLASSES  = 3;

extern const float IRIS_X[IRIS_N_SAMPLES][IRIS_N_FEATURES];
extern const int   IRIS_Y[IRIS_N_SAMPLES];
}
```

`IRIS_X` and `IRIS_Y` are embedded as `static const` arrays in the firmware — no file I/O required.

## MODEL

```cpp
MLP model({4, 16, 8, 3}, ActType::RELU, /*use_softmax=*/true, /*use_bias=*/true);
```

Expands to:

```txt
Dense(4 → 16) + ReLU
Dense(16 → 8) + ReLU
Dense(8  → 3) + Softmax
```

## TRAINING CONFIG

- **Split**: `dataset.split(0.2f, train_ds, test_ds, 42)` → 120 train / 30 test.
- **Optimiser**: `Adam(lr=1e-3, β1=0.9, β2=0.999)`.
- **Loss**: `LossType::CROSS_ENTROPY` (raw logits in; softmax baked into the loss).
- **Hyper**: `epochs=100, batch_size=16, print_every=20`.

## TRAINING FLOW

```cpp
Trainer trainer(&model, &opt, LossType::CROSS_ENTROPY);
Trainer::Config cfg;
cfg.epochs      = 100;
cfg.batch_size  = 16;
cfg.print_every = 20;
trainer.fit(train_ds, cfg, &test_ds);

float train_acc = trainer.evaluate_accuracy(train_ds);
float test_acc  = trainer.evaluate_accuracy(test_ds);
```

`Trainer::fit` will:

1. On first call, run `model.collect_params(params)` and `opt.init(params)`.
2. Shuffle the training set every epoch via `train_ds.shuffle(epoch + 1)`.
3. Loop with `batch_size = 16` calling `next_batch`, then forward / loss_forward / loss_backward / `model.backward` / `opt.step`.
4. Print `loss + val_acc` every `print_every` epochs.

## INT8 PTQ DEMO

```cpp
QuantParams wp = calibrate(demo_w, TINY_DTYPE_INT8);
int8_t *w_int8 = (int8_t *)TINY_AI_MALLOC(demo_w.size);
tiny_quant_params_t cp = wp.to_c();
tiny_quant_f32_to_int8(demo_w.data, w_int8, demo_w.size, &cp);
```

The demo prints `scale / zero_point / quantised value / dequantised value`, so you can see the symmetric quantisation precision visually. For full integer inference, see `tiny_quant_dense_forward_int8` in [QUANT/INT/notes](../../QUANT/INT/notes.md).

## New: BatchNorm1D demo

After the main pipeline, `example_mlp` calls `batchnorm1d_demo(train_ds, test_ds)` (only when `TINY_AI_TRAINING_ENABLED` is enabled):

```txt
Dense(4,16) + BN1D(16) + ReLU
Dense(16,8) + BN1D(8)  + ReLU
Dense(8,3)  + Softmax
```

Key behaviors:

- `model.set_training_mode(true)`: BN uses current-batch stats and updates `running_mean/running_var`.
- `model.set_training_mode(false)`: BN switches to running stats for inference.
- The demo prints the first BN layer's leading `running_mean/running_var` values for quick sanity checks.

## Measured output (2026-04-30)

```txt
========================================
  tiny_ai  |  MLP Example (Iris)
========================================
Dataset split: 120 train / 30 test
Sequential model  (6 layers)
--------------------
  [ 0] dense
  [ 1] activation
  [ 2] dense
  [ 3] activation
  [ 4] dense
  [ 5] activation
--------------------
Training...
Epoch [ 20/100]  loss: 0.820218  val_acc: 0.7667
Epoch [ 40/100]  loss: 0.690300  val_acc: 0.9333
Epoch [ 60/100]  loss: 0.629859  val_acc: 0.9000
Epoch [ 80/100]  loss: 0.602956  val_acc: 0.9000
Epoch [100/100]  loss: 0.586117  val_acc: 0.9000

--- Float32 Results ---
  Train accuracy: 97.50%
  Test  accuracy: 90.00%

--- INT8 PTQ Inference ---
  Quantisation demo: calibrating weight tensor...
  Weight scale = 0.006299  zero_point = 0
  Original w[0]=-0.8000  Quantised=-127  Dequantised=-0.8000
  INT8 accuracy (whole dataset): 8.67%

[BN1D] training_mode = ON
[train mode] Train: 97.50%  Test: 80.00%
[BN1D] training_mode = OFF  (running stats active)
[infer mode] Train: 99.17%  Test: 96.67%
BN1D[0] running_mean (first 4): -0.0179 -0.0019 0.0345 0.0276
BN1D[0] running_var  (first 4): 0.6317 0.1680 0.9003 0.7524
example_mlp  DONE
```

## Interpretation

- **FP32 main pipeline converges normally**: `Train 97.50% / Test 90.00%`, a reasonable Iris result.
- **INT8=8.67% is not the real quantized accuracy**: current `run_int8_inference()` predicts on `X_test` but compares against full-dataset labels `IRIS_Y`; this mismatched pairing invalidates the reported metric.
- **BN1D inference mode outperforms train-mode eval**: test accuracy improves from `80.00%` to `96.67%` when switching to running stats, which matches expected BatchNorm behavior.
- **Running stats are non-trivial**: printed `running_mean/running_var` confirm BN stats are actually being accumulated.

## ENTRY POINT

`tiny_ai.h` exposes `void example_mlp(void)` for both C and C++ callers:

```cpp
extern "C" void example_mlp(void);

// in AIoTNode.cpp / app_main.c:
example_mlp();
```
