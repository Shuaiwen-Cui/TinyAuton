# Notes

!!! note "Notes"
    `CNN1D` is a convenience wrapper around `Sequential` for 1-D convolutional neural networks. It uses a `CNN1DConfig` struct to describe each conv block's filter count, kernel size, pool window and the final classification head, then auto-builds the full "Conv1D + ReLU + MaxPool1D" pipeline.

## CNN1DConfig

```cpp
struct CNN1DConfig
{
    int signal_length;       // input length (e.g. 64)
    int in_channels    = 1;
    int num_classes    = 3;

    std::vector<int> filters;   // output channels per block, e.g. {16, 32}
    std::vector<int> kernels;   // kernel sizes per block,  e.g. {3, 3}
    int pool_size      = 2;     // MaxPool1D window after each block

    int  fc_units      = 32;    // hidden Dense units (0 to skip the hidden Dense)
    bool use_softmax   = true;
};
```

## CONSTRUCTION LOGIC

`CNN1D::CNN1D(const CNN1DConfig &cfg)`:

1. For `i = 0..filters.size()-1`:
    - `Conv1D(in_ch, filters[i], kernels[i] (or 3), 1, 0, true)` (stride=1, no padding).
    - `ActivationLayer(ActType::RELU)`.
    - `MaxPool1D(pool_size, pool_size)`.
    - Update `L = (L - k + 1) / pool_size`, `in_ch = filters[i]`.
2. `Flatten()`: flatten `[B, in_ch, L]` → `[B, in_ch*L]`.
3. Classification head:
    - If `fc_units > 0`: `Dense(flat, fc_units) → ReLU → Dense(fc_units, num_classes)`.
    - Else: `Dense(flat, num_classes)`.
4. If `use_softmax`: add `ActivationLayer(ActType::SOFTMAX)`.

`flat_features()` returns the flattened dimension so you can size the Dense correctly.

## EXAMPLE

```cpp
CNN1DConfig cfg;
cfg.signal_length = 64;
cfg.in_channels   = 1;
cfg.num_classes   = 3;
cfg.filters       = {16, 32};
cfg.kernels       = {3, 3};
cfg.pool_size     = 2;
cfg.fc_units      = 32;
cfg.use_softmax   = true;

CNN1D model(cfg);
model.summary();
```

Input `[B, 1, 64]` flows through:

```txt
Conv1D(1→16, k=3, p=0)  -> [B, 16, 62]
ReLU                    -> [B, 16, 62]
MaxPool1D(2)            -> [B, 16, 31]
Conv1D(16→32, k=3)      -> [B, 32, 29]
ReLU                    -> [B, 32, 29]
MaxPool1D(2)            -> [B, 32, 14]
Flatten                 -> [B, 32*14 = 448]
Dense(448 → 32)         -> [B, 32]
ReLU                    -> [B, 32]
Dense(32 → 3)           -> [B, 3]
SOFTMAX                 -> [B, 3]
```

## COMPUTE / MEMORY

- **Parameter count**: depends on `filters / kernels / fc_units`. `{16, 32}` + `fc_units=32` is roughly 14 KB of float weights.
- **Activation memory**: each conv block stores `B × ch × L` activations; training also caches the inputs.
- **PSRAM**: `example_cnn.cpp` runs at `B=8` comfortably on ESP32-S3 with its 8 MB PSRAM.

## USE CASES

- Vibration / accelerometer classification.
- ECG, EMG, voice-frame classification.
- Any multi-class problem on 1-D time-series.

A full training + FP8 quantisation walk-through lives in [EXAMPLES/CNN](../../EXAMPLES/CNN/notes.md).
