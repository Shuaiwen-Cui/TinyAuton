# 说明

!!! note "说明"
    `CNN1D` 是 1-D 卷积神经网络的便捷封装，继承自 `Sequential`。它通过 `CNN1DConfig` 描述每个卷积块的输出通道、卷积核大小、池化窗口和最终的全连接头，自动构造完整的「Conv1D + ReLU + MaxPool1D」流水线。

## CNN1DConfig

```cpp
struct CNN1DConfig
{
    int signal_length;       // 输入序列长度 (例如 64)
    int in_channels    = 1;  // 输入通道
    int num_classes    = 3;  // 输出类别数

    std::vector<int> filters;   // 每个卷积块的输出通道，例如 {16, 32}
    std::vector<int> kernels;   // 每个卷积块的核大小，例如 {3, 3}
    int pool_size      = 2;     // 每个卷积块后的 MaxPool1D 窗口

    int  fc_units      = 32;    // 中间 Dense 单元数；0 表示不加中间 Dense
    bool use_softmax   = true;
};
```

## 构造逻辑

`CNN1D::CNN1D(const CNN1DConfig &cfg)` 流程：

1. 对 `i = 0..filters.size()-1`：
    - `Conv1D(in_ch, filters[i], kernels[i] (or 3), 1, 0, true)`：stride=1，无 padding。
    - `ActivationLayer(ActType::RELU)`。
    - `MaxPool1D(pool_size, pool_size)`。
    - 更新 `L = (L - k + 1) / pool_size`，`in_ch = filters[i]`。
2. `Flatten()`：把 `[B, in_ch, L]` 展平为 `[B, in_ch*L]`。
3. 全连接头：
    - 若 `fc_units > 0`：`Dense(flat, fc_units) → ReLU → Dense(fc_units, num_classes)`。
    - 否则：`Dense(flat, num_classes)`。
4. 若 `use_softmax`：再加一层 `ActivationLayer(ActType::SOFTMAX)`。

`flat_features()` 返回展平后维度，便于推断 Dense 大小。

## 使用示例

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

输入 `[B, 1, 64]` 会经历：

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

## 计算 / 内存

- **参数量**：取决于 `filters / kernels / fc_units`。`{16, 32}` + `fc_units=32` 大约 14 KB float 权重。
- **激活内存**：每个卷积块的中间张量按 `B × ch × L` 大小存放；训练开启时还要再缓存一份输入。
- **PSRAM**：`example_cnn.cpp` 中默认在 `B=8` 下推理；ESP32-S3 PSRAM 8 MB 完全够用。

## 适用场景

- 振动 / 加速度信号分类。
- 心电、肌电、语音帧分类。
- 任何 1-D 时序信号的多类分类问题。

完整训练 + FP8 量化演示见 [EXAMPLES/CNN](../../EXAMPLES/CNN/notes.md)。
