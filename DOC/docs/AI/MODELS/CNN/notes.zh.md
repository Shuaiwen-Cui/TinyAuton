# TinyAI · 模型 · CNN — 设计说明 {#_1}

!!! info "实现依据与记录"
    本节接口以 `CODE/AIoTNode-TinyAuton-AI/middleware/` 为依据。源码摘录与串口输出包含历史记录；是否运行某项测试，请核对工程入口和启用开关。

!!! note "说明"
    `CNN1D` 是 1-D 卷积神经网络的便捷封装，继承自 `Sequential`。它通过 `CNN1DConfig` 描述每个卷积块的输出通道、卷积核大小、池化窗口和最终的全连接头，自动构造完整的「Conv1D + ReLU + MaxPool1D」流水线。

!!! abstract "CNN — 卷积神经网络：让 Conv 做特征提取，Dense 做分类"
    对于 SHM 时序信号，CNN1D 是默认选择：Conv 层自动学习频域/时域特征，Pool 层压缩，Dense 层分类。

## 算法直觉 {#_2}

### CNN1D 处理流水线 {#cnn1d}

```
输入信号 → [Conv1D → ReLU → MaxPool] × N → Flatten → Dense → 输出
```

1. **Conv1D**: 用多个卷积核扫描信号，学不同模式（频率、冲击、趋势）
2. **MaxPool**: 下采样，保留最强响应，减少数据量
3. **Flatten**: 把多维特征展平成一维向量
4. **Dense**: 在展平的特征上做分类

### CNN1DConfig 快速配置 {#cnn1dconfig}

```cpp
CNN1DConfig config;
config.in_channels  = 1;       // 单通道加速度信号
config.out_channels = {16, 32}; // 第1层16个核，第2层32个核
config.kernel_sizes = {9, 5};  // 第1层核宽9，第2层核宽5
config.strides = {2, 2};       // 每层步长2
config.num_classes = 3;        // 3类输出
```

---

## CNN1DConfig {#cnn1dconfig_1}

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

## 构造逻辑 {#_3}

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

## 使用示例 {#_4}

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

## 计算 / 内存 {#_5}

- **参数量**：取决于 `filters / kernels / fc_units`。`{16, 32}` + `fc_units=32` 大约 14 KB float 权重。
- **激活内存**：每个卷积块的中间张量按 `B × ch × L` 大小存放；训练开启时还要再缓存一份输入。
- **PSRAM**：`example_cnn.cpp` 中默认在 `B=8` 下推理；ESP32-S3 PSRAM 8 MB 完全够用。

## 适用场景 {#_6}

- 振动 / 加速度信号分类。
- 心电、肌电、语音帧分类。
- 任何 1-D 时序信号的多类分类问题。

完整训练 + FP8 量化演示见 [EXAMPLES/CNN](../../EXAMPLES/CNN/notes.md)。
