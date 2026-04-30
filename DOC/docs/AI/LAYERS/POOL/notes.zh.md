# 说明

!!! note "说明"
    `tiny_pool` 提供 1-D / 2-D 的最大池化与平均池化层。池化在通道维独立进行，不带可学习参数；训练时仅记录 argmax 位置（仅 MaxPool）以便反向传播。

## 形状约定

| 层 | 输入 | 输出 |
| --- | --- | --- |
| MaxPool1D / AvgPool1D | `[batch, channels, length]` | `[batch, channels, (L - K) / S + 1]` |
| MaxPool2D / AvgPool2D | `[batch, channels, height, width]` | `[batch, channels, (H - kH) / sH + 1, (W - kW) / sW + 1]` |

构造时 `stride` 默认为 `kernel`（即非重叠池化）。

## MaxPool1D

```cpp
class MaxPool1D : public Layer
{
    explicit MaxPool1D(int kernel_size, int stride = -1);
    Tensor forward (const Tensor &x) override;
    Tensor backward(const Tensor &grad_out) override;
};
```

### 前向

```cpp
out[b, c, l] = max(x[b, c, l*S + 0..K-1])
```

并把 argmax 的绝对下标存到 `mask_[b, c, l]`（仅训练时）。

### 反向

把 `grad_out[b, c, l]` 写回 `g[b, c, mask_[b, c, l]]`（梯度只流向最大值位置）。

## AvgPool1D

```cpp
class AvgPool1D : public Layer { ... };
```

### 前向

```cpp
out[b, c, l] = (1/K) * Σ_k x[b, c, l*S + k]
```

### 反向

每个输出位置把 `grad_out / K` 平均回写到对应的 K 个输入位置。

## MaxPool2D

```cpp
class MaxPool2D : public Layer
{
    MaxPool2D(int kH, int kW, int sH = -1, int sW = -1);
};
```

`mask_` 形状 `[B, C, OH, OW * 2]`，把 argmax 的 `(ih, iw)` 以两个相邻 float 的形式存放（节省一次 reshape）。反向时按相同打包方式读出，把 `grad_out` 写到原最大值位置。

## AvgPool2D

实现与 AvgPool1D 类似，前向求 `kH * kW` 元素的平均，反向把 `grad_out / (kH * kW)` 平均分配回去。

## 使用说明

- **构造时 `stride` 默认为 `kernel`** → 非重叠池化（最常见）。
- **MaxPool 反向梯度稀疏**：仅最大位置接收梯度，其它位置为 0；这种「赢者通吃」是训练时常见的特征选择行为。
- **AvgPool 梯度均匀分布**：更稳定但缺少特征选择性，常用于全局平均池化（比如 [LAYERS/BASE/notes](../BASE/notes.md) 的 `GlobalAvgPool`）。
- **不带 padding**：当前实现假设 `(L - K)` 和 `(H/W - kH/kW)` 能被 `stride` 整除；调用方需要保证形状匹配，否则会丢弃边界元素。

## 与 Conv 的搭配

CNN1D 标准模板：

```txt
Conv1D + ReLU + MaxPool1D + (重复) → Flatten → Dense → Softmax
```

每个 `MaxPool1D(2)` 把序列长度减半，在保持感受野扩展的同时降低后续 Dense 层的参数量。

## 计算与内存

- **复杂度**：`O(B · C · OH · OW · kH · kW)`，没有矩阵乘。
- **训练显存**（仅 MaxPool）：`mask_` 与输出张量同量级 (1D) 或 2× (2D)。
- **PSRAM**：池化层本身不占权重，激活则按 batch / 通道 / 长度而定。
