# 说明

!!! note "说明"
    当前 `tiny_norm` 已包含三类归一化层：`LayerNorm`、`BatchNorm1D`、`BatchNorm2D`。其中 BatchNorm 系列默认以**推理模式**运行，直接使用 `running_mean/running_var`，便于端侧部署时对齐 PC 训练结果。

## LayerNorm

`LayerNorm` 沿最后一维（`feat`）归一化，不依赖 batch 统计量：

\[
\mu = \frac{1}{F}\sum_f x_f,\quad
\sigma^2 = \frac{1}{F}\sum_f (x_f-\mu)^2,\quad
\hat{x}_f = \frac{x_f-\mu}{\sqrt{\sigma^2+\varepsilon}}
\]

\[
y_f = \gamma_f \hat{x}_f + \beta_f
\]

- 参数：`gamma/beta` 形状为 `[feat]`。
- 默认 `epsilon=1e-5`。
- 适用输入：任意维度，只要最后一维等于 `feat`。

## BatchNorm1D（Dense/MLP）

- 输入/输出：`[batch, feat]`。
- 构造：`BatchNorm1D(int feat, float momentum=0.1f, float epsilon=1e-5f)`。
- 训练模式：按 batch 维统计每个特征的 `mu/var`，并更新运行统计量：
  - `running_mean = (1-m) * running_mean + m * mu`
  - `running_var  = (1-m) * running_var  + m * var`
- 推理模式：对每个特征融合为常数 `scale+shift`，避免重复计算。

## BatchNorm2D（Conv 输出）

- 输入/输出：`[N,C,L]`（Conv1D）或 `[N,C,H,W]`（Conv2D），输出同形状。
- 构造：`BatchNorm2D(int num_channels, float momentum=0.1f, float epsilon=1e-5f)`。
- 统计方式：对每个通道 `c`，在 `N * spatial` 上统计均值方差（除通道轴外全部归一化）。
- 推理模式同样使用 `running_mean/running_var` 做融合。

## 训练/推理模式切换

`BatchNorm1D/2D` 都提供：

```cpp
bn->set_training(true);   // 使用当前 batch 统计并更新 running stats
bn->set_training(false);  // 使用 running_mean/running_var
```

在 `Sequential` 里可统一切换：

```cpp
model.set_training_mode(true);   // 训练
model.set_training_mode(false);  // 推理
```

## 使用建议

- **部署推理**：先在 PC 训练并导入 `gamma/beta/running_mean/running_var`，端上保持 `training_mode=false`。
- **小 batch 训练**：若 batch 很小且波动大，优先考虑 `LayerNorm`。
- **MLP/CNN 示例**：`example_mlp` 新增 `BatchNorm1D` demo，`example_cnn` 新增 `BatchNorm2D` demo，可直接参考运行日志与模式切换流程。
