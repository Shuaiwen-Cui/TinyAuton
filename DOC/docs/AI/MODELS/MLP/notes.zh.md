# 说明

!!! note "说明"
    `MLP` 是多层感知机（Multi-Layer Perceptron）的便捷封装，继承自 `Sequential`。它接收一组维度列表 `{in, h1, h2, ..., out}`，自动添加 Dense + 激活，并在末端可选 Softmax。

## 类定义

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

## 构造逻辑

设 `dims = {d0, d1, ..., d_{N-1}}`，循环 `i = 0..N-2`：

1. `add(new Dense(d_i, d_{i+1}, use_bias))`。
2. 若 `i < N-2`：`add(new ActivationLayer(hidden_act))`（隐藏层后的激活）。
3. 若 `i == N-2` 且 `use_softmax == true`：`add(new ActivationLayer(SOFTMAX))`（输出层 softmax）。

## 使用示例

```cpp
// Iris：4 → 16 → 8 → 3
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

// 接 Trainer + CrossEntropy
Adam opt(1e-3f);
Trainer trainer(&model, &opt, LossType::CROSS_ENTROPY);
trainer.fit(train_data, cfg);
```

## 何时关闭 Softmax

`cross_entropy_forward` 已经内置 softmax，所以训练时模型最后一层加不加 Softmax **不会改变损失值**。但是：

- 想让 `model.predict()` / `accuracy()` 直接读概率 → 保持 `use_softmax = true`。
- 想要纯 logits 输出（例如做温度缩放、知识蒸馏）→ 设 `use_softmax = false`。

## 参数量与内存预算

设 `dims = {F, h1, h2, ..., C}`，则：

\[
\text{params} = \sum_{i=0}^{N-2} (d_i \cdot d_{i+1} + d_{i+1})
\]

例：`{4, 16, 8, 3}` ≈ `4*16 + 16 + 16*8 + 8 + 8*3 + 3 = 251` 个 float ≈ 1 KB。即使训练（多 m/v 缓冲）也只需要 ~3 KB，能完全放在 ESP32-S3 内部 SRAM。

## 与原始 Sequential 的关系

`MLP` 只重写了构造函数，所有 `forward / backward / summary / predict / accuracy` 都直接复用 `Sequential`。如果需要更复杂的拓扑（残差、分支、跳连），改用 `Sequential` 直接堆 `Layer*`，或继承 `Sequential` 自定义。
