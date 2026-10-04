# TinyAI · 模型 · MLP — 设计说明 {#_1}

!!! info "实现依据与记录"
    本节接口以 `CODE/AIoTNode-TinyAuton-AI/middleware/` 为依据。源码摘录与串口输出包含历史记录；是否运行某项测试，请核对工程入口和启用开关。

!!! note "说明"
    `MLP` 是多层感知机（Multi-Layer Perceptron）的便捷封装，继承自 `Sequential`。它接收一组维度列表 `{in, h1, h2, ..., out}`，自动添加 Dense + 激活，并在末端可选 Softmax。

!!! abstract "MLP — 多层感知机：通往复杂模式的万能钥匙"
    MLP = Dense + Activation 的堆叠。只要层数和宽度够，理论上可以拟合任何函数。

## 算法直觉 {#_2}

### 通用函数逼近器 {#_3}

MLP 是**通用函数逼近器**——意思是只要有足够多的神经元，它可以近似任何连续函数。

**结构**：
```
Input → [Dense → ReLU] × N → Dense (输出)
```

### 相比手工特征 {#_4}

传统机器学习需要工程师手动设计特征（"信号的第 2 个频段的能量和第 5 个频段能量的比值"），MLP 自动学什么特征重要。

### IRIS 示例（快速入门标准案例） {#iris}

```
Dense(4→64) → ReLU → Dense(64→3) → Softmax
```

- 输入：萼片长/宽、花瓣长/宽（4 维）
- 输出：3 个种类（Iris Setosa / Versicolor / Virginica）
- 隐藏层 64 个神经元足够学习花瓣和萼片的非线性组合

---

## 类定义 {#_5}

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

## 构造逻辑 {#_6}

设 `dims = {d0, d1, ..., d_{N-1}}`，循环 `i = 0..N-2`：

1. `add(new Dense(d_i, d_{i+1}, use_bias))`。
2. 若 `i < N-2`：`add(new ActivationLayer(hidden_act))`（隐藏层后的激活）。
3. 若 `i == N-2` 且 `use_softmax == true`：`add(new ActivationLayer(SOFTMAX))`（输出层 softmax）。

## 使用示例 {#_7}

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

## 何时关闭 Softmax {#softmax}

`cross_entropy_forward` 已经内置 softmax，所以训练时模型最后一层加不加 Softmax **不会改变损失值**。但是：

- 想让 `model.predict()` / `accuracy()` 直接读概率 → 保持 `use_softmax = true`。
- 想要纯 logits 输出（例如做温度缩放、知识蒸馏）→ 设 `use_softmax = false`。

## 参数量与内存预算 {#_8}

设 `dims = {F, h1, h2, ..., C}`，则：

\[
\text{params} = \sum_{i=0}^{N-2} (d_i \cdot d_{i+1} + d_{i+1})
\]

例：`{4, 16, 8, 3}` ≈ `4*16 + 16 + 16*8 + 8 + 8*3 + 3 = 251` 个 float ≈ 1 KB。即使训练（多 m/v 缓冲）也只需要 ~3 KB，能完全放在 ESP32-S3 内部 SRAM。

## 与原始 Sequential 的关系 {#sequential}

`MLP` 只重写了构造函数，所有 `forward / backward / summary / predict / accuracy` 都直接复用 `Sequential`。如果需要更复杂的拓扑（残差、分支、跳连），改用 `Sequential` 直接堆 `Layer*`，或继承 `Sequential` 自定义。
