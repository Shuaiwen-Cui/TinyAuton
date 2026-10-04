# TinyAI · 层 · 全连接层 — 设计说明 {#_1}

!!! info "实现依据与记录"
    本节接口以 `CODE/AIoTNode-TinyAuton-AI/middleware/` 为依据。源码摘录与串口输出包含历史记录；是否运行某项测试，请核对工程入口和启用开关。

!!! note "说明"
    `Dense` 是全连接层（fully-connected / linear），公式为 \( y = x W^\top + b \)。它是 MLP、分类头等场景的基本构件，权重采用 Xavier-uniform 初始化以缓解深网络的梯度衰减。

!!! abstract "Dense — 全连接层：每个输出看所有输入"
    \(\mathbf{y} = \mathbf{x} \cdot W^T + \mathbf{b}\) —— 最基础的"加权求和"。

## 算法直觉 {#_2}

### 每个神经元 = 一个加权投票 {#_3}

一个输出神经元的工作：

1. 查看**所有**输入特征
2. 给每个特征分配一个**权重**（重要特征权重大）
3. 加权求和 + 偏置
4. 经过激活函数决定要不要"激活"

### 权重矩阵的含义 {#_4}

\(W\) 的形状是 `[out_features, in_features]` —— 每一行对应一个输出神经元需要的所有权重。

### Xavier 初始化 {#xavier}

为什么需要特定的初始化？如果权重太小→信号逐层消失；权重太大→信号逐层爆炸。

**Xavier 均匀分布**：\(W \sim U\left[-\sqrt{6/(n_{in}+n_{out})},\; \sqrt{6/(n_{in}+n_{out})}\right]\)
保证输入输出的方差一致，信号在深层网络中能稳定传递。

!!! tip "Dense = 一个二线制的矩阵乘法"
    ```cpp
    // 等价实现
    Tensor output = matmul(input, weight.T()) + bias;  // [B, out_features]
    ```

---

## 数学定义 {#_5}

输入 `x` 形状 `[batch, in_features]`，输出 `y` 形状 `[batch, out_features]`：

\[
y_{b, o} = \sum_{i=0}^{F-1} W_{o, i}\, x_{b, i} + b_o
\]

权重张量形状 `[out_features, in_features]`（行 = 输出维），偏置形状 `[out_features]`。

### Xavier-uniform 初始化 {#xavier-uniform}

\[
W_{o, i} \sim \mathcal{U}(-L, L),\quad
L = \sqrt{\frac{6}{F_\text{in} + F_\text{out}}}
\]

偏置统一初始化为 0。

## 类定义 {#_6}

```cpp
class Dense : public Layer
{
public:
    Tensor weight;   // [out_features, in_features]
    Tensor bias;     // [out_features]   （use_bias=false 时为空）

#if TINY_AI_TRAINING_ENABLED
    Tensor dweight;  // 与 weight 同形状的梯度
    Tensor dbias;    // 与 bias 同形状的梯度
#endif

    Dense(int in_features, int out_features, bool use_bias = true);

    Tensor forward(const Tensor &x) override;     // [B, in_feat] → [B, out_feat]
    Tensor backward(const Tensor &grad_out) override;
    void   collect_params(std::vector<ParamGroup> &groups) override;

    int in_features()  const;
    int out_features() const;
};
```

## 反向传播 {#_7}

输入缓存 `x_cache_`：`forward` 内部 `clone()` 一份输入。反向公式：

\[
\frac{\partial L}{\partial W_{o, i}} \mathrel{+}= \sum_b \mathrm{grad\_out}_{b, o}\,x_{b, i}
\]

\[
\frac{\partial L}{\partial b_o} \mathrel{+}= \sum_b \mathrm{grad\_out}_{b, o}
\]

\[
\frac{\partial L}{\partial x_{b, i}} = \sum_o \mathrm{grad\_out}_{b, o}\,W_{o, i}
\]

注意 `dweight`、`dbias` 是 **累加** 写入；`Optimizer::zero_grad()` 在每个 batch 之前清零。

## 参数采集 {#_8}

```cpp
void Dense::collect_params(std::vector<ParamGroup> &groups)
{
    groups.push_back({&weight, &dweight});
    if (use_bias_) groups.push_back({&bias, &dbias});
}
```

如果构造时 `use_bias = false`，则 `bias` 为空张量，且不会进入 `collect_params`。

## 使用示例 {#_9}

```cpp
Dense fc1(F, 128);                  // [B, F] → [B, 128]
Dense fc2(128, num_classes);        // [B, 128] → [B, num_classes]

Sequential m;
m.add(new Dense(F, 128));
m.add(new ActivationLayer(ActType::RELU));
m.add(new Dense(128, num_classes));
m.add(new ActivationLayer(ActType::SOFTMAX));
```

也可以使用 `MLP` 便捷封装：

```cpp
MLP m({F, 128, 64, num_classes}, ActType::RELU);
```

它会自动插入 ReLU 与最终的 Softmax。

## 性能与内存 {#_10}

- **参数量**：`F_in * F_out + F_out`（含 bias）。
- **复杂度**：`forward` 是 `O(B * F_in * F_out)`，`backward` 同阶。
- **内存**：训练开启时多 ~2× 权重（`dweight`）+ ~1× bias（`dbias`）。
- **PSRAM 建议**：当 `F_in * F_out ≥ 64 KiB` 时把 `weight` 放进 PSRAM 视图（用 `Tensor::from_data`）。

## 与量化的衔接 {#_11}

- INT8 PTQ：使用 `quantize_weights(weight, qp)` 得到 `int8_t*`，再传入 `tiny_quant_dense_forward_int8` 完成纯整数推理。
- FP8：`calibrate(weight, TINY_DTYPE_FP8_E4M3)` + `quantize(weight, buf, qp)`，存储节省 4× 但需要在使用时 `dequantize`。

详见 [QUANT/INT](../../QUANT/INT/notes.md) 与 [QUANT/FP8](../../QUANT/FP8/notes.md)。
