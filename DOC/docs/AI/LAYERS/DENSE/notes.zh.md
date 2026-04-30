# 说明

!!! note "说明"
    `Dense` 是全连接层（fully-connected / linear），公式为 \( y = x W^\top + b \)。它是 MLP、分类头等场景的基本构件，权重采用 Xavier-uniform 初始化以缓解深网络的梯度衰减。

## 数学定义

输入 `x` 形状 `[batch, in_features]`，输出 `y` 形状 `[batch, out_features]`：

\[
y_{b, o} = \sum_{i=0}^{F-1} W_{o, i}\, x_{b, i} + b_o
\]

权重张量形状 `[out_features, in_features]`（行 = 输出维），偏置形状 `[out_features]`。

### Xavier-uniform 初始化

\[
W_{o, i} \sim \mathcal{U}(-L, L),\quad
L = \sqrt{\frac{6}{F_\text{in} + F_\text{out}}}
\]

偏置统一初始化为 0。

## 类定义

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

## 反向传播

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

## 参数采集

```cpp
void Dense::collect_params(std::vector<ParamGroup> &groups)
{
    groups.push_back({&weight, &dweight});
    if (use_bias_) groups.push_back({&bias, &dbias});
}
```

如果构造时 `use_bias = false`，则 `bias` 为空张量，且不会进入 `collect_params`。

## 使用示例

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

## 性能与内存

- **参数量**：`F_in * F_out + F_out`（含 bias）。
- **复杂度**：`forward` 是 `O(B * F_in * F_out)`，`backward` 同阶。
- **内存**：训练开启时多 ~2× 权重（`dweight`）+ ~1× bias（`dbias`）。
- **PSRAM 建议**：当 `F_in * F_out ≥ 64 KiB` 时把 `weight` 放进 PSRAM 视图（用 `Tensor::from_data`）。

## 与量化的衔接

- INT8 PTQ：使用 `quantize_weights(weight, qp)` 得到 `int8_t*`，再传入 `tiny_quant_dense_forward_int8` 完成纯整数推理。
- FP8：`calibrate(weight, TINY_DTYPE_FP8_E4M3)` + `quantize(weight, buf, qp)`，存储节省 4× 但需要在使用时 `dequantize`。

详见 [QUANT/INT](../../QUANT/INT/notes.md) 与 [QUANT/FP8](../../QUANT/FP8/notes.md)。
