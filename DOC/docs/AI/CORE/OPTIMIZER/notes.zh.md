# 说明

!!! note "说明"
    `tiny_optimizer` 提供两种针对 ESP32-S3 内存预算优化的梯度下降优化器：带动量与 L2 正则的 SGD，以及 Adam（lite 版）。所有优化器都通过 `ParamGroup` 组成的 `std::vector` 与各层的可学习参数 / 梯度对接。

## ParamGroup

```cpp
struct ParamGroup
{
    Tensor *param;  // 权重 / 偏置 张量
    Tensor *grad;   // 对应的梯度张量
};
```

每个可训练层（`Dense`、`Conv1D`、`Conv2D`、`LayerNorm`、`Attention`）都重载 `Layer::collect_params()`，把自己的 `(weight, dweight)`、`(bias, dbias)` 等成对压入 `std::vector<ParamGroup>`。`Sequential::collect_params()` 自动汇总整个网络。

## Optimizer 抽象基类

```cpp
class Optimizer
{
public:
    virtual void init(const std::vector<ParamGroup> &groups) = 0;  // 初始化内部缓冲
    virtual void step(std::vector<ParamGroup> &groups)       = 0;  // 一步更新
    virtual void zero_grad(std::vector<ParamGroup> &groups);       // 清零梯度
};
```

调用顺序：

1. **构造**：`SGD opt(lr, mom)` 或 `Adam opt(lr, β1, β2, ε)`。
2. **采集参数**：`model.collect_params(params)`。
3. **初始化**：`opt.init(params)`。仅在此时根据 `params.size()` 与每个张量形状分配动量 / 一二阶矩缓冲。
4. **训练循环**：每个 batch 执行 `opt.zero_grad(params)` → forward → backward → `opt.step(params)`。

## SGD（带动量与 L2）

```cpp
SGD(float lr = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f);
```

更新公式：

\[
g \leftarrow \nabla_\theta + \lambda\,\theta\quad(\text{若}~\lambda > 0)
\]

\[
v \leftarrow \mu\,v + g\quad(\text{若}~\mu > 0)
\]

\[
\theta \leftarrow \theta - \eta \cdot v
\]

参数：

- `lr`：学习率 \(\eta\)。
- `momentum`：动量系数 \(\mu\)。设为 0 时退化为标准 SGD。
- `weight_decay`：L2 正则系数 \(\lambda\)。

`init()` 会为每个参数分配同形状的 velocity 张量；`zero_grad()` 默认实现已经在基类提供。

## Adam（lite 版）

```cpp
Adam(float lr     = 1e-3f,
     float beta1  = 0.9f,
     float beta2  = 0.999f,
     float epsilon = 1e-8f,
     float weight_decay = 0.0f);
```

每步：

\[
g \leftarrow \nabla_\theta + \lambda\,\theta
\]

\[
m \leftarrow \beta_1 m + (1-\beta_1) g,\quad
v \leftarrow \beta_2 v + (1-\beta_2) g^2
\]

实现采用「整体 lr 偏差校正」节省每元素计算：

\[
\eta_t = \eta\;\frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
\]

\[
\theta \leftarrow \theta - \eta_t\;\frac{m}{\sqrt{v} + \varepsilon}
\]

`init()` 为每个参数分配 `m` 与 `v` 两个张量；`step()` 内部时间步 `t_` 自动 +1。

!!! tip "实战建议"
    - 对结构健康监测、生物信号等小数据 / 不稳定数据，优先用 Adam（默认参数即可）。
    - 对高度稀疏 / 大 batch 训练，可试 SGD + 较大 lr + 动量 0.9。
    - `weight_decay > 0` 等价于 PyTorch 的 L2 正则；只对权重生效，建议不要把 bias 一起 decay（`tiny_ai` 中 bias 也参与，但量级可忽略）。

## 显存与 PSRAM 影响

- **SGD**：每个参数额外一份 velocity → 内存约 ×2。
- **Adam**：每个参数额外两份 (m, v) → 内存约 ×3。

如果模型权重已经放进 PSRAM，建议同步把动量缓冲也放 PSRAM。`Tensor` 默认走 `TINY_AI_MALLOC`，需要时可在外层把权重张量替换为 `Tensor::from_data(psram_buf, ...)` 视图。

## 与 Trainer 的协作

`Trainer::ensure_params_collected()` 在第一次 `fit()` 时执行：

```cpp
model_->collect_params(params_);
optimizer_->init(params_);
params_collected_ = true;
```

之后每个 batch：

```cpp
optimizer_->zero_grad(params_);
auto logits = model_->forward(X_batch);
auto grad   = loss_backward(logits, ..., loss_type_, y_batch);
model_->backward(grad);
optimizer_->step(params_);
```

因此你完全可以重新实现一个自定义优化器（继承 `Optimizer` 并实现 `init / step` 即可），无需改动模型层或 Trainer。
