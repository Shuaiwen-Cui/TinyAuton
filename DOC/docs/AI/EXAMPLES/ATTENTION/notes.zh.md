# TinyAI · 示例 · 注意力 — 设计说明 {#_1}

<!-- Original section links retained for compatibility. -->
<span id="2026-04-30"></span>

!!! info "训练输出与损失"
    当前交叉熵内部对 logits 计算 Softmax。历史示例含末尾 Softmax 层，作为当时配置保留；新模型应按损失接口决定输出层。MLP 的所谓 INT8 评估走浮点 predict 且标签集合不一致，不能据此评价 INT8 kernel。

!!! info "实现依据与记录"
    本节接口以 `CODE/AIoTNode-TinyAuton-AI/middleware/` 为依据。源码摘录与串口输出包含历史记录；是否运行某项测试，请核对工程入口和启用开关。

## 阅读这次试验 {#doc-阅读这次试验}

Iris 示例训练 99.17%、测试 90.00%；记录中的损失从 0.319672 降至 0.027159。该单次运行展示收敛，不足以证明多次随机划分或真实结构数据上的稳定性。

下方输出记录于 2026-04-30，所有行与数值原样保留。本次整理没有重新运行训练；设备与固件提交未在原记录中注明。先阅读结果和解释，再查看数据、模型及训练过程。

!!! info "示例说明"
    `example_attention` 演示了一个最小化的 Transformer 风格分类器：把 4 个 Iris 特征「编码为 4 个 token × 8 维」之后送入 Multi-Head Self-Attention，再用全局平均池化 + 线性分类器输出。整个示例用来验证 `Attention`、`GlobalAvgPool`、`Dense` 等组件可以在小数据集上完成端侧训练。

## 实测输出（2026-04-30） {#2026-04-30}

```txt
========================================
  tiny_ai  |  Attention Example (Iris)
========================================
Dataset split: 120 train / 30 test
Model summary:
  Dense(4, 32) + ReLU
  reshape [B, 32] -> [B, 4, 8]  (tokens, embed_dim)
  Attention(embed_dim=8, heads=2, head_dim=4)
  GlobalAvgPool
  Dense(8, 3)  [raw logits]

Training...
Epoch [ 20/100]  loss: 0.319672
Epoch [ 40/100]  loss: 0.110597
Epoch [ 60/100]  loss: 0.046880
Epoch [ 80/100]  loss: 0.033336
Epoch [100/100]  loss: 0.027159

--- Float32 Results ---
  Train accuracy: 99.17%
  Test  accuracy: 90.00%
example_attention  DONE
```

## 结果解读 {#_6}

- **训练收敛良好**：loss 从 `0.319 -> 0.027`，说明注意力分支与分类头都正常学习。
- **存在一定泛化差距**：`Train 99.17%` 对 `Test 90.00%`，在小数据集（Iris）上属于常见现象，可通过早停、正则或多次种子评估进一步确认稳定性。
- **与 MLP 对比**：当前输出里 Attention 与 MLP 的测试精度都在 `90%`，说明在该超参下 Attention 没有明显领先，但验证了 tiny_ai 的 Attention 训练链路可用。

## 数据来源 {#_2}

```cpp
Dataset dataset(
    &IRIS_X[0][0], IRIS_Y,
    IRIS_N_SAMPLES, IRIS_N_FEATURES, IRIS_N_CLASSES);

Dataset train_ds(dataset), test_ds(dataset);
dataset.split(0.2f, train_ds, test_ds, 42);
```

数据来自 `iris_data.hpp`，与 MLP 示例完全一样：150 条 4-维样本，3 个类别。

## 模型结构 {#_3}

```txt
Input          : [B, 4]
embed_proj     : Dense(4, 32) + ReLU       -> [B, 32]
reshape        : [B, 32] -> [B, 4, 8]      (4 tokens × embed_dim 8)
attn           : Attention(embed_dim=8, heads=2)  -> [B, 4, 8]
gap            : GlobalAvgPool             -> [B, 8]
classifier     : Dense(8, 3)               -> [B, 3]   (raw logits)
```

要点：

- `embed_proj` 把 4 维特征展开为 `4 * 8 = 32` 维向量，再 reshape 成 4 个 8 维 token。
- `Attention(embed_dim=8, heads=2, head_dim=4)` 为标准多头自注意力。
- 因为最后用 `cross_entropy_forward(logits, y)` 直接吃 logits，分类器**不需要** softmax 层。
- 由于数据流要在 `[B, 32]` 与 `[B, 4, 8]` 之间切换，本示例没有放进 `Sequential`，而是直接以「散件」组合。

## 训练循环 {#_4}

由于需要在中间做 `reshape_2d / reshape_3d`，示例自己写了循环，关键步骤如下：

```cpp
std::vector<ParamGroup> params;
embed_proj.collect_params(params);
attn.collect_params(params);
classifier.collect_params(params);
opt.init(params);

for (int epoch = 0; epoch < 100; epoch++)
{
    train_ds.shuffle(epoch + 1);
    while (true)
    {
        int actual = train_ds.next_batch(X_batch, y_batch, 16);
        if (actual == 0) break;

        // forward
        Tensor e0 = embed_proj.forward(X_batch);
        Tensor e1 = embed_act.forward(e0);
        e1.reshape_3d(actual, 4, 8);
        Tensor a0     = attn.forward(e1);
        Tensor p0     = gap.forward(a0);
        Tensor logits = classifier.forward(p0);

        float loss = cross_entropy_forward(logits, y_batch);

        // backward
        opt.zero_grad(params);
        Tensor dlogits = cross_entropy_backward(logits, y_batch);
        Tensor dp0     = classifier.backward(dlogits);
        Tensor da0     = gap.backward(dp0);
        Tensor de1     = attn.backward(da0);
        de1.reshape_2d(actual, 32);
        Tensor de0     = embed_act.backward(de1);
        embed_proj.backward(de0);
        opt.step(params);
    }
}
```

注意：

- 必须先在所有「会更新参数」的层上调用 `collect_params`，再 `opt.init(params)`，否则 SGD/Adam 没有任何参数可更新。
- 反向链条与 forward 严格对称，包括 `reshape_3d → reshape_2d` 这一对。
- `Attention` 内部已经为 backward 缓存了所有需要的中间张量，所以无需手动保存 `Q/K/V/A`。

## 评估 {#_5}

```cpp
auto eval_accuracy = [&](Dataset &ds, const char *tag) {
    ds.reset();
    while (true)
    {
        int actual = ds.next_batch(Xb, yb, batch_size);
        if (actual == 0) break;
        Tensor e0     = embed_proj.forward(Xb);
        Tensor e1     = embed_act.forward(e0);
        e1.reshape_3d(actual, 4, 8);
        Tensor a0     = attn.forward(e1);
        Tensor p0     = gap.forward(a0);
        Tensor logits = classifier.forward(p0);
        // argmax over logits...
    }
};
```

由于 `Sequential::predict` 不便表达 reshape，示例用了内联的 argmax 逻辑。

## 资源消耗 {#_7}

- 注意力层在每个 batch 都会缓存 `[B, S, F]`、`[B, H, S, dh]`、`[B, H, S, S]` 等张量；当 batch=16, S=4, F=8 时仍然非常小（约几 KB）。
- 整个示例不到 100 KB 训练时内存，能在 ESP32-S3 主 RAM 中跑通；如果数据更大，应考虑把 `embed_proj` 与 `attn` 的中间张量分配到 PSRAM。

## 入口 {#_8}

```cpp
extern "C" void example_attention(void);
example_attention();
```
