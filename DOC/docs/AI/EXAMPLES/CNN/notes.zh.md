# 说明

!!! info "示例说明"
    `example_cnn` 现在包含三部分：1-D CNN 训练、FP8 E4M3FN 压缩演示、以及新增的 `BatchNorm2D` demo（展示 BN 在 Conv 场景下的训练/推理切换）。

## 数据来源

```cpp
namespace tiny_data
{
constexpr int SIG_N_CLASSES        = 3;
constexpr int SIG_SAMPLES_PER_CLASS = 50;
constexpr int SIG_N_SAMPLES         = SIG_N_CLASSES * SIG_SAMPLES_PER_CLASS;  // 150
constexpr int SIG_SIGNAL_LEN        = 64;

void generate_signal_dataset(float *X, int *Y);
}
```

`generate_signal_dataset` 在 PSRAM 缓冲里写入 150 条 64 点合成信号（典型为正弦、方波、加噪等），再用作 `Dataset` 的源数据。

## 模型结构

```cpp
CNN1DConfig cfg;
cfg.in_channels   = 1;
cfg.signal_length = 64;
cfg.num_classes   = 3;
cfg.filters       = {8, 16};
cfg.kernels       = {5, 3};
cfg.pool_size     = 2;
cfg.fc_units      = 32;

CNN1D model(cfg);
```

展开为：

```txt
Conv1D(1→8,  k=5)  + ReLU + MaxPool1D(2)   -> [B, 8, (64-5+1)/2 = 30]
Conv1D(8→16, k=3)  + ReLU + MaxPool1D(2)   -> [B,16, (30-3+1)/2 = 14]
Flatten                                     -> [B, 16*14 = 224]
Dense(224 → 32) + ReLU
Dense(32  → 3)  + Softmax
```

`model.flat_features()` 在构造后返回展平维度（这里是 224），便于检查。

## 自定义训练循环

由于 `Dataset` 输出的是 `[B, L]`，而 `Conv1D` 期待 `[B, 1, L]`，这个示例没有直接调用 `Trainer::fit`，而是手写训练循环：

```cpp
std::vector<ParamGroup> params;
model.collect_params(params);
opt.init(params);

for (int epoch = 0; epoch < 50; epoch++)
{
    train_ds.shuffle(epoch + 1);
    while (next_batch returns > 0)
    {
        Tensor X3(actual, 1, SIG_SIGNAL_LEN);  // [B, L] → [B, 1, L]
        ... copy ...

        Tensor logits = model.forward(X3);
        float  loss   = cross_entropy_forward(logits, y_batch);
        opt.zero_grad(params);
        Tensor grad   = cross_entropy_backward(logits, y_batch);
        model.backward(grad);
        opt.step(params);
    }
}
```

这种「手写循环」的写法是 `Trainer::fit` 用 `Sequential::forward(X_batch)` 的局限的对策，也展示了 `Optimizer + Sequential + Dataset` 一组件拆开来用的灵活性。

## 评估

```cpp
auto eval_accuracy = [&](Dataset &ds, const char *tag) {
    ds.reset();
    ...
    Tensor X3(actual, 1, SIG_SIGNAL_LEN);
    model.predict(X3, yp);
    ...
};
eval_accuracy(train_ds, "Train");
eval_accuracy(test_ds,  "Test ");
```

每个 batch 同样完成 `[B, L] → [B, 1, L]` 的 reshape，再调用 `Sequential::predict` 取 argmax。

## FP8 压缩演示

```cpp
Tensor w(16, 8, 3);      // 模拟 Conv1D 权重
QuantParams qp = calibrate(w, TINY_DTYPE_FP8_E4M3);

uint8_t *fp8_buf = (uint8_t *)TINY_AI_MALLOC(w.size);
quantize(w, fp8_buf, qp);

Tensor w_recon = Tensor::zeros_like(w);
dequantize(fp8_buf, w_recon, qp);
```

打印：

- 4:1 压缩比（`size * 4` 字节 → `size` 字节）。
- 最大绝对误差 `max_err`、均方误差 `mse`。
- 前 4 个元素的原值 / 字节编码 / 重建值。

可作为对比 INT8 压缩的参考。

## 新增：BatchNorm2D Demo

`example_cnn` 在末尾新增 `batchnorm2d_demo(train_ds, test_ds)`（仅训练版编译）：

```txt
Conv1D(1,8,5)  + BN2D(8)  + ReLU + MaxPool1D(2)
Conv1D(8,16,3) + BN2D(16) + ReLU + MaxPool1D(2)
Flatten -> Dense(224,32)+ReLU -> Dense(32,3)+Softmax
```

关键点：

- BN2D 用于 `Conv1D` 输出 `[B, C, L]`，按通道在 `N*L` 上统计。
- `training_mode=ON` 时更新 `running_mean/running_var`。
- `training_mode=OFF` 时使用 running stats 推理。
- demo 同时打印 train/test 在两种模式下的准确率，并输出首层 BN 的通道统计值。

## 实测输出（2026-04-30）

```txt
========================================
  tiny_ai  |  CNN Example (Signal)
========================================
Generated 150 signals (3 classes × 50 samples, 64 pts each)
Split: 120 train / 30 test
Sequential model  (11 layers)
--------------------
  [ 0] conv1d
  [ 1] activation
  [ 2] max_pool1d
  [ 3] conv1d
  [ 4] activation
  [ 5] max_pool1d
  [ 6] flatten
  [ 7] dense
  [ 8] activation
  [ 9] dense
  [10] activation
--------------------
  Flat features after conv blocks: 224
Training...
Epoch [ 10/ 50]  loss: 0.558837
Epoch [ 20/ 50]  loss: 0.552400
Epoch [ 30/ 50]  loss: 0.551829
Epoch [ 40/ 50]  loss: 0.551640
Epoch [ 50/ 50]  loss: 0.551561

--- Float32 Results ---
  Train accuracy: 100.00%
  Test  accuracy: 100.00%

--- FP8 E4M3FN Compression Demo ---
  E4M3 scale = 0.008571
  Size: FP32=1536 bytes  FP8=384 bytes  (4:1 compression)
  Max abs error = 0.990000   MSE = 0.16210581
  w[0] orig=-3.8400  fp8=0xC4  recon=-3.0000
  ...

[BN2D] training_mode = ON
  [train mode] Train acc: 100.00%
  [train mode] Test  acc: 100.00%
[BN2D] training_mode = OFF  (running stats active)
  [infer mode] Train acc: 100.00%
  [infer mode] Test  acc: 100.00%
BN2D[0] running_mean (first 4 ch): 0.0000 -0.0023 -0.0046 0.0108
BN2D[0] running_var  (first 4 ch): 0.2838 1.0601 0.6727 2.8801
example_cnn  DONE
```

## 结果解读

- **分类任务已学会**：训练和测试均为 `100%`，数据分布对当前模型较友好。
- **loss 平台较高但准确率满分**：softmax + cross-entropy 下只要预测类别正确，accuracy 可高；loss 仍受置信度影响，所以两者并不矛盾。
- **FP8 压缩损失明显**：`max_err=0.99`、`MSE=0.162` 表明 E4M3 对该权重范围有较强截断/量化误差；这是一组“压缩误差展示值”，不是部署后精度结论。
- **BN2D 两种模式表现一致**：说明 running stats 与当前数据分布匹配良好，切换到推理模式没有造成精度退化。

## 入口

```cpp
extern "C" void example_cnn(void);
example_cnn();
```
