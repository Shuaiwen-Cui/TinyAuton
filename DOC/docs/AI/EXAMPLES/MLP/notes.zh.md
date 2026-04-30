# 说明

!!! info "示例说明"
    `example_mlp` 现在包含两部分：主流程（Iris 上的 MLP 训练 + INT8 PTQ 演示）以及新增的 `BatchNorm1D` 专项 demo，用于展示训练模式与推理模式切换。

## 数据来源

数据来自 `example/data/iris_data.hpp`：

```cpp
namespace tiny_data
{
constexpr int IRIS_N_SAMPLES  = 150;
constexpr int IRIS_N_FEATURES = 4;
constexpr int IRIS_N_CLASSES  = 3;

extern const float IRIS_X[IRIS_N_SAMPLES][IRIS_N_FEATURES];
extern const int   IRIS_Y[IRIS_N_SAMPLES];
}
```

`IRIS_X` 与 `IRIS_Y` 直接以 `static const` 数组形式嵌入到固件，无需任何文件 I/O。

## 模型结构

```cpp
MLP model({4, 16, 8, 3}, ActType::RELU, /*use_softmax=*/true, /*use_bias=*/true);
```

展开为：

```txt
Dense(4 → 16) + ReLU
Dense(16 → 8) + ReLU
Dense(8  → 3) + Softmax
```

## 训练配置

- **数据划分**：`dataset.split(0.2f, train_ds, test_ds, 42)` → 120 训练 / 30 测试。
- **优化器**：`Adam(lr=1e-3, β1=0.9, β2=0.999)`。
- **损失**：`LossType::CROSS_ENTROPY`（输入 raw logits，softmax 内置在损失里）。
- **超参**：`epochs=100, batch_size=16, print_every=20`。

## 训练流程

```cpp
Trainer trainer(&model, &opt, LossType::CROSS_ENTROPY);
Trainer::Config cfg;
cfg.epochs      = 100;
cfg.batch_size  = 16;
cfg.print_every = 20;
trainer.fit(train_ds, cfg, &test_ds);

float train_acc = trainer.evaluate_accuracy(train_ds);
float test_acc  = trainer.evaluate_accuracy(test_ds);
```

`Trainer::fit` 会：

1. 第一次调用时 `model.collect_params(params)` 与 `opt.init(params)`。
2. 每个 epoch 用 `train_ds.shuffle(epoch + 1)` 洗牌。
3. 内层循环按 `batch_size = 16` 滚动 `next_batch`，依次执行 forward / loss_forward / loss_backward / model.backward / opt.step。
4. 每 `print_every` 个 epoch 打印 `loss + val_acc`。

## INT8 PTQ 演示

```cpp
QuantParams wp = calibrate(demo_w, TINY_DTYPE_INT8);
int8_t *w_int8 = (int8_t *)TINY_AI_MALLOC(demo_w.size);
tiny_quant_params_t cp = wp.to_c();
tiny_quant_f32_to_int8(demo_w.data, w_int8, demo_w.size, &cp);
```

打印 `scale / zero_point / quantised value / dequantised value`，让你直观感受到对称量化的精度。完整 INT8 推理见 [QUANT/INT/notes](../../QUANT/INT/notes.md) 中的 `tiny_quant_dense_forward_int8`。

## 新增：BatchNorm1D Demo

`example_mlp` 在主流程后会调用 `batchnorm1d_demo(train_ds, test_ds)`（仅在 `TINY_AI_TRAINING_ENABLED` 打开时编译）：

```txt
Dense(4,16) + BN1D(16) + ReLU
Dense(16,8) + BN1D(8)  + ReLU
Dense(8,3)  + Softmax
```

关键行为：

- `model.set_training_mode(true)`：BN 使用当前 batch 统计量并累计 `running_mean/running_var`。
- `model.set_training_mode(false)`：BN 改用累计的 running stats 进行推理。
- demo 会打印首层 BN 的前几个 `running_mean/running_var`，用于确认统计量已更新。

## 实测输出（2026-04-30）

```txt
========================================
  tiny_ai  |  MLP Example (Iris)
========================================
Dataset split: 120 train / 30 test
Sequential model  (6 layers)
--------------------
  [ 0] dense
  [ 1] activation
  [ 2] dense
  [ 3] activation
  [ 4] dense
  [ 5] activation
--------------------
Training...
Epoch [ 20/100]  loss: 0.820218  val_acc: 0.7667
Epoch [ 40/100]  loss: 0.690300  val_acc: 0.9333
Epoch [ 60/100]  loss: 0.629859  val_acc: 0.9000
Epoch [ 80/100]  loss: 0.602956  val_acc: 0.9000
Epoch [100/100]  loss: 0.586117  val_acc: 0.9000

--- Float32 Results ---
  Train accuracy: 97.50%
  Test  accuracy: 90.00%

--- INT8 PTQ Inference ---
  Quantisation demo: calibrating weight tensor...
  Weight scale = 0.006299  zero_point = 0
  Original w[0]=-0.8000  Quantised=-127  Dequantised=-0.8000
  INT8 accuracy (whole dataset): 8.67%

[BN1D] training_mode = ON
[train mode] Train: 97.50%  Test: 80.00%
[BN1D] training_mode = OFF  (running stats active)
[infer mode] Train: 99.17%  Test: 96.67%
BN1D[0] running_mean (first 4): -0.0179 -0.0019 0.0345 0.0276
BN1D[0] running_var  (first 4): 0.6317 0.1680 0.9003 0.7524
example_mlp  DONE
```

## 结果解读

- **FP32 主流程正常收敛**：`Train 97.50% / Test 90.00%`，对 Iris 来说属于可接受范围。
- **INT8=8.67% 不是“量化后真实性能”**：当前 `run_int8_inference()` 实际调用的是 `model.predict` 浮点路径，但标签传入是整集 `IRIS_Y` 且预测输入是 `X_test`，两者样本集合不一致，准确率统计失真，文档不要把它当作 INT8 推理结论。
- **BN1D 推理模式优于训练模式评估**：`training_mode=OFF` 下测试从 `80.00% -> 96.67%`，符合 BatchNorm 在推理时使用稳定 running stats 的预期。
- **running stats 已有效累积**：`running_mean/running_var` 已为非平凡值，说明 BN 在训练阶段确实在更新统计量。

## 入口

`tiny_ai.h` 暴露 `void example_mlp(void)` 供 C / C++ 调用：

```cpp
extern "C" void example_mlp(void);

// 在 AIoTNode.cpp / app_main.c 中：
example_mlp();
```
