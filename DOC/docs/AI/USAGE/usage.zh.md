# TinyAI 使用方法

当前示例以 Dataset、MLP 和 Trainer 头文件为依据。Dataset 没有无参构造函数；评估使用 `evaluate_accuracy()` 或 `evaluate_loss()`。交叉熵接收 logits，因此这里关闭末尾 Softmax。

```cpp
#include "tiny_ai.h"
#include <cstdio>

void tiny_ai_usage_demo()
{
    using namespace tiny;
    static const float X[16] = {
        0, 0, 0, 1, 1, 0, 1, 1,
        3, 3, 3, 4, 4, 3, 4, 4
    };
    static const int y[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    Dataset full(X, y, 8, 2, 2);
    Dataset train(full), test(full);
    full.split(0.25f, train, test, 42);
    MLP model({2, 4, 2}, ActType::RELU, false);
    Adam opt(1e-3f);
    Trainer trainer(&model, &opt, LossType::CROSS_ENTROPY);
    Trainer::Config cfg;
    cfg.epochs = 50;
    cfg.batch_size = 2;
    trainer.fit(train, cfg, &test);
    printf("test accuracy = %.2f%%\n",
           trainer.evaluate_accuracy(test) * 100.0f);
}
```

输入数组必须比引用它们的 Dataset 活得更久。示例需要启用 `TINY_AI_TRAINING_ENABLED`；小型合成数据仅展示调用方法。

[MLP 历史示例](../EXAMPLES/MLP/notes.md) · [训练器](../TRAIN/TRAINER/notes.md) · [量化接口](../QUANT/INT/notes.md)

## 原始引入方式与历史示例 {#auton-original-usage}

旧流程中的 `Dataset train, test` 和 `quantize_weights(model_layer.weight, ...)` 是不能直接编译的示意。当前调用以上方为准，旧内容保留供对照。

<details class="auton-source" markdown="1">
<summary>展开原始页面</summary>

# 使用说明 {#_1}

!!! info "实现依据与记录"
    本节接口以 `CODE/AIoTNode-TinyAuton-AI/middleware/` 为依据。源码摘录与串口输出包含历史记录；是否运行某项测试，请核对工程入口和启用开关。

!!! info "使用说明"
    该文档是对 `tiny_ai` 模块的使用说明。

## 整体引入 TinyAI {#tinyai}

!!! info
    适用于绝大多数 C++ 项目，单条 `#include` 即可拉入张量、层、模型、量化、训练全部 API。

```cpp
#include "tiny_ai.h"
```

## 分模块引入 TinyAI {#tinyai_1}

!!! info
    适用于需要精确控制依赖、或者只做推理 / 只用量化工具的项目。

```cpp
// 顶层配置（必含，提供宏与错误码）
#include "tiny_ai_config.h"

// 核心 (core/)
#include "tiny_tensor.hpp"      // N 维 float32 Tensor
#include "tiny_activation.hpp"  // 激活函数
#include "tiny_loss.hpp"        // 损失函数
#include "tiny_optimizer.hpp"   // SGD / Adam

// 网络层 (layers/)
#include "tiny_layer.hpp"       // Layer 抽象 / ActivationLayer / Flatten / GlobalAvgPool
#include "tiny_dense.hpp"       // 全连接层
#include "tiny_conv.hpp"        // Conv1D / Conv2D
#include "tiny_pool.hpp"        // MaxPool / AvgPool 1D & 2D
#include "tiny_norm.hpp"        // LayerNorm
#include "tiny_attention.hpp"   // 多头自注意力

// 模型 (models/)
#include "tiny_sequential.hpp"  // Sequential
#include "tiny_mlp.hpp"         // MLP
#include "tiny_cnn.hpp"         // CNN1D

// 量化 (quant/)
#include "tiny_quant_config.h"  // dtype 与参数结构
#include "tiny_quant.h"         // C 接口 INT8 / INT16
#include "tiny_quant.hpp"       // C++ Tensor 级 PTQ 工具
#include "tiny_fp8.hpp"         // FP8 E4M3FN / E5M2

// 训练 (train/)
#include "tiny_dataset.hpp"     // 数据集 + 小批量迭代
#include "tiny_trainer.hpp"     // 训练循环
```

## 典型使用流程 {#_2}

```cpp
using namespace tiny;

// 1) 准备数据
Dataset full(X, y, N, F, C);
Dataset train, test;
full.split(0.2f, train, test, 42);

// 2) 构建模型
MLP model({F, 16, 8, C}, ActType::RELU);
model.summary();

// 3) 准备优化器与训练器
Adam opt(1e-3f);
Trainer trainer(&model, &opt, LossType::CROSS_ENTROPY);

Trainer::Config cfg;
cfg.epochs = 100;
cfg.batch_size = 16;

// 4) 训练 + 评估
trainer.fit(train, cfg, &test);
printf("Test acc = %.2f\n", trainer.evaluate_accuracy(test) * 100.0f);

// 5) 推理 / 量化（可选 PTQ）
QuantParams qp;
int8_t *w_int8 = quantize_weights(model_layer.weight, qp);
```

!!! tip
    具体的使用方法请参考 [EXAMPLES](../EXAMPLES/MLP/notes.md) 中的三个完整示例。

</details>
