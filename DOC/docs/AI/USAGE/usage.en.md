# TinyAI usage

This example follows the Dataset, MLP and Trainer headers. Dataset has no default constructor; evaluation uses `evaluate_accuracy()` or `evaluate_loss()`. Cross entropy consumes logits, so the final Softmax is disabled here.

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

Input arrays must outlive every referencing Dataset. Enable `TINY_AI_TRAINING_ENABLED`; the small synthetic data demonstrates calls without promising accuracy.

[Historical MLP example](../EXAMPLES/MLP/notes.md) · [Trainer](../TRAIN/TRAINER/notes.md) · [Quantization](../QUANT/INT/notes.md)

## Original includes and historical example {#auton-original-usage}

The old `Dataset train, test` and `quantize_weights(model_layer.weight, ...)` are non-compilable schematic calls. Use the current example above; original content remains for comparison.

<details class="auton-source" markdown="1">
<summary>Expand original page</summary>

# USAGE {#usage}

!!! info "Implementation and records"
    APIs in this section are based on `CODE/AIoTNode-TinyAuton-AI/middleware/`. Source excerpts and serial output include historical records; check the project entry and enabled selectors before reproducing a test.

!!! info "USAGE"
    This document explains how to consume the `tiny_ai` module.

## INCLUDE THE WHOLE TINY_AI {#include-the-whole-tiny_ai}

!!! info
    Recommended for most C++ projects: a single `#include` pulls in tensors, layers, models, quantisation and training.

```cpp
#include "tiny_ai.h"
```

## INCLUDE PER-MODULE {#include-per-module}

!!! info
    Use this when you need fine-grained control over dependencies, or you only want the quantisation utilities for inference-only deployments.

```cpp
// Top-level config (always required: macros + error codes)
#include "tiny_ai_config.h"

// Core (core/)
#include "tiny_tensor.hpp"      // N-D float32 Tensor
#include "tiny_activation.hpp"  // activations
#include "tiny_loss.hpp"        // loss functions
#include "tiny_optimizer.hpp"   // SGD / Adam

// Layers (layers/)
#include "tiny_layer.hpp"       // abstract Layer / ActivationLayer / Flatten / GlobalAvgPool
#include "tiny_dense.hpp"       // fully-connected layer
#include "tiny_conv.hpp"        // Conv1D / Conv2D
#include "tiny_pool.hpp"        // MaxPool / AvgPool 1D & 2D
#include "tiny_norm.hpp"        // LayerNorm
#include "tiny_attention.hpp"   // multi-head self-attention

// Models (models/)
#include "tiny_sequential.hpp"  // Sequential
#include "tiny_mlp.hpp"         // MLP
#include "tiny_cnn.hpp"         // CNN1D

// Quantisation (quant/)
#include "tiny_quant_config.h"  // dtype + param struct
#include "tiny_quant.h"         // C API for INT8 / INT16
#include "tiny_quant.hpp"       // C++ Tensor-level PTQ helpers
#include "tiny_fp8.hpp"         // FP8 E4M3FN / E5M2

// Training (train/)
#include "tiny_dataset.hpp"     // dataset + mini-batch iteration
#include "tiny_trainer.hpp"     // training loop
```

## TYPICAL WORKFLOW {#typical-workflow}

```cpp
using namespace tiny;

// 1) Prepare data
Dataset full(X, y, N, F, C);
Dataset train, test;
full.split(0.2f, train, test, 42);

// 2) Build model
MLP model({F, 16, 8, C}, ActType::RELU);
model.summary();

// 3) Optimiser + trainer
Adam opt(1e-3f);
Trainer trainer(&model, &opt, LossType::CROSS_ENTROPY);

Trainer::Config cfg;
cfg.epochs = 100;
cfg.batch_size = 16;

// 4) Train + evaluate
trainer.fit(train, cfg, &test);
printf("Test acc = %.2f\n", trainer.evaluate_accuracy(test) * 100.0f);

// 5) Inference / optional PTQ
QuantParams qp;
int8_t *w_int8 = quantize_weights(model_layer.weight, qp);
```

!!! tip
    For complete walk-throughs, see the three end-to-end demos under [EXAMPLES](../EXAMPLES/MLP/notes.md).

</details>
