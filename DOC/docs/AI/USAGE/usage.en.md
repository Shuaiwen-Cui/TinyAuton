# USAGE

!!! info "USAGE"
    This document explains how to consume the `tiny_ai` module.

## INCLUDE THE WHOLE TINY_AI

!!! info
    Recommended for most C++ projects: a single `#include` pulls in tensors, layers, models, quantisation and training.

```cpp
#include "tiny_ai.h"
```

## INCLUDE PER-MODULE

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

## TYPICAL WORKFLOW

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
