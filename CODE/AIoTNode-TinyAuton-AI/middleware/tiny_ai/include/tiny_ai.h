/**
 * @file tiny_ai.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_ai | Unified entry-point header — include this to access all AI
 *        functionality in the tiny_ai middleware.
 *
 * @details
 *  Dependency chain:  tiny_toolbox → tiny_math → tiny_dsp → tiny_ai
 *
 *  Include hierarchy:
 *    include/tiny_ai_config.h      — feature flags, memory macros, error codes
 *    quant/tiny_quant_config.h     — dtype enum, QuantParams C struct
 *    core/tiny_tensor.hpp          — N-D float32 tensor
 *    core/tiny_activation.hpp      — activation functions (ReLU, Sigmoid …)
 *    core/tiny_loss.hpp            — loss functions (MSE, CrossEntropy …)
 *    core/tiny_optimizer.hpp       — SGD, Adam
 *    quant/tiny_fp8.hpp            — FP8 E4M3FN / E5M2 software emulation
 *    quant/tiny_quant.h            — C-compatible INT8/INT16 quantisation API
 *    quant/tiny_quant.hpp          — C++ QuantParams + calibrate/quantize helpers
 *    layers/tiny_layer.hpp         — abstract Layer, ActivationLayer, Flatten,
 *                                    GlobalAvgPool
 *    layers/tiny_dense.hpp         — fully-connected Dense layer
 *    layers/tiny_conv.hpp          — Conv1D / Conv2D
 *    layers/tiny_pool.hpp          — MaxPool1D/2D, AvgPool1D/2D
 *    layers/tiny_norm.hpp          — LayerNorm
 *    layers/tiny_attention.hpp     — Multi-Head Self-Attention
 *    models/tiny_sequential.hpp    — Sequential container
 *    models/tiny_mlp.hpp           — MLP convenience wrapper
 *    models/tiny_cnn.hpp           — CNN1D convenience wrapper
 *    train/tiny_dataset.hpp        — Dataset (shuffle, split, mini-batch)
 *    train/tiny_trainer.hpp        — Trainer (fit, evaluate)
 *
 * @version 1.0
 * @date 2026-04-30
 * @copyright Copyright (c) 2026
 */

#pragma once

/* ============================================================================
 * Configuration (platform macros, feature flags, error codes)
 * ============================================================================ */
#include "tiny_ai_config.h"

/* ============================================================================
 * C++ modules
 * ============================================================================ */
#ifdef __cplusplus

/* Quantisation ---------------------------------------------------------------- */
#include "tiny_quant_config.h"
#include "tiny_fp8.hpp"
#include "tiny_quant.h"
#include "tiny_quant.hpp"

/* Core ----------------------------------------------------------------------- */
#include "tiny_tensor.hpp"
#include "tiny_activation.hpp"
#include "tiny_loss.hpp"
#include "tiny_optimizer.hpp"

/* Layers --------------------------------------------------------------------- */
#include "tiny_layer.hpp"
#include "tiny_dense.hpp"
#include "tiny_conv.hpp"
#include "tiny_pool.hpp"
#include "tiny_norm.hpp"
#include "tiny_attention.hpp"

/* Models --------------------------------------------------------------------- */
#include "tiny_sequential.hpp"
#include "tiny_mlp.hpp"
#include "tiny_cnn.hpp"

/* Training utilities --------------------------------------------------------- */
#include "tiny_dataset.hpp"
#include "tiny_trainer.hpp"

#endif /* __cplusplus */

/* ============================================================================
 * Example entry points (callable from C and C++)
 * ============================================================================ */
#ifdef __cplusplus
extern "C" {
#endif

void example_mlp(void);
void example_cnn(void);
void example_attention(void);

#ifdef __cplusplus
}
#endif
