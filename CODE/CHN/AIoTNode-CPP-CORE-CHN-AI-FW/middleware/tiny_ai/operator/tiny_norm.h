/**
 * @file tiny_norm.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Normalization Layer for neural networks
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 *
 * @details
 * This module provides normalization layer functionality for neural networks.
 * Supports Batch Normalization (BatchNorm).
 * 
 * Features:
 * - Forward propagation: normalization with learnable scale and shift
 * - Backward propagation: gradient computation
 * - Training and inference modes
 * - Running statistics for inference
 * - Integration with computation graph
 * 
 * Design:
 * - Lightweight implementation for MCU
 * - Memory-efficient operation
 * - Supports multi-channel input/output
 */

#pragma once

#include "tiny_ai_config.h"
#include "tiny_tensor.h"
#include "tiny_graph.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* ============================================================================
 * TYPE DEFINITIONS
 * ============================================================================ */

/**
 * @brief Batch Normalization parameters structure
 */
typedef struct tiny_batchnorm_params_t
{
    tiny_tensor_t* gamma;          // Scale parameter [channels] (learnable)
    tiny_tensor_t* beta;            // Shift parameter [channels] (learnable)
    tiny_tensor_t* running_mean;    // Running mean [channels] (for inference)
    tiny_tensor_t* running_var;    // Running variance [channels] (for inference)
    int num_channels;               // Number of channels
    float momentum;                 // Momentum for running statistics (0.0-1.0)
    float eps;                      // Small constant for numerical stability
    bool affine;                    // Whether to use learnable scale and shift
} tiny_batchnorm_params_t;

/* ============================================================================
 * FUNCTION PROTOTYPES - Layer Creation and Destruction
 * ============================================================================ */

/**
 * @brief Create a batch normalization layer
 * 
 * @param num_channels Number of channels
 * @param momentum Momentum for running statistics (typically 0.9-0.99)
 * @param eps Small constant for numerical stability (typically 1e-5)
 * @param affine Whether to use learnable scale (gamma) and shift (beta)
 * @return tiny_batchnorm_params_t* Pointer to created layer parameters, NULL on failure
 * 
 * @note Gamma and beta are initialized to 1.0 and 0.0 respectively
 * @note Running mean and variance are initialized to 0.0 and 1.0 respectively
 */
tiny_batchnorm_params_t* tiny_batchnorm_create(int num_channels, 
                                                float momentum, 
                                                float eps, 
                                                bool affine);

/**
 * @brief Destroy a batch normalization layer and free all resources
 * 
 * @param params Layer parameters to destroy (can be NULL)
 */
void tiny_batchnorm_destroy(tiny_batchnorm_params_t* params);

/* ============================================================================
 * FUNCTION PROTOTYPES - Forward and Backward Propagation
 * ============================================================================ */

/**
 * @brief Forward propagation for batch normalization layer
 * 
 * @param node Graph node containing the layer
 * 
 * @details
 * Performs: output = gamma * normalize(input) + beta
 * - input: [batch, channels, h, w] or [batch, channels]
 * - output: [batch, channels, h, w] or [batch, channels]
 * 
 * In training mode:
 * - Computes batch statistics (mean, variance)
 * - Updates running statistics
 * - Normalizes using batch statistics
 * 
 * In inference mode:
 * - Normalizes using running statistics
 * 
 * @note This function is designed to be called by the computation graph
 * @note Input tensor should be in node->inputs[0]
 * @note Output tensor should be in node->outputs[0]
 * @note Parameters should be in node->params (cast to tiny_batchnorm_params_t*)
 * @note Training mode is determined by graph->training_mode
 */
void tiny_batchnorm_forward(tiny_graph_node_t* node);

/**
 * @brief Backward propagation for batch normalization layer
 * 
 * @param node Graph node containing the layer
 * 
 * @details
 * Computes gradients:
 * - input_grad: gradient w.r.t. input
 * - gamma_grad: gradient w.r.t. gamma (if affine)
 * - beta_grad: gradient w.r.t. beta (if affine)
 * 
 * @note This function is designed to be called by the computation graph
 * @note Only computes gradients if tensors have requires_grad = true
 * @note Only works in training mode
 */
void tiny_batchnorm_backward(tiny_graph_node_t* node);

/* ============================================================================
 * FUNCTION PROTOTYPES - Utility
 * ============================================================================ */

/**
 * @brief Reset running statistics to initial values
 * 
 * @param params Layer parameters
 * @return tiny_error_t TINY_OK on success, error code on failure
 */
tiny_error_t tiny_batchnorm_reset_running_stats(tiny_batchnorm_params_t* params);

#ifdef __cplusplus
}
#endif

