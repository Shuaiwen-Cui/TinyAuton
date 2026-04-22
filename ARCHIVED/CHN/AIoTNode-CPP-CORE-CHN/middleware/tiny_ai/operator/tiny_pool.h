/**
 * @file tiny_pool.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Pooling Layer for neural networks
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 *
 * @details
 * This module provides 2D pooling layer functionality for neural networks.
 * Supports Max Pooling and Average Pooling operations.
 * 
 * Features:
 * - Forward propagation: Max/Average pooling
 * - Backward propagation: gradient computation
 * - Support for stride and padding
 * - Integration with computation graph
 * - Support for training and inference modes
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
 * @brief Pooling type enumeration
 */
typedef enum
{
    TINY_POOL_MAX = 0,      // Max pooling
    TINY_POOL_AVG = 1       // Average pooling
} tiny_pool_type_t;

/**
 * @brief Pooling Layer parameters structure
 */
typedef struct tiny_pool_params_t
{
    tiny_pool_type_t pool_type;  // Type of pooling (MAX or AVG)
    int kernel_h;                 // Pooling kernel height
    int kernel_w;                 // Pooling kernel width
    int stride_h;                 // Stride in height dimension
    int stride_w;                 // Stride in width dimension
    int pad_h;                    // Padding in height dimension (symmetric)
    int pad_w;                    // Padding in width dimension (symmetric)
} tiny_pool_params_t;

/* ============================================================================
 * FUNCTION PROTOTYPES - Layer Creation and Destruction
 * ============================================================================ */

/**
 * @brief Create a pooling layer
 * 
 * @param pool_type Type of pooling (TINY_POOL_MAX or TINY_POOL_AVG)
 * @param kernel_h Pooling kernel height
 * @param kernel_w Pooling kernel width
 * @param stride_h Stride in height dimension
 * @param stride_w Stride in width dimension
 * @param pad_h Padding in height dimension (symmetric, applied to both sides)
 * @param pad_w Padding in width dimension (symmetric, applied to both sides)
 * @return tiny_pool_params_t* Pointer to created layer parameters, NULL on failure
 */
tiny_pool_params_t* tiny_pool_create(tiny_pool_type_t pool_type,
                                     int kernel_h, int kernel_w,
                                     int stride_h, int stride_w,
                                     int pad_h, int pad_w);

/**
 * @brief Destroy a pooling layer and free all resources
 * 
 * @param params Layer parameters to destroy (can be NULL)
 */
void tiny_pool_destroy(tiny_pool_params_t* params);

/* ============================================================================
 * FUNCTION PROTOTYPES - Forward and Backward Propagation
 * ============================================================================ */

/**
 * @brief Forward propagation for pooling layer
 * 
 * @param node Graph node containing the layer
 * 
 * @details
 * Performs: output = pool(input)
 * - input: [batch, channels, in_h, in_w]
 * - output: [batch, channels, out_h, out_w]
 * 
 * Output dimensions:
 * - out_h = (in_h + 2*pad_h - kernel_h) / stride_h + 1
 * - out_w = (in_w + 2*pad_w - kernel_w) / stride_w + 1
 * 
 * @note This function is designed to be called by the computation graph
 * @note Input tensor should be in node->inputs[0]
 * @note Output tensor should be in node->outputs[0]
 * @note Parameters should be in node->params (cast to tiny_pool_params_t*)
 */
void tiny_pool_forward(tiny_graph_node_t* node);

/**
 * @brief Backward propagation for pooling layer
 * 
 * @param node Graph node containing the layer
 * 
 * @details
 * Computes gradients:
 * - input_grad: gradient w.r.t. input
 * 
 * For Max Pooling: gradient flows only to the maximum element
 * For Average Pooling: gradient is distributed equally to all elements
 * 
 * @note This function is designed to be called by the computation graph
 * @note Only computes gradients if tensors have requires_grad = true
 */
void tiny_pool_backward(tiny_graph_node_t* node);

#ifdef __cplusplus
}
#endif

