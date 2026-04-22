/**
 * @file tiny_activations.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Activation functions for neural networks
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 *
 * @details
 * This module provides activation functions for neural networks.
 * All activation functions support both forward and backward propagation.
 * 
 * Supported activations:
 * - ReLU: Rectified Linear Unit
 * - Sigmoid: Sigmoid activation
 * - Tanh: Hyperbolic tangent
 * 
 * Features:
 * - Forward propagation: element-wise activation
 * - Backward propagation: gradient computation
 * - Integration with computation graph
 * - In-place operations support (optional)
 * 
 * Design:
 * - Lightweight implementation for MCU
 * - Efficient element-wise operations
 * - Memory-efficient (can operate in-place)
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
 * FUNCTION PROTOTYPES - ReLU Activation
 * ============================================================================ */

/**
 * @brief Forward propagation for ReLU activation
 * 
 * @param node Graph node containing the activation
 * 
 * @details
 * Performs: output = max(0, input)
 * - Element-wise operation
 * - output[i] = input[i] if input[i] > 0, else 0
 * 
 * @note Input tensor should be in node->inputs[0]
 * @note Output tensor should be in node->outputs[0]
 * @note Can operate in-place (input == output)
 */
void tiny_relu_forward(tiny_graph_node_t* node);

/**
 * @brief Backward propagation for ReLU activation
 * 
 * @param node Graph node containing the activation
 * 
 * @details
 * Computes gradient: input_grad = output_grad * (input > 0 ? 1 : 0)
 * - Element-wise gradient passing
 * - Only passes gradient where input > 0
 * 
 * @note Only computes gradients if tensors have requires_grad = true
 */
void tiny_relu_backward(tiny_graph_node_t* node);

/* ============================================================================
 * FUNCTION PROTOTYPES - Sigmoid Activation
 * ============================================================================ */

/**
 * @brief Forward propagation for Sigmoid activation
 * 
 * @param node Graph node containing the activation
 * 
 * @details
 * Performs: output = 1 / (1 + exp(-input))
 * - Element-wise operation
 * - Output range: (0, 1)
 * 
 * @note Input tensor should be in node->inputs[0]
 * @note Output tensor should be in node->outputs[0]
 * @note Can operate in-place (input == output)
 */
void tiny_sigmoid_forward(tiny_graph_node_t* node);

/**
 * @brief Backward propagation for Sigmoid activation
 * 
 * @param node Graph node containing the activation
 * 
 * @details
 * Computes gradient: input_grad = output_grad * output * (1 - output)
 * - Uses the property: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
 * - More numerically stable than recomputing
 * 
 * @note Only computes gradients if tensors have requires_grad = true
 */
void tiny_sigmoid_backward(tiny_graph_node_t* node);

/* ============================================================================
 * FUNCTION PROTOTYPES - Tanh Activation
 * ============================================================================ */

/**
 * @brief Forward propagation for Tanh activation
 * 
 * @param node Graph node containing the activation
 * 
 * @details
 * Performs: output = tanh(input)
 * - Element-wise operation
 * - Output range: (-1, 1)
 * 
 * @note Input tensor should be in node->inputs[0]
 * @note Output tensor should be in node->outputs[0]
 * @note Can operate in-place (input == output)
 */
void tiny_tanh_forward(tiny_graph_node_t* node);

/**
 * @brief Backward propagation for Tanh activation
 * 
 * @param node Graph node containing the activation
 * 
 * @details
 * Computes gradient: input_grad = output_grad * (1 - output^2)
 * - Uses the property: tanh'(x) = 1 - tanh(x)^2
 * - More numerically stable than recomputing
 * 
 * @note Only computes gradients if tensors have requires_grad = true
 */
void tiny_tanh_backward(tiny_graph_node_t* node);

#ifdef __cplusplus
}
#endif

