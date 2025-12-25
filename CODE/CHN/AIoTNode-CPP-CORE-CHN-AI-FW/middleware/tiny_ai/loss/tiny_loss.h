/**
 * @file tiny_loss.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Loss functions for neural network training
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 *
 * @details
 * This module provides loss functions for neural network training.
 * All loss functions support both forward and backward propagation.
 * 
 * Supported loss functions:
 * - MSE: Mean Squared Error (for regression)
 * - MAE: Mean Absolute Error (for regression)
 * - Cross Entropy: Multi-class classification loss
 * - Binary Cross Entropy: Binary classification loss
 * 
 * Features:
 * - Forward propagation: compute loss value
 * - Backward propagation: compute gradients
 * - Integration with computation graph
 * - Batch support
 * 
 * Design:
 * - Lightweight implementation for MCU
 * - Efficient batch processing
 * - Numerically stable implementations
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
 * FUNCTION PROTOTYPES - MSE Loss (Mean Squared Error)
 * ============================================================================ */

/**
 * @brief Forward propagation for MSE loss
 * 
 * @param node Graph node containing the loss
 * 
 * @details
 * Computes: loss = mean((pred - target)^2)
 * - Input: node->inputs[0] = predictions, node->inputs[1] = targets
 * - Output: node->outputs[0] = scalar loss value
 * - Supports batch processing
 * 
 * @note Both predictions and targets must have the same shape
 * @note Output is a scalar tensor (shape = [1])
 */
void tiny_mse_forward(tiny_graph_node_t* node);

/**
 * @brief Backward propagation for MSE loss
 * 
 * @param node Graph node containing the loss
 * 
 * @details
 * Computes gradient: pred_grad = 2 * (pred - target) / batch_size
 * - Only computes gradient for predictions (not targets)
 * - Gradient is averaged over batch
 * 
 * @note Only computes gradients if predictions have requires_grad = true
 */
void tiny_mse_backward(tiny_graph_node_t* node);

/* ============================================================================
 * FUNCTION PROTOTYPES - MAE Loss (Mean Absolute Error)
 * ============================================================================ */

/**
 * @brief Forward propagation for MAE loss
 * 
 * @param node Graph node containing the loss
 * 
 * @details
 * Computes: loss = mean(|pred - target|)
 * - Input: node->inputs[0] = predictions, node->inputs[1] = targets
 * - Output: node->outputs[0] = scalar loss value
 * - Supports batch processing
 * 
 * @note Both predictions and targets must have the same shape
 * @note Output is a scalar tensor (shape = [1])
 */
void tiny_mae_forward(tiny_graph_node_t* node);

/**
 * @brief Backward propagation for MAE loss
 * 
 * @param node Graph node containing the loss
 * 
 * @details
 * Computes gradient: pred_grad = sign(pred - target) / batch_size
 * - Only computes gradient for predictions (not targets)
 * - Gradient is averaged over batch
 * 
 * @note Only computes gradients if predictions have requires_grad = true
 */
void tiny_mae_backward(tiny_graph_node_t* node);

/* ============================================================================
 * FUNCTION PROTOTYPES - Cross Entropy Loss
 * ============================================================================ */

/**
 * @brief Forward propagation for Cross Entropy loss
 * 
 * @param node Graph node containing the loss
 * 
 * @details
 * Computes: loss = -mean(sum(target * log(softmax(pred)), dim=-1))
 * - Input: node->inputs[0] = logits (before softmax), node->inputs[1] = target labels (one-hot or class indices)
 * - Output: node->outputs[0] = scalar loss value
 * - Supports both one-hot encoded targets and class indices
 * 
 * @note Logits shape: [batch, num_classes]
 * @note Targets shape: [batch, num_classes] (one-hot) or [batch] (class indices)
 * @note Output is a scalar tensor (shape = [1])
 */
void tiny_cross_entropy_forward(tiny_graph_node_t* node);

/**
 * @brief Backward propagation for Cross Entropy loss
 * 
 * @param node Graph node containing the loss
 * 
 * @details
 * Computes gradient: logits_grad = (softmax(logits) - target) / batch_size
 * - Only computes gradient for logits (not targets)
 * - Gradient is averaged over batch
 * 
 * @note Only computes gradients if logits have requires_grad = true
 */
void tiny_cross_entropy_backward(tiny_graph_node_t* node);

/* ============================================================================
 * FUNCTION PROTOTYPES - Binary Cross Entropy Loss
 * ============================================================================ */

/**
 * @brief Forward propagation for Binary Cross Entropy loss
 * 
 * @param node Graph node containing the loss
 * 
 * @details
 * Computes: loss = -mean(target * log(sigmoid(pred)) + (1-target) * log(1-sigmoid(pred)))
 * - Input: node->inputs[0] = predictions (logits), node->inputs[1] = targets (0 or 1)
 * - Output: node->outputs[0] = scalar loss value
 * - Supports batch processing
 * 
 * @note Both predictions and targets must have the same shape
 * @note Targets should be in range [0, 1]
 * @note Output is a scalar tensor (shape = [1])
 */
void tiny_binary_cross_entropy_forward(tiny_graph_node_t* node);

/**
 * @brief Backward propagation for Binary Cross Entropy loss
 * 
 * @param node Graph node containing the loss
 * 
 * @details
 * Computes gradient: pred_grad = (sigmoid(pred) - target) / batch_size
 * - Only computes gradient for predictions (not targets)
 * - Gradient is averaged over batch
 * 
 * @note Only computes gradients if predictions have requires_grad = true
 */
void tiny_binary_cross_entropy_backward(tiny_graph_node_t* node);

#ifdef __cplusplus
}
#endif

