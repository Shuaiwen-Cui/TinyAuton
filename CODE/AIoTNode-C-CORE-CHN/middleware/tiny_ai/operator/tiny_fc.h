/**
 * @file tiny_fc.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Fully Connected (Dense) Layer for neural networks
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 *
 * @details
 * This module provides fully connected (dense) layer functionality for neural networks.
 * A fully connected layer performs: output = input @ weights^T + bias
 * 
 * Features:
 * - Forward propagation: matrix multiplication + bias addition
 * - Backward propagation: gradient computation for input, weights, and bias
 * - Integration with computation graph
 * - Support for training and inference modes
 * 
 * Design:
 * - Lightweight implementation for MCU
 * - Uses tiny_math for optimized matrix operations
 * - Memory-efficient parameter storage
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
 * @brief Fully Connected Layer parameters structure
 */
typedef struct tiny_fc_params_t
{
    tiny_tensor_t* weights;      // Weight matrix [out_features, in_features]
    tiny_tensor_t* bias;         // Bias vector [out_features] (can be NULL)
    int in_features;              // Number of input features
    int out_features;             // Number of output features
    bool use_bias;                // Whether to use bias
} tiny_fc_params_t;

/* ============================================================================
 * FUNCTION PROTOTYPES - Layer Creation and Destruction
 * ============================================================================ */

/**
 * @brief Create a fully connected layer
 * 
 * @param in_features Number of input features
 * @param out_features Number of output features
 * @param use_bias Whether to use bias (true) or not (false)
 * @return tiny_fc_params_t* Pointer to created layer parameters, NULL on failure
 * 
 * @note Weights are initialized to zero (should be initialized externally)
 * @note Bias is allocated only if use_bias is true
 * @note Both weights and bias have requires_grad = true by default
 */
tiny_fc_params_t* tiny_fc_create(int in_features, int out_features, bool use_bias);

/**
 * @brief Destroy a fully connected layer and free all resources
 * 
 * @param params Layer parameters to destroy (can be NULL)
 * 
 * @note This destroys the weights and bias tensors
 */
void tiny_fc_destroy(tiny_fc_params_t* params);

/* ============================================================================
 * FUNCTION PROTOTYPES - Forward and Backward Propagation
 * ============================================================================ */

/**
 * @brief Forward propagation for fully connected layer
 * 
 * @param node Graph node containing the layer
 * 
 * @details
 * Performs: output = input @ weights^T + bias
 * - input: [batch, in_features]
 * - weights: [out_features, in_features]
 * - bias: [out_features] (if use_bias)
 * - output: [batch, out_features]
 * 
 * @note This function is designed to be called by the computation graph
 * @note Input tensor should be in node->inputs[0]
 * @note Output tensor should be in node->outputs[0]
 * @note Parameters should be in node->params (cast to tiny_fc_params_t*)
 */
void tiny_fc_forward(tiny_graph_node_t* node);

/**
 * @brief Backward propagation for fully connected layer
 * 
 * @param node Graph node containing the layer
 * 
 * @details
 * Computes gradients:
 * - input_grad: gradient w.r.t. input
 * - weight_grad: gradient w.r.t. weights
 * - bias_grad: gradient w.r.t. bias (if use_bias)
 * 
 * @note This function is designed to be called by the computation graph
 * @note Only computes gradients if tensors have requires_grad = true
 */
void tiny_fc_backward(tiny_graph_node_t* node);

/* ============================================================================
 * FUNCTION PROTOTYPES - Utility
 * ============================================================================ */

/**
 * @brief Initialize weights with random values (Xavier/Glorot initialization)
 * 
 * @param params Layer parameters
 * @param seed Random seed (0 for time-based seed)
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @note Uses uniform distribution: [-sqrt(6/(in+out)), sqrt(6/(in+out))]
 */
tiny_error_t tiny_fc_init_weights_xavier(tiny_fc_params_t* params, unsigned int seed);

/**
 * @brief Initialize weights with zeros
 * 
 * @param params Layer parameters
 * @return tiny_error_t TINY_OK on success, error code on failure
 */
tiny_error_t tiny_fc_init_weights_zero(tiny_fc_params_t* params);

#ifdef __cplusplus
}
#endif

