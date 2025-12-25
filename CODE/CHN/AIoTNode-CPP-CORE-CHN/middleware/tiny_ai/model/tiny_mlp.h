/**
 * @file tiny_mlp.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Multi-Layer Perceptron (MLP) Model Wrapper
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 *
 * @details
 * This module provides a high-level wrapper for building and training
 * Multi-Layer Perceptron (MLP) neural networks.
 * 
 * Features:
 * - Automatic graph construction
 * - Automatic tensor management
 * - Simplified layer addition
 * - Built-in training interface
 * - Parameter access and initialization
 * 
 * Design:
 * - Lightweight implementation for MCU
 * - Minimal memory overhead
 * - Easy-to-use API
 */

#pragma once

#include "tiny_ai_config.h"
#include "tiny_tensor.h"
#include "tiny_graph.h"
#include "tiny_fc.h"
#include "tiny_activations.h"
#include "tiny_loss.h"
#include "tiny_optimizer.h"
#include "tiny_trainer.h"
#include "tiny_dataloader.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* ============================================================================
 * TYPE DEFINITIONS
 * ============================================================================ */

/**
 * @brief Activation function type
 */
typedef enum
{
    TINY_MLP_ACT_NONE = 0,    // No activation
    TINY_MLP_ACT_RELU,        // ReLU activation
    TINY_MLP_ACT_SIGMOID,     // Sigmoid activation
    TINY_MLP_ACT_TANH         // Tanh activation
} tiny_mlp_activation_t;

/**
 * @brief MLP layer configuration
 */
typedef struct tiny_mlp_layer_config_t
{
    int in_features;              // Input features
    int out_features;             // Output features
    bool use_bias;                // Whether to use bias
    tiny_mlp_activation_t activation;  // Activation function
} tiny_mlp_layer_config_t;

/**
 * @brief MLP model structure
 */
typedef struct tiny_mlp_model_t
{
    tiny_graph_t* graph;          // Computation graph
    tiny_graph_node_t** fc_nodes; // Array of FC layer nodes
    tiny_graph_node_t** act_nodes;// Array of activation nodes
    tiny_graph_node_t* loss_node; // Loss node
    tiny_tensor_t* input_tensor;  // Input tensor
    tiny_tensor_t* output_tensor; // Output tensor
    tiny_tensor_t* target_tensor; // Target tensor
    tiny_tensor_t* loss_tensor;   // Loss tensor
    tiny_fc_params_t** fc_params; // Array of FC layer parameters
    int num_layers;                // Number of layers
    int input_size;                // Input size
    int output_size;               // Output size
    bool initialized;              // Whether model is initialized
} tiny_mlp_model_t;

/* ============================================================================
 * FUNCTION PROTOTYPES - Model Creation and Destruction
 * ============================================================================ */

/**
 * @brief Create a new MLP model
 * 
 * @param input_size Input feature size
 * @param layer_configs Array of layer configurations
 * @param num_layers Number of layers
 * @return tiny_mlp_model_t* Pointer to created model, NULL on failure
 * 
 * @note The layer_configs array should have num_layers elements
 * @note The first layer's in_features should match input_size
 * @note Each subsequent layer's in_features should match previous layer's out_features
 */
tiny_mlp_model_t* tiny_mlp_create(int input_size,
                                   const tiny_mlp_layer_config_t* layer_configs,
                                   int num_layers);

/**
 * @brief Destroy an MLP model and free all resources
 * 
 * @param model Model to destroy (can be NULL)
 */
void tiny_mlp_destroy(tiny_mlp_model_t* model);

/* ============================================================================
 * FUNCTION PROTOTYPES - Model Configuration
 * ============================================================================ */

/**
 * @brief Initialize all weights using Xavier initialization
 * 
 * @param model Model to initialize
 * @param seed Random seed (0 for time-based seed)
 * @return tiny_error_t TINY_OK on success, error code on failure
 */
tiny_error_t tiny_mlp_init_weights_xavier(tiny_mlp_model_t* model, unsigned int seed);

/**
 * @brief Initialize all weights to zero
 * 
 * @param model Model to initialize
 * @return tiny_error_t TINY_OK on success, error code on failure
 */
tiny_error_t tiny_mlp_init_weights_zero(tiny_mlp_model_t* model);

/**
 * @brief Set loss function
 * 
 * @param model Model to configure
 * @param loss_type Loss type (TINY_AI_OP_MSE_LOSS, TINY_AI_OP_CROSS_ENTROPY_LOSS, etc.)
 * @return tiny_error_t TINY_OK on success, error code on failure
 */
tiny_error_t tiny_mlp_set_loss(tiny_mlp_model_t* model, tiny_ai_op_type_t loss_type);

/* ============================================================================
 * FUNCTION PROTOTYPES - Model Execution
 * ============================================================================ */

/**
 * @brief Forward propagation (inference)
 * 
 * @param model Model to run
 * @param input_data Input data (array of floats)
 * @param output_data Output buffer (must be pre-allocated, size = output_size)
 * @return tiny_error_t TINY_OK on success, error code on failure
 */
tiny_error_t tiny_mlp_forward(tiny_mlp_model_t* model,
                               const float* input_data,
                               float* output_data);

/**
 * @brief Get all trainable parameters
 * 
 * @param model Model to query
 * @param params Output array of parameter tensors (must be pre-allocated)
 * @param max_params Maximum number of parameters to return
 * @return int Actual number of parameters returned
 * 
 * @note Returns weights and biases from all FC layers
 */
int tiny_mlp_get_parameters(tiny_mlp_model_t* model,
                            tiny_tensor_t** params,
                            int max_params);

/* ============================================================================
 * FUNCTION PROTOTYPES - Training
 * ============================================================================ */

/**
 * @brief Train the model for one epoch
 * 
 * @param model Model to train
 * @param loader DataLoader for training data
 * @param loss_type Loss type
 * @param optimizers Array of optimizers (one per parameter)
 * @param num_optimizers Number of optimizers
 * @param callback Optional callback function (can be NULL)
 * @param user_data User data for callback (can be NULL)
 * @return tiny_error_t TINY_OK on success, error code on failure
 */
tiny_error_t tiny_mlp_train_epoch(tiny_mlp_model_t* model,
                                   tiny_dataloader_t* loader,
                                   tiny_ai_op_type_t loss_type,
                                   void** optimizers,
                                   int num_optimizers,
                                   tiny_training_callback_t callback,
                                   void* user_data);

#ifdef __cplusplus
}
#endif

