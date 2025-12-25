/**
 * @file tiny_cnn.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Convolutional Neural Network (CNN) Model Wrapper
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 *
 * @details
 * This module provides a high-level wrapper for building and training
 * Convolutional Neural Networks (CNN).
 * 
 * Features:
 * - Automatic graph construction
 * - Automatic tensor management
 * - Simplified layer addition (Conv, Pool, Norm, FC)
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
#include "tiny_convolution.h"
#include "tiny_pool.h"
#include "tiny_norm.h"
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
 * @brief Activation function type (same as MLP)
 */
typedef enum
{
    TINY_CNN_ACT_NONE = 0,    // No activation
    TINY_CNN_ACT_RELU,        // ReLU activation
    TINY_CNN_ACT_SIGMOID,     // Sigmoid activation
    TINY_CNN_ACT_TANH         // Tanh activation
} tiny_cnn_activation_t;

/**
 * @brief CNN block type
 */
typedef enum
{
    TINY_CNN_BLOCK_CONV = 0,  // Convolution block
    TINY_CNN_BLOCK_POOL,      // Pooling block
    TINY_CNN_BLOCK_NORM,      // Normalization block
    TINY_CNN_BLOCK_FC         // Fully connected block
} tiny_cnn_block_type_t;

/**
 * @brief CNN convolution block configuration
 */
typedef struct tiny_cnn_conv_block_t
{
    int in_channels;              // Input channels
    int out_channels;             // Output channels
    int kernel_h;                 // Kernel height
    int kernel_w;                 // Kernel width
    int stride_h;                 // Stride in height (default: 1)
    int stride_w;                 // Stride in width (default: 1)
    int pad_h;                    // Padding in height (default: 0)
    int pad_w;                    // Padding in width (default: 0)
    bool use_bias;                // Whether to use bias
    tiny_cnn_activation_t activation;  // Activation function
    bool use_batchnorm;           // Whether to use batch normalization after conv
} tiny_cnn_conv_block_t;

/**
 * @brief CNN pooling block configuration
 */
typedef struct tiny_cnn_pool_block_t
{
    tiny_pool_type_t pool_type;  // Pooling type (MAX or AVG)
    int kernel_h;                 // Pooling kernel height
    int kernel_w;                 // Pooling kernel width
    int stride_h;                 // Stride in height (default: kernel_h)
    int stride_w;                 // Stride in width (default: kernel_w)
    int pad_h;                    // Padding in height (default: 0)
    int pad_w;                    // Padding in width (default: 0)
} tiny_cnn_pool_block_t;

/**
 * @brief CNN fully connected block configuration
 */
typedef struct tiny_cnn_fc_block_t
{
    int in_features;              // Input features
    int out_features;             // Output features
    bool use_bias;                // Whether to use bias
    tiny_cnn_activation_t activation;  // Activation function
} tiny_cnn_fc_block_t;

/**
 * @brief CNN block configuration (union of all block types)
 */
typedef struct tiny_cnn_block_config_t
{
    tiny_cnn_block_type_t block_type;  // Block type
    
    union {
        tiny_cnn_conv_block_t conv;     // Convolution block config
        tiny_cnn_pool_block_t pool;    // Pooling block config
        tiny_cnn_fc_block_t fc;         // Fully connected block config
    } config;
} tiny_cnn_block_config_t;

/**
 * @brief CNN model structure
 */
typedef struct tiny_cnn_model_t
{
    tiny_graph_t* graph;              // Computation graph
    tiny_graph_node_t** block_nodes;  // Array of block nodes
    tiny_graph_node_t* loss_node;     // Loss node
    tiny_tensor_t* input_tensor;       // Input tensor [batch, channels, h, w]
    tiny_tensor_t* output_tensor;      // Output tensor
    tiny_tensor_t* target_tensor;     // Target tensor
    tiny_tensor_t* loss_tensor;       // Loss tensor
    
    // Layer parameters (for initialization and optimization)
    void** conv_params;               // Array of conv layer parameters
    void** pool_params;               // Array of pool layer parameters
    void** norm_params;               // Array of norm layer parameters
    void** fc_params;                 // Array of FC layer parameters
    
    int num_blocks;                   // Number of blocks
    int num_conv;                     // Number of conv layers
    int num_pool;                     // Number of pool layers
    int num_norm;                     // Number of norm layers
    int num_fc;                       // Number of FC layers
    
    // Input/output dimensions
    int input_channels;               // Input channels
    int input_h;                      // Input height
    int input_w;                      // Input width
    int output_size;                  // Output size (for classification)
    
    bool initialized;                 // Whether model is initialized
} tiny_cnn_model_t;

/* ============================================================================
 * FUNCTION PROTOTYPES - Model Creation and Destruction
 * ============================================================================ */

/**
 * @brief Create a new CNN model
 * 
 * @param input_channels Input channels
 * @param input_h Input height
 * @param input_w Input width
 * @param block_configs Array of block configurations
 * @param num_blocks Number of blocks
 * @return tiny_cnn_model_t* Pointer to created model, NULL on failure
 * 
 * @note The block_configs array should have num_blocks elements
 * @note First block should be a convolution block with in_channels matching input_channels
 * @note Last block should typically be a fully connected block for classification
 */
tiny_cnn_model_t* tiny_cnn_create(int input_channels, int input_h, int input_w,
                                  const tiny_cnn_block_config_t* block_configs,
                                  int num_blocks);

/**
 * @brief Destroy a CNN model and free all resources
 * 
 * @param model Model to destroy (can be NULL)
 */
void tiny_cnn_destroy(tiny_cnn_model_t* model);

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
tiny_error_t tiny_cnn_init_weights_xavier(tiny_cnn_model_t* model, unsigned int seed);

/**
 * @brief Initialize all weights to zero
 * 
 * @param model Model to initialize
 * @return tiny_error_t TINY_OK on success, error code on failure
 */
tiny_error_t tiny_cnn_init_weights_zero(tiny_cnn_model_t* model);

/**
 * @brief Set loss function
 * 
 * @param model Model to configure
 * @param loss_type Loss type (TINY_AI_OP_MSE_LOSS, TINY_AI_OP_CROSS_ENTROPY_LOSS, etc.)
 * @return tiny_error_t TINY_OK on success, error code on failure
 */
tiny_error_t tiny_cnn_set_loss(tiny_cnn_model_t* model, tiny_ai_op_type_t loss_type);

/* ============================================================================
 * FUNCTION PROTOTYPES - Model Execution
 * ============================================================================ */

/**
 * @brief Forward propagation (inference)
 * 
 * @param model Model to run
 * @param input_data Input data [batch, channels, h, w] (flattened)
 * @param batch_size Batch size
 * @param output_data Output buffer (must be pre-allocated, size = batch_size * output_size)
 * @return tiny_error_t TINY_OK on success, error code on failure
 */
tiny_error_t tiny_cnn_forward(tiny_cnn_model_t* model,
                              const float* input_data,
                              int batch_size,
                              float* output_data);

/**
 * @brief Get all trainable parameters
 * 
 * @param model Model to query
 * @param params Output array of parameter tensors (must be pre-allocated)
 * @param max_params Maximum number of parameters to return
 * @return int Actual number of parameters returned
 * 
 * @note Returns weights and biases from all layers
 */
int tiny_cnn_get_parameters(tiny_cnn_model_t* model,
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
tiny_error_t tiny_cnn_train_epoch(tiny_cnn_model_t* model,
                                   tiny_dataloader_t* loader,
                                   tiny_ai_op_type_t loss_type,
                                   void** optimizers,
                                   int num_optimizers,
                                   tiny_training_callback_t callback,
                                   void* user_data);

#ifdef __cplusplus
}
#endif

