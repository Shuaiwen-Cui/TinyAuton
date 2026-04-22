/**
 * @file tiny_trainer.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Training loop and utilities for neural networks
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 *
 * @details
 * This module provides training loop functionality for neural networks.
 * It integrates forward propagation, loss computation, backward propagation,
 * and parameter updates into a unified training interface.
 * 
 * Features:
 * - Training loop management (epochs, batches)
 * - Integration with computation graph
 * - Support for different optimizers
 * - Loss tracking and reporting
 * - Training/validation mode switching
 * 
 * Design:
 * - Lightweight implementation for MCU
 * - Flexible optimizer support
 * - Efficient batch processing
 */

#pragma once

#include "tiny_ai_config.h"
#include "tiny_tensor.h"
#include "tiny_graph.h"
#include "tiny_optimizer.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* ============================================================================
 * TYPE DEFINITIONS
 * ============================================================================ */

/**
 * @brief Training statistics structure
 */
typedef struct tiny_training_stats_t
{
    float current_loss;        // Current batch loss
    float epoch_loss;          // Average loss for current epoch
    int current_epoch;         // Current epoch number
    int current_batch;          // Current batch number
    int total_batches;          // Total batches in current epoch
} tiny_training_stats_t;

/**
 * @brief Callback function type for training events
 * 
 * @param stats Training statistics
 * @param user_data User-provided data pointer
 */
typedef void (*tiny_training_callback_t)(const tiny_training_stats_t* stats, void* user_data);

/* ============================================================================
 * FUNCTION PROTOTYPES - Training Step
 * ============================================================================ */

/**
 * @brief Perform a single training step (forward + backward + update)
 * 
 * @param graph Computation graph
 * @param input Input tensor
 * @param target Target tensor (for loss computation)
 * @param loss_node Loss node in the graph
 * @param optimizers Array of optimizers (one per parameter tensor)
 * @param num_optimizers Number of optimizers
 * @param param_tensors Array of parameter tensors to update
 * @param num_params Number of parameter tensors
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @details
 * This function performs:
 * 1. Forward propagation through the graph
 * 2. Loss computation
 * 3. Backward propagation
 * 4. Parameter updates using optimizers
 * 
 * @note The loss node should have inputs[0] = predictions, inputs[1] = targets
 * @note Each optimizer corresponds to one parameter tensor
 * @note Parameter tensors must have gradients enabled
 */
tiny_error_t tiny_trainer_step(tiny_graph_t* graph, tiny_tensor_t* input, tiny_tensor_t* target,
                                tiny_graph_node_t* loss_node, void** optimizers, int num_optimizers,
                                tiny_tensor_t** param_tensors, int num_params);

/**
 * @brief Perform a single inference step (forward only)
 * 
 * @param graph Computation graph
 * @param input Input tensor
 * @param output Output tensor (will be filled with predictions)
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @details
 * This function performs forward propagation only (no gradients computed)
 * 
 * @note Graph should be in inference mode (training_mode = false)
 */
tiny_error_t tiny_trainer_inference(tiny_graph_t* graph, tiny_tensor_t* input, tiny_tensor_t* output);

/* ============================================================================
 * FUNCTION PROTOTYPES - Training Loop
 * ============================================================================ */

/**
 * @brief Train for one epoch
 * 
 * @param graph Computation graph
 * @param inputs Array of input tensors (one per sample)
 * @param targets Array of target tensors (one per sample)
 * @param num_samples Number of training samples
 * @param batch_size Batch size (1 = online learning, >1 = mini-batch)
 * @param loss_node Loss node in the graph
 * @param optimizers Array of optimizers
 * @param num_optimizers Number of optimizers
 * @param param_tensors Array of parameter tensors
 * @param num_params Number of parameter tensors
 * @param callback Optional callback function called after each batch
 * @param user_data User data passed to callback
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @details
 * Trains the model for one epoch:
 * - Processes samples in batches
 * - Calls tiny_trainer_step for each batch
 * - Accumulates loss statistics
 * - Calls callback after each batch (if provided)
 */
tiny_error_t tiny_trainer_train_epoch(tiny_graph_t* graph, tiny_tensor_t** inputs, tiny_tensor_t** targets,
                                      int num_samples, int batch_size, tiny_graph_node_t* loss_node,
                                      void** optimizers, int num_optimizers, tiny_tensor_t** param_tensors, int num_params,
                                      tiny_training_callback_t callback, void* user_data);

/**
 * @brief Train for multiple epochs
 * 
 * @param graph Computation graph
 * @param inputs Array of input tensors
 * @param targets Array of target tensors
 * @param num_samples Number of training samples
 * @param batch_size Batch size
 * @param num_epochs Number of epochs to train
 * @param loss_node Loss node in the graph
 * @param optimizers Array of optimizers
 * @param num_optimizers Number of optimizers
 * @param param_tensors Array of parameter tensors
 * @param num_params Number of parameter tensors
 * @param callback Optional callback function called after each batch
 * @param user_data User data passed to callback
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @details
 * Trains the model for multiple epochs by calling tiny_trainer_train_epoch repeatedly
 */
tiny_error_t tiny_trainer_train(tiny_graph_t* graph, tiny_tensor_t** inputs, tiny_tensor_t** targets,
                                int num_samples, int batch_size, int num_epochs, tiny_graph_node_t* loss_node,
                                void** optimizers, int num_optimizers, tiny_tensor_t** param_tensors, int num_params,
                                tiny_training_callback_t callback, void* user_data);

/* ============================================================================
 * FUNCTION PROTOTYPES - Utility
 * ============================================================================ */

/**
 * @brief Zero all gradients in the graph
 * 
 * @param graph Computation graph
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @note This should be called at the start of each training step
 */
tiny_error_t tiny_trainer_zero_gradients(tiny_graph_t* graph);

/**
 * @brief Get training statistics
 * 
 * @param stats Pointer to statistics structure to fill
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @note Statistics are maintained internally and updated during training
 */
tiny_error_t tiny_trainer_get_stats(tiny_training_stats_t* stats);

#ifdef __cplusplus
}
#endif

