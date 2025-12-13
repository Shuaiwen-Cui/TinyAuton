/**
 * @file tiny_trainer.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Training loop implementation
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 */

#include "tiny_trainer.h"
#include <string.h>
#include <stdlib.h>

// Watchdog support for ESP-IDF
#ifdef ESP_PLATFORM
#include "esp_task_wdt.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
static bool g_wdt_initialized = false;
static void tiny_ai_feed_watchdog_safe(void) {
    // Add task to watchdog only once
    if (!g_wdt_initialized) {
        esp_err_t add_err = esp_task_wdt_add(NULL);
        if (add_err == ESP_OK || add_err == ESP_ERR_INVALID_STATE) {
            g_wdt_initialized = true;
        } else {
            // If add failed, don't try again and skip reset
            return;
        }
    }
    // Task is in watchdog, safe to reset
    esp_task_wdt_reset();
}
#define TINY_AI_FEED_WATCHDOG() tiny_ai_feed_watchdog_safe()
#define TINY_AI_YIELD_CPU() vTaskDelay(1)  // Yield CPU to other tasks
#else
#define TINY_AI_FEED_WATCHDOG() do { } while(0)
#define TINY_AI_YIELD_CPU() do { } while(0)
#endif

/* ============================================================================
 * INTERNAL STATE
 * ============================================================================ */

static tiny_training_stats_t g_training_stats = {0};

/* ============================================================================
 * TRAINING STEP
 * ============================================================================ */

tiny_error_t tiny_trainer_step(tiny_graph_t* graph, tiny_tensor_t* input, tiny_tensor_t* target,
                                tiny_graph_node_t* loss_node, void** optimizers, int num_optimizers,
                                tiny_tensor_t** param_tensors, int num_params)
{
    if (graph == NULL || input == NULL || target == NULL || loss_node == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    if (optimizers == NULL || param_tensors == NULL || num_optimizers != num_params) {
        return TINY_ERR_AI_INVALID_ARG;
    }
    
    // Set training mode
    tiny_graph_set_training_mode(graph, true);
    
    // Zero gradients
    tiny_error_t err = tiny_trainer_zero_gradients(graph);
    if (err != TINY_OK) {
        return err;
    }
    
    // Set input and target for loss node
    if (loss_node->inputs[0] == NULL || loss_node->inputs[1] == NULL) {
        return TINY_ERR_AI_INVALID_STATE;
    }
    
    // Copy target data to loss node's target input
    // Note: input data should already be in the graph's input tensor (set by model)
    // but we ensure target is correctly set here
    tiny_error_t copy_err = tiny_tensor_copy(target, loss_node->inputs[1]);
    if (copy_err != TINY_OK) {
        return copy_err;
    }
    
    // Forward propagation
    err = tiny_graph_forward(graph);
    if (err != TINY_OK) {
        return err;
    }
    
    // Get loss value
    float* loss_data = (float*)tiny_tensor_data(loss_node->outputs[0]);
    if (loss_data == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    // Set loss gradient to 1.0 (for backpropagation)
    void* loss_grad_ptr = tiny_tensor_grad(loss_node->outputs[0]);
    if (loss_grad_ptr != NULL) {
        ((float*)loss_grad_ptr)[0] = 1.0f;
    }
    
    // Backward propagation
    err = tiny_graph_backward(graph);
    if (err != TINY_OK) {
        return err;
    }
    
    // Update parameters using optimizers
    for (int i = 0; i < num_params; i++) {
        // Feed watchdog every 10 parameters to prevent timeout
        if (i > 0 && i % 10 == 0) {
            TINY_AI_FEED_WATCHDOG();
        }
        
        if (param_tensors[i] == NULL || optimizers[i] == NULL) {
            continue;
        }
        
        tiny_optimizer_t* opt_base = (tiny_optimizer_t*)optimizers[i];
        tiny_error_t update_err = TINY_OK;
        
        switch (opt_base->type) {
            case TINY_OPTIMIZER_SGD:
                update_err = tiny_optimizer_sgd_step((tiny_sgd_optimizer_t*)optimizers[i], param_tensors[i]);
                break;
                
            case TINY_OPTIMIZER_SGD_MOMENTUM:
                update_err = tiny_optimizer_sgd_momentum_step((tiny_sgd_momentum_optimizer_t*)optimizers[i], param_tensors[i]);
                break;
                
            case TINY_OPTIMIZER_ADAM:
                update_err = tiny_optimizer_adam_step((tiny_adam_optimizer_t*)optimizers[i], param_tensors[i]);
                break;
                
            case TINY_OPTIMIZER_RMSPROP:
                update_err = tiny_optimizer_rmsprop_step((tiny_rmsprop_optimizer_t*)optimizers[i], param_tensors[i]);
                break;
                
            case TINY_OPTIMIZER_ADAGRAD:
                update_err = tiny_optimizer_adagrad_step((tiny_adagrad_optimizer_t*)optimizers[i], param_tensors[i]);
                break;
                
            default:
                update_err = TINY_ERR_AI_NOT_SUPPORTED;
                break;
        }
        
        if (update_err != TINY_OK) {
            return update_err;
        }
    }
    
    // Update statistics
    g_training_stats.current_loss = loss_data[0];
    
    return TINY_OK;
}

tiny_error_t tiny_trainer_inference(tiny_graph_t* graph, tiny_tensor_t* input, tiny_tensor_t* output)
{
    if (graph == NULL || input == NULL || output == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    // Set inference mode
    tiny_graph_set_training_mode(graph, false);
    
    // Forward propagation only
    tiny_error_t err = tiny_graph_forward(graph);
    if (err != TINY_OK) {
        return err;
    }
    
    return TINY_OK;
}

/* ============================================================================
 * TRAINING LOOP
 * ============================================================================ */

tiny_error_t tiny_trainer_train_epoch(tiny_graph_t* graph, tiny_tensor_t** inputs, tiny_tensor_t** targets,
                                      int num_samples, int batch_size, tiny_graph_node_t* loss_node,
                                      void** optimizers, int num_optimizers, tiny_tensor_t** param_tensors, int num_params,
                                      tiny_training_callback_t callback, void* user_data)
{
    if (graph == NULL || inputs == NULL || targets == NULL || loss_node == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    if (num_samples <= 0 || batch_size <= 0) {
        return TINY_ERR_AI_INVALID_ARG;
    }
    
    int num_batches = (num_samples + batch_size - 1) / batch_size;  // Ceiling division
    float epoch_loss_sum = 0.0f;
    int samples_processed = 0;
    
    g_training_stats.total_batches = num_batches;
    g_training_stats.current_batch = 0;
    
    // Process samples in batches
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int batch_start = batch_idx * batch_size;
        int batch_end = (batch_start + batch_size < num_samples) ? (batch_start + batch_size) : num_samples;
        int actual_batch_size = batch_end - batch_start;
        
        // For simplicity, we process one sample at a time in the batch
        // In a more advanced implementation, we could process the entire batch at once
        for (int i = batch_start; i < batch_end; i++) {
            if (inputs[i] == NULL || targets[i] == NULL) {
                continue;
            }
            
            // Set input and target for loss node
            // Assuming the graph's first node takes the input
            // and loss node takes predictions and targets
            if (loss_node->inputs[0] != NULL && loss_node->inputs[1] != NULL) {
                // The prediction should come from the graph's output
                // For now, we assume the graph structure is set up correctly
                // and we just need to set the target
                tiny_tensor_t* target_tensor = loss_node->inputs[1];
                tiny_tensor_copy(targets[i], target_tensor);
            }
            
            // Perform training step
            tiny_error_t err = tiny_trainer_step(graph, inputs[i], targets[i], loss_node,
                                                optimizers, num_optimizers, param_tensors, num_params);
            if (err != TINY_OK) {
                return err;
            }
            
            epoch_loss_sum += g_training_stats.current_loss;
            samples_processed++;
        }
        
        g_training_stats.current_batch = batch_idx + 1;
        g_training_stats.epoch_loss = epoch_loss_sum / samples_processed;
        
        // Call callback if provided
        if (callback != NULL) {
            callback(&g_training_stats, user_data);
        }
    }
    
    return TINY_OK;
}

tiny_error_t tiny_trainer_train(tiny_graph_t* graph, tiny_tensor_t** inputs, tiny_tensor_t** targets,
                                int num_samples, int batch_size, int num_epochs, tiny_graph_node_t* loss_node,
                                void** optimizers, int num_optimizers, tiny_tensor_t** param_tensors, int num_params,
                                tiny_training_callback_t callback, void* user_data)
{
    if (graph == NULL || num_epochs <= 0) {
        return TINY_ERR_AI_INVALID_ARG;
    }
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        g_training_stats.current_epoch = epoch + 1;
        
        tiny_error_t err = tiny_trainer_train_epoch(graph, inputs, targets, num_samples, batch_size,
                                                   loss_node, optimizers, num_optimizers,
                                                   param_tensors, num_params, callback, user_data);
        if (err != TINY_OK) {
            return err;
        }
    }
    
    return TINY_OK;
}

/* ============================================================================
 * UTILITY
 * ============================================================================ */

tiny_error_t tiny_trainer_zero_gradients(tiny_graph_t* graph)
{
    if (graph == NULL || !graph->initialized) {
        return graph == NULL ? TINY_ERR_AI_NULL_POINTER : TINY_ERR_AI_UNINITIALIZED;
    }
    
    // Zero gradients for all tensors in the graph
    for (int i = 0; i < graph->num_nodes; i++) {
        tiny_graph_node_t* node = graph->nodes[i];
        
        // Zero input gradients
        for (int j = 0; j < node->num_inputs; j++) {
            if (node->inputs[j] != NULL) {
                tiny_tensor_zero_grad(node->inputs[j]);
            }
        }
        
        // Zero output gradients
        for (int j = 0; j < node->num_outputs; j++) {
            if (node->outputs[j] != NULL) {
                tiny_tensor_zero_grad(node->outputs[j]);
            }
        }
    }
    
    return TINY_OK;
}

tiny_error_t tiny_trainer_get_stats(tiny_training_stats_t* stats)
{
    if (stats == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    memcpy(stats, &g_training_stats, sizeof(tiny_training_stats_t));
    return TINY_OK;
}

