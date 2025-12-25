/**
 * @file tiny_mlp.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief MLP Model implementation
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 */

#include "tiny_mlp.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

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
 * MODEL CREATION AND DESTRUCTION
 * ============================================================================ */

tiny_mlp_model_t* tiny_mlp_create(int input_size,
                                  const tiny_mlp_layer_config_t* layer_configs,
                                  int num_layers)
{
    if (input_size <= 0 || layer_configs == NULL || num_layers <= 0) {
        return NULL;
    }
    
    // Validate layer configurations
    if (layer_configs[0].in_features != input_size) {
        return NULL;  // First layer input must match model input
    }
    
    for (int i = 1; i < num_layers; i++) {
        if (layer_configs[i].in_features != layer_configs[i-1].out_features) {
            return NULL;  // Layer inputs must match previous layer outputs
        }
    }
    
    // Allocate model structure
    tiny_mlp_model_t* model = (tiny_mlp_model_t*)malloc(sizeof(tiny_mlp_model_t));
    if (model == NULL) {
        return NULL;
    }
    
    memset(model, 0, sizeof(tiny_mlp_model_t));
    model->input_size = input_size;
    model->output_size = layer_configs[num_layers - 1].out_features;
    model->num_layers = num_layers;
    
    // Create computation graph
    int graph_capacity = num_layers * 2 + 2;  // FC + activation nodes + input + loss
    model->graph = tiny_graph_create(graph_capacity);
    if (model->graph == NULL) {
        free(model);
        return NULL;
    }
    
    // Allocate arrays for nodes and parameters
    model->fc_nodes = (tiny_graph_node_t**)malloc(num_layers * sizeof(tiny_graph_node_t*));
    model->act_nodes = (tiny_graph_node_t**)malloc(num_layers * sizeof(tiny_graph_node_t*));
    model->fc_params = (tiny_fc_params_t**)malloc(num_layers * sizeof(tiny_fc_params_t*));
    
    if (model->fc_nodes == NULL || model->act_nodes == NULL || model->fc_params == NULL) {
        tiny_graph_destroy(model->graph);
        if (model->fc_nodes) free(model->fc_nodes);
        if (model->act_nodes) free(model->act_nodes);
        if (model->fc_params) free(model->fc_params);
        free(model);
        return NULL;
    }
    
    memset(model->fc_nodes, 0, num_layers * sizeof(tiny_graph_node_t*));
    memset(model->act_nodes, 0, num_layers * sizeof(tiny_graph_node_t*));
    memset(model->fc_params, 0, num_layers * sizeof(tiny_fc_params_t*));
    
    // Create input tensor
    int input_shape[] = {input_size};
    model->input_tensor = tiny_tensor_create(input_shape, 1, TINY_AI_DTYPE_FLOAT32);
    if (model->input_tensor == NULL) {
        tiny_graph_destroy(model->graph);
        free(model->fc_nodes);
        free(model->act_nodes);
        free(model->fc_params);
        free(model);
        return NULL;
    }
    
#if TINY_AI_ENABLE_GRADIENTS
    tiny_tensor_requires_grad(model->input_tensor, true);
#endif
    
    // Create layers and tensors
    tiny_tensor_t* prev_output = model->input_tensor;
    
    for (int i = 0; i < num_layers; i++) {
        const tiny_mlp_layer_config_t* config = &layer_configs[i];
        
        // Create FC layer
        model->fc_params[i] = tiny_fc_create(config->in_features, config->out_features, config->use_bias);
        if (model->fc_params[i] == NULL) {
            // Cleanup
            for (int j = 0; j < i; j++) {
                tiny_fc_destroy(model->fc_params[j]);
            }
            tiny_tensor_destroy(model->input_tensor);
            tiny_graph_destroy(model->graph);
            free(model->fc_nodes);
            free(model->act_nodes);
            free(model->fc_params);
            free(model);
            return NULL;
        }
        
#if TINY_AI_ENABLE_GRADIENTS
        tiny_tensor_requires_grad(model->fc_params[i]->weights, true);
        if (model->fc_params[i]->bias != NULL) {
            tiny_tensor_requires_grad(model->fc_params[i]->bias, true);
        }
#endif
        
        // Create FC node
        model->fc_nodes[i] = tiny_graph_add_node(model->graph, TINY_AI_OP_FULLY_CONNECTED, 1, 1);
        if (model->fc_nodes[i] == NULL) {
            // Cleanup
            for (int j = 0; j <= i; j++) {
                tiny_fc_destroy(model->fc_params[j]);
            }
            tiny_tensor_destroy(model->input_tensor);
            tiny_graph_destroy(model->graph);
            free(model->fc_nodes);
            free(model->act_nodes);
            free(model->fc_params);
            free(model);
            return NULL;
        }
        
        model->fc_nodes[i]->params = model->fc_params[i];
        model->fc_nodes[i]->forward_func = tiny_fc_forward;
        model->fc_nodes[i]->backward_func = tiny_fc_backward;
        model->fc_nodes[i]->inputs[0] = prev_output;
        
        // Create output tensor for FC layer
        int fc_output_shape[] = {config->out_features};
        tiny_tensor_t* fc_output = tiny_tensor_create(fc_output_shape, 1, TINY_AI_DTYPE_FLOAT32);
        if (fc_output == NULL) {
            // Cleanup
            for (int j = 0; j <= i; j++) {
                tiny_fc_destroy(model->fc_params[j]);
            }
            tiny_tensor_destroy(model->input_tensor);
            tiny_graph_destroy(model->graph);
            free(model->fc_nodes);
            free(model->act_nodes);
            free(model->fc_params);
            free(model);
            return NULL;
        }
        
#if TINY_AI_ENABLE_GRADIENTS
        tiny_tensor_requires_grad(fc_output, true);
#endif
        
        model->fc_nodes[i]->outputs[0] = fc_output;
        
        // Connect FC node
        if (i > 0) {
            // Connect previous activation to current FC
            tiny_graph_connect(model->graph, model->act_nodes[i-1], 0, model->fc_nodes[i], 0);
        } else {
            // First layer: connect input to FC
            // Input is not a node, so we just set the connection manually
            // (already done by setting inputs[0])
        }
        
        // Create activation if needed
        if (config->activation != TINY_MLP_ACT_NONE) {
            tiny_ai_op_type_t act_op;
            void (*act_forward)(tiny_graph_node_t*) = NULL;
            void (*act_backward)(tiny_graph_node_t*) = NULL;
            
            switch (config->activation) {
                case TINY_MLP_ACT_RELU:
                    act_op = TINY_AI_OP_RELU;
                    act_forward = tiny_relu_forward;
                    act_backward = tiny_relu_backward;
                    break;
                case TINY_MLP_ACT_SIGMOID:
                    act_op = TINY_AI_OP_SIGMOID;
                    act_forward = tiny_sigmoid_forward;
                    act_backward = tiny_sigmoid_backward;
                    break;
                case TINY_MLP_ACT_TANH:
                    act_op = TINY_AI_OP_TANH;
                    act_forward = tiny_tanh_forward;
                    act_backward = tiny_tanh_backward;
                    break;
                default:
                    act_op = TINY_AI_OP_RELU;
                    act_forward = tiny_relu_forward;
                    act_backward = tiny_relu_backward;
                    break;
            }
            
            model->act_nodes[i] = tiny_graph_add_node(model->graph, act_op, 1, 1);
            if (model->act_nodes[i] == NULL) {
                // Cleanup
                for (int j = 0; j <= i; j++) {
                    tiny_fc_destroy(model->fc_params[j]);
                }
                tiny_tensor_destroy(model->input_tensor);
                tiny_graph_destroy(model->graph);
                free(model->fc_nodes);
                free(model->act_nodes);
                free(model->fc_params);
                free(model);
                return NULL;
            }
            
            model->act_nodes[i]->forward_func = act_forward;
            model->act_nodes[i]->backward_func = act_backward;
            
            // Create activation output tensor
            int act_output_shape[] = {config->out_features};
            tiny_tensor_t* act_output = tiny_tensor_create(act_output_shape, 1, TINY_AI_DTYPE_FLOAT32);
            if (act_output == NULL) {
                // Cleanup
                for (int j = 0; j <= i; j++) {
                    tiny_fc_destroy(model->fc_params[j]);
                }
                tiny_tensor_destroy(model->input_tensor);
                tiny_graph_destroy(model->graph);
                free(model->fc_nodes);
                free(model->act_nodes);
                free(model->fc_params);
                free(model);
                return NULL;
            }
            
#if TINY_AI_ENABLE_GRADIENTS
            tiny_tensor_requires_grad(act_output, true);
#endif
            
            model->act_nodes[i]->outputs[0] = act_output;
            
            // Connect FC to activation
            tiny_graph_connect(model->graph, model->fc_nodes[i], 0, model->act_nodes[i], 0);
            
            prev_output = act_output;
        } else {
            // No activation
            model->act_nodes[i] = NULL;
            prev_output = fc_output;
        }
    }
    
    // Set output tensor
    model->output_tensor = prev_output;
    
    // Create target and loss tensors
    int target_shape[] = {model->output_size};
    model->target_tensor = tiny_tensor_create(target_shape, 1, TINY_AI_DTYPE_FLOAT32);
    
    int loss_shape[] = {1};
    model->loss_tensor = tiny_tensor_create(loss_shape, 1, TINY_AI_DTYPE_FLOAT32);
    
    if (model->target_tensor == NULL || model->loss_tensor == NULL) {
        // Cleanup
        for (int i = 0; i < num_layers; i++) {
            tiny_fc_destroy(model->fc_params[i]);
        }
        tiny_tensor_destroy(model->input_tensor);
        tiny_graph_destroy(model->graph);
        free(model->fc_nodes);
        free(model->act_nodes);
        free(model->fc_params);
        free(model);
        return NULL;
    }
    
#if TINY_AI_ENABLE_GRADIENTS
    tiny_tensor_requires_grad(model->loss_tensor, true);
#endif
    
    // Create loss node (will be configured later)
    model->loss_node = tiny_graph_add_node(model->graph, TINY_AI_OP_MSE_LOSS, 2, 1);
    if (model->loss_node == NULL) {
        // Cleanup
        for (int i = 0; i < num_layers; i++) {
            tiny_fc_destroy(model->fc_params[i]);
        }
        tiny_tensor_destroy(model->input_tensor);
        tiny_tensor_destroy(model->target_tensor);
        tiny_tensor_destroy(model->loss_tensor);
        tiny_graph_destroy(model->graph);
        free(model->fc_nodes);
        free(model->act_nodes);
        free(model->fc_params);
        free(model);
        return NULL;
    }
    
    // Connect last layer to loss
    tiny_graph_node_t* last_output_node = (model->act_nodes[num_layers - 1] != NULL) ?
                                          model->act_nodes[num_layers - 1] : model->fc_nodes[num_layers - 1];
    tiny_graph_connect(model->graph, last_output_node, 0, model->loss_node, 0);
    
    model->loss_node->inputs[1] = model->target_tensor;
    model->loss_node->outputs[0] = model->loss_tensor;
    model->loss_node->forward_func = tiny_mse_forward;
    model->loss_node->backward_func = tiny_mse_backward;
    
    // Build execution order
    tiny_error_t err = tiny_graph_build_order(model->graph);
    if (err != TINY_OK) {
        // Cleanup
        for (int i = 0; i < num_layers; i++) {
            tiny_fc_destroy(model->fc_params[i]);
        }
        tiny_tensor_destroy(model->input_tensor);
        tiny_tensor_destroy(model->target_tensor);
        tiny_tensor_destroy(model->loss_tensor);
        tiny_graph_destroy(model->graph);
        free(model->fc_nodes);
        free(model->act_nodes);
        free(model->fc_params);
        free(model);
        return NULL;
    }
    
    model->initialized = true;
    return model;
}

void tiny_mlp_destroy(tiny_mlp_model_t* model)
{
    if (model == NULL) {
        return;
    }
    
    // Destroy FC parameters
    if (model->fc_params != NULL) {
        for (int i = 0; i < model->num_layers; i++) {
            if (model->fc_params[i] != NULL) {
                tiny_fc_destroy(model->fc_params[i]);
            }
        }
        free(model->fc_params);
    }
    
    // Destroy tensors (graph will handle node tensors, but we need to destroy input/target/loss)
    if (model->input_tensor != NULL) {
        tiny_tensor_destroy(model->input_tensor);
    }
    if (model->target_tensor != NULL) {
        tiny_tensor_destroy(model->target_tensor);
    }
    if (model->loss_tensor != NULL) {
        tiny_tensor_destroy(model->loss_tensor);
    }
    
    // Destroy graph (this will destroy all node tensors)
    if (model->graph != NULL) {
        tiny_graph_destroy(model->graph);
    }
    
    // Free arrays
    if (model->fc_nodes != NULL) {
        free(model->fc_nodes);
    }
    if (model->act_nodes != NULL) {
        free(model->act_nodes);
    }
    
    free(model);
}

/* ============================================================================
 * MODEL CONFIGURATION
 * ============================================================================ */

tiny_error_t tiny_mlp_init_weights_xavier(tiny_mlp_model_t* model, unsigned int seed)
{
    if (model == NULL || !model->initialized) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    unsigned int current_seed = seed;
    for (int i = 0; i < model->num_layers; i++) {
        if (model->fc_params[i] != NULL) {
            tiny_error_t err = tiny_fc_init_weights_xavier(model->fc_params[i], current_seed);
            if (err != TINY_OK) {
                return err;
            }
            current_seed = (current_seed * 1103515245u + 12345u) & 0x7FFFFFFFu;
        }
    }
    
    return TINY_OK;
}

tiny_error_t tiny_mlp_init_weights_zero(tiny_mlp_model_t* model)
{
    if (model == NULL || !model->initialized) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    for (int i = 0; i < model->num_layers; i++) {
        if (model->fc_params[i] != NULL) {
            tiny_error_t err = tiny_fc_init_weights_zero(model->fc_params[i]);
            if (err != TINY_OK) {
                return err;
            }
        }
    }
    
    return TINY_OK;
}

tiny_error_t tiny_mlp_set_loss(tiny_mlp_model_t* model, tiny_ai_op_type_t loss_type)
{
    if (model == NULL || !model->initialized || model->loss_node == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    switch (loss_type) {
        case TINY_AI_OP_MSE_LOSS:
            model->loss_node->forward_func = tiny_mse_forward;
            model->loss_node->backward_func = tiny_mse_backward;
            break;
        case TINY_AI_OP_CROSS_ENTROPY_LOSS:
            model->loss_node->forward_func = tiny_cross_entropy_forward;
            model->loss_node->backward_func = tiny_cross_entropy_backward;
            break;
        default:
            return TINY_ERR_AI_NOT_SUPPORTED;
    }
    
    model->loss_node->op_type = loss_type;
    return TINY_OK;
}

/* ============================================================================
 * MODEL EXECUTION
 * ============================================================================ */

tiny_error_t tiny_mlp_forward(tiny_mlp_model_t* model,
                              const float* input_data,
                              float* output_data)
{
    if (model == NULL || !model->initialized || input_data == NULL || output_data == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    // Copy input data to input tensor
    float* input_tensor_data = (float*)tiny_tensor_data(model->input_tensor);
    if (input_tensor_data == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    memcpy(input_tensor_data, input_data, model->input_size * sizeof(float));
    
    // Set inference mode
    tiny_graph_set_training_mode(model->graph, false);
    
    // Forward propagation
    tiny_error_t err = tiny_graph_forward(model->graph);
    if (err != TINY_OK) {
        return err;
    }
    
    // Copy output data from output tensor
    float* output_tensor_data = (float*)tiny_tensor_data(model->output_tensor);
    if (output_tensor_data == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    memcpy(output_data, output_tensor_data, model->output_size * sizeof(float));
    
    return TINY_OK;
}

int tiny_mlp_get_parameters(tiny_mlp_model_t* model,
                            tiny_tensor_t** params,
                            int max_params)
{
    if (model == NULL || !model->initialized || params == NULL || max_params <= 0) {
        return 0;
    }
    
    int count = 0;
    for (int i = 0; i < model->num_layers && count < max_params; i++) {
        if (model->fc_params[i] != NULL) {
            if (count < max_params) {
                params[count++] = model->fc_params[i]->weights;
            }
            if (model->fc_params[i]->bias != NULL && count < max_params) {
                params[count++] = model->fc_params[i]->bias;
            }
        }
    }
    
    return count;
}

/* ============================================================================
 * TRAINING
 * ============================================================================ */

tiny_error_t tiny_mlp_train_epoch(tiny_mlp_model_t* model,
                                   tiny_dataloader_t* loader,
                                   tiny_ai_op_type_t loss_type,
                                   void** optimizers,
                                   int num_optimizers,
                                   tiny_training_callback_t callback,
                                   void* user_data)
{
    if (model == NULL || !model->initialized || loader == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    // Set loss function
    tiny_error_t err = tiny_mlp_set_loss(model, loss_type);
    if (err != TINY_OK) {
        return err;
    }
    
    // Get parameters
    tiny_tensor_t* params[64];  // Max 64 parameters (should be enough for most models)
    int num_params = tiny_mlp_get_parameters(model, params, 64);
    
    if (num_params != num_optimizers) {
        return TINY_ERR_AI_INVALID_ARG;  // Number of optimizers must match parameters
    }
    
    // Reset dataloader
    err = tiny_dataloader_reset(loader);
    if (err != TINY_OK) {
        return err;
    }
    
    // Training loop
    tiny_tensor_t* batch_inputs[32];  // Max batch size 32
    tiny_tensor_t* batch_targets[32];
    int actual_batch_size;
    
    int batch_idx = 0;
    while (tiny_dataloader_has_next(loader)) {
        err = tiny_dataloader_get_batch(loader, batch_inputs, batch_targets, &actual_batch_size);
        if (err != TINY_OK) {
            return err;
        }
        
        // For each sample in batch
        for (int i = 0; i < actual_batch_size; i++) {
            // Feed watchdog and yield CPU every 10 samples to prevent timeout
            if (i % 10 == 0) {
                TINY_AI_FEED_WATCHDOG();
                TINY_AI_YIELD_CPU();
            }
            
            // Copy input and target to model tensors
            float* input_data = (float*)tiny_tensor_data(batch_inputs[i]);
            float* target_data = (float*)tiny_tensor_data(batch_targets[i]);
            float* model_input = (float*)tiny_tensor_data(model->input_tensor);
            float* model_target = (float*)tiny_tensor_data(model->target_tensor);
            
            if (input_data == NULL || target_data == NULL || 
                model_input == NULL || model_target == NULL) {
                continue;
            }
            
            memcpy(model_input, input_data, model->input_size * sizeof(float));
            memcpy(model_target, target_data, model->output_size * sizeof(float));
            
            // Training step
            err = tiny_trainer_step(model->graph, model->input_tensor, model->target_tensor,
                                   model->loss_node, optimizers, num_optimizers,
                                   params, num_params);
            if (err != TINY_OK) {
                return err;
            }
            
            // Get loss value from loss node output
            float loss = 0.0f;
            if (model->loss_node != NULL && model->loss_node->outputs != NULL && 
                model->loss_node->outputs[0] != NULL) {
                float* loss_data = (float*)tiny_tensor_data(model->loss_node->outputs[0]);
                if (loss_data != NULL) {
                    loss = loss_data[0];
                }
            }
            
            // Callback
            if (callback != NULL) {
                tiny_training_stats_t stats = {0};
                stats.current_loss = loss;
                stats.current_epoch = 0;  // Will be set by trainer if needed
                stats.current_batch = batch_idx * actual_batch_size + i;
                stats.total_batches = loader->num_batches;
                callback(&stats, user_data);
            }
        }
        
        // Feed watchdog and yield CPU after each batch
        TINY_AI_FEED_WATCHDOG();
        TINY_AI_YIELD_CPU();
        
        batch_idx++;
    }
    
    return TINY_OK;
}

