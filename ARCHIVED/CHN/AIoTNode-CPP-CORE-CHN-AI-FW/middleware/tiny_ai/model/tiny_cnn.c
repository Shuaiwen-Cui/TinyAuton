/**
 * @file tiny_cnn.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief CNN Model implementation
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 */

#include "tiny_cnn.h"
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
 * HELPER FUNCTIONS
 * ============================================================================ */

/**
 * @brief Calculate output dimensions after convolution
 */
static void calc_conv_output_size(int in_h, int in_w, int kernel_h, int kernel_w,
                                  int stride_h, int stride_w, int pad_h, int pad_w,
                                  int* out_h, int* out_w)
{
    *out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    *out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
}

/**
 * @brief Calculate output dimensions after pooling
 */
static void calc_pool_output_size(int in_h, int in_w, int kernel_h, int kernel_w,
                                 int stride_h, int stride_w, int pad_h, int pad_w,
                                 int* out_h, int* out_w)
{
    *out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    *out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
}

/**
 * @brief Convert activation type to op type and functions
 */
static void get_activation_funcs(tiny_cnn_activation_t act_type,
                                 tiny_ai_op_type_t* op_type,
                                 void (**forward_func)(tiny_graph_node_t*),
                                 void (**backward_func)(tiny_graph_node_t*))
{
    switch (act_type) {
        case TINY_CNN_ACT_RELU:
            *op_type = TINY_AI_OP_RELU;
            *forward_func = tiny_relu_forward;
            *backward_func = tiny_relu_backward;
            break;
        case TINY_CNN_ACT_SIGMOID:
            *op_type = TINY_AI_OP_SIGMOID;
            *forward_func = tiny_sigmoid_forward;
            *backward_func = tiny_sigmoid_backward;
            break;
        case TINY_CNN_ACT_TANH:
            *op_type = TINY_AI_OP_TANH;
            *forward_func = tiny_tanh_forward;
            *backward_func = tiny_tanh_backward;
            break;
        default:
            *op_type = TINY_AI_OP_RELU;
            *forward_func = tiny_relu_forward;
            *backward_func = tiny_relu_backward;
            break;
    }
}

/* ============================================================================
 * MODEL CREATION AND DESTRUCTION
 * ============================================================================ */

tiny_cnn_model_t* tiny_cnn_create(int input_channels, int input_h, int input_w,
                                   const tiny_cnn_block_config_t* block_configs,
                                   int num_blocks)
{
    if (input_channels <= 0 || input_h <= 0 || input_w <= 0 ||
        block_configs == NULL || num_blocks <= 0) {
        return NULL;
    }
    
    // Validate first block
    if (block_configs[0].block_type != TINY_CNN_BLOCK_CONV) {
        return NULL;  // First block should be convolution
    }
    if (block_configs[0].config.conv.in_channels != input_channels) {
        return NULL;  // First conv input channels must match model input
    }
    
    // Allocate model structure
    tiny_cnn_model_t* model = (tiny_cnn_model_t*)malloc(sizeof(tiny_cnn_model_t));
    if (model == NULL) {
        return NULL;
    }
    
    memset(model, 0, sizeof(tiny_cnn_model_t));
    model->input_channels = input_channels;
    model->input_h = input_h;
    model->input_w = input_w;
    model->num_blocks = num_blocks;
    
    // Count different layer types
    for (int i = 0; i < num_blocks; i++) {
        switch (block_configs[i].block_type) {
            case TINY_CNN_BLOCK_CONV:
                model->num_conv++;
                if (block_configs[i].config.conv.use_batchnorm) {
                    model->num_norm++;
                }
                break;
            case TINY_CNN_BLOCK_POOL:
                model->num_pool++;
                break;
            case TINY_CNN_BLOCK_NORM:
                model->num_norm++;
                break;
            case TINY_CNN_BLOCK_FC:
                model->num_fc++;
                break;
        }
    }
    
    // Create computation graph
    int graph_capacity = num_blocks * 3 + 10;  // Extra capacity for activations and norms
    model->graph = tiny_graph_create(graph_capacity);
    if (model->graph == NULL) {
        free(model);
        return NULL;
    }
    
    // Allocate arrays
    model->block_nodes = (tiny_graph_node_t**)malloc(num_blocks * 3 * sizeof(tiny_graph_node_t*));
    model->conv_params = (void**)malloc(model->num_conv * sizeof(void*));
    model->pool_params = (void**)malloc(model->num_pool * sizeof(void*));
    model->norm_params = (void**)malloc(model->num_norm * sizeof(void*));
    model->fc_params = (void**)malloc(model->num_fc * sizeof(void*));
    
    if (model->block_nodes == NULL || model->conv_params == NULL ||
        model->pool_params == NULL || model->norm_params == NULL ||
        model->fc_params == NULL) {
        tiny_graph_destroy(model->graph);
        if (model->block_nodes) free(model->block_nodes);
        if (model->conv_params) free(model->conv_params);
        if (model->pool_params) free(model->pool_params);
        if (model->norm_params) free(model->norm_params);
        if (model->fc_params) free(model->fc_params);
        free(model);
        return NULL;
    }
    
    memset(model->block_nodes, 0, num_blocks * 3 * sizeof(tiny_graph_node_t*));
    memset(model->conv_params, 0, model->num_conv * sizeof(void*));
    memset(model->pool_params, 0, model->num_pool * sizeof(void*));
    memset(model->norm_params, 0, model->num_norm * sizeof(void*));
    memset(model->fc_params, 0, model->num_fc * sizeof(void*));
    
    // Create input tensor [batch, channels, h, w] (batch=1 for now)
    int input_shape[] = {1, input_channels, input_h, input_w};
    model->input_tensor = tiny_tensor_create(input_shape, 4, TINY_AI_DTYPE_FLOAT32);
    if (model->input_tensor == NULL) {
        tiny_graph_destroy(model->graph);
        free(model->block_nodes);
        free(model->conv_params);
        free(model->pool_params);
        free(model->norm_params);
        free(model->fc_params);
        free(model);
        return NULL;
    }
    
#if TINY_AI_ENABLE_GRADIENTS
    tiny_tensor_requires_grad(model->input_tensor, true);
#endif
    
    // Build model by processing each block
    tiny_tensor_t* prev_output = model->input_tensor;
    int current_h = input_h;
    int current_w = input_w;
    int current_channels = input_channels;
    int conv_idx = 0, pool_idx = 0, norm_idx = 0, fc_idx = 0;
    int node_idx = 0;
    
    for (int i = 0; i < num_blocks; i++) {
        const tiny_cnn_block_config_t* block = &block_configs[i];
        
        switch (block->block_type) {
            case TINY_CNN_BLOCK_CONV: {
                const tiny_cnn_conv_block_t* conv_config = &block->config.conv;
                
                // Validate input channels
                if (conv_config->in_channels != current_channels) {
                    // Cleanup and return
                    goto cleanup_error;
                }
                
                // Calculate output dimensions
                int out_h, out_w;
                calc_conv_output_size(current_h, current_w,
                                     conv_config->kernel_h, conv_config->kernel_w,
                                     conv_config->stride_h, conv_config->stride_w,
                                     conv_config->pad_h, conv_config->pad_w,
                                     &out_h, &out_w);
                
                // Create conv layer
                tiny_convolution_params_t* conv_params = tiny_convolution_create(
                    conv_config->in_channels, conv_config->out_channels,
                    conv_config->kernel_h, conv_config->kernel_w,
                    conv_config->stride_h, conv_config->stride_w,
                    conv_config->pad_h, conv_config->pad_w,
                    conv_config->use_bias
                );
                
                if (conv_params == NULL) {
                    goto cleanup_error;
                }
                
                model->conv_params[conv_idx++] = conv_params;
                
#if TINY_AI_ENABLE_GRADIENTS
                tiny_tensor_requires_grad(conv_params->weights, true);
                if (conv_params->bias != NULL) {
                    tiny_tensor_requires_grad(conv_params->bias, true);
                }
#endif
                
                // Create conv node
                tiny_graph_node_t* conv_node = tiny_graph_add_node(
                    model->graph, TINY_AI_OP_CONV2D, 1, 1
                );
                if (conv_node == NULL) {
                    tiny_convolution_destroy(conv_params);
                    goto cleanup_error;
                }
                
                conv_node->params = conv_params;
                conv_node->forward_func = tiny_convolution_forward;
                conv_node->backward_func = tiny_convolution_backward;
                conv_node->inputs[0] = prev_output;
                
                // Create conv output tensor
                int conv_output_shape[] = {1, conv_config->out_channels, out_h, out_w};
                tiny_tensor_t* conv_output = tiny_tensor_create(conv_output_shape, 4, TINY_AI_DTYPE_FLOAT32);
                if (conv_output == NULL) {
                    tiny_convolution_destroy(conv_params);
                    goto cleanup_error;
                }
                
#if TINY_AI_ENABLE_GRADIENTS
                tiny_tensor_requires_grad(conv_output, true);
#endif
                
                conv_node->outputs[0] = conv_output;
                
                // Connect
                if (i > 0) {
                    tiny_graph_connect(model->graph, model->block_nodes[node_idx - 1], 0, conv_node, 0);
                }
                
                model->block_nodes[node_idx++] = conv_node;
                prev_output = conv_output;
                current_channels = conv_config->out_channels;
                current_h = out_h;
                current_w = out_w;
                
                // Add batch normalization if requested
                if (conv_config->use_batchnorm) {
                    tiny_batchnorm_params_t* norm_params = tiny_batchnorm_create(
                        current_channels, 0.9f, 1e-5f, true
                    );
                    
                    if (norm_params == NULL) {
                        goto cleanup_error;
                    }
                    
                    model->norm_params[norm_idx++] = norm_params;
                    
                    tiny_graph_node_t* norm_node = tiny_graph_add_node(
                        model->graph, TINY_AI_OP_BATCHNORM, 1, 1
                    );
                    
                    if (norm_node == NULL) {
                        tiny_batchnorm_destroy(norm_params);
                        goto cleanup_error;
                    }
                    
                    norm_node->params = norm_params;
                    norm_node->forward_func = tiny_batchnorm_forward;
                    norm_node->backward_func = tiny_batchnorm_backward;
                    
                    int norm_output_shape[] = {1, current_channels, current_h, current_w};
                    tiny_tensor_t* norm_output = tiny_tensor_create(norm_output_shape, 4, TINY_AI_DTYPE_FLOAT32);
                    if (norm_output == NULL) {
                        tiny_batchnorm_destroy(norm_params);
                        goto cleanup_error;
                    }
                    
#if TINY_AI_ENABLE_GRADIENTS
                    tiny_tensor_requires_grad(norm_output, true);
#endif
                    
                    norm_node->outputs[0] = norm_output;
                    tiny_graph_connect(model->graph, conv_node, 0, norm_node, 0);
                    
                    model->block_nodes[node_idx++] = norm_node;
                    prev_output = norm_output;
                }
                
                // Add activation if specified
                if (conv_config->activation != TINY_CNN_ACT_NONE) {
                    tiny_ai_op_type_t act_op;
                    void (*act_forward)(tiny_graph_node_t*) = NULL;
                    void (*act_backward)(tiny_graph_node_t*) = NULL;
                    get_activation_funcs(conv_config->activation, &act_op, &act_forward, &act_backward);
                    
                    tiny_graph_node_t* act_node = tiny_graph_add_node(model->graph, act_op, 1, 1);
                    if (act_node == NULL) {
                        goto cleanup_error;
                    }
                    
                    act_node->forward_func = act_forward;
                    act_node->backward_func = act_backward;
                    
                    int act_output_shape[] = {1, current_channels, current_h, current_w};
                    tiny_tensor_t* act_output = tiny_tensor_create(act_output_shape, 4, TINY_AI_DTYPE_FLOAT32);
                    if (act_output == NULL) {
                        goto cleanup_error;
                    }
                    
#if TINY_AI_ENABLE_GRADIENTS
                    tiny_tensor_requires_grad(act_output, true);
#endif
                    
                    act_node->outputs[0] = act_output;
                    tiny_graph_connect(model->graph, prev_output == conv_output ? conv_node : model->block_nodes[node_idx - 1], 0, act_node, 0);
                    
                    model->block_nodes[node_idx++] = act_node;
                    prev_output = act_output;
                }
                
                break;
            }
            
            case TINY_CNN_BLOCK_POOL: {
                const tiny_cnn_pool_block_t* pool_config = &block->config.pool;
                
                // Calculate output dimensions
                int out_h, out_w;
                calc_pool_output_size(current_h, current_w,
                                     pool_config->kernel_h, pool_config->kernel_w,
                                     pool_config->stride_h, pool_config->stride_w,
                                     pool_config->pad_h, pool_config->pad_w,
                                     &out_h, &out_w);
                
                // Create pool layer
                tiny_pool_params_t* pool_params = tiny_pool_create(
                    pool_config->pool_type,
                    pool_config->kernel_h, pool_config->kernel_w,
                    pool_config->stride_h, pool_config->stride_w,
                    pool_config->pad_h, pool_config->pad_w
                );
                
                if (pool_params == NULL) {
                    goto cleanup_error;
                }
                
                model->pool_params[pool_idx++] = pool_params;
                
                // Create pool node
                tiny_graph_node_t* pool_node = tiny_graph_add_node(
                    model->graph, TINY_AI_OP_POOL2D, 1, 1
                );
                
                if (pool_node == NULL) {
                    tiny_pool_destroy(pool_params);
                    goto cleanup_error;
                }
                
                pool_node->params = pool_params;
                pool_node->forward_func = tiny_pool_forward;
                pool_node->backward_func = tiny_pool_backward;
                
                int pool_output_shape[] = {1, current_channels, out_h, out_w};
                tiny_tensor_t* pool_output = tiny_tensor_create(pool_output_shape, 4, TINY_AI_DTYPE_FLOAT32);
                if (pool_output == NULL) {
                    tiny_pool_destroy(pool_params);
                    goto cleanup_error;
                }
                
#if TINY_AI_ENABLE_GRADIENTS
                tiny_tensor_requires_grad(pool_output, true);
#endif
                
                pool_node->outputs[0] = pool_output;
                tiny_graph_connect(model->graph, model->block_nodes[node_idx - 1], 0, pool_node, 0);
                
                model->block_nodes[node_idx++] = pool_node;
                prev_output = pool_output;
                current_h = out_h;
                current_w = out_w;
                // Channels remain the same
                
                break;
            }
            
            case TINY_CNN_BLOCK_FC: {
                const tiny_cnn_fc_block_t* fc_config = &block->config.fc;
                
                // Flatten: convert [batch, channels, h, w] to [batch, features]
                int flattened_features = current_channels * current_h * current_w;
                
                if (fc_config->in_features != flattened_features && i > 0) {
                    // Need to create a flatten operation or adjust
                    // For simplicity, we'll require in_features to match
                    if (fc_config->in_features != flattened_features) {
                        goto cleanup_error;
                    }
                }
                
                // Create FC layer
                tiny_fc_params_t* fc_params = tiny_fc_create(
                    fc_config->in_features, fc_config->out_features, fc_config->use_bias
                );
                
                if (fc_params == NULL) {
                    goto cleanup_error;
                }
                
                model->fc_params[fc_idx++] = fc_params;
                
#if TINY_AI_ENABLE_GRADIENTS
                tiny_tensor_requires_grad(fc_params->weights, true);
                if (fc_params->bias != NULL) {
                    tiny_tensor_requires_grad(fc_params->bias, true);
                }
#endif
                
                // Create FC node
                tiny_graph_node_t* fc_node = tiny_graph_add_node(
                    model->graph, TINY_AI_OP_FULLY_CONNECTED, 1, 1
                );
                
                if (fc_node == NULL) {
                    tiny_fc_destroy(fc_params);
                    goto cleanup_error;
                }
                
                fc_node->params = fc_params;
                fc_node->forward_func = tiny_fc_forward;
                fc_node->backward_func = tiny_fc_backward;
                
                // For FC, we need to reshape input from 4D to 2D
                // Create a reshaped input tensor (view or copy)
                // For simplicity, we'll create a new tensor and copy data in forward pass
                // Or we can require the user to add a flatten layer before FC
                // For now, let's assume prev_output is already flattened or we handle it
                
                // Create FC input tensor (flattened)
                int fc_input_shape[] = {flattened_features};
                tiny_tensor_t* fc_input = tiny_tensor_create(fc_input_shape, 1, TINY_AI_DTYPE_FLOAT32);
                if (fc_input == NULL) {
                    tiny_fc_destroy(fc_params);
                    goto cleanup_error;
                }
                
                // Note: In a real implementation, we'd need to handle the reshape/flatten
                // For now, we'll connect the previous output directly
                // This requires the previous layer to output the correct shape
                
                fc_node->inputs[0] = prev_output;  // This might need reshaping
                
                int fc_output_shape[] = {fc_config->out_features};
                tiny_tensor_t* fc_output = tiny_tensor_create(fc_output_shape, 1, TINY_AI_DTYPE_FLOAT32);
                if (fc_output == NULL) {
                    tiny_tensor_destroy(fc_input);
                    tiny_fc_destroy(fc_params);
                    goto cleanup_error;
                }
                
#if TINY_AI_ENABLE_GRADIENTS
                tiny_tensor_requires_grad(fc_output, true);
#endif
                
                fc_node->outputs[0] = fc_output;
                tiny_graph_connect(model->graph, model->block_nodes[node_idx - 1], 0, fc_node, 0);
                
                model->block_nodes[node_idx++] = fc_node;
                prev_output = fc_output;
                
                // Add activation if specified
                if (fc_config->activation != TINY_CNN_ACT_NONE) {
                    tiny_ai_op_type_t act_op;
                    void (*act_forward)(tiny_graph_node_t*) = NULL;
                    void (*act_backward)(tiny_graph_node_t*) = NULL;
                    get_activation_funcs(fc_config->activation, &act_op, &act_forward, &act_backward);
                    
                    tiny_graph_node_t* act_node = tiny_graph_add_node(model->graph, act_op, 1, 1);
                    if (act_node == NULL) {
                        goto cleanup_error;
                    }
                    
                    act_node->forward_func = act_forward;
                    act_node->backward_func = act_backward;
                    
                    int act_output_shape[] = {fc_config->out_features};
                    tiny_tensor_t* act_output = tiny_tensor_create(act_output_shape, 1, TINY_AI_DTYPE_FLOAT32);
                    if (act_output == NULL) {
                        goto cleanup_error;
                    }
                    
#if TINY_AI_ENABLE_GRADIENTS
                    tiny_tensor_requires_grad(act_output, true);
#endif
                    
                    act_node->outputs[0] = act_output;
                    tiny_graph_connect(model->graph, fc_node, 0, act_node, 0);
                    
                    model->block_nodes[node_idx++] = act_node;
                    prev_output = act_output;
                }
                
                model->output_size = fc_config->out_features;
                break;
            }
            
            default:
                goto cleanup_error;
        }
    }
    
    // Create target and loss tensors
    int target_shape[] = {model->output_size};
    model->target_tensor = tiny_tensor_create(target_shape, 1, TINY_AI_DTYPE_FLOAT32);
    
    int loss_shape[] = {1};
    model->loss_tensor = tiny_tensor_create(loss_shape, 1, TINY_AI_DTYPE_FLOAT32);
    
    if (model->target_tensor == NULL || model->loss_tensor == NULL) {
        goto cleanup_error;
    }
    
#if TINY_AI_ENABLE_GRADIENTS
    tiny_tensor_requires_grad(model->loss_tensor, true);
#endif
    
    // Set output tensor
    model->output_tensor = prev_output;
    
    // Create loss node
    model->loss_node = tiny_graph_add_node(model->graph, TINY_AI_OP_MSE_LOSS, 2, 1);
    if (model->loss_node == NULL) {
        goto cleanup_error;
    }
    
    tiny_graph_connect(model->graph, model->block_nodes[node_idx - 1], 0, model->loss_node, 0);
    model->loss_node->inputs[1] = model->target_tensor;
    model->loss_node->outputs[0] = model->loss_tensor;
    model->loss_node->forward_func = tiny_mse_forward;
    model->loss_node->backward_func = tiny_mse_backward;
    
    // Build execution order
    tiny_error_t err = tiny_graph_build_order(model->graph);
    if (err != TINY_OK) {
        goto cleanup_error;
    }
    
    model->initialized = true;
    return model;
    
cleanup_error:
    // Cleanup all allocated resources
    for (int i = 0; i < conv_idx; i++) {
        if (model->conv_params[i] != NULL) {
            tiny_convolution_destroy((tiny_convolution_params_t*)model->conv_params[i]);
        }
    }
    for (int i = 0; i < pool_idx; i++) {
        if (model->pool_params[i] != NULL) {
            tiny_pool_destroy((tiny_pool_params_t*)model->pool_params[i]);
        }
    }
    for (int i = 0; i < norm_idx; i++) {
        if (model->norm_params[i] != NULL) {
            tiny_batchnorm_destroy((tiny_batchnorm_params_t*)model->norm_params[i]);
        }
    }
    for (int i = 0; i < fc_idx; i++) {
        if (model->fc_params[i] != NULL) {
            tiny_fc_destroy((tiny_fc_params_t*)model->fc_params[i]);
        }
    }
    
    if (model->input_tensor != NULL) {
        tiny_tensor_destroy(model->input_tensor);
    }
    if (model->target_tensor != NULL) {
        tiny_tensor_destroy(model->target_tensor);
    }
    if (model->loss_tensor != NULL) {
        tiny_tensor_destroy(model->loss_tensor);
    }
    
    if (model->graph != NULL) {
        tiny_graph_destroy(model->graph);
    }
    
    if (model->block_nodes != NULL) {
        free(model->block_nodes);
    }
    if (model->conv_params != NULL) {
        free(model->conv_params);
    }
    if (model->pool_params != NULL) {
        free(model->pool_params);
    }
    if (model->norm_params != NULL) {
        free(model->norm_params);
    }
    if (model->fc_params != NULL) {
        free(model->fc_params);
    }
    
    free(model);
    return NULL;
}

void tiny_cnn_destroy(tiny_cnn_model_t* model)
{
    if (model == NULL) {
        return;
    }
    
    // Destroy layer parameters
    if (model->conv_params != NULL) {
        for (int i = 0; i < model->num_conv; i++) {
            if (model->conv_params[i] != NULL) {
                tiny_convolution_destroy((tiny_convolution_params_t*)model->conv_params[i]);
            }
        }
        free(model->conv_params);
    }
    
    if (model->pool_params != NULL) {
        for (int i = 0; i < model->num_pool; i++) {
            if (model->pool_params[i] != NULL) {
                tiny_pool_destroy((tiny_pool_params_t*)model->pool_params[i]);
            }
        }
        free(model->pool_params);
    }
    
    if (model->norm_params != NULL) {
        for (int i = 0; i < model->num_norm; i++) {
            if (model->norm_params[i] != NULL) {
                tiny_batchnorm_destroy((tiny_batchnorm_params_t*)model->norm_params[i]);
            }
        }
        free(model->norm_params);
    }
    
    if (model->fc_params != NULL) {
        for (int i = 0; i < model->num_fc; i++) {
            if (model->fc_params[i] != NULL) {
                tiny_fc_destroy((tiny_fc_params_t*)model->fc_params[i]);
            }
        }
        free(model->fc_params);
    }
    
    // Destroy tensors
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
    if (model->block_nodes != NULL) {
        free(model->block_nodes);
    }
    
    free(model);
}

/* ============================================================================
 * MODEL CONFIGURATION
 * ============================================================================ */

tiny_error_t tiny_cnn_init_weights_xavier(tiny_cnn_model_t* model, unsigned int seed)
{
    if (model == NULL || !model->initialized) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    unsigned int current_seed = seed;
    
    // Initialize conv layers
    for (int i = 0; i < model->num_conv; i++) {
        if (model->conv_params[i] != NULL) {
            tiny_error_t err = tiny_convolution_init_weights_xavier(
                (tiny_convolution_params_t*)model->conv_params[i], current_seed
            );
            if (err != TINY_OK) {
                return err;
            }
            current_seed = (current_seed * 1103515245u + 12345u) & 0x7FFFFFFFu;
        }
    }
    
    // Initialize FC layers
    for (int i = 0; i < model->num_fc; i++) {
        if (model->fc_params[i] != NULL) {
            tiny_error_t err = tiny_fc_init_weights_xavier(
                (tiny_fc_params_t*)model->fc_params[i], current_seed
            );
            if (err != TINY_OK) {
                return err;
            }
            current_seed = (current_seed * 1103515245u + 12345u) & 0x7FFFFFFFu;
        }
    }
    
    return TINY_OK;
}

tiny_error_t tiny_cnn_init_weights_zero(tiny_cnn_model_t* model)
{
    if (model == NULL || !model->initialized) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    // Initialize conv layers
    for (int i = 0; i < model->num_conv; i++) {
        if (model->conv_params[i] != NULL) {
            tiny_error_t err = tiny_convolution_init_weights_zero((tiny_convolution_params_t*)model->conv_params[i]);
            if (err != TINY_OK) {
                return err;
            }
        }
    }
    
    // Initialize FC layers
    for (int i = 0; i < model->num_fc; i++) {
        if (model->fc_params[i] != NULL) {
            tiny_error_t err = tiny_fc_init_weights_zero((tiny_fc_params_t*)model->fc_params[i]);
            if (err != TINY_OK) {
                return err;
            }
        }
    }
    
    return TINY_OK;
}

tiny_error_t tiny_cnn_set_loss(tiny_cnn_model_t* model, tiny_ai_op_type_t loss_type)
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

tiny_error_t tiny_cnn_forward(tiny_cnn_model_t* model,
                              const float* input_data,
                              int batch_size,
                              float* output_data)
{
    if (model == NULL || !model->initialized || input_data == NULL || output_data == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    if (batch_size <= 0) {
        return TINY_ERR_AI_INVALID_ARG;
    }
    
    // Reshape input tensor if batch size changed
    // For simplicity, we assume batch_size = 1 for now
    // In a full implementation, we'd need to handle dynamic batch sizes
    
    // Copy input data to input tensor
    float* input_tensor_data = (float*)tiny_tensor_data(model->input_tensor);
    if (input_tensor_data == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    int input_size = model->input_channels * model->input_h * model->input_w;
    memcpy(input_tensor_data, input_data, input_size * sizeof(float));
    
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

int tiny_cnn_get_parameters(tiny_cnn_model_t* model,
                            tiny_tensor_t** params,
                            int max_params)
{
    if (model == NULL || !model->initialized || params == NULL || max_params <= 0) {
        return 0;
    }
    
    int count = 0;
    
    // Get conv parameters
    for (int i = 0; i < model->num_conv && count < max_params; i++) {
        if (model->conv_params[i] != NULL) {
            tiny_convolution_params_t* conv = (tiny_convolution_params_t*)model->conv_params[i];
            if (count < max_params) {
                params[count++] = conv->weights;
            }
            if (conv->bias != NULL && count < max_params) {
                params[count++] = conv->bias;
            }
        }
    }
    
    // Get norm parameters (gamma and beta)
    for (int i = 0; i < model->num_norm && count < max_params; i++) {
        if (model->norm_params[i] != NULL) {
            tiny_batchnorm_params_t* norm = (tiny_batchnorm_params_t*)model->norm_params[i];
            if (norm->gamma != NULL && count < max_params) {
                params[count++] = norm->gamma;
            }
            if (norm->beta != NULL && count < max_params) {
                params[count++] = norm->beta;
            }
        }
    }
    
    // Get FC parameters
    for (int i = 0; i < model->num_fc && count < max_params; i++) {
        if (model->fc_params[i] != NULL) {
            tiny_fc_params_t* fc = (tiny_fc_params_t*)model->fc_params[i];
            if (count < max_params) {
                params[count++] = fc->weights;
            }
            if (fc->bias != NULL && count < max_params) {
                params[count++] = fc->bias;
            }
        }
    }
    
    return count;
}

/* ============================================================================
 * TRAINING
 * ============================================================================ */

tiny_error_t tiny_cnn_train_epoch(tiny_cnn_model_t* model,
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
    tiny_error_t err = tiny_cnn_set_loss(model, loss_type);
    if (err != TINY_OK) {
        return err;
    }
    
    // Get parameters
    tiny_tensor_t* params[128];  // Max 128 parameters
    int num_params = tiny_cnn_get_parameters(model, params, 128);
    
    if (num_params != num_optimizers) {
        return TINY_ERR_AI_INVALID_ARG;
    }
    
    // Reset dataloader
    err = tiny_dataloader_reset(loader);
    if (err != TINY_OK) {
        return err;
    }
    
    // Training loop
    tiny_tensor_t* batch_inputs[32];
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
            
            int input_size = model->input_channels * model->input_h * model->input_w;
            memcpy(model_input, input_data, input_size * sizeof(float));
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

