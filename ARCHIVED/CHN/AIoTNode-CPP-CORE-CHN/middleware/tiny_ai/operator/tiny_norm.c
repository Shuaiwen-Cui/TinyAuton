/**
 * @file tiny_norm.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Normalization Layer implementation
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 */

#include "tiny_norm.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ============================================================================
 * LAYER CREATION AND DESTRUCTION
 * ============================================================================ */

tiny_batchnorm_params_t* tiny_batchnorm_create(int num_channels, 
                                                float momentum, 
                                                float eps, 
                                                bool affine)
{
    if (num_channels <= 0 || momentum < 0.0f || momentum > 1.0f || eps <= 0.0f) {
        return NULL;
    }
    
    // Allocate parameters structure
    tiny_batchnorm_params_t* params = (tiny_batchnorm_params_t*)malloc(sizeof(tiny_batchnorm_params_t));
    if (params == NULL) {
        return NULL;
    }
    
    memset(params, 0, sizeof(tiny_batchnorm_params_t));
    params->num_channels = num_channels;
    params->momentum = momentum;
    params->eps = eps;
    params->affine = affine;
    
    // Create gamma (scale) tensor [channels] if affine
    if (affine) {
        int gamma_shape[] = {num_channels};
        params->gamma = tiny_tensor_create(gamma_shape, 1, TINY_AI_DTYPE_FLOAT32);
        if (params->gamma == NULL) {
            free(params);
            return NULL;
        }
        
        // Initialize gamma to 1.0
        float* gamma_data = (float*)tiny_tensor_data(params->gamma);
        for (int i = 0; i < num_channels; i++) {
            gamma_data[i] = 1.0f;
        }
        
        // Enable gradients for gamma
#if TINY_AI_ENABLE_GRADIENTS
        tiny_tensor_requires_grad(params->gamma, true);
#endif
        
        // Create beta (shift) tensor [channels]
        int beta_shape[] = {num_channels};
        params->beta = tiny_tensor_create(beta_shape, 1, TINY_AI_DTYPE_FLOAT32);
        if (params->beta == NULL) {
            tiny_tensor_destroy(params->gamma);
            free(params);
            return NULL;
        }
        
        // Initialize beta to 0.0
        float* beta_data = (float*)tiny_tensor_data(params->beta);
        for (int i = 0; i < num_channels; i++) {
            beta_data[i] = 0.0f;
        }
        
        // Enable gradients for beta
#if TINY_AI_ENABLE_GRADIENTS
        tiny_tensor_requires_grad(params->beta, true);
#endif
    } else {
        params->gamma = NULL;
        params->beta = NULL;
    }
    
    // Create running mean tensor [channels]
    int mean_shape[] = {num_channels};
    params->running_mean = tiny_tensor_create(mean_shape, 1, TINY_AI_DTYPE_FLOAT32);
    if (params->running_mean == NULL) {
        if (params->gamma != NULL) tiny_tensor_destroy(params->gamma);
        if (params->beta != NULL) tiny_tensor_destroy(params->beta);
        free(params);
        return NULL;
    }
    
    // Initialize running mean to 0.0
    tiny_tensor_zero(params->running_mean);
    
    // Create running variance tensor [channels]
    int var_shape[] = {num_channels};
    params->running_var = tiny_tensor_create(var_shape, 1, TINY_AI_DTYPE_FLOAT32);
    if (params->running_var == NULL) {
        if (params->gamma != NULL) tiny_tensor_destroy(params->gamma);
        if (params->beta != NULL) tiny_tensor_destroy(params->beta);
        tiny_tensor_destroy(params->running_mean);
        free(params);
        return NULL;
    }
    
    // Initialize running variance to 1.0
    float* var_data = (float*)tiny_tensor_data(params->running_var);
    for (int i = 0; i < num_channels; i++) {
        var_data[i] = 1.0f;
    }
    
    return params;
}

void tiny_batchnorm_destroy(tiny_batchnorm_params_t* params)
{
    if (params == NULL) {
        return;
    }
    
    if (params->gamma != NULL) {
        tiny_tensor_destroy(params->gamma);
    }
    
    if (params->beta != NULL) {
        tiny_tensor_destroy(params->beta);
    }
    
    if (params->running_mean != NULL) {
        tiny_tensor_destroy(params->running_mean);
    }
    
    if (params->running_var != NULL) {
        tiny_tensor_destroy(params->running_var);
    }
    
    free(params);
}

/* ============================================================================
 * FORWARD AND BACKWARD PROPAGATION
 * ============================================================================ */

void tiny_batchnorm_forward(tiny_graph_node_t* node)
{
    if (node == NULL || node->params == NULL) {
        return;
    }
    
    tiny_batchnorm_params_t* params = (tiny_batchnorm_params_t*)node->params;
    
    // Get input and output tensors
    if (node->inputs == NULL || node->inputs[0] == NULL ||
        node->outputs == NULL || node->outputs[0] == NULL) {
        return;
    }
    
    tiny_tensor_t* input = node->inputs[0];
    tiny_tensor_t* output = node->outputs[0];
    
    // Get data pointers
    float* input_data = (float*)tiny_tensor_data(input);
    float* output_data = (float*)tiny_tensor_data(output);
    
    if (input_data == NULL || output_data == NULL) {
        return;
    }
    
    // Get graph to check training mode
    // Note: We need to get the graph from the node, but the node doesn't have a direct pointer
    // For now, we'll assume training mode if gradients are enabled
    bool training_mode = true;
#if TINY_AI_ENABLE_GRADIENTS
    // Check if output requires grad (indicator of training mode)
    training_mode = tiny_tensor_get_requires_grad(output);
#endif
    
    // Get dimensions
    int input_ndim = tiny_tensor_ndim(input);
    if (input_ndim < 2) {
        return;  // Invalid input shape
    }
    
    int batch_size = tiny_tensor_shape(input, 0);
    int channels = tiny_tensor_shape(input, 1);
    
    if (channels != params->num_channels) {
        return;
    }
    
    // Compute spatial dimensions
    int spatial_size = 1;
    for (int i = 2; i < input_ndim; i++) {
        spatial_size *= tiny_tensor_shape(input, i);
    }
    
    int num_elements_per_channel = batch_size * spatial_size;
    
    // Get gamma and beta
    float* gamma_data = NULL;
    float* beta_data = NULL;
    if (params->affine) {
        if (params->gamma != NULL) {
            gamma_data = (float*)tiny_tensor_data(params->gamma);
        }
        if (params->beta != NULL) {
            beta_data = (float*)tiny_tensor_data(params->beta);
        }
    }
    
    // Get running statistics
    float* running_mean_data = (float*)tiny_tensor_data(params->running_mean);
    float* running_var_data = (float*)tiny_tensor_data(params->running_var);
    
    if (running_mean_data == NULL || running_var_data == NULL) {
        return;
    }
    
    if (training_mode) {
        // Training mode: compute batch statistics
        
        // Compute mean for each channel
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                for (int s = 0; s < spatial_size; s++) {
                    int idx = b * (channels * spatial_size) + c * spatial_size + s;
                    sum += input_data[idx];
                }
            }
            float mean = sum / num_elements_per_channel;
            
            // Compute variance for each channel
            float sum_var = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                for (int s = 0; s < spatial_size; s++) {
                    int idx = b * (channels * spatial_size) + c * spatial_size + s;
                    float diff = input_data[idx] - mean;
                    sum_var += diff * diff;
                }
            }
            float var = sum_var / num_elements_per_channel;
            
            // Update running statistics
            running_mean_data[c] = params->momentum * running_mean_data[c] + (1.0f - params->momentum) * mean;
            running_var_data[c] = params->momentum * running_var_data[c] + (1.0f - params->momentum) * var;
            
            // Normalize: (x - mean) / sqrt(var + eps)
            float inv_std = 1.0f / sqrtf(var + params->eps);
            float gamma_val = (gamma_data != NULL) ? gamma_data[c] : 1.0f;
            float beta_val = (beta_data != NULL) ? beta_data[c] : 0.0f;
            
            for (int b = 0; b < batch_size; b++) {
                for (int s = 0; s < spatial_size; s++) {
                    int idx = b * (channels * spatial_size) + c * spatial_size + s;
                    float normalized = (input_data[idx] - mean) * inv_std;
                    output_data[idx] = gamma_val * normalized + beta_val;
                }
            }
        }
    } else {
        // Inference mode: use running statistics
        for (int c = 0; c < channels; c++) {
            float mean = running_mean_data[c];
            float var = running_var_data[c];
            float inv_std = 1.0f / sqrtf(var + params->eps);
            float gamma_val = (gamma_data != NULL) ? gamma_data[c] : 1.0f;
            float beta_val = (beta_data != NULL) ? beta_data[c] : 0.0f;
            
            for (int b = 0; b < batch_size; b++) {
                for (int s = 0; s < spatial_size; s++) {
                    int idx = b * (channels * spatial_size) + c * spatial_size + s;
                    float normalized = (input_data[idx] - mean) * inv_std;
                    output_data[idx] = gamma_val * normalized + beta_val;
                }
            }
        }
    }
}

void tiny_batchnorm_backward(tiny_graph_node_t* node)
{
    if (node == NULL || node->params == NULL) {
        return;
    }
    
    tiny_batchnorm_params_t* params = (tiny_batchnorm_params_t*)node->params;
    
    // Get input, output, and gradient tensors
    if (node->inputs == NULL || node->inputs[0] == NULL ||
        node->outputs == NULL || node->outputs[0] == NULL) {
        return;
    }
    
    tiny_tensor_t* input = node->inputs[0];
    tiny_tensor_t* output = node->outputs[0];
    
    // Get data pointers
    float* input_data = (float*)tiny_tensor_data(input);
    float* output_data = (float*)tiny_tensor_data(output);
    
    if (input_data == NULL || output_data == NULL) {
        return;
    }
    
    // Get output gradient
    float* output_grad_data = NULL;
#if TINY_AI_ENABLE_GRADIENTS
    if (tiny_tensor_get_requires_grad(output)) {
        output_grad_data = (float*)tiny_tensor_grad(output);
    }
#endif
    
    if (output_grad_data == NULL) {
        return;  // No gradient to propagate
    }
    
    // Get dimensions
    int input_ndim = tiny_tensor_ndim(input);
    if (input_ndim < 2) {
        return;
    }
    
    int batch_size = tiny_tensor_shape(input, 0);
    int channels = tiny_tensor_shape(input, 1);
    
    // Compute spatial dimensions
    int spatial_size = 1;
    for (int i = 2; i < input_ndim; i++) {
        spatial_size *= tiny_tensor_shape(input, i);
    }
    
    int num_elements_per_channel = batch_size * spatial_size;
    
    // Compute input gradient
    float* input_grad_data = NULL;
#if TINY_AI_ENABLE_GRADIENTS
    if (tiny_tensor_get_requires_grad(input)) {
        input_grad_data = (float*)tiny_tensor_grad(input);
        if (input_grad_data != NULL) {
            // Zero out input gradient (will accumulate)
            int input_size = batch_size * channels * spatial_size;
            memset(input_grad_data, 0, input_size * sizeof(float));
        }
    }
#endif
    
    // Compute gamma and beta gradients
    float* gamma_grad_data = NULL;
    float* beta_grad_data = NULL;
    if (params->affine) {
#if TINY_AI_ENABLE_GRADIENTS
        if (params->gamma != NULL && tiny_tensor_get_requires_grad(params->gamma)) {
            gamma_grad_data = (float*)tiny_tensor_grad(params->gamma);
            if (gamma_grad_data != NULL) {
                memset(gamma_grad_data, 0, channels * sizeof(float));
            }
        }
        if (params->beta != NULL && tiny_tensor_get_requires_grad(params->beta)) {
            beta_grad_data = (float*)tiny_tensor_grad(params->beta);
            if (beta_grad_data != NULL) {
                memset(beta_grad_data, 0, channels * sizeof(float));
            }
        }
#endif
    }
    
    // Get gamma
    float* gamma_data = NULL;
    if (params->affine && params->gamma != NULL) {
        gamma_data = (float*)tiny_tensor_data(params->gamma);
    }
    
    // Recompute batch statistics (needed for backward)
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            for (int s = 0; s < spatial_size; s++) {
                int idx = b * (channels * spatial_size) + c * spatial_size + s;
                sum += input_data[idx];
            }
        }
        float mean = sum / num_elements_per_channel;
        
        float sum_var = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            for (int s = 0; s < spatial_size; s++) {
                int idx = b * (channels * spatial_size) + c * spatial_size + s;
                float diff = input_data[idx] - mean;
                sum_var += diff * diff;
            }
        }
        float var = sum_var / num_elements_per_channel;
        float inv_std = 1.0f / sqrtf(var + params->eps);
        float gamma_val = (gamma_data != NULL) ? gamma_data[c] : 1.0f;
        
        // Compute gradients
        float sum_grad = 0.0f;
        float sum_grad_x_centered = 0.0f;
        
        for (int b = 0; b < batch_size; b++) {
            for (int s = 0; s < spatial_size; s++) {
                int idx = b * (channels * spatial_size) + c * spatial_size + s;
                float grad_out = output_grad_data[idx];
                float x_centered = (input_data[idx] - mean) * inv_std;
                
                sum_grad += grad_out;
                sum_grad_x_centered += grad_out * x_centered;
                
                // Beta gradient
                if (beta_grad_data != NULL) {
                    beta_grad_data[c] += grad_out;
                }
                
                // Gamma gradient
                if (gamma_grad_data != NULL) {
                    gamma_grad_data[c] += grad_out * x_centered;
                }
            }
        }
        
        // Input gradient
        if (input_grad_data != NULL) {
            float grad_mean = sum_grad / num_elements_per_channel;
            float grad_var = sum_grad_x_centered / num_elements_per_channel;
            
            for (int b = 0; b < batch_size; b++) {
                for (int s = 0; s < spatial_size; s++) {
                    int idx = b * (channels * spatial_size) + c * spatial_size + s;
                    float x_centered = (input_data[idx] - mean) * inv_std;
                    float grad_out = output_grad_data[idx];
                    
                    float grad_input = gamma_val * inv_std * (
                        grad_out - grad_mean - x_centered * grad_var
                    );
                    
                    input_grad_data[idx] += grad_input;
                }
            }
        }
    }
}

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

tiny_error_t tiny_batchnorm_reset_running_stats(tiny_batchnorm_params_t* params)
{
    if (params == NULL || params->running_mean == NULL || params->running_var == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    // Reset running mean to 0.0
    tiny_tensor_zero(params->running_mean);
    
    // Reset running variance to 1.0
    float* var_data = (float*)tiny_tensor_data(params->running_var);
    for (int i = 0; i < params->num_channels; i++) {
        var_data[i] = 1.0f;
    }
    
    return TINY_OK;
}

