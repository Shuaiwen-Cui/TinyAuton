/**
 * @file tiny_fc.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Fully Connected Layer implementation
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 */

#include "tiny_fc.h"
#include "tiny_mat.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ============================================================================
 * LAYER CREATION AND DESTRUCTION
 * ============================================================================ */

tiny_fc_params_t* tiny_fc_create(int in_features, int out_features, bool use_bias)
{
    if (in_features <= 0 || out_features <= 0) {
        return NULL;
    }
    
    // Allocate parameters structure
    tiny_fc_params_t* params = (tiny_fc_params_t*)malloc(sizeof(tiny_fc_params_t));
    if (params == NULL) {
        return NULL;
    }
    
    memset(params, 0, sizeof(tiny_fc_params_t));
    params->in_features = in_features;
    params->out_features = out_features;
    params->use_bias = use_bias;
    
    // Create weight tensor [out_features, in_features]
    int weight_shape[] = {out_features, in_features};
    params->weights = tiny_tensor_create(weight_shape, 2, TINY_AI_DTYPE_FLOAT32);
    if (params->weights == NULL) {
        free(params);
        return NULL;
    }
    
    // Enable gradients for weights
#if TINY_AI_ENABLE_GRADIENTS
    tiny_tensor_requires_grad(params->weights, true);
#endif
    
    // Create bias tensor [out_features] if needed
    if (use_bias) {
        int bias_shape[] = {out_features};
        params->bias = tiny_tensor_create(bias_shape, 1, TINY_AI_DTYPE_FLOAT32);
        if (params->bias == NULL) {
            tiny_tensor_destroy(params->weights);
            free(params);
            return NULL;
        }
        
        // Enable gradients for bias
#if TINY_AI_ENABLE_GRADIENTS
        tiny_tensor_requires_grad(params->bias, true);
#endif
    } else {
        params->bias = NULL;
    }
    
    return params;
}

void tiny_fc_destroy(tiny_fc_params_t* params)
{
    if (params == NULL) {
        return;
    }
    
    if (params->weights != NULL) {
        tiny_tensor_destroy(params->weights);
    }
    
    if (params->bias != NULL) {
        tiny_tensor_destroy(params->bias);
    }
    
    free(params);
}

/* ============================================================================
 * FORWARD AND BACKWARD PROPAGATION
 * ============================================================================ */

void tiny_fc_forward(tiny_graph_node_t* node)
{
    if (node == NULL || node->params == NULL) {
        return;
    }
    
    tiny_fc_params_t* params = (tiny_fc_params_t*)node->params;
    
    // Get input and output tensors
    if (node->inputs == NULL || node->inputs[0] == NULL ||
        node->outputs == NULL || node->outputs[0] == NULL) {
        return;
    }
    
    tiny_tensor_t* input = node->inputs[0];
    tiny_tensor_t* output = node->outputs[0];
    tiny_tensor_t* weights = params->weights;
    
    // Get data pointers
    float* input_data = (float*)tiny_tensor_data(input);
    float* output_data = (float*)tiny_tensor_data(output);
    float* weight_data = (float*)tiny_tensor_data(weights);
    
    if (input_data == NULL || output_data == NULL || weight_data == NULL) {
        return;
    }
    
    // Get dimensions
    // Input shape: [batch, in_features] or [in_features] (batch=1)
    int batch_size = 1;
    int input_ndim = tiny_tensor_ndim(input);
    if (input_ndim == 2) {
        batch_size = tiny_tensor_shape(input, 0);
    }
    
    int in_features = params->in_features;
    int out_features = params->out_features;
    
    // Perform: output = input @ weights^T
    // input: [batch, in_features]
    // weights: [out_features, in_features]
    // weights^T: [in_features, out_features]
    // output: [batch, out_features]
    
    // For each sample in batch
    for (int b = 0; b < batch_size; b++) {
        float* input_row = input_data + b * in_features;
        float* output_row = output_data + b * out_features;
        
        // output_row = input_row @ weights^T
        // This is equivalent to: output_row = weights @ input_row^T
        // We use: C = A @ B where A=input_row (1 x in), B=weights^T (in x out)
        // But tiny_mat_mult_f32 expects: C = A @ B where A is (m x n), B is (n x k)
        // So we need: output (1 x out) = input (1 x in) @ weights^T (in x out)
        // Which is: output = input @ weights^T
        
        // Actually, we can compute: output = weights @ input^T
        // weights: [out_features, in_features]
        // input_row: [1, in_features] (as row vector)
        // We want: output_row = input_row @ weights^T
        // Which is: output_row^T = weights @ input_row^T
        
        // Compute: output_row = input_row @ weights^T
        // This is: output_row[j] = sum_i(input_row[i] * weights[j][i])
        for (int j = 0; j < out_features; j++) {
            float sum = 0.0f;
            for (int i = 0; i < in_features; i++) {
                sum += input_row[i] * weight_data[j * in_features + i];
            }
            output_row[j] = sum;
        }
        
        // Add bias if present
        if (params->use_bias && params->bias != NULL) {
            float* bias_data = (float*)tiny_tensor_data(params->bias);
            if (bias_data != NULL) {
                for (int j = 0; j < out_features; j++) {
                    output_row[j] += bias_data[j];
                }
            }
        }
    }
}

void tiny_fc_backward(tiny_graph_node_t* node)
{
    if (node == NULL || node->params == NULL) {
        return;
    }
    
    tiny_fc_params_t* params = (tiny_fc_params_t*)node->params;
    
    // Get tensors
    if (node->inputs == NULL || node->inputs[0] == NULL ||
        node->outputs == NULL || node->outputs[0] == NULL) {
        return;
    }
    
    tiny_tensor_t* input = node->inputs[0];
    tiny_tensor_t* output = node->outputs[0];
    tiny_tensor_t* weights = params->weights;
    
    // Get data pointers
    float* input_data = (float*)tiny_tensor_data(input);
    float* output_data = (float*)tiny_tensor_data(output);
    float* weight_data = (float*)tiny_tensor_data(weights);
    
    if (input_data == NULL || output_data == NULL || weight_data == NULL) {
        return;
    }
    
    // Get output gradient (from next layer)
    void* output_grad_ptr = tiny_tensor_grad(output);
    if (output_grad_ptr == NULL) {
        return;  // No gradient to propagate
    }
    float* output_grad = (float*)output_grad_ptr;
    
    // Get dimensions
    int batch_size = 1;
    int input_ndim = tiny_tensor_ndim(input);
    if (input_ndim == 2) {
        batch_size = tiny_tensor_shape(input, 0);
    }
    
    int in_features = params->in_features;
    int out_features = params->out_features;
    
    // 1. Compute input gradient: input_grad = output_grad @ weights
    // output_grad: [batch, out_features]
    // weights: [out_features, in_features]
    // input_grad: [batch, in_features]
    void* input_grad_ptr = tiny_tensor_grad(input);
    if (input_grad_ptr != NULL && tiny_tensor_get_requires_grad(input)) {
        float* input_grad = (float*)input_grad_ptr;
        
        // Initialize input_grad to zero if not already accumulated
        // (In case of multiple outputs, gradients are accumulated)
        // For now, we'll accumulate: input_grad += output_grad @ weights
        
        for (int b = 0; b < batch_size; b++) {
            float* output_grad_row = output_grad + b * out_features;
            float* input_grad_row = input_grad + b * in_features;
            
            // input_grad_row = output_grad_row @ weights
            // input_grad_row[i] = sum_j(output_grad_row[j] * weights[j][i])
            for (int i = 0; i < in_features; i++) {
                float sum = 0.0f;
                for (int j = 0; j < out_features; j++) {
                    sum += output_grad_row[j] * weight_data[j * in_features + i];
                }
                input_grad_row[i] += sum;  // Accumulate gradient
            }
        }
    }
    
    // 2. Compute weight gradient: weight_grad = output_grad^T @ input
    // output_grad: [batch, out_features]
    // input: [batch, in_features]
    // weight_grad: [out_features, in_features]
    void* weight_grad_ptr = tiny_tensor_grad(weights);
    if (weight_grad_ptr != NULL && tiny_tensor_get_requires_grad(weights)) {
        float* weight_grad = (float*)weight_grad_ptr;
        
        // Initialize weight_grad to zero if first time
        // Then accumulate: weight_grad += output_grad^T @ input
        
        // For each output feature
        for (int j = 0; j < out_features; j++) {
            // For each input feature
            for (int i = 0; i < in_features; i++) {
                float sum = 0.0f;
                // Sum over batch: weight_grad[j][i] = sum_b(output_grad[b][j] * input[b][i])
                for (int b = 0; b < batch_size; b++) {
                    float* output_grad_row = output_grad + b * out_features;
                    float* input_row = input_data + b * in_features;
                    sum += output_grad_row[j] * input_row[i];
                }
                weight_grad[j * in_features + i] += sum;  // Accumulate gradient
            }
        }
    }
    
    // 3. Compute bias gradient: bias_grad = sum(output_grad, axis=0)
    // output_grad: [batch, out_features]
    // bias_grad: [out_features]
    if (params->use_bias && params->bias != NULL) {
        void* bias_grad_ptr = tiny_tensor_grad(params->bias);
        if (bias_grad_ptr != NULL && tiny_tensor_get_requires_grad(params->bias)) {
            float* bias_grad = (float*)bias_grad_ptr;
            
            // Initialize bias_grad to zero, then accumulate
            // bias_grad[j] = sum_b(output_grad[b][j])
            for (int j = 0; j < out_features; j++) {
                float sum = 0.0f;
                for (int b = 0; b < batch_size; b++) {
                    float* output_grad_row = output_grad + b * out_features;
                    sum += output_grad_row[j];
                }
                bias_grad[j] += sum;  // Accumulate gradient
            }
        }
    }
}

/* ============================================================================
 * UTILITY
 * ============================================================================ */

tiny_error_t tiny_fc_init_weights_xavier(tiny_fc_params_t* params, unsigned int seed)
{
    if (params == NULL || params->weights == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    // Xavier/Glorot initialization: uniform distribution
    // Range: [-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out))]
    int fan_in = params->in_features;
    int fan_out = params->out_features;
    
    float limit = sqrtf(6.0f / (fan_in + fan_out));
    
    // Simple random number generator (linear congruential generator)
    // If seed is 0, use a default seed
    unsigned int rng_state = (seed == 0) ? 12345 : seed;
    
    float* weight_data = (float*)tiny_tensor_data(params->weights);
    if (weight_data == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    int numel = tiny_tensor_numel(params->weights);
    
    // Initialize weights with uniform distribution
    for (int i = 0; i < numel; i++) {
        // Generate random number in [0, 1)
        rng_state = rng_state * 1103515245 + 12345;
        float random_val = ((float)(rng_state >> 16) / 65536.0f);
        
        // Map to [-limit, limit]
        weight_data[i] = (random_val * 2.0f - 1.0f) * limit;
    }
    
    return TINY_OK;
}

tiny_error_t tiny_fc_init_weights_zero(tiny_fc_params_t* params)
{
    if (params == NULL || params->weights == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    return tiny_tensor_zero(params->weights);
}

