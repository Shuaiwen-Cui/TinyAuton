/**
 * @file tiny_pool.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Pooling Layer implementation
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 */

#include "tiny_pool.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/* ============================================================================
 * LAYER CREATION AND DESTRUCTION
 * ============================================================================ */

tiny_pool_params_t* tiny_pool_create(tiny_pool_type_t pool_type,
                                     int kernel_h, int kernel_w,
                                     int stride_h, int stride_w,
                                     int pad_h, int pad_w)
{
    if (kernel_h <= 0 || kernel_w <= 0 ||
        stride_h <= 0 || stride_w <= 0 ||
        pad_h < 0 || pad_w < 0) {
        return NULL;
    }
    
    // Allocate parameters structure
    tiny_pool_params_t* params = (tiny_pool_params_t*)malloc(sizeof(tiny_pool_params_t));
    if (params == NULL) {
        return NULL;
    }
    
    params->pool_type = pool_type;
    params->kernel_h = kernel_h;
    params->kernel_w = kernel_w;
    params->stride_h = stride_h;
    params->stride_w = stride_w;
    params->pad_h = pad_h;
    params->pad_w = pad_w;
    
    return params;
}

void tiny_pool_destroy(tiny_pool_params_t* params)
{
    if (params == NULL) {
        return;
    }
    
    free(params);
}

/* ============================================================================
 * FORWARD AND BACKWARD PROPAGATION
 * ============================================================================ */

void tiny_pool_forward(tiny_graph_node_t* node)
{
    if (node == NULL || node->params == NULL) {
        return;
    }
    
    tiny_pool_params_t* params = (tiny_pool_params_t*)node->params;
    
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
    
    // Get dimensions
    // Input shape: [batch, channels, in_h, in_w]
    int input_ndim = tiny_tensor_ndim(input);
    if (input_ndim != 4) {
        return;  // Invalid input shape
    }
    
    int batch_size = tiny_tensor_shape(input, 0);
    int channels = tiny_tensor_shape(input, 1);
    int in_h = tiny_tensor_shape(input, 2);
    int in_w = tiny_tensor_shape(input, 3);
    
    // Compute output dimensions
    // out_h = (in_h + 2*pad_h - kernel_h) / stride_h + 1
    // out_w = (in_w + 2*pad_w - kernel_w) / stride_w + 1
    int out_h = (in_h + 2 * params->pad_h - params->kernel_h) / params->stride_h + 1;
    int out_w = (in_w + 2 * params->pad_w - params->kernel_w) / params->stride_w + 1;
    
    // Verify output shape matches
    int output_ndim = tiny_tensor_ndim(output);
    if (output_ndim != 4) {
        return;
    }
    
    int out_batch = tiny_tensor_shape(output, 0);
    int out_channels = tiny_tensor_shape(output, 1);
    int out_h_actual = tiny_tensor_shape(output, 2);
    int out_w_actual = tiny_tensor_shape(output, 3);
    
    if (out_batch != batch_size || out_channels != channels ||
        out_h_actual != out_h || out_w_actual != out_w) {
        return;
    }
    
    // Perform pooling for each sample in batch
    for (int b = 0; b < batch_size; b++) {
        // Get input slice for this batch
        float* input_batch = input_data + b * (channels * in_h * in_w);
        
        // For each channel
        for (int c = 0; c < channels; c++) {
            float* input_channel = input_batch + c * (in_h * in_w);
            
            // For each output position
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float result;
                    
                    if (params->pool_type == TINY_POOL_MAX) {
                        // Max pooling: find maximum value in kernel window
                        float max_val = -FLT_MAX;
                        
                        int in_h_start = oh * params->stride_h - params->pad_h;
                        int in_w_start = ow * params->stride_w - params->pad_w;
                        
                        for (int kh = 0; kh < params->kernel_h; kh++) {
                            for (int kw = 0; kw < params->kernel_w; kw++) {
                                int in_h_pos = in_h_start + kh;
                                int in_w_pos = in_w_start + kw;
                                
                                // Check bounds (handle padding)
                                if (in_h_pos >= 0 && in_h_pos < in_h && 
                                    in_w_pos >= 0 && in_w_pos < in_w) {
                                    int input_idx = in_h_pos * in_w + in_w_pos;
                                    float val = input_channel[input_idx];
                                    if (val > max_val) {
                                        max_val = val;
                                    }
                                }
                                // If out of bounds, treat as -infinity (for max pooling)
                            }
                        }
                        
                        result = max_val;
                    } else {
                        // Average pooling: compute average value in kernel window
                        float sum = 0.0f;
                        int count = 0;
                        
                        int in_h_start = oh * params->stride_h - params->pad_h;
                        int in_w_start = ow * params->stride_w - params->pad_w;
                        
                        for (int kh = 0; kh < params->kernel_h; kh++) {
                            for (int kw = 0; kw < params->kernel_w; kw++) {
                                int in_h_pos = in_h_start + kh;
                                int in_w_pos = in_w_start + kw;
                                
                                // Check bounds (handle padding)
                                if (in_h_pos >= 0 && in_h_pos < in_h && 
                                    in_w_pos >= 0 && in_w_pos < in_w) {
                                    int input_idx = in_h_pos * in_w + in_w_pos;
                                    sum += input_channel[input_idx];
                                    count++;
                                }
                                // If out of bounds, don't count (zero padding)
                            }
                        }
                        
                        result = (count > 0) ? (sum / count) : 0.0f;
                    }
                    
                    // Store output
                    int output_idx = c * (out_h * out_w) + oh * out_w + ow;
                    output_data[b * (channels * out_h * out_w) + output_idx] = result;
                }
            }
        }
    }
}

void tiny_pool_backward(tiny_graph_node_t* node)
{
    if (node == NULL || node->params == NULL) {
        return;
    }
    
    tiny_pool_params_t* params = (tiny_pool_params_t*)node->params;
    
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
    int batch_size = tiny_tensor_shape(input, 0);
    int channels = tiny_tensor_shape(input, 1);
    int in_h = tiny_tensor_shape(input, 2);
    int in_w = tiny_tensor_shape(input, 3);
    
    int out_h = tiny_tensor_shape(output, 2);
    int out_w = tiny_tensor_shape(output, 3);
    
    // Compute input gradient
    float* input_grad_data = NULL;
#if TINY_AI_ENABLE_GRADIENTS
    if (tiny_tensor_get_requires_grad(input)) {
        input_grad_data = (float*)tiny_tensor_grad(input);
        if (input_grad_data != NULL) {
            // Zero out input gradient (will accumulate)
            int input_size = batch_size * channels * in_h * in_w;
            memset(input_grad_data, 0, input_size * sizeof(float));
        }
    }
#endif
    
    if (input_grad_data == NULL) {
        return;  // No gradient to compute
    }
    
    // Backward propagation
    for (int b = 0; b < batch_size; b++) {
        float* input_batch = input_data + b * (channels * in_h * in_w);
        float* input_grad_batch = input_grad_data + b * (channels * in_h * in_w);
        float* output_grad_batch = output_grad_data + b * (channels * out_h * out_w);
        
        // For each channel
        for (int c = 0; c < channels; c++) {
            float* input_channel = input_batch + c * (in_h * in_w);
            float* input_grad_channel = input_grad_batch + c * (in_h * in_w);
            float* output_grad_channel = output_grad_batch + c * (out_h * out_w);
            
            // For each output position
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float grad_val = output_grad_channel[oh * out_w + ow];
                    
                    int in_h_start = oh * params->stride_h - params->pad_h;
                    int in_w_start = ow * params->stride_w - params->pad_w;
                    
                    if (params->pool_type == TINY_POOL_MAX) {
                        // Max pooling: gradient flows only to the maximum element
                        float max_val = -FLT_MAX;
                        int max_h = -1, max_w = -1;
                        
                        for (int kh = 0; kh < params->kernel_h; kh++) {
                            for (int kw = 0; kw < params->kernel_w; kw++) {
                                int in_h_pos = in_h_start + kh;
                                int in_w_pos = in_w_start + kw;
                                
                                if (in_h_pos >= 0 && in_h_pos < in_h && 
                                    in_w_pos >= 0 && in_w_pos < in_w) {
                                    int input_idx = in_h_pos * in_w + in_w_pos;
                                    float val = input_channel[input_idx];
                                    if (val > max_val) {
                                        max_val = val;
                                        max_h = in_h_pos;
                                        max_w = in_w_pos;
                                    }
                                }
                            }
                        }
                        
                        // Add gradient to the maximum element
                        if (max_h >= 0 && max_w >= 0) {
                            int input_grad_idx = max_h * in_w + max_w;
                            input_grad_channel[input_grad_idx] += grad_val;
                        }
                    } else {
                        // Average pooling: gradient is distributed equally to all elements
                        int count = 0;
                        
                        // First pass: count valid elements
                        for (int kh = 0; kh < params->kernel_h; kh++) {
                            for (int kw = 0; kw < params->kernel_w; kw++) {
                                int in_h_pos = in_h_start + kh;
                                int in_w_pos = in_w_start + kw;
                                
                                if (in_h_pos >= 0 && in_h_pos < in_h && 
                                    in_w_pos >= 0 && in_w_pos < in_w) {
                                    count++;
                                }
                            }
                        }
                        
                        // Second pass: distribute gradient
                        if (count > 0) {
                            float grad_per_element = grad_val / count;
                            
                            for (int kh = 0; kh < params->kernel_h; kh++) {
                                for (int kw = 0; kw < params->kernel_w; kw++) {
                                    int in_h_pos = in_h_start + kh;
                                    int in_w_pos = in_w_start + kw;
                                    
                                    if (in_h_pos >= 0 && in_h_pos < in_h && 
                                        in_w_pos >= 0 && in_w_pos < in_w) {
                                        int input_grad_idx = in_h_pos * in_w + in_w_pos;
                                        input_grad_channel[input_grad_idx] += grad_per_element;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

