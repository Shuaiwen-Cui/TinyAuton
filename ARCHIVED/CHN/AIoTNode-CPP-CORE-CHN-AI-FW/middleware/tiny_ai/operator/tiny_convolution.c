/**
 * @file tiny_convolution.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Convolution Layer implementation
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 */

#include "tiny_ai_config.h"  // Include config first to ensure macros are defined
#include "tiny_tensor.h"      // Include tensor.h to get function declarations
#include "tiny_convolution.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ============================================================================
 * LAYER CREATION AND DESTRUCTION
 * ============================================================================ */

tiny_convolution_params_t* tiny_convolution_create_1d(int in_channels, int out_channels,
                                                      int kernel_w, int stride_w, int pad_w,
                                                      bool use_bias)
{
    if (in_channels <= 0 || out_channels <= 0 || 
        kernel_w <= 0 || stride_w <= 0 || pad_w < 0) {
        return NULL;
    }
    
    // Allocate parameters structure
    tiny_convolution_params_t* params = (tiny_convolution_params_t*)malloc(sizeof(tiny_convolution_params_t));
    if (params == NULL) {
        return NULL;
    }
    
    memset(params, 0, sizeof(tiny_convolution_params_t));
    params->spatial_dims = 1;
    params->in_channels = in_channels;
    params->out_channels = out_channels;
    params->kernel_h = 1;  // Unused for 1D
    params->kernel_w = kernel_w;
    params->stride_h = 1;  // Unused for 1D
    params->stride_w = stride_w;
    params->pad_h = 0;     // Unused for 1D
    params->pad_w = pad_w;
    params->kernel_d = 1;  // Unused for 1D
    params->stride_d = 1;  // Unused for 1D
    params->pad_d = 0;     // Unused for 1D
    params->use_bias = use_bias;
    
    // Create weight tensor [out_channels, in_channels, kernel_w]
    int weight_shape[] = {out_channels, in_channels, kernel_w};
    params->weights = tiny_tensor_create(weight_shape, 3, TINY_AI_DTYPE_FLOAT32);
    if (params->weights == NULL) {
        free(params);
        return NULL;
    }
    
    // Enable gradients for weights
#if TINY_AI_ENABLE_GRADIENTS
    tiny_tensor_requires_grad(params->weights, true);
#endif
    
    // Create bias tensor [out_channels] if needed
    if (use_bias) {
        int bias_shape[] = {out_channels};
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

tiny_convolution_params_t* tiny_convolution_create_2d(int in_channels, int out_channels,
                                                      int kernel_h, int kernel_w,
                                                      int stride_h, int stride_w,
                                                      int pad_h, int pad_w,
                                                      bool use_bias)
{
    if (in_channels <= 0 || out_channels <= 0 || 
        kernel_h <= 0 || kernel_w <= 0 ||
        stride_h <= 0 || stride_w <= 0 ||
        pad_h < 0 || pad_w < 0) {
        return NULL;
    }
    
    // Allocate parameters structure
    tiny_convolution_params_t* params = (tiny_convolution_params_t*)malloc(sizeof(tiny_convolution_params_t));
    if (params == NULL) {
        return NULL;
    }
    
    memset(params, 0, sizeof(tiny_convolution_params_t));
    params->spatial_dims = 2;
    params->in_channels = in_channels;
    params->out_channels = out_channels;
    params->kernel_h = kernel_h;
    params->kernel_w = kernel_w;
    params->stride_h = stride_h;
    params->stride_w = stride_w;
    params->pad_h = pad_h;
    params->pad_w = pad_w;
    params->kernel_d = 1;  // Unused for 2D
    params->stride_d = 1;  // Unused for 2D
    params->pad_d = 0;     // Unused for 2D
    params->use_bias = use_bias;
    
    // Create weight tensor [out_channels, in_channels, kernel_h, kernel_w]
    int weight_shape[] = {out_channels, in_channels, kernel_h, kernel_w};
    params->weights = tiny_tensor_create(weight_shape, 4, TINY_AI_DTYPE_FLOAT32);
    if (params->weights == NULL) {
        free(params);
        return NULL;
    }
    
    // Enable gradients for weights
#if TINY_AI_ENABLE_GRADIENTS
    tiny_tensor_requires_grad(params->weights, true);
#endif
    
    // Create bias tensor [out_channels] if needed
    if (use_bias) {
        int bias_shape[] = {out_channels};
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

tiny_convolution_params_t* tiny_convolution_create_3d(int in_channels, int out_channels,
                                                      int kernel_d, int kernel_h, int kernel_w,
                                                      int stride_d, int stride_h, int stride_w,
                                                      int pad_d, int pad_h, int pad_w,
                                                      bool use_bias)
{
    if (in_channels <= 0 || out_channels <= 0 || 
        kernel_d <= 0 || kernel_h <= 0 || kernel_w <= 0 ||
        stride_d <= 0 || stride_h <= 0 || stride_w <= 0 ||
        pad_d < 0 || pad_h < 0 || pad_w < 0) {
        return NULL;
    }
    
    // Allocate parameters structure
    tiny_convolution_params_t* params = (tiny_convolution_params_t*)malloc(sizeof(tiny_convolution_params_t));
    if (params == NULL) {
        return NULL;
    }
    
    memset(params, 0, sizeof(tiny_convolution_params_t));
    params->spatial_dims = 3;
    params->in_channels = in_channels;
    params->out_channels = out_channels;
    params->kernel_d = kernel_d;
    params->kernel_h = kernel_h;
    params->kernel_w = kernel_w;
    params->stride_d = stride_d;
    params->stride_h = stride_h;
    params->stride_w = stride_w;
    params->pad_d = pad_d;
    params->pad_h = pad_h;
    params->pad_w = pad_w;
    params->use_bias = use_bias;
    
    // Create weight tensor [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
    int weight_shape[] = {out_channels, in_channels, kernel_d, kernel_h, kernel_w};
    params->weights = tiny_tensor_create(weight_shape, 5, TINY_AI_DTYPE_FLOAT32);
    if (params->weights == NULL) {
        free(params);
        return NULL;
    }
    
    // Enable gradients for weights
#if TINY_AI_ENABLE_GRADIENTS
    tiny_tensor_requires_grad(params->weights, true);
#endif
    
    // Create bias tensor [out_channels] if needed
    if (use_bias) {
        int bias_shape[] = {out_channels};
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

void tiny_convolution_destroy(tiny_convolution_params_t* params)
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

void tiny_convolution_forward(tiny_graph_node_t* node)
{
    if (node == NULL || node->params == NULL) {
        return;
    }
    
    tiny_convolution_params_t* params = (tiny_convolution_params_t*)node->params;
    
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
    
    int batch_size = tiny_tensor_shape(input, 0);
    int in_channels = tiny_tensor_shape(input, 1);
    
    // Verify input channels match
    if (in_channels != params->in_channels) {
        return;
    }
    
    // Get bias data
    float* bias_data = NULL;
    if (params->use_bias && params->bias != NULL) {
        bias_data = (float*)tiny_tensor_data(params->bias);
    }
    
    // Branch based on spatial dimensions
    if (params->spatial_dims == 1) {
        // 1D Convolution: [batch, in_channels, length]
        int in_length = tiny_tensor_shape(input, 2);
        int out_length = (in_length + 2 * params->pad_w - params->kernel_w) / params->stride_w + 1;
        
        // Verify output shape
        if (tiny_tensor_ndim(output) != 3 ||
            tiny_tensor_shape(output, 0) != batch_size ||
            tiny_tensor_shape(output, 1) != params->out_channels ||
            tiny_tensor_shape(output, 2) != out_length) {
            return;
        }
        
        // Perform 1D convolution
        for (int b = 0; b < batch_size; b++) {
            float* input_batch = input_data + b * (in_channels * in_length);
            
            for (int c_out = 0; c_out < params->out_channels; c_out++) {
                float* weight_channel = weight_data + c_out * (in_channels * params->kernel_w);
                
                for (int ow = 0; ow < out_length; ow++) {
                    float sum = 0.0f;
                    
                    for (int c_in = 0; c_in < in_channels; c_in++) {
                        int in_w_start = ow * params->stride_w - params->pad_w;
                        
                        for (int kw = 0; kw < params->kernel_w; kw++) {
                            int in_w_pos = in_w_start + kw;
                            
                            if (in_w_pos >= 0 && in_w_pos < in_length) {
                                int input_idx = c_in * in_length + in_w_pos;
                                float input_val = input_batch[input_idx];
                                
                                int weight_idx = c_in * params->kernel_w + kw;
                                float weight_val = weight_channel[weight_idx];
                                
                                sum += input_val * weight_val;
                            }
                        }
                    }
                    
                    if (bias_data != NULL) {
                        sum += bias_data[c_out];
                    }
                    
                    int output_idx = c_out * out_length + ow;
                    output_data[b * (params->out_channels * out_length) + output_idx] = sum;
                }
            }
        }
    } else if (params->spatial_dims == 2) {
        // 2D Convolution: [batch, in_channels, height, width]
        int in_h = tiny_tensor_shape(input, 2);
        int in_w = tiny_tensor_shape(input, 3);
        int out_h = (in_h + 2 * params->pad_h - params->kernel_h) / params->stride_h + 1;
        int out_w = (in_w + 2 * params->pad_w - params->kernel_w) / params->stride_w + 1;
        
        // Verify output shape
        if (tiny_tensor_ndim(output) != 4 ||
            tiny_tensor_shape(output, 0) != batch_size ||
            tiny_tensor_shape(output, 1) != params->out_channels ||
            tiny_tensor_shape(output, 2) != out_h ||
            tiny_tensor_shape(output, 3) != out_w) {
            return;
        }
        
        // Perform 2D convolution
        for (int b = 0; b < batch_size; b++) {
            float* input_batch = input_data + b * (in_channels * in_h * in_w);
            
            for (int c_out = 0; c_out < params->out_channels; c_out++) {
                float* weight_channel = weight_data + c_out * (in_channels * params->kernel_h * params->kernel_w);
                
                for (int oh = 0; oh < out_h; oh++) {
                    for (int ow = 0; ow < out_w; ow++) {
                        float sum = 0.0f;
                        
                        for (int c_in = 0; c_in < in_channels; c_in++) {
                            int in_h_start = oh * params->stride_h - params->pad_h;
                            int in_w_start = ow * params->stride_w - params->pad_w;
                            
                            for (int kh = 0; kh < params->kernel_h; kh++) {
                                for (int kw = 0; kw < params->kernel_w; kw++) {
                                    int in_h_pos = in_h_start + kh;
                                    int in_w_pos = in_w_start + kw;
                                    
                                    if (in_h_pos >= 0 && in_h_pos < in_h && 
                                        in_w_pos >= 0 && in_w_pos < in_w) {
                                        int input_idx = c_in * (in_h * in_w) + in_h_pos * in_w + in_w_pos;
                                        float input_val = input_batch[input_idx];
                                        
                                        int weight_idx = c_in * (params->kernel_h * params->kernel_w) + 
                                                        kh * params->kernel_w + kw;
                                        float weight_val = weight_channel[weight_idx];
                                        
                                        sum += input_val * weight_val;
                                    }
                                }
                            }
                        }
                        
                        if (bias_data != NULL) {
                            sum += bias_data[c_out];
                        }
                        
                        int output_idx = c_out * (out_h * out_w) + oh * out_w + ow;
                        output_data[b * (params->out_channels * out_h * out_w) + output_idx] = sum;
                    }
                }
            }
        }
    } else if (params->spatial_dims == 3) {
        // 3D Convolution: [batch, in_channels, depth, height, width]
        int in_d = tiny_tensor_shape(input, 2);
        int in_h = tiny_tensor_shape(input, 3);
        int in_w = tiny_tensor_shape(input, 4);
        int out_d = (in_d + 2 * params->pad_d - params->kernel_d) / params->stride_d + 1;
        int out_h = (in_h + 2 * params->pad_h - params->kernel_h) / params->stride_h + 1;
        int out_w = (in_w + 2 * params->pad_w - params->kernel_w) / params->stride_w + 1;
        
        // Verify output shape
        if (tiny_tensor_ndim(output) != 5 ||
            tiny_tensor_shape(output, 0) != batch_size ||
            tiny_tensor_shape(output, 1) != params->out_channels ||
            tiny_tensor_shape(output, 2) != out_d ||
            tiny_tensor_shape(output, 3) != out_h ||
            tiny_tensor_shape(output, 4) != out_w) {
            return;
        }
        
        // Perform 3D convolution
        for (int b = 0; b < batch_size; b++) {
            float* input_batch = input_data + b * (in_channels * in_d * in_h * in_w);
            
            for (int c_out = 0; c_out < params->out_channels; c_out++) {
                float* weight_channel = weight_data + c_out * (in_channels * params->kernel_d * params->kernel_h * params->kernel_w);
                
                for (int od = 0; od < out_d; od++) {
                    for (int oh = 0; oh < out_h; oh++) {
                        for (int ow = 0; ow < out_w; ow++) {
                            float sum = 0.0f;
                            
                            for (int c_in = 0; c_in < in_channels; c_in++) {
                                int in_d_start = od * params->stride_d - params->pad_d;
                                int in_h_start = oh * params->stride_h - params->pad_h;
                                int in_w_start = ow * params->stride_w - params->pad_w;
                                
                                for (int kd = 0; kd < params->kernel_d; kd++) {
                                    for (int kh = 0; kh < params->kernel_h; kh++) {
                                        for (int kw = 0; kw < params->kernel_w; kw++) {
                                            int in_d_pos = in_d_start + kd;
                                            int in_h_pos = in_h_start + kh;
                                            int in_w_pos = in_w_start + kw;
                                            
                                            if (in_d_pos >= 0 && in_d_pos < in_d &&
                                                in_h_pos >= 0 && in_h_pos < in_h && 
                                                in_w_pos >= 0 && in_w_pos < in_w) {
                                                int input_idx = c_in * (in_d * in_h * in_w) + 
                                                               in_d_pos * (in_h * in_w) + 
                                                               in_h_pos * in_w + in_w_pos;
                                                float input_val = input_batch[input_idx];
                                                
                                                int weight_idx = c_in * (params->kernel_d * params->kernel_h * params->kernel_w) + 
                                                                kd * (params->kernel_h * params->kernel_w) + 
                                                                kh * params->kernel_w + kw;
                                                float weight_val = weight_channel[weight_idx];
                                                
                                                sum += input_val * weight_val;
                                            }
                                        }
                                    }
                                }
                            }
                            
                            if (bias_data != NULL) {
                                sum += bias_data[c_out];
                            }
                            
                            int output_idx = c_out * (out_d * out_h * out_w) + 
                                           od * (out_h * out_w) + 
                                           oh * out_w + ow;
                            output_data[b * (params->out_channels * out_d * out_h * out_w) + output_idx] = sum;
                        }
                    }
                }
            }
        }
    }
}

void tiny_convolution_backward(tiny_graph_node_t* node)
{
    if (node == NULL || node->params == NULL) {
        return;
    }
    
    tiny_convolution_params_t* params = (tiny_convolution_params_t*)node->params;
    
    // Get input, output, and gradient tensors
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
    
    int batch_size = tiny_tensor_shape(input, 0);
    int in_channels = tiny_tensor_shape(input, 1);
    int out_channels = tiny_tensor_shape(output, 1);
    
    // Compute input gradient
    float* input_grad_data = NULL;
#if TINY_AI_ENABLE_GRADIENTS
    if (tiny_tensor_get_requires_grad(input)) {
        input_grad_data = (float*)tiny_tensor_grad(input);
        if (input_grad_data != NULL) {
            int input_size = tiny_tensor_numel(input);
            memset(input_grad_data, 0, input_size * sizeof(float));
        }
    }
#endif
    
    // Compute weight gradient
    float* weight_grad_data = NULL;
#if TINY_AI_ENABLE_GRADIENTS
    if (tiny_tensor_get_requires_grad(weights)) {
        weight_grad_data = (float*)tiny_tensor_grad(weights);
        if (weight_grad_data != NULL) {
            int weight_size = tiny_tensor_numel(weights);
            memset(weight_grad_data, 0, weight_size * sizeof(float));
        }
    }
#endif
    
    // Compute bias gradient
    float* bias_grad_data = NULL;
    if (params->use_bias && params->bias != NULL) {
#if TINY_AI_ENABLE_GRADIENTS
        if (tiny_tensor_get_requires_grad(params->bias)) {
            bias_grad_data = (float*)tiny_tensor_grad(params->bias);
            if (bias_grad_data != NULL) {
                memset(bias_grad_data, 0, out_channels * sizeof(float));
            }
        }
#endif
    }
    
    // Branch based on spatial dimensions
    if (params->spatial_dims == 1) {
        // 1D Backward
        int in_length = tiny_tensor_shape(input, 2);
        int out_length = tiny_tensor_shape(output, 2);
        
        for (int b = 0; b < batch_size; b++) {
            float* input_batch = input_data + b * (in_channels * in_length);
            float* output_grad_batch = output_grad_data + b * (out_channels * out_length);
            
            // Input gradient
            if (input_grad_data != NULL) {
                float* input_grad_batch = input_grad_data + b * (in_channels * in_length);
                
                for (int c_out = 0; c_out < out_channels; c_out++) {
                    float* weight_channel = weight_data + c_out * (in_channels * params->kernel_w);
                    
                    for (int ow = 0; ow < out_length; ow++) {
                        float grad_val = output_grad_batch[c_out * out_length + ow];
                        int in_w_start = ow * params->stride_w - params->pad_w;
                        
                        for (int c_in = 0; c_in < in_channels; c_in++) {
                            for (int kw = 0; kw < params->kernel_w; kw++) {
                                int in_w_pos = in_w_start + kw;
                                if (in_w_pos >= 0 && in_w_pos < in_length) {
                                    int input_grad_idx = c_in * in_length + in_w_pos;
                                    int weight_idx = c_in * params->kernel_w + kw;
                                    float weight_val = weight_channel[weight_idx];
                                    input_grad_batch[input_grad_idx] += grad_val * weight_val;
                                }
                            }
                        }
                    }
                }
            }
            
            // Weight gradient
            if (weight_grad_data != NULL) {
                for (int c_out = 0; c_out < out_channels; c_out++) {
                    float* weight_grad_channel = weight_grad_data + c_out * (in_channels * params->kernel_w);
                    
                    for (int c_in = 0; c_in < in_channels; c_in++) {
                        for (int kw = 0; kw < params->kernel_w; kw++) {
                            float sum = 0.0f;
                            
                            for (int ow = 0; ow < out_length; ow++) {
                                float grad_val = output_grad_batch[c_out * out_length + ow];
                                int in_w_pos = ow * params->stride_w - params->pad_w + kw;
                                
                                if (in_w_pos >= 0 && in_w_pos < in_length) {
                                    int input_idx = c_in * in_length + in_w_pos;
                                    float input_val = input_batch[input_idx];
                                    sum += grad_val * input_val;
                                }
                            }
                            
                            int weight_grad_idx = c_in * params->kernel_w + kw;
                            weight_grad_channel[weight_grad_idx] += sum;
                        }
                    }
                }
            }
            
            // Bias gradient
            if (bias_grad_data != NULL) {
                for (int c_out = 0; c_out < out_channels; c_out++) {
                    float sum = 0.0f;
                    for (int ow = 0; ow < out_length; ow++) {
                        sum += output_grad_batch[c_out * out_length + ow];
                    }
                    bias_grad_data[c_out] += sum;
                }
            }
        }
    } else if (params->spatial_dims == 2) {
        // 2D Backward (existing implementation)
        int in_h = tiny_tensor_shape(input, 2);
        int in_w = tiny_tensor_shape(input, 3);
        int out_h = tiny_tensor_shape(output, 2);
        int out_w = tiny_tensor_shape(output, 3);
        
        for (int b = 0; b < batch_size; b++) {
            float* input_batch = input_data + b * (in_channels * in_h * in_w);
            float* output_grad_batch = output_grad_data + b * (out_channels * out_h * out_w);
            
            // Input gradient
            if (input_grad_data != NULL) {
                float* input_grad_batch = input_grad_data + b * (in_channels * in_h * in_w);
                
                for (int c_out = 0; c_out < out_channels; c_out++) {
                    float* weight_channel = weight_data + c_out * (in_channels * params->kernel_h * params->kernel_w);
                    
                    for (int oh = 0; oh < out_h; oh++) {
                        for (int ow = 0; ow < out_w; ow++) {
                            float grad_val = output_grad_batch[c_out * (out_h * out_w) + oh * out_w + ow];
                            int in_h_start = oh * params->stride_h - params->pad_h;
                            int in_w_start = ow * params->stride_w - params->pad_w;
                            
                            for (int c_in = 0; c_in < in_channels; c_in++) {
                                for (int kh = 0; kh < params->kernel_h; kh++) {
                                    for (int kw = 0; kw < params->kernel_w; kw++) {
                                        int in_h_pos = in_h_start + kh;
                                        int in_w_pos = in_w_start + kw;
                                        
                                        if (in_h_pos >= 0 && in_h_pos < in_h && 
                                            in_w_pos >= 0 && in_w_pos < in_w) {
                                            int input_grad_idx = c_in * (in_h * in_w) + in_h_pos * in_w + in_w_pos;
                                            int weight_idx = c_in * (params->kernel_h * params->kernel_w) + 
                                                            kh * params->kernel_w + kw;
                                            float weight_val = weight_channel[weight_idx];
                                            input_grad_batch[input_grad_idx] += grad_val * weight_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // Weight gradient
            if (weight_grad_data != NULL) {
                for (int c_out = 0; c_out < out_channels; c_out++) {
                    float* weight_grad_channel = weight_grad_data + c_out * (in_channels * params->kernel_h * params->kernel_w);
                    
                    for (int c_in = 0; c_in < in_channels; c_in++) {
                        for (int kh = 0; kh < params->kernel_h; kh++) {
                            for (int kw = 0; kw < params->kernel_w; kw++) {
                                float sum = 0.0f;
                                
                                for (int oh = 0; oh < out_h; oh++) {
                                    for (int ow = 0; ow < out_w; ow++) {
                                        float grad_val = output_grad_batch[c_out * (out_h * out_w) + oh * out_w + ow];
                                        int in_h_pos = oh * params->stride_h - params->pad_h + kh;
                                        int in_w_pos = ow * params->stride_w - params->pad_w + kw;
                                        
                                        if (in_h_pos >= 0 && in_h_pos < in_h && 
                                            in_w_pos >= 0 && in_w_pos < in_w) {
                                            int input_idx = c_in * (in_h * in_w) + in_h_pos * in_w + in_w_pos;
                                            float input_val = input_batch[input_idx];
                                            sum += grad_val * input_val;
                                        }
                                    }
                                }
                                
                                int weight_grad_idx = c_in * (params->kernel_h * params->kernel_w) + 
                                                     kh * params->kernel_w + kw;
                                weight_grad_channel[weight_grad_idx] += sum;
                            }
                        }
                    }
                }
            }
            
            // Bias gradient
            if (bias_grad_data != NULL) {
                for (int c_out = 0; c_out < out_channels; c_out++) {
                    float sum = 0.0f;
                    for (int oh = 0; oh < out_h; oh++) {
                        for (int ow = 0; ow < out_w; ow++) {
                            sum += output_grad_batch[c_out * (out_h * out_w) + oh * out_w + ow];
                        }
                    }
                    bias_grad_data[c_out] += sum;
                }
            }
        }
    } else if (params->spatial_dims == 3) {
        // 3D Backward
        int in_d = tiny_tensor_shape(input, 2);
        int in_h = tiny_tensor_shape(input, 3);
        int in_w = tiny_tensor_shape(input, 4);
        int out_d = tiny_tensor_shape(output, 2);
        int out_h = tiny_tensor_shape(output, 3);
        int out_w = tiny_tensor_shape(output, 4);
        
        for (int b = 0; b < batch_size; b++) {
            float* input_batch = input_data + b * (in_channels * in_d * in_h * in_w);
            float* output_grad_batch = output_grad_data + b * (out_channels * out_d * out_h * out_w);
            
            // Input gradient
            if (input_grad_data != NULL) {
                float* input_grad_batch = input_grad_data + b * (in_channels * in_d * in_h * in_w);
                
                for (int c_out = 0; c_out < out_channels; c_out++) {
                    float* weight_channel = weight_data + c_out * (in_channels * params->kernel_d * params->kernel_h * params->kernel_w);
                    
                    for (int od = 0; od < out_d; od++) {
                        for (int oh = 0; oh < out_h; oh++) {
                            for (int ow = 0; ow < out_w; ow++) {
                                float grad_val = output_grad_batch[c_out * (out_d * out_h * out_w) + 
                                                                   od * (out_h * out_w) + 
                                                                   oh * out_w + ow];
                                int in_d_start = od * params->stride_d - params->pad_d;
                                int in_h_start = oh * params->stride_h - params->pad_h;
                                int in_w_start = ow * params->stride_w - params->pad_w;
                                
                                for (int c_in = 0; c_in < in_channels; c_in++) {
                                    for (int kd = 0; kd < params->kernel_d; kd++) {
                                        for (int kh = 0; kh < params->kernel_h; kh++) {
                                            for (int kw = 0; kw < params->kernel_w; kw++) {
                                                int in_d_pos = in_d_start + kd;
                                                int in_h_pos = in_h_start + kh;
                                                int in_w_pos = in_w_start + kw;
                                                
                                                if (in_d_pos >= 0 && in_d_pos < in_d &&
                                                    in_h_pos >= 0 && in_h_pos < in_h && 
                                                    in_w_pos >= 0 && in_w_pos < in_w) {
                                                    int input_grad_idx = c_in * (in_d * in_h * in_w) + 
                                                                       in_d_pos * (in_h * in_w) + 
                                                                       in_h_pos * in_w + in_w_pos;
                                                    int weight_idx = c_in * (params->kernel_d * params->kernel_h * params->kernel_w) + 
                                                                    kd * (params->kernel_h * params->kernel_w) + 
                                                                    kh * params->kernel_w + kw;
                                                    float weight_val = weight_channel[weight_idx];
                                                    input_grad_batch[input_grad_idx] += grad_val * weight_val;
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
            
            // Weight gradient
            if (weight_grad_data != NULL) {
                for (int c_out = 0; c_out < out_channels; c_out++) {
                    float* weight_grad_channel = weight_grad_data + c_out * (in_channels * params->kernel_d * params->kernel_h * params->kernel_w);
                    
                    for (int c_in = 0; c_in < in_channels; c_in++) {
                        for (int kd = 0; kd < params->kernel_d; kd++) {
                            for (int kh = 0; kh < params->kernel_h; kh++) {
                                for (int kw = 0; kw < params->kernel_w; kw++) {
                                    float sum = 0.0f;
                                    
                                    for (int od = 0; od < out_d; od++) {
                                        for (int oh = 0; oh < out_h; oh++) {
                                            for (int ow = 0; ow < out_w; ow++) {
                                                float grad_val = output_grad_batch[c_out * (out_d * out_h * out_w) + 
                                                                                   od * (out_h * out_w) + 
                                                                                   oh * out_w + ow];
                                                int in_d_pos = od * params->stride_d - params->pad_d + kd;
                                                int in_h_pos = oh * params->stride_h - params->pad_h + kh;
                                                int in_w_pos = ow * params->stride_w - params->pad_w + kw;
                                                
                                                if (in_d_pos >= 0 && in_d_pos < in_d &&
                                                    in_h_pos >= 0 && in_h_pos < in_h && 
                                                    in_w_pos >= 0 && in_w_pos < in_w) {
                                                    int input_idx = c_in * (in_d * in_h * in_w) + 
                                                                   in_d_pos * (in_h * in_w) + 
                                                                   in_h_pos * in_w + in_w_pos;
                                                    float input_val = input_batch[input_idx];
                                                    sum += grad_val * input_val;
                                                }
                                            }
                                        }
                                    }
                                    
                                    int weight_grad_idx = c_in * (params->kernel_d * params->kernel_h * params->kernel_w) + 
                                                         kd * (params->kernel_h * params->kernel_w) + 
                                                         kh * params->kernel_w + kw;
                                    weight_grad_channel[weight_grad_idx] += sum;
                                }
                            }
                        }
                    }
                }
            }
            
            // Bias gradient
            if (bias_grad_data != NULL) {
                for (int c_out = 0; c_out < out_channels; c_out++) {
                    float sum = 0.0f;
                    for (int od = 0; od < out_d; od++) {
                        for (int oh = 0; oh < out_h; oh++) {
                            for (int ow = 0; ow < out_w; ow++) {
                                sum += output_grad_batch[c_out * (out_d * out_h * out_w) + 
                                                        od * (out_h * out_w) + 
                                                        oh * out_w + ow];
                            }
                        }
                    }
                    bias_grad_data[c_out] += sum;
                }
            }
        }
    }
}

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

tiny_error_t tiny_convolution_init_weights_xavier(tiny_convolution_params_t* params, unsigned int seed)
{
    if (params == NULL || params->weights == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    int in_features, out_features;
    
    // Calculate features based on spatial dimensions
    if (params->spatial_dims == 1) {
        in_features = params->in_channels * params->kernel_w;
        out_features = params->out_channels * params->kernel_w;
    } else if (params->spatial_dims == 2) {
        in_features = params->in_channels * params->kernel_h * params->kernel_w;
        out_features = params->out_channels * params->kernel_h * params->kernel_w;
    } else if (params->spatial_dims == 3) {
        in_features = params->in_channels * params->kernel_d * params->kernel_h * params->kernel_w;
        out_features = params->out_channels * params->kernel_d * params->kernel_h * params->kernel_w;
    } else {
        return TINY_ERR_AI_INVALID_ARG;
    }
    
    if (in_features <= 0 || out_features <= 0) {
        return TINY_ERR_AI_INVALID_ARG;
    }
    
    // Xavier/Glorot initialization: uniform[-sqrt(6/(in+out)), sqrt(6/(in+out))]
    float limit = sqrtf(6.0f / (in_features + out_features));
    
    // Simple LCG for random number generation
    unsigned int state = seed;
    if (seed == 0) {
        state = 12345;  // Default seed
    }
    
    float* weight_data = (float*)tiny_tensor_data(params->weights);
    int weight_size = tiny_tensor_numel(params->weights);
    
    for (int i = 0; i < weight_size; i++) {
        // LCG: state = (a * state + c) mod m
        state = (1103515245u * state + 12345u) & 0x7FFFFFFFu;
        float r = (float)state / 2147483648.0f;  // Normalize to [0, 1)
        weight_data[i] = (r * 2.0f - 1.0f) * limit;  // Scale to [-limit, limit]
    }
    
    return TINY_OK;
}

tiny_error_t tiny_convolution_init_weights_zero(tiny_convolution_params_t* params)
{
    if (params == NULL || params->weights == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    tiny_tensor_zero(params->weights);
    
    if (params->use_bias && params->bias != NULL) {
        tiny_tensor_zero(params->bias);
    }
    
    return TINY_OK;
}

