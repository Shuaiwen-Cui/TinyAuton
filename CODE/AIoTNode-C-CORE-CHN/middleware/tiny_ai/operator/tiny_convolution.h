/**
 * @file tiny_convolution.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Convolution Layer for neural networks (1D, 2D, 3D)
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 *
 * @details
 * This module provides 1D, 2D, and 3D convolution layer functionality for neural networks.
 * A convolution layer performs: output = conv(input, weights) + bias
 * 
 * Features:
 * - Forward propagation: 1D/2D/3D convolution + bias addition
 * - Backward propagation: gradient computation for input, weights, and bias
 * - Support for stride and padding
 * - Integration with computation graph
 * - Support for training and inference modes
 * 
 * Design:
 * - Lightweight implementation for MCU
 * - Memory-efficient parameter storage
 * - Supports multi-channel input/output
 * - Supports 1D (temporal), 2D (spatial), and 3D (volumetric) convolutions
 */

#pragma once

#include "tiny_ai_config.h"
#include "tiny_tensor.h"
#include "tiny_graph.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* ============================================================================
 * TYPE DEFINITIONS
 * ============================================================================ */

/**
 * @brief Convolution Layer parameters structure
 * 
 * @details
 * Supports 1D, 2D, and 3D convolutions:
 * - 1D: spatial_dims=1, input shape [batch, in_channels, length]
 *        weights shape [out_channels, in_channels, kernel_w]
 *        uses kernel_w, stride_w, pad_w (kernel_h=1, stride_h=1, pad_h=0)
 * - 2D: spatial_dims=2, input shape [batch, in_channels, height, width]
 *        weights shape [out_channels, in_channels, kernel_h, kernel_w]
 *        uses kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w
 * - 3D: spatial_dims=3, input shape [batch, in_channels, depth, height, width]
 *        weights shape [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
 *        uses all kernel, stride, and pad parameters
 */
typedef struct tiny_convolution_params_t
{
    tiny_tensor_t* weights;      // Weight tensor (shape depends on spatial_dims)
    tiny_tensor_t* bias;          // Bias vector [out_channels] (can be NULL)
    int spatial_dims;            // Number of spatial dimensions (1, 2, or 3)
    int in_channels;              // Number of input channels
    int out_channels;             // Number of output channels
    // 2D parameters (also used for 1D with kernel_h=1, stride_h=1, pad_h=0)
    int kernel_h;                 // Kernel height (1D: unused, 2D/3D: used)
    int kernel_w;                 // Kernel width (1D/2D/3D: used)
    int stride_h;                 // Stride in height dimension
    int stride_w;                 // Stride in width dimension
    int pad_h;                    // Padding in height dimension (symmetric)
    int pad_w;                    // Padding in width dimension (symmetric)
    // 3D parameters (only used when spatial_dims=3)
    int kernel_d;                 // Kernel depth (3D only)
    int stride_d;                 // Stride in depth dimension (3D only)
    int pad_d;                    // Padding in depth dimension (3D only, symmetric)
    bool use_bias;                // Whether to use bias
} tiny_convolution_params_t;

/* ============================================================================
 * FUNCTION PROTOTYPES - Layer Creation and Destruction
 * ============================================================================ */

/**
 * @brief Create a 1D convolution layer
 * 
 * @param in_channels Number of input channels
 * @param out_channels Number of output channels
 * @param kernel_w Kernel width (temporal dimension)
 * @param stride_w Stride in width dimension
 * @param pad_w Padding in width dimension (symmetric, applied to both sides)
 * @param use_bias Whether to use bias (true) or not (false)
 * @return tiny_convolution_params_t* Pointer to created layer parameters, NULL on failure
 * 
 * @note Input shape: [batch, in_channels, length]
 * @note Output shape: [batch, out_channels, out_length]
 * @note Weights shape: [out_channels, in_channels, kernel_w]
 * @note out_length = (length + 2*pad_w - kernel_w) / stride_w + 1
 */
tiny_convolution_params_t* tiny_convolution_create_1d(int in_channels, int out_channels,
                                                      int kernel_w, int stride_w, int pad_w,
                                                      bool use_bias);

/**
 * @brief Create a 2D convolution layer
 * 
 * @param in_channels Number of input channels
 * @param out_channels Number of output channels
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 * @param stride_h Stride in height dimension
 * @param stride_w Stride in width dimension
 * @param pad_h Padding in height dimension (symmetric, applied to both sides)
 * @param pad_w Padding in width dimension (symmetric, applied to both sides)
 * @param use_bias Whether to use bias (true) or not (false)
 * @return tiny_convolution_params_t* Pointer to created layer parameters, NULL on failure
 * 
 * @note Input shape: [batch, in_channels, height, width]
 * @note Output shape: [batch, out_channels, out_height, out_width]
 * @note Weights shape: [out_channels, in_channels, kernel_h, kernel_w]
 * @note out_height = (height + 2*pad_h - kernel_h) / stride_h + 1
 * @note out_width = (width + 2*pad_w - kernel_w) / stride_w + 1
 */
tiny_convolution_params_t* tiny_convolution_create_2d(int in_channels, int out_channels,
                                                      int kernel_h, int kernel_w,
                                                      int stride_h, int stride_w,
                                                      int pad_h, int pad_w,
                                                      bool use_bias);

/**
 * @brief Create a 3D convolution layer
 * 
 * @param in_channels Number of input channels
 * @param out_channels Number of output channels
 * @param kernel_d Kernel depth
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 * @param stride_d Stride in depth dimension
 * @param stride_h Stride in height dimension
 * @param stride_w Stride in width dimension
 * @param pad_d Padding in depth dimension (symmetric, applied to both sides)
 * @param pad_h Padding in height dimension (symmetric, applied to both sides)
 * @param pad_w Padding in width dimension (symmetric, applied to both sides)
 * @param use_bias Whether to use bias (true) or not (false)
 * @return tiny_convolution_params_t* Pointer to created layer parameters, NULL on failure
 * 
 * @note Input shape: [batch, in_channels, depth, height, width]
 * @note Output shape: [batch, out_channels, out_depth, out_height, out_width]
 * @note Weights shape: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
 * @note out_depth = (depth + 2*pad_d - kernel_d) / stride_d + 1
 * @note out_height = (height + 2*pad_h - kernel_h) / stride_h + 1
 * @note out_width = (width + 2*pad_w - kernel_w) / stride_w + 1
 */
tiny_convolution_params_t* tiny_convolution_create_3d(int in_channels, int out_channels,
                                                      int kernel_d, int kernel_h, int kernel_w,
                                                      int stride_d, int stride_h, int stride_w,
                                                      int pad_d, int pad_h, int pad_w,
                                                      bool use_bias);

/**
 * @brief Create a 2D convolution layer (backward compatibility)
 * 
 * @deprecated Use tiny_convolution_create_2d() instead
 */
#define tiny_convolution_create tiny_convolution_create_2d

/**
 * @brief Destroy a convolution layer and free all resources
 * 
 * @param params Layer parameters to destroy (can be NULL)
 * 
 * @note This destroys the weights and bias tensors
 */
void tiny_convolution_destroy(tiny_convolution_params_t* params);

/* ============================================================================
 * FUNCTION PROTOTYPES - Forward and Backward Propagation
 * ============================================================================ */

/**
 * @brief Forward propagation for convolution layer
 * 
 * @param node Graph node containing the layer
 * 
 * @details
 * Performs: output = conv(input, weights) + bias
 * 
 * For 1D convolution:
 * - input: [batch, in_channels, length]
 * - weights: [out_channels, in_channels, kernel_w]
 * - output: [batch, out_channels, out_length]
 * 
 * For 2D convolution:
 * - input: [batch, in_channels, height, width]
 * - weights: [out_channels, in_channels, kernel_h, kernel_w]
 * - output: [batch, out_channels, out_height, out_width]
 * 
 * For 3D convolution:
 * - input: [batch, in_channels, depth, height, width]
 * - weights: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
 * - output: [batch, out_channels, out_depth, out_height, out_width]
 * 
 * @note This function is designed to be called by the computation graph
 * @note Input tensor should be in node->inputs[0]
 * @note Output tensor should be in node->outputs[0]
 * @note Parameters should be in node->params (cast to tiny_convolution_params_t*)
 * @note Automatically detects spatial_dims from params and processes accordingly
 */
void tiny_convolution_forward(tiny_graph_node_t* node);

/**
 * @brief Backward propagation for convolution layer
 * 
 * @param node Graph node containing the layer
 * 
 * @details
 * Computes gradients:
 * - input_grad: gradient w.r.t. input
 * - weight_grad: gradient w.r.t. weights
 * - bias_grad: gradient w.r.t. bias (if use_bias)
 * 
 * @note This function is designed to be called by the computation graph
 * @note Only computes gradients if tensors have requires_grad = true
 */
void tiny_convolution_backward(tiny_graph_node_t* node);

/* ============================================================================
 * FUNCTION PROTOTYPES - Utility
 * ============================================================================ */

/**
 * @brief Initialize weights with random values (Xavier/Glorot initialization)
 * 
 * @param params Layer parameters
 * @param seed Random seed (0 for time-based seed)
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @note Uses uniform distribution: [-sqrt(6/(in*out)), sqrt(6/(in*out))]
 * @note For 1D: in = in_channels * kernel_w, out = out_channels * kernel_w
 * @note For 2D: in = in_channels * kernel_h * kernel_w, out = out_channels * kernel_h * kernel_w
 * @note For 3D: in = in_channels * kernel_d * kernel_h * kernel_w, out = out_channels * kernel_d * kernel_h * kernel_w
 */
tiny_error_t tiny_convolution_init_weights_xavier(tiny_convolution_params_t* params, unsigned int seed);

/**
 * @brief Initialize weights with zeros
 * 
 * @param params Layer parameters
 * @return tiny_error_t TINY_OK on success, error code on failure
 */
tiny_error_t tiny_convolution_init_weights_zero(tiny_convolution_params_t* params);

#ifdef __cplusplus
}
#endif

