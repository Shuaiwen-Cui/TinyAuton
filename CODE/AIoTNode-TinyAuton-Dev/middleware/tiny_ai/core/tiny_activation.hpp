/**
 * @file tiny_activation.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Activation functions for tiny_ai — forward and (optionally) backward.
 *        All operations are stateless free functions; an Activation Layer
 *        wrapping these is defined in layers/tiny_layer.hpp.
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#pragma once

#include "tiny_tensor.hpp"

#ifdef __cplusplus

namespace tiny
{

/* ============================================================================
 * Activation type enum
 * ============================================================================ */
enum class ActType
{
    RELU = 0,        ///< max(0, x)
    LEAKY_RELU,      ///< x > 0 ? x : alpha*x  (alpha default 0.01)
    SIGMOID,         ///< 1 / (1 + exp(-x))
    TANH,            ///< tanh(x)
    SOFTMAX,         ///< exp(xi) / sum(exp(xj))  — applied along last dim
    GELU,            ///< x * Phi(x) approx via tanh polynomial
    LINEAR           ///< identity — no-op
};

/* ============================================================================
 * Forward passes (operate on flat data, return new Tensor)
 * ============================================================================ */

/// ReLU: max(0, x), element-wise
Tensor relu_forward(const Tensor &x);

/// Leaky ReLU: x > 0 ? x : alpha * x
Tensor leaky_relu_forward(const Tensor &x, float alpha = 0.01f);

/// Sigmoid: 1 / (1 + exp(-x))
Tensor sigmoid_forward(const Tensor &x);

/// Tanh
Tensor tanh_forward(const Tensor &x);

/**
 * @brief Softmax along the last dimension.
 *        Input shape: [batch, classes] or any shape where the last dim is the
 *        class/feature axis.
 */
Tensor softmax_forward(const Tensor &x);

/// GELU approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
Tensor gelu_forward(const Tensor &x);

/* ============================================================================
 * In-place forward passes (modify x directly, return reference)
 * ============================================================================ */

void relu_inplace(Tensor &x);
void leaky_relu_inplace(Tensor &x, float alpha = 0.01f);
void sigmoid_inplace(Tensor &x);
void tanh_inplace(Tensor &x);
void softmax_inplace(Tensor &x);
void gelu_inplace(Tensor &x);

/* ============================================================================
 * Backward passes (return gradient w.r.t. input)
 * ============================================================================ */
#if TINY_AI_TRAINING_ENABLED

/// ReLU backward: grad_out * (x > 0)
Tensor relu_backward(const Tensor &x, const Tensor &grad_out);

/// Leaky ReLU backward
Tensor leaky_relu_backward(const Tensor &x, const Tensor &grad_out, float alpha = 0.01f);

/// Sigmoid backward: grad_out * sigmoid(x) * (1 - sigmoid(x))
/// Pass in the *output* of sigmoid_forward as `y` to avoid recomputation.
Tensor sigmoid_backward(const Tensor &y, const Tensor &grad_out);

/// Tanh backward: grad_out * (1 - tanh(x)^2)
/// Pass in the *output* of tanh_forward as `y`.
Tensor tanh_backward(const Tensor &y, const Tensor &grad_out);

/// Softmax backward: Jacobian-vector product
/// Pass in the *output* of softmax_forward as `y`.
Tensor softmax_backward(const Tensor &y, const Tensor &grad_out);

/// GELU backward (numerical via tanh approximation)
Tensor gelu_backward(const Tensor &x, const Tensor &grad_out);

#endif // TINY_AI_TRAINING_ENABLED

/* ============================================================================
 * Dispatch helpers
 * ============================================================================ */

Tensor act_forward(const Tensor &x, ActType type, float alpha = 0.01f);
void   act_inplace(Tensor &x, ActType type, float alpha = 0.01f);

#if TINY_AI_TRAINING_ENABLED
/// cache: the activation input (for ReLU/LeakyReLU/GELU) or output (for Sigmoid/Tanh/Softmax)
Tensor act_backward(const Tensor &cache, const Tensor &grad_out,
                    ActType type, float alpha = 0.01f);
#endif

} // namespace tiny

#endif // __cplusplus
