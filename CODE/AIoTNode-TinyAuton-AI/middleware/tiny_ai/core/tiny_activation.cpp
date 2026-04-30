/**
 * @file tiny_activation.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Activation functions implementation.
 * @version 1.0
 * @date 2026-04-30
 * @copyright Copyright (c) 2026
 */

#include "tiny_activation.hpp"
#include <cmath>

#ifdef __cplusplus

namespace tiny
{

// ============================================================================
// Forward passes
// ============================================================================

Tensor relu_forward(const Tensor &x)
{
    Tensor out(x.clone());
    relu_inplace(out);
    return out;
}

Tensor leaky_relu_forward(const Tensor &x, float alpha)
{
    Tensor out(x.clone());
    leaky_relu_inplace(out, alpha);
    return out;
}

Tensor sigmoid_forward(const Tensor &x)
{
    Tensor out(x.clone());
    sigmoid_inplace(out);
    return out;
}

Tensor tanh_forward(const Tensor &x)
{
    Tensor out(x.clone());
    tanh_inplace(out);
    return out;
}

Tensor softmax_forward(const Tensor &x)
{
    Tensor out(x.clone());
    softmax_inplace(out);
    return out;
}

Tensor gelu_forward(const Tensor &x)
{
    Tensor out(x.clone());
    gelu_inplace(out);
    return out;
}

// ============================================================================
// In-place forward passes
// ============================================================================

void relu_inplace(Tensor &x)
{
    for (int i = 0; i < x.size; i++)
        if (x.data[i] < 0.0f) x.data[i] = 0.0f;
}

void leaky_relu_inplace(Tensor &x, float alpha)
{
    for (int i = 0; i < x.size; i++)
        if (x.data[i] < 0.0f) x.data[i] *= alpha;
}

void sigmoid_inplace(Tensor &x)
{
    for (int i = 0; i < x.size; i++)
        x.data[i] = 1.0f / (1.0f + expf(-x.data[i]));
}

void tanh_inplace(Tensor &x)
{
    for (int i = 0; i < x.size; i++)
        x.data[i] = tanhf(x.data[i]);
}

void softmax_inplace(Tensor &x)
{
    // Apply row-wise (last dim is class axis)
    int rows = x.size / x.cols();
    int cls  = x.cols();

    for (int r = 0; r < rows; r++)
    {
        float *row = x.data + r * cls;

        // Numerically stable: subtract row max before exp
        float mx = row[0];
        for (int c = 1; c < cls; c++) if (row[c] > mx) mx = row[c];

        float sum = 0.0f;
        for (int c = 0; c < cls; c++) { row[c] = expf(row[c] - mx); sum += row[c]; }

        float inv = 1.0f / (sum + TINY_MATH_MIN_DENOMINATOR);
        for (int c = 0; c < cls; c++) row[c] *= inv;
    }
}

void gelu_inplace(Tensor &x)
{
    // GELU approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    for (int i = 0; i < x.size; i++)
    {
        float v  = x.data[i];
        float c  = SQRT_2_OVER_PI * (v + 0.044715f * v * v * v);
        x.data[i] = 0.5f * v * (1.0f + tanhf(c));
    }
}

// ============================================================================
// Backward passes
// ============================================================================
#if TINY_AI_TRAINING_ENABLED

Tensor relu_backward(const Tensor &x, const Tensor &grad_out)
{
    Tensor g(x.clone());
    for (int i = 0; i < x.size; i++)
        g.data[i] = (x.data[i] > 0.0f) ? grad_out.data[i] : 0.0f;
    return g;
}

Tensor leaky_relu_backward(const Tensor &x, const Tensor &grad_out, float alpha)
{
    Tensor g(grad_out.clone());
    for (int i = 0; i < x.size; i++)
        if (x.data[i] < 0.0f) g.data[i] *= alpha;
    return g;
}

// y = sigmoid(x), grad_input = grad_out * y * (1 - y)
Tensor sigmoid_backward(const Tensor &y, const Tensor &grad_out)
{
    Tensor g(grad_out.clone());
    for (int i = 0; i < y.size; i++)
        g.data[i] *= y.data[i] * (1.0f - y.data[i]);
    return g;
}

// y = tanh(x), grad_input = grad_out * (1 - y^2)
Tensor tanh_backward(const Tensor &y, const Tensor &grad_out)
{
    Tensor g(grad_out.clone());
    for (int i = 0; i < y.size; i++)
        g.data[i] *= (1.0f - y.data[i] * y.data[i]);
    return g;
}

// Softmax Jacobian-vector product:
//   dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
Tensor softmax_backward(const Tensor &y, const Tensor &grad_out)
{
    Tensor g(grad_out.clone());
    int rows = y.size / y.cols();
    int cls  = y.cols();

    for (int r = 0; r < rows; r++)
    {
        const float *yr  = y.data       + r * cls;
        const float *gor = grad_out.data + r * cls;
        float       *gr  = g.data        + r * cls;

        float dot = 0.0f;
        for (int c = 0; c < cls; c++) dot += gor[c] * yr[c];
        for (int c = 0; c < cls; c++) gr[c] = yr[c] * (gor[c] - dot);
    }
    return g;
}

Tensor gelu_backward(const Tensor &x, const Tensor &grad_out)
{
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    Tensor g(grad_out.clone());
    for (int i = 0; i < x.size; i++)
    {
        float v  = x.data[i];
        float c  = SQRT_2_OVER_PI * (v + 0.044715f * v * v * v);
        float tc = tanhf(c);
        float dc = SQRT_2_OVER_PI * (1.0f + 3.0f * 0.044715f * v * v);
        float d  = 0.5f * (1.0f + tc) + 0.5f * v * (1.0f - tc * tc) * dc;
        g.data[i] *= d;
    }
    return g;
}

#endif // TINY_AI_TRAINING_ENABLED

// ============================================================================
// Dispatch helpers
// ============================================================================

Tensor act_forward(const Tensor &x, ActType type, float alpha)
{
    switch (type)
    {
        case ActType::RELU:       return relu_forward(x);
        case ActType::LEAKY_RELU: return leaky_relu_forward(x, alpha);
        case ActType::SIGMOID:    return sigmoid_forward(x);
        case ActType::TANH:       return tanh_forward(x);
        case ActType::SOFTMAX:    return softmax_forward(x);
        case ActType::GELU:       return gelu_forward(x);
        default:                  return x.clone();
    }
}

void act_inplace(Tensor &x, ActType type, float alpha)
{
    switch (type)
    {
        case ActType::RELU:       relu_inplace(x);            break;
        case ActType::LEAKY_RELU: leaky_relu_inplace(x, alpha); break;
        case ActType::SIGMOID:    sigmoid_inplace(x);          break;
        case ActType::TANH:       tanh_inplace(x);             break;
        case ActType::SOFTMAX:    softmax_inplace(x);          break;
        case ActType::GELU:       gelu_inplace(x);             break;
        default: break;
    }
}

#if TINY_AI_TRAINING_ENABLED

Tensor act_backward(const Tensor &cache, const Tensor &grad_out,
                    ActType type, float alpha)
{
    switch (type)
    {
        case ActType::RELU:       return relu_backward(cache, grad_out);
        case ActType::LEAKY_RELU: return leaky_relu_backward(cache, grad_out, alpha);
        case ActType::SIGMOID:    return sigmoid_backward(cache, grad_out);
        case ActType::TANH:       return tanh_backward(cache, grad_out);
        case ActType::SOFTMAX:    return softmax_backward(cache, grad_out);
        case ActType::GELU:       return gelu_backward(cache, grad_out);
        default:                  return grad_out.clone();
    }
}

#endif // TINY_AI_TRAINING_ENABLED

} // namespace tiny

#endif // __cplusplus
