# Code

## tiny_activation.hpp

```cpp
/**
 * @file tiny_activation.hpp
 * @brief Activation functions for tiny_ai — forward and (optionally) backward.
 */

#pragma once

#include "tiny_tensor.hpp"

#ifdef __cplusplus

namespace tiny
{

enum class ActType
{
    RELU = 0,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    SOFTMAX,
    GELU,
    LINEAR
};

Tensor relu_forward       (const Tensor &x);
Tensor leaky_relu_forward (const Tensor &x, float alpha = 0.01f);
Tensor sigmoid_forward    (const Tensor &x);
Tensor tanh_forward       (const Tensor &x);
Tensor softmax_forward    (const Tensor &x);
Tensor gelu_forward       (const Tensor &x);

void relu_inplace       (Tensor &x);
void leaky_relu_inplace (Tensor &x, float alpha = 0.01f);
void sigmoid_inplace    (Tensor &x);
void tanh_inplace       (Tensor &x);
void softmax_inplace    (Tensor &x);
void gelu_inplace       (Tensor &x);

#if TINY_AI_TRAINING_ENABLED
Tensor relu_backward       (const Tensor &x, const Tensor &grad_out);
Tensor leaky_relu_backward (const Tensor &x, const Tensor &grad_out, float alpha = 0.01f);
Tensor sigmoid_backward    (const Tensor &y, const Tensor &grad_out);
Tensor tanh_backward       (const Tensor &y, const Tensor &grad_out);
Tensor softmax_backward    (const Tensor &y, const Tensor &grad_out);
Tensor gelu_backward       (const Tensor &x, const Tensor &grad_out);
#endif

Tensor act_forward (const Tensor &x, ActType type, float alpha = 0.01f);
void   act_inplace (Tensor &x,       ActType type, float alpha = 0.01f);

#if TINY_AI_TRAINING_ENABLED
Tensor act_backward(const Tensor &cache, const Tensor &grad_out,
                    ActType type, float alpha = 0.01f);
#endif

} // namespace tiny

#endif // __cplusplus
```

## tiny_activation.cpp

```cpp
/**
 * @file tiny_activation.cpp
 * @brief Activation functions implementation.
 */

#include "tiny_activation.hpp"
#include <cmath>

#ifdef __cplusplus

namespace tiny
{

Tensor relu_forward(const Tensor &x)        { Tensor o(x.clone()); relu_inplace(o);        return o; }
Tensor leaky_relu_forward(const Tensor &x, float a) { Tensor o(x.clone()); leaky_relu_inplace(o, a); return o; }
Tensor sigmoid_forward(const Tensor &x)     { Tensor o(x.clone()); sigmoid_inplace(o);     return o; }
Tensor tanh_forward(const Tensor &x)        { Tensor o(x.clone()); tanh_inplace(o);        return o; }
Tensor softmax_forward(const Tensor &x)     { Tensor o(x.clone()); softmax_inplace(o);     return o; }
Tensor gelu_forward(const Tensor &x)        { Tensor o(x.clone()); gelu_inplace(o);        return o; }

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
    int rows = x.size / x.cols();
    int cls  = x.cols();
    for (int r = 0; r < rows; r++)
    {
        float *row = x.data + r * cls;
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
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    for (int i = 0; i < x.size; i++)
    {
        float v = x.data[i];
        float c = SQRT_2_OVER_PI * (v + 0.044715f * v * v * v);
        x.data[i] = 0.5f * v * (1.0f + tanhf(c));
    }
}

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

Tensor sigmoid_backward(const Tensor &y, const Tensor &grad_out)
{
    Tensor g(grad_out.clone());
    for (int i = 0; i < y.size; i++)
        g.data[i] *= y.data[i] * (1.0f - y.data[i]);
    return g;
}

Tensor tanh_backward(const Tensor &y, const Tensor &grad_out)
{
    Tensor g(grad_out.clone());
    for (int i = 0; i < y.size; i++)
        g.data[i] *= (1.0f - y.data[i] * y.data[i]);
    return g;
}

Tensor softmax_backward(const Tensor &y, const Tensor &grad_out)
{
    Tensor g(grad_out.clone());
    int rows = y.size / y.cols();
    int cls  = y.cols();
    for (int r = 0; r < rows; r++)
    {
        const float *yr  = y.data        + r * cls;
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
```
