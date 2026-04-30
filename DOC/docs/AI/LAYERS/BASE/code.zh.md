# 代码

## tiny_layer.hpp

```cpp
/**
 * @file tiny_layer.hpp
 * @brief Abstract base class for all tiny_ai neural network layers.
 */

#pragma once

#include "tiny_tensor.hpp"
#include "tiny_optimizer.hpp"
#include "tiny_activation.hpp"

#ifdef __cplusplus

#include <vector>

namespace tiny
{

class Layer
{
public:
    const char *name;
    bool        trainable;

    explicit Layer(const char *name = "layer", bool trainable = false)
        : name(name), trainable(trainable) {}

    virtual ~Layer() {}

    virtual Tensor forward(const Tensor &x) = 0;

#if TINY_AI_TRAINING_ENABLED
    virtual Tensor backward(const Tensor &grad_out) = 0;
    virtual void   collect_params(std::vector<ParamGroup> &groups) {}
#endif
};

class ActivationLayer : public Layer
{
public:
    explicit ActivationLayer(ActType type, float alpha = 0.01f)
        : Layer("activation", false), type_(type), alpha_(alpha) {}

    Tensor forward(const Tensor &x) override;

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override;
#endif

private:
    ActType type_;
    float   alpha_;
    Tensor  cache_;
};

class Flatten : public Layer
{
public:
    Flatten() : Layer("flatten", false) {}

    Tensor forward(const Tensor &x) override;

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override;
#endif

private:
    int in_shape[4];
    int in_ndim;
};

class GlobalAvgPool : public Layer
{
public:
    GlobalAvgPool() : Layer("global_avg_pool", false) {}

    Tensor forward(const Tensor &x) override;

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override;
#endif

private:
    int in_seq;
    int in_feat;
    int in_batch;
};

} // namespace tiny

#endif // __cplusplus
```

## tiny_layer.cpp

```cpp
/**
 * @file tiny_layer.cpp
 * @brief ActivationLayer, Flatten, and GlobalAvgPool implementations.
 */

#include "tiny_layer.hpp"
#include <cstring>

#ifdef __cplusplus

namespace tiny
{

// ---- ActivationLayer ------------------------------------------------------
Tensor ActivationLayer::forward(const Tensor &x)
{
    Tensor out = act_forward(x, type_, alpha_);
#if TINY_AI_TRAINING_ENABLED
    if (type_ == ActType::SIGMOID || type_ == ActType::TANH || type_ == ActType::SOFTMAX)
        cache_ = out.clone();
    else
        cache_ = x.clone();
#endif
    return out;
}

#if TINY_AI_TRAINING_ENABLED
Tensor ActivationLayer::backward(const Tensor &grad_out)
{
    return act_backward(cache_, grad_out, type_, alpha_);
}
#endif

// ---- Flatten --------------------------------------------------------------
Tensor Flatten::forward(const Tensor &x)
{
#if TINY_AI_TRAINING_ENABLED
    in_ndim = x.ndim;
    memcpy(in_shape, x.shape, sizeof(in_shape));
#endif
    int batch = x.shape[0];
    int flat  = x.size / batch;
    Tensor out = x.clone();
    out.reshape_2d(batch, flat);
    return out;
}

#if TINY_AI_TRAINING_ENABLED
Tensor Flatten::backward(const Tensor &grad_out)
{
    Tensor g = grad_out.clone();
    g.reshape(in_ndim, in_shape);
    return g;
}
#endif

// ---- GlobalAvgPool --------------------------------------------------------
Tensor GlobalAvgPool::forward(const Tensor &x)
{
    int batch = x.shape[0];
    int seq   = x.shape[1];
    int feat  = x.shape[2];

#if TINY_AI_TRAINING_ENABLED
    in_batch = batch; in_seq = seq; in_feat = feat;
#endif

    Tensor out(batch, feat);
    float inv_seq = 1.0f / (float)seq;
    for (int b = 0; b < batch; b++)
        for (int f = 0; f < feat; f++)
        {
            float sum = 0.0f;
            for (int s = 0; s < seq; s++) sum += x.at(b, s, f);
            out.at(b, f) = sum * inv_seq;
        }
    return out;
}

#if TINY_AI_TRAINING_ENABLED
Tensor GlobalAvgPool::backward(const Tensor &grad_out)
{
    Tensor g(in_batch, in_seq, in_feat);
    float inv_seq = 1.0f / (float)in_seq;
    for (int b = 0; b < in_batch; b++)
        for (int s = 0; s < in_seq; s++)
            for (int f = 0; f < in_feat; f++)
                g.at(b, s, f) = grad_out.at(b, f) * inv_seq;
    return g;
}
#endif

} // namespace tiny

#endif // __cplusplus
```
