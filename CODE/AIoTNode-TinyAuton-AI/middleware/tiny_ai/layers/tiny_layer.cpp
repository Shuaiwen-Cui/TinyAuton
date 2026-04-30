/**
 * @file tiny_layer.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief ActivationLayer, Flatten, and GlobalAvgPool implementations.
 * @version 1.0
 * @date 2026-04-30
 * @copyright Copyright (c) 2026
 */

#include "tiny_layer.hpp"
#include <cstring>

#ifdef __cplusplus

namespace tiny
{

// ============================================================================
// ActivationLayer
// ============================================================================

Tensor ActivationLayer::forward(const Tensor &x)
{
    // Cache input for backward (output for Sigmoid/Tanh/Softmax)
    Tensor out = act_forward(x, type_, alpha_);
#if TINY_AI_TRAINING_ENABLED
    // For Sigmoid/Tanh/Softmax backward we need the *output*; for others the *input*
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

// ============================================================================
// Flatten
// ============================================================================

Tensor Flatten::forward(const Tensor &x)
{
#if TINY_AI_TRAINING_ENABLED
    in_ndim = x.ndim;
    memcpy(in_shape, x.shape, sizeof(in_shape));
#endif
    int batch    = x.shape[0];
    int flat     = x.size / batch;
    Tensor out   = x.clone();
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

// ============================================================================
// GlobalAvgPool
// ============================================================================

// Input:  [batch, seq, feat]
// Output: [batch, feat]   (mean over seq dimension)
Tensor GlobalAvgPool::forward(const Tensor &x)
{
    int batch = x.shape[0];
    int seq   = x.shape[1];
    int feat  = x.shape[2];

#if TINY_AI_TRAINING_ENABLED
    in_batch = batch;
    in_seq   = seq;
    in_feat  = feat;
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

// Distribute gradient equally across seq positions
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
