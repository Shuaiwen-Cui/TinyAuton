/**
 * @file tiny_dense.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Dense layer implementation.
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#include "tiny_dense.hpp"
#include <cmath>
#include <cstdlib>

#ifdef __cplusplus

namespace tiny
{

// ============================================================================
// Constructor
// ============================================================================

Dense::Dense(int in_features, int out_features, bool use_bias)
    : Layer("dense", true),
      weight(out_features, in_features),
      bias(use_bias ? out_features : 0),
      in_feat_(in_features), out_feat_(out_features), use_bias_(use_bias)
{
#if TINY_AI_TRAINING_ENABLED
    dweight = Tensor::zeros_like(weight);
    if (use_bias_) dbias = Tensor(out_features);
#endif
    xavier_init();
}

// Xavier-uniform:  W ~ U(-limit, +limit),  limit = sqrt(6 / (fan_in + fan_out))
void Dense::xavier_init()
{
    float limit = sqrtf(6.0f / (float)(in_feat_ + out_feat_));
    for (int i = 0; i < weight.size; i++)
    {
        float r    = (float)rand() / (float)RAND_MAX;  // [0, 1)
        weight.data[i] = (r * 2.0f - 1.0f) * limit;
    }
    if (use_bias_) bias.zero();
}

// ============================================================================
// Forward:  out[b, o] = sum_i(W[o, i] * x[b, i]) + b[o]
// ============================================================================

Tensor Dense::forward(const Tensor &x)
{
    int batch = x.shape[0];
    Tensor out(batch, out_feat_);

#if TINY_AI_TRAINING_ENABLED
    x_cache_ = x.clone();
#endif

    for (int b = 0; b < batch; b++)
        for (int o = 0; o < out_feat_; o++)
        {
            float sum = use_bias_ ? bias.data[o] : 0.0f;
            for (int i = 0; i < in_feat_; i++)
                sum += weight.at(o, i) * x.at(b, i);
            out.at(b, o) = sum;
        }
    return out;
}

// ============================================================================
// Backward
// ============================================================================
#if TINY_AI_TRAINING_ENABLED

Tensor Dense::backward(const Tensor &grad_out)
{
    int batch = x_cache_.shape[0];
    Tensor grad_input(batch, in_feat_);

    // dW += grad_out^T @ x_cache
    for (int o = 0; o < out_feat_; o++)
        for (int i = 0; i < in_feat_; i++)
        {
            float acc = 0.0f;
            for (int b = 0; b < batch; b++)
                acc += grad_out.at(b, o) * x_cache_.at(b, i);
            dweight.at(o, i) += acc;
        }

    // db += sum_batch(grad_out)
    if (use_bias_)
        for (int o = 0; o < out_feat_; o++)
        {
            float acc = 0.0f;
            for (int b = 0; b < batch; b++) acc += grad_out.at(b, o);
            dbias.data[o] += acc;
        }

    // grad_input = grad_out @ W
    for (int b = 0; b < batch; b++)
        for (int i = 0; i < in_feat_; i++)
        {
            float acc = 0.0f;
            for (int o = 0; o < out_feat_; o++)
                acc += grad_out.at(b, o) * weight.at(o, i);
            grad_input.at(b, i) = acc;
        }
    return grad_input;
}

void Dense::collect_params(std::vector<ParamGroup> &groups)
{
    groups.push_back({&weight, &dweight});
    if (use_bias_) groups.push_back({&bias, &dbias});
}

#endif // TINY_AI_TRAINING_ENABLED

} // namespace tiny

#endif // __cplusplus
