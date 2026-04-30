/**
 * @file tiny_optimizer.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief SGD and Adam optimizer implementations.
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#include "tiny_optimizer.hpp"
#include <cmath>
#include <cstring>

#ifdef __cplusplus

namespace tiny
{

// ============================================================================
// Optimizer base
// ============================================================================

void Optimizer::zero_grad(std::vector<ParamGroup> &groups)
{
    for (auto &pg : groups)
        if (pg.grad) pg.grad->zero();
}

// ============================================================================
// SGD
// ============================================================================

SGD::SGD(float lr, float momentum, float weight_decay)
    : lr_(lr), momentum_(momentum), weight_decay_(weight_decay)
{
}

void SGD::init(const std::vector<ParamGroup> &groups)
{
    velocity_.clear();
    velocity_.reserve(groups.size());
    for (const auto &pg : groups)
        velocity_.push_back(Tensor::zeros_like(*pg.param));
}

void SGD::step(std::vector<ParamGroup> &groups)
{
    for (size_t p = 0; p < groups.size(); p++)
    {
        Tensor &param = *groups[p].param;
        Tensor &grad  = *groups[p].grad;
        Tensor &vel   = velocity_[p];

        for (int i = 0; i < param.size; i++)
        {
            float g = grad.data[i];

            // L2 weight decay adds λ·w to gradient
            if (weight_decay_ > 0.0f) g += weight_decay_ * param.data[i];

            // Momentum update: v = μ·v + g
            if (momentum_ > 0.0f)
            {
                vel.data[i] = momentum_ * vel.data[i] + g;
                g = vel.data[i];
            }

            param.data[i] -= lr_ * g;
        }
    }
}

// ============================================================================
// Adam
// ============================================================================

Adam::Adam(float lr, float beta1, float beta2, float epsilon, float weight_decay)
    : lr_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
      weight_decay_(weight_decay), t_(0)
{
}

void Adam::init(const std::vector<ParamGroup> &groups)
{
    t_ = 0;
    m_.clear();
    v_.clear();
    m_.reserve(groups.size());
    v_.reserve(groups.size());
    for (const auto &pg : groups)
    {
        m_.push_back(Tensor::zeros_like(*pg.param));
        v_.push_back(Tensor::zeros_like(*pg.param));
    }
}

void Adam::step(std::vector<ParamGroup> &groups)
{
    t_++;

    // Bias-corrected learning rate (avoids computing per-element bias correction)
    float bc1  = 1.0f - powf(beta1_, (float)t_);
    float bc2  = 1.0f - powf(beta2_, (float)t_);
    float lr_t = lr_ * sqrtf(bc2) / (bc1 + TINY_MATH_MIN_DENOMINATOR);

    for (size_t p = 0; p < groups.size(); p++)
    {
        Tensor &param = *groups[p].param;
        Tensor &grad  = *groups[p].grad;
        Tensor &m     = m_[p];
        Tensor &v     = v_[p];

        for (int i = 0; i < param.size; i++)
        {
            float g = grad.data[i];
            if (weight_decay_ > 0.0f) g += weight_decay_ * param.data[i];

            m.data[i] = beta1_ * m.data[i] + (1.0f - beta1_) * g;
            v.data[i] = beta2_ * v.data[i] + (1.0f - beta2_) * g * g;

            param.data[i] -= lr_t * m.data[i] / (sqrtf(v.data[i]) + epsilon_);
        }
    }
}

} // namespace tiny

#endif // __cplusplus
