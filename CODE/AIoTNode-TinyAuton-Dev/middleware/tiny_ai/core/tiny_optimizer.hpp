/**
 * @file tiny_optimizer.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Optimizers for tiny_ai — SGD (with momentum) and Adam-lite.
 *        Designed for on-device training on ESP32-S3 with up to 8 MB PSRAM.
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#pragma once

#include "tiny_tensor.hpp"

#ifdef __cplusplus

#include <vector>

namespace tiny
{

/* ============================================================================
 * ParamGroup — pairs a parameter tensor with its gradient tensor.
 *              Layers register (param, grad) pairs via collect_params().
 * ============================================================================ */
struct ParamGroup
{
    Tensor *param;  ///< Pointer to the weight / bias tensor
    Tensor *grad;   ///< Pointer to the corresponding gradient tensor
};

/* ============================================================================
 * Optimizer — abstract base class
 * ============================================================================ */
class Optimizer
{
public:
    virtual ~Optimizer() {}

    /**
     * @brief Initialise internal buffers (moments, velocities …) to match the
     *        shapes of all registered parameters.  Must be called once after
     *        collect_params().
     */
    virtual void init(const std::vector<ParamGroup> &groups) = 0;

    /// Apply one gradient-descent step to every registered parameter.
    virtual void step(std::vector<ParamGroup> &groups) = 0;

    /// Zero all gradient tensors.
    virtual void zero_grad(std::vector<ParamGroup> &groups);
};

/* ============================================================================
 * SGD with optional momentum and weight decay
 * ============================================================================ */
class SGD : public Optimizer
{
public:
    /**
     * @param lr           Learning rate
     * @param momentum     Momentum factor (0 = vanilla SGD)
     * @param weight_decay L2 regularisation coefficient
     */
    SGD(float lr = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f);

    void init(const std::vector<ParamGroup> &groups) override;
    void step(std::vector<ParamGroup> &groups) override;

private:
    float lr_;
    float momentum_;
    float weight_decay_;
    std::vector<Tensor> velocity_;  ///< Momentum buffer (one per param)
};

/* ============================================================================
 * Adam (Kingma & Ba, 2015) — lightweight edition
 * ============================================================================ */
class Adam : public Optimizer
{
public:
    /**
     * @param lr           Learning rate (default 1e-3)
     * @param beta1        First moment decay  (default 0.9)
     * @param beta2        Second moment decay (default 0.999)
     * @param epsilon      Numerical stability constant (default 1e-8)
     * @param weight_decay L2 regularisation coefficient (default 0)
     */
    Adam(float lr       = 1e-3f,
         float beta1    = 0.9f,
         float beta2    = 0.999f,
         float epsilon  = 1e-8f,
         float weight_decay = 0.0f);

    void init(const std::vector<ParamGroup> &groups) override;
    void step(std::vector<ParamGroup> &groups) override;

private:
    float lr_;
    float beta1_;
    float beta2_;
    float epsilon_;
    float weight_decay_;
    int   t_;                       ///< Time step (incremented each step() call)
    std::vector<Tensor> m_;         ///< First moment estimates
    std::vector<Tensor> v_;         ///< Second moment estimates
};

} // namespace tiny

#endif // __cplusplus
