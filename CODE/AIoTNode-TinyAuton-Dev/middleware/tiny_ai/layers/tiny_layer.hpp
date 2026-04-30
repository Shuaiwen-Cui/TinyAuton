/**
 * @file tiny_layer.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Abstract base class for all tiny_ai neural network layers.
 *        Defines the forward / backward interface and parameter registration.
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#pragma once

#include "tiny_tensor.hpp"
#include "tiny_optimizer.hpp"
#include "tiny_activation.hpp"

#ifdef __cplusplus

#include <vector>

namespace tiny
{

/* ============================================================================
 * Layer — abstract base
 * ============================================================================ */
class Layer
{
public:
    const char *name;      ///< Human-readable name (set by subclass)
    bool        trainable; ///< Whether this layer has learnable parameters

    explicit Layer(const char *name = "layer", bool trainable = false)
        : name(name), trainable(trainable) {}

    virtual ~Layer() {}

    // =========================================================================
    // Core interface
    // =========================================================================

    /**
     * @brief Forward pass.
     * @param x Input tensor (shape depends on layer type)
     * @return  Output tensor
     */
    virtual Tensor forward(const Tensor &x) = 0;

#if TINY_AI_TRAINING_ENABLED
    /**
     * @brief Backward pass.
     *        Must be called only after a corresponding forward() call.
     * @param grad_out Gradient of the loss w.r.t. this layer's output
     * @return         Gradient of the loss w.r.t. this layer's input
     */
    virtual Tensor backward(const Tensor &grad_out) = 0;

    /**
     * @brief Register all learnable (param, grad) pairs into groups.
     *        Called by Sequential / Trainer to build the full parameter list.
     */
    virtual void collect_params(std::vector<ParamGroup> &groups) {}
#endif
};

/* ============================================================================
 * Activation Layer — wraps stateless activation functions as a Layer
 * ============================================================================ */
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
    Tensor  cache_;   ///< Saves input (for ReLU/GELU) or output (for Sigmoid/Tanh/Softmax)
};

/* ============================================================================
 * Flatten Layer — reshapes [batch, ...] → [batch, flat_size]
 * ============================================================================ */
class Flatten : public Layer
{
public:
    Flatten() : Layer("flatten", false) {}

    Tensor forward(const Tensor &x) override;

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override;
#endif

private:
    int in_shape[4];  ///< Saved input shape for backward
    int in_ndim;
};

/* ============================================================================
 * GlobalAvgPool Layer — reduces [batch, seq, feat] → [batch, feat]
 *                       Used in Attention models to aggregate sequence output
 * ============================================================================ */
class GlobalAvgPool : public Layer
{
public:
    GlobalAvgPool() : Layer("global_avg_pool", false) {}

    Tensor forward(const Tensor &x) override;

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override;
#endif

private:
    int in_seq;   ///< Saved sequence length for backward
    int in_feat;  ///< Saved feature dim
    int in_batch;
};

} // namespace tiny

#endif // __cplusplus
