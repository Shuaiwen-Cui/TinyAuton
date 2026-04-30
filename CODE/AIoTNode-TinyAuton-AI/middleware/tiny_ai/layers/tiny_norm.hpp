/**
 * @file tiny_norm.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Normalisation layers for tiny_ai.
 *
 *  LayerNorm   — normalises over last (feature) dimension; no running stats.
 *  BatchNorm1D — normalises over batch for [batch, feat] inputs.
 *                Compatible with PyTorch nn.BatchNorm1d.
 *  BatchNorm2D — normalises over batch + spatial dims for [N, C, ...] inputs.
 *                Compatible with PyTorch nn.BatchNorm2d / nn.BatchNorm1d (3-D).
 *
 * @version 1.1
 * @date 2026-04-30
 * @copyright Copyright (c) 2026
 */

#pragma once

#include "tiny_layer.hpp"

#ifdef __cplusplus

namespace tiny
{

class LayerNorm : public Layer
{
public:
    Tensor gamma;   ///< Scale parameter  [feat] — initialised to 1
    Tensor beta;    ///< Shift parameter  [feat] — initialised to 0

#if TINY_AI_TRAINING_ENABLED
    Tensor dgamma;
    Tensor dbeta;
#endif

    /**
     * @param feat     Size of the normalised dimension (last dim of input)
     * @param epsilon  Numerical stability constant (default 1e-5)
     */
    LayerNorm(int feat, float epsilon = 1e-5f);

    Tensor forward(const Tensor &x) override;

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override;
    void   collect_params(std::vector<ParamGroup> &groups) override;
#endif

private:
    int   feat_;
    float eps_;

#if TINY_AI_TRAINING_ENABLED
    Tensor x_norm_;   ///< Cached normalised input
    Tensor x_cache_;  ///< Cached input
    Tensor mean_;     ///< Per-row mean    [rows]
    Tensor var_;      ///< Per-row variance [rows]
#endif
};

/* ============================================================================
 * BatchNorm1D — Batch Normalisation for Dense / MLP layers
 * ============================================================================ */
/**
 *  input:  [batch, feat]
 *  output: [batch, feat]
 *
 *  Inference (default): uses running_mean / running_var loaded from
 *                       PC-trained weights — no batch statistics needed.
 *  Training:            computes per-feature batch statistics and updates
 *                       running stats with EMA momentum.
 */
class BatchNorm1D : public Layer
{
public:
    Tensor gamma;        ///< Scale     [feat] — initialised to 1
    Tensor beta;         ///< Shift     [feat] — initialised to 0
    Tensor running_mean; ///< Running mean     [feat] — initialised to 0
    Tensor running_var;  ///< Running variance [feat] — initialised to 1

#if TINY_AI_TRAINING_ENABLED
    Tensor dgamma;
    Tensor dbeta;
#endif

    /**
     * @param feat      Number of features
     * @param momentum  EMA momentum for running-stat update (PyTorch default 0.1)
     * @param epsilon   Numerical-stability constant (default 1e-5)
     */
    BatchNorm1D(int feat, float momentum = 0.1f, float epsilon = 1e-5f);

    void set_training(bool mode) override { training_mode_ = mode; }
    bool is_training() const              { return training_mode_; }

    Tensor forward(const Tensor &x) override;

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override;
    void   collect_params(std::vector<ParamGroup> &groups) override;
#endif

private:
    int   feat_;
    float momentum_;
    float eps_;
    bool  training_mode_; ///< false by default (inference uses running stats)

#if TINY_AI_TRAINING_ENABLED
    Tensor x_norm_;  ///< Cached normalised input [batch, feat]
    Tensor inv_std_; ///< Cached per-feature inv-std    [feat]
#endif
};

/* ============================================================================
 * BatchNorm2D — Batch Normalisation for channel-first Conv outputs
 * ============================================================================ */
/**
 *  input:  [batch, channels, length]           (Conv1D output)
 *       or [batch, channels, height, width]     (Conv2D output)
 *  output: same shape as input
 *
 *  Normalises over all axes except the channel axis (axis 1).
 *  Inference (default): uses running_mean / running_var loaded from
 *                       PC-trained weights.
 */
class BatchNorm2D : public Layer
{
public:
    Tensor gamma;        ///< Scale     [channels] — initialised to 1
    Tensor beta;         ///< Shift     [channels] — initialised to 0
    Tensor running_mean; ///< Running mean     [channels] — initialised to 0
    Tensor running_var;  ///< Running variance [channels] — initialised to 1

#if TINY_AI_TRAINING_ENABLED
    Tensor dgamma;
    Tensor dbeta;
#endif

    /**
     * @param num_channels  Number of channels (C in [N, C, ...])
     * @param momentum      EMA momentum for running-stat update (default 0.1)
     * @param epsilon       Numerical-stability constant (default 1e-5)
     */
    BatchNorm2D(int num_channels, float momentum = 0.1f, float epsilon = 1e-5f);

    void set_training(bool mode) override { training_mode_ = mode; }
    bool is_training() const              { return training_mode_; }

    Tensor forward(const Tensor &x) override;

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override;
    void   collect_params(std::vector<ParamGroup> &groups) override;
#endif

private:
    int   num_channels_;
    float momentum_;
    float eps_;
    bool  training_mode_; ///< false by default (inference uses running stats)

#if TINY_AI_TRAINING_ENABLED
    Tensor x_norm_;  ///< Cached normalised input, same shape as input
    Tensor inv_std_; ///< Cached per-channel inv-std [channels]
#endif
};

} // namespace tiny

#endif // __cplusplus
