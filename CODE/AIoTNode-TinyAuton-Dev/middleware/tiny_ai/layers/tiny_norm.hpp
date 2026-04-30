/**
 * @file tiny_norm.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief LayerNorm for tiny_ai.
 *        Preferred over BatchNorm for MCU inference (no running-stats dependency).
 *        Normalises over the last dimension (features).
 *
 *  input:  [batch, ..., feat]
 *  output: [batch, ..., feat]  (normalised + affine γ,β)
 *
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
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

} // namespace tiny

#endif // __cplusplus
