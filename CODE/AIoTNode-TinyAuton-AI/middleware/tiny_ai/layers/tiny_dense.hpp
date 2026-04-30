/**
 * @file tiny_dense.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Fully-connected (Dense / Linear) layer for tiny_ai.
 *        output = input × weight^T + bias
 *        input:  [batch, in_features]
 *        output: [batch, out_features]
 * @version 1.0
 * @date 2026-04-30
 * @copyright Copyright (c) 2026
 */

#pragma once

#include "tiny_layer.hpp"

#ifdef __cplusplus

namespace tiny
{

class Dense : public Layer
{
public:
    // =========================================================================
    // Learnable parameters
    // =========================================================================
    Tensor weight;   ///< [out_features, in_features]
    Tensor bias;     ///< [out_features]

#if TINY_AI_TRAINING_ENABLED
    Tensor dweight;  ///< Gradient of weight
    Tensor dbias;    ///< Gradient of bias
#endif

    // =========================================================================
    // Constructor
    // =========================================================================
    /**
     * @param in_features   Number of input features
     * @param out_features  Number of output features
     * @param use_bias      Whether to add a learnable bias (default true)
     *
     * Weights are initialised with Xavier-uniform (Glorot) initialisation.
     * Bias is zero-initialised.
     */
    Dense(int in_features, int out_features, bool use_bias = true);

    // =========================================================================
    // Layer interface
    // =========================================================================
    Tensor forward(const Tensor &x) override;

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override;
    void   collect_params(std::vector<ParamGroup> &groups) override;
#endif

    // =========================================================================
    // Accessors
    // =========================================================================
    int in_features()  const { return in_feat_; }
    int out_features() const { return out_feat_; }

private:
    int  in_feat_;
    int  out_feat_;
    bool use_bias_;

#if TINY_AI_TRAINING_ENABLED
    Tensor x_cache_;  ///< Cached input for backward pass
#endif

    void xavier_init();
};

} // namespace tiny

#endif // __cplusplus
