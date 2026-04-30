/**
 * @file tiny_norm.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief LayerNorm implementation.
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#include "tiny_norm.hpp"
#include <cmath>

#ifdef __cplusplus

namespace tiny
{

LayerNorm::LayerNorm(int feat, float epsilon)
    : Layer("layer_norm", true),
      gamma(feat), beta(feat),
      feat_(feat), eps_(epsilon)
{
    gamma.fill(1.0f);
    beta.zero();
#if TINY_AI_TRAINING_ENABLED
    dgamma = Tensor::zeros_like(gamma);
    dbeta  = Tensor::zeros_like(beta);
#endif
}

// Normalise over last dimension (feat_) independently per row
Tensor LayerNorm::forward(const Tensor &x)
{
    int rows = x.size / feat_;
    Tensor out = x.clone();

#if TINY_AI_TRAINING_ENABLED
    x_cache_ = x.clone();
    x_norm_  = Tensor::zeros_like(x);
    mean_    = Tensor(rows);
    var_     = Tensor(rows);
#endif

    for (int r = 0; r < rows; r++)
    {
        float *xr  = x.data    + r * feat_;
        float *outr = out.data  + r * feat_;

        // Compute mean
        float mu = 0.0f;
        for (int f = 0; f < feat_; f++) mu += xr[f];
        mu /= (float)feat_;

        // Compute variance
        float var = 0.0f;
        for (int f = 0; f < feat_; f++) { float d = xr[f] - mu; var += d * d; }
        var /= (float)feat_;

        float inv_std = 1.0f / sqrtf(var + eps_);

        for (int f = 0; f < feat_; f++)
        {
            float xn = (xr[f] - mu) * inv_std;
#if TINY_AI_TRAINING_ENABLED
            x_norm_.data[r * feat_ + f] = xn;
            mean_.data[r]  = mu;
            var_.data[r]   = var;
#endif
            outr[f] = gamma.data[f] * xn + beta.data[f];
        }
    }
    return out;
}

#if TINY_AI_TRAINING_ENABLED

Tensor LayerNorm::backward(const Tensor &grad_out)
{
    int rows = x_cache_.size / feat_;
    Tensor g = Tensor::zeros_like(grad_out);

    for (int r = 0; r < rows; r++)
    {
        const float *xnr   = x_norm_.data  + r * feat_;
        const float *gr    = grad_out.data  + r * feat_;
        float       *gout  = g.data         + r * feat_;

        float var     = var_.data[r];
        float inv_std = 1.0f / sqrtf(var + eps_);

        // dgamma += grad_out * x_norm
        // dbeta  += grad_out
        for (int f = 0; f < feat_; f++)
        {
            dgamma.data[f] += gr[f] * xnr[f];
            dbeta.data[f]  += gr[f];
        }

        // Backward through normalisation (standard LayerNorm gradient)
        // dx = (1/feat) * inv_std * (feat*grad_gamma - sum(grad_gamma) - x_norm * sum(grad_gamma * x_norm))
        float sum_g  = 0.0f, sum_gxn = 0.0f;
        for (int f = 0; f < feat_; f++)
        {
            float gg = gr[f] * gamma.data[f];
            sum_g   += gg;
            sum_gxn += gg * xnr[f];
        }
        float inv_feat = 1.0f / (float)feat_;
        for (int f = 0; f < feat_; f++)
        {
            float gg = gr[f] * gamma.data[f];
            gout[f] = inv_std * (gg - inv_feat * sum_g - inv_feat * xnr[f] * sum_gxn);
        }
    }
    return g;
}

void LayerNorm::collect_params(std::vector<ParamGroup> &groups)
{
    groups.push_back({&gamma, &dgamma});
    groups.push_back({&beta,  &dbeta});
}

#endif // TINY_AI_TRAINING_ENABLED

} // namespace tiny

#endif // __cplusplus
