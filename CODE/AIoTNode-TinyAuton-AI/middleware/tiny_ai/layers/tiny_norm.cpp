/**
 * @file tiny_norm.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief LayerNorm, BatchNorm1D, BatchNorm2D implementations.
 * @version 1.1
 * @date 2026-04-30
 * @copyright Copyright (c) 2026
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

/* ============================================================================
 * BatchNorm1D
 * ============================================================================ */

BatchNorm1D::BatchNorm1D(int feat, float momentum, float epsilon)
    : Layer("batch_norm_1d", true),
      gamma(feat), beta(feat),
      running_mean(feat), running_var(feat),
      feat_(feat), momentum_(momentum), eps_(epsilon),
      training_mode_(false)
{
    gamma.fill(1.0f);
    beta.zero();
    running_mean.zero();
    running_var.fill(1.0f);
#if TINY_AI_TRAINING_ENABLED
    dgamma = Tensor::zeros_like(gamma);
    dbeta  = Tensor::zeros_like(beta);
#endif
}

Tensor BatchNorm1D::forward(const Tensor &x)
{
    int batch = x.shape[0];
    Tensor out = x.clone();

#if TINY_AI_TRAINING_ENABLED
    if (training_mode_)
    {
        x_norm_  = Tensor::zeros_like(x);
        inv_std_ = Tensor(feat_);

        for (int f = 0; f < feat_; f++)
        {
            float mu = 0.0f;
            for (int n = 0; n < batch; n++) mu += x.at(n, f);
            mu /= (float)batch;

            float var = 0.0f;
            for (int n = 0; n < batch; n++) { float d = x.at(n, f) - mu; var += d * d; }
            var /= (float)batch;

            float inv_std = 1.0f / sqrtf(var + eps_);
            inv_std_.data[f] = inv_std;

            for (int n = 0; n < batch; n++)
            {
                float xn = (x.at(n, f) - mu) * inv_std;
                x_norm_.at(n, f) = xn;
                out.at(n, f) = gamma.data[f] * xn + beta.data[f];
            }

            running_mean.data[f] = (1.0f - momentum_) * running_mean.data[f] + momentum_ * mu;
            running_var.data[f]  = (1.0f - momentum_) * running_var.data[f]  + momentum_ * var;
        }
        return out;
    }
#endif

    // Inference: fuse scale+shift into two constants per feature
    for (int f = 0; f < feat_; f++)
    {
        float scale = gamma.data[f] / sqrtf(running_var.data[f] + eps_);
        float shift = beta.data[f] - scale * running_mean.data[f];
        for (int n = 0; n < batch; n++)
            out.at(n, f) = scale * x.at(n, f) + shift;
    }
    return out;
}

#if TINY_AI_TRAINING_ENABLED

Tensor BatchNorm1D::backward(const Tensor &grad_out)
{
    int   batch  = x_norm_.shape[0];
    float inv_N  = 1.0f / (float)batch;
    Tensor g = Tensor::zeros_like(grad_out);

    for (int f = 0; f < feat_; f++)
    {
        float inv_std = inv_std_.data[f];
        float gf      = gamma.data[f];
        float dg_sum  = 0.0f, db_sum = 0.0f;
        float dxh_sum = 0.0f, dxh_xn = 0.0f;

        for (int n = 0; n < batch; n++)
        {
            float go = grad_out.at(n, f);
            float xn = x_norm_.at(n, f);
            dg_sum  += go * xn;
            db_sum  += go;
            float dxh = go * gf;
            dxh_sum += dxh;
            dxh_xn  += dxh * xn;
        }

        dgamma.data[f] += dg_sum;
        dbeta.data[f]  += db_sum;

        for (int n = 0; n < batch; n++)
        {
            float dxh = grad_out.at(n, f) * gf;
            float xn  = x_norm_.at(n, f);
            g.at(n, f) = inv_std * inv_N * ((float)batch * dxh - dxh_sum - xn * dxh_xn);
        }
    }
    return g;
}

void BatchNorm1D::collect_params(std::vector<ParamGroup> &groups)
{
    groups.push_back({&gamma, &dgamma});
    groups.push_back({&beta,  &dbeta});
}

#endif // TINY_AI_TRAINING_ENABLED

/* ============================================================================
 * BatchNorm2D
 * ============================================================================ */

BatchNorm2D::BatchNorm2D(int num_channels, float momentum, float epsilon)
    : Layer("batch_norm_2d", true),
      gamma(num_channels), beta(num_channels),
      running_mean(num_channels), running_var(num_channels),
      num_channels_(num_channels), momentum_(momentum), eps_(epsilon),
      training_mode_(false)
{
    gamma.fill(1.0f);
    beta.zero();
    running_mean.zero();
    running_var.fill(1.0f);
#if TINY_AI_TRAINING_ENABLED
    dgamma = Tensor::zeros_like(gamma);
    dbeta  = Tensor::zeros_like(beta);
#endif
}

// Input layout: [N, C, S...] where S = L (3D) or H*W (4D)
// Elements for channel c: data[n*C*S + c*S + s]  for n in [0,N), s in [0,S)
Tensor BatchNorm2D::forward(const Tensor &x)
{
    int N = x.shape[0];
    int C = x.shape[1];
    int S = x.size / (N * C);   // spatial elements per (sample, channel) pair
    int M = N * S;               // total elements per channel

    Tensor out = x.clone();

#if TINY_AI_TRAINING_ENABLED
    if (training_mode_)
    {
        x_norm_  = Tensor::zeros_like(x);
        inv_std_ = Tensor(C);

        for (int c = 0; c < C; c++)
        {
            float mu = 0.0f;
            for (int n = 0; n < N; n++)
                for (int s = 0; s < S; s++)
                    mu += x.data[n * C * S + c * S + s];
            mu /= (float)M;

            float var = 0.0f;
            for (int n = 0; n < N; n++)
                for (int s = 0; s < S; s++) {
                    float d = x.data[n * C * S + c * S + s] - mu;
                    var += d * d;
                }
            var /= (float)M;

            float inv_std = 1.0f / sqrtf(var + eps_);
            inv_std_.data[c] = inv_std;

            for (int n = 0; n < N; n++)
                for (int s = 0; s < S; s++) {
                    int   idx = n * C * S + c * S + s;
                    float xn  = (x.data[idx] - mu) * inv_std;
                    x_norm_.data[idx] = xn;
                    out.data[idx] = gamma.data[c] * xn + beta.data[c];
                }

            running_mean.data[c] = (1.0f - momentum_) * running_mean.data[c] + momentum_ * mu;
            running_var.data[c]  = (1.0f - momentum_) * running_var.data[c]  + momentum_ * var;
        }
        return out;
    }
#endif

    // Inference: fuse scale+shift into two constants per channel
    for (int c = 0; c < C; c++)
    {
        float scale = gamma.data[c] / sqrtf(running_var.data[c] + eps_);
        float shift = beta.data[c] - scale * running_mean.data[c];
        for (int n = 0; n < N; n++)
            for (int s = 0; s < S; s++) {
                int idx = n * C * S + c * S + s;
                out.data[idx] = scale * x.data[idx] + shift;
            }
    }
    return out;
}

#if TINY_AI_TRAINING_ENABLED

Tensor BatchNorm2D::backward(const Tensor &grad_out)
{
    int   N     = x_norm_.shape[0];
    int   C     = x_norm_.shape[1];
    int   S     = x_norm_.size / (N * C);
    int   M     = N * S;
    float inv_M = 1.0f / (float)M;

    Tensor g = Tensor::zeros_like(grad_out);

    for (int c = 0; c < C; c++)
    {
        float inv_std = inv_std_.data[c];
        float gf      = gamma.data[c];
        float dg_sum  = 0.0f, db_sum = 0.0f;
        float dxh_sum = 0.0f, dxh_xn = 0.0f;

        for (int n = 0; n < N; n++)
            for (int s = 0; s < S; s++) {
                int   idx = n * C * S + c * S + s;
                float go  = grad_out.data[idx];
                float xn  = x_norm_.data[idx];
                dg_sum  += go * xn;
                db_sum  += go;
                float dxh = go * gf;
                dxh_sum += dxh;
                dxh_xn  += dxh * xn;
            }

        dgamma.data[c] += dg_sum;
        dbeta.data[c]  += db_sum;

        for (int n = 0; n < N; n++)
            for (int s = 0; s < S; s++) {
                int   idx = n * C * S + c * S + s;
                float dxh = grad_out.data[idx] * gf;
                float xn  = x_norm_.data[idx];
                g.data[idx] = inv_std * inv_M * ((float)M * dxh - dxh_sum - xn * dxh_xn);
            }
    }
    return g;
}

void BatchNorm2D::collect_params(std::vector<ParamGroup> &groups)
{
    groups.push_back({&gamma, &dgamma});
    groups.push_back({&beta,  &dbeta});
}

#endif // TINY_AI_TRAINING_ENABLED

} // namespace tiny

#endif // __cplusplus
