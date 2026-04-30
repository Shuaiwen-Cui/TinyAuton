/**
 * @file tiny_attention.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Multi-Head Self-Attention implementation.
 * @version 1.0
 * @date 2026-04-30
 * @copyright Copyright (c) 2026
 */

#include "tiny_attention.hpp"
#include "tiny_activation.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <initializer_list>

#ifdef __cplusplus

namespace tiny
{

// ============================================================================
// Constructor
// ============================================================================

Attention::Attention(int embed_dim, int num_heads, bool use_bias)
    : Layer("attention", true),
      Wq(embed_dim, embed_dim), Wk(embed_dim, embed_dim),
      Wv(embed_dim, embed_dim), Wo(embed_dim, embed_dim),
      bq(use_bias ? embed_dim : 0), bk(use_bias ? embed_dim : 0),
      bv(use_bias ? embed_dim : 0), bo(use_bias ? embed_dim : 0),
      embed_dim_(embed_dim), num_heads_(num_heads),
      head_dim_(embed_dim / num_heads), use_bias_(use_bias)
{
    // Xavier init for all projection weights
    float limit = sqrtf(6.0f / (float)(embed_dim + embed_dim));
    for (Tensor *W : {&Wq, &Wk, &Wv, &Wo})
        for (int i = 0; i < W->size; i++)
        {
            float r = (float)rand() / (float)RAND_MAX;
            W->data[i] = (r * 2.0f - 1.0f) * limit;
        }

#if TINY_AI_TRAINING_ENABLED
    dWq = Tensor::zeros_like(Wq); dWk = Tensor::zeros_like(Wk);
    dWv = Tensor::zeros_like(Wv); dWo = Tensor::zeros_like(Wo);
    if (use_bias_)
    {
        dbq = Tensor::zeros_like(bq); dbk = Tensor::zeros_like(bk);
        dbv = Tensor::zeros_like(bv); dbo = Tensor::zeros_like(bo);
    }
#endif
}

// ============================================================================
// Helpers
// ============================================================================

// Project x [B, S, E] through W [E, E] → out [B, S, E]
// out[b, s, o] = sum_e W[o, e] * x[b, s, e]  (+ bias if add_bias)
Tensor Attention::project(const Tensor &x, const Tensor &W,
                           const Tensor &b, bool add_bias) const
{
    int B = x.shape[0], S = x.shape[1], E = x.shape[2];
    Tensor out(B, S, E);

    for (int bs = 0; bs < B * S; bs++)
    {
        const float *xr = x.data + bs * E;
        float       *or_ = out.data + bs * E;
        for (int o = 0; o < E; o++)
        {
            float sum = add_bias ? b.data[o] : 0.0f;
            const float *wr = W.data + o * E;
            for (int e = 0; e < E; e++) sum += wr[e] * xr[e];
            or_[o] = sum;
        }
    }
    return out;
}

// Scaled dot-product attention for a single head
// Q, K, V each sliced to [B, S, D] (starting at head offset)
void Attention::sdp_attention(const Tensor &Q, const Tensor &K, const Tensor &V,
                               int B, int S, int D,
                               Tensor &out, Tensor &A_out) const
{
    float scale = 1.0f / sqrtf((float)D);

    for (int b = 0; b < B; b++)
    {
        // score [S, S] = Q[b] @ K[b]^T * scale
        for (int s1 = 0; s1 < S; s1++)
            for (int s2 = 0; s2 < S; s2++)
            {
                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += Q.at(b, s1, d) * K.at(b, s2, d);
                A_out.at(b, s1, s2) = dot * scale;
            }

        // softmax over last dim (s2)
        for (int s1 = 0; s1 < S; s1++)
        {
            float mx = A_out.at(b, s1, 0);
            for (int s2 = 1; s2 < S; s2++)
                if (A_out.at(b, s1, s2) > mx) mx = A_out.at(b, s1, s2);

            float sum = 0.0f;
            for (int s2 = 0; s2 < S; s2++)
            {
                A_out.at(b, s1, s2) = expf(A_out.at(b, s1, s2) - mx);
                sum += A_out.at(b, s1, s2);
            }
            float inv = 1.0f / (sum + TINY_MATH_MIN_DENOMINATOR);
            for (int s2 = 0; s2 < S; s2++) A_out.at(b, s1, s2) *= inv;
        }

        // context [b, s1, d] = sum_s2  A[b, s1, s2] * V[b, s2, d]
        for (int s1 = 0; s1 < S; s1++)
            for (int d = 0; d < D; d++)
            {
                float val = 0.0f;
                for (int s2 = 0; s2 < S; s2++)
                    val += A_out.at(b, s1, s2) * V.at(b, s2, d);
                out.at(b, s1, d) = val;
            }
    }
}

// ============================================================================
// Forward
// ============================================================================

Tensor Attention::forward(const Tensor &x)
{
    int B = x.shape[0], S = x.shape[1], E = x.shape[2];

    // Full projections: [B, S, E]
    Tensor Q = project(x, Wq, bq, use_bias_);
    Tensor K = project(x, Wk, bk, use_bias_);
    Tensor V = project(x, Wv, bv, use_bias_);

#if TINY_AI_TRAINING_ENABLED
    x_cache_ = x.clone();
    Q_cache_ = Q.clone();
    K_cache_ = K.clone();
    V_cache_ = V.clone();
    A_cache_ = Tensor(B * num_heads_, S, S);
#endif

    // Concatenated context: [B, S, E]
    Tensor ctx(B, S, E);

    for (int h = 0; h < num_heads_; h++)
    {
        int offset = h * head_dim_;

        // Slice Q, K, V for head h → [B, S, head_dim]
        Tensor Qh(B, S, head_dim_), Kh(B, S, head_dim_), Vh(B, S, head_dim_);
        Tensor Ah(B, S, S);

        for (int b = 0; b < B; b++)
            for (int s = 0; s < S; s++)
                for (int d = 0; d < head_dim_; d++)
                {
                    Qh.at(b, s, d) = Q.at(b, s, offset + d);
                    Kh.at(b, s, d) = K.at(b, s, offset + d);
                    Vh.at(b, s, d) = V.at(b, s, offset + d);
                }

        Tensor ctx_h(B, S, head_dim_);
        sdp_attention(Qh, Kh, Vh, B, S, head_dim_, ctx_h, Ah);

#if TINY_AI_TRAINING_ENABLED
        // Store attention weights for head h
        for (int b = 0; b < B; b++)
            for (int s1 = 0; s1 < S; s1++)
                for (int s2 = 0; s2 < S; s2++)
                    A_cache_.at(b * num_heads_ + h, s1, s2) = Ah.at(b, s1, s2);
#endif

        // Write head output into ctx
        for (int b = 0; b < B; b++)
            for (int s = 0; s < S; s++)
                for (int d = 0; d < head_dim_; d++)
                    ctx.at(b, s, offset + d) = ctx_h.at(b, s, d);
    }

#if TINY_AI_TRAINING_ENABLED
    ctx_cache_ = ctx.clone();
#endif

    // Output projection: ctx @ Wo^T + bo
    return project(ctx, Wo, bo, use_bias_);
}

// ============================================================================
// Backward
// ============================================================================
#if TINY_AI_TRAINING_ENABLED

Tensor Attention::backward(const Tensor &grad_out)
{
    int B = x_cache_.shape[0], S = x_cache_.shape[1], E = x_cache_.shape[2];

    // --- Backward through output projection (Wo) ---
    // grad_out: [B, S, E]   Wo: [E, E]
    // dWo += ctx^T @ grad_out (summed over B*S)
    // d_ctx = grad_out @ Wo
    Tensor d_ctx(B, S, E);

    for (int bs = 0; bs < B * S; bs++)
    {
        const float *gor = grad_out.data + bs * E;
        const float *cr  = ctx_cache_.data + bs * E;
        float       *dcr = d_ctx.data     + bs * E;

        for (int o = 0; o < E; o++)
        {
            if (use_bias_) dbo.data[o] += gor[o];
            float *dwr = dWo.data + o * E;
            for (int e = 0; e < E; e++) dwr[e] += gor[o] * cr[e];
        }
        for (int e = 0; e < E; e++)
        {
            float sum = 0.0f;
            for (int o = 0; o < E; o++) sum += gor[o] * Wo.at(o, e);
            dcr[e] = sum;
        }
    }

    // --- Backward through multi-head attention and projections ---
    Tensor dQ(B, S, E), dK(B, S, E), dV(B, S, E);

    for (int h = 0; h < num_heads_; h++)
    {
        int offset = h * head_dim_;
        int D = head_dim_;

        // Recover head slices
        Tensor Qh(B, S, D), Kh(B, S, D), Vh(B, S, D), Ah(B, S, S);
        Tensor dctx_h(B, S, D);
        for (int b = 0; b < B; b++)
            for (int s = 0; s < S; s++)
            {
                for (int d = 0; d < D; d++)
                {
                    Qh.at(b, s, d)    = Q_cache_.at(b, s, offset + d);
                    Kh.at(b, s, d)    = K_cache_.at(b, s, offset + d);
                    Vh.at(b, s, d)    = V_cache_.at(b, s, offset + d);
                    dctx_h.at(b, s, d) = d_ctx.at(b, s, offset + d);
                }
                for (int s2 = 0; s2 < S; s2++)
                    Ah.at(b, s, s2) = A_cache_.at(b * num_heads_ + h, s, s2);
            }

        float scale = 1.0f / sqrtf((float)D);

        Tensor dQh(B, S, D), dKh(B, S, D), dVh(B, S, D);

        for (int b = 0; b < B; b++)
        {
            // dV += A^T @ dctx_h
            for (int s2 = 0; s2 < S; s2++)
                for (int d = 0; d < D; d++)
                {
                    float acc = 0.0f;
                    for (int s1 = 0; s1 < S; s1++) acc += Ah.at(b, s1, s2) * dctx_h.at(b, s1, d);
                    dVh.at(b, s2, d) = acc;
                }

            // dA = dctx_h @ V^T
            Tensor dA(S, S);
            for (int s1 = 0; s1 < S; s1++)
                for (int s2 = 0; s2 < S; s2++)
                {
                    float acc = 0.0f;
                    for (int d = 0; d < D; d++) acc += dctx_h.at(b, s1, d) * Vh.at(b, s2, d);
                    dA.at(s1, s2) = acc;
                }

            // Softmax backward: dS = A * (dA - rowsum(dA * A))
            for (int s1 = 0; s1 < S; s1++)
            {
                float dot = 0.0f;
                for (int s2 = 0; s2 < S; s2++) dot += dA.at(s1, s2) * Ah.at(b, s1, s2);
                for (int s2 = 0; s2 < S; s2++)
                    dA.at(s1, s2) = Ah.at(b, s1, s2) * (dA.at(s1, s2) - dot) * scale;
            }

            // dQ += dA @ K
            for (int s1 = 0; s1 < S; s1++)
                for (int d = 0; d < D; d++)
                {
                    float acc = 0.0f;
                    for (int s2 = 0; s2 < S; s2++) acc += dA.at(s1, s2) * Kh.at(b, s2, d);
                    dQh.at(b, s1, d) = acc;
                }

            // dK += dA^T @ Q
            for (int s2 = 0; s2 < S; s2++)
                for (int d = 0; d < D; d++)
                {
                    float acc = 0.0f;
                    for (int s1 = 0; s1 < S; s1++) acc += dA.at(s1, s2) * Qh.at(b, s1, d);
                    dKh.at(b, s2, d) = acc;
                }
        }

        // Write back to full-dim grad tensors
        for (int b = 0; b < B; b++)
            for (int s = 0; s < S; s++)
                for (int d = 0; d < D; d++)
                {
                    dQ.at(b, s, offset + d) += dQh.at(b, s, d);
                    dK.at(b, s, offset + d) += dKh.at(b, s, d);
                    dV.at(b, s, offset + d) += dVh.at(b, s, d);
                }
    }

    // --- Backward through Wq, Wk, Wv projections ---
    // dW += x^T @ dProj   (per row of B*S)
    // dx  += dProj @ W

    Tensor dx(B, S, E);

    auto proj_backward = [&](const Tensor &dP, Tensor &dW, Tensor &db)
    {
        for (int bs = 0; bs < B * S; bs++)
        {
            const float *xr  = x_cache_.data + bs * E;
            const float *dpr = dP.data       + bs * E;
            for (int o = 0; o < E; o++)
            {
                if (use_bias_) db.data[o] += dpr[o];
                float *dwr = dW.data + o * E;
                for (int e = 0; e < E; e++) dwr[e] += dpr[o] * xr[e];
            }
        }
    };

    proj_backward(dQ, dWq, dbq);
    proj_backward(dK, dWk, dbk);
    proj_backward(dV, dWv, dbv);

    // dx = dQ @ Wq + dK @ Wk + dV @ Wv
    for (int bs = 0; bs < B * S; bs++)
    {
        float *dxr = dx.data + bs * E;
        for (const auto &pair : std::initializer_list<std::pair<const Tensor *, const Tensor *>>
                                { {&dQ, &Wq}, {&dK, &Wk}, {&dV, &Wv} })
        {
            const float *dpr = pair.first->data  + bs * E;
            const Tensor &W  = *pair.second;
            for (int e = 0; e < E; e++)
            {
                float acc = 0.0f;
                for (int o = 0; o < E; o++) acc += dpr[o] * W.at(o, e);
                dxr[e] += acc;
            }
        }
    }

    return dx;
}

void Attention::collect_params(std::vector<ParamGroup> &groups)
{
    groups.push_back({&Wq, &dWq}); groups.push_back({&Wk, &dWk});
    groups.push_back({&Wv, &dWv}); groups.push_back({&Wo, &dWo});
    if (use_bias_)
    {
        groups.push_back({&bq, &dbq}); groups.push_back({&bk, &dbk});
        groups.push_back({&bv, &dbv}); groups.push_back({&bo, &dbo});
    }
}

#endif // TINY_AI_TRAINING_ENABLED

} // namespace tiny

#endif // __cplusplus
