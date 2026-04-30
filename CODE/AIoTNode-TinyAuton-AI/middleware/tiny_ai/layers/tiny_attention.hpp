/**
 * @file tiny_attention.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Multi-Head Self-Attention layer for tiny_ai.
 *
 *  input:  [batch, seq_len, embed_dim]
 *  output: [batch, seq_len, embed_dim]
 *
 *  Architecture:
 *    For each head h (head_dim = embed_dim / num_heads):
 *      Q_h = x @ Wq_h^T   [batch, seq, head_dim]
 *      K_h = x @ Wk_h^T   [batch, seq, head_dim]
 *      V_h = x @ Wv_h^T   [batch, seq, head_dim]
 *      score_h = Q_h @ K_h^T / sqrt(head_dim)
 *      attn_h  = softmax(score_h)
 *      out_h   = attn_h @ V_h
 *    concat all heads → [batch, seq, embed_dim]
 *    output = concat @ Wo^T
 *
 *  Weights are merged: Wq, Wk, Wv each have shape [embed_dim, embed_dim];
 *  they are split into num_heads slices at forward time (no extra copies).
 *
 * @version 1.0
 * @date 2026-04-30
 * @copyright Copyright (c) 2026
 */

#pragma once

#include "tiny_layer.hpp"

#ifdef __cplusplus

namespace tiny
{

class Attention : public Layer
{
public:
    // =========================================================================
    // Learnable weights
    // =========================================================================
    Tensor Wq;   ///< [embed_dim, embed_dim]
    Tensor Wk;   ///< [embed_dim, embed_dim]
    Tensor Wv;   ///< [embed_dim, embed_dim]
    Tensor Wo;   ///< [embed_dim, embed_dim]  output projection

    Tensor bq, bk, bv, bo;  ///< Bias vectors [embed_dim]

#if TINY_AI_TRAINING_ENABLED
    Tensor dWq, dWk, dWv, dWo;
    Tensor dbq, dbk, dbv, dbo;
#endif

    /**
     * @param embed_dim  Total embedding dimension (must be divisible by num_heads)
     * @param num_heads  Number of attention heads (default 1)
     * @param use_bias   Whether to add projection biases (default true)
     */
    Attention(int embed_dim, int num_heads = 1, bool use_bias = true);

    Tensor forward(const Tensor &x) override;

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override;
    void   collect_params(std::vector<ParamGroup> &groups) override;
#endif

    int embed_dim()  const { return embed_dim_; }
    int num_heads()  const { return num_heads_; }
    int head_dim()   const { return head_dim_; }

private:
    int  embed_dim_;
    int  num_heads_;
    int  head_dim_;    // embed_dim_ / num_heads_
    bool use_bias_;

#if TINY_AI_TRAINING_ENABLED
    // Caches needed for backward
    Tensor x_cache_;     // input  [B, S, E]
    Tensor Q_cache_;     // all-head Q projected [B, S, E]
    Tensor K_cache_;     // all-head K projected [B, S, E]
    Tensor V_cache_;     // all-head V projected [B, S, E]
    Tensor A_cache_;     // softmax attention weights [B*H, S, S]
    Tensor ctx_cache_;   // weighted context before Wo  [B, S, E]
#endif

    // Project x [B,S,E] through W [E,E] + b [E] → out [B,S,E]
    Tensor project(const Tensor &x, const Tensor &W, const Tensor &b, bool add_bias) const;

    // Single-head scaled dot-product attention
    // Q,K,V: [B, S, D]  →  out [B, S, D],  writes attn weights to A_out [B, S, S]
    void sdp_attention(const Tensor &Q, const Tensor &K, const Tensor &V,
                       int B, int S, int D,
                       Tensor &out, Tensor &A_out) const;
};

} // namespace tiny

#endif // __cplusplus
