/**
 * @file tiny_pool.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Pooling layer implementations.
 * @version 1.0
 * @date 2026-04-30
 * @copyright Copyright (c) 2026
 */

#include "tiny_pool.hpp"
#include <cfloat>
#include <cstring>

#ifdef __cplusplus

namespace tiny
{

// ============================================================================
// MaxPool1D
// ============================================================================

MaxPool1D::MaxPool1D(int kernel_size, int stride)
    : Layer("max_pool1d", false),
      kernel_(kernel_size),
      stride_(stride < 0 ? kernel_size : stride)
{}

Tensor MaxPool1D::forward(const Tensor &x)
{
    // x: [batch, channels, length]
    int B   = x.shape[0];
    int C   = x.shape[1];
    int Lin = x.shape[2];
    int Lo  = (Lin - kernel_) / stride_ + 1;

    Tensor out(B, C, Lo);

#if TINY_AI_TRAINING_ENABLED
    in_batch_ = B; in_ch_ = C; in_len_ = Lin;
    mask_ = Tensor(B, C, Lo);  // stores argmax position as float
#endif

    for (int b = 0; b < B; b++)
        for (int c = 0; c < C; c++)
            for (int l = 0; l < Lo; l++)
            {
                int start = l * stride_;
                float mx  = -FLT_MAX;
                int   idx = start;
                for (int k = 0; k < kernel_ && start + k < Lin; k++)
                {
                    float v = x.at(b, c, start + k);
                    if (v > mx) { mx = v; idx = start + k; }
                }
                out.at(b, c, l) = mx;
#if TINY_AI_TRAINING_ENABLED
                mask_.at(b, c, l) = (float)idx;
#endif
            }
    return out;
}

#if TINY_AI_TRAINING_ENABLED

Tensor MaxPool1D::backward(const Tensor &grad_out)
{
    int B   = in_batch_;
    int C   = in_ch_;
    int Lin = in_len_;
    int Lo  = mask_.shape[2];

    Tensor g(B, C, Lin);
    for (int b = 0; b < B; b++)
        for (int c = 0; c < C; c++)
            for (int l = 0; l < Lo; l++)
            {
                int idx = (int)mask_.at(b, c, l);
                g.at(b, c, idx) += grad_out.at(b, c, l);
            }
    return g;
}

#endif

// ============================================================================
// AvgPool1D
// ============================================================================

AvgPool1D::AvgPool1D(int kernel_size, int stride)
    : Layer("avg_pool1d", false),
      kernel_(kernel_size),
      stride_(stride < 0 ? kernel_size : stride)
{}

Tensor AvgPool1D::forward(const Tensor &x)
{
    int B   = x.shape[0];
    int C   = x.shape[1];
    int Lin = x.shape[2];
    int Lo  = (Lin - kernel_) / stride_ + 1;

    Tensor out(B, C, Lo);

#if TINY_AI_TRAINING_ENABLED
    in_batch_ = B; in_ch_ = C; in_len_ = Lin;
#endif

    float inv_k = 1.0f / (float)kernel_;
    for (int b = 0; b < B; b++)
        for (int c = 0; c < C; c++)
            for (int l = 0; l < Lo; l++)
            {
                float sum = 0.0f;
                int start = l * stride_;
                for (int k = 0; k < kernel_ && start + k < Lin; k++)
                    sum += x.at(b, c, start + k);
                out.at(b, c, l) = sum * inv_k;
            }
    return out;
}

#if TINY_AI_TRAINING_ENABLED

Tensor AvgPool1D::backward(const Tensor &grad_out)
{
    int B   = in_batch_;
    int C   = in_ch_;
    int Lin = in_len_;
    int Lo  = grad_out.shape[2];

    Tensor g(B, C, Lin);
    float inv_k = 1.0f / (float)kernel_;
    for (int b = 0; b < B; b++)
        for (int c = 0; c < C; c++)
            for (int l = 0; l < Lo; l++)
            {
                int start = l * stride_;
                for (int k = 0; k < kernel_ && start + k < Lin; k++)
                    g.at(b, c, start + k) += grad_out.at(b, c, l) * inv_k;
            }
    return g;
}

#endif

// ============================================================================
// MaxPool2D
// ============================================================================

MaxPool2D::MaxPool2D(int kH, int kW, int sH, int sW)
    : Layer("max_pool2d", false),
      kH_(kH), kW_(kW),
      sH_(sH < 0 ? kH : sH),
      sW_(sW < 0 ? kW : sW)
{}

// x: [B, C, H, W]  — stored as Tensor with ndim=4
Tensor MaxPool2D::forward(const Tensor &x)
{
    int B  = x.shape[0];
    int C  = x.shape[1];
    int H  = x.shape[2];
    int W  = x.shape[3];
    int OH = (H - kH_) / sH_ + 1;
    int OW = (W - kW_) / sW_ + 1;

    Tensor out(B, C, OH, OW);

#if TINY_AI_TRAINING_ENABLED
    in_b_ = B; in_c_ = C; in_h_ = H; in_w_ = W;
    // mask stores packed (ih, iw) as two consecutive floats → shape [B, C, OH, OW*2]
    mask_ = Tensor(B, C, OH, OW * 2);
#endif

    for (int b = 0; b < B; b++)
        for (int c = 0; c < C; c++)
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++)
                {
                    float mx = -FLT_MAX;
                    int   mi = oh * sH_, mj = ow * sW_;
                    for (int kh = 0; kh < kH_; kh++)
                        for (int kw = 0; kw < kW_; kw++)
                        {
                            float v = x.at(b, c, oh * sH_ + kh, ow * sW_ + kw);
                            if (v > mx) { mx = v; mi = oh * sH_ + kh; mj = ow * sW_ + kw; }
                        }
                    out.at(b, c, oh, ow) = mx;
#if TINY_AI_TRAINING_ENABLED
                    // pack as two consecutive floats in mask_
                    int idx = ((b * C + c) * OH + oh) * (OW * 2) + ow * 2;
                    mask_.data[idx]     = (float)mi;
                    mask_.data[idx + 1] = (float)mj;
#endif
                }
    return out;
}

#if TINY_AI_TRAINING_ENABLED

Tensor MaxPool2D::backward(const Tensor &grad_out)
{
    int OH = grad_out.shape[2];
    int OW = grad_out.shape[3];

    Tensor g(in_b_, in_c_, in_h_, in_w_);

    for (int b = 0; b < in_b_; b++)
        for (int c = 0; c < in_c_; c++)
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++)
                {
                    int idx = ((b * in_c_ + c) * OH + oh) * (OW * 2) + ow * 2;
                    int mi  = (int)mask_.data[idx];
                    int mj  = (int)mask_.data[idx + 1];
                    g.at(b, c, mi, mj) += grad_out.at(b, c, oh, ow);
                }
    return g;
}

#endif

// ============================================================================
// AvgPool2D
// ============================================================================

AvgPool2D::AvgPool2D(int kH, int kW, int sH, int sW)
    : Layer("avg_pool2d", false),
      kH_(kH), kW_(kW),
      sH_(sH < 0 ? kH : sH),
      sW_(sW < 0 ? kW : sW)
{}

Tensor AvgPool2D::forward(const Tensor &x)
{
    int B  = x.shape[0];
    int C  = x.shape[1];
    int H  = x.shape[2];
    int W  = x.shape[3];
    int OH = (H - kH_) / sH_ + 1;
    int OW = (W - kW_) / sW_ + 1;

    Tensor out(B, C, OH, OW);

#if TINY_AI_TRAINING_ENABLED
    in_b_ = B; in_c_ = C; in_h_ = H; in_w_ = W;
#endif

    float inv = 1.0f / (float)(kH_ * kW_);
    for (int b = 0; b < B; b++)
        for (int c = 0; c < C; c++)
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++)
                {
                    float sum = 0.0f;
                    for (int kh = 0; kh < kH_; kh++)
                        for (int kw = 0; kw < kW_; kw++)
                            sum += x.at(b, c, oh * sH_ + kh, ow * sW_ + kw);
                    out.at(b, c, oh, ow) = sum * inv;
                }
    return out;
}

#if TINY_AI_TRAINING_ENABLED

Tensor AvgPool2D::backward(const Tensor &grad_out)
{
    int OH = grad_out.shape[2];
    int OW = grad_out.shape[3];

    Tensor g(in_b_, in_c_, in_h_, in_w_);
    float inv = 1.0f / (float)(kH_ * kW_);

    for (int b = 0; b < in_b_; b++)
        for (int c = 0; c < in_c_; c++)
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++)
                    for (int kh = 0; kh < kH_; kh++)
                        for (int kw = 0; kw < kW_; kw++)
                            g.at(b, c, oh * sH_ + kh, ow * sW_ + kw) +=
                                grad_out.at(b, c, oh, ow) * inv;
    return g;
}

#endif

} // namespace tiny

#endif // __cplusplus
