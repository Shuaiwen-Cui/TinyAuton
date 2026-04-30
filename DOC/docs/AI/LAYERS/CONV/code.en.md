# Code

## tiny_conv.hpp

```cpp
/**
 * @file tiny_conv.hpp
 * @brief Convolutional layers for tiny_ai (Conv1D / Conv2D).
 */

#pragma once

#include "tiny_layer.hpp"

#ifdef __cplusplus

namespace tiny
{

class Conv1D : public Layer
{
public:
    Tensor weight;   // [out_ch, in_ch, kernel]
    Tensor bias;     // [out_ch]
#if TINY_AI_TRAINING_ENABLED
    Tensor dweight;
    Tensor dbias;
#endif

    Conv1D(int in_channels, int out_channels, int kernel_size,
           int stride = 1, int padding = 0, bool use_bias = true);

    Tensor forward(const Tensor &x) override;

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override;
    void   collect_params(std::vector<ParamGroup> &groups) override;
#endif

    int in_channels()  const { return in_ch_; }
    int out_channels() const { return out_ch_; }

private:
    int  in_ch_, out_ch_, kernel_, stride_, padding_;
    bool use_bias_;
#if TINY_AI_TRAINING_ENABLED
    Tensor x_cache_;
#endif
    void he_init();
};

class Conv2D : public Layer
{
public:
    Tensor weight;   // [out_ch, in_ch, kH, kW]
    Tensor bias;     // [out_ch]
#if TINY_AI_TRAINING_ENABLED
    Tensor dweight;
    Tensor dbias;
#endif

    Conv2D(int in_channels, int out_channels, int kH, int kW,
           int sH = 1, int sW = 1, int pH = 0, int pW = 0,
           bool use_bias = true);

    Tensor forward(const Tensor &x) override;

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override;
    void   collect_params(std::vector<ParamGroup> &groups) override;
#endif

private:
    int  in_ch_, out_ch_, kH_, kW_, sH_, sW_, pH_, pW_;
    bool use_bias_;
#if TINY_AI_TRAINING_ENABLED
    Tensor x_cache_;
#endif
    void he_init();
};

} // namespace tiny

#endif // __cplusplus
```

## tiny_conv.cpp

```cpp
/**
 * @file tiny_conv.cpp
 * @brief Conv1D and Conv2D implementations.
 */

#include "tiny_conv.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>

#ifdef __cplusplus

namespace tiny
{

// ============================================================================
// Conv1D
// ============================================================================

Conv1D::Conv1D(int in_ch, int out_ch, int kernel, int stride, int padding, bool use_bias)
    : Layer("conv1d", true),
      weight(out_ch, in_ch, kernel),
      bias(use_bias ? out_ch : 0),
      in_ch_(in_ch), out_ch_(out_ch), kernel_(kernel),
      stride_(stride), padding_(padding), use_bias_(use_bias)
{
#if TINY_AI_TRAINING_ENABLED
    dweight = Tensor::zeros_like(weight);
    if (use_bias_) dbias = Tensor(out_ch);
#endif
    he_init();
}

void Conv1D::he_init()
{
    float std_dev = sqrtf(2.0f / (float)(in_ch_ * kernel_));
    for (int i = 0; i < weight.size; i++)
    {
        float u1 = (float)(rand() + 1) / ((float)RAND_MAX + 1.0f);
        float u2 = (float)rand()       / ((float)RAND_MAX + 1.0f);
        float n  = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * TINY_PI * u2);
        weight.data[i] = n * std_dev;
    }
    if (use_bias_) bias.zero();
}

Tensor Conv1D::forward(const Tensor &x)
{
    int B   = x.shape[0];
    int Lin = x.shape[2];
    int Lo  = (Lin + 2 * padding_ - kernel_) / stride_ + 1;

    int Lp = Lin + 2 * padding_;
    Tensor xp(B, in_ch_, Lp);
    for (int b = 0; b < B; b++)
        for (int c = 0; c < in_ch_; c++)
            for (int l = 0; l < Lin; l++)
                xp.at(b, c, l + padding_) = x.at(b, c, l);

#if TINY_AI_TRAINING_ENABLED
    x_cache_ = xp.clone();
#endif

    Tensor out(B, out_ch_, Lo);
    for (int b = 0; b < B; b++)
        for (int oc = 0; oc < out_ch_; oc++)
            for (int t = 0; t < Lo; t++)
            {
                float sum = use_bias_ ? bias.data[oc] : 0.0f;
                for (int ic = 0; ic < in_ch_; ic++)
                    for (int k = 0; k < kernel_; k++)
                        sum += weight.at(oc, ic, k) * xp.at(b, ic, t * stride_ + k);
                out.at(b, oc, t) = sum;
            }
    return out;
}

#if TINY_AI_TRAINING_ENABLED

Tensor Conv1D::backward(const Tensor &grad_out)
{
    int B   = x_cache_.shape[0];
    int Lp  = x_cache_.shape[2];
    int Lo  = grad_out.shape[2];
    int Lin = Lp - 2 * padding_;

    Tensor g_xp(B, in_ch_, Lp);

    for (int b = 0; b < B; b++)
        for (int oc = 0; oc < out_ch_; oc++)
            for (int t = 0; t < Lo; t++)
            {
                float go = grad_out.at(b, oc, t);
                for (int ic = 0; ic < in_ch_; ic++)
                    for (int k = 0; k < kernel_; k++)
                        dweight.at(oc, ic, k) += go * x_cache_.at(b, ic, t * stride_ + k);
                if (use_bias_) dbias.data[oc] += go;
            }

    for (int b = 0; b < B; b++)
        for (int oc = 0; oc < out_ch_; oc++)
            for (int t = 0; t < Lo; t++)
            {
                float go = grad_out.at(b, oc, t);
                for (int ic = 0; ic < in_ch_; ic++)
                    for (int k = 0; k < kernel_; k++)
                        g_xp.at(b, ic, t * stride_ + k) += go * weight.at(oc, ic, k);
            }

    Tensor g_x(B, in_ch_, Lin);
    for (int b = 0; b < B; b++)
        for (int ic = 0; ic < in_ch_; ic++)
            for (int l = 0; l < Lin; l++)
                g_x.at(b, ic, l) = g_xp.at(b, ic, l + padding_);
    return g_x;
}

void Conv1D::collect_params(std::vector<ParamGroup> &groups)
{
    groups.push_back({&weight, &dweight});
    if (use_bias_) groups.push_back({&bias, &dbias});
}

#endif

// ============================================================================
// Conv2D
// ============================================================================

Conv2D::Conv2D(int in_ch, int out_ch, int kH, int kW,
               int sH, int sW, int pH, int pW, bool use_bias)
    : Layer("conv2d", true),
      weight(out_ch, in_ch, kH, kW),
      bias(use_bias ? out_ch : 0),
      in_ch_(in_ch), out_ch_(out_ch),
      kH_(kH), kW_(kW), sH_(sH), sW_(sW), pH_(pH), pW_(pW),
      use_bias_(use_bias)
{
#if TINY_AI_TRAINING_ENABLED
    dweight = Tensor::zeros_like(weight);
    if (use_bias_) dbias = Tensor(out_ch);
#endif
    he_init();
}

void Conv2D::he_init()
{
    float std_dev = sqrtf(2.0f / (float)(in_ch_ * kH_ * kW_));
    for (int i = 0; i < weight.size; i++)
    {
        float u1 = (float)(rand() + 1) / ((float)RAND_MAX + 1.0f);
        float u2 = (float)rand()       / ((float)RAND_MAX + 1.0f);
        float n  = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * TINY_PI * u2);
        weight.data[i] = n * std_dev;
    }
    if (use_bias_) bias.zero();
}

Tensor Conv2D::forward(const Tensor &x)
{
    int B  = x.shape[0];
    int H  = x.shape[2];
    int W  = x.shape[3];
    int OH = (H + 2 * pH_ - kH_) / sH_ + 1;
    int OW = (W + 2 * pW_ - kW_) / sW_ + 1;

    int Hp = H + 2 * pH_;
    int Wp = W + 2 * pW_;
    Tensor xp(B, in_ch_, Hp, Wp);
    for (int b = 0; b < B; b++)
        for (int c = 0; c < in_ch_; c++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    xp.at(b, c, h + pH_, w + pW_) = x.at(b, c, h, w);

#if TINY_AI_TRAINING_ENABLED
    x_cache_ = xp.clone();
#endif

    Tensor out(B, out_ch_, OH, OW);
    for (int b = 0; b < B; b++)
        for (int oc = 0; oc < out_ch_; oc++)
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++)
                {
                    float sum = use_bias_ ? bias.data[oc] : 0.0f;
                    for (int ic = 0; ic < in_ch_; ic++)
                        for (int kh = 0; kh < kH_; kh++)
                            for (int kw = 0; kw < kW_; kw++)
                                sum += weight.at(oc, ic, kh, kw) *
                                       xp.at(b, ic, oh * sH_ + kh, ow * sW_ + kw);
                    out.at(b, oc, oh, ow) = sum;
                }
    return out;
}

#if TINY_AI_TRAINING_ENABLED

Tensor Conv2D::backward(const Tensor &grad_out)
{
    int B  = x_cache_.shape[0];
    int Hp = x_cache_.shape[2];
    int Wp = x_cache_.shape[3];
    int OH = grad_out.shape[2];
    int OW = grad_out.shape[3];

    Tensor g_xp(B, in_ch_, Hp, Wp);

    for (int b = 0; b < B; b++)
        for (int oc = 0; oc < out_ch_; oc++)
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++)
                {
                    float go = grad_out.at(b, oc, oh, ow);
                    if (use_bias_) dbias.data[oc] += go;
                    for (int ic = 0; ic < in_ch_; ic++)
                        for (int kh = 0; kh < kH_; kh++)
                            for (int kw = 0; kw < kW_; kw++)
                            {
                                dweight.at(oc, ic, kh, kw) +=
                                    go * x_cache_.at(b, ic, oh * sH_ + kh, ow * sW_ + kw);
                                g_xp.at(b, ic, oh * sH_ + kh, ow * sW_ + kw) +=
                                    go * weight.at(oc, ic, kh, kw);
                            }
                }

    int H = Hp - 2 * pH_;
    int W = Wp - 2 * pW_;
    Tensor g_x(B, in_ch_, H, W);
    for (int b = 0; b < B; b++)
        for (int c = 0; c < in_ch_; c++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    g_x.at(b, c, h, w) = g_xp.at(b, c, h + pH_, w + pW_);
    return g_x;
}

void Conv2D::collect_params(std::vector<ParamGroup> &groups)
{
    groups.push_back({&weight, &dweight});
    if (use_bias_) groups.push_back({&bias, &dbias});
}

#endif // TINY_AI_TRAINING_ENABLED

} // namespace tiny

#endif // __cplusplus
```
