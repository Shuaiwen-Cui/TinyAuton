/**
 * @file tiny_pool.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Max-pooling and average-pooling layers (1-D and 2-D) for tiny_ai.
 *
 *  MaxPool1D / AvgPool1D
 *    input:  [batch, channels, length]
 *    output: [batch, channels, length / kernel_size]  (non-overlapping)
 *
 *  MaxPool2D / AvgPool2D
 *    input:  [batch, channels, height, width]   (stored as flat 4-D tensor)
 *    output: [batch, channels, height/kH, width/kW]
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

/* ============================================================================
 * MaxPool1D
 * ============================================================================ */
class MaxPool1D : public Layer
{
public:
    explicit MaxPool1D(int kernel_size, int stride = -1);

    Tensor forward(const Tensor &x) override;

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override;
#endif

private:
    int kernel_;
    int stride_;    // defaults to kernel_ (non-overlapping)

#if TINY_AI_TRAINING_ENABLED
    Tensor mask_;   // Saves argmax positions as float indices
    int    in_len_;
    int    in_batch_, in_ch_;
#endif
};

/* ============================================================================
 * AvgPool1D
 * ============================================================================ */
class AvgPool1D : public Layer
{
public:
    explicit AvgPool1D(int kernel_size, int stride = -1);

    Tensor forward(const Tensor &x) override;

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override;
#endif

private:
    int kernel_;
    int stride_;
#if TINY_AI_TRAINING_ENABLED
    int in_len_, in_batch_, in_ch_;
#endif
};

/* ============================================================================
 * MaxPool2D
 * ============================================================================ */
class MaxPool2D : public Layer
{
public:
    MaxPool2D(int kH, int kW, int sH = -1, int sW = -1);

    Tensor forward(const Tensor &x) override;

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override;
#endif

private:
    int kH_, kW_, sH_, sW_;
#if TINY_AI_TRAINING_ENABLED
    // argmax mask: stores (ih, iw) as interleaved int pairs packed in a float buffer
    Tensor mask_;
    int in_b_, in_c_, in_h_, in_w_;
#endif
};

/* ============================================================================
 * AvgPool2D
 * ============================================================================ */
class AvgPool2D : public Layer
{
public:
    AvgPool2D(int kH, int kW, int sH = -1, int sW = -1);

    Tensor forward(const Tensor &x) override;

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override;
#endif

private:
    int kH_, kW_, sH_, sW_;
#if TINY_AI_TRAINING_ENABLED
    int in_b_, in_c_, in_h_, in_w_;
#endif
};

} // namespace tiny

#endif // __cplusplus
