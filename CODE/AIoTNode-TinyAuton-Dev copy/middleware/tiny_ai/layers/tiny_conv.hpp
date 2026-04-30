/**
 * @file tiny_conv.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Convolutional layers for tiny_ai.
 *
 *  Conv1D — 1-D convolution for time-series / signal data
 *    input:  [batch, in_channels, length]
 *    output: [batch, out_channels, out_length]
 *    out_length = (length + 2*padding - kernel_size) / stride + 1
 *
 *  Conv2D — 2-D convolution for image / feature-map data
 *    input:  [batch, in_channels, height, width]
 *    output: [batch, out_channels, out_height, out_width]
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

/* ============================================================================
 * Conv1D
 * ============================================================================ */
class Conv1D : public Layer
{
public:
    Tensor weight;   ///< [out_ch, in_ch, kernel]
    Tensor bias;     ///< [out_ch]

#if TINY_AI_TRAINING_ENABLED
    Tensor dweight;
    Tensor dbias;
#endif

    /**
     * @param in_channels   Input channels
     * @param out_channels  Output channels (number of filters)
     * @param kernel_size   Kernel width
     * @param stride        Convolution stride (default 1)
     * @param padding       Zero-padding on each side (default 0)
     * @param use_bias      Whether to add bias (default true)
     */
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
    Tensor x_cache_;  // padded input cached for backward
#endif

    void he_init();
};

/* ============================================================================
 * Conv2D
 * ============================================================================ */
class Conv2D : public Layer
{
public:
    Tensor weight;   ///< [out_ch, in_ch, kH, kW]
    Tensor bias;     ///< [out_ch]

#if TINY_AI_TRAINING_ENABLED
    Tensor dweight;
    Tensor dbias;
#endif

    /**
     * @param in_channels   Input channels
     * @param out_channels  Number of filters
     * @param kH            Kernel height
     * @param kW            Kernel width
     * @param sH            Stride H (default 1)
     * @param sW            Stride W (default 1)
     * @param pH            Padding H (default 0)
     * @param pW            Padding W (default 0)
     * @param use_bias      Whether to add bias (default true)
     */
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
