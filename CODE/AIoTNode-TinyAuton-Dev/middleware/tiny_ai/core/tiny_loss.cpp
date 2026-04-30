/**
 * @file tiny_loss.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Loss function implementations for tiny_ai.
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#include "tiny_loss.hpp"
#include "tiny_activation.hpp"
#include <cmath>

#ifdef __cplusplus

namespace tiny
{

// ============================================================================
// MSE
// ============================================================================

float mse_forward(const Tensor &pred, const Tensor &target)
{
    float sum = 0.0f;
    for (int i = 0; i < pred.size; i++)
    {
        float d = pred.data[i] - target.data[i];
        sum += d * d;
    }
    return sum / (float)pred.size;
}

#if TINY_AI_TRAINING_ENABLED

Tensor mse_backward(const Tensor &pred, const Tensor &target)
{
    Tensor g = Tensor::zeros_like(pred);
    float scale = 2.0f / (float)pred.size;
    for (int i = 0; i < pred.size; i++)
        g.data[i] = scale * (pred.data[i] - target.data[i]);
    return g;
}

#endif

// ============================================================================
// MAE
// ============================================================================

float mae_forward(const Tensor &pred, const Tensor &target)
{
    float sum = 0.0f;
    for (int i = 0; i < pred.size; i++)
        sum += fabsf(pred.data[i] - target.data[i]);
    return sum / (float)pred.size;
}

#if TINY_AI_TRAINING_ENABLED

Tensor mae_backward(const Tensor &pred, const Tensor &target)
{
    Tensor g = Tensor::zeros_like(pred);
    float scale = 1.0f / (float)pred.size;
    for (int i = 0; i < pred.size; i++)
    {
        float d = pred.data[i] - target.data[i];
        g.data[i] = (d > 0.0f ? scale : (d < 0.0f ? -scale : 0.0f));
    }
    return g;
}

#endif

// ============================================================================
// Cross-Entropy (numerically stable, expects logits)
// ============================================================================

float cross_entropy_forward(const Tensor &logits, const int *labels)
{
    // logits: [batch, num_classes]
    int batch = logits.rows();
    int cls   = logits.cols();
    float loss = 0.0f;

    for (int b = 0; b < batch; b++)
    {
        const float *row = logits.data + b * cls;
        int lbl          = labels[b];

        // log-sum-exp trick for numerical stability
        float mx = row[0];
        for (int c = 1; c < cls; c++) if (row[c] > mx) mx = row[c];

        float sum_exp = 0.0f;
        for (int c = 0; c < cls; c++) sum_exp += expf(row[c] - mx);

        // NLL = -(row[lbl] - mx) + log(sum_exp)
        loss += -(row[lbl] - mx) + logf(sum_exp + TINY_MATH_MIN_DENOMINATOR);
    }
    return loss / (float)batch;
}

#if TINY_AI_TRAINING_ENABLED

// Gradient: softmax(logits) - one_hot(labels)
Tensor cross_entropy_backward(const Tensor &logits, const int *labels)
{
    int batch = logits.rows();
    int cls   = logits.cols();

    // Start with softmax probabilities
    Tensor g = softmax_forward(logits);

    // Subtract 1 at the true class position, normalise by batch
    float inv_batch = 1.0f / (float)batch;
    for (int b = 0; b < batch; b++)
    {
        g.data[b * cls + labels[b]] -= 1.0f;
        for (int c = 0; c < cls; c++)
            g.data[b * cls + c] *= inv_batch;
    }
    return g;
}

#endif

// ============================================================================
// Binary Cross-Entropy
// ============================================================================

float binary_ce_forward(const Tensor &pred, const Tensor &target)
{
    float loss = 0.0f;
    float eps  = TINY_MATH_MIN_POSITIVE_INPUT_F32;
    for (int i = 0; i < pred.size; i++)
    {
        float p = pred.data[i];
        float t = target.data[i];
        loss -= t * logf(p + eps) + (1.0f - t) * logf(1.0f - p + eps);
    }
    return loss / (float)pred.size;
}

#if TINY_AI_TRAINING_ENABLED

Tensor binary_ce_backward(const Tensor &pred, const Tensor &target)
{
    Tensor g(pred.clone());
    float  eps   = TINY_MATH_MIN_POSITIVE_INPUT_F32;
    float  scale = 1.0f / (float)pred.size;
    for (int i = 0; i < pred.size; i++)
    {
        float p = pred.data[i];
        float t = target.data[i];
        g.data[i] = scale * (-(t / (p + eps)) + (1.0f - t) / (1.0f - p + eps));
    }
    return g;
}

#endif

// ============================================================================
// Dispatch
// ============================================================================

float loss_forward(const Tensor &pred, const Tensor &target,
                   LossType type, const int *labels)
{
    switch (type)
    {
        case LossType::MSE:          return mse_forward(pred, target);
        case LossType::MAE:          return mae_forward(pred, target);
        case LossType::CROSS_ENTROPY:return cross_entropy_forward(pred, labels);
        case LossType::BINARY_CE:    return binary_ce_forward(pred, target);
        default: return 0.0f;
    }
}

#if TINY_AI_TRAINING_ENABLED

Tensor loss_backward(const Tensor &pred, const Tensor &target,
                     LossType type, const int *labels)
{
    switch (type)
    {
        case LossType::MSE:          return mse_backward(pred, target);
        case LossType::MAE:          return mae_backward(pred, target);
        case LossType::CROSS_ENTROPY:return cross_entropy_backward(pred, labels);
        case LossType::BINARY_CE:    return binary_ce_backward(pred, target);
        default:                     return pred.clone();
    }
}

#endif

} // namespace tiny

#endif // __cplusplus
