/**
 * @file tiny_loss.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Loss functions for tiny_ai — MSE, MAE, Cross-Entropy, Binary-CE.
 *        Each loss provides a scalar forward value and an input-gradient tensor.
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#pragma once

#include "tiny_tensor.hpp"

#ifdef __cplusplus

namespace tiny
{

/* ============================================================================
 * Loss type enum
 * ============================================================================ */
enum class LossType
{
    MSE = 0,           ///< Mean Squared Error
    MAE,               ///< Mean Absolute Error
    CROSS_ENTROPY,     ///< Softmax + Cross-Entropy (expects logits as input)
    BINARY_CE          ///< Binary Cross-Entropy (expects sigmoid output)
};

/* ============================================================================
 * MSE  — Mean Squared Error
 * ============================================================================ */

/**
 * @brief MSE forward: L = mean((pred - target)^2)
 * @param pred   Predicted values  [batch, *]
 * @param target Ground truth      [batch, *]  (same shape as pred)
 * @return Scalar loss value
 */
float mse_forward(const Tensor &pred, const Tensor &target);

#if TINY_AI_TRAINING_ENABLED
/**
 * @brief MSE backward: dL/d_pred = 2*(pred - target) / N
 */
Tensor mse_backward(const Tensor &pred, const Tensor &target);
#endif

/* ============================================================================
 * MAE  — Mean Absolute Error
 * ============================================================================ */

float mae_forward(const Tensor &pred, const Tensor &target);

#if TINY_AI_TRAINING_ENABLED
Tensor mae_backward(const Tensor &pred, const Tensor &target);
#endif

/* ============================================================================
 * Cross-Entropy  (with built-in numerically stable Softmax)
 * ============================================================================ */

/**
 * @brief Softmax Cross-Entropy forward.
 *        Expects raw logits; applies log-softmax internally.
 *
 * @param logits  Raw class scores  [batch, num_classes]
 * @param labels  Integer class labels [batch]  (values in [0, num_classes))
 * @return Mean cross-entropy loss over the batch
 */
float cross_entropy_forward(const Tensor &logits, const int *labels);

#if TINY_AI_TRAINING_ENABLED
/**
 * @brief Cross-Entropy backward.
 *        Returns dL/d_logits = softmax(logits) - one_hot(labels),  normalised by batch.
 */
Tensor cross_entropy_backward(const Tensor &logits, const int *labels);
#endif

/* ============================================================================
 * Binary Cross-Entropy
 * ============================================================================ */

/**
 * @brief Binary CE forward: expects sigmoid probabilities in pred.
 *        target values should be 0.0 or 1.0.
 */
float binary_ce_forward(const Tensor &pred, const Tensor &target);

#if TINY_AI_TRAINING_ENABLED
Tensor binary_ce_backward(const Tensor &pred, const Tensor &target);
#endif

/* ============================================================================
 * Dispatch helpers
 * ============================================================================ */

float loss_forward(const Tensor &pred, const Tensor &target,
                   LossType type, const int *labels = nullptr);

#if TINY_AI_TRAINING_ENABLED
Tensor loss_backward(const Tensor &pred, const Tensor &target,
                     LossType type, const int *labels = nullptr);
#endif

} // namespace tiny

#endif // __cplusplus
