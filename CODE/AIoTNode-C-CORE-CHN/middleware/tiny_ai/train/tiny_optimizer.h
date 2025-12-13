/**
 * @file tiny_optimizer.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Optimizers for neural network training
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 *
 * @details
 * This module provides optimization algorithms for updating neural network parameters.
 * All optimizers support gradient-based parameter updates.
 * 
 * Supported optimizers:
 * - SGD: Stochastic Gradient Descent
 * - SGD with Momentum: SGD with momentum term
 * - Adam: Adaptive Moment Estimation
 * - RMSprop: Root Mean Square Propagation
 * - AdaGrad: Adaptive Gradient Algorithm
 * 
 * Features:
 * - Parameter update: w = w - lr * grad (with optimizer-specific modifications)
 * - Learning rate scheduling support
 * - Memory-efficient implementations for MCU
 * - Support for per-parameter learning rates
 * 
 * Design:
 * - Lightweight implementation for MCU
 * - Efficient memory usage
 * - Numerically stable implementations
 */

#pragma once

#include "tiny_ai_config.h"
#include "tiny_tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* ============================================================================
 * TYPE DEFINITIONS
 * ============================================================================ */

/**
 * @brief Optimizer type enumeration
 */
typedef enum
{
    TINY_OPTIMIZER_SGD = 0,
    TINY_OPTIMIZER_SGD_MOMENTUM,
    TINY_OPTIMIZER_ADAM,
    TINY_OPTIMIZER_RMSPROP,
    TINY_OPTIMIZER_ADAGRAD,
} tiny_optimizer_type_t;

/**
 * @brief Base optimizer structure
 */
typedef struct tiny_optimizer_t
{
    tiny_optimizer_type_t type;   // Optimizer type
    float learning_rate;           // Learning rate
    int step_count;                // Current step count (for Adam, etc.)
} tiny_optimizer_t;

/**
 * @brief SGD optimizer parameters
 */
typedef struct tiny_sgd_optimizer_t
{
    tiny_optimizer_t base;         // Base optimizer
    // No additional parameters for basic SGD
} tiny_sgd_optimizer_t;

/**
 * @brief SGD with Momentum optimizer parameters
 */
typedef struct tiny_sgd_momentum_optimizer_t
{
    tiny_optimizer_t base;         // Base optimizer
    float momentum;                 // Momentum coefficient (typically 0.9)
    tiny_tensor_t* velocity;        // Velocity buffer (same shape as parameters)
} tiny_sgd_momentum_optimizer_t;

/**
 * @brief Adam optimizer parameters
 */
typedef struct tiny_adam_optimizer_t
{
    tiny_optimizer_t base;         // Base optimizer
    float beta1;                    // First moment decay (typically 0.9)
    float beta2;                    // Second moment decay (typically 0.999)
    float epsilon;                  // Small constant for numerical stability (typically 1e-8)
    tiny_tensor_t* m;               // First moment estimate
    tiny_tensor_t* v;               // Second moment estimate
} tiny_adam_optimizer_t;

/**
 * @brief RMSprop optimizer parameters
 */
typedef struct tiny_rmsprop_optimizer_t
{
    tiny_optimizer_t base;         // Base optimizer
    float alpha;                    // Decay rate (typically 0.9)
    float epsilon;                  // Small constant (typically 1e-8)
    tiny_tensor_t* cache;           // Running average of squared gradients
} tiny_rmsprop_optimizer_t;

/**
 * @brief AdaGrad optimizer parameters
 */
typedef struct tiny_adagrad_optimizer_t
{
    tiny_optimizer_t base;         // Base optimizer
    float epsilon;                  // Small constant (typically 1e-8)
    tiny_tensor_t* cache;           // Accumulated squared gradients
} tiny_adagrad_optimizer_t;

/* ============================================================================
 * FUNCTION PROTOTYPES - Optimizer Creation and Destruction
 * ============================================================================ */

/**
 * @brief Create SGD optimizer
 * 
 * @param learning_rate Learning rate
 * @return tiny_sgd_optimizer_t* Pointer to created optimizer, NULL on failure
 */
tiny_sgd_optimizer_t* tiny_optimizer_sgd_create(float learning_rate);

/**
 * @brief Create SGD with Momentum optimizer
 * 
 * @param learning_rate Learning rate
 * @param momentum Momentum coefficient (typically 0.9)
 * @return tiny_sgd_momentum_optimizer_t* Pointer to created optimizer, NULL on failure
 */
tiny_sgd_momentum_optimizer_t* tiny_optimizer_sgd_momentum_create(float learning_rate, float momentum);

/**
 * @brief Create Adam optimizer
 * 
 * @param learning_rate Learning rate
 * @param beta1 First moment decay (typically 0.9)
 * @param beta2 Second moment decay (typically 0.999)
 * @param epsilon Small constant (typically 1e-8)
 * @return tiny_adam_optimizer_t* Pointer to created optimizer, NULL on failure
 */
tiny_adam_optimizer_t* tiny_optimizer_adam_create(float learning_rate, float beta1, float beta2, float epsilon);

/**
 * @brief Create RMSprop optimizer
 * 
 * @param learning_rate Learning rate
 * @param alpha Decay rate (typically 0.9)
 * @param epsilon Small constant (typically 1e-8)
 * @return tiny_rmsprop_optimizer_t* Pointer to created optimizer, NULL on failure
 */
tiny_rmsprop_optimizer_t* tiny_optimizer_rmsprop_create(float learning_rate, float alpha, float epsilon);

/**
 * @brief Create AdaGrad optimizer
 * 
 * @param learning_rate Learning rate
 * @param epsilon Small constant (typically 1e-8)
 * @return tiny_adagrad_optimizer_t* Pointer to created optimizer, NULL on failure
 */
tiny_adagrad_optimizer_t* tiny_optimizer_adagrad_create(float learning_rate, float epsilon);

/**
 * @brief Destroy optimizer and free all resources
 * 
 * @param optimizer Optimizer to destroy (can be NULL)
 * 
 * @note This function handles all optimizer types
 */
void tiny_optimizer_destroy(void* optimizer);

/* ============================================================================
 * FUNCTION PROTOTYPES - Parameter Update
 * ============================================================================ */

/**
 * @brief Update parameters using SGD optimizer
 * 
 * @param optimizer SGD optimizer
 * @param param Parameter tensor to update
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @note Performs: param = param - lr * grad
 * @note Parameter must have gradients enabled
 */
tiny_error_t tiny_optimizer_sgd_step(tiny_sgd_optimizer_t* optimizer, tiny_tensor_t* param);

/**
 * @brief Update parameters using SGD with Momentum optimizer
 * 
 * @param optimizer SGD with Momentum optimizer
 * @param param Parameter tensor to update
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @note Performs: v = momentum * v + grad; param = param - lr * v
 * @note Parameter must have gradients enabled
 */
tiny_error_t tiny_optimizer_sgd_momentum_step(tiny_sgd_momentum_optimizer_t* optimizer, tiny_tensor_t* param);

/**
 * @brief Update parameters using Adam optimizer
 * 
 * @param optimizer Adam optimizer
 * @param param Parameter tensor to update
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @note Performs Adam update with bias correction
 * @note Parameter must have gradients enabled
 */
tiny_error_t tiny_optimizer_adam_step(tiny_adam_optimizer_t* optimizer, tiny_tensor_t* param);

/**
 * @brief Update parameters using RMSprop optimizer
 * 
 * @param optimizer RMSprop optimizer
 * @param param Parameter tensor to update
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @note Performs: cache = alpha * cache + (1-alpha) * grad^2; param = param - lr * grad / sqrt(cache + eps)
 * @note Parameter must have gradients enabled
 */
tiny_error_t tiny_optimizer_rmsprop_step(tiny_rmsprop_optimizer_t* optimizer, tiny_tensor_t* param);

/**
 * @brief Update parameters using AdaGrad optimizer
 * 
 * @param optimizer AdaGrad optimizer
 * @param param Parameter tensor to update
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @note Performs: cache = cache + grad^2; param = param - lr * grad / sqrt(cache + eps)
 * @note Parameter must have gradients enabled
 */
tiny_error_t tiny_optimizer_adagrad_step(tiny_adagrad_optimizer_t* optimizer, tiny_tensor_t* param);

/* ============================================================================
 * FUNCTION PROTOTYPES - Utility
 * ============================================================================ */

/**
 * @brief Set learning rate for optimizer
 * 
 * @param optimizer Optimizer to configure
 * @param learning_rate New learning rate
 */
void tiny_optimizer_set_learning_rate(void* optimizer, float learning_rate);

/**
 * @brief Get learning rate from optimizer
 * 
 * @param optimizer Optimizer to query
 * @return float Current learning rate
 */
float tiny_optimizer_get_learning_rate(const void* optimizer);

#ifdef __cplusplus
}
#endif

