/**
 * @file tiny_ai.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_ai | Main header file - Unified entry point for all AI functionality
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 *
 * @details
 * This header file provides a unified interface to all AI (Artificial Intelligence)
 * functionality in the tiny_ai middleware. It includes:
 *
 * - Core: Tensor data structure, computation graph, memory management
 * - Operators: Activation functions, fully connected, convolution, pooling, normalization
 * - Loss: Loss functions for training
 * - Models: MLP, CNN model wrappers
 * - Training: Optimizers, training loop, data loading
 * - Utils: Preprocessing and postprocessing utilities
 *
 * Usage:
 *   Simply include this header to access all AI functions:
 *   @code
 *   #include "tiny_ai.h"
 *   @endcode
 */

#pragma once

/* ============================================================================
 * DEPENDENCIES
 * ============================================================================ */

// Core configuration
#include "tiny_ai_config.h"

/* ============================================================================
 * CORE MODULES
 * ============================================================================ */

/**
 * @name Core - Tensor
 * @brief Tensor data structure for multi-dimensional arrays
 * @details
 * - Multi-dimensional array representation
 * - Support for gradients (for training)
 * - Memory-efficient storage
 */
#include "tiny_tensor.h"
#include "tiny_tensor_test.h"

/**
 * @name Core - Computation Graph
 * @brief Computation graph for forward and backward propagation
 * @details
 * - Graph-based model representation
 * - Forward and backward propagation
 * - Automatic gradient computation
 */
#include "tiny_graph.h"
#include "tiny_graph_test.h"

/**
 * @name Core - Memory Management
 * @brief Memory allocation and management utilities
 * @details
 * - Static buffer allocation
 * - Memory pool for temporary buffers
 * - Gradient buffer management
 */
// #include "tiny_memory.h"
// #include "tiny_memory_test.h"

/* ============================================================================
 * OPERATOR MODULES
 * ============================================================================ */

/**
 * @name Operator - Activation Functions
 * @brief Activation functions with forward and backward passes
 * @details
 * - ReLU, Sigmoid, Tanh, Softmax
 * - Forward and backward propagation
 * - In-place operations support
 */
#include "tiny_activations.h"
// #include "tiny_activations_test.h"

/**
 * @name Operator - Fully Connected Layer
 * @brief Fully connected (dense) layer with forward and backward passes
 * @details
 * - Matrix multiplication: output = input * weights + bias
 * - Forward and backward propagation
 * - Gradient computation for weights and bias
 */
#include "tiny_fc.h"
// #include "tiny_fc_test.h"

/**
 * @name Operator - Convolution Layer
 * @brief Convolution layer (2D) with forward and backward passes
 * @details
 * - 2D convolution support
 * - Configurable padding and stride
 * - Forward and backward propagation
 * - Gradient computation for filters
 */
#include "tiny_convolution.h"
// #include "tiny_convolution_test.h"

/**
 * @name Operator - Pooling Layer
 * @brief Pooling operations (Max/Average) with forward and backward passes
 * @details
 * - Max pooling and average pooling
 * - Forward and backward propagation
 * - Gradient computation
 */
#include "tiny_pool.h"
// #include "tiny_pool_test.h"

/**
 * @name Operator - Normalization Layer
 * @brief Normalization layers (BatchNorm) with forward and backward passes
 * @details
 * - Batch normalization
 * - Forward and backward propagation
 * - Training and inference modes
 */
#include "tiny_norm.h"
// #include "tiny_norm_test.h"

/* ============================================================================
 * LOSS MODULES
 * ============================================================================ */

/**
 * @name Loss - Loss Functions
 * @brief Loss functions for training
 * @details
 * - Mean Squared Error (MSE)
 * - Mean Absolute Error (MAE)
 * - Cross Entropy
 * - Binary Cross Entropy
 * - Forward and backward passes
 */
#include "tiny_loss.h"
// #include "tiny_loss_test.h"

/* ============================================================================
 * MODEL MODULES
 * ============================================================================ */

/**
 * @name Model - MLP (Multi-Layer Perceptron)
 * @brief Multi-layer perceptron model wrapper
 * @details
 * - Stacked fully connected layers
 * - Forward and backward propagation
 * - Training and inference support
 */
#include "tiny_mlp.h"
// #include "tiny_mlp_test.h"

/**
 * @name Model - CNN (Convolutional Neural Network)
 * @brief Convolutional neural network model wrapper
 * @details
 * - Stacked convolution and pooling layers
 * - Forward and backward propagation
 * - Training and inference support
 */
#include "tiny_cnn.h"
// #include "tiny_cnn_test.h"

/* ============================================================================
 * TRAINING MODULES
 * ============================================================================ */

/**
 * @name Training - Optimizers
 * @brief Optimization algorithms for training
 * @details
 * - SGD (Stochastic Gradient Descent)
 * - SGD with Momentum
 * - Adam (Adaptive Moment Estimation)
 * - RMSprop
 * - AdaGrad
 * - Learning rate scheduling
 */
#include "tiny_optimizer.h"
// #include "tiny_optimizer_test.h"

/**
 * @name Training - Trainer
 * @brief Training loop and utilities
 * @details
 * - Training loop implementation
 * - Epoch and batch management
 * - Loss tracking
 * - Integration with optimizers and computation graph
 */
#include "tiny_trainer.h"
// #include "tiny_trainer_test.h"

/* ============================================================================
 * TEST MODULES
 * ============================================================================ */

/**
 * @name Test - Integration Tests
 * @brief End-to-end integration tests
 * @details
 * - Complete training pipeline tests
 * - Model training verification
 * - Loss convergence tests
 */
// #include "tiny_ai_integration_test.h"

/**
 * @name Test - Sine Regression Tests
 * @brief Sine function regression tests
 * @details
 * - Sine function regression with MLP/CNN models
 * - Data generation (sine + noise)
 * - Training and inference comparison
 */
// #include "tiny_ai_sine_regression_test.h"

/**
 * @name Training - Data Loader
 * @brief Data loading and batching utilities
 * @details
 * - Batch data loading
 * - Data shuffling
 * - Data preprocessing integration
 */
#include "tiny_dataloader.h"
// #include "tiny_dataloader_test.h"

/* ============================================================================
 * UTILITY MODULES
 * ============================================================================ */

/**
 * @name Utils - Preprocessing
 * @brief Data preprocessing utilities
 * @details
 * - Normalization and standardization
 * - Data augmentation
 * - Feature scaling
 */
// #include "tiny_preprocess.h"
// #include "tiny_preprocess_test.h"

/**
 * @name Utils - Postprocessing
 * @brief Postprocessing utilities
 * @details
 * - Softmax and argmax
 * - Result interpretation
 * - Confidence scoring
 */
// #include "tiny_postprocess.h"
// #include "tiny_postprocess_test.h"

/* ============================================================================
 * C++ COMPATIBILITY
 * ============================================================================ */

#ifdef __cplusplus
extern "C"
{
#endif

    // All AI functions are C-compatible and can be called from C++

#ifdef __cplusplus
}
#endif

