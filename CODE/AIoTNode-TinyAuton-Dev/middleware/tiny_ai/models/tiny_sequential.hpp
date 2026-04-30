/**
 * @file tiny_sequential.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Sequential model container for tiny_ai.
 *        Stacks layers in order; forward/backward pass are applied sequentially.
 *        Owns all layers added via add() and frees them in the destructor.
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#pragma once

#include "tiny_layer.hpp"

#ifdef __cplusplus

#include <vector>

namespace tiny
{

class Sequential
{
public:
    Sequential()  = default;
    ~Sequential();

    // =========================================================================
    // Layer management
    // =========================================================================

    /**
     * @brief Append a layer.  Sequential takes ownership and will delete it.
     * @param layer  Heap-allocated Layer pointer (e.g. new Dense(…))
     */
    void add(Layer *layer);

    int num_layers() const { return (int)layers_.size(); }

    // =========================================================================
    // Forward / Backward
    // =========================================================================

    Tensor forward(const Tensor &x);

#if TINY_AI_TRAINING_ENABLED
    /**
     * @brief Run backward pass through all layers in reverse order.
     * @param grad_out  Gradient of the loss w.r.t. the model output
     * @return          Gradient of the loss w.r.t. the model input (rarely used)
     */
    Tensor backward(const Tensor &grad_out);

    /**
     * @brief Collect all (param, grad) pairs from trainable layers.
     *        Must be called before Optimizer::init().
     */
    void collect_params(std::vector<ParamGroup> &groups);
#endif

    // =========================================================================
    // Utilities
    // =========================================================================

    /// Print a summary of layer names to stdout
    void summary() const;

    /**
     * @brief Predict class labels for input x (argmax of forward output).
     * @param x      Input tensor [batch, features]
     * @param labels Output int array (length = batch)
     */
    void predict(const Tensor &x, int *labels);

    /**
     * @brief Compute classification accuracy on (x, labels).
     * @return Fraction of correctly classified samples in [0, 1]
     */
    float accuracy(const Tensor &x, const int *labels, int n_samples);

private:
    std::vector<Layer *> layers_;
};

} // namespace tiny

#endif // __cplusplus
