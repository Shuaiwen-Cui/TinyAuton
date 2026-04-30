/**
 * @file tiny_mlp.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Convenience MLP (Multi-Layer Perceptron) model wrapper for tiny_ai.
 *        Builds a Sequential from a list of layer widths and an activation.
 *
 *  Example (Iris classifier):
 *    MLP model({4, 16, 8, 3}, ActType::RELU);
 *    // → Dense(4,16) + ReLU + Dense(16,8) + ReLU + Dense(8,3) + Softmax
 *
 * @version 1.0
 * @date 2026-04-30
 * @copyright Copyright (c) 2026
 */

#pragma once

#include "tiny_sequential.hpp"
#include "tiny_dense.hpp"
#include "tiny_activation.hpp"

#ifdef __cplusplus

#include <initializer_list>

namespace tiny
{

class MLP : public Sequential
{
public:
    /**
     * @param dims         List of layer widths including input and output,
     *                     e.g. {4, 16, 8, 3} → two hidden layers
     * @param hidden_act   Activation after each hidden layer (default RELU)
     * @param use_softmax  If true, append a Softmax after the last Dense layer
     *                     (for classification; set false for regression)
     * @param use_bias     Whether each Dense layer has a bias term
     */
    MLP(std::initializer_list<int> dims,
        ActType hidden_act  = ActType::RELU,
        bool    use_softmax = true,
        bool    use_bias    = true);

    int in_features()  const { return in_feat_; }
    int out_features() const { return out_feat_; }

private:
    int in_feat_;
    int out_feat_;
};

} // namespace tiny

#endif // __cplusplus
