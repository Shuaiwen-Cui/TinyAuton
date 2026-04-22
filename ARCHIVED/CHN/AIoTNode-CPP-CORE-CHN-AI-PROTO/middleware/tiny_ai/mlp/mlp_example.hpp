/**
 * @file mlp_example.hpp
 * @brief Simple MLP (Multi-Layer Perceptron) for microcontroller AI
 * @version 1.0
 * 
 * @details
 * 简单的 MLP 实现，用于单片机 AI 推理
 * 基于 tiny_math 的 Mat 类进行矩阵运算
 */

#pragma once

#include "tiny_math.h"
#include <vector>
#include <cmath>
#include <string.h>

#ifdef __cplusplus

namespace tiny_ai {

/**
 * @brief Activation function types
 */
enum ActivationType {
    ACT_LINEAR = 0,  // Linear (no activation)
    ACT_RELU,        // ReLU: max(0, x)
    ACT_SIGMOID,     // Sigmoid: 1 / (1 + exp(-x))
    ACT_TANH,        // Tanh: tanh(x)
    ACT_SOFTMAX      // Softmax: exp(x_i) / sum(exp(x))
};

/**
 * @brief Simple MLP class for inference
 */
class MLP {
public:
    /**
     * @brief Constructor
     * @param layer_sizes Vector of layer sizes, e.g., {4, 16, 8, 3}
     * @param activations Vector of activation functions for each layer (except input layer)
     */
    MLP(const std::vector<int>& layer_sizes, const std::vector<ActivationType>& activations = {});

    /**
     * @brief Destructor
     */
    ~MLP();

    /**
     * @brief Set weights for a specific layer
     * @param layer_idx Layer index (0 = first hidden layer)
     * @param weights Pointer to weight matrix data (row-major: out_features x in_features)
     */
    void set_weights(int layer_idx, const float* weights);

    /**
     * @brief Set bias for a specific layer
     * @param layer_idx Layer index (0 = first hidden layer)
     * @param bias Pointer to bias vector (size = out_features)
     */
    void set_bias(int layer_idx, const float* bias);

    /**
     * @brief Forward pass inference
     * @param input Input vector (size = input_size)
     * @param output Output vector (size = output_size, must be pre-allocated)
     */
    void forward(const float* input, float* output);

    /**
     * @brief Get number of layers (including input and output)
     */
    int get_num_layers() const { return num_layers_; }

    /**
     * @brief Get size of a specific layer
     */
    int get_layer_size(int layer_idx) const { return layer_sizes_[layer_idx]; }

    /**
     * @brief Print network architecture (for debugging)
     */
    void print_architecture() const;

private:
    /**
     * @brief Apply activation function element-wise
     */
    void apply_activation(tiny::Mat& mat, ActivationType act_type);

    /**
     * @brief Apply ReLU activation
     */
    static float relu(float x) { return x > 0.0f ? x : 0.0f; }

    /**
     * @brief Apply Sigmoid activation
     */
    static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

    /**
     * @brief Apply Tanh activation
     */
    static float tanh_act(float x) { return tanhf(x); }

    /**
     * @brief Apply Softmax activation
     */
    void softmax(tiny::Mat& mat);

    std::vector<int> layer_sizes_;              // Size of each layer
    std::vector<ActivationType> activations_;   // Activation function for each layer
    int num_layers_;                            // Total number of layers
    std::vector<tiny::Mat> weights_;            // Weight matrices for each layer
    std::vector<tiny::Mat> biases_;             // Bias vectors for each layer
    std::vector<tiny::Mat> activations_cache_;  // Cache for intermediate activations
};

} // namespace tiny_ai

#endif // __cplusplus
