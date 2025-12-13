/**
 * @file cnn_example.hpp
 * @brief Simple 1D CNN (Convolutional Neural Network) for microcontroller AI
 * @version 1.0
 * 
 * @details
 * Simple 1D CNN implementation for microcontroller AI inference
 * Based on tiny_math Mat class for matrix operations
 */

#pragma once

#include "tiny_math.h"
#include <vector>
#include <cmath>
#include <string.h>

#ifdef __cplusplus

namespace tiny_ai {

/**
 * @brief Simple 1D CNN class for inference
 * 
 * Model structure:
 * - Conv1D: in_channels -> out_channels, kernel_size, padding
 * - ReLU activation
 * - Global Average Pooling
 * - Fully Connected layer
 */
class Simple1DCNN {
public:
    /**
     * @brief Constructor
     * @param in_channels Input channels (e.g., 3 for accelerometer data)
     * @param conv1_out_channels First conv layer output channels
     * @param conv2_out_channels Second conv layer output channels
     * @param num_classes Number of output classes
     * @param seq_len Sequence length (time steps)
     * @param kernel_size Convolution kernel size (default 3)
     */
    Simple1DCNN(int in_channels, int conv1_out_channels, int conv2_out_channels, 
                int num_classes, int seq_len, int kernel_size = 3);

    /**
     * @brief Destructor
     */
    ~Simple1DCNN();

    /**
     * @brief Set weights for first conv layer
     * @param weights Weight tensor (out_channels, in_channels, kernel_size)
     */
    void set_conv1_weights(const float* weights);

    /**
     * @brief Set bias for first conv layer
     * @param bias Bias vector (out_channels)
     */
    void set_conv1_bias(const float* bias);

    /**
     * @brief Set weights for second conv layer
     * @param weights Weight tensor (out_channels, in_channels, kernel_size)
     */
    void set_conv2_weights(const float* weights);

    /**
     * @brief Set bias for second conv layer
     * @param bias Bias vector (out_channels)
     */
    void set_conv2_bias(const float* bias);

    /**
     * @brief Set weights for fully connected layer
     * @param weights Weight matrix (num_classes, conv2_out_channels)
     */
    void set_fc_weights(const float* weights);

    /**
     * @brief Set bias for fully connected layer
     * @param bias Bias vector (num_classes)
     */
    void set_fc_bias(const float* bias);

    /**
     * @brief Forward pass inference
     * @param input Input tensor (in_channels, seq_len) - row-major
     * @param output Output vector (num_classes, must be pre-allocated)
     */
    void forward(const float* input, float* output);

    /**
     * @brief Print network architecture (for debugging)
     */
    void print_architecture() const;

private:
    /**
     * @brief Apply 1D convolution with padding
     * @param input Input tensor (in_channels, seq_len)
     * @param output Output tensor (out_channels, seq_len)
     * @param weights Weight tensor (out_channels, in_channels, kernel_size)
     * @param bias Bias vector (out_channels)
     * @param in_channels Input channels
     * @param out_channels Output channels
     * @param seq_len Sequence length
     */
    void conv1d(const float* input, float* output, const float* weights, const float* bias,
                int in_channels, int out_channels, int seq_len);

    /**
     * @brief Apply ReLU activation
     */
    static float relu(float x) { return x > 0.0f ? x : 0.0f; }

    /**
     * @brief Apply global average pooling
     * @param input Input tensor (channels, seq_len)
     * @param output Output vector (channels)
     * @param channels Number of channels
     * @param seq_len Sequence length
     */
    void global_avg_pool(const float* input, float* output, int channels, int seq_len);

    int in_channels_;
    int conv1_out_channels_;
    int conv2_out_channels_;
    int num_classes_;
    int seq_len_;
    int kernel_size_;
    
    // Weight and bias storage
    std::vector<float> conv1_weights_;
    std::vector<float> conv1_bias_;
    std::vector<float> conv2_weights_;
    std::vector<float> conv2_bias_;
    std::vector<float> fc_weights_;
    std::vector<float> fc_bias_;
    
    // Intermediate buffers
    std::vector<float> conv1_output_;
    std::vector<float> conv2_output_;
    std::vector<float> pooled_output_;
};

} // namespace tiny_ai

#endif // __cplusplus

