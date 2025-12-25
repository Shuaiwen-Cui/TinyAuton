/**
 * @file cnn_example.cpp
 * @brief Implementation of 1D CNN example
 */

#include "cnn_example.hpp"
#include <stdio.h>
#include <string.h>

namespace tiny_ai {

Simple1DCNN::Simple1DCNN(int in_channels, int conv1_out_channels, int conv2_out_channels,
                         int num_classes, int seq_len, int kernel_size)
    : in_channels_(in_channels),
      conv1_out_channels_(conv1_out_channels),
      conv2_out_channels_(conv2_out_channels),
      num_classes_(num_classes),
      seq_len_(seq_len),
      kernel_size_(kernel_size)
{
    // Allocate weight and bias storage
    conv1_weights_.resize(conv1_out_channels * in_channels * kernel_size);
    conv1_bias_.resize(conv1_out_channels);
    conv2_weights_.resize(conv2_out_channels * conv1_out_channels * kernel_size);
    conv2_bias_.resize(conv2_out_channels);
    fc_weights_.resize(num_classes * conv2_out_channels);
    fc_bias_.resize(num_classes);
    
    // Allocate intermediate buffers
    conv1_output_.resize(conv1_out_channels * seq_len);
    conv2_output_.resize(conv2_out_channels * seq_len);
    pooled_output_.resize(conv2_out_channels);
    
    // Initialize to zero
    memset(conv1_weights_.data(), 0, conv1_weights_.size() * sizeof(float));
    memset(conv1_bias_.data(), 0, conv1_bias_.size() * sizeof(float));
    memset(conv2_weights_.data(), 0, conv2_weights_.size() * sizeof(float));
    memset(conv2_bias_.data(), 0, conv2_bias_.size() * sizeof(float));
    memset(fc_weights_.data(), 0, fc_weights_.size() * sizeof(float));
    memset(fc_bias_.data(), 0, fc_bias_.size() * sizeof(float));
}

Simple1DCNN::~Simple1DCNN() {
    // Vectors will handle cleanup automatically
}

void Simple1DCNN::set_conv1_weights(const float* weights) {
    memcpy(conv1_weights_.data(), weights, conv1_weights_.size() * sizeof(float));
}

void Simple1DCNN::set_conv1_bias(const float* bias) {
    memcpy(conv1_bias_.data(), bias, conv1_bias_.size() * sizeof(float));
}

void Simple1DCNN::set_conv2_weights(const float* weights) {
    memcpy(conv2_weights_.data(), weights, conv2_weights_.size() * sizeof(float));
}

void Simple1DCNN::set_conv2_bias(const float* bias) {
    memcpy(conv2_bias_.data(), bias, conv2_bias_.size() * sizeof(float));
}

void Simple1DCNN::set_fc_weights(const float* weights) {
    memcpy(fc_weights_.data(), weights, fc_weights_.size() * sizeof(float));
}

void Simple1DCNN::set_fc_bias(const float* bias) {
    memcpy(fc_bias_.data(), bias, fc_bias_.size() * sizeof(float));
}

void Simple1DCNN::conv1d(const float* input, float* output, const float* weights, const float* bias,
                         int in_channels, int out_channels, int seq_len) {
    // 1D convolution with padding=1 (kernel_size=3)
    // Input: (in_channels, seq_len) - row-major: input[ch * seq_len + t]
    // Output: (out_channels, seq_len) - row-major: output[ch * seq_len + t]
    // Weights: (out_channels, in_channels, kernel_size) - row-major
    
    int pad = kernel_size_ / 2;  // padding = 1 for kernel_size = 3
    
    for (int out_ch = 0; out_ch < out_channels; out_ch++) {
        for (int t = 0; t < seq_len; t++) {
            float sum = 0.0f;
            
            // Convolution operation
            for (int in_ch = 0; in_ch < in_channels; in_ch++) {
                for (int k = 0; k < kernel_size_; k++) {
                    int t_idx = t + k - pad;  // Account for padding
                    
                    // Handle padding (zero-padding)
                    if (t_idx >= 0 && t_idx < seq_len) {
                        float input_val = input[in_ch * seq_len + t_idx];
                        float weight_val = weights[out_ch * (in_channels * kernel_size_) + 
                                                   in_ch * kernel_size_ + k];
                        sum += input_val * weight_val;
                    }
                }
            }
            
            // Add bias
            sum += bias[out_ch];
            
            output[out_ch * seq_len + t] = sum;
        }
    }
}

void Simple1DCNN::global_avg_pool(const float* input, float* output, int channels, int seq_len) {
    // Global average pooling: average over sequence dimension
    for (int ch = 0; ch < channels; ch++) {
        float sum = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            sum += input[ch * seq_len + t];
        }
        output[ch] = sum / seq_len;
    }
}

void Simple1DCNN::forward(const float* input, float* output) {
    // Step 1: First convolution
    conv1d(input, conv1_output_.data(), conv1_weights_.data(), conv1_bias_.data(),
           in_channels_, conv1_out_channels_, seq_len_);
    
    // Step 2: ReLU activation on conv1 output
    for (size_t i = 0; i < conv1_output_.size(); i++) {
        conv1_output_[i] = relu(conv1_output_[i]);
    }
    
    // Step 3: Second convolution
    conv1d(conv1_output_.data(), conv2_output_.data(), conv2_weights_.data(), conv2_bias_.data(),
           conv1_out_channels_, conv2_out_channels_, seq_len_);
    
    // Step 4: ReLU activation on conv2 output
    for (size_t i = 0; i < conv2_output_.size(); i++) {
        conv2_output_[i] = relu(conv2_output_[i]);
    }
    
    // Step 5: Global average pooling
    global_avg_pool(conv2_output_.data(), pooled_output_.data(), conv2_out_channels_, seq_len_);
    
    // Step 6: Fully connected layer
    for (int cls = 0; cls < num_classes_; cls++) {
        float sum = 0.0f;
        for (int ch = 0; ch < conv2_out_channels_; ch++) {
            sum += pooled_output_[ch] * fc_weights_[cls * conv2_out_channels_ + ch];
        }
        output[cls] = sum + fc_bias_[cls];
    }
}

void Simple1DCNN::print_architecture() const {
    printf("1D CNN Architecture:\n");
    printf("  Input: (%d channels, %d time steps)\n", in_channels_, seq_len_);
    printf("  Conv1: %d -> %d channels, kernel_size=%d, padding=1\n", 
           in_channels_, conv1_out_channels_, kernel_size_);
    printf("  ReLU\n");
    printf("  Conv2: %d -> %d channels, kernel_size=%d, padding=1\n", 
           conv1_out_channels_, conv2_out_channels_, kernel_size_);
    printf("  ReLU\n");
    printf("  Global Average Pooling\n");
    printf("  FC: %d -> %d classes\n", conv2_out_channels_, num_classes_);
}

} // namespace tiny_ai

