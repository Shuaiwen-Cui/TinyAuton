/**
 * @file mlp_example.cpp
 * @brief Implementation of MLP example
 */

#include "mlp_example.hpp"
#include <stdio.h>
#include <string.h>

namespace tiny_ai {

MLP::MLP(const std::vector<int>& layer_sizes, const std::vector<ActivationType>& activations)
    : layer_sizes_(layer_sizes), num_layers_(layer_sizes.size())
{
    // Validate input
    if (layer_sizes.size() < 2) {
        printf("Error: MLP requires at least 2 layers (input and output)\n");
        return;
    }

    // Set default activations if not provided
    activations_.resize(num_layers_ - 1);
    if (activations.empty()) {
        // Default: ReLU for hidden layers, Linear for output
        for (int i = 0; i < num_layers_ - 2; i++) {
            activations_[i] = ACT_RELU;
        }
        activations_[num_layers_ - 2] = ACT_LINEAR;
    } else {
        // Use provided activations
        for (size_t i = 0; i < activations.size() && i < activations_.size(); i++) {
            activations_[i] = activations[i];
        }
        // Fill remaining with default if needed
        for (size_t i = activations.size(); i < activations_.size(); i++) {
            activations_[i] = (i == activations_.size() - 1) ? ACT_LINEAR : ACT_RELU;
        }
    }

    // Allocate weight matrices and bias vectors
    weights_.resize(num_layers_ - 1);
    biases_.resize(num_layers_ - 1);
    activations_cache_.resize(num_layers_);

    // Initialize weight matrices: weights_[i] is (layer_sizes[i+1] x layer_sizes[i])
    for (int i = 0; i < num_layers_ - 1; i++) {
        weights_[i] = tiny::Mat(layer_sizes[i + 1], layer_sizes[i]);
        biases_[i] = tiny::Mat(layer_sizes[i + 1], 1);
        
        // Initialize to zero (user should set actual weights)
        weights_[i].clear();
        biases_[i].clear();
    }

    // Initialize activation cache
    for (int i = 0; i < num_layers_; i++) {
        activations_cache_[i] = tiny::Mat(layer_sizes[i], 1);
    }
}

MLP::~MLP() {
    // Mat destructor will handle memory cleanup
}

void MLP::set_weights(int layer_idx, const float* weights) {
    if (layer_idx < 0 || layer_idx >= num_layers_ - 1) {
        printf("Error: Invalid layer index %d\n", layer_idx);
        return;
    }

    int rows = layer_sizes_[layer_idx + 1];
    int cols = layer_sizes_[layer_idx];
    int size = rows * cols;

    // Copy weights (assuming row-major order)
    memcpy(weights_[layer_idx].data, weights, size * sizeof(float));
}

void MLP::set_bias(int layer_idx, const float* bias) {
    if (layer_idx < 0 || layer_idx >= num_layers_ - 1) {
        printf("Error: Invalid layer index %d\n", layer_idx);
        return;
    }

    int size = layer_sizes_[layer_idx + 1];
    memcpy(biases_[layer_idx].data, bias, size * sizeof(float));
}

void MLP::forward(const float* input, float* output) {
    // Copy input to first activation cache
    int input_size = layer_sizes_[0];
    memcpy(activations_cache_[0].data, input, input_size * sizeof(float));

    // Forward pass through each layer
    for (int i = 0; i < num_layers_ - 1; i++) {
        // Compute: output = weights * input + bias
        // activations_cache_[i+1] = weights_[i] * activations_cache_[i] + biases_[i]
        
        // Matrix multiplication: weights_[i] * activations_cache_[i]
        tiny::Mat temp = weights_[i] * activations_cache_[i];
        
        // Add bias
        activations_cache_[i + 1] = temp + biases_[i];
        
        // Apply activation function
        apply_activation(activations_cache_[i + 1], activations_[i]);
    }

    // Copy output
    int output_size = layer_sizes_[num_layers_ - 1];
    memcpy(output, activations_cache_[num_layers_ - 1].data, output_size * sizeof(float));
}

void MLP::apply_activation(tiny::Mat& mat, ActivationType act_type) {
    int size = mat.row * mat.col;
    
    switch (act_type) {
        case ACT_LINEAR:
            // No operation needed
            break;
            
        case ACT_RELU:
            for (int i = 0; i < size; i++) {
                mat.data[i] = relu(mat.data[i]);
            }
            break;
            
        case ACT_SIGMOID:
            for (int i = 0; i < size; i++) {
                mat.data[i] = sigmoid(mat.data[i]);
            }
            break;
            
        case ACT_TANH:
            for (int i = 0; i < size; i++) {
                mat.data[i] = tanh_act(mat.data[i]);
            }
            break;
            
        case ACT_SOFTMAX:
            softmax(mat);
            break;
    }
}

void MLP::softmax(tiny::Mat& mat) {
    int size = mat.row * mat.col;
    
    // Find maximum for numerical stability
    float max_val = mat.data[0];
    for (int i = 1; i < size; i++) {
        if (mat.data[i] > max_val) {
            max_val = mat.data[i];
        }
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        mat.data[i] = expf(mat.data[i] - max_val);
        sum += mat.data[i];
    }
    
    // Normalize
    if (sum > 0.0f) {
        for (int i = 0; i < size; i++) {
            mat.data[i] /= sum;
        }
    }
}

void MLP::print_architecture() const {
    printf("MLP Architecture:\n");
    printf("  Input layer: %d neurons\n", layer_sizes_[0]);
    
    for (int i = 1; i < num_layers_ - 1; i++) {
        const char* act_name = "Unknown";
        switch (activations_[i - 1]) {
            case ACT_LINEAR: act_name = "Linear"; break;
            case ACT_RELU: act_name = "ReLU"; break;
            case ACT_SIGMOID: act_name = "Sigmoid"; break;
            case ACT_TANH: act_name = "Tanh"; break;
            case ACT_SOFTMAX: act_name = "Softmax"; break;
        }
        printf("  Hidden layer %d: %d neurons, activation: %s\n", i, layer_sizes_[i], act_name);
    }
    
    if (num_layers_ > 1) {
        const char* act_name = "Unknown";
        switch (activations_[num_layers_ - 2]) {
            case ACT_LINEAR: act_name = "Linear"; break;
            case ACT_RELU: act_name = "ReLU"; break;
            case ACT_SIGMOID: act_name = "Sigmoid"; break;
            case ACT_TANH: act_name = "Tanh"; break;
            case ACT_SOFTMAX: act_name = "Softmax"; break;
        }
        printf("  Output layer: %d neurons, activation: %s\n", 
               layer_sizes_[num_layers_ - 1], act_name);
    }
}

} // namespace tiny_ai

