/**
 * @file mlp_iris_demo.cpp
 * @brief Iris Classification MLP Example Implementation
 * 
 * Simple example demonstrating how to use MLP for Iris classification
 */

#include "mlp_iris_demo.h"
#include "iris_mlp_weights.h"
#include "mlp_example.hpp"
#include <stdio.h>
#include <vector>
#include <string.h>

#ifdef ESP_PLATFORM
#include "esp_timer.h"
#include "esp_heap_caps.h"
#endif

using namespace tiny_ai;

/**
 * @brief Data standardization function (StandardScaler: (x - mean) / std)
 * @param input Input data (will be modified to standardized values)
 * @param size Data dimension (should be 4)
 */
static void standardize_input(float* input, int size)
{
    if (size != IRIS_MLP_INPUT_SIZE) {
        printf("Warning: Input size mismatch for standardization\n");
        return;
    }
    
    // StandardScaler: (x - mean) / std
    for (int i = 0; i < size; i++) {
        input[i] = (input[i] - iris_scaler_mean[i]) / iris_scaler_std[i];
    }
}

/**
 * @brief Run Iris MLP basic example
 */
void mlp_iris_demo_run(void)
{
    printf("\n");
    printf("========================================\n");
    printf("Iris MLP Classification Example\n");
    printf("========================================\n");
    printf("Model Architecture: 4 inputs -> 16 hidden(ReLU) -> 8 hidden(ReLU) -> 3 outputs\n\n");

    // 1. Create MLP network
    std::vector<int> layer_sizes = {
        IRIS_MLP_INPUT_SIZE,
        IRIS_MLP_HIDDEN1_SIZE,
        IRIS_MLP_HIDDEN2_SIZE,
        IRIS_MLP_OUTPUT_SIZE
    };
    
    std::vector<ActivationType> activations = {
        ACT_RELU,   // First hidden layer
        ACT_RELU,   // Second hidden layer
        ACT_LINEAR  // Output layer (usually Softmax for classification, but using Linear and finding max manually)
    };
    
    MLP mlp(layer_sizes, activations);
    
    // 2. Set weights and biases
    mlp.set_weights(0, iris_mlp_weights_layer0);
    mlp.set_bias(0, iris_mlp_bias_layer0);
    mlp.set_weights(1, iris_mlp_weights_layer1);
    mlp.set_bias(1, iris_mlp_bias_layer1);
    mlp.set_weights(2, iris_mlp_weights_layer2);
    mlp.set_bias(2, iris_mlp_bias_layer2);
    
    printf("Model weights loaded\n\n");

    // 3. Prepare test data (Iris samples) - Same as Python project test set
    // Format: [sepal_length, sepal_width, petal_length, petal_width]
    // Test set from Python project: train_test_split(test_size=0.2, random_state=42, stratify=y)
    float test_inputs[][4] = {
        {4.4f, 3.0f, 1.3f, 0.2f},  // Setosa (class 0)
        {6.1f, 3.0f, 4.9f, 1.8f},  // Virginica (class 2)
        {4.9f, 2.4f, 3.3f, 1.0f},  // Versicolor (class 1)
        {5.0f, 2.3f, 3.3f, 1.0f},  // Versicolor (class 1)
        {4.4f, 3.2f, 1.3f, 0.2f},  // Setosa (class 0)
        {6.3f, 3.3f, 4.7f, 1.6f},  // Versicolor (class 1)
        {4.6f, 3.6f, 1.0f, 0.2f},  // Setosa (class 0)
        {5.4f, 3.4f, 1.7f, 0.2f},  // Setosa (class 0)
        {6.5f, 3.0f, 5.2f, 2.0f},  // Virginica (class 2)
        {5.4f, 3.0f, 4.5f, 1.5f},  // Versicolor (class 1)
        {7.3f, 2.9f, 6.3f, 1.8f},  // Virginica (class 2)
        {6.9f, 3.1f, 5.1f, 2.3f},  // Virginica (class 2)
        {6.5f, 3.0f, 5.8f, 2.2f},  // Virginica (class 2)
        {6.4f, 3.2f, 4.5f, 1.5f},  // Versicolor (class 1)
        {5.0f, 3.4f, 1.5f, 0.2f},  // Setosa (class 0)
        {5.0f, 3.3f, 1.4f, 0.2f},  // Setosa (class 0)
        {5.8f, 4.0f, 1.2f, 0.2f},  // Setosa (class 0)
        {5.6f, 2.5f, 3.9f, 1.1f},  // Versicolor (class 1)
        {6.1f, 2.9f, 4.7f, 1.4f},  // Versicolor (class 1)
        {6.0f, 3.0f, 4.8f, 1.8f},  // Virginica (class 2)
        {5.4f, 3.7f, 1.5f, 0.2f},  // Setosa (class 0)
        {6.7f, 3.1f, 5.6f, 2.4f},  // Virginica (class 2)
        {6.6f, 2.9f, 4.6f, 1.3f},  // Versicolor (class 1)
        {6.1f, 2.6f, 5.6f, 1.4f},  // Virginica (class 2)
        {6.4f, 2.8f, 5.6f, 2.2f},  // Virginica (class 2)
        {6.7f, 3.0f, 5.0f, 1.7f},  // Versicolor (class 1)
        {6.6f, 3.0f, 4.4f, 1.4f},  // Versicolor (class 1)
        {5.7f, 3.8f, 1.7f, 0.3f},  // Setosa (class 0)
        {6.5f, 3.0f, 5.5f, 1.8f},  // Virginica (class 2)
        {5.2f, 3.4f, 1.4f, 0.2f}   // Setosa (class 0)
    };
    
    const char* class_names[] = {"Setosa", "Versicolor", "Virginica"};
    int expected_classes[] = {
        0, 2, 1, 1, 0, 1, 0, 0, 2, 1,
        2, 2, 2, 1, 0, 0, 0, 1, 1, 2,
        0, 2, 1, 2, 2, 1, 1, 0, 2, 0
    };
    int num_test = 30;

    // 4. Run inference
    printf("Starting inference test...\n");
    printf("----------------------------------------\n");
    
    float output[IRIS_MLP_OUTPUT_SIZE];
    int correct = 0;
    
    for (int i = 0; i < num_test; i++) {
        // Prepare input data
        float input[4];
        for (int j = 0; j < 4; j++) {
            input[j] = test_inputs[i][j];
        }
        
        // Standardize input (consistent with PC-AI project)
        float input_normalized[4];
        memcpy(input_normalized, input, sizeof(input));
        standardize_input(input_normalized, 4);
        
        // Forward pass
        mlp.forward(input_normalized, output);
        
        // Find index of maximum output (predicted class)
        int predicted_class = 0;
        float max_output = output[0];
        for (int j = 1; j < IRIS_MLP_OUTPUT_SIZE; j++) {
            if (output[j] > max_output) {
                max_output = output[j];
                predicted_class = j;
            }
        }
        
        // Print results for all samples
        printf("Sample %2d: Input [%.1f, %.1f, %.1f, %.1f]\n",
               i + 1, input[0], input[1], input[2], input[3]);
        printf("  Output: [%.4f, %.4f, %.4f]\n",
               output[0], output[1], output[2]);
        printf("  Predicted: %s (class %d), Expected: %s (class %d) %s\n",
               class_names[predicted_class], predicted_class,
               class_names[expected_classes[i]], expected_classes[i],
               (predicted_class == expected_classes[i]) ? "OK" : "FAIL");
        
        if (predicted_class == expected_classes[i]) {
            correct++;
        } else {
            printf("  *** MISMATCH ***\n");
        }
        printf("\n");
    }
    
    printf("----------------------------------------\n");
    printf("Test Results Summary:\n");
    printf("  Total samples: %d\n", num_test);
    printf("  Correct predictions: %d\n", correct);
    printf("  Incorrect predictions: %d\n", num_test - correct);
    printf("  Accuracy: %d/%d = %.2f%%\n", correct, num_test, 100.0f * correct / num_test);
    printf("========================================\n");
    printf("Basic example completed\n");
    printf("========================================\n\n");
}

/**
 * @brief Run performance benchmark test
 */
void mlp_iris_demo_benchmark(int num_iterations)
{
    printf("\n");
    printf("========================================\n");
    printf("Iris MLP Performance Benchmark\n");
    printf("========================================\n");
    printf("Iterations: %d\n\n", num_iterations);

    // 1. Create and initialize MLP
    std::vector<int> layer_sizes = {
        IRIS_MLP_INPUT_SIZE,
        IRIS_MLP_HIDDEN1_SIZE,
        IRIS_MLP_HIDDEN2_SIZE,
        IRIS_MLP_OUTPUT_SIZE
    };
    
    std::vector<ActivationType> activations = {
        ACT_RELU,
        ACT_RELU,
        ACT_LINEAR
    };
    
    MLP mlp(layer_sizes, activations);
    mlp.set_weights(0, iris_mlp_weights_layer0);
    mlp.set_bias(0, iris_mlp_bias_layer0);
    mlp.set_weights(1, iris_mlp_weights_layer1);
    mlp.set_bias(1, iris_mlp_bias_layer1);
    mlp.set_weights(2, iris_mlp_weights_layer2);
    mlp.set_bias(2, iris_mlp_bias_layer2);

    // 2. Prepare test input (using one sample)
    float test_input[4] = {5.1f, 3.5f, 1.4f, 0.2f};
    standardize_input(test_input, 4);
    
    float output[IRIS_MLP_OUTPUT_SIZE];

#ifdef ESP_PLATFORM
    // 3. Measure memory usage (ESP32)
    size_t free_heap_before = esp_get_free_heap_size();
    size_t min_free_heap_before = esp_get_minimum_free_heap_size();
    
    // 4. Performance test - measure inference time
    int64_t start_time = esp_timer_get_time();
    
    for (int i = 0; i < num_iterations; i++) {
        mlp.forward(test_input, output);
    }
    
    int64_t end_time = esp_timer_get_time();
    int64_t total_time_us = end_time - start_time;
    float avg_time_us = (float)total_time_us / num_iterations;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    // 5. Measure memory usage (ESP32)
    size_t free_heap_after = esp_get_free_heap_size();
    size_t min_free_heap_after = esp_get_minimum_free_heap_size();
    size_t heap_used = (free_heap_before > free_heap_after) ? (free_heap_before - free_heap_after) : 0;
    
    // 6. Print performance results
    printf("Performance Test Results:\n");
    printf("----------------------------------------\n");
    printf("Total inference time: %.2f ms (%.2f us)\n", total_time_us / 1000.0f, (float)total_time_us);
    printf("Average inference time: %.4f ms (%.2f us)\n", avg_time_ms, avg_time_us);
    printf("Inference speed: %.2f inferences/sec\n", 1000000.0f / avg_time_us);
    printf("\n");
    printf("Memory Usage:\n");
    printf("  Free heap before: %zu bytes\n", free_heap_before);
    printf("  Free heap after: %zu bytes\n", free_heap_after);
    printf("  Heap used: %zu bytes\n", heap_used);
    printf("  Min free heap (before): %zu bytes\n", min_free_heap_before);
    printf("  Min free heap (after): %zu bytes\n", min_free_heap_after);
    
#else
    // Simple time measurement for non-ESP32 platforms
    printf("Performance Test Results (Non-ESP32 Platform):\n");
    printf("----------------------------------------\n");
    printf("Completed %d inferences\n", num_iterations);
    printf("Note: ESP32 platform required for accurate performance data\n");
#endif

    printf("========================================\n");
    printf("Performance test completed\n");
    printf("========================================\n\n");
}

