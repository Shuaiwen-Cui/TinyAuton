/**
 * @file mlp_iris_demo.h
 * @brief Iris Classification MLP Example - Based on PC-AI Project Model
 * @version 1.0
 * @date 2025-01-XX
 * 
 * @details
 * Complete Iris classification MLP example demonstrating neural network inference on ESP32S3.
 * Model architecture: 4 inputs -> 16 hidden(ReLU) -> 8 hidden(ReLU) -> 3 outputs
 * 
 * Features:
 * - Load trained model weights
 * - Data standardization
 * - Forward inference
 * - Performance benchmarking (inference time, memory usage)
 * - Accuracy testing
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Run complete Iris MLP example (including performance test)
 */
void mlp_iris_demo_run(void);

/**
 * @brief Run performance benchmark test
 * @param num_iterations Number of test iterations
 */
void mlp_iris_demo_benchmark(int num_iterations);

#ifdef __cplusplus
}
#endif

