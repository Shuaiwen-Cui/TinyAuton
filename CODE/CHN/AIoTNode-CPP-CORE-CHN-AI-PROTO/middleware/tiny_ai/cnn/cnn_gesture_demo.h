/**
 * @file cnn_gesture_demo.h
 * @brief Gesture Recognition 1D CNN Example - Based on PC-AI Project Model
 * @version 1.0
 * @date 2025-01-XX
 * 
 * @details
 * Complete gesture recognition 1D CNN example demonstrating neural network inference on ESP32S3.
 * Model architecture: 3 channels -> Conv1D(8) -> Conv1D(16) -> GlobalAvgPool -> FC(5)
 * Input: 3-axis accelerometer data (3 channels, 64 time steps)
 * Output: 5 gesture classes (swipe_left, swipe_right, tap, circle, wave)
 * 
 * Features:
 * - Load trained model weights
 * - Data standardization
 * - Forward inference
 * - Performance benchmarking
 * - Accuracy testing
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Run complete Gesture CNN example (including performance test)
 */
void cnn_gesture_demo_run(void);

/**
 * @brief Run performance benchmark test
 * @param num_iterations Number of test iterations
 */
void cnn_gesture_demo_benchmark(int num_iterations);

#ifdef __cplusplus
}
#endif

