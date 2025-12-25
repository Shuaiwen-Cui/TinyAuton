/**
 * @file cnn_gesture_demo.cpp
 * @brief Gesture Recognition 1D CNN Example Implementation
 * 
 * Simple example demonstrating how to use 1D CNN for gesture recognition
 */

#include "cnn_gesture_demo.h"
#include "gesture_cnn_weights.h"
#include "cnn_example.hpp"
#include <stdio.h>
#include <string.h>

#ifdef ESP_PLATFORM
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_task_wdt.h"
#endif

using namespace tiny_ai;

// Test data from Python project test set
// First test sample (class 4)
static const float test_input_0[3 * 64] = {
    // Channel 0
    0.042261f, 0.056194f, 0.337932f, 0.109831f, 0.392982f, 0.014657f, 0.363591f, 0.367090f, 0.242784f, 0.472558f, 0.373349f, 0.286688f, 0.218153f, 0.128302f, 0.039146f, 0.230079f, 
    -0.075614f, 0.128275f, -0.077109f, -0.015333f, -0.113465f, -0.447770f, -0.267374f, -0.393923f, -0.220294f, -0.269237f, -0.255871f, -0.217917f, -0.139271f, -0.127523f, -0.227036f, -0.124798f, 
    0.062893f, 0.230666f, 0.128289f, 0.137228f, 0.397493f, 0.135555f, 0.231144f, 0.250083f, 0.183237f, 0.128797f, 0.176020f, 0.387573f, 0.210235f, 0.171357f, 0.130156f, 0.257760f, 
    -0.162998f, -0.020660f, -0.033493f, -0.149960f, -0.315523f, -0.355840f, 0.022043f, -0.395544f, -0.214936f, -0.296245f, -0.203090f, -0.248420f, -0.165693f, -0.128689f, -0.093054f, 0.065954f,
    // Channel 1
    0.169320f, 0.037383f, 0.069451f, -0.022266f, 0.118163f, -0.196642f, 0.024947f, -0.042768f, 0.203190f, -0.111166f, -0.032931f, -0.119780f, 0.005824f, -0.068542f, 0.055333f, 0.043217f, 
    -0.030066f, -0.074661f, 0.054227f, 0.053915f, -0.033267f, 0.067225f, 0.035028f, -0.007694f, 0.042087f, -0.065251f, -0.079981f, 0.146581f, 0.054682f, 0.079683f, 0.000789f, -0.010399f, 
    -0.020268f, -0.045201f, 0.088560f, -0.050371f, -0.075897f, -0.047780f, -0.012937f, -0.116446f, 0.098113f, -0.075578f, 0.120667f, -0.123681f, 0.181744f, -0.170623f, 0.090938f, 0.015690f, 
    -0.083214f, 0.129383f, -0.045216f, 0.148948f, 0.110531f, 0.229932f, -0.073005f, 0.148468f, -0.046597f, -0.006744f, 0.031411f, 0.063414f, -0.033663f, 0.002044f, 0.052291f, 0.124839f,
    // Channel 2
    0.931636f, 1.114916f, 1.103107f, 1.062139f, 0.967710f, 1.066604f, 0.942766f, 0.882565f, 0.969853f, 1.057263f, 0.994511f, 0.940443f, 0.934004f, 0.991814f, 0.946734f, 1.132807f, 
    0.944710f, 1.037775f, 1.097000f, 0.839740f, 0.915200f, 0.898474f, 0.906576f, 1.031665f, 1.097980f, 0.915666f, 0.903722f, 1.049507f, 1.274518f, 0.838121f, 0.970905f, 1.161145f, 
    1.280787f, 0.991545f, 1.076017f, 1.183255f, 0.949439f, 1.100398f, 1.126082f, 1.007190f, 1.010873f, 1.081573f, 0.912303f, 1.096438f, 0.825182f, 0.858344f, 0.917625f, 1.062600f, 
    0.986608f, 0.960281f, 1.180688f, 0.944702f, 0.930817f, 1.007396f, 0.944609f, 1.071235f, 1.003288f, 0.974947f, 1.122478f, 1.096077f, 0.897709f, 0.911907f, 1.147435f, 1.077263f
};

static const int expected_class_0 = 4;

// Gesture class names
static const char* gesture_names[] = {
    "swipe_left",   // 0
    "swipe_right",  // 1
    "tap",          // 2
    "circle",       // 3
    "wave"          // 4
};

/**
 * @brief Data standardization function (StandardScaler: (x - mean) / std)
 * @param input Input data (3 channels, seq_len) - will be modified
 * @param channels Number of channels (3)
 * @param seq_len Sequence length (64)
 * @param scaler_mean Mean values for each timestep (64 values)
 * @param scaler_std Standard deviation values for each timestep (64 values)
 * 
 * Note: Python StandardScaler standardizes per timestep (across all channels and samples),
 * not per channel. So for each timestep t, we use scaler_mean[t] and scaler_std[t].
 */
static void standardize_input(float* input, int channels, int seq_len,
                             const float* scaler_mean, const float* scaler_std)
{
    // StandardScaler: (x - mean) / std
    // Applied per timestep (not per channel)
    // For each timestep t, standardize all channels using mean[t] and std[t]
    for (int t = 0; t < seq_len; t++) {
        for (int ch = 0; ch < channels; ch++) {
            int idx = ch * seq_len + t;
            input[idx] = (input[idx] - scaler_mean[t]) / scaler_std[t];
        }
    }
}

/**
 * @brief Run Gesture CNN basic example
 */
void cnn_gesture_demo_run(void)
{
    printf("\n");
    printf("========================================\n");
    printf("Gesture Recognition 1D CNN Example\n");
    printf("========================================\n");
    printf("Model Architecture: 3 channels -> Conv1D(8) -> Conv1D(16) -> GlobalAvgPool -> FC(5)\n");
    printf("Input: 3-axis accelerometer data (3 channels, 64 time steps)\n");
    printf("Output: 5 gesture classes\n\n");

    // 1. Create CNN network
    Simple1DCNN cnn(GESTURE_CNN_IN_CHANNELS,
                    GESTURE_CNN_CONV1_OUT_CHANNELS,
                    GESTURE_CNN_CONV2_OUT_CHANNELS,
                    GESTURE_CNN_NUM_CLASSES,
                    GESTURE_CNN_SEQ_LEN,
                    3);  // kernel_size = 3
    
    // 2. Set weights and biases
    cnn.set_conv1_weights(gesture_cnn_conv1_weights);
    cnn.set_conv1_bias(gesture_cnn_conv1_bias);
    cnn.set_conv2_weights(gesture_cnn_conv2_weights);
    cnn.set_conv2_bias(gesture_cnn_conv2_bias);
    cnn.set_fc_weights(gesture_cnn_fc_weights);
    cnn.set_fc_bias(gesture_cnn_fc_bias);
    
    printf("Model weights loaded\n\n");

    // 3. Test with first sample
    printf("Testing with first sample...\n");
    printf("----------------------------------------\n");
    
    // Prepare input data (copy from test data)
    float input[3 * 64];
    memcpy(input, test_input_0, sizeof(test_input_0));
    
    // Standardize input
    standardize_input(input, GESTURE_CNN_IN_CHANNELS, GESTURE_CNN_SEQ_LEN,
                     gesture_scaler_mean, gesture_scaler_std);
    
    // Forward pass
    float output[GESTURE_CNN_NUM_CLASSES];
    cnn.forward(input, output);
    
    // Find predicted class (index with maximum output)
    int predicted_class = 0;
    float max_output = output[0];
    for (int i = 1; i < GESTURE_CNN_NUM_CLASSES; i++) {
        if (output[i] > max_output) {
            max_output = output[i];
            predicted_class = i;
        }
    }
    
    // Print results
    printf("Output scores: [");
    for (int i = 0; i < GESTURE_CNN_NUM_CLASSES; i++) {
        printf("%.4f", output[i]);
        if (i < GESTURE_CNN_NUM_CLASSES - 1) {
            printf(", ");
        }
    }
    printf("]\n");
    printf("Predicted: %s (class %d)\n", gesture_names[predicted_class], predicted_class);
    printf("Expected:  %s (class %d) %s\n", 
           gesture_names[expected_class_0], expected_class_0,
           (predicted_class == expected_class_0) ? "OK" : "FAIL");
    printf("----------------------------------------\n");
    
    printf("========================================\n");
    printf("Basic example completed\n");
    printf("========================================\n\n");
}

/**
 * @brief Run performance benchmark test
 */
void cnn_gesture_demo_benchmark(int num_iterations)
{
    printf("\n");
    printf("========================================\n");
    printf("Gesture CNN Performance Benchmark\n");
    printf("========================================\n");
    printf("Iterations: %d\n\n", num_iterations);
    
    // 1. Create CNN network
    Simple1DCNN cnn(GESTURE_CNN_IN_CHANNELS,
                    GESTURE_CNN_CONV1_OUT_CHANNELS,
                    GESTURE_CNN_CONV2_OUT_CHANNELS,
                    GESTURE_CNN_NUM_CLASSES,
                    GESTURE_CNN_SEQ_LEN,
                    3);
    
    // 2. Set weights and biases
    cnn.set_conv1_weights(gesture_cnn_conv1_weights);
    cnn.set_conv1_bias(gesture_cnn_conv1_bias);
    cnn.set_conv2_weights(gesture_cnn_conv2_weights);
    cnn.set_conv2_bias(gesture_cnn_conv2_bias);
    cnn.set_fc_weights(gesture_cnn_fc_weights);
    cnn.set_fc_bias(gesture_cnn_fc_bias);
    
    // 3. Prepare test input (using first sample)
    float input[3 * 64];
    memcpy(input, test_input_0, sizeof(test_input_0));
    standardize_input(input, GESTURE_CNN_IN_CHANNELS, GESTURE_CNN_SEQ_LEN,
                     gesture_scaler_mean, gesture_scaler_std);
    
    float output[GESTURE_CNN_NUM_CLASSES];
    
#ifdef ESP_PLATFORM
    // 4. Measure memory usage (ESP32)
    size_t free_heap_before = esp_get_free_heap_size();
    size_t min_free_heap_before = esp_get_minimum_free_heap_size();
    
    // 5. Performance test - measure inference time
    int64_t start_time = esp_timer_get_time();
    
    for (int i = 0; i < num_iterations; i++) {
        cnn.forward(input, output);
        // Feed watchdog every 100 iterations to prevent timeout
        // Only reset if task is already added to watchdog
        if ((i + 1) % 100 == 0) {
            esp_err_t ret = esp_task_wdt_reset();
            // Ignore ESP_ERR_NOT_FOUND error (task not in watchdog)
            (void)ret;
        }
    }
    
    int64_t end_time = esp_timer_get_time();
    int64_t total_time_us = end_time - start_time;
    float avg_time_us = (float)total_time_us / num_iterations;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    // 6. Measure memory usage (ESP32)
    size_t free_heap_after = esp_get_free_heap_size();
    size_t min_free_heap_after = esp_get_minimum_free_heap_size();
    size_t heap_used = (free_heap_before > free_heap_after) ? (free_heap_before - free_heap_after) : 0;
    
    // 7. Print performance results
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

