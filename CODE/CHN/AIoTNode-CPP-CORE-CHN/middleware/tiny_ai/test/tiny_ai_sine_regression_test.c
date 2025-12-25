/**
 * @file tiny_ai_sine_regression_test.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Sine function regression test implementation
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 */

#include "tiny_ai_sine_regression_test.h"
#include "tiny_mlp.h"
#include "tiny_dataloader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("TEST FAILED: %s\n", msg); \
            return TINY_FAIL; \
        } \
    } while(0)

/* ============================================================================
 * DATA GENERATION
 * ============================================================================ */

/**
 * @brief Generate sine function data with white noise
 * 
 * @param x Input values (array of floats)
 * @param y Output values (array of floats, will be filled)
 * @param num_samples Number of samples to generate
 * @param noise_level Standard deviation of white noise (0.0 for no noise)
 */
static void generate_sine_data(float* x, float* y, int num_samples, float noise_level)
{
    // Simple LCG for random number generation
    static unsigned int seed = 12345;
    
    for (int i = 0; i < num_samples; i++) {
        // Generate x in range [0, 2*PI]
        x[i] = (float)i / (num_samples - 1) * 2.0f * M_PI;
        
        // Generate y = sin(x) + noise
        float sine_val = sinf(x[i]);
        
        // Add white noise
        if (noise_level > 0.0f) {
            // Generate random number in [0, 1)
            seed = (1103515245u * seed + 12345u) & 0x7FFFFFFFu;
            float r1 = (float)seed / 2147483648.0f;
            seed = (1103515245u * seed + 12345u) & 0x7FFFFFFFu;
            float r2 = (float)seed / 2147483648.0f;
            
            // Box-Muller transform for Gaussian noise
            float z = sqrtf(-2.0f * logf(r1 + 1e-10f)) * cosf(2.0f * M_PI * r2);
            float noise = z * noise_level;
            
            y[i] = sine_val + noise;
        } else {
            y[i] = sine_val;
        }
    }
}

/* ============================================================================
 * TEST FUNCTIONS
 * ============================================================================ */

static tiny_error_t test_sine_regression_mlp(void)
{
    printf("Testing sine function regression with MLP...\n");
    
    // Parameters
    const int num_train_samples = 200;
    const int num_test_samples = 50;
    const float noise_level = 0.1f;
    const int num_epochs = 50;
    const int batch_size = 16;
    const float learning_rate = 0.01f;
    
    printf("  Generating training data (%d samples, noise=%.3f)...\n", 
           num_train_samples, noise_level);
    
    // Generate training data
    float* train_x = (float*)malloc(num_train_samples * sizeof(float));
    float* train_y = (float*)malloc(num_train_samples * sizeof(float));
    TEST_ASSERT(train_x != NULL && train_y != NULL, "Failed to allocate training data");
    
    generate_sine_data(train_x, train_y, num_train_samples, noise_level);
    
    printf("  Generating test data (%d samples, no noise)...\n", num_test_samples);
    
    // Generate test data (no noise for evaluation)
    float* test_x = (float*)malloc(num_test_samples * sizeof(float));
    float* test_y_true = (float*)malloc(num_test_samples * sizeof(float));
    TEST_ASSERT(test_x != NULL && test_y_true != NULL, "Failed to allocate test data");
    
    generate_sine_data(test_x, test_y_true, num_test_samples, 0.0f);
    
    // Create dataset
    printf("  Creating dataset...\n");
    tiny_dataset_t* dataset = tiny_dataset_create(num_train_samples);
    TEST_ASSERT(dataset != NULL, "Failed to create dataset");
    
    // Add samples to dataset
    for (int i = 0; i < num_train_samples; i++) {
        // Create input tensor [1] (single feature: x value)
        int input_shape[] = {1};
        tiny_tensor_t* input_tensor = tiny_tensor_create(input_shape, 1, TINY_AI_DTYPE_FLOAT32);
        TEST_ASSERT(input_tensor != NULL, "Failed to create input tensor");
        
        // Create target tensor [1] (single output: y value)
        int target_shape[] = {1};
        tiny_tensor_t* target_tensor = tiny_tensor_create(target_shape, 1, TINY_AI_DTYPE_FLOAT32);
        TEST_ASSERT(target_tensor != NULL, "Failed to create target tensor");
        
        // Set values
        float* input_data = (float*)tiny_tensor_data(input_tensor);
        float* target_data = (float*)tiny_tensor_data(target_tensor);
        input_data[0] = train_x[i];
        target_data[0] = train_y[i];
        
        // Add to dataset
        tiny_error_t err = tiny_dataset_add_sample(dataset, input_tensor, target_tensor);
        TEST_ASSERT(err == TINY_OK, "Failed to add sample to dataset");
    }
    
    printf("  Dataset created with %d samples\n", tiny_dataset_size(dataset));
    
    // Create DataLoader
    printf("  Creating DataLoader (batch_size=%d, shuffle=true)...\n", batch_size);
    tiny_dataloader_t* loader = tiny_dataloader_create(dataset, batch_size, true);
    TEST_ASSERT(loader != NULL, "Failed to create DataLoader");
    
    // Create MLP model: 1 -> 32 -> 32 -> 1
    printf("  Creating MLP model (1 -> 32 -> 32 -> 1)...\n");
    tiny_mlp_layer_config_t layers[] = {
        {1, 32, true, TINY_MLP_ACT_RELU},   // Input layer: 1 -> 32, ReLU
        {32, 32, true, TINY_MLP_ACT_RELU},  // Hidden layer: 32 -> 32, ReLU
        {32, 1, true, TINY_MLP_ACT_NONE}   // Output layer: 32 -> 1, no activation
    };
    
    tiny_mlp_model_t* model = tiny_mlp_create(1, layers, 3);
    TEST_ASSERT(model != NULL, "Failed to create MLP model");
    
    // Initialize weights
    printf("  Initializing weights (Xavier)...\n");
    tiny_error_t err = tiny_mlp_init_weights_xavier(model, 12345);
    TEST_ASSERT(err == TINY_OK, "Failed to initialize weights");
    
    // Create optimizers
    printf("  Creating optimizers (Adam, lr=%.4f)...\n", learning_rate);
    tiny_tensor_t* params[128];
    int num_params = tiny_mlp_get_parameters(model, params, 128);
    TEST_ASSERT(num_params > 0, "Failed to get model parameters");
    
    void** optimizers = (void**)malloc(num_params * sizeof(void*));
    TEST_ASSERT(optimizers != NULL, "Failed to allocate optimizers array");
    
    for (int i = 0; i < num_params; i++) {
        optimizers[i] = tiny_optimizer_adam_create(learning_rate, 0.9f, 0.999f, 1e-8f);
        TEST_ASSERT(optimizers[i] != NULL, "Failed to create optimizer");
    }
    
    printf("  Created %d optimizers\n", num_params);
    
    // Training statistics structure
    typedef struct {
        float epoch_loss_sum;
        int epoch_batch_count;
        int current_epoch;
    } training_stats_t;
    
    training_stats_t train_stats = {0};
    
    // Training callback - track loss for each epoch
    void training_callback(const tiny_training_stats_t* stats, void* user_data) {
        training_stats_t* ts = (training_stats_t*)user_data;
        
        // Update epoch statistics
        if (stats->current_epoch != ts->current_epoch) {
            // New epoch started, print previous epoch summary
            if (ts->current_epoch >= 0 && ts->epoch_batch_count > 0) {
                float avg_loss = ts->epoch_loss_sum / ts->epoch_batch_count;
                printf("  Epoch %3d: Average Loss = %.6f (from %d batches)\n", 
                       ts->current_epoch, avg_loss, ts->epoch_batch_count);
            }
            // Reset for new epoch
            ts->current_epoch = stats->current_epoch;
            ts->epoch_loss_sum = 0.0f;
            ts->epoch_batch_count = 0;
        }
        
        ts->epoch_loss_sum += stats->current_loss;
        ts->epoch_batch_count++;
        
        // Print batch loss every few batches
        if (stats->current_batch % 5 == 0 || stats->current_batch == 0) {
            printf("    Epoch %3d, Batch %3d: Loss = %.6f\n", 
                   stats->current_epoch, stats->current_batch, stats->current_loss);
        }
    }
    
    // Train model
    printf("  Training model (%d epochs)...\n", num_epochs);
    printf("  Training progress:\n");
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Reset epoch statistics
        train_stats.epoch_loss_sum = 0.0f;
        train_stats.epoch_batch_count = 0;
        train_stats.current_epoch = epoch;
        
        err = tiny_mlp_train_epoch(model, loader, TINY_AI_OP_MSE_LOSS,
                                   optimizers, num_params,
                                   training_callback, &train_stats);
        TEST_ASSERT(err == TINY_OK, "Training epoch failed");
        
        // Print epoch summary
        if (train_stats.epoch_batch_count > 0) {
            float avg_loss = train_stats.epoch_loss_sum / train_stats.epoch_batch_count;
            printf("  Epoch %3d: Average Loss = %.6f (from %d batches)\n", 
                   epoch, avg_loss, train_stats.epoch_batch_count);
        }
    }
    
    printf("  Training completed.\n");
    
    // Inference and evaluation
    printf("  Evaluating on test set (%d samples)...\n", num_test_samples);
    
    float* test_y_pred = (float*)malloc(num_test_samples * sizeof(float));
    TEST_ASSERT(test_y_pred != NULL, "Failed to allocate prediction array");
    
    float total_error = 0.0f;
    float max_error = 0.0f;
    int num_correct = 0;  // Predictions within 0.1 of true value
    
    printf("  Comparing predictions with ground truth:\n");
    printf("    Sample | Input (x) | True (sin) | Predicted | Error\n");
    printf("    ------|----------|------------|-----------|--------\n");
    
    for (int i = 0; i < num_test_samples; i++) {
        // Forward pass
        err = tiny_mlp_forward(model, &test_x[i], &test_y_pred[i]);
        TEST_ASSERT(err == TINY_OK, "Forward pass failed");
        
        // Calculate error
        float error = fabsf(test_y_pred[i] - test_y_true[i]);
        total_error += error;
        if (error > max_error) {
            max_error = error;
        }
        if (error < 0.1f) {
            num_correct++;
        }
        
        // Print some samples
        if (i < 10 || i % 10 == 0) {
            printf("    %6d | %8.4f | %10.6f | %9.6f | %.6f\n",
                   i, test_x[i], test_y_true[i], test_y_pred[i], error);
        }
    }
    
    float mean_error = total_error / num_test_samples;
    float accuracy = (float)num_correct / num_test_samples * 100.0f;
    
    printf("\n  Evaluation Results:\n");
    printf("    Mean Absolute Error (MAE): %.6f\n", mean_error);
    printf("    Max Error: %.6f\n", max_error);
    printf("    Accuracy (error < 0.1): %.2f%% (%d/%d)\n", 
           accuracy, num_correct, num_test_samples);
    
    // Verify reasonable performance
    TEST_ASSERT(mean_error < 0.2f, "Mean error too high (expected < 0.2)");
    TEST_ASSERT(max_error < 0.5f, "Max error too high (expected < 0.5)");
    TEST_ASSERT(accuracy > 70.0f, "Accuracy too low (expected > 70%)");
    
    printf("  ✓ Sine regression test passed!\n");
    
    // Cleanup
    free(test_y_pred);
    for (int i = 0; i < num_params; i++) {
        tiny_optimizer_destroy(optimizers[i]);
    }
    free(optimizers);
    tiny_mlp_destroy(model);
    tiny_dataloader_destroy(loader);
    
    // Destroy dataset tensors
    for (int i = 0; i < num_train_samples; i++) {
        tiny_tensor_destroy(dataset->inputs[i]);
        tiny_tensor_destroy(dataset->targets[i]);
    }
    tiny_dataset_destroy(dataset);
    
    free(train_x);
    free(train_y);
    free(test_x);
    free(test_y_true);
    
    return TINY_OK;
}

tiny_error_t tiny_ai_test_sine_regression_mlp(void)
{
    return test_sine_regression_mlp();
}

tiny_error_t tiny_ai_sine_regression_test_all(void)
{
    printf("========== TinyAI Sine Regression Tests ==========\n\n");
    
    tiny_error_t err;
    
    err = test_sine_regression_mlp();
    if (err != TINY_OK) return err;
    
    printf("\n========== All Sine Regression Tests Passed ==========\n\n");
    return TINY_OK;
}

