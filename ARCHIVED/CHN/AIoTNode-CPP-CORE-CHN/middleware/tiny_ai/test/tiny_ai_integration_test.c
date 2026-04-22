/**
 * @file tiny_ai_integration_test.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Integration tests implementation
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 */

#include "tiny_ai_integration_test.h"
#include "tiny_graph.h"
#include "tiny_fc.h"
#include "tiny_activations.h"
#include "tiny_loss.h"
#include "tiny_optimizer.h"
#include "tiny_trainer.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("TEST FAILED: %s\n", msg); \
            return TINY_FAIL; \
        } \
    } while(0)

static tiny_error_t test_simple_mlp_training(void)
{
    printf("Testing simple MLP training (FC -> ReLU -> FC -> MSE)...\n");
    
    // Create computation graph
    printf("  Creating computation graph...\n");
    tiny_graph_t* graph = tiny_graph_create(10);
    TEST_ASSERT(graph != NULL, "Failed to create graph");
    
    // Create FC layer 1: 2 inputs -> 4 outputs
    printf("  Creating FC layer 1 (2 -> 4)...\n");
    tiny_fc_params_t* fc1_params = tiny_fc_create(2, 4, true);
    TEST_ASSERT(fc1_params != NULL, "Failed to create FC1");
    
    // Create FC layer 2: 4 inputs -> 1 output
    printf("  Creating FC layer 2 (4 -> 1)...\n");
    tiny_fc_params_t* fc2_params = tiny_fc_create(4, 1, true);
    TEST_ASSERT(fc2_params != NULL, "Failed to create FC2");
    
    // Create tensors first (needed for node setup)
    printf("  Creating tensors...\n");
    int input_shape[] = {2};
    int hidden_shape[] = {4};
    int relu_output_shape[] = {4};  // ReLU output has same shape as input
    int output_shape[] = {1};
    int loss_shape[] = {1};
    int target_shape[] = {1};
    
    tiny_tensor_t* input = tiny_tensor_create(input_shape, 1, TINY_AI_DTYPE_FLOAT32);
    tiny_tensor_t* hidden = tiny_tensor_create(hidden_shape, 1, TINY_AI_DTYPE_FLOAT32);
    tiny_tensor_t* relu_output = tiny_tensor_create(relu_output_shape, 1, TINY_AI_DTYPE_FLOAT32);
    tiny_tensor_t* output = tiny_tensor_create(output_shape, 1, TINY_AI_DTYPE_FLOAT32);
    tiny_tensor_t* target = tiny_tensor_create(target_shape, 1, TINY_AI_DTYPE_FLOAT32);
    tiny_tensor_t* loss = tiny_tensor_create(loss_shape, 1, TINY_AI_DTYPE_FLOAT32);
    
    TEST_ASSERT(input != NULL && hidden != NULL && relu_output != NULL && 
                output != NULL && target != NULL && loss != NULL, "Failed to create tensors");
    
    // Enable gradients
#if TINY_AI_ENABLE_GRADIENTS
    tiny_tensor_requires_grad(input, true);
    tiny_tensor_requires_grad(hidden, true);
    tiny_tensor_requires_grad(relu_output, true);
    tiny_tensor_requires_grad(output, true);
    tiny_tensor_requires_grad(loss, true);
    tiny_tensor_requires_grad(fc1_params->weights, true);
    tiny_tensor_requires_grad(fc1_params->bias, true);
    tiny_tensor_requires_grad(fc2_params->weights, true);
    tiny_tensor_requires_grad(fc2_params->bias, true);
#endif
    
    // Create FC nodes and connect them
    printf("  Setting up graph connections...\n");
    tiny_graph_node_t* fc1_node = tiny_graph_add_node(graph, TINY_AI_OP_FULLY_CONNECTED, 1, 1);
    fc1_node->inputs[0] = input;
    fc1_node->outputs[0] = hidden;
    fc1_node->params = fc1_params;
    fc1_node->forward_func = tiny_fc_forward;
    fc1_node->backward_func = tiny_fc_backward;
    
    // Create ReLU activation
    printf("  Creating ReLU activation...\n");
    tiny_graph_node_t* relu_node = tiny_graph_add_node(graph, TINY_AI_OP_RELU, 1, 1);
    TEST_ASSERT(relu_node != NULL, "Failed to create ReLU node");
    relu_node->outputs[0] = relu_output;  // Set ReLU output tensor
    relu_node->forward_func = tiny_relu_forward;
    relu_node->backward_func = tiny_relu_backward;
    
    tiny_graph_node_t* fc2_node = tiny_graph_add_node(graph, TINY_AI_OP_FULLY_CONNECTED, 1, 1);
    fc2_node->outputs[0] = output;
    fc2_node->params = fc2_params;
    fc2_node->forward_func = tiny_fc_forward;
    fc2_node->backward_func = tiny_fc_backward;
    
    // Create MSE loss node
    printf("  Creating MSE loss node...\n");
    tiny_graph_node_t* loss_node = tiny_graph_add_node(graph, TINY_AI_OP_MSE_LOSS, 2, 1);
    TEST_ASSERT(loss_node != NULL, "Failed to create loss node");
    loss_node->inputs[1] = target;  // Set target (second input)
    loss_node->outputs[0] = loss;   // Set loss output
    loss_node->forward_func = tiny_mse_forward;
    loss_node->backward_func = tiny_mse_backward;
    
    // Connect nodes (this will set inputs automatically)
    tiny_graph_connect(graph, fc1_node, 0, relu_node, 0);  // Sets relu_node->inputs[0] = hidden
    tiny_graph_connect(graph, relu_node, 0, fc2_node, 0);  // Sets fc2_node->inputs[0] = relu_output
    tiny_graph_connect(graph, fc2_node, 0, loss_node, 0);  // Sets loss_node->inputs[0] = output
    
    // Enable gradients for loss output
#if TINY_AI_ENABLE_GRADIENTS
    tiny_tensor_requires_grad(loss, true);
#endif
    
    // Build execution order
    printf("  Building execution order...\n");
    tiny_error_t err = tiny_graph_build_order(graph);
    TEST_ASSERT(err == TINY_OK, "Failed to build order");
    
    // Initialize weights with small random values
    printf("  Initializing weights...\n");
    tiny_fc_init_weights_xavier(fc1_params, 12345);
    tiny_fc_init_weights_xavier(fc2_params, 54321);
    
    // Create optimizer
    printf("  Creating Adam optimizer...\n");
    tiny_adam_optimizer_t* optimizer_fc1_w = tiny_optimizer_adam_create(0.01f, 0.9f, 0.999f, 1e-8f);
    tiny_adam_optimizer_t* optimizer_fc1_b = tiny_optimizer_adam_create(0.01f, 0.9f, 0.999f, 1e-8f);
    tiny_adam_optimizer_t* optimizer_fc2_w = tiny_optimizer_adam_create(0.01f, 0.9f, 0.999f, 1e-8f);
    tiny_adam_optimizer_t* optimizer_fc2_b = tiny_optimizer_adam_create(0.01f, 0.9f, 0.999f, 1e-8f);
    TEST_ASSERT(optimizer_fc1_w != NULL && optimizer_fc1_b != NULL &&
                optimizer_fc2_w != NULL && optimizer_fc2_b != NULL, "Failed to create optimizers");
    
    void* optimizers[] = {optimizer_fc1_w, optimizer_fc1_b, optimizer_fc2_w, optimizer_fc2_b};
    tiny_tensor_t* params[] = {fc1_params->weights, fc1_params->bias, 
                               fc2_params->weights, fc2_params->bias};
    
    // Training data: simple linear regression task
    // y = 2*x1 + 3*x2 + 1
    printf("  Setting up training data...\n");
    float training_inputs[][2] = {{1.0f, 1.0f}, {2.0f, 1.0f}, {1.0f, 2.0f}, {2.0f, 2.0f}};
    float training_targets[] = {6.0f, 8.0f, 9.0f, 11.0f};  // 2*x1 + 3*x2 + 1
    
    int num_samples = 4;
    tiny_tensor_t* input_tensors[4];
    tiny_tensor_t* target_tensors[4];
    
    for (int i = 0; i < num_samples; i++) {
        input_tensors[i] = tiny_tensor_create(input_shape, 1, TINY_AI_DTYPE_FLOAT32);
        target_tensors[i] = tiny_tensor_create(target_shape, 1, TINY_AI_DTYPE_FLOAT32);
        float* input_data = (float*)tiny_tensor_data(input_tensors[i]);
        float* target_data = (float*)tiny_tensor_data(target_tensors[i]);
        input_data[0] = training_inputs[i][0];
        input_data[1] = training_inputs[i][1];
        target_data[0] = training_targets[i];
    }
    
    // Train for a few steps
    printf("  Training model (5 steps)...\n");
    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    
    for (int step = 0; step < 5; step++) {
        // Use first sample for training
        float* input_data = (float*)tiny_tensor_data(input_tensors[0]);
        float* target_data = (float*)tiny_tensor_data(target_tensors[0]);
        
        // Copy to graph tensors
        memcpy(tiny_tensor_data(input), input_data, 2 * sizeof(float));
        memcpy(tiny_tensor_data(target), target_data, sizeof(float));
        
        // Debug: Print input and target before training step
        if (step == 0) {
            float* input_data_debug = (float*)tiny_tensor_data(input);
            float* target_data_debug = (float*)tiny_tensor_data(target);
            printf("    Debug - Input: [%.2f, %.2f], Target: %.2f\n", 
                   input_data_debug[0], input_data_debug[1], target_data_debug[0]);
        }
        
        // Training step
        err = tiny_trainer_step(graph, input, target, loss_node, optimizers, 4, params, 4);
        TEST_ASSERT(err == TINY_OK, "Training step failed");
        
        // Debug: Check if loss was computed
        float* loss_data = (float*)tiny_tensor_data(loss);
        float* output_data = (float*)tiny_tensor_data(output);
        if (step == 0) {
            printf("    Debug - Output after forward: %.4f\n", output_data[0]);
            printf("    Debug - Loss after forward: %.4f\n", loss_data[0]);
        }
        
        if (step == 0) {
            initial_loss = loss_data[0];
        }
        if (step == 4) {
            final_loss = loss_data[0];
        }
        
        printf("    Step %d: Loss = %.4f, Output = %.4f\n", step + 1, loss_data[0], output_data[0]);
    }
    
    printf("    Initial loss: %.4f\n", initial_loss);
    printf("    Final loss: %.4f\n", final_loss);
    printf("    Loss reduction: %.2f%%\n", (initial_loss - final_loss) / initial_loss * 100.0f);
    
    // Verify loss decreased
    TEST_ASSERT(final_loss < initial_loss, "Loss should decrease during training");
    
    // Cleanup
    printf("  Cleaning up...\n");
    for (int i = 0; i < num_samples; i++) {
        tiny_tensor_destroy(input_tensors[i]);
        tiny_tensor_destroy(target_tensors[i]);
    }
    tiny_optimizer_destroy(optimizer_fc1_w);
    tiny_optimizer_destroy(optimizer_fc1_b);
    tiny_optimizer_destroy(optimizer_fc2_w);
    tiny_optimizer_destroy(optimizer_fc2_b);
    tiny_tensor_destroy(input);
    tiny_tensor_destroy(hidden);
    tiny_tensor_destroy(relu_output);
    tiny_tensor_destroy(output);
    tiny_tensor_destroy(target);
    tiny_tensor_destroy(loss);
    tiny_fc_destroy(fc1_params);
    tiny_fc_destroy(fc2_params);
    tiny_graph_destroy(graph);
    
    printf("  ✓ Simple MLP training test passed\n");
    return TINY_OK;
}

tiny_error_t tiny_ai_integration_test_all(void)
{
    printf("========== TinyAI Integration Tests ==========\n\n");
    
    tiny_error_t err;
    
    err = test_simple_mlp_training();
    if (err != TINY_OK) return err;
    
    printf("\n========== All Integration Tests Passed ==========\n\n");
    return TINY_OK;
}

