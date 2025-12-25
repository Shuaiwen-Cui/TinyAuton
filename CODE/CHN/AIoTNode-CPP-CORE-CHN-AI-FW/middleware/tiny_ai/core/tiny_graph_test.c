/**
 * @file tiny_graph_test.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Test implementation for graph module
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 */

#include "tiny_graph_test.h"
#include "tiny_fc.h"
#include <stdio.h>
#include <math.h>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("TEST FAILED: %s\n", msg); \
            return TINY_FAIL; \
        } \
    } while(0)

static tiny_error_t test_create_destroy(void)
{
    printf("Testing graph create/destroy...\n");
    
    printf("  Creating graph with initial capacity 8...\n");
    tiny_graph_t* graph = tiny_graph_create(8);
    TEST_ASSERT(graph != NULL, "Failed to create graph");
    printf("    - Created successfully\n");
    printf("    - Capacity: %d\n", graph->capacity);
    printf("    - Num nodes: %d (expected: 0)\n", graph->num_nodes);
    TEST_ASSERT(graph->num_nodes == 0, "Graph should start with 0 nodes");
    TEST_ASSERT(graph->capacity == 8, "Graph capacity should be 8");
    
    tiny_graph_destroy(graph);
    printf("    - Destroyed successfully\n");
    
    printf("  ✓ Create/destroy tests passed\n");
    return TINY_OK;
}

static tiny_error_t test_add_node(void)
{
    printf("Testing add node...\n");
    
    tiny_graph_t* graph = tiny_graph_create(4);
    TEST_ASSERT(graph != NULL, "Failed to create graph");
    
    printf("  Adding FC node (2 inputs, 1 output)...\n");
    tiny_graph_node_t* node1 = tiny_graph_add_node(graph, TINY_AI_OP_FULLY_CONNECTED, 2, 1);
    TEST_ASSERT(node1 != NULL, "Failed to add node");
    TEST_ASSERT(graph->num_nodes == 1, "Graph should have 1 node");
    printf("    - Node added successfully\n");
    printf("    - Node op_type: %d (FC)\n", node1->op_type);
    printf("    - Node inputs: %d\n", node1->num_inputs);
    printf("    - Node outputs: %d\n", node1->num_outputs);
    
    printf("  Adding ReLU node (1 input, 1 output)...\n");
    tiny_graph_node_t* node2 = tiny_graph_add_node(graph, TINY_AI_OP_RELU, 1, 1);
    TEST_ASSERT(node2 != NULL, "Failed to add second node");
    TEST_ASSERT(graph->num_nodes == 2, "Graph should have 2 nodes");
    printf("    - Second node added successfully\n");
    
    tiny_graph_destroy(graph);
    
    printf("  ✓ Add node tests passed\n");
    return TINY_OK;
}

static tiny_error_t test_connect_and_build_order(void)
{
    printf("Testing connect and build order...\n");
    
    tiny_graph_t* graph = tiny_graph_create(4);
    TEST_ASSERT(graph != NULL, "Failed to create graph");
    
    // Create nodes: FC -> ReLU -> FC
    printf("  Creating nodes: FC -> ReLU -> FC\n");
    tiny_graph_node_t* fc1 = tiny_graph_add_node(graph, TINY_AI_OP_FULLY_CONNECTED, 1, 1);
    tiny_graph_node_t* relu = tiny_graph_add_node(graph, TINY_AI_OP_RELU, 1, 1);
    tiny_graph_node_t* fc2 = tiny_graph_add_node(graph, TINY_AI_OP_FULLY_CONNECTED, 1, 1);
    TEST_ASSERT(fc1 != NULL && relu != NULL && fc2 != NULL, "Failed to create nodes");
    printf("    - Created 3 nodes\n");
    
    // Connect nodes
    printf("  Connecting nodes...\n");
    tiny_error_t err = tiny_graph_connect(graph, fc1, 0, relu, 0);
    TEST_ASSERT(err == TINY_OK, "Failed to connect fc1 -> relu");
    printf("    - Connected FC1 -> ReLU\n");
    
    err = tiny_graph_connect(graph, relu, 0, fc2, 0);
    TEST_ASSERT(err == TINY_OK, "Failed to connect relu -> fc2");
    printf("    - Connected ReLU -> FC2\n");
    
    // Verify connections
    TEST_ASSERT(relu->num_parents == 1, "ReLU should have 1 parent");
    TEST_ASSERT(relu->parents[0] == fc1, "ReLU's parent should be FC1");
    TEST_ASSERT(fc1->num_children == 1, "FC1 should have 1 child");
    TEST_ASSERT(fc1->children[0] == relu, "FC1's child should be ReLU");
    printf("    - Parent/child relationships verified\n");
    
    // Build execution order
    printf("  Building execution order (topological sort)...\n");
    err = tiny_graph_build_order(graph);
    TEST_ASSERT(err == TINY_OK, "Failed to build order");
    TEST_ASSERT(graph->forward_order != NULL, "Forward order should be built");
    TEST_ASSERT(graph->backward_order != NULL, "Backward order should be built");
    printf("    - Execution order built successfully\n");
    printf("    - Forward order: ");
    for (int i = 0; i < graph->num_nodes; i++) {
        printf("Node%d ", i);
    }
    printf("\n");
    
    // Verify order: fc1 should come before relu, relu before fc2
    int fc1_idx = -1, relu_idx = -1, fc2_idx = -1;
    for (int i = 0; i < graph->num_nodes; i++) {
        if (graph->forward_order[i] == fc1) fc1_idx = i;
        if (graph->forward_order[i] == relu) relu_idx = i;
        if (graph->forward_order[i] == fc2) fc2_idx = i;
    }
    TEST_ASSERT(fc1_idx < relu_idx, "FC1 should come before ReLU");
    TEST_ASSERT(relu_idx < fc2_idx, "ReLU should come before FC2");
    printf("    - Order verified: FC1 (%d) -> ReLU (%d) -> FC2 (%d)\n", fc1_idx, relu_idx, fc2_idx);
    
    tiny_graph_destroy(graph);
    
    printf("  ✓ Connect and build order tests passed\n");
    return TINY_OK;
}

static tiny_error_t test_forward_backward(void)
{
    printf("Testing forward/backward propagation...\n");
    
    tiny_graph_t* graph = tiny_graph_create(4);
    TEST_ASSERT(graph != NULL, "Failed to create graph");
    
    // Create a simple graph: Input -> FC -> Output
    printf("  Creating simple graph: Input -> FC\n");
    tiny_graph_node_t* fc = tiny_graph_add_node(graph, TINY_AI_OP_FULLY_CONNECTED, 1, 1);
    TEST_ASSERT(fc != NULL, "Failed to create FC node");
    
    // Create input and output tensors
    int input_shape[] = {4};  // 4 features
    int output_shape[] = {2}; // 2 output features
    
    tiny_tensor_t* input = tiny_tensor_create(input_shape, 1, TINY_AI_DTYPE_FLOAT32);
    tiny_tensor_t* output = tiny_tensor_create(output_shape, 1, TINY_AI_DTYPE_FLOAT32);
    TEST_ASSERT(input != NULL && output != NULL, "Failed to create tensors");
    
    // Set input/output for FC node
    fc->inputs[0] = input;
    fc->outputs[0] = output;
    
    // Build execution order
    printf("  Building execution order...\n");
    tiny_error_t err = tiny_graph_build_order(graph);
    TEST_ASSERT(err == TINY_OK, "Failed to build order");
    
    // Test forward propagation (should not crash even without forward_func)
    printf("  Testing forward propagation...\n");
    err = tiny_graph_forward(graph);
    TEST_ASSERT(err == TINY_OK, "Forward propagation failed");
    printf("    - Forward propagation completed (no forward_func set, skipped)\n");
    
    // Test backward propagation in inference mode (should skip)
    printf("  Testing backward propagation (inference mode)...\n");
    tiny_graph_set_training_mode(graph, false);
    err = tiny_graph_backward(graph);
    TEST_ASSERT(err == TINY_OK, "Backward propagation failed");
    printf("    - Backward propagation skipped (inference mode)\n");
    
    // Test backward propagation in training mode (should not crash even without backward_func)
    printf("  Testing backward propagation (training mode)...\n");
    tiny_graph_set_training_mode(graph, true);
#if TINY_AI_ENABLE_GRADIENTS
    tiny_tensor_requires_grad(output, true);
#endif
    err = tiny_graph_backward(graph);
    TEST_ASSERT(err == TINY_OK, "Backward propagation failed");
    printf("    - Backward propagation completed (no backward_func set, skipped)\n");
    
    // Cleanup
    tiny_tensor_destroy(input);
    tiny_tensor_destroy(output);
    tiny_graph_destroy(graph);
    
    printf("  ✓ Forward/backward tests passed\n");
    return TINY_OK;
}

static tiny_error_t test_fc_end_to_end(void)
{
    printf("Testing FC layer end-to-end...\n");
    
    // Create FC layer: 4 inputs -> 2 outputs
    printf("  Creating FC layer (4 inputs, 2 outputs, with bias)...\n");
    tiny_fc_params_t* fc_params = tiny_fc_create(4, 2, true);
    TEST_ASSERT(fc_params != NULL, "Failed to create FC layer");
    printf("    - FC layer created successfully\n");
    
    // Initialize weights with known values for testing
    printf("  Initializing weights with test values...\n");
    float* weight_data = (float*)tiny_tensor_data(fc_params->weights);
    // Set weights to identity-like pattern for easy verification
    // weights[0] = [1, 0, 0, 0]
    // weights[1] = [0, 1, 0, 0]
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            weight_data[i * 4 + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    printf("    - Weights initialized\n");
    
    // Set bias to [0.5, 0.5]
    float* bias_data = (float*)tiny_tensor_data(fc_params->bias);
    bias_data[0] = 0.5f;
    bias_data[1] = 0.5f;
    printf("    - Bias initialized to [0.5, 0.5]\n");
    
    // Create input tensor [1, 2, 3, 4]
    printf("  Creating input tensor [1, 2, 3, 4]...\n");
    int input_shape[] = {4};
    tiny_tensor_t* input = tiny_tensor_create(input_shape, 1, TINY_AI_DTYPE_FLOAT32);
    TEST_ASSERT(input != NULL, "Failed to create input tensor");
    float* input_data = (float*)tiny_tensor_data(input);
    input_data[0] = 1.0f;
    input_data[1] = 2.0f;
    input_data[2] = 3.0f;
    input_data[3] = 4.0f;
    printf("    - Input tensor created\n");
    
    // Create output tensor
    int output_shape[] = {2};
    tiny_tensor_t* output = tiny_tensor_create(output_shape, 1, TINY_AI_DTYPE_FLOAT32);
    TEST_ASSERT(output != NULL, "Failed to create output tensor");
    printf("    - Output tensor created\n");
    
    // Enable gradients for training
#if TINY_AI_ENABLE_GRADIENTS
    tiny_tensor_requires_grad(input, true);
    tiny_tensor_requires_grad(output, true);
    tiny_tensor_requires_grad(fc_params->weights, true);
    tiny_tensor_requires_grad(fc_params->bias, true);
    printf("    - Gradients enabled\n");
#endif
    
    // Create graph node and set up
    printf("  Setting up computation graph...\n");
    tiny_graph_t* graph = tiny_graph_create(4);
    tiny_graph_node_t* fc_node = tiny_graph_add_node(graph, TINY_AI_OP_FULLY_CONNECTED, 1, 1);
    fc_node->inputs[0] = input;
    fc_node->outputs[0] = output;
    fc_node->params = fc_params;
    fc_node->forward_func = tiny_fc_forward;
    fc_node->backward_func = tiny_fc_backward;
    printf("    - Graph node configured\n");
    
    // Build execution order
    tiny_error_t err = tiny_graph_build_order(graph);
    TEST_ASSERT(err == TINY_OK, "Failed to build order");
    printf("    - Execution order built\n");
    
    // Test forward propagation
    printf("  Testing forward propagation...\n");
    tiny_graph_set_training_mode(graph, false);
    err = tiny_graph_forward(graph);
    TEST_ASSERT(err == TINY_OK, "Forward propagation failed");
    
    // Verify output: should be [1*1 + 0*2 + 0*3 + 0*4 + 0.5, 0*1 + 1*2 + 0*3 + 0*4 + 0.5]
    // = [1.5, 2.5]
    float* output_data = (float*)tiny_tensor_data(output);
    printf("    - Output: [%.2f, %.2f] (expected: [1.50, 2.50])\n", output_data[0], output_data[1]);
    TEST_ASSERT(fabsf(output_data[0] - 1.5f) < 0.01f, "Output[0] incorrect");
    TEST_ASSERT(fabsf(output_data[1] - 2.5f) < 0.01f, "Output[1] incorrect");
    printf("    - Forward propagation verified ✓\n");
    
    // Test backward propagation
    printf("  Testing backward propagation...\n");
    tiny_graph_set_training_mode(graph, true);
    
    // Set output gradient to [1.0, 1.0]
    float* output_grad = (float*)tiny_tensor_grad(output);
    if (output_grad != NULL) {
        output_grad[0] = 1.0f;
        output_grad[1] = 1.0f;
        printf("    - Output gradient set to [1.0, 1.0]\n");
    }
    
    // Zero input and weight gradients
    tiny_tensor_zero_grad(input);
    tiny_tensor_zero_grad(fc_params->weights);
    tiny_tensor_zero_grad(fc_params->bias);
    
    err = tiny_graph_backward(graph);
    TEST_ASSERT(err == TINY_OK, "Backward propagation failed");
    printf("    - Backward propagation completed\n");
    
    // Verify gradients
    float* input_grad = (float*)tiny_tensor_grad(input);
    float* weight_grad = (float*)tiny_tensor_grad(fc_params->weights);
    float* bias_grad = (float*)tiny_tensor_grad(fc_params->bias);
    
    if (input_grad != NULL) {
        printf("    - Input gradient: [%.2f, %.2f, %.2f, %.2f]\n", 
               input_grad[0], input_grad[1], input_grad[2], input_grad[3]);
        // input_grad = output_grad @ weights = [1, 1] @ [[1,0,0,0], [0,1,0,0]] = [1, 1, 0, 0]
        TEST_ASSERT(fabsf(input_grad[0] - 1.0f) < 0.01f, "Input grad[0] incorrect");
        TEST_ASSERT(fabsf(input_grad[1] - 1.0f) < 0.01f, "Input grad[1] incorrect");
    }
    
    if (bias_grad != NULL) {
        printf("    - Bias gradient: [%.2f, %.2f] (expected: [1.0, 1.0])\n", bias_grad[0], bias_grad[1]);
        TEST_ASSERT(fabsf(bias_grad[0] - 1.0f) < 0.01f, "Bias grad[0] incorrect");
        TEST_ASSERT(fabsf(bias_grad[1] - 1.0f) < 0.01f, "Bias grad[1] incorrect");
    }
    
    printf("    - Backward propagation verified ✓\n");
    
    // Cleanup
    tiny_tensor_destroy(input);
    tiny_tensor_destroy(output);
    tiny_fc_destroy(fc_params);
    tiny_graph_destroy(graph);
    
    printf("  ✓ FC end-to-end tests passed\n");
    return TINY_OK;
}

tiny_error_t tiny_graph_test_all(void)
{
    printf("========== Graph Tests ==========\n\n");
    
    tiny_error_t err;
    
    err = test_create_destroy();
    if (err != TINY_OK) return err;
    
    err = test_add_node();
    if (err != TINY_OK) return err;
    
    err = test_connect_and_build_order();
    if (err != TINY_OK) return err;
    
    err = test_forward_backward();
    if (err != TINY_OK) return err;
    
    err = test_fc_end_to_end();
    if (err != TINY_OK) return err;
    
    printf("\n========== All Graph Tests Passed ==========\n\n");
    return TINY_OK;
}

