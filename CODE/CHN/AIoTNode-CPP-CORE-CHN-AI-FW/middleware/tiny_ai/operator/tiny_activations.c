/**
 * @file tiny_activations.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Activation functions implementation
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 */

#include "tiny_activations.h"
#include <math.h>
#include <string.h>

/* ============================================================================
 * RELU ACTIVATION
 * ============================================================================ */

void tiny_relu_forward(tiny_graph_node_t* node)
{
    if (node == NULL || node->inputs == NULL || node->inputs[0] == NULL ||
        node->outputs == NULL || node->outputs[0] == NULL) {
        return;
    }
    
    tiny_tensor_t* input = node->inputs[0];
    tiny_tensor_t* output = node->outputs[0];
    
    float* input_data = (float*)tiny_tensor_data(input);
    float* output_data = (float*)tiny_tensor_data(output);
    
    if (input_data == NULL || output_data == NULL) {
        return;
    }
    
    int numel = tiny_tensor_numel(input);
    
    // ReLU: output = max(0, input)
    for (int i = 0; i < numel; i++) {
        output_data[i] = (input_data[i] > 0.0f) ? input_data[i] : 0.0f;
    }
}

void tiny_relu_backward(tiny_graph_node_t* node)
{
    if (node == NULL || node->inputs == NULL || node->inputs[0] == NULL ||
        node->outputs == NULL || node->outputs[0] == NULL) {
        return;
    }
    
    tiny_tensor_t* input = node->inputs[0];
    tiny_tensor_t* output = node->outputs[0];
    
    // Check if input needs gradients
    if (!tiny_tensor_get_requires_grad(input)) {
        return;
    }
    
    float* input_data = (float*)tiny_tensor_data(input);
    float* output_data = (float*)tiny_tensor_data(output);
    void* output_grad_ptr = tiny_tensor_grad(output);
    void* input_grad_ptr = tiny_tensor_grad(input);
    
    if (input_data == NULL || output_data == NULL || 
        output_grad_ptr == NULL || input_grad_ptr == NULL) {
        return;
    }
    
    float* output_grad = (float*)output_grad_ptr;
    float* input_grad = (float*)input_grad_ptr;
    
    int numel = tiny_tensor_numel(input);
    
    // ReLU backward: input_grad = output_grad * (input > 0 ? 1 : 0)
    for (int i = 0; i < numel; i++) {
        input_grad[i] += output_grad[i] * (input_data[i] > 0.0f ? 1.0f : 0.0f);
    }
}

/* ============================================================================
 * SIGMOID ACTIVATION
 * ============================================================================ */

void tiny_sigmoid_forward(tiny_graph_node_t* node)
{
    if (node == NULL || node->inputs == NULL || node->inputs[0] == NULL ||
        node->outputs == NULL || node->outputs[0] == NULL) {
        return;
    }
    
    tiny_tensor_t* input = node->inputs[0];
    tiny_tensor_t* output = node->outputs[0];
    
    float* input_data = (float*)tiny_tensor_data(input);
    float* output_data = (float*)tiny_tensor_data(output);
    
    if (input_data == NULL || output_data == NULL) {
        return;
    }
    
    int numel = tiny_tensor_numel(input);
    
    // Sigmoid: output = 1 / (1 + exp(-input))
    for (int i = 0; i < numel; i++) {
        // Clamp input to avoid overflow
        float x = input_data[i];
        if (x > 10.0f) {
            output_data[i] = 1.0f;
        } else if (x < -10.0f) {
            output_data[i] = 0.0f;
        } else {
            output_data[i] = 1.0f / (1.0f + expf(-x));
        }
    }
}

void tiny_sigmoid_backward(tiny_graph_node_t* node)
{
    if (node == NULL || node->inputs == NULL || node->inputs[0] == NULL ||
        node->outputs == NULL || node->outputs[0] == NULL) {
        return;
    }
    
    tiny_tensor_t* input = node->inputs[0];
    tiny_tensor_t* output = node->outputs[0];
    
    // Check if input needs gradients
    if (!tiny_tensor_get_requires_grad(input)) {
        return;
    }
    
    float* output_data = (float*)tiny_tensor_data(output);
    void* output_grad_ptr = tiny_tensor_grad(output);
    void* input_grad_ptr = tiny_tensor_grad(input);
    
    if (output_data == NULL || output_grad_ptr == NULL || input_grad_ptr == NULL) {
        return;
    }
    
    float* output_grad = (float*)output_grad_ptr;
    float* input_grad = (float*)input_grad_ptr;
    
    int numel = tiny_tensor_numel(output);
    
    // Sigmoid backward: input_grad = output_grad * output * (1 - output)
    // Uses: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    for (int i = 0; i < numel; i++) {
        float sigmoid_val = output_data[i];
        input_grad[i] += output_grad[i] * sigmoid_val * (1.0f - sigmoid_val);
    }
}

/* ============================================================================
 * TANH ACTIVATION
 * ============================================================================ */

void tiny_tanh_forward(tiny_graph_node_t* node)
{
    if (node == NULL || node->inputs == NULL || node->inputs[0] == NULL ||
        node->outputs == NULL || node->outputs[0] == NULL) {
        return;
    }
    
    tiny_tensor_t* input = node->inputs[0];
    tiny_tensor_t* output = node->outputs[0];
    
    float* input_data = (float*)tiny_tensor_data(input);
    float* output_data = (float*)tiny_tensor_data(output);
    
    if (input_data == NULL || output_data == NULL) {
        return;
    }
    
    int numel = tiny_tensor_numel(input);
    
    // Tanh: output = tanh(input)
    for (int i = 0; i < numel; i++) {
        output_data[i] = tanhf(input_data[i]);
    }
}

void tiny_tanh_backward(tiny_graph_node_t* node)
{
    if (node == NULL || node->inputs == NULL || node->inputs[0] == NULL ||
        node->outputs == NULL || node->outputs[0] == NULL) {
        return;
    }
    
    tiny_tensor_t* input = node->inputs[0];
    tiny_tensor_t* output = node->outputs[0];
    
    // Check if input needs gradients
    if (!tiny_tensor_get_requires_grad(input)) {
        return;
    }
    
    float* output_data = (float*)tiny_tensor_data(output);
    void* output_grad_ptr = tiny_tensor_grad(output);
    void* input_grad_ptr = tiny_tensor_grad(input);
    
    if (output_data == NULL || output_grad_ptr == NULL || input_grad_ptr == NULL) {
        return;
    }
    
    float* output_grad = (float*)output_grad_ptr;
    float* input_grad = (float*)input_grad_ptr;
    
    int numel = tiny_tensor_numel(output);
    
    // Tanh backward: input_grad = output_grad * (1 - output^2)
    // Uses: tanh'(x) = 1 - tanh(x)^2
    for (int i = 0; i < numel; i++) {
        float tanh_val = output_data[i];
        input_grad[i] += output_grad[i] * (1.0f - tanh_val * tanh_val);
    }
}

