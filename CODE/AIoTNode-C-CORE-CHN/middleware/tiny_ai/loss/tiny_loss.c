/**
 * @file tiny_loss.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Loss functions implementation
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 */

#include "tiny_loss.h"
#include <math.h>
#include <string.h>

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

/**
 * @brief Compute softmax in-place (numerically stable)
 */
static void softmax_stable(float* logits, int num_classes)
{
    // Find maximum for numerical stability
    float max_val = logits[0];
    for (int i = 1; i < num_classes; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    
    // Normalize
    if (sum > 0.0f) {
        for (int i = 0; i < num_classes; i++) {
            logits[i] /= sum;
        }
    }
}

/**
 * @brief Compute sigmoid
 */
static float sigmoid(float x)
{
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

/* ============================================================================
 * MSE LOSS
 * ============================================================================ */

void tiny_mse_forward(tiny_graph_node_t* node)
{
    if (node == NULL || node->inputs == NULL || 
        node->inputs[0] == NULL || node->inputs[1] == NULL ||
        node->outputs == NULL || node->outputs[0] == NULL) {
        return;
    }
    
    tiny_tensor_t* pred = node->inputs[0];
    tiny_tensor_t* target = node->inputs[1];
    tiny_tensor_t* loss = node->outputs[0];
    
    float* pred_data = (float*)tiny_tensor_data(pred);
    float* target_data = (float*)tiny_tensor_data(target);
    float* loss_data = (float*)tiny_tensor_data(loss);
    
    if (pred_data == NULL || target_data == NULL || loss_data == NULL) {
        return;
    }
    
    int numel = tiny_tensor_numel(pred);
    
    // MSE: loss = mean((pred - target)^2)
    float sum = 0.0f;
    for (int i = 0; i < numel; i++) {
        float diff = pred_data[i] - target_data[i];
        sum += diff * diff;
    }
    
    loss_data[0] = sum / numel;
}

void tiny_mse_backward(tiny_graph_node_t* node)
{
    if (node == NULL || node->inputs == NULL || 
        node->inputs[0] == NULL || node->inputs[1] == NULL) {
        return;
    }
    
    tiny_tensor_t* pred = node->inputs[0];
    tiny_tensor_t* target = node->inputs[1];
    tiny_tensor_t* loss = node->outputs[0];
    
    // Check if predictions need gradients
    if (!tiny_tensor_get_requires_grad(pred)) {
        return;
    }
    
    float* pred_data = (float*)tiny_tensor_data(pred);
    float* target_data = (float*)tiny_tensor_data(target);
    void* loss_grad_ptr = tiny_tensor_grad(loss);
    void* pred_grad_ptr = tiny_tensor_grad(pred);
    
    if (pred_data == NULL || target_data == NULL || 
        loss_grad_ptr == NULL || pred_grad_ptr == NULL) {
        return;
    }
    
    float loss_grad = ((float*)loss_grad_ptr)[0];  // Usually 1.0
    float* pred_grad = (float*)pred_grad_ptr;
    
    int numel = tiny_tensor_numel(pred);
    
    // MSE backward: pred_grad = 2 * (pred - target) / numel * loss_grad
    for (int i = 0; i < numel; i++) {
        pred_grad[i] += 2.0f * (pred_data[i] - target_data[i]) / numel * loss_grad;
    }
}

/* ============================================================================
 * MAE LOSS
 * ============================================================================ */

void tiny_mae_forward(tiny_graph_node_t* node)
{
    if (node == NULL || node->inputs == NULL || 
        node->inputs[0] == NULL || node->inputs[1] == NULL ||
        node->outputs == NULL || node->outputs[0] == NULL) {
        return;
    }
    
    tiny_tensor_t* pred = node->inputs[0];
    tiny_tensor_t* target = node->inputs[1];
    tiny_tensor_t* loss = node->outputs[0];
    
    float* pred_data = (float*)tiny_tensor_data(pred);
    float* target_data = (float*)tiny_tensor_data(target);
    float* loss_data = (float*)tiny_tensor_data(loss);
    
    if (pred_data == NULL || target_data == NULL || loss_data == NULL) {
        return;
    }
    
    int numel = tiny_tensor_numel(pred);
    
    // MAE: loss = mean(|pred - target|)
    float sum = 0.0f;
    for (int i = 0; i < numel; i++) {
        float diff = pred_data[i] - target_data[i];
        sum += (diff > 0.0f) ? diff : -diff;  // abs
    }
    
    loss_data[0] = sum / numel;
}

void tiny_mae_backward(tiny_graph_node_t* node)
{
    if (node == NULL || node->inputs == NULL || 
        node->inputs[0] == NULL || node->inputs[1] == NULL) {
        return;
    }
    
    tiny_tensor_t* pred = node->inputs[0];
    tiny_tensor_t* target = node->inputs[1];
    tiny_tensor_t* loss = node->outputs[0];
    
    // Check if predictions need gradients
    if (!tiny_tensor_get_requires_grad(pred)) {
        return;
    }
    
    float* pred_data = (float*)tiny_tensor_data(pred);
    float* target_data = (float*)tiny_tensor_data(target);
    void* loss_grad_ptr = tiny_tensor_grad(loss);
    void* pred_grad_ptr = tiny_tensor_grad(pred);
    
    if (pred_data == NULL || target_data == NULL || 
        loss_grad_ptr == NULL || pred_grad_ptr == NULL) {
        return;
    }
    
    float loss_grad = ((float*)loss_grad_ptr)[0];  // Usually 1.0
    float* pred_grad = (float*)pred_grad_ptr;
    
    int numel = tiny_tensor_numel(pred);
    
    // MAE backward: pred_grad = sign(pred - target) / numel * loss_grad
    for (int i = 0; i < numel; i++) {
        float diff = pred_data[i] - target_data[i];
        pred_grad[i] += ((diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f)) / numel * loss_grad;
    }
}

/* ============================================================================
 * CROSS ENTROPY LOSS
 * ============================================================================ */

void tiny_cross_entropy_forward(tiny_graph_node_t* node)
{
    if (node == NULL || node->inputs == NULL || 
        node->inputs[0] == NULL || node->inputs[1] == NULL ||
        node->outputs == NULL || node->outputs[0] == NULL) {
        return;
    }
    
    tiny_tensor_t* logits = node->inputs[0];
    tiny_tensor_t* targets = node->inputs[1];
    tiny_tensor_t* loss = node->outputs[0];
    
    float* logits_data = (float*)tiny_tensor_data(logits);
    float* targets_data = (float*)tiny_tensor_data(targets);
    float* loss_data = (float*)tiny_tensor_data(loss);
    
    if (logits_data == NULL || targets_data == NULL || loss_data == NULL) {
        return;
    }
    
    // Get dimensions
    int logits_ndim = tiny_tensor_ndim(logits);
    int batch_size = 1;
    int num_classes = 1;
    
    if (logits_ndim == 2) {
        batch_size = tiny_tensor_shape(logits, 0);
        num_classes = tiny_tensor_shape(logits, 1);
    } else if (logits_ndim == 1) {
        num_classes = tiny_tensor_shape(logits, 0);
    } else {
        return;  // Invalid shape
    }
    
    // Check if targets are one-hot or class indices
    int targets_ndim = tiny_tensor_ndim(targets);
    bool one_hot = (targets_ndim == 2) || (targets_ndim == 1 && tiny_tensor_numel(targets) == batch_size * num_classes);
    
    // Compute cross entropy loss
    float sum_loss = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        float* logits_row = logits_data + b * num_classes;
        float* targets_row = targets_data + (one_hot ? b * num_classes : b);
        
        // Compute softmax (in-place on logits_row for efficiency)
        // We'll use a temporary buffer if needed, but for MCU we assume reasonable num_classes
        // For larger num_classes, we could allocate dynamically, but for MCU static is better
        float softmax_probs[32];  // Static buffer for up to 32 classes (reasonable for MCU)
        if (num_classes > 32) {
            // For MCU, 32 classes should be enough for most use cases
            // If more classes are needed, consider dynamic allocation
            return;
        }
        
        memcpy(softmax_probs, logits_row, num_classes * sizeof(float));
        softmax_stable(softmax_probs, num_classes);
        
        // Compute cross entropy for this sample
        float sample_loss = 0.0f;
        if (one_hot) {
            // One-hot encoded targets
            for (int c = 0; c < num_classes; c++) {
                if (targets_row[c] > 0.0f) {
                    sample_loss -= targets_row[c] * logf(softmax_probs[c] + 1e-8f);  // Add epsilon for numerical stability
                }
            }
        } else {
            // Class index targets
            int class_idx = (int)targets_row[0];
            if (class_idx >= 0 && class_idx < num_classes) {
                sample_loss = -logf(softmax_probs[class_idx] + 1e-8f);
            }
        }
        
        sum_loss += sample_loss;
    }
    
    loss_data[0] = sum_loss / batch_size;
}

void tiny_cross_entropy_backward(tiny_graph_node_t* node)
{
    if (node == NULL || node->inputs == NULL || 
        node->inputs[0] == NULL || node->inputs[1] == NULL) {
        return;
    }
    
    tiny_tensor_t* logits = node->inputs[0];
    tiny_tensor_t* targets = node->inputs[1];
    tiny_tensor_t* loss = node->outputs[0];
    
    // Check if logits need gradients
    if (!tiny_tensor_get_requires_grad(logits)) {
        return;
    }
    
    float* logits_data = (float*)tiny_tensor_data(logits);
    float* targets_data = (float*)tiny_tensor_data(targets);
    void* loss_grad_ptr = tiny_tensor_grad(loss);
    void* logits_grad_ptr = tiny_tensor_grad(logits);
    
    if (logits_data == NULL || targets_data == NULL || 
        loss_grad_ptr == NULL || logits_grad_ptr == NULL) {
        return;
    }
    
    float loss_grad = ((float*)loss_grad_ptr)[0];  // Usually 1.0
    float* logits_grad = (float*)logits_grad_ptr;
    
    // Get dimensions
    int logits_ndim = tiny_tensor_ndim(logits);
    int batch_size = 1;
    int num_classes = 1;
    
    if (logits_ndim == 2) {
        batch_size = tiny_tensor_shape(logits, 0);
        num_classes = tiny_tensor_shape(logits, 1);
    } else if (logits_ndim == 1) {
        num_classes = tiny_tensor_shape(logits, 0);
    } else {
        return;
    }
    
    // Check if targets are one-hot or class indices
    int targets_ndim = tiny_tensor_ndim(targets);
    bool one_hot = (targets_ndim == 2) || (targets_ndim == 1 && tiny_tensor_numel(targets) == batch_size * num_classes);
    
    // Compute gradients
    for (int b = 0; b < batch_size; b++) {
        float* logits_row = logits_data + b * num_classes;
        float* targets_row = targets_data + (one_hot ? b * num_classes : b);
        float* logits_grad_row = logits_grad + b * num_classes;
        
        // Compute softmax
        float softmax_probs[32];  // Static buffer for up to 32 classes
        if (num_classes > 32) {
            return;
        }
        
        memcpy(softmax_probs, logits_row, num_classes * sizeof(float));
        softmax_stable(softmax_probs, num_classes);
        
        // Compute gradient: logits_grad = (softmax - target) / batch_size
        if (one_hot) {
            for (int c = 0; c < num_classes; c++) {
                logits_grad_row[c] += (softmax_probs[c] - targets_row[c]) / batch_size * loss_grad;
            }
        } else {
            int class_idx = (int)targets_row[0];
            for (int c = 0; c < num_classes; c++) {
                float target_val = (c == class_idx) ? 1.0f : 0.0f;
                logits_grad_row[c] += (softmax_probs[c] - target_val) / batch_size * loss_grad;
            }
        }
    }
}

/* ============================================================================
 * BINARY CROSS ENTROPY LOSS
 * ============================================================================ */

void tiny_binary_cross_entropy_forward(tiny_graph_node_t* node)
{
    if (node == NULL || node->inputs == NULL || 
        node->inputs[0] == NULL || node->inputs[1] == NULL ||
        node->outputs == NULL || node->outputs[0] == NULL) {
        return;
    }
    
    tiny_tensor_t* pred = node->inputs[0];
    tiny_tensor_t* target = node->inputs[1];
    tiny_tensor_t* loss = node->outputs[0];
    
    float* pred_data = (float*)tiny_tensor_data(pred);
    float* target_data = (float*)tiny_tensor_data(target);
    float* loss_data = (float*)tiny_tensor_data(loss);
    
    if (pred_data == NULL || target_data == NULL || loss_data == NULL) {
        return;
    }
    
    int numel = tiny_tensor_numel(pred);
    
    // Binary Cross Entropy: loss = -mean(target * log(sigmoid(pred)) + (1-target) * log(1-sigmoid(pred)))
    float sum = 0.0f;
    const float epsilon = 1e-8f;  // For numerical stability
    
    for (int i = 0; i < numel; i++) {
        float sigmoid_val = sigmoid(pred_data[i]);
        float t = target_data[i];
        
        // Clamp sigmoid_val to avoid log(0)
        if (sigmoid_val < epsilon) sigmoid_val = epsilon;
        if (sigmoid_val > 1.0f - epsilon) sigmoid_val = 1.0f - epsilon;
        
        sum -= t * logf(sigmoid_val) + (1.0f - t) * logf(1.0f - sigmoid_val);
    }
    
    loss_data[0] = sum / numel;
}

void tiny_binary_cross_entropy_backward(tiny_graph_node_t* node)
{
    if (node == NULL || node->inputs == NULL || 
        node->inputs[0] == NULL || node->inputs[1] == NULL) {
        return;
    }
    
    tiny_tensor_t* pred = node->inputs[0];
    tiny_tensor_t* target = node->inputs[1];
    tiny_tensor_t* loss = node->outputs[0];
    
    // Check if predictions need gradients
    if (!tiny_tensor_get_requires_grad(pred)) {
        return;
    }
    
    float* pred_data = (float*)tiny_tensor_data(pred);
    float* target_data = (float*)tiny_tensor_data(target);
    void* loss_grad_ptr = tiny_tensor_grad(loss);
    void* pred_grad_ptr = tiny_tensor_grad(pred);
    
    if (pred_data == NULL || target_data == NULL || 
        loss_grad_ptr == NULL || pred_grad_ptr == NULL) {
        return;
    }
    
    float loss_grad = ((float*)loss_grad_ptr)[0];  // Usually 1.0
    float* pred_grad = (float*)pred_grad_ptr;
    
    int numel = tiny_tensor_numel(pred);
    
    // Binary Cross Entropy backward: pred_grad = (sigmoid(pred) - target) / numel * loss_grad
    for (int i = 0; i < numel; i++) {
        float sigmoid_val = sigmoid(pred_data[i]);
        pred_grad[i] += (sigmoid_val - target_data[i]) / numel * loss_grad;
    }
}

