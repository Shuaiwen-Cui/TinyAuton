/**
 * @file tiny_optimizer.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Optimizers implementation
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 */

#include "tiny_optimizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * OPTIMIZER CREATION AND DESTRUCTION
 * ============================================================================ */

tiny_sgd_optimizer_t* tiny_optimizer_sgd_create(float learning_rate)
{
    if (learning_rate <= 0.0f) {
        return NULL;
    }
    
    tiny_sgd_optimizer_t* optimizer = (tiny_sgd_optimizer_t*)malloc(sizeof(tiny_sgd_optimizer_t));
    if (optimizer == NULL) {
        return NULL;
    }
    
    memset(optimizer, 0, sizeof(tiny_sgd_optimizer_t));
    optimizer->base.type = TINY_OPTIMIZER_SGD;
    optimizer->base.learning_rate = learning_rate;
    optimizer->base.step_count = 0;
    
    return optimizer;
}

tiny_sgd_momentum_optimizer_t* tiny_optimizer_sgd_momentum_create(float learning_rate, float momentum)
{
    if (learning_rate <= 0.0f || momentum < 0.0f || momentum >= 1.0f) {
        return NULL;
    }
    
    tiny_sgd_momentum_optimizer_t* optimizer = (tiny_sgd_momentum_optimizer_t*)malloc(sizeof(tiny_sgd_momentum_optimizer_t));
    if (optimizer == NULL) {
        return NULL;
    }
    
    memset(optimizer, 0, sizeof(tiny_sgd_momentum_optimizer_t));
    optimizer->base.type = TINY_OPTIMIZER_SGD_MOMENTUM;
    optimizer->base.learning_rate = learning_rate;
    optimizer->base.step_count = 0;
    optimizer->momentum = momentum;
    optimizer->velocity = NULL;  // Will be created on first step
    
    return optimizer;
}

tiny_adam_optimizer_t* tiny_optimizer_adam_create(float learning_rate, float beta1, float beta2, float epsilon)
{
    if (learning_rate <= 0.0f || beta1 < 0.0f || beta1 >= 1.0f ||
        beta2 < 0.0f || beta2 >= 1.0f || epsilon <= 0.0f) {
        return NULL;
    }
    
    tiny_adam_optimizer_t* optimizer = (tiny_adam_optimizer_t*)malloc(sizeof(tiny_adam_optimizer_t));
    if (optimizer == NULL) {
        return NULL;
    }
    
    memset(optimizer, 0, sizeof(tiny_adam_optimizer_t));
    optimizer->base.type = TINY_OPTIMIZER_ADAM;
    optimizer->base.learning_rate = learning_rate;
    optimizer->base.step_count = 0;
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->epsilon = epsilon;
    optimizer->m = NULL;  // Will be created on first step
    optimizer->v = NULL;  // Will be created on first step
    
    return optimizer;
}

tiny_rmsprop_optimizer_t* tiny_optimizer_rmsprop_create(float learning_rate, float alpha, float epsilon)
{
    if (learning_rate <= 0.0f || alpha < 0.0f || alpha >= 1.0f || epsilon <= 0.0f) {
        return NULL;
    }
    
    tiny_rmsprop_optimizer_t* optimizer = (tiny_rmsprop_optimizer_t*)malloc(sizeof(tiny_rmsprop_optimizer_t));
    if (optimizer == NULL) {
        return NULL;
    }
    
    memset(optimizer, 0, sizeof(tiny_rmsprop_optimizer_t));
    optimizer->base.type = TINY_OPTIMIZER_RMSPROP;
    optimizer->base.learning_rate = learning_rate;
    optimizer->base.step_count = 0;
    optimizer->alpha = alpha;
    optimizer->epsilon = epsilon;
    optimizer->cache = NULL;  // Will be created on first step
    
    return optimizer;
}

tiny_adagrad_optimizer_t* tiny_optimizer_adagrad_create(float learning_rate, float epsilon)
{
    if (learning_rate <= 0.0f || epsilon <= 0.0f) {
        return NULL;
    }
    
    tiny_adagrad_optimizer_t* optimizer = (tiny_adagrad_optimizer_t*)malloc(sizeof(tiny_adagrad_optimizer_t));
    if (optimizer == NULL) {
        return NULL;
    }
    
    memset(optimizer, 0, sizeof(tiny_adagrad_optimizer_t));
    optimizer->base.type = TINY_OPTIMIZER_ADAGRAD;
    optimizer->base.learning_rate = learning_rate;
    optimizer->base.step_count = 0;
    optimizer->epsilon = epsilon;
    optimizer->cache = NULL;  // Will be created on first step
    
    return optimizer;
}

void tiny_optimizer_destroy(void* optimizer)
{
    if (optimizer == NULL) {
        return;
    }
    
    tiny_optimizer_t* base = (tiny_optimizer_t*)optimizer;
    
    switch (base->type) {
        case TINY_OPTIMIZER_SGD:
            free(optimizer);
            break;
            
        case TINY_OPTIMIZER_SGD_MOMENTUM: {
            tiny_sgd_momentum_optimizer_t* opt = (tiny_sgd_momentum_optimizer_t*)optimizer;
            if (opt->velocity != NULL) {
                tiny_tensor_destroy(opt->velocity);
            }
            free(optimizer);
            break;
        }
            
        case TINY_OPTIMIZER_ADAM: {
            tiny_adam_optimizer_t* opt = (tiny_adam_optimizer_t*)optimizer;
            if (opt->m != NULL) {
                tiny_tensor_destroy(opt->m);
            }
            if (opt->v != NULL) {
                tiny_tensor_destroy(opt->v);
            }
            free(optimizer);
            break;
        }
            
        case TINY_OPTIMIZER_RMSPROP: {
            tiny_rmsprop_optimizer_t* opt = (tiny_rmsprop_optimizer_t*)optimizer;
            if (opt->cache != NULL) {
                tiny_tensor_destroy(opt->cache);
            }
            free(optimizer);
            break;
        }
            
        case TINY_OPTIMIZER_ADAGRAD: {
            tiny_adagrad_optimizer_t* opt = (tiny_adagrad_optimizer_t*)optimizer;
            if (opt->cache != NULL) {
                tiny_tensor_destroy(opt->cache);
            }
            free(optimizer);
            break;
        }
            
        default:
            free(optimizer);
            break;
    }
}

/* ============================================================================
 * PARAMETER UPDATE
 * ============================================================================ */

tiny_error_t tiny_optimizer_sgd_step(tiny_sgd_optimizer_t* optimizer, tiny_tensor_t* param)
{
    if (optimizer == NULL || param == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    if (!tiny_tensor_get_requires_grad(param)) {
        return TINY_ERR_AI_INVALID_STATE;
    }
    
    void* grad_ptr = tiny_tensor_grad(param);
    float* param_data = (float*)tiny_tensor_data(param);
    
    if (grad_ptr == NULL || param_data == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    float* grad = (float*)grad_ptr;
    int numel = tiny_tensor_numel(param);
    
    // SGD: param = param - lr * grad
    for (int i = 0; i < numel; i++) {
        param_data[i] -= optimizer->base.learning_rate * grad[i];
    }
    
    optimizer->base.step_count++;
    return TINY_OK;
}

tiny_error_t tiny_optimizer_sgd_momentum_step(tiny_sgd_momentum_optimizer_t* optimizer, tiny_tensor_t* param)
{
    if (optimizer == NULL || param == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    if (!tiny_tensor_get_requires_grad(param)) {
        return TINY_ERR_AI_INVALID_STATE;
    }
    
    void* grad_ptr = tiny_tensor_grad(param);
    float* param_data = (float*)tiny_tensor_data(param);
    
    if (grad_ptr == NULL || param_data == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    float* grad = (float*)grad_ptr;
    int numel = tiny_tensor_numel(param);
    
    // Initialize velocity buffer if needed
    if (optimizer->velocity == NULL) {
        // Create velocity tensor with same shape as param
        int ndim = tiny_tensor_ndim(param);
        int shape[TINY_AI_TENSOR_MAX_DIMS];
        for (int i = 0; i < ndim; i++) {
            shape[i] = tiny_tensor_shape(param, i);
        }
        optimizer->velocity = tiny_tensor_create(shape, ndim, TINY_AI_DTYPE_FLOAT32);
        if (optimizer->velocity == NULL) {
            return TINY_ERR_AI_NO_MEM;
        }
        tiny_tensor_zero(optimizer->velocity);
    }
    
    float* velocity = (float*)tiny_tensor_data(optimizer->velocity);
    
    // SGD with Momentum: v = momentum * v + grad; param = param - lr * v
    for (int i = 0; i < numel; i++) {
        velocity[i] = optimizer->momentum * velocity[i] + grad[i];
        param_data[i] -= optimizer->base.learning_rate * velocity[i];
    }
    
    optimizer->base.step_count++;
    return TINY_OK;
}

tiny_error_t tiny_optimizer_adam_step(tiny_adam_optimizer_t* optimizer, tiny_tensor_t* param)
{
    if (optimizer == NULL || param == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    if (!tiny_tensor_get_requires_grad(param)) {
        return TINY_ERR_AI_INVALID_STATE;
    }
    
    void* grad_ptr = tiny_tensor_grad(param);
    float* param_data = (float*)tiny_tensor_data(param);
    
    if (grad_ptr == NULL || param_data == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    float* grad = (float*)grad_ptr;
    int numel = tiny_tensor_numel(param);
    
    // Initialize moment buffers if needed
    if (optimizer->m == NULL) {
        int ndim = tiny_tensor_ndim(param);
        int shape[TINY_AI_TENSOR_MAX_DIMS];
        for (int i = 0; i < ndim; i++) {
            shape[i] = tiny_tensor_shape(param, i);
        }
        optimizer->m = tiny_tensor_create(shape, ndim, TINY_AI_DTYPE_FLOAT32);
        optimizer->v = tiny_tensor_create(shape, ndim, TINY_AI_DTYPE_FLOAT32);
        if (optimizer->m == NULL || optimizer->v == NULL) {
            if (optimizer->m != NULL) tiny_tensor_destroy(optimizer->m);
            if (optimizer->v != NULL) tiny_tensor_destroy(optimizer->v);
            optimizer->m = NULL;
            optimizer->v = NULL;
            return TINY_ERR_AI_NO_MEM;
        }
        tiny_tensor_zero(optimizer->m);
        tiny_tensor_zero(optimizer->v);
    }
    
    float* m = (float*)tiny_tensor_data(optimizer->m);
    float* v = (float*)tiny_tensor_data(optimizer->v);
    
    optimizer->base.step_count++;
    int t = optimizer->base.step_count;
    
    // Adam: m = beta1 * m + (1 - beta1) * grad
    //       v = beta2 * v + (1 - beta2) * grad^2
    //       m_hat = m / (1 - beta1^t)
    //       v_hat = v / (1 - beta2^t)
    //       param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
    
    float beta1_t = 1.0f - powf(optimizer->beta1, t);
    float beta2_t = 1.0f - powf(optimizer->beta2, t);
    
    for (int i = 0; i < numel; i++) {
        // Update biased first moment estimate
        m[i] = optimizer->beta1 * m[i] + (1.0f - optimizer->beta1) * grad[i];
        
        // Update biased second raw moment estimate
        v[i] = optimizer->beta2 * v[i] + (1.0f - optimizer->beta2) * grad[i] * grad[i];
        
        // Compute bias-corrected first moment estimate
        float m_hat = m[i] / beta1_t;
        
        // Compute bias-corrected second raw moment estimate
        float v_hat = v[i] / beta2_t;
        
        // Update parameters
        param_data[i] -= optimizer->base.learning_rate * m_hat / (sqrtf(v_hat) + optimizer->epsilon);
    }
    
    return TINY_OK;
}

tiny_error_t tiny_optimizer_rmsprop_step(tiny_rmsprop_optimizer_t* optimizer, tiny_tensor_t* param)
{
    if (optimizer == NULL || param == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    if (!tiny_tensor_get_requires_grad(param)) {
        return TINY_ERR_AI_INVALID_STATE;
    }
    
    void* grad_ptr = tiny_tensor_grad(param);
    float* param_data = (float*)tiny_tensor_data(param);
    
    if (grad_ptr == NULL || param_data == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    float* grad = (float*)grad_ptr;
    int numel = tiny_tensor_numel(param);
    
    // Initialize cache if needed
    if (optimizer->cache == NULL) {
        int ndim = tiny_tensor_ndim(param);
        int shape[TINY_AI_TENSOR_MAX_DIMS];
        for (int i = 0; i < ndim; i++) {
            shape[i] = tiny_tensor_shape(param, i);
        }
        optimizer->cache = tiny_tensor_create(shape, ndim, TINY_AI_DTYPE_FLOAT32);
        if (optimizer->cache == NULL) {
            return TINY_ERR_AI_NO_MEM;
        }
        tiny_tensor_zero(optimizer->cache);
    }
    
    float* cache = (float*)tiny_tensor_data(optimizer->cache);
    
    // RMSprop: cache = alpha * cache + (1 - alpha) * grad^2
    //          param = param - lr * grad / (sqrt(cache) + epsilon)
    for (int i = 0; i < numel; i++) {
        cache[i] = optimizer->alpha * cache[i] + (1.0f - optimizer->alpha) * grad[i] * grad[i];
        param_data[i] -= optimizer->base.learning_rate * grad[i] / (sqrtf(cache[i]) + optimizer->epsilon);
    }
    
    optimizer->base.step_count++;
    return TINY_OK;
}

tiny_error_t tiny_optimizer_adagrad_step(tiny_adagrad_optimizer_t* optimizer, tiny_tensor_t* param)
{
    if (optimizer == NULL || param == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    if (!tiny_tensor_get_requires_grad(param)) {
        return TINY_ERR_AI_INVALID_STATE;
    }
    
    void* grad_ptr = tiny_tensor_grad(param);
    float* param_data = (float*)tiny_tensor_data(param);
    
    if (grad_ptr == NULL || param_data == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    float* grad = (float*)grad_ptr;
    int numel = tiny_tensor_numel(param);
    
    // Initialize cache if needed
    if (optimizer->cache == NULL) {
        int ndim = tiny_tensor_ndim(param);
        int shape[TINY_AI_TENSOR_MAX_DIMS];
        for (int i = 0; i < ndim; i++) {
            shape[i] = tiny_tensor_shape(param, i);
        }
        optimizer->cache = tiny_tensor_create(shape, ndim, TINY_AI_DTYPE_FLOAT32);
        if (optimizer->cache == NULL) {
            return TINY_ERR_AI_NO_MEM;
        }
        tiny_tensor_zero(optimizer->cache);
    }
    
    float* cache = (float*)tiny_tensor_data(optimizer->cache);
    
    // AdaGrad: cache = cache + grad^2
    //          param = param - lr * grad / (sqrt(cache) + epsilon)
    for (int i = 0; i < numel; i++) {
        cache[i] += grad[i] * grad[i];
        param_data[i] -= optimizer->base.learning_rate * grad[i] / (sqrtf(cache[i]) + optimizer->epsilon);
    }
    
    optimizer->base.step_count++;
    return TINY_OK;
}

/* ============================================================================
 * UTILITY
 * ============================================================================ */

void tiny_optimizer_set_learning_rate(void* optimizer, float learning_rate)
{
    if (optimizer == NULL || learning_rate <= 0.0f) {
        return;
    }
    
    tiny_optimizer_t* base = (tiny_optimizer_t*)optimizer;
    base->learning_rate = learning_rate;
}

float tiny_optimizer_get_learning_rate(const void* optimizer)
{
    if (optimizer == NULL) {
        return 0.0f;
    }
    
    const tiny_optimizer_t* base = (const tiny_optimizer_t*)optimizer;
    return base->learning_rate;
}

