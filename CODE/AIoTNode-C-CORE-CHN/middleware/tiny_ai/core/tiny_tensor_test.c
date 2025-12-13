/**
 * @file tiny_tensor_test.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Test implementation for tensor module
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 */

#include "tiny_tensor_test.h"
#include <stdio.h>
#include <assert.h>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("TEST FAILED: %s\n", msg); \
            return TINY_FAIL; \
        } \
    } while(0)

static tiny_error_t test_create_destroy(void)
{
    printf("Testing create/destroy...\n");
    
    // Test 1D tensor (vector of 10 elements)
    printf("  Creating 1D tensor: shape=[10]\n");
    int shape1d[] = {10};
    tiny_tensor_t* t1 = tiny_tensor_create(shape1d, 1, TINY_AI_DTYPE_FLOAT32);
    TEST_ASSERT(t1 != NULL, "Failed to create 1D tensor");
    printf("    - Created successfully\n");
    printf("    - Numel: %d (expected: 10)\n", tiny_tensor_numel(t1));
    printf("    - Ndim: %d (expected: 1)\n", tiny_tensor_ndim(t1));
    TEST_ASSERT(tiny_tensor_numel(t1) == 10, "Wrong numel for 1D tensor");
    TEST_ASSERT(tiny_tensor_ndim(t1) == 1, "Wrong ndim for 1D tensor");
    tiny_tensor_destroy(t1);
    printf("    - Destroyed successfully\n");
    
    // Test 2D tensor (3x4 matrix)
    printf("  Creating 2D tensor: shape=[3, 4] (3 rows, 4 cols)\n");
    int shape2d[] = {3, 4};
    tiny_tensor_t* t2 = tiny_tensor_create(shape2d, 2, TINY_AI_DTYPE_FLOAT32);
    TEST_ASSERT(t2 != NULL, "Failed to create 2D tensor");
    printf("    - Created successfully\n");
    printf("    - Numel: %d (expected: 12)\n", tiny_tensor_numel(t2));
    printf("    - Ndim: %d (expected: 2)\n", tiny_tensor_ndim(t2));
    printf("    - Shape[0]: %d (expected: 3)\n", tiny_tensor_shape(t2, 0));
    printf("    - Shape[1]: %d (expected: 4)\n", tiny_tensor_shape(t2, 1));
    TEST_ASSERT(tiny_tensor_numel(t2) == 12, "Wrong numel for 2D tensor");
    TEST_ASSERT(tiny_tensor_ndim(t2) == 2, "Wrong ndim for 2D tensor");
    TEST_ASSERT(tiny_tensor_shape(t2, 0) == 3, "Wrong shape[0] for 2D tensor");
    TEST_ASSERT(tiny_tensor_shape(t2, 1) == 4, "Wrong shape[1] for 2D tensor");
    tiny_tensor_destroy(t2);
    printf("    - Destroyed successfully\n");
    
    // Test invalid shape
    printf("  Testing invalid shape: shape=[0] (should fail)\n");
    int invalid_shape[] = {0};
    tiny_tensor_t* t_invalid = tiny_tensor_create(invalid_shape, 1, TINY_AI_DTYPE_FLOAT32);
    TEST_ASSERT(t_invalid == NULL, "Should fail to create tensor with invalid shape");
    printf("    - Correctly rejected invalid shape\n");
    
    printf("  ✓ Create/destroy tests passed\n");
    return TINY_OK;
}

static tiny_error_t test_data_access(void)
{
    printf("Testing data access...\n");
    
    printf("  Creating 1D tensor: shape=[5]\n");
    int shape[] = {5};
    tiny_tensor_t* t = tiny_tensor_create(shape, 1, TINY_AI_DTYPE_FLOAT32);
    TEST_ASSERT(t != NULL, "Failed to create tensor");
    
    // Test set/get with specific values
    printf("  Setting values: [0.0, 1.5, 3.0, 4.5, 6.0]\n");
    float expected_values[] = {0.0f, 1.5f, 3.0f, 4.5f, 6.0f};
    for (int i = 0; i < 5; i++) {
        tiny_error_t err = tiny_tensor_set_f32(t, i, expected_values[i]);
        TEST_ASSERT(err == TINY_OK, "Failed to set value");
    }
    
    printf("  Reading back values:\n");
    for (int i = 0; i < 5; i++) {
        float val = tiny_tensor_get_f32(t, i);
        printf("    - Index %d: got %.2f (expected %.2f)\n", i, val, expected_values[i]);
        TEST_ASSERT(val == expected_values[i], "Wrong value retrieved");
    }
    
    tiny_tensor_destroy(t);
    
    printf("  ✓ Data access tests passed\n");
    return TINY_OK;
}

static tiny_error_t test_copy(void)
{
    printf("Testing copy...\n");
    
    printf("  Creating 2D tensors: shape=[3, 4] (source and destination)\n");
    int shape[] = {3, 4};
    tiny_tensor_t* src = tiny_tensor_create(shape, 2, TINY_AI_DTYPE_FLOAT32);
    tiny_tensor_t* dst = tiny_tensor_create(shape, 2, TINY_AI_DTYPE_FLOAT32);
    TEST_ASSERT(src != NULL && dst != NULL, "Failed to create tensors");
    
    // Fill source with values (0.0, 1.0, 2.0, ..., 11.0)
    printf("  Filling source tensor with values 0.0 to 11.0\n");
    for (int i = 0; i < 12; i++) {
        tiny_tensor_set_f32(src, i, (float)i);
    }
    
    // Copy
    printf("  Copying source to destination...\n");
    tiny_error_t err = tiny_tensor_copy(src, dst);
    TEST_ASSERT(err == TINY_OK, "Failed to copy tensor");
    
    // Verify a few sample values
    printf("  Verifying copy (checking sample indices):\n");
    int sample_indices[] = {0, 5, 11};
    for (int j = 0; j < 3; j++) {
        int idx = sample_indices[j];
        float src_val = tiny_tensor_get_f32(src, idx);
        float dst_val = tiny_tensor_get_f32(dst, idx);
        printf("    - Index %d: src=%.1f, dst=%.1f\n", idx, src_val, dst_val);
        TEST_ASSERT(src_val == dst_val, "Copy failed - values don't match");
    }
    
    // Verify all values
    for (int i = 0; i < 12; i++) {
        float src_val = tiny_tensor_get_f32(src, i);
        float dst_val = tiny_tensor_get_f32(dst, i);
        TEST_ASSERT(src_val == dst_val, "Copy failed - values don't match");
    }
    printf("    - All 12 values match correctly\n");
    
    tiny_tensor_destroy(src);
    tiny_tensor_destroy(dst);
    
    printf("  ✓ Copy tests passed\n");
    return TINY_OK;
}

static tiny_error_t test_fill_zero(void)
{
    printf("Testing fill/zero...\n");
    
    printf("  Creating 1D tensor: shape=[10]\n");
    int shape[] = {10};
    tiny_tensor_t* t = tiny_tensor_create(shape, 1, TINY_AI_DTYPE_FLOAT32);
    TEST_ASSERT(t != NULL, "Failed to create tensor");
    
    // Test fill with 3.14
    printf("  Filling tensor with value 3.14...\n");
    tiny_error_t err = tiny_tensor_fill(t, 3.14f);
    TEST_ASSERT(err == TINY_OK, "Failed to fill tensor");
    
    printf("  Verifying fill (checking first 3 and last 3 elements):\n");
    for (int i = 0; i < 3; i++) {
        float val = tiny_tensor_get_f32(t, i);
        printf("    - Index %d: %.2f (expected 3.14)\n", i, val);
        TEST_ASSERT(val == 3.14f, "Fill failed");
    }
    for (int i = 7; i < 10; i++) {
        float val = tiny_tensor_get_f32(t, i);
        printf("    - Index %d: %.2f (expected 3.14)\n", i, val);
        TEST_ASSERT(val == 3.14f, "Fill failed");
    }
    
    // Test zero
    printf("  Zeroing tensor...\n");
    err = tiny_tensor_zero(t);
    TEST_ASSERT(err == TINY_OK, "Failed to zero tensor");
    
    printf("  Verifying zero (checking first 3 elements):\n");
    for (int i = 0; i < 3; i++) {
        float val = tiny_tensor_get_f32(t, i);
        printf("    - Index %d: %.2f (expected 0.00)\n", i, val);
        TEST_ASSERT(val == 0.0f, "Zero failed");
    }
    
    // Verify all are zero
    for (int i = 0; i < 10; i++) {
        float val = tiny_tensor_get_f32(t, i);
        TEST_ASSERT(val == 0.0f, "Zero failed");
    }
    
    tiny_tensor_destroy(t);
    
    printf("  ✓ Fill/zero tests passed\n");
    return TINY_OK;
}

#if TINY_AI_ENABLE_GRADIENTS
static tiny_error_t test_gradients(void)
{
    printf("Testing gradients...\n");
    
    printf("  Creating 1D tensor: shape=[5]\n");
    int shape[] = {5};
    tiny_tensor_t* t = tiny_tensor_create(shape, 1, TINY_AI_DTYPE_FLOAT32);
    TEST_ASSERT(t != NULL, "Failed to create tensor");
    
    // Test requires_grad
    printf("  Enabling gradients (requires_grad = true)...\n");
    tiny_error_t err = tiny_tensor_requires_grad(t, true);
    TEST_ASSERT(err == TINY_OK, "Failed to set requires_grad");
    bool requires = tiny_tensor_get_requires_grad(t);
    printf("    - requires_grad: %s (expected: true)\n", requires ? "true" : "false");
    TEST_ASSERT(requires == true, "requires_grad not set correctly");
    
    // Test gradient buffer exists
    void* grad = tiny_tensor_grad(t);
    TEST_ASSERT(grad != NULL, "Gradient buffer should exist");
    printf("    - Gradient buffer allocated successfully\n");
    
    // Test zero_grad
    printf("  Setting gradient[0] = 1.0, then zeroing...\n");
    float* grad_data = (float*)grad;
    grad_data[0] = 1.0f;  // Set some gradient
    printf("    - Before zero: grad[0] = %.2f\n", grad_data[0]);
    err = tiny_tensor_zero_grad(t);
    TEST_ASSERT(err == TINY_OK, "Failed to zero gradients");
    printf("    - After zero: grad[0] = %.2f (expected: 0.00)\n", grad_data[0]);
    TEST_ASSERT(grad_data[0] == 0.0f, "Zero grad failed");
    
    tiny_tensor_destroy(t);
    
    printf("  ✓ Gradient tests passed\n");
    return TINY_OK;
}
#endif

tiny_error_t tiny_tensor_test_all(void)
{
    printf("========== Tensor Tests ==========\n\n");
    
    tiny_error_t err;
    
    err = test_create_destroy();
    if (err != TINY_OK) return err;
    
    err = test_data_access();
    if (err != TINY_OK) return err;
    
    err = test_copy();
    if (err != TINY_OK) return err;
    
    err = test_fill_zero();
    if (err != TINY_OK) return err;
    
#if TINY_AI_ENABLE_GRADIENTS
    err = test_gradients();
    if (err != TINY_OK) return err;
#endif
    
    printf("\n========== All Tensor Tests Passed ==========\n\n");
    return TINY_OK;
}

