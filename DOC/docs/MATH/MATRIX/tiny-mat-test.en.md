# TINY_MAT TEST

## TEST CODE

### tiny_mat_test.h

```cpp
/**
 * @file tiny_mat_test.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the header file for the test of the submodule mat (basic matrix operations) of the tiny_math middleware.
 * @version 1.0
 * @date 2025-04-17
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
#include "tiny_math_config.h"
#include "tiny_mat.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tiny_mat_test(void);

#ifdef __cplusplus
}
#endif

```

### tiny_mat_test.c

```cpp
/**
 * @file tiny_mat_test.c
 * @brief Comprehensive stress tests for tiny_mat module, targeting edge cases and potential weaknesses.
 * @note Tests include: step parameters, different paddings, extreme values, boundary cases, and complex memory layouts.
 */

#include "tiny_mat_test.h"
#include <stdio.h>
#include <string.h>

/**
 * @brief Test tiny_mat_add_f32 with pad=0 and step=1 (contiguous memory layout)
 * 
 * Test Scenario:
 *   - This test case uses contiguous memory layout (no padding, step=1)
 *   - On ESP32 platform, this should trigger ESP-DSP optimized implementation
 *   - On other platforms, uses the standard implementation
 * 
 * Memory Layout:
 *   - Input1: 3x4 matrix stored contiguously in memory
 *     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
 *     Logical view:
 *       1.0   2.0   3.0   4.0
 *       5.0   6.0   7.0   8.0
 *       9.0  10.0  11.0  12.0
 * 
 *   - Input2: 3x4 matrix stored contiguously in memory
 *     [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
 *     Logical view:
 *       0.5   1.5   2.5   3.5
 *       4.5   5.5   6.5   7.5
 *       8.5   9.5  10.5  11.5
 * 
 * Expected Output:
 *   - Output: 3x4 matrix, each element = input1[i][j] + input2[i][j]
 *     Expected logical view:
 *       1.5   3.5   5.5   7.5
 *       9.5  11.5  13.5  15.5
 *      17.5  19.5  21.5  23.5
 * 
 * Parameters:
 *   - rows = 3, cols = 4
 *   - padd1 = 0, padd2 = 0, padd_out = 0 (no padding)
 *   - step1 = 1, step2 = 1, step_out = 1 (contiguous access)
 */
void test_tiny_mat_add_f32_contiguous(void)
{
    printf("\n");
    printf("================================================================================\n\r");
    printf("Test Case 1: tiny_mat_add_f32 - Contiguous Memory Layout (pad=0, step=1)\n\r");
    printf("================================================================================\n\r");
    printf("Parameters: rows=3, cols=4, pad=0, step=1\n\r");
    printf("\n\r");
    
    const int rows = 3;
    const int cols = 4;
    
    // Input matrices (contiguous, no padding)
    float input1[12] = {1.0f, 2.0f, 3.0f, 4.0f,
                        5.0f, 6.0f, 7.0f, 8.0f,
                        9.0f, 10.0f, 11.0f, 12.0f};
    
    float input2[12] = {0.5f, 1.5f, 2.5f, 3.5f,
                        4.5f, 5.5f, 6.5f, 7.5f,
                        8.5f, 9.5f, 10.5f, 11.5f};
    
    float output[12];
    memset(output, 0, sizeof(output));
    
    printf("Input1 Memory Layout (12 elements, contiguous):\n\r");
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 12; i++) {
        printf("%5.1f ", input1[i]);
    }
    printf("\n\r");
    printf("  Matrix: [1.0  2.0  3.0  4.0]  <- Row 0\n\r");
    printf("          [5.0  6.0  7.0  8.0]  <- Row 1\n\r");
    printf("          [9.0 10.0 11.0 12.0]  <- Row 2\n\r");
    printf("\n\r");
    
    printf("Input2 Memory Layout (12 elements, contiguous):\n\r");
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 12; i++) {
        printf("%5.1f ", input2[i]);
    }
    printf("\n\r");
    printf("  Matrix: [0.5  1.5  2.5  3.5]  <- Row 0\n\r");
    printf("          [4.5  5.5  6.5  7.5]  <- Row 1\n\r");
    printf("          [8.5  9.5 10.5 11.5]  <- Row 2\n\r");
    printf("\n\r");
    
    printf("Expected Output Memory Layout (12 elements, contiguous):\n\r");
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 12; i++) {
        printf("%5.1f ", input1[i] + input2[i]);
    }
    printf("\n\r");
    printf("  Matrix: [1.5  3.5  5.5  7.5]  <- Row 0\n\r");
    printf("          [9.5 11.5 13.5 15.5]  <- Row 1\n\r");
    printf("          [17.5 19.5 21.5 23.5] <- Row 2\n\r");
    printf("\n\r");
    
    // Test with pad=0, step=1 (should use ESP-DSP on ESP32)
    tiny_error_t err = tiny_mat_add_f32(input1, input2, output, rows, cols, 
                                        0, 0, 0,  // padd1=0, padd2=0, padd_out=0
                                        1, 1, 1); // step1=1, step2=1, step_out=1
    
    if (err == TINY_OK) {
        printf("Output Memory Layout (12 elements, contiguous):\n\r");
        printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]\n\r");
        printf("  Value:  ");
        for (int i = 0; i < 12; i++) {
            printf("%5.1f ", output[i]);
        }
        printf("\n\r");
        printf("  Matrix: [1.5  3.5  5.5  7.5]  <- Row 0\n\r");
        printf("          [9.5 11.5 13.5 15.5]  <- Row 1\n\r");
        printf("          [17.5 19.5 21.5 23.5] <- Row 2\n\r");
        printf("\n\r");
        
        // Verify results
        int all_correct = 1;
        for (int i = 0; i < rows * cols; i++) {
            float expected = input1[i] + input2[i];
            float tolerance = 1e-6f;
            float diff = (output[i] > expected) ? (output[i] - expected) : (expected - output[i]);
            if (diff > tolerance) {
                all_correct = 0;
                break;
            }
        }
        
        if (all_correct) {
            printf("✓ Test PASSED\n\r");
        } else {
            printf("✗ Test FAILED\n\r");
        }
    } else {
        printf("✗ Test FAILED: Error code = %d\n\r", err);
    }
    
    printf("================================================================================\n\r\n\r");
}

/**
 * @brief Test tiny_mat_add_f32 with pad!=0 and step>1 (non-contiguous memory layout)
 * 
 * ================================================================================
 * Test Scenario:
 * ================================================================================
 * This test case demonstrates how to handle matrix addition with non-contiguous
 * memory layout. In real applications, matrix data may not be stored contiguously:
 *   1. Padding: Extra space at the end of each row
 *   2. Stride/Step: Gaps between elements
 * 
 * For example, a 2x3 logical matrix with padding=2 and step=2 may have memory layout:
 *   Logical matrix:        Memory array (first 10 elements):
 *   [1.0  2.0  3.0]  [1.0, 0, 2.0, 0, 3.0, 0, 0, 0, 0, 0, ...]
 *   [4.0  5.0  6.0]  [4.0, 0, 5.0, 0, 6.0, 0, 0, 0, 0, 0, ...]
 * 
 * ================================================================================
 * Index Calculation Formula:
 * ================================================================================
 * For matrix element [row][col], the memory index is calculated as:
 *   index = row * (cols + padding) + col * step
 * 
 * Where:
 *   - row: Row number (starting from 0)
 *   - col: Column number (starting from 0)
 *   - cols: Number of columns in the matrix
 *   - padding: Number of padding elements at the end of each row
 *   - step: Stride between columns (element spacing)
 * 
 * ================================================================================
 * Input1 Details (2x3 matrix, padding=2, step=2):
 * ================================================================================
 * Logical matrix view:
 *   [1.0  2.0  3.0]
 *   [4.0  5.0  6.0]
 * 
 * Memory array (input1[20]) actual content:
 *   Index:  0    1    2    3    4    5    6    7    8    9   10   11   12   ...
 *   Value:  [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...]
 *           |---- row 0 ----|  |-- padding --|  |---- row 1 ----|  |-- padding --|
 * 
 * Index calculation process:
 *   [0][0]: index = 0*(3+2) + 0*2 = 0*5 + 0 = 0  → input1[0] = 1.0
 *   [0][1]: index = 0*(3+2) + 1*2 = 0*5 + 2 = 2  → input1[2] = 2.0
 *   [0][2]: index = 0*(3+2) + 2*2 = 0*5 + 4 = 4  → input1[4] = 3.0
 *   [1][0]: index = 1*(3+2) + 0*2 = 1*5 + 0 = 5  → input1[5] = 4.0
 *   [1][1]: index = 1*(3+2) + 1*2 = 1*5 + 2 = 7  → input1[7] = 5.0
 *   [1][2]: index = 1*(3+2) + 2*2 = 1*5 + 4 = 9  → input1[9] = 6.0
 * 
 * Memory layout visualization:
 *   Position: [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] ...
 *   Value:     1.0  0.0  2.0  0.0  3.0  0.0  0.0  0.0  0.0  6.0   0.0   0.0   0.0 ...
 *   Label:     R0C0      R0C1      R0C2  pad  pad  pad  pad  R1C2   pad   pad   pad ...
 * 
 * ================================================================================
 * Input2 Details (2x3 matrix, padding=1, step=3):
 * ================================================================================
 * Logical matrix view:
 *   [0.5  1.5  2.5]
 *   [3.5  4.5  5.5]
 * 
 * Memory array (input2[16]) actual content:
 *   Index:  0    1    2    3    4    5    6    7    8    9   10   11   ...
 *   Value:  [0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 5.5, 0.0, ...]
 *           |---- row 0 ----|  |-- padding --|  |---- row 1 ----|  |-- padding --|
 * 
 * Index calculation process:
 *   [0][0]: index = 0*(3+1) + 0*3 = 0*4 + 0 = 0  → input2[0] = 0.5
 *   [0][1]: index = 0*(3+1) + 1*3 = 0*4 + 3 = 3  → input2[3] = 1.5
 *   [0][2]: index = 0*(3+1) + 2*3 = 0*4 + 6 = 6  → input2[6] = 2.5
 *   [1][0]: index = 1*(3+1) + 0*3 = 1*4 + 0 = 4  → input2[4] = 3.5
 *   [1][1]: index = 1*(3+1) + 1*3 = 1*4 + 3 = 7  → input2[7] = 4.5
 *   [1][2]: index = 1*(3+1) + 2*3 = 1*4 + 6 = 10 → input2[10] = 5.5
 * 
 * Memory layout visualization:
 *   Position: [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] ...
 *   Value:     0.5  0.0  0.0  1.5  3.5  0.0  2.5  4.5  0.0  0.0  5.5   0.0 ...
 *   Label:     R0C0      R0C1      R1C0      R0C2  R1C1      R1C2   pad ...
 * 
 * ================================================================================
 * Expected Output (2x3 matrix, padding=2, step=2):
 * ================================================================================
 * Logical matrix view:
 *   [1.5  3.5  5.5]
 *   [7.5  9.5 11.5]
 * 
 * Calculation process:
 *   output[0][0] = input1[0][0] + input2[0][0] = 1.0 + 0.5 = 1.5
 *   output[0][1] = input1[0][1] + input2[0][1] = 2.0 + 1.5 = 3.5
 *   output[0][2] = input1[0][2] + input2[0][2] = 3.0 + 2.5 = 5.5
 *   output[1][0] = input1[1][0] + input2[1][0] = 4.0 + 3.5 = 7.5
 *   output[1][1] = input1[1][1] + input2[1][1] = 5.0 + 4.5 = 9.5
 *   output[1][2] = input1[1][2] + input2[1][2] = 6.0 + 5.5 = 11.5
 * 
 * Output memory array (output[20]) expected content:
 *   Index:  0    1    2    3    4    5    6    7    8    9   10   11   12   ...
 *   Value:  [1.5, 0.0, 3.5, 0.0, 5.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...]
 *           |---- row 0 ----|  |-- padding --|  |---- row 1 ----|  |-- padding --|
 * 
 * ================================================================================
 * Test Parameters:
 * ================================================================================
 *   rows = 2, cols = 3
 *   padd1 = 2, padd2 = 1, padd_out = 2 (with padding)
 *   step1 = 2, step2 = 3, step_out = 2 (non-contiguous access)
 */
void test_tiny_mat_add_f32_padded_strided(void)
{
    printf("\n");
    printf("================================================================================\n\r");
    printf("Test Case 2: tiny_mat_add_f32 - Non-Contiguous Memory Layout (pad!=0, step>1)\n\r");
    printf("================================================================================\n\r");
    
    const int rows = 2;
    const int cols = 3;
    const int padd1 = 2;
    const int padd2 = 1;
    const int padd_out = 2;
    const int step1 = 2;
    const int step2 = 3;
    const int step_out = 2;
    
    printf("Parameters: rows=%d, cols=%d, pad1=%d, pad2=%d, pad_out=%d, step1=%d, step2=%d, step_out=%d\n\r",
           rows, cols, padd1, padd2, padd_out, step1, step2, step_out);
    printf("Index formula: index = row * (cols + padding) + col * step\n\r");
    printf("\n\r");
    
    // Input1: 2 rows, 3 cols, padding=2, step=2
    // Row stride = cols + padd1 = 3 + 2 = 5
    // Memory index calculation: base = row * (cols + padd1), offset = col * step1
    float input1[20] = {0}; // Allocate enough space, initialize to zero
    // Row 0: elements at indices 0, 2, 4
    input1[0 * 5 + 0 * 2] = 1.0f; // row 0, col 0: index = 0*5 + 0*2 = 0
    input1[0 * 5 + 1 * 2] = 2.0f; // row 0, col 1: index = 0*5 + 1*2 = 2
    input1[0 * 5 + 2 * 2] = 3.0f; // row 0, col 2: index = 0*5 + 2*2 = 4
    // Row 1: elements at indices 5, 7, 9
    input1[1 * 5 + 0 * 2] = 4.0f; // row 1, col 0: index = 1*5 + 0*2 = 5
    input1[1 * 5 + 1 * 2] = 5.0f; // row 1, col 1: index = 1*5 + 1*2 = 7
    input1[1 * 5 + 2 * 2] = 6.0f; // row 1, col 2: index = 1*5 + 2*2 = 9
    
    // Input2: 2 rows, 3 cols, padding=1, step=3
    // Row stride = cols + padd2 = 3 + 1 = 4
    // Memory index calculation: base = row * (cols + padd2), offset = col * step2
    float input2[16] = {0}; // Allocate enough space, initialize to zero
    // Row 0: elements at indices 0, 3, 6
    input2[0 * 4 + 0 * 3] = 0.5f; // row 0, col 0: index = 0*4 + 0*3 = 0
    input2[0 * 4 + 1 * 3] = 1.5f; // row 0, col 1: index = 0*4 + 1*3 = 3
    input2[0 * 4 + 2 * 3] = 2.5f; // row 0, col 2: index = 0*4 + 2*3 = 6
    // Row 1: elements at indices 4, 7, 10
    input2[1 * 4 + 0 * 3] = 3.5f; // row 1, col 0: index = 1*4 + 0*3 = 4
    input2[1 * 4 + 1 * 3] = 4.5f; // row 1, col 1: index = 1*4 + 1*3 = 7
    input2[1 * 4 + 2 * 3] = 5.5f; // row 1, col 2: index = 1*4 + 2*3 = 10
    
    // Output: 2 rows, 3 cols, padding=2, step=2
    // Row stride = cols + padd_out = 3 + 2 = 5
    float output[20] = {0}; // Allocate enough space, initialize to zero
    
    printf("Input1 Memory Layout (20 elements, pad=%d, step=%d):\n\r", padd1, step1);
    printf("  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 15; i++) {
        printf("%4.1f ", input1[i]);
    }
    printf("...\n\r");
    printf("  Matrix: [1.0  X  2.0  X  3.0]  <- Row 0 (indices: 0, 2, 4)\n\r");
    printf("          [4.0  X  5.0  X  6.0]  <- Row 1 (indices: 5, 7, 9)\n\r");
    printf("          (X = padding/unused)\n\r");
    printf("\n\r");
    
    printf("Input2 Memory Layout (16 elements, pad=%d, step=%d):\n\r", padd2, step2);
    printf("  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] ...\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 12; i++) {
        printf("%4.1f ", input2[i]);
    }
    printf("...\n\r");
    printf("  Matrix: [0.5  X  X  1.5  X  X  2.5]  <- Row 0 (indices: 0, 3, 6)\n\r");
    printf("          [3.5  X  X  4.5  X  X  5.5]  <- Row 1 (indices: 4, 7, 10)\n\r");
    printf("          (X = padding/unused)\n\r");
    printf("\n\r");
    
    // Calculate expected output
    float expected_output[20] = {0};
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int idx1 = row * (cols + padd1) + col * step1;
            int idx2 = row * (cols + padd2) + col * step2;
            int idx_out = row * (cols + padd_out) + col * step_out;
            expected_output[idx_out] = input1[idx1] + input2[idx2];
        }
    }
    
    printf("Expected Output Memory Layout (20 elements, pad=%d, step=%d):\n\r", padd_out, step_out);
    printf("  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 15; i++) {
        printf("%4.1f ", expected_output[i]);
    }
    printf("...\n\r");
    printf("  Matrix: [1.5  X  3.5  X  5.5]  <- Row 0 (indices: 0, 2, 4)\n\r");
    printf("          [7.5  X  9.5  X 11.5]  <- Row 1 (indices: 5, 7, 9)\n\r");
    printf("          (X = padding/unused)\n\r");
    printf("\n\r");
    
    tiny_error_t err = tiny_mat_add_f32(input1, input2, output, rows, cols,
                                        padd1, padd2, padd_out,
                                        step1, step2, step_out);
    
    if (err == TINY_OK) {
        printf("Output Memory Layout (20 elements, pad=%d, step=%d):\n\r", padd_out, step_out);
        printf("  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...\n\r");
        printf("  Value:  ");
        for (int i = 0; i < 15; i++) {
            printf("%4.1f ", output[i]);
        }
        printf("...\n\r");
        printf("  Matrix: [1.5  X  3.5  X  5.5]  <- Row 0 (indices: 0, 2, 4)\n\r");
        printf("          [7.5  X  9.5  X 11.5]  <- Row 1 (indices: 5, 7, 9)\n\r");
        printf("          (X = padding/unused)\n\r");
        printf("\n\r");
        
        // Verify results
        int all_correct = 1;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                int idx_out = row * (cols + padd_out) + col * step_out;
                int idx1 = row * (cols + padd1) + col * step1;
                int idx2 = row * (cols + padd2) + col * step2;
                float expected = input1[idx1] + input2[idx2];
                float tolerance = 1e-6f;
                float diff = (output[idx_out] > expected) ? (output[idx_out] - expected) : (expected - output[idx_out]);
                if (diff > tolerance) {
                    all_correct = 0;
                    break;
                }
            }
            if (!all_correct) break;
        }
        
        if (all_correct) {
            printf("✓ Test PASSED\n\r");
        } else {
            printf("✗ Test FAILED\n\r");
        }
    } else {
        printf("✗ Test FAILED: Error code = %d\n\r", err);
    }
    
    printf("================================================================================\n\r\n\r");
}

/**
 * @brief Test tiny_mat_addc_f32 with pad=0 and step=1 (contiguous memory layout)
 */
void test_tiny_mat_addc_f32_contiguous(void)
{
    printf("\n");
    printf("================================================================================\n\r");
    printf("Test Case 3: tiny_mat_addc_f32 - Contiguous Memory Layout (pad=0, step=1)\n\r");
    printf("================================================================================\n\r");
    printf("Parameters: rows=3, cols=4, pad=0, step=1, C=2.5\n\r");
    printf("\n\r");
    
    const int rows = 3;
    const int cols = 4;
    const float C = 2.5f;
    
    // Input matrix (contiguous, no padding)
    float input[12] = {1.0f, 2.0f, 3.0f, 4.0f,
                       5.0f, 6.0f, 7.0f, 8.0f,
                       9.0f, 10.0f, 11.0f, 12.0f};
    
    float output[12];
    memset(output, 0, sizeof(output));
    
    printf("Input Memory Layout (12 elements, contiguous):\n\r");
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 12; i++) {
        printf("%5.1f ", input[i]);
    }
    printf("\n\r");
    printf("  Matrix: [1.0  2.0  3.0  4.0]  <- Row 0\n\r");
    printf("          [5.0  6.0  7.0  8.0]  <- Row 1\n\r");
    printf("          [9.0 10.0 11.0 12.0]  <- Row 2\n\r");
    printf("\n\r");
    
    printf("Constant C = %5.1f\n\r", C);
    printf("\n\r");
    
    printf("Expected Output Memory Layout (12 elements, contiguous):\n\r");
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 12; i++) {
        printf("%5.1f ", input[i] + C);
    }
    printf("\n\r");
    printf("  Matrix: [3.5  4.5  5.5  6.5]  <- Row 0\n\r");
    printf("          [7.5  8.5  9.5 10.5]  <- Row 1\n\r");
    printf("          [11.5 12.5 13.5 14.5] <- Row 2\n\r");
    printf("\n\r");
    
    // Test with pad=0, step=1 (should use ESP-DSP on ESP32)
    tiny_error_t err = tiny_mat_addc_f32(input, output, C, rows, cols,
                                         0, 0,  // padd_in=0, padd_out=0
                                         1, 1); // step_in=1, step_out=1
    
    if (err == TINY_OK) {
        printf("Output Memory Layout (12 elements, contiguous):\n\r");
        printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]\n\r");
        printf("  Value:  ");
        for (int i = 0; i < 12; i++) {
            printf("%5.1f ", output[i]);
        }
        printf("\n\r");
        printf("  Matrix: [3.5  4.5  5.5  6.5]  <- Row 0\n\r");
        printf("          [7.5  8.5  9.5 10.5]  <- Row 1\n\r");
        printf("          [11.5 12.5 13.5 14.5] <- Row 2\n\r");
        printf("\n\r");
        
        // Verify results
        int all_correct = 1;
        for (int i = 0; i < rows * cols; i++) {
            float expected = input[i] + C;
            float tolerance = 1e-6f;
            float diff = (output[i] > expected) ? (output[i] - expected) : (expected - output[i]);
            if (diff > tolerance) {
                all_correct = 0;
                break;
            }
        }
        
        if (all_correct) {
            printf("✓ Test PASSED\n\r");
        } else {
            printf("✗ Test FAILED\n\r");
        }
    } else {
        printf("✗ Test FAILED: Error code = %d\n\r", err);
    }
    
    printf("================================================================================\n\r\n\r");
}

/**
 * @brief Test tiny_mat_addc_f32 with pad!=0 and step>1 (non-contiguous memory layout)
 */
void test_tiny_mat_addc_f32_padded_strided(void)
{
    printf("\n");
    printf("================================================================================\n\r");
    printf("Test Case 4: tiny_mat_addc_f32 - Non-Contiguous Memory Layout (pad!=0, step>1)\n\r");
    printf("================================================================================\n\r");
    
    const int rows = 2;
    const int cols = 3;
    const int padd_in = 2;
    const int padd_out = 2;
    const int step_in = 2;
    const int step_out = 2;
    const float C = 1.5f;
    
    printf("Parameters: rows=%d, cols=%d, pad_in=%d, pad_out=%d, step_in=%d, step_out=%d, C=%5.1f\n\r",
           rows, cols, padd_in, padd_out, step_in, step_out, C);
    printf("Index formula: index = row * (cols + padding) + col * step\n\r");
    printf("\n\r");
    
    // Input: 2 rows, 3 cols, padding=2, step=2
    // Row stride = cols + padd_in = 3 + 2 = 5
    float input[20] = {0}; // Allocate enough space, initialize to zero
    // Row 0: elements at indices 0, 2, 4
    input[0 * 5 + 0 * 2] = 1.0f; // row 0, col 0: index = 0*5 + 0*2 = 0
    input[0 * 5 + 1 * 2] = 2.0f; // row 0, col 1: index = 0*5 + 1*2 = 2
    input[0 * 5 + 2 * 2] = 3.0f; // row 0, col 2: index = 0*5 + 2*2 = 4
    // Row 1: elements at indices 5, 7, 9
    input[1 * 5 + 0 * 2] = 4.0f; // row 1, col 0: index = 1*5 + 0*2 = 5
    input[1 * 5 + 1 * 2] = 5.0f; // row 1, col 1: index = 1*5 + 1*2 = 7
    input[1 * 5 + 2 * 2] = 6.0f; // row 1, col 2: index = 1*5 + 2*2 = 9
    
    // Output: 2 rows, 3 cols, padding=2, step=2
    // Row stride = cols + padd_out = 3 + 2 = 5
    float output[20] = {0}; // Allocate enough space, initialize to zero
    
    printf("Input Memory Layout (20 elements, pad=%d, step=%d):\n\r", padd_in, step_in);
    printf("  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 15; i++) {
        printf("%4.1f ", input[i]);
    }
    printf("...\n\r");
    printf("  Matrix: [1.0  X  2.0  X  3.0]  <- Row 0 (indices: 0, 2, 4)\n\r");
    printf("          [4.0  X  5.0  X  6.0]  <- Row 1 (indices: 5, 7, 9)\n\r");
    printf("          (X = padding/unused)\n\r");
    printf("\n\r");
    
    printf("Constant C = %5.1f\n\r", C);
    printf("\n\r");
    
    // Calculate expected output
    float expected_output[20] = {0};
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int idx_in = row * (cols + padd_in) + col * step_in;
            int idx_out = row * (cols + padd_out) + col * step_out;
            expected_output[idx_out] = input[idx_in] + C;
        }
    }
    
    printf("Expected Output Memory Layout (20 elements, pad=%d, step=%d):\n\r", padd_out, step_out);
    printf("  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 15; i++) {
        printf("%4.1f ", expected_output[i]);
    }
    printf("...\n\r");
    printf("  Matrix: [2.5  X  3.5  X  4.5]  <- Row 0 (indices: 0, 2, 4)\n\r");
    printf("          [5.5  X  6.5  X  7.5]  <- Row 1 (indices: 5, 7, 9)\n\r");
    printf("          (X = padding/unused)\n\r");
    printf("\n\r");
    
    tiny_error_t err = tiny_mat_addc_f32(input, output, C, rows, cols,
                                         padd_in, padd_out,
                                         step_in, step_out);
    
    if (err == TINY_OK) {
        printf("Output Memory Layout (20 elements, pad=%d, step=%d):\n\r", padd_out, step_out);
        printf("  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...\n\r");
        printf("  Value:  ");
        for (int i = 0; i < 15; i++) {
            printf("%4.1f ", output[i]);
        }
        printf("...\n\r");
        printf("  Matrix: [2.5  X  3.5  X  4.5]  <- Row 0 (indices: 0, 2, 4)\n\r");
        printf("          [5.5  X  6.5  X  7.5]  <- Row 1 (indices: 5, 7, 9)\n\r");
        printf("          (X = padding/unused)\n\r");
        printf("\n\r");
        
        // Verify results
        int all_correct = 1;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                int idx_in = row * (cols + padd_in) + col * step_in;
                int idx_out = row * (cols + padd_out) + col * step_out;
                float expected = input[idx_in] + C;
                float tolerance = 1e-6f;
                float diff = (output[idx_out] > expected) ? (output[idx_out] - expected) : (expected - output[idx_out]);
                if (diff > tolerance) {
                    all_correct = 0;
                    break;
                }
            }
            if (!all_correct) break;
        }
        
        if (all_correct) {
            printf("✓ Test PASSED\n\r");
        } else {
            printf("✗ Test FAILED\n\r");
        }
    } else {
        printf("✗ Test FAILED: Error code = %d\n\r", err);
    }
    
    printf("================================================================================\n\r\n\r");
}

/**
 * @brief Test tiny_mat_sub_f32 with pad=0 and step=1 (contiguous memory layout)
 */
void test_tiny_mat_sub_f32_contiguous(void)
{
    printf("\n");
    printf("================================================================================\n\r");
    printf("Test Case 5: tiny_mat_sub_f32 - Contiguous Memory Layout (pad=0, step=1)\n\r");
    printf("================================================================================\n\r");
    printf("Parameters: rows=3, cols=4, pad=0, step=1\n\r");
    printf("\n\r");
    
    const int rows = 3;
    const int cols = 4;
    
    // Input matrices (contiguous, no padding)
    float input1[12] = {1.0f, 2.0f, 3.0f, 4.0f,
                        5.0f, 6.0f, 7.0f, 8.0f,
                        9.0f, 10.0f, 11.0f, 12.0f};
    
    float input2[12] = {0.5f, 1.5f, 2.5f, 3.5f,
                        4.5f, 5.5f, 6.5f, 7.5f,
                        8.5f, 9.5f, 10.5f, 11.5f};
    
    float output[12];
    memset(output, 0, sizeof(output));
    
    printf("Input1 Memory Layout (12 elements, contiguous):\n\r");
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 12; i++) {
        printf("%5.1f ", input1[i]);
    }
    printf("\n\r");
    printf("  Matrix: [1.0  2.0  3.0  4.0]  <- Row 0\n\r");
    printf("          [5.0  6.0  7.0  8.0]  <- Row 1\n\r");
    printf("          [9.0 10.0 11.0 12.0]  <- Row 2\n\r");
    printf("\n\r");
    
    printf("Input2 Memory Layout (12 elements, contiguous):\n\r");
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 12; i++) {
        printf("%5.1f ", input2[i]);
    }
    printf("\n\r");
    printf("  Matrix: [0.5  1.5  2.5  3.5]  <- Row 0\n\r");
    printf("          [4.5  5.5  6.5  7.5]  <- Row 1\n\r");
    printf("          [8.5  9.5 10.5 11.5]  <- Row 2\n\r");
    printf("\n\r");
    
    printf("Expected Output Memory Layout (12 elements, contiguous):\n\r");
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 12; i++) {
        printf("%5.1f ", input1[i] - input2[i]);
    }
    printf("\n\r");
    printf("  Matrix: [0.5  0.5  0.5  0.5]  <- Row 0\n\r");
    printf("          [0.5  0.5  0.5  0.5]  <- Row 1\n\r");
    printf("          [0.5  0.5  0.5  0.5] <- Row 2\n\r");
    printf("\n\r");
    
    // Test with pad=0, step=1 (should use ESP-DSP on ESP32)
    tiny_error_t err = tiny_mat_sub_f32(input1, input2, output, rows, cols, 
                                        0, 0, 0,  // padd1=0, padd2=0, padd_out=0
                                        1, 1, 1); // step1=1, step2=1, step_out=1
    
    if (err == TINY_OK) {
        printf("Output Memory Layout (12 elements, contiguous):\n\r");
        printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]\n\r");
        printf("  Value:  ");
        for (int i = 0; i < 12; i++) {
            printf("%5.1f ", output[i]);
        }
        printf("\n\r");
        printf("  Matrix: [0.5  0.5  0.5  0.5]  <- Row 0\n\r");
        printf("          [0.5  0.5  0.5  0.5]  <- Row 1\n\r");
        printf("          [0.5  0.5  0.5  0.5] <- Row 2\n\r");
        printf("\n\r");
        
        // Verify results
        int all_correct = 1;
        for (int i = 0; i < rows * cols; i++) {
            float expected = input1[i] - input2[i];
            float tolerance = 1e-6f;
            float diff = (output[i] > expected) ? (output[i] - expected) : (expected - output[i]);
            if (diff > tolerance) {
                all_correct = 0;
                break;
            }
        }
        
        if (all_correct) {
            printf("✓ Test PASSED\n\r");
        } else {
            printf("✗ Test FAILED\n\r");
        }
    } else {
        printf("✗ Test FAILED: Error code = %d\n\r", err);
    }
    
    printf("================================================================================\n\r\n\r");
}

/**
 * @brief Test tiny_mat_sub_f32 with pad!=0 and step>1 (non-contiguous memory layout)
 */
void test_tiny_mat_sub_f32_padded_strided(void)
{
    printf("\n");
    printf("================================================================================\n\r");
    printf("Test Case 6: tiny_mat_sub_f32 - Non-Contiguous Memory Layout (pad!=0, step>1)\n\r");
    printf("================================================================================\n\r");
    
    const int rows = 2;
    const int cols = 3;
    const int padd1 = 2;
    const int padd2 = 1;
    const int padd_out = 2;
    const int step1 = 2;
    const int step2 = 3;
    const int step_out = 2;
    
    printf("Parameters: rows=%d, cols=%d, pad1=%d, pad2=%d, pad_out=%d, step1=%d, step2=%d, step_out=%d\n\r",
           rows, cols, padd1, padd2, padd_out, step1, step2, step_out);
    printf("Index formula: index = row * (cols + padding) + col * step\n\r");
    printf("\n\r");
    
    // Input1: 2 rows, 3 cols, padding=2, step=2
    // Row stride = cols + padd1 = 3 + 2 = 5
    float input1[20] = {0}; // Allocate enough space, initialize to zero
    // Row 0: elements at indices 0, 2, 4
    input1[0 * 5 + 0 * 2] = 1.0f; // row 0, col 0: index = 0*5 + 0*2 = 0
    input1[0 * 5 + 1 * 2] = 2.0f; // row 0, col 1: index = 0*5 + 1*2 = 2
    input1[0 * 5 + 2 * 2] = 3.0f; // row 0, col 2: index = 0*5 + 2*2 = 4
    // Row 1: elements at indices 5, 7, 9
    input1[1 * 5 + 0 * 2] = 4.0f; // row 1, col 0: index = 1*5 + 0*2 = 5
    input1[1 * 5 + 1 * 2] = 5.0f; // row 1, col 1: index = 1*5 + 1*2 = 7
    input1[1 * 5 + 2 * 2] = 6.0f; // row 1, col 2: index = 1*5 + 2*2 = 9
    
    // Input2: 2 rows, 3 cols, padding=1, step=3
    // Row stride = cols + padd2 = 3 + 1 = 4
    float input2[16] = {0}; // Allocate enough space, initialize to zero
    // Row 0: elements at indices 0, 3, 6
    input2[0 * 4 + 0 * 3] = 0.5f; // row 0, col 0: index = 0*4 + 0*3 = 0
    input2[0 * 4 + 1 * 3] = 1.5f; // row 0, col 1: index = 0*4 + 1*3 = 3
    input2[0 * 4 + 2 * 3] = 2.5f; // row 0, col 2: index = 0*4 + 2*3 = 6
    // Row 1: elements at indices 4, 7, 10
    input2[1 * 4 + 0 * 3] = 3.5f; // row 1, col 0: index = 1*4 + 0*3 = 4
    input2[1 * 4 + 1 * 3] = 4.5f; // row 1, col 1: index = 1*4 + 1*3 = 7
    input2[1 * 4 + 2 * 3] = 5.5f; // row 1, col 2: index = 1*4 + 2*3 = 10
    
    // Output: 2 rows, 3 cols, padding=2, step=2
    // Row stride = cols + padd_out = 3 + 2 = 5
    float output[20] = {0}; // Allocate enough space, initialize to zero
    
    printf("Input1 Memory Layout (20 elements, pad=%d, step=%d):\n\r", padd1, step1);
    printf("  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 15; i++) {
        printf("%4.1f ", input1[i]);
    }
    printf("...\n\r");
    printf("  Matrix: [1.0  X  2.0  X  3.0]  <- Row 0 (indices: 0, 2, 4)\n\r");
    printf("          [4.0  X  5.0  X  6.0]  <- Row 1 (indices: 5, 7, 9)\n\r");
    printf("          (X = padding/unused)\n\r");
    printf("\n\r");
    
    printf("Input2 Memory Layout (16 elements, pad=%d, step=%d):\n\r", padd2, step2);
    printf("  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] ...\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 12; i++) {
        printf("%4.1f ", input2[i]);
    }
    printf("...\n\r");
    printf("  Matrix: [0.5  X  X  1.5  X  X  2.5]  <- Row 0 (indices: 0, 3, 6)\n\r");
    printf("          [3.5  X  X  4.5  X  X  5.5]  <- Row 1 (indices: 4, 7, 10)\n\r");
    printf("          (X = padding/unused)\n\r");
    printf("\n\r");
    
    // Calculate expected output
    float expected_output[20] = {0};
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int idx1 = row * (cols + padd1) + col * step1;
            int idx2 = row * (cols + padd2) + col * step2;
            int idx_out = row * (cols + padd_out) + col * step_out;
            expected_output[idx_out] = input1[idx1] - input2[idx2];
        }
    }
    
    printf("Expected Output Memory Layout (20 elements, pad=%d, step=%d):\n\r", padd_out, step_out);
    printf("  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 15; i++) {
        printf("%4.1f ", expected_output[i]);
    }
    printf("...\n\r");
    printf("  Matrix: [0.5  X  0.5  X  0.5]  <- Row 0 (indices: 0, 2, 4)\n\r");
    printf("          [0.5  X  0.5  X  0.5]  <- Row 1 (indices: 5, 7, 9)\n\r");
    printf("          (X = padding/unused)\n\r");
    printf("\n\r");
    
    tiny_error_t err = tiny_mat_sub_f32(input1, input2, output, rows, cols,
                                        padd1, padd2, padd_out,
                                        step1, step2, step_out);
    
    if (err == TINY_OK) {
        printf("Output Memory Layout (20 elements, pad=%d, step=%d):\n\r", padd_out, step_out);
        printf("  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...\n\r");
        printf("  Value:  ");
        for (int i = 0; i < 15; i++) {
            printf("%4.1f ", output[i]);
        }
        printf("...\n\r");
        printf("  Matrix: [0.5  X  0.5  X  0.5]  <- Row 0 (indices: 0, 2, 4)\n\r");
        printf("          [0.5  X  0.5  X  0.5]  <- Row 1 (indices: 5, 7, 9)\n\r");
        printf("          (X = padding/unused)\n\r");
        printf("\n\r");
        
        // Verify results
        int all_correct = 1;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                int idx1 = row * (cols + padd1) + col * step1;
                int idx2 = row * (cols + padd2) + col * step2;
                int idx_out = row * (cols + padd_out) + col * step_out;
                float expected = input1[idx1] - input2[idx2];
                float tolerance = 1e-6f;
                float diff = (output[idx_out] > expected) ? (output[idx_out] - expected) : (expected - output[idx_out]);
                if (diff > tolerance) {
                    all_correct = 0;
                    break;
                }
            }
            if (!all_correct) break;
        }
        
        if (all_correct) {
            printf("✓ Test PASSED\n\r");
        } else {
            printf("✗ Test FAILED\n\r");
        }
    } else {
        printf("✗ Test FAILED: Error code = %d\n\r", err);
    }
    
    printf("================================================================================\n\r\n\r");
}

/**
 * @brief Test tiny_mat_subc_f32 with pad=0 and step=1 (contiguous memory layout)
 */
void test_tiny_mat_subc_f32_contiguous(void)
{
    printf("\n");
    printf("================================================================================\n\r");
    printf("Test Case 7: tiny_mat_subc_f32 - Contiguous Memory Layout (pad=0, step=1)\n\r");
    printf("================================================================================\n\r");
    printf("Parameters: rows=3, cols=4, pad=0, step=1, C=2.5\n\r");
    printf("\n\r");
    
    const int rows = 3;
    const int cols = 4;
    const float C = 2.5f;
    
    // Input matrix (contiguous, no padding)
    float input[12] = {1.0f, 2.0f, 3.0f, 4.0f,
                       5.0f, 6.0f, 7.0f, 8.0f,
                       9.0f, 10.0f, 11.0f, 12.0f};
    
    float output[12];
    memset(output, 0, sizeof(output));
    
    printf("Input Memory Layout (12 elements, contiguous):\n\r");
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 12; i++) {
        printf("%5.1f ", input[i]);
    }
    printf("\n\r");
    printf("  Matrix: [1.0  2.0  3.0  4.0]  <- Row 0\n\r");
    printf("          [5.0  6.0  7.0  8.0]  <- Row 1\n\r");
    printf("          [9.0 10.0 11.0 12.0]  <- Row 2\n\r");
    printf("\n\r");
    
    printf("Constant C = %5.1f\n\r", C);
    printf("\n\r");
    
    printf("Expected Output Memory Layout (12 elements, contiguous):\n\r");
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 12; i++) {
        printf("%5.1f ", input[i] - C);
    }
    printf("\n\r");
    printf("  Matrix: [-1.5 -0.5  0.5  1.5]  <- Row 0\n\r");
    printf("          [ 2.5  3.5  4.5  5.5]  <- Row 1\n\r");
    printf("          [ 6.5  7.5  8.5  9.5] <- Row 2\n\r");
    printf("\n\r");
    
    // Test with pad=0, step=1 (should use ESP-DSP on ESP32)
    tiny_error_t err = tiny_mat_subc_f32(input, output, C, rows, cols,
                                         0, 0,  // padd_in=0, padd_out=0
                                         1, 1); // step_in=1, step_out=1
    
    if (err == TINY_OK) {
        printf("Output Memory Layout (12 elements, contiguous):\n\r");
        printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]\n\r");
        printf("  Value:  ");
        for (int i = 0; i < 12; i++) {
            printf("%5.1f ", output[i]);
        }
        printf("\n\r");
        printf("  Matrix: [-1.5 -0.5  0.5  1.5]  <- Row 0\n\r");
        printf("          [ 2.5  3.5  4.5  5.5]  <- Row 1\n\r");
        printf("          [ 6.5  7.5  8.5  9.5] <- Row 2\n\r");
        printf("\n\r");
        
        // Verify results
        int all_correct = 1;
        for (int i = 0; i < rows * cols; i++) {
            float expected = input[i] - C;
            float tolerance = 1e-6f;
            float diff = (output[i] > expected) ? (output[i] - expected) : (expected - output[i]);
            if (diff > tolerance) {
                all_correct = 0;
                break;
            }
        }
        
        if (all_correct) {
            printf("✓ Test PASSED\n\r");
        } else {
            printf("✗ Test FAILED\n\r");
        }
    } else {
        printf("✗ Test FAILED: Error code = %d\n\r", err);
    }
    
    printf("================================================================================\n\r\n\r");
}

/**
 * @brief Test tiny_mat_subc_f32 with pad!=0 and step>1 (non-contiguous memory layout)
 */
void test_tiny_mat_subc_f32_padded_strided(void)
{
    printf("\n");
    printf("================================================================================\n\r");
    printf("Test Case 8: tiny_mat_subc_f32 - Non-Contiguous Memory Layout (pad!=0, step>1)\n\r");
    printf("================================================================================\n\r");
    
    const int rows = 2;
    const int cols = 3;
    const int padd_in = 2;
    const int padd_out = 2;
    const int step_in = 2;
    const int step_out = 2;
    const float C = 1.5f;
    
    printf("Parameters: rows=%d, cols=%d, pad_in=%d, pad_out=%d, step_in=%d, step_out=%d, C=%5.1f\n\r",
           rows, cols, padd_in, padd_out, step_in, step_out, C);
    printf("Index formula: index = row * (cols + padding) + col * step\n\r");
    printf("\n\r");
    
    // Input: 2 rows, 3 cols, padding=2, step=2
    // Row stride = cols + padd_in = 3 + 2 = 5
    float input[20] = {0}; // Allocate enough space, initialize to zero
    // Row 0: elements at indices 0, 2, 4
    input[0 * 5 + 0 * 2] = 1.0f; // row 0, col 0: index = 0*5 + 0*2 = 0
    input[0 * 5 + 1 * 2] = 2.0f; // row 0, col 1: index = 0*5 + 1*2 = 2
    input[0 * 5 + 2 * 2] = 3.0f; // row 0, col 2: index = 0*5 + 2*2 = 4
    // Row 1: elements at indices 5, 7, 9
    input[1 * 5 + 0 * 2] = 4.0f; // row 1, col 0: index = 1*5 + 0*2 = 5
    input[1 * 5 + 1 * 2] = 5.0f; // row 1, col 1: index = 1*5 + 1*2 = 7
    input[1 * 5 + 2 * 2] = 6.0f; // row 1, col 2: index = 1*5 + 2*2 = 9
    
    // Output: 2 rows, 3 cols, padding=2, step=2
    // Row stride = cols + padd_out = 3 + 2 = 5
    float output[20] = {0}; // Allocate enough space, initialize to zero
    
    printf("Input Memory Layout (20 elements, pad=%d, step=%d):\n\r", padd_in, step_in);
    printf("  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 15; i++) {
        printf("%4.1f ", input[i]);
    }
    printf("...\n\r");
    printf("  Matrix: [1.0  X  2.0  X  3.0]  <- Row 0 (indices: 0, 2, 4)\n\r");
    printf("          [4.0  X  5.0  X  6.0]  <- Row 1 (indices: 5, 7, 9)\n\r");
    printf("          (X = padding/unused)\n\r");
    printf("\n\r");
    
    printf("Constant C = %5.1f\n\r", C);
    printf("\n\r");
    
    // Calculate expected output
    float expected_output[20] = {0};
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int idx_in = row * (cols + padd_in) + col * step_in;
            int idx_out = row * (cols + padd_out) + col * step_out;
            expected_output[idx_out] = input[idx_in] - C;
        }
    }
    
    printf("Expected Output Memory Layout (20 elements, pad=%d, step=%d):\n\r", padd_out, step_out);
    printf("  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 15; i++) {
        printf("%4.1f ", expected_output[i]);
    }
    printf("...\n\r");
    printf("  Matrix: [-0.5  X  0.5  X  1.5]  <- Row 0 (indices: 0, 2, 4)\n\r");
    printf("          [ 2.5  X  3.5  X  4.5]  <- Row 1 (indices: 5, 7, 9)\n\r");
    printf("          (X = padding/unused)\n\r");
    printf("\n\r");
    
    tiny_error_t err = tiny_mat_subc_f32(input, output, C, rows, cols,
                                         padd_in, padd_out,
                                         step_in, step_out);
    
    if (err == TINY_OK) {
        printf("Output Memory Layout (20 elements, pad=%d, step=%d):\n\r", padd_out, step_out);
        printf("  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...\n\r");
        printf("  Value:  ");
        for (int i = 0; i < 15; i++) {
            printf("%4.1f ", output[i]);
        }
        printf("...\n\r");
        printf("  Matrix: [-0.5  X  0.5  X  1.5]  <- Row 0 (indices: 0, 2, 4)\n\r");
        printf("          [ 2.5  X  3.5  X  4.5]  <- Row 1 (indices: 5, 7, 9)\n\r");
        printf("          (X = padding/unused)\n\r");
        printf("\n\r");
        
        // Verify results
        int all_correct = 1;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                int idx_in = row * (cols + padd_in) + col * step_in;
                int idx_out = row * (cols + padd_out) + col * step_out;
                float expected = input[idx_in] - C;
                float tolerance = 1e-6f;
                float diff = (output[idx_out] > expected) ? (output[idx_out] - expected) : (expected - output[idx_out]);
                if (diff > tolerance) {
                    all_correct = 0;
                    break;
                }
            }
            if (!all_correct) break;
        }
        
        if (all_correct) {
            printf("✓ Test PASSED\n\r");
        } else {
            printf("✗ Test FAILED\n\r");
        }
    } else {
        printf("✗ Test FAILED: Error code = %d\n\r", err);
    }
    
    printf("================================================================================\n\r\n\r");
}

/**
 * @brief Test tiny_mat_mult_f32 with basic matrix multiplication
 */
void test_tiny_mat_mult_f32_basic(void)
{
    printf("\n");
    printf("================================================================================\n\r");
    printf("Test Case 9: tiny_mat_mult_f32 - Basic Matrix Multiplication\n\r");
    printf("================================================================================\n\r");
    printf("Parameters: m=3, n=4, k=2 (A is 3x4, B is 4x2, C is 3x2)\n\r");
    printf("Note: This function always uses ESP-DSP on ESP32, standard implementation otherwise\n\r");
    printf("\n\r");
    
    const int m = 3; // rows of A
    const int n = 4; // cols of A and rows of B
    const int k = 2; // cols of B
    
    // Matrix A: 3x4
    // Memory layout: [row0_col0, row0_col1, row0_col2, row0_col3, row1_col0, ...]
    float A[12] = {1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f,
                    9.0f, 10.0f, 11.0f, 12.0f};
    
    // Matrix B: 4x2
    float B[8] = {0.5f, 1.5f,
                  2.5f, 3.5f,
                  4.5f, 5.5f,
                  6.5f, 7.5f};
    
    // Matrix C: 3x2 (output)
    float C[6];
    memset(C, 0, sizeof(C));
    
    printf("Matrix A Memory Layout (3x4, 12 elements, contiguous):\n\r");
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 12; i++) {
        printf("%5.1f ", A[i]);
    }
    printf("\n\r");
    printf("  Matrix: [1.0  2.0  3.0  4.0]  <- Row 0\n\r");
    printf("          [5.0  6.0  7.0  8.0]  <- Row 1\n\r");
    printf("          [9.0 10.0 11.0 12.0]  <- Row 2\n\r");
    printf("\n\r");
    
    printf("Matrix B Memory Layout (4x2, 8 elements, contiguous):\n\r");
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 8; i++) {
        printf("%5.1f ", B[i]);
    }
    printf("\n\r");
    printf("  Matrix: [0.5  1.5]  <- Row 0\n\r");
    printf("          [2.5  3.5]  <- Row 1\n\r");
    printf("          [4.5  5.5]  <- Row 2\n\r");
    printf("          [6.5  7.5]  <- Row 3\n\r");
    printf("\n\r");
    
    // Calculate expected output: C = A * B
    // C[i][j] = sum_{s=0}^{n-1} A[i][s] * B[s][j]
    float expected_C[6] = {0};
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int s = 0; s < n; s++) {
                sum += A[i * n + s] * B[s * k + j];
            }
            expected_C[i * k + j] = sum;
        }
    }
    
    printf("Expected Output Matrix C Memory Layout (3x2, 6 elements, contiguous):\n\r");
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 6; i++) {
        printf("%6.1f ", expected_C[i]);
    }
    printf("\n\r");
    printf("  Matrix: [%5.1f  %5.1f]  <- Row 0\n\r", expected_C[0], expected_C[1]);
    printf("          [%5.1f  %5.1f]  <- Row 1\n\r", expected_C[2], expected_C[3]);
    printf("          [%5.1f  %5.1f] <- Row 2\n\r", expected_C[4], expected_C[5]);
    printf("  Calculation:\n\r");
    printf("    C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0] + A[0][3]*B[3][0]\n\r");
    printf("            = 1.0*0.5 + 2.0*2.5 + 3.0*4.5 + 4.0*6.5 = %5.1f\n\r", expected_C[0]);
    printf("    C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1] + A[0][3]*B[3][1]\n\r");
    printf("            = 1.0*1.5 + 2.0*3.5 + 3.0*5.5 + 4.0*7.5 = %5.1f\n\r", expected_C[1]);
    printf("\n\r");
    
    // Test matrix multiplication
    tiny_error_t err = tiny_mat_mult_f32(A, B, C, m, n, k);
    
    if (err == TINY_OK) {
        printf("Output Matrix C Memory Layout (3x2, 6 elements, contiguous):\n\r");
        printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]\n\r");
        printf("  Value:  ");
        for (int i = 0; i < 6; i++) {
            printf("%6.1f ", C[i]);
        }
        printf("\n\r");
        printf("  Matrix: [%5.1f  %5.1f]  <- Row 0\n\r", C[0], C[1]);
        printf("          [%5.1f  %5.1f]  <- Row 1\n\r", C[2], C[3]);
        printf("          [%5.1f  %5.1f] <- Row 2\n\r", C[4], C[5]);
        printf("\n\r");
        
        // Verify results
        int all_correct = 1;
        for (int i = 0; i < m * k; i++) {
            float tolerance = 1e-5f;
            float diff = (C[i] > expected_C[i]) ? (C[i] - expected_C[i]) : (expected_C[i] - C[i]);
            if (diff > tolerance) {
                int row = i / k;
                int col = i % k;
                printf("  ERROR at [%d][%d]: output = %10.6f, expected = %10.6f, diff = %e\n\r", 
                       row, col, C[i], expected_C[i], diff);
                all_correct = 0;
            }
        }
        
        if (all_correct) {
            printf("✓ Test PASSED\n\r");
        } else {
            printf("✗ Test FAILED\n\r");
        }
    } else {
        printf("✗ Test FAILED: Error code = %d\n\r", err);
    }
    
    printf("================================================================================\n\r\n\r");
}

/**
 * @brief Test tiny_mat_mult_f32 with square matrices
 */
void test_tiny_mat_mult_f32_square(void)
{
    printf("\n");
    printf("================================================================================\n\r");
    printf("Test Case 10: tiny_mat_mult_f32 - Square Matrix Multiplication\n\r");
    printf("================================================================================\n\r");
    printf("Parameters: m=3, n=3, k=3 (A is 3x3, B is 3x3, C is 3x3)\n\r");
    printf("Note: This function always uses ESP-DSP on ESP32, standard implementation otherwise\n\r");
    printf("\n\r");
    
    const int m = 3; // rows of A
    const int n = 3; // cols of A and rows of B
    const int k = 3; // cols of B
    
    // Matrix A: 3x3
    float A[9] = {1.0f, 2.0f, 3.0f,
                  4.0f, 5.0f, 6.0f,
                  7.0f, 8.0f, 9.0f};
    
    // Matrix B: 3x3
    float B[9] = {0.5f, 1.0f, 1.5f,
                  2.0f, 2.5f, 3.0f,
                  3.5f, 4.0f, 4.5f};
    
    // Matrix C: 3x3 (output)
    float C[9];
    memset(C, 0, sizeof(C));
    
    printf("Matrix A Memory Layout (3x3, 9 elements, contiguous):\n\r");
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 9; i++) {
        printf("%5.1f ", A[i]);
    }
    printf("\n\r");
    printf("  Matrix: [1.0  2.0  3.0]  <- Row 0\n\r");
    printf("          [4.0  5.0  6.0]  <- Row 1\n\r");
    printf("          [7.0  8.0  9.0]  <- Row 2\n\r");
    printf("\n\r");
    
    printf("Matrix B Memory Layout (3x3, 9 elements, contiguous):\n\r");
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 9; i++) {
        printf("%5.1f ", B[i]);
    }
    printf("\n\r");
    printf("  Matrix: [0.5  1.0  1.5]  <- Row 0\n\r");
    printf("          [2.0  2.5  3.0]  <- Row 1\n\r");
    printf("          [3.5  4.0  4.5]  <- Row 2\n\r");
    printf("\n\r");
    
    // Calculate expected output: C = A * B
    float expected_C[9] = {0};
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int s = 0; s < n; s++) {
                sum += A[i * n + s] * B[s * k + j];
            }
            expected_C[i * k + j] = sum;
        }
    }
    
    printf("Expected Output Matrix C Memory Layout (3x3, 9 elements, contiguous):\n\r");
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 9; i++) {
        printf("%6.1f ", expected_C[i]);
    }
    printf("\n\r");
    printf("  Matrix: [%5.1f  %5.1f  %5.1f]  <- Row 0\n\r", expected_C[0], expected_C[1], expected_C[2]);
    printf("          [%5.1f  %5.1f  %5.1f]  <- Row 1\n\r", expected_C[3], expected_C[4], expected_C[5]);
    printf("          [%5.1f  %5.1f  %5.1f] <- Row 2\n\r", expected_C[6], expected_C[7], expected_C[8]);
    printf("\n\r");
    
    // Test matrix multiplication
    tiny_error_t err = tiny_mat_mult_f32(A, B, C, m, n, k);
    
    if (err == TINY_OK) {
        printf("Output Matrix C Memory Layout (3x3, 9 elements, contiguous):\n\r");
        printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]\n\r");
        printf("  Value:  ");
        for (int i = 0; i < 9; i++) {
            printf("%6.1f ", C[i]);
        }
        printf("\n\r");
        printf("  Matrix: [%5.1f  %5.1f  %5.1f]  <- Row 0\n\r", C[0], C[1], C[2]);
        printf("          [%5.1f  %5.1f  %5.1f]  <- Row 1\n\r", C[3], C[4], C[5]);
        printf("          [%5.1f  %5.1f  %5.1f] <- Row 2\n\r", C[6], C[7], C[8]);
        printf("\n\r");
        
        // Verify results
        int all_correct = 1;
        for (int i = 0; i < m * k; i++) {
            float tolerance = 1e-5f;
            float diff = (C[i] > expected_C[i]) ? (C[i] - expected_C[i]) : (expected_C[i] - C[i]);
            if (diff > tolerance) {
                all_correct = 0;
                break;
            }
        }
        
        if (all_correct) {
            printf("✓ Test PASSED\n\r");
        } else {
            printf("✗ Test FAILED\n\r");
        }
    } else {
        printf("✗ Test FAILED: Error code = %d\n\r", err);
    }
    
    printf("================================================================================\n\r\n\r");
}

/**
 * @brief Test tiny_mat_mult_ex_f32 with contiguous matrices (pad=0)
 */
void test_tiny_mat_mult_ex_f32_contiguous(void)
{
    printf("\n");
    printf("================================================================================\n\r");
    printf("Test Case 11: tiny_mat_mult_ex_f32 - Contiguous Matrix Multiplication\n\r");
    printf("================================================================================\n\r");
    printf("Parameters: A_rows=3, A_cols=4, B_cols=2, A_padding=0, B_padding=0, C_padding=0\n\r");
    printf("Matrix dimensions: A is 3x4, B is 4x2, C is 3x2\n\r");
    printf("Note: This should use ESP-DSP on ESP32 when all paddings are 0\n\r");
    printf("\n\r");
    
    const int A_rows = 3;
    const int A_cols = 4;
    const int B_cols = 2;
    const int A_padding = 0;
    const int B_padding = 0;
    const int C_padding = 0;
    
    const int A_step = A_cols + A_padding; // 4
    const int B_step = B_cols + B_padding; // 2
    const int C_step = B_cols + C_padding; // 2
    
    // Matrix A: 3x4 (contiguous, no padding)
    float A[12] = {1.0f, 2.0f, 3.0f, 4.0f,
                   5.0f, 6.0f, 7.0f, 8.0f,
                   9.0f, 10.0f, 11.0f, 12.0f};
    
    // Matrix B: 4x2 (contiguous, no padding)
    float B[8] = {0.5f, 1.5f,
                  2.5f, 3.5f,
                  4.5f, 5.5f,
                  6.5f, 7.5f};
    
    // Matrix C: 3x2 (output, contiguous, no padding)
    float C[6];
    memset(C, 0, sizeof(C));
    
    printf("Matrix A Memory Layout (3x4, 12 elements, contiguous, pad=%d):\n\r", A_padding);
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 12; i++) {
        printf("%5.1f ", A[i]);
    }
    printf("\n\r");
    printf("  Matrix: [1.0  2.0  3.0  4.0]  <- Row 0 (indices: 0-3)\n\r");
    printf("          [5.0  6.0  7.0  8.0]  <- Row 1 (indices: 4-7)\n\r");
    printf("          [9.0 10.0 11.0 12.0]  <- Row 2 (indices: 8-11)\n\r");
    printf("  Step size: %d (A_cols + A_padding = %d + %d)\n\r", A_step, A_cols, A_padding);
    printf("\n\r");
    
    printf("Matrix B Memory Layout (4x2, 8 elements, contiguous, pad=%d):\n\r", B_padding);
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 8; i++) {
        printf("%5.1f ", B[i]);
    }
    printf("\n\r");
    printf("  Matrix: [0.5  1.5]  <- Row 0 (indices: 0-1)\n\r");
    printf("          [2.5  3.5]  <- Row 1 (indices: 2-3)\n\r");
    printf("          [4.5  5.5]  <- Row 2 (indices: 4-5)\n\r");
    printf("          [6.5  7.5]  <- Row 3 (indices: 6-7)\n\r");
    printf("  Step size: %d (B_cols + B_padding = %d + %d)\n\r", B_step, B_cols, B_padding);
    printf("\n\r");
    
    // Calculate expected output: C = A * B
    // C[i][j] = sum_{s=0}^{A_cols-1} A[i][s] * B[s][j]
    // Index calculation: A[i * A_step + s], B[s * B_step + j], C[i * C_step + j]
    float expected_C[6] = {0};
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            float sum = 0.0f;
            for (int s = 0; s < A_cols; s++) {
                int idx_A = i * A_step + s;
                int idx_B = s * B_step + j;
                sum += A[idx_A] * B[idx_B];
            }
            expected_C[i * C_step + j] = sum;
        }
    }
    
    printf("Expected Output Matrix C Memory Layout (3x2, 6 elements, contiguous, pad=%d):\n\r", C_padding);
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 6; i++) {
        printf("%6.1f ", expected_C[i]);
    }
    printf("\n\r");
    printf("  Matrix: [%5.1f  %5.1f]  <- Row 0 (indices: 0-1)\n\r", expected_C[0], expected_C[1]);
    printf("          [%5.1f  %5.1f]  <- Row 1 (indices: 2-3)\n\r", expected_C[2], expected_C[3]);
    printf("          [%5.1f  %5.1f] <- Row 2 (indices: 4-5)\n\r", expected_C[4], expected_C[5]);
    printf("  Step size: %d (B_cols + C_padding = %d + %d)\n\r", C_step, B_cols, C_padding);
    printf("  Calculation example:\n\r");
    printf("    C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0] + A[0][3]*B[3][0]\n\r");
    printf("            = 1.0*0.5 + 2.0*2.5 + 3.0*4.5 + 4.0*6.5 = %5.1f\n\r", expected_C[0]);
    printf("\n\r");
    
    // Test matrix multiplication
    tiny_error_t err = tiny_mat_mult_ex_f32(A, B, C, A_rows, A_cols, B_cols, A_padding, B_padding, C_padding);
    
    if (err == TINY_OK) {
        printf("Output Matrix C Memory Layout (3x2, 6 elements, contiguous, pad=%d):\n\r", C_padding);
        printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]\n\r");
        printf("  Value:  ");
        for (int i = 0; i < 6; i++) {
            printf("%6.1f ", C[i]);
        }
        printf("\n\r");
        printf("  Matrix: [%5.1f  %5.1f]  <- Row 0 (indices: 0-1)\n\r", C[0], C[1]);
        printf("          [%5.1f  %5.1f]  <- Row 1 (indices: 2-3)\n\r", C[2], C[3]);
        printf("          [%5.1f  %5.1f] <- Row 2 (indices: 4-5)\n\r", C[4], C[5]);
        printf("\n\r");
        
        // Verify results
        int all_correct = 1;
        for (int i = 0; i < A_rows; i++) {
            for (int j = 0; j < B_cols; j++) {
                int idx = i * C_step + j;
                float tolerance = 1e-5f;
                float diff = (C[idx] > expected_C[idx]) ? (C[idx] - expected_C[idx]) : (expected_C[idx] - C[idx]);
                if (diff > tolerance) {
                    printf("  ERROR at [%d][%d] (index %d): output = %10.6f, expected = %10.6f, diff = %e\n\r", 
                           i, j, idx, C[idx], expected_C[idx], diff);
                    all_correct = 0;
                }
            }
        }
        
        if (all_correct) {
            printf("✓ Test PASSED\n\r");
        } else {
            printf("✗ Test FAILED\n\r");
        }
    } else {
        printf("✗ Test FAILED: Error code = %d\n\r", err);
    }
    
    printf("================================================================================\n\r\n\r");
}

/**
 * @brief Test tiny_mat_mult_ex_f32 with padded matrices (pad!=0)
 */
void test_tiny_mat_mult_ex_f32_padded(void)
{
    printf("\n");
    printf("================================================================================\n\r");
    printf("Test Case 12: tiny_mat_mult_ex_f32 - Padded Matrix Multiplication\n\r");
    printf("================================================================================\n\r");
    printf("Parameters: A_rows=2, A_cols=3, B_cols=2, A_padding=2, B_padding=1, C_padding=1\n\r");
    printf("Matrix dimensions: A is 2x3, B is 3x2, C is 2x2\n\r");
    printf("Note: This should use own implementation when padding is non-zero\n\r");
    printf("\n\r");
    
    const int A_rows = 2;
    const int A_cols = 3;
    const int B_cols = 2;
    const int A_padding = 2;
    const int B_padding = 1;
    const int C_padding = 1;
    
    const int A_step = A_cols + A_padding; // 3 + 2 = 5
    const int B_step = B_cols + B_padding; // 2 + 1 = 3
    const int C_step = B_cols + C_padding; // 2 + 1 = 3
    
    // Matrix A: 2x3 with padding=2, so each row has 5 elements (3 data + 2 padding)
    // Total memory: 2 rows * 5 elements = 10 elements
    float A[10] = {1.0f, 2.0f, 3.0f, 0.0f, 0.0f,  // Row 0: [1.0, 2.0, 3.0, X, X]
                   4.0f, 5.0f, 6.0f, 0.0f, 0.0f}; // Row 1: [4.0, 5.0, 6.0, X, X]
    
    // Matrix B: 3x2 with padding=1, so each row has 3 elements (2 data + 1 padding)
    // Total memory: 3 rows * 3 elements = 9 elements
    float B[9] = {0.5f, 1.5f, 0.0f,  // Row 0: [0.5, 1.5, X]
                  2.5f, 3.5f, 0.0f,  // Row 1: [2.5, 3.5, X]
                  4.5f, 5.5f, 0.0f}; // Row 2: [4.5, 5.5, X]
    
    // Matrix C: 2x2 with padding=1, so each row has 3 elements (2 data + 1 padding)
    // Total memory: 2 rows * 3 elements = 6 elements
    float C[6];
    memset(C, 0, sizeof(C));
    
    printf("Matrix A Memory Layout (2x3, pad=%d, step=%d, 10 elements):\n\r", A_padding, A_step);
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 10; i++) {
        printf("%4.1f ", A[i]);
    }
    printf("\n\r");
    printf("  Matrix: [1.0  2.0  3.0  X   X]  <- Row 0 (indices: 0, 1, 2, 3, 4)\n\r");
    printf("          [4.0  5.0  6.0  X   X]  <- Row 1 (indices: 5, 6, 7, 8, 9)\n\r");
    printf("          (X = padding/unused)\n\r");
    printf("  Index calculation: A[i][j] = A[i * %d + j]\n\r", A_step);
    printf("    Row 0: indices 0, 1, 2 (data), 3, 4 (padding)\n\r");
    printf("    Row 1: indices 5, 6, 7 (data), 8, 9 (padding)\n\r");
    printf("\n\r");
    
    printf("Matrix B Memory Layout (3x2, pad=%d, step=%d, 9 elements):\n\r", B_padding, B_step);
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 9; i++) {
        printf("%4.1f ", B[i]);
    }
    printf("\n\r");
    printf("  Matrix: [0.5  1.5  X]  <- Row 0 (indices: 0, 1, 2)\n\r");
    printf("          [2.5  3.5  X]  <- Row 1 (indices: 3, 4, 5)\n\r");
    printf("          [4.5  5.5  X]  <- Row 2 (indices: 6, 7, 8)\n\r");
    printf("          (X = padding/unused)\n\r");
    printf("  Index calculation: B[i][j] = B[i * %d + j]\n\r", B_step);
    printf("    Row 0: indices 0, 1 (data), 2 (padding)\n\r");
    printf("    Row 1: indices 3, 4 (data), 5 (padding)\n\r");
    printf("    Row 2: indices 6, 7 (data), 8 (padding)\n\r");
    printf("\n\r");
    
    // Calculate expected output: C = A * B
    // C[i][j] = sum_{s=0}^{A_cols-1} A[i][s] * B[s][j]
    // Index calculation: A[i * A_step + s], B[s * B_step + j], C[i * C_step + j]
    float expected_C[6] = {0};
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            float sum = 0.0f;
            for (int s = 0; s < A_cols; s++) {
                int idx_A = i * A_step + s;
                int idx_B = s * B_step + j;
                sum += A[idx_A] * B[idx_B];
            }
            expected_C[i * C_step + j] = sum;
        }
    }
    
    printf("Expected Output Matrix C Memory Layout (2x2, pad=%d, step=%d, 6 elements):\n\r", C_padding, C_step);
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 6; i++) {
        printf("%6.1f ", expected_C[i]);
    }
    printf("\n\r");
    printf("  Matrix: [%5.1f  %5.1f  X]  <- Row 0 (indices: 0, 1, 2)\n\r", expected_C[0], expected_C[1]);
    printf("          [%5.1f  %5.1f  X]  <- Row 1 (indices: 3, 4, 5)\n\r", expected_C[3], expected_C[4]);
    printf("          (X = padding/unused)\n\r");
    printf("  Index calculation: C[i][j] = C[i * %d + j]\n\r", C_step);
    printf("  Calculation:\n\r");
    printf("    C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0]\n\r");
    printf("            = A[%d]*B[%d] + A[%d]*B[%d] + A[%d]*B[%d]\n\r", 
           0*A_step+0, 0*B_step+0, 0*A_step+1, 1*B_step+0, 0*A_step+2, 2*B_step+0);
    printf("            = 1.0*0.5 + 2.0*2.5 + 3.0*4.5 = %5.1f\n\r", expected_C[0]);
    printf("    C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1]\n\r");
    printf("            = 1.0*1.5 + 2.0*3.5 + 3.0*5.5 = %5.1f\n\r", expected_C[1]);
    printf("    C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0]\n\r");
    printf("            = 4.0*0.5 + 5.0*2.5 + 6.0*4.5 = %5.1f\n\r", expected_C[3]);
    printf("    C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1]\n\r");
    printf("            = 4.0*1.5 + 5.0*3.5 + 6.0*5.5 = %5.1f\n\r", expected_C[4]);
    printf("\n\r");
    
    // Test matrix multiplication
    tiny_error_t err = tiny_mat_mult_ex_f32(A, B, C, A_rows, A_cols, B_cols, A_padding, B_padding, C_padding);
    
    if (err == TINY_OK) {
        printf("Output Matrix C Memory Layout (2x2, pad=%d, step=%d, 6 elements):\n\r", C_padding, C_step);
        printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]\n\r");
        printf("  Value:  ");
        for (int i = 0; i < 6; i++) {
            printf("%6.1f ", C[i]);
        }
        printf("\n\r");
        printf("  Matrix: [%5.1f  %5.1f  X]  <- Row 0 (indices: 0, 1, 2)\n\r", C[0], C[1]);
        printf("          [%5.1f  %5.1f  X]  <- Row 1 (indices: 3, 4, 5)\n\r", C[3], C[4]);
        printf("\n\r");
        
        // Verify results
        int all_correct = 1;
        for (int i = 0; i < A_rows; i++) {
            for (int j = 0; j < B_cols; j++) {
                int idx = i * C_step + j;
                float tolerance = 1e-5f;
                float diff = (C[idx] > expected_C[idx]) ? (C[idx] - expected_C[idx]) : (expected_C[idx] - C[idx]);
                if (diff > tolerance) {
                    printf("  ERROR at [%d][%d] (index %d): output = %10.6f, expected = %10.6f, diff = %e\n\r", 
                           i, j, idx, C[idx], expected_C[idx], diff);
                    all_correct = 0;
                }
            }
        }
        
        if (all_correct) {
            printf("✓ Test PASSED\n\r");
        } else {
            printf("✗ Test FAILED\n\r");
        }
    } else {
        printf("✗ Test FAILED: Error code = %d\n\r", err);
    }
    
    printf("================================================================================\n\r\n\r");
}

/**
 * @brief Test tiny_mat_multc_f32 with contiguous matrix (pad=0, step=1)
 */
void test_tiny_mat_multc_f32_contiguous(void)
{
    printf("\n");
    printf("================================================================================\n\r");
    printf("Test Case 13: tiny_mat_multc_f32 - Contiguous Matrix Multiply Constant\n\r");
    printf("================================================================================\n\r");
    printf("Parameters: rows=3, cols=3, padd_in=0, padd_out=0, step_in=1, step_out=1\n\r");
    printf("Matrix dimensions: 3x3\n\r");
    printf("Constant C: 2.5\n\r");
    printf("Note: This should use ESP-DSP on ESP32 when all paddings are 0 and all steps are 1\n\r");
    printf("\n\r");
    
    const int rows = 3;
    const int cols = 3;
    const int padd_in = 0;
    const int padd_out = 0;
    const int step_in = 1;
    const int step_out = 1;
    const float C = 2.5f;
    
    const int in_row_stride = cols + padd_in;  // 3
    const int out_row_stride = cols + padd_out; // 3
    
    // Input matrix: 3x3 (contiguous, no padding)
    float input[9] = {1.0f, 2.0f, 3.0f,
                       4.0f, 5.0f, 6.0f,
                       7.0f, 8.0f, 9.0f};
    
    // Output matrix: 3x3 (contiguous, no padding)
    float output[9];
    memset(output, 0, sizeof(output));
    
    printf("Input Matrix Memory Layout (3x3, 9 elements, contiguous, pad=%d, step=%d):\n\r", padd_in, step_in);
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 9; i++) {
        printf("%5.1f ", input[i]);
    }
    printf("\n\r");
    printf("  Matrix: [1.0  2.0  3.0]  <- Row 0 (indices: 0, 1, 2)\n\r");
    printf("          [4.0  5.0  6.0]  <- Row 1 (indices: 3, 4, 5)\n\r");
    printf("          [7.0  8.0  9.0]  <- Row 2 (indices: 6, 7, 8)\n\r");
    printf("  Row stride: %d (cols + padd_in = %d + %d)\n\r", in_row_stride, cols, padd_in);
    printf("  Index calculation: input[i][j] = input[i * %d + j * %d]\n\r", in_row_stride, step_in);
    printf("\n\r");
    
    // Calculate expected output: output[i][j] = input[i][j] * C
    float expected_output[9] = {0};
    for (int row = 0; row < rows; row++) {
        int base_in = row * in_row_stride;
        int base_out = row * out_row_stride;
        for (int col = 0; col < cols; col++) {
            int idx_in = base_in + col * step_in;
            int idx_out = base_out + col * step_out;
            expected_output[idx_out] = input[idx_in] * C;
        }
    }
    
    printf("Expected Output Matrix Memory Layout (3x3, 9 elements, contiguous, pad=%d, step=%d):\n\r", padd_out, step_out);
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 9; i++) {
        printf("%6.1f ", expected_output[i]);
    }
    printf("\n\r");
    printf("  Matrix: [%5.1f  %5.1f  %5.1f]  <- Row 0 (indices: 0, 1, 2)\n\r", 
           expected_output[0], expected_output[1], expected_output[2]);
    printf("          [%5.1f  %5.1f  %5.1f]  <- Row 1 (indices: 3, 4, 5)\n\r", 
           expected_output[3], expected_output[4], expected_output[5]);
    printf("          [%5.1f  %5.1f  %5.1f] <- Row 2 (indices: 6, 7, 8)\n\r", 
           expected_output[6], expected_output[7], expected_output[8]);
    printf("  Row stride: %d (cols + padd_out = %d + %d)\n\r", out_row_stride, cols, padd_out);
    printf("  Index calculation: output[i][j] = output[i * %d + j * %d]\n\r", out_row_stride, step_out);
    printf("  Calculation: output[i][j] = input[i][j] * %.1f\n\r", C);
    printf("    Example: output[0][0] = input[0][0] * %.1f = 1.0 * %.1f = %.1f\n\r", C, C, expected_output[0]);
    printf("\n\r");
    
    // Test matrix multiply constant
    tiny_error_t err = tiny_mat_multc_f32(input, output, C, rows, cols, padd_in, padd_out, step_in, step_out);
    
    if (err == TINY_OK) {
        printf("Output Matrix Memory Layout (3x3, 9 elements, contiguous, pad=%d, step=%d):\n\r", padd_out, step_out);
        printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]\n\r");
        printf("  Value:  ");
        for (int i = 0; i < 9; i++) {
            printf("%6.1f ", output[i]);
        }
        printf("\n\r");
        printf("  Matrix: [%5.1f  %5.1f  %5.1f]  <- Row 0 (indices: 0, 1, 2)\n\r", 
               output[0], output[1], output[2]);
        printf("          [%5.1f  %5.1f  %5.1f]  <- Row 1 (indices: 3, 4, 5)\n\r", 
               output[3], output[4], output[5]);
        printf("          [%5.1f  %5.1f  %5.1f] <- Row 2 (indices: 6, 7, 8)\n\r", 
               output[6], output[7], output[8]);
        printf("\n\r");
        
        // Verify results
        int all_correct = 1;
        for (int row = 0; row < rows; row++) {
            int base_out = row * out_row_stride;
            for (int col = 0; col < cols; col++) {
                int idx_out = base_out + col * step_out;
                float tolerance = 1e-5f;
                float diff = (output[idx_out] > expected_output[idx_out]) ? 
                            (output[idx_out] - expected_output[idx_out]) : 
                            (expected_output[idx_out] - output[idx_out]);
                if (diff > tolerance) {
                    printf("  ERROR at [%d][%d] (index %d): output = %10.6f, expected = %10.6f, diff = %e\n\r", 
                           row, col, idx_out, output[idx_out], expected_output[idx_out], diff);
                    all_correct = 0;
                }
            }
        }
        
        if (all_correct) {
            printf("✓ Test PASSED\n\r");
        } else {
            printf("✗ Test FAILED\n\r");
        }
    } else {
        printf("✗ Test FAILED: Error code = %d\n\r", err);
    }
    
    printf("================================================================================\n\r\n\r");
}

/**
 * @brief Test tiny_mat_multc_f32 with padded and strided matrix (pad!=0, step>1)
 */
void test_tiny_mat_multc_f32_padded_strided(void)
{
    printf("\n");
    printf("================================================================================\n\r");
    printf("Test Case 14: tiny_mat_multc_f32 - Padded and Strided Matrix Multiply Constant\n\r");
    printf("================================================================================\n\r");
    printf("Parameters: rows=2, cols=3, padd_in=2, padd_out=1, step_in=2, step_out=1\n\r");
    printf("Matrix dimensions: 2x3\n\r");
    printf("Constant C: 3.0\n\r");
    printf("Note: This should use own implementation when padding is non-zero or step > 1\n\r");
    printf("\n\r");
    
    const int rows = 2;
    const int cols = 3;
    const int padd_in = 2;
    const int padd_out = 1;
    const int step_in = 2;
    const int step_out = 1;
    const float C = 3.0f;
    
    const int in_row_stride = cols + padd_in;  // 3 + 2 = 5
    const int out_row_stride = cols + padd_out; // 3 + 1 = 4
    
    // Input matrix: 2x3 with padding=2, step=2
    // Each row has 5 elements in memory, but we only use every 2nd element (step=2)
    // Total memory: 2 rows * 5 elements = 10 elements
    // Row 0: indices 0, 2, 4 (data), 1, 3 (unused)
    // Row 1: indices 5, 7, 9 (data), 6, 8 (unused)
    float input[10] = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f,  // Row 0: [1.0, X, 2.0, X, 3.0]
                       4.0f, 0.0f, 5.0f, 0.0f, 6.0f}; // Row 1: [4.0, X, 5.0, X, 6.0]
    
    // Output matrix: 2x3 with padding=1, step=1
    // Each row has 4 elements (3 data + 1 padding)
    // Total memory: 2 rows * 4 elements = 8 elements
    float output[8];
    memset(output, 0, sizeof(output));
    
    printf("Input Matrix Memory Layout (2x3, pad=%d, step=%d, 10 elements):\n\r", padd_in, step_in);
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 10; i++) {
        printf("%4.1f ", input[i]);
    }
    printf("\n\r");
    printf("  Matrix: [1.0  X  2.0  X  3.0]  <- Row 0 (data indices: 0, 2, 4)\n\r");
    printf("          [4.0  X  5.0  X  6.0]  <- Row 1 (data indices: 5, 7, 9)\n\r");
    printf("          (X = unused/padding)\n\r");
    printf("  Row stride: %d (cols + padd_in = %d + %d)\n\r", in_row_stride, cols, padd_in);
    printf("  Index calculation: input[i][j] = input[i * %d + j * %d]\n\r", in_row_stride, step_in);
    printf("    Row 0: input[0][0]=input[%d]=%.1f, input[0][1]=input[%d]=%.1f, input[0][2]=input[%d]=%.1f\n\r",
           0*in_row_stride+0*step_in, input[0*in_row_stride+0*step_in],
           0*in_row_stride+1*step_in, input[0*in_row_stride+1*step_in],
           0*in_row_stride+2*step_in, input[0*in_row_stride+2*step_in]);
    printf("    Row 1: input[1][0]=input[%d]=%.1f, input[1][1]=input[%d]=%.1f, input[1][2]=input[%d]=%.1f\n\r",
           1*in_row_stride+0*step_in, input[1*in_row_stride+0*step_in],
           1*in_row_stride+1*step_in, input[1*in_row_stride+1*step_in],
           1*in_row_stride+2*step_in, input[1*in_row_stride+2*step_in]);
    printf("\n\r");
    
    // Calculate expected output: output[i][j] = input[i][j] * C
    float expected_output[8] = {0};
    for (int row = 0; row < rows; row++) {
        int base_in = row * in_row_stride;
        int base_out = row * out_row_stride;
        for (int col = 0; col < cols; col++) {
            int idx_in = base_in + col * step_in;
            int idx_out = base_out + col * step_out;
            expected_output[idx_out] = input[idx_in] * C;
        }
    }
    
    printf("Expected Output Matrix Memory Layout (2x3, pad=%d, step=%d, 8 elements):\n\r", padd_out, step_out);
    printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]\n\r");
    printf("  Value:  ");
    for (int i = 0; i < 8; i++) {
        printf("%6.1f ", expected_output[i]);
    }
    printf("\n\r");
    printf("  Matrix: [%5.1f  %5.1f  %5.1f  X]  <- Row 0 (indices: 0, 1, 2, 3)\n\r", 
           expected_output[0], expected_output[1], expected_output[2]);
    printf("          [%5.1f  %5.1f  %5.1f  X]  <- Row 1 (indices: 4, 5, 6, 7)\n\r", 
           expected_output[4], expected_output[5], expected_output[6]);
    printf("          (X = padding/unused)\n\r");
    printf("  Row stride: %d (cols + padd_out = %d + %d)\n\r", out_row_stride, cols, padd_out);
    printf("  Index calculation: output[i][j] = output[i * %d + j * %d]\n\r", out_row_stride, step_out);
    printf("  Calculation: output[i][j] = input[i][j] * %.1f\n\r", C);
    printf("    Row 0: output[0][0] = input[0][0] * %.1f = %.1f * %.1f = %.1f (index %d)\n\r",
           C, input[0*in_row_stride+0*step_in], C, expected_output[0*out_row_stride+0*step_out], 0*out_row_stride+0*step_out);
    printf("           output[0][1] = input[0][1] * %.1f = %.1f * %.1f = %.1f (index %d)\n\r",
           C, input[0*in_row_stride+1*step_in], C, expected_output[0*out_row_stride+1*step_out], 0*out_row_stride+1*step_out);
    printf("           output[0][2] = input[0][2] * %.1f = %.1f * %.1f = %.1f (index %d)\n\r",
           C, input[0*in_row_stride+2*step_in], C, expected_output[0*out_row_stride+2*step_out], 0*out_row_stride+2*step_out);
    printf("\n\r");
    
    // Test matrix multiply constant
    tiny_error_t err = tiny_mat_multc_f32(input, output, C, rows, cols, padd_in, padd_out, step_in, step_out);
    
    if (err == TINY_OK) {
        printf("Output Matrix Memory Layout (2x3, pad=%d, step=%d, 8 elements):\n\r", padd_out, step_out);
        printf("  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]\n\r");
        printf("  Value:  ");
        for (int i = 0; i < 8; i++) {
            printf("%6.1f ", output[i]);
        }
        printf("\n\r");
        printf("  Matrix: [%5.1f  %5.1f  %5.1f  X]  <- Row 0 (indices: 0, 1, 2, 3)\n\r", 
               output[0], output[1], output[2]);
        printf("          [%5.1f  %5.1f  %5.1f  X]  <- Row 1 (indices: 4, 5, 6, 7)\n\r", 
               output[4], output[5], output[6]);
        printf("\n\r");
        
        // Verify results
        int all_correct = 1;
        for (int row = 0; row < rows; row++) {
            int base_out = row * out_row_stride;
            for (int col = 0; col < cols; col++) {
                int idx_out = base_out + col * step_out;
                float tolerance = 1e-5f;
                float diff = (output[idx_out] > expected_output[idx_out]) ? 
                            (output[idx_out] - expected_output[idx_out]) : 
                            (expected_output[idx_out] - output[idx_out]);
                if (diff > tolerance) {
                    printf("  ERROR at [%d][%d] (index %d): output = %10.6f, expected = %10.6f, diff = %e\n\r", 
                           row, col, idx_out, output[idx_out], expected_output[idx_out], diff);
                    all_correct = 0;
                }
            }
        }
        
        if (all_correct) {
            printf("✓ Test PASSED\n\r");
        } else {
            printf("✗ Test FAILED\n\r");
        }
    } else {
        printf("✗ Test FAILED: Error code = %d\n\r", err);
    }
    
    printf("================================================================================\n\r\n\r");
}

void tiny_mat_test(void)
{
    printf("============ [tiny_mat_test] ============\n\r");
    
    // Test 1: Contiguous matrices (pad=0, step=1) - should use ESP-DSP on ESP32
    test_tiny_mat_add_f32_contiguous();
    
    // Test 2: Padded and strided matrices (pad!=0, step>1) - should use own implementation
    test_tiny_mat_add_f32_padded_strided();
    
    // Test 3: Contiguous matrix add constant (pad=0, step=1) - should use ESP-DSP on ESP32
    test_tiny_mat_addc_f32_contiguous();
    
    // Test 4: Padded and strided matrix add constant (pad!=0, step>1) - should use own implementation
    test_tiny_mat_addc_f32_padded_strided();
    
    // Test 5: Contiguous matrices subtraction (pad=0, step=1) - should use ESP-DSP on ESP32
    test_tiny_mat_sub_f32_contiguous();
    
    // Test 6: Padded and strided matrices subtraction (pad!=0, step>1) - should use own implementation
    test_tiny_mat_sub_f32_padded_strided();
    
    // Test 7: Contiguous matrix subtract constant (pad=0, step=1) - should use ESP-DSP on ESP32
    test_tiny_mat_subc_f32_contiguous();
    
    // Test 8: Padded and strided matrix subtract constant (pad!=0, step>1) - should use own implementation
    test_tiny_mat_subc_f32_padded_strided();
    
    // Test 9: Basic matrix multiplication (3x4 * 4x2 = 3x2)
    test_tiny_mat_mult_f32_basic();
    
    // Test 10: Square matrix multiplication (3x3 * 3x3 = 3x3)
    test_tiny_mat_mult_f32_square();
    
    // Test 11: Contiguous matrix multiplication with padding (pad=0) - should use ESP-DSP on ESP32
    test_tiny_mat_mult_ex_f32_contiguous();
    
    // Test 12: Padded matrix multiplication (pad!=0) - should use own implementation
    test_tiny_mat_mult_ex_f32_padded();
    
    // Test 13: Contiguous matrix multiply constant (pad=0, step=1) - should use ESP-DSP on ESP32
    test_tiny_mat_multc_f32_contiguous();
    
    // Test 14: Padded and strided matrix multiply constant (pad!=0, step>1) - should use own implementation
    test_tiny_mat_multc_f32_padded_strided();
    
    printf("============ [test complete] ============\n\r");
}

```

## TEST RESULTS

```bash
============ [tiny_mat_test] ============

================================================================================
Test Case 1: tiny_mat_add_f32 - Contiguous Memory Layout (pad=0, step=1)
================================================================================
Parameters: rows=3, cols=4, pad=0, step=1

Input1 Memory Layout (12 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]
  Value:    1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0  11.0  12.0 
  Matrix: [1.0  2.0  3.0  4.0]  <- Row 0
          [5.0  6.0  7.0  8.0]  <- Row 1
          [9.0 10.0 11.0 12.0]  <- Row 2

Input2 Memory Layout (12 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]
  Value:    0.5   1.5   2.5   3.5   4.5   5.5   6.5   7.5   8.5   9.5  10.5  11.5 
  Matrix: [0.5  1.5  2.5  3.5]  <- Row 0
          [4.5  5.5  6.5  7.5]  <- Row 1
          [8.5  9.5 10.5 11.5]  <- Row 2

Expected Output Memory Layout (12 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]
  Value:    1.5   3.5   5.5   7.5   9.5  11.5  13.5  15.5  17.5  19.5  21.5  23.5 
  Matrix: [1.5  3.5  5.5  7.5]  <- Row 0
          [9.5 11.5 13.5 15.5]  <- Row 1
          [17.5 19.5 21.5 23.5] <- Row 2

Output Memory Layout (12 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]
  Value:    1.5   3.5   5.5   7.5   9.5  11.5  13.5  15.5  17.5  19.5  21.5  23.5 
  Matrix: [1.5  3.5  5.5  7.5]  <- Row 0
          [9.5 11.5 13.5 15.5]  <- Row 1
          [17.5 19.5 21.5 23.5] <- Row 2

✓ Test PASSED
================================================================================


================================================================================
Test Case 2: tiny_mat_add_f32 - Non-Contiguous Memory Layout (pad!=0, step>1)
================================================================================
Parameters: rows=2, cols=3, pad1=2, pad2=1, pad_out=2, step1=2, step2=3, step_out=2
Index formula: index = row * (cols + padding) + col * step

Input1 Memory Layout (20 elements, pad=2, step=2):
  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...
  Value:   1.0  0.0  2.0  0.0  3.0  4.0  0.0  5.0  0.0  6.0  0.0  0.0  0.0  0.0  0.0 ...
  Matrix: [1.0  X  2.0  X  3.0]  <- Row 0 (indices: 0, 2, 4)
          [4.0  X  5.0  X  6.0]  <- Row 1 (indices: 5, 7, 9)
          (X = padding/unused)

Input2 Memory Layout (16 elements, pad=1, step=3):
  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] ...
  Value:   0.5  0.0  0.0  1.5  3.5  0.0  2.5  4.5  0.0  0.0  5.5  0.0 ...
  Matrix: [0.5  X  X  1.5  X  X  2.5]  <- Row 0 (indices: 0, 3, 6)
          [3.5  X  X  4.5  X  X  5.5]  <- Row 1 (indices: 4, 7, 10)
          (X = padding/unused)

Expected Output Memory Layout (20 elements, pad=2, step=2):
  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...
  Value:   1.5  0.0  3.5  0.0  5.5  7.5  0.0  9.5  0.0 11.5  0.0  0.0  0.0  0.0  0.0 ...
  Matrix: [1.5  X  3.5  X  5.5]  <- Row 0 (indices: 0, 2, 4)
          [7.5  X  9.5  X 11.5]  <- Row 1 (indices: 5, 7, 9)
          (X = padding/unused)

Output Memory Layout (20 elements, pad=2, step=2):
  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...
  Value:   1.5  0.0  3.5  0.0  5.5  7.5  0.0  9.5  0.0 11.5  0.0  0.0  0.0  0.0  0.0 ...
  Matrix: [1.5  X  3.5  X  5.5]  <- Row 0 (indices: 0, 2, 4)
          [7.5  X  9.5  X 11.5]  <- Row 1 (indices: 5, 7, 9)
          (X = padding/unused)

✓ Test PASSED
================================================================================


================================================================================
Test Case 3: tiny_mat_addc_f32 - Contiguous Memory Layout (pad=0, step=1)
================================================================================
Parameters: rows=3, cols=4, pad=0, step=1, C=2.5

Input Memory Layout (12 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]
  Value:    1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0  11.0  12.0 
  Matrix: [1.0  2.0  3.0  4.0]  <- Row 0
          [5.0  6.0  7.0  8.0]  <- Row 1
          [9.0 10.0 11.0 12.0]  <- Row 2

Constant C =   2.5

Expected Output Memory Layout (12 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]
  Value:    3.5   4.5   5.5   6.5   7.5   8.5   9.5  10.5  11.5  12.5  13.5  14.5 
  Matrix: [3.5  4.5  5.5  6.5]  <- Row 0
          [7.5  8.5  9.5 10.5]  <- Row 1
          [11.5 12.5 13.5 14.5] <- Row 2

Output Memory Layout (12 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]
  Value:    3.5   4.5   5.5   6.5   7.5   8.5   9.5  10.5  11.5  12.5  13.5  14.5 
  Matrix: [3.5  4.5  5.5  6.5]  <- Row 0
          [7.5  8.5  9.5 10.5]  <- Row 1
          [11.5 12.5 13.5 14.5] <- Row 2

✓ Test PASSED
================================================================================


================================================================================
Test Case 4: tiny_mat_addc_f32 - Non-Contiguous Memory Layout (pad!=0, step>1)
================================================================================
Parameters: rows=2, cols=3, pad_in=2, pad_out=2, step_in=2, step_out=2, C=  1.5
Index formula: index = row * (cols + padding) + col * step

Input Memory Layout (20 elements, pad=2, step=2):
  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...
  Value:   1.0  0.0  2.0  0.0  3.0  4.0  0.0  5.0  0.0  6.0  0.0  0.0  0.0  0.0  0.0 ...
  Matrix: [1.0  X  2.0  X  3.0]  <- Row 0 (indices: 0, 2, 4)
          [4.0  X  5.0  X  6.0]  <- Row 1 (indices: 5, 7, 9)
          (X = padding/unused)

Constant C =   1.5

Expected Output Memory Layout (20 elements, pad=2, step=2):
  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...
  Value:   2.5  0.0  3.5  0.0  4.5  5.5  0.0  6.5  0.0  7.5  0.0  0.0  0.0  0.0  0.0 ...
  Matrix: [2.5  X  3.5  X  4.5]  <- Row 0 (indices: 0, 2, 4)
          [5.5  X  6.5  X  7.5]  <- Row 1 (indices: 5, 7, 9)
          (X = padding/unused)

Output Memory Layout (20 elements, pad=2, step=2):
  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...
  Value:   2.5  0.0  3.5  0.0  4.5  5.5  0.0  6.5  0.0  7.5  0.0  0.0  0.0  0.0  0.0 ...
  Matrix: [2.5  X  3.5  X  4.5]  <- Row 0 (indices: 0, 2, 4)
          [5.5  X  6.5  X  7.5]  <- Row 1 (indices: 5, 7, 9)
          (X = padding/unused)

✓ Test PASSED
================================================================================


================================================================================
Test Case 5: tiny_mat_sub_f32 - Contiguous Memory Layout (pad=0, step=1)
================================================================================
Parameters: rows=3, cols=4, pad=0, step=1

Input1 Memory Layout (12 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]
  Value:    1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0  11.0  12.0 
  Matrix: [1.0  2.0  3.0  4.0]  <- Row 0
          [5.0  6.0  7.0  8.0]  <- Row 1
          [9.0 10.0 11.0 12.0]  <- Row 2

Input2 Memory Layout (12 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]
  Value:    0.5   1.5   2.5   3.5   4.5   5.5   6.5   7.5   8.5   9.5  10.5  11.5 
  Matrix: [0.5  1.5  2.5  3.5]  <- Row 0
          [4.5  5.5  6.5  7.5]  <- Row 1
          [8.5  9.5 10.5 11.5]  <- Row 2

Expected Output Memory Layout (12 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]
  Value:    0.5   0.5   0.5   0.5   0.5   0.5   0.5   0.5   0.5   0.5   0.5   0.5 
  Matrix: [0.5  0.5  0.5  0.5]  <- Row 0
          [0.5  0.5  0.5  0.5]  <- Row 1
          [0.5  0.5  0.5  0.5] <- Row 2

Output Memory Layout (12 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]
  Value:    0.5   0.5   0.5   0.5   0.5   0.5   0.5   0.5   0.5   0.5   0.5   0.5 
  Matrix: [0.5  0.5  0.5  0.5]  <- Row 0
          [0.5  0.5  0.5  0.5]  <- Row 1
          [0.5  0.5  0.5  0.5] <- Row 2

✓ Test PASSED
================================================================================


================================================================================
Test Case 6: tiny_mat_sub_f32 - Non-Contiguous Memory Layout (pad!=0, step>1)
================================================================================
Parameters: rows=2, cols=3, pad1=2, pad2=1, pad_out=2, step1=2, step2=3, step_out=2
Index formula: index = row * (cols + padding) + col * step

Input1 Memory Layout (20 elements, pad=2, step=2):
  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...
  Value:   1.0  0.0  2.0  0.0  3.0  4.0  0.0  5.0  0.0  6.0  0.0  0.0  0.0  0.0  0.0 ...
  Matrix: [1.0  X  2.0  X  3.0]  <- Row 0 (indices: 0, 2, 4)
          [4.0  X  5.0  X  6.0]  <- Row 1 (indices: 5, 7, 9)
          (X = padding/unused)

Input2 Memory Layout (16 elements, pad=1, step=3):
  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] ...
  Value:   0.5  0.0  0.0  1.5  3.5  0.0  2.5  4.5  0.0  0.0  5.5  0.0 ...
  Matrix: [0.5  X  X  1.5  X  X  2.5]  <- Row 0 (indices: 0, 3, 6)
          [3.5  X  X  4.5  X  X  5.5]  <- Row 1 (indices: 4, 7, 10)
          (X = padding/unused)

Expected Output Memory Layout (20 elements, pad=2, step=2):
  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...
  Value:   0.5  0.0  0.5  0.0  0.5  0.5  0.0  0.5  0.0  0.5  0.0  0.0  0.0  0.0  0.0 ...
  Matrix: [0.5  X  0.5  X  0.5]  <- Row 0 (indices: 0, 2, 4)
          [0.5  X  0.5  X  0.5]  <- Row 1 (indices: 5, 7, 9)
          (X = padding/unused)

Output Memory Layout (20 elements, pad=2, step=2):
  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...
  Value:   0.5  0.0  0.5  0.0  0.5  0.5  0.0  0.5  0.0  0.5  0.0  0.0  0.0  0.0  0.0 ...
  Matrix: [0.5  X  0.5  X  0.5]  <- Row 0 (indices: 0, 2, 4)
          [0.5  X  0.5  X  0.5]  <- Row 1 (indices: 5, 7, 9)
          (X = padding/unused)

✓ Test PASSED
================================================================================


================================================================================
Test Case 7: tiny_mat_subc_f32 - Contiguous Memory Layout (pad=0, step=1)
================================================================================
Parameters: rows=3, cols=4, pad=0, step=1, C=2.5

Input Memory Layout (12 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]
  Value:    1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0  11.0  12.0 
  Matrix: [1.0  2.0  3.0  4.0]  <- Row 0
          [5.0  6.0  7.0  8.0]  <- Row 1
          [9.0 10.0 11.0 12.0]  <- Row 2

Constant C =   2.5

Expected Output Memory Layout (12 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]
  Value:   -1.5  -0.5   0.5   1.5   2.5   3.5   4.5   5.5   6.5   7.5   8.5   9.5 
  Matrix: [-1.5 -0.5  0.5  1.5]  <- Row 0
          [ 2.5  3.5  4.5  5.5]  <- Row 1
          [ 6.5  7.5  8.5  9.5] <- Row 2

Output Memory Layout (12 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]
  Value:   -1.5  -0.5   0.5   1.5   2.5   3.5   4.5   5.5   6.5   7.5   8.5   9.5 
  Matrix: [-1.5 -0.5  0.5  1.5]  <- Row 0
          [ 2.5  3.5  4.5  5.5]  <- Row 1
          [ 6.5  7.5  8.5  9.5] <- Row 2

✓ Test PASSED
================================================================================


================================================================================
Test Case 8: tiny_mat_subc_f32 - Non-Contiguous Memory Layout (pad!=0, step>1)
================================================================================
Parameters: rows=2, cols=3, pad_in=2, pad_out=2, step_in=2, step_out=2, C=  1.5
Index formula: index = row * (cols + padding) + col * step

Input Memory Layout (20 elements, pad=2, step=2):
  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...
  Value:   1.0  0.0  2.0  0.0  3.0  4.0  0.0  5.0  0.0  6.0  0.0  0.0  0.0  0.0  0.0 ...
  Matrix: [1.0  X  2.0  X  3.0]  <- Row 0 (indices: 0, 2, 4)
          [4.0  X  5.0  X  6.0]  <- Row 1 (indices: 5, 7, 9)
          (X = padding/unused)

Constant C =   1.5

Expected Output Memory Layout (20 elements, pad=2, step=2):
  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...
  Value:  -0.5  0.0  0.5  0.0  1.5  2.5  0.0  3.5  0.0  4.5  0.0  0.0  0.0  0.0  0.0 ...
  Matrix: [-0.5  X  0.5  X  1.5]  <- Row 0 (indices: 0, 2, 4)
          [ 2.5  X  3.5  X  4.5]  <- Row 1 (indices: 5, 7, 9)
          (X = padding/unused)

Output Memory Layout (20 elements, pad=2, step=2):
  Index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12] [13] [14] ...
  Value:  -0.5  0.0  0.5  0.0  1.5  2.5  0.0  3.5  0.0  4.5  0.0  0.0  0.0  0.0  0.0 ...
  Matrix: [-0.5  X  0.5  X  1.5]  <- Row 0 (indices: 0, 2, 4)
          [ 2.5  X  3.5  X  4.5]  <- Row 1 (indices: 5, 7, 9)
          (X = padding/unused)

✓ Test PASSED
================================================================================


================================================================================
Test Case 9: tiny_mat_mult_f32 - Basic Matrix Multiplication
================================================================================
Parameters: m=3, n=4, k=2 (A is 3x4, B is 4x2, C is 3x2)
Note: This function always uses ESP-DSP on ESP32, standard implementation otherwise

Matrix A Memory Layout (3x4, 12 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]
  Value:    1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0  11.0  12.0 
  Matrix: [1.0  2.0  3.0  4.0]  <- Row 0
          [5.0  6.0  7.0  8.0]  <- Row 1
          [9.0 10.0 11.0 12.0]  <- Row 2

Matrix B Memory Layout (4x2, 8 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]
  Value:    0.5   1.5   2.5   3.5   4.5   5.5   6.5   7.5 
  Matrix: [0.5  1.5]  <- Row 0
          [2.5  3.5]  <- Row 1
          [4.5  5.5]  <- Row 2
          [6.5  7.5]  <- Row 3

Expected Output Matrix C Memory Layout (3x2, 6 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]
  Value:    45.0   55.0  101.0  127.0  157.0  199.0 
  Matrix: [ 45.0   55.0]  <- Row 0
          [101.0  127.0]  <- Row 1
          [157.0  199.0] <- Row 2
  Calculation:
    C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0] + A[0][3]*B[3][0]
            = 1.0*0.5 + 2.0*2.5 + 3.0*4.5 + 4.0*6.5 =  45.0
    C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1] + A[0][3]*B[3][1]
            = 1.0*1.5 + 2.0*3.5 + 3.0*5.5 + 4.0*7.5 =  55.0

Output Matrix C Memory Layout (3x2, 6 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]
  Value:    45.0   55.0  101.0  127.0  157.0  199.0 
  Matrix: [ 45.0   55.0]  <- Row 0
          [101.0  127.0]  <- Row 1
          [157.0  199.0] <- Row 2

✓ Test PASSED
================================================================================


================================================================================
Test Case 10: tiny_mat_mult_f32 - Square Matrix Multiplication
================================================================================
Parameters: m=3, n=3, k=3 (A is 3x3, B is 3x3, C is 3x3)
Note: This function always uses ESP-DSP on ESP32, standard implementation otherwise

Matrix A Memory Layout (3x3, 9 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]
  Value:    1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0 
  Matrix: [1.0  2.0  3.0]  <- Row 0
          [4.0  5.0  6.0]  <- Row 1
          [7.0  8.0  9.0]  <- Row 2

Matrix B Memory Layout (3x3, 9 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]
  Value:    0.5   1.0   1.5   2.0   2.5   3.0   3.5   4.0   4.5 
  Matrix: [0.5  1.0  1.5]  <- Row 0
          [2.0  2.5  3.0]  <- Row 1
          [3.5  4.0  4.5]  <- Row 2

Expected Output Matrix C Memory Layout (3x3, 9 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]
  Value:    15.0   18.0   21.0   33.0   40.5   48.0   51.0   63.0   75.0 
  Matrix: [ 15.0   18.0   21.0]  <- Row 0
          [ 33.0   40.5   48.0]  <- Row 1
          [ 51.0   63.0   75.0] <- Row 2

Output Matrix C Memory Layout (3x3, 9 elements, contiguous):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]
  Value:    15.0   18.0   21.0   33.0   40.5   48.0   51.0   63.0   75.0 
  Matrix: [ 15.0   18.0   21.0]  <- Row 0
          [ 33.0   40.5   48.0]  <- Row 1
          [ 51.0   63.0   75.0] <- Row 2

✓ Test PASSED
================================================================================


================================================================================
Test Case 11: tiny_mat_mult_ex_f32 - Contiguous Matrix Multiplication
================================================================================
Parameters: A_rows=3, A_cols=4, B_cols=2, A_padding=0, B_padding=0, C_padding=0
Matrix dimensions: A is 3x4, B is 4x2, C is 3x2
Note: This should use ESP-DSP on ESP32 when all paddings are 0

Matrix A Memory Layout (3x4, 12 elements, contiguous, pad=0):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]
  Value:    1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0  11.0  12.0 
  Matrix: [1.0  2.0  3.0  4.0]  <- Row 0 (indices: 0-3)
          [5.0  6.0  7.0  8.0]  <- Row 1 (indices: 4-7)
          [9.0 10.0 11.0 12.0]  <- Row 2 (indices: 8-11)
  Step size: 4 (A_cols + A_padding = 4 + 0)

Matrix B Memory Layout (4x2, 8 elements, contiguous, pad=0):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]
  Value:    0.5   1.5   2.5   3.5   4.5   5.5   6.5   7.5 
  Matrix: [0.5  1.5]  <- Row 0 (indices: 0-1)
          [2.5  3.5]  <- Row 1 (indices: 2-3)
          [4.5  5.5]  <- Row 2 (indices: 4-5)
          [6.5  7.5]  <- Row 3 (indices: 6-7)
  Step size: 2 (B_cols + B_padding = 2 + 0)

Expected Output Matrix C Memory Layout (3x2, 6 elements, contiguous, pad=0):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]
  Value:    45.0   55.0  101.0  127.0  157.0  199.0 
  Matrix: [ 45.0   55.0]  <- Row 0 (indices: 0-1)
          [101.0  127.0]  <- Row 1 (indices: 2-3)
          [157.0  199.0] <- Row 2 (indices: 4-5)
  Step size: 2 (B_cols + C_padding = 2 + 0)
  Calculation example:
    C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0] + A[0][3]*B[3][0]
            = 1.0*0.5 + 2.0*2.5 + 3.0*4.5 + 4.0*6.5 =  45.0

Output Matrix C Memory Layout (3x2, 6 elements, contiguous, pad=0):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]
  Value:    45.0   55.0  101.0  127.0  157.0  199.0 
  Matrix: [ 45.0   55.0]  <- Row 0 (indices: 0-1)
          [101.0  127.0]  <- Row 1 (indices: 2-3)
          [157.0  199.0] <- Row 2 (indices: 4-5)

✓ Test PASSED
================================================================================


================================================================================
Test Case 12: tiny_mat_mult_ex_f32 - Padded Matrix Multiplication
================================================================================
Parameters: A_rows=2, A_cols=3, B_cols=2, A_padding=2, B_padding=1, C_padding=1
Matrix dimensions: A is 2x3, B is 3x2, C is 2x2
Note: This should use own implementation when padding is non-zero

Matrix A Memory Layout (2x3, pad=2, step=5, 10 elements):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]
  Value:   1.0  2.0  3.0  0.0  0.0  4.0  5.0  6.0  0.0  0.0 
  Matrix: [1.0  2.0  3.0  X   X]  <- Row 0 (indices: 0, 1, 2, 3, 4)
          [4.0  5.0  6.0  X   X]  <- Row 1 (indices: 5, 6, 7, 8, 9)
          (X = padding/unused)
  Index calculation: A[i][j] = A[i * 5 + j]
    Row 0: indices 0, 1, 2 (data), 3, 4 (padding)
    Row 1: indices 5, 6, 7 (data), 8, 9 (padding)

Matrix B Memory Layout (3x2, pad=1, step=3, 9 elements):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]
  Value:   0.5  1.5  0.0  2.5  3.5  0.0  4.5  5.5  0.0 
  Matrix: [0.5  1.5  X]  <- Row 0 (indices: 0, 1, 2)
          [2.5  3.5  X]  <- Row 1 (indices: 3, 4, 5)
          [4.5  5.5  X]  <- Row 2 (indices: 6, 7, 8)
          (X = padding/unused)
  Index calculation: B[i][j] = B[i * 3 + j]
    Row 0: indices 0, 1 (data), 2 (padding)
    Row 1: indices 3, 4 (data), 5 (padding)
    Row 2: indices 6, 7 (data), 8 (padding)

Expected Output Matrix C Memory Layout (2x2, pad=1, step=3, 6 elements):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]
  Value:    19.0   25.0    0.0   41.5   56.5    0.0 
  Matrix: [ 19.0   25.0  X]  <- Row 0 (indices: 0, 1, 2)
          [ 41.5   56.5  X]  <- Row 1 (indices: 3, 4, 5)
          (X = padding/unused)
  Index calculation: C[i][j] = C[i * 3 + j]
  Calculation:
    C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0]
            = A[0]*B[0] + A[1]*B[3] + A[2]*B[6]
            = 1.0*0.5 + 2.0*2.5 + 3.0*4.5 =  19.0
    C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1]
            = 1.0*1.5 + 2.0*3.5 + 3.0*5.5 =  25.0
    C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0]
            = 4.0*0.5 + 5.0*2.5 + 6.0*4.5 =  41.5
    C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1]
            = 4.0*1.5 + 5.0*3.5 + 6.0*5.5 =  56.5

Output Matrix C Memory Layout (2x2, pad=1, step=3, 6 elements):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]
  Value:    19.0   25.0    0.0   41.5   56.5    0.0 
  Matrix: [ 19.0   25.0  X]  <- Row 0 (indices: 0, 1, 2)
          [ 41.5   56.5  X]  <- Row 1 (indices: 3, 4, 5)

✓ Test PASSED
================================================================================


================================================================================
Test Case 13: tiny_mat_multc_f32 - Contiguous Matrix Multiply Constant
================================================================================
Parameters: rows=3, cols=3, padd_in=0, padd_out=0, step_in=1, step_out=1
Matrix dimensions: 3x3
Constant C: 2.5
Note: This should use ESP-DSP on ESP32 when all paddings are 0 and all steps are 1

Input Matrix Memory Layout (3x3, 9 elements, contiguous, pad=0, step=1):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]
  Value:    1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0 
  Matrix: [1.0  2.0  3.0]  <- Row 0 (indices: 0, 1, 2)
          [4.0  5.0  6.0]  <- Row 1 (indices: 3, 4, 5)
          [7.0  8.0  9.0]  <- Row 2 (indices: 6, 7, 8)
  Row stride: 3 (cols + padd_in = 3 + 0)
  Index calculation: input[i][j] = input[i * 3 + j * 1]

Expected Output Matrix Memory Layout (3x3, 9 elements, contiguous, pad=0, step=1):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]
  Value:     2.5    5.0    7.5   10.0   12.5   15.0   17.5   20.0   22.5 
  Matrix: [  2.5    5.0    7.5]  <- Row 0 (indices: 0, 1, 2)
          [ 10.0   12.5   15.0]  <- Row 1 (indices: 3, 4, 5)
          [ 17.5   20.0   22.5] <- Row 2 (indices: 6, 7, 8)
  Row stride: 3 (cols + padd_out = 3 + 0)
  Index calculation: output[i][j] = output[i * 3 + j * 1]
  Calculation: output[i][j] = input[i][j] * 2.5
    Example: output[0][0] = input[0][0] * 2.5 = 1.0 * 2.5 = 2.5

Output Matrix Memory Layout (3x3, 9 elements, contiguous, pad=0, step=1):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]
  Value:     2.5    5.0    7.5   10.0   12.5   15.0   17.5   20.0   22.5 
  Matrix: [  2.5    5.0    7.5]  <- Row 0 (indices: 0, 1, 2)
          [ 10.0   12.5   15.0]  <- Row 1 (indices: 3, 4, 5)
          [ 17.5   20.0   22.5] <- Row 2 (indices: 6, 7, 8)

✓ Test PASSED
================================================================================


================================================================================
Test Case 14: tiny_mat_multc_f32 - Padded and Strided Matrix Multiply Constant
================================================================================
Parameters: rows=2, cols=3, padd_in=2, padd_out=1, step_in=2, step_out=1
Matrix dimensions: 2x3
Constant C: 3.0
Note: This should use own implementation when padding is non-zero or step > 1

Input Matrix Memory Layout (2x3, pad=2, step=2, 10 elements):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]
  Value:   1.0  0.0  2.0  0.0  3.0  4.0  0.0  5.0  0.0  6.0 
  Matrix: [1.0  X  2.0  X  3.0]  <- Row 0 (data indices: 0, 2, 4)
          [4.0  X  5.0  X  6.0]  <- Row 1 (data indices: 5, 7, 9)
          (X = unused/padding)
  Row stride: 5 (cols + padd_in = 3 + 2)
  Index calculation: input[i][j] = input[i * 5 + j * 2]
    Row 0: input[0][0]=input[0]=1.0, input[0][1]=input[2]=2.0, input[0][2]=input[4]=3.0
    Row 1: input[1][0]=input[5]=4.0, input[1][1]=input[7]=5.0, input[1][2]=input[9]=6.0

Expected Output Matrix Memory Layout (2x3, pad=1, step=1, 8 elements):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]
  Value:     3.0    6.0    9.0    0.0   12.0   15.0   18.0    0.0 
  Matrix: [  3.0    6.0    9.0  X]  <- Row 0 (indices: 0, 1, 2, 3)
          [ 12.0   15.0   18.0  X]  <- Row 1 (indices: 4, 5, 6, 7)
          (X = padding/unused)
  Row stride: 4 (cols + padd_out = 3 + 1)
  Index calculation: output[i][j] = output[i * 4 + j * 1]
  Calculation: output[i][j] = input[i][j] * 3.0
    Row 0: output[0][0] = input[0][0] * 3.0 = 1.0 * 3.0 = 3.0 (index 0)
           output[0][1] = input[0][1] * 3.0 = 2.0 * 3.0 = 6.0 (index 1)
           output[0][2] = input[0][2] * 3.0 = 3.0 * 3.0 = 9.0 (index 2)

Output Matrix Memory Layout (2x3, pad=1, step=1, 8 elements):
  Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]
  Value:     3.0    6.0    9.0    0.0   12.0   15.0   18.0    0.0 
  Matrix: [  3.0    6.0    9.0  X]  <- Row 0 (indices: 0, 1, 2, 3)
          [ 12.0   15.0   18.0  X]  <- Row 1 (indices: 4, 5, 6, 7)

✓ Test PASSED
================================================================================

============ [test complete] ============

```