# TINY_MAT 测试

## 测试代码

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

#ifndef __TINY_MAT_TEST__
#define __TINY_MAT_TEST__

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

#endif /* __TINY_MAT_TEST__ */
```

### tiny_mat_test.c

```cpp
/**
 * @file tiny_mat_test.c
 * @brief This file implements test functions for the submodule mat of the tiny_math middleware.
 */

#include "tiny_mat_test.h"

#define ROWS 2
#define COLS 3
#define PAD 1
#define FULL_WIDTH (COLS + PAD)

void tiny_mat_test(void)
{
    float A[ROWS * FULL_WIDTH] = {
        1, 2, 3, 0,
        4, 5, 6, 0};

    float B[COLS * FULL_WIDTH] = {
        7, 8, 9, 0,
        10, 11, 12, 0,
        13, 14, 15, 0};

    float A_compact[ROWS * COLS] = {0};
    float B_compact[COLS * COLS] = {0};
    float C_compact[ROWS * COLS] = {0};

    float output_general[ROWS * FULL_WIDTH] = {0};
    float constant = 2.0f;

    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++)
            A_compact[i * COLS + j] = A[i * FULL_WIDTH + j];

    for (int i = 0; i < COLS; i++)
        for (int j = 0; j < COLS; j++)
            B_compact[i * COLS + j] = B[i * FULL_WIDTH + j];

    printf("============ [tiny_mat_test - Full Coverage] ============\n\r");
    printf("[TEST SETTINGS] ROWS=%d, COLS=%d, PAD=%d, FULL_WIDTH=%d\n\r", ROWS, COLS, PAD, FULL_WIDTH);

    print_matrix_padded("A (padded) | memory layout", A, ROWS, FULL_WIDTH, FULL_WIDTH);
    print_matrix_padded("B (padded) | memory layout", B, COLS, FULL_WIDTH, FULL_WIDTH);
    print_matrix("A_compact", A_compact, ROWS, COLS);
    print_matrix("B_compact", B_compact, COLS, COLS);
    printf("constant = %f\n\r\n\r", constant);

    TinyTimeMark_t t0, t1;
    tiny_error_t err;

    t0 = tiny_get_running_time();
    err = tiny_mat_add_f32(A, B, output_general, ROWS, COLS, PAD, PAD, PAD, 1, 1, 1);
    t1 = tiny_get_running_time();
    printf("--- tiny_mat_add_f32(rows=%d, cols=%d, paddings=1) | Time: %6.2f us | Error: %d ---\n\r", ROWS, COLS, (double)(t1 - t0), err);
    print_matrix_padded("Output (padded)", output_general, ROWS, COLS, FULL_WIDTH);

    t0 = tiny_get_running_time();
    err = tiny_mat_addc_f32(A, output_general, constant, ROWS, COLS, PAD, PAD, 1, 1);
    t1 = tiny_get_running_time();
    printf("--- tiny_mat_addc_f32(C=%f) | Time: %6.2f us | Error: %d ---\n\r", constant, (double)(t1 - t0), err);
    print_matrix_padded("Output (padded)", output_general, ROWS, COLS, FULL_WIDTH);

    t0 = tiny_get_running_time();
    err = tiny_mat_sub_f32(A, B, output_general, ROWS, COLS, PAD, PAD, PAD, 1, 1, 1);
    t1 = tiny_get_running_time();
    printf("--- tiny_mat_sub_f32(rows=%d, cols=%d) | Time: %6.2f us | Error: %d ---\n\r", ROWS, COLS, (double)(t1 - t0), err);
    print_matrix_padded("Output (padded)", output_general, ROWS, COLS, FULL_WIDTH);

    t0 = tiny_get_running_time();
    err = tiny_mat_subc_f32(A, output_general, constant, ROWS, COLS, PAD, PAD, 1, 1);
    t1 = tiny_get_running_time();
    printf("--- tiny_mat_subc_f32(C=%f) | Time: %6.2f us | Error: %d ---\n\r", constant, (double)(t1 - t0), err);
    print_matrix_padded("Output (padded)", output_general, ROWS, COLS, FULL_WIDTH);

    t0 = tiny_get_running_time();
    err = tiny_mat_multc_f32(A, output_general, constant, ROWS, COLS, PAD, PAD, 1, 1);
    t1 = tiny_get_running_time();
    printf("--- tiny_mat_multc_f32(C=%f) | Time: %6.2f us | Error: %d ---\n\r", constant, (double)(t1 - t0), err);
    print_matrix_padded("Output (padded)", output_general, ROWS, COLS, FULL_WIDTH);

    t0 = tiny_get_running_time();
    err = tiny_mat_mult_f32(A_compact, B_compact, C_compact, ROWS, COLS, COLS);
    t1 = tiny_get_running_time();
    printf("--- tiny_mat_mult_f32 (compact) | Time: %6.2f us | Error: %d ---\n\r", (double)(t1 - t0), err);
    print_matrix("Output (compact)", C_compact, ROWS, COLS);

    t0 = tiny_get_running_time();
    err = tiny_mat_mult_ex_f32(A, B, output_general, ROWS, COLS, COLS, PAD, PAD, PAD);
    t1 = tiny_get_running_time();
    printf("--- tiny_mat_mult_ex_f32(padded) | Time: %6.2f us | Error: %d ---\n\r", (double)(t1 - t0), err);
    print_matrix_padded("Output (padded)", output_general, ROWS, COLS, FULL_WIDTH);

    printf("============ [test complete] ============\n\r");
}

```

## 测试结果

```bash
============ [tiny_mat_test - Full Coverage] ============
[TEST SETTINGS] ROWS=2, COLS=3, PAD=1, FULL_WIDTH=4
A (padded) | memory layout =
  1.000000   2.000000   3.000000   0.000000 
  4.000000   5.000000   6.000000   0.000000 

B (padded) | memory layout =
  7.000000   8.000000   9.000000   0.000000 
 10.000000  11.000000  12.000000   0.000000 
 13.000000  14.000000  15.000000   0.000000 

A_compact =
  1.000000   2.000000   3.000000 
  2.000000   3.000000   4.000000 

B_compact =
  7.000000   8.000000   9.000000 
  8.000000   9.000000  10.000000 
  9.000000  10.000000  11.000000 

constant = 2.000000

--- tiny_mat_add_f32(rows=2, cols=3, paddings=1) | Time:  18.00 us | Error: 0 ---
Output (padded) =
  8.000000  10.000000  12.000000 
 14.000000  16.000000  18.000000 

--- tiny_mat_addc_f32(C=2.000000) | Time:  10.00 us | Error: 0 ---
Output (padded) =
  3.000000   4.000000   5.000000 
  6.000000   7.000000   8.000000 

--- tiny_mat_sub_f32(rows=2, cols=3) | Time:  13.00 us | Error: 0 ---
Output (padded) =
 -6.000000  -6.000000  -6.000000 
 -6.000000  -6.000000  -6.000000 

--- tiny_mat_subc_f32(C=2.000000) | Time:   7.00 us | Error: 0 ---
Output (padded) =
 -1.000000   0.000000   1.000000 
  2.000000   3.000000   4.000000 

--- tiny_mat_multc_f32(C=2.000000) | Time:   8.00 us | Error: 0 ---
Output (padded) =
  2.000000   4.000000   6.000000 
  8.000000  10.000000  12.000000 

--- tiny_mat_mult_f32 (compact) | Time:  10.00 us | Error: 0 ---
Output (compact) =
 66.000000  72.000000  78.000000 
 72.000000  78.000000 156.000000 

--- tiny_mat_mult_ex_f32(padded) | Time:  13.00 us | Error: 0 ---
Output (padded) =
 66.000000  72.000000  78.000000 
156.000000 171.000000 186.000000 

============ [test complete] ============
```
