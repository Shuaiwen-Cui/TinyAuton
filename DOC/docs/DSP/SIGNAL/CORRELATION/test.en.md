# TESTS

## tiny_corr_test.h

```c

/**
 * @file tiny_corr_test.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_corr | test | header
 * @version 1.0
 * @date 2025-04-27
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
#include "tiny_corr.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tiny_signal_corr_ccorr_test(void);

#ifdef __cplusplus
}
#endif

```

## tiny_corr_test.c

```c
/**
 * @file tiny_corr_test.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_corr | test | source
 * @version 1.0
 * @date 2025-04-27
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_corr_test.h"

#define EPSILON 1e-3f // Tolerance for floating-point comparison

// Helper function to print float arrays
static void print_float_array(const char *label, const float *array, int length) {
    printf("%s: [", label);
    for (int i = 0; i < length; i++) {
        printf("%.2f", array[i]);
        if (i != length -1) {
            printf(", ");
        }
    }
    printf("]\n");
}

void tiny_signal_corr_ccorr_test(void)
{
    printf("\n========== Correlation & Cross-Correlation Test ==========\n");

    /*** Test 1: Correlation Function - tiny_corr_f32 ***/
    const float signal[]  = {1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 1.0f};
    const float pattern[] = {2.0f, 1.0f, 0.0f};
    const int siglen  = sizeof(signal) / sizeof(signal[0]);
    const int patlen  = sizeof(pattern) / sizeof(pattern[0]);
    const int corr_len = siglen - patlen + 1;

    float corr_output[corr_len];
    const float expected_corr[] = {4.0f, 7.0f, 10.0f, 10.0f};

    printf("\n--- Test 1: tiny_corr_f32 ---\n");
    print_float_array("Input Signal ", signal, siglen);
    print_float_array("Pattern      ", pattern, patlen);

    tiny_error_t status_corr = tiny_corr_f32(signal, siglen, pattern, patlen, corr_output);

    if (status_corr != TINY_OK) {
        printf("[tiny_corr_f32 Test] Failed with error code: %d\n", status_corr);
    } else {
        printf("Output vs Expected:\n");
        int pass = 1;
        for (int i = 0; i < corr_len; i++) {
            printf("  [%d] Output = %.3f | Expected = %.3f\n", i, corr_output[i], expected_corr[i]);
            if (fabs(corr_output[i] - expected_corr[i]) > EPSILON) {
                pass = 0;
            }
        }
        printf("[tiny_corr_f32 Test] [%s]\n", pass ? "PASS" : "FAIL");
    }

    /*** Test 2: Cross-Correlation Function - tiny_ccorr_f32 ***/
    const float x[] = {1.0f, 3.0f, 2.0f, 0.0f, 1.0f, 2.0f};
    const float y[] = {2.0f, 1.0f, 0.0f, -1.0f};
    const int len_x = sizeof(x) / sizeof(x[0]);
    const int len_y = sizeof(y) / sizeof(y[0]);
    const int ccorr_len = len_x + len_y - 1;

    float ccorr_output[ccorr_len];
    const float expected_ccorr[] = {-1.0f, -3.0f, -1.0f, 5.0f, 7.0f, 2.0f, 1.0f, 4.0f, 4.0f};

    printf("\n--- Test 2: tiny_ccorr_f32 ---\n");
    print_float_array("Input Signal X", x, len_x);
    print_float_array("Input Signal Y", y, len_y);

    tiny_error_t status_ccorr = tiny_ccorr_f32(x, len_x, y, len_y, ccorr_output);

    if (status_ccorr != TINY_OK) {
        printf("[tiny_ccorr_f32 Test] Failed with error code: %d\n", status_ccorr);
    } else {
        printf("Output vs Expected:\n");
        int pass = 1;
        for (int i = 0; i < ccorr_len; i++) {
            printf("  [%d] Output = %.3f | Expected = %.3f\n", i, ccorr_output[i], expected_ccorr[i]);
            if (fabs(ccorr_output[i] - expected_ccorr[i]) > EPSILON) {
                pass = 0;
            }
        }
        printf("[tiny_ccorr_f32 Test] [%s]\n", pass ? "PASS" : "FAIL");
    }

    printf("==========================================================\n");
}

```


## TEST RESULTS

```c
========== Correlation & Cross-Correlation Test ==========

--- Test 1: tiny_corr_f32 ---
Input Signal : [1.00, 2.00, 3.00, 4.00, 2.00, 1.00]
Pattern      : [2.00, 1.00, 0.00]
Output vs Expected:
  [0] Output = 4.000 | Expected = 4.000
  [1] Output = 7.000 | Expected = 7.000
  [2] Output = 10.000 | Expected = 10.000
  [3] Output = 10.000 | Expected = 10.000
[tiny_corr_f32 Test] [PASS]

--- Test 2: tiny_ccorr_f32 ---
Input Signal X: [1.00, 3.00, 2.00, 0.00, 1.00, 2.00]
Input Signal Y: [2.00, 1.00, 0.00, -1.00]
Output vs Expected:
  [0] Output = -1.000 | Expected = -1.000
  [1] Output = -3.000 | Expected = -3.000
  [2] Output = -1.000 | Expected = -1.000
  [3] Output = 5.000 | Expected = 5.000
  [4] Output = 7.000 | Expected = 7.000
  [5] Output = 2.000 | Expected = 2.000
  [6] Output = 1.000 | Expected = 1.000
  [7] Output = 4.000 | Expected = 4.000
  [8] Output = 4.000 | Expected = 4.000
[tiny_ccorr_f32 Test] [PASS]
==========================================================
```