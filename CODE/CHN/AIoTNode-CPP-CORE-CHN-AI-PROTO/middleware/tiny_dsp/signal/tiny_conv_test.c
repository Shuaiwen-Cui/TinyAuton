/**
 * @file tiny_conv_test.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_conv | test | header
 * @version 1.0
 * @date 2025-04-28
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
// tiny_conv
#include "tiny_conv_test.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EPSILON 1e-5  // Tolerance for floating-point comparison

// Helper function to compare two float arrays
int compare_float_arrays(const float *a, const float *b, int len, float tol)
{
    for (int i = 0; i < len; i++)
    {
        if (fabs(a[i] - b[i]) > tol)
        {
            return 0; // Not equal
        }
    }
    return 1; // Equal
}

// Test function
void tiny_signal_conv_test(void)
{
    printf("===== tiny_conv_f32 and tiny_conv_ex_f32 Test Start =====\n");

    // Define sample signal and kernel
    float signal[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
    float kernel[] = {0.2f, 0.5f, 0.3f};
    int siglen = sizeof(signal) / sizeof(signal[0]);
    int kernlen = sizeof(kernel) / sizeof(kernel[0]);

    int convlen_full = siglen + kernlen - 1;
    float convout_ref[convlen_full];
    float convout_test[convlen_full];

    // show signal and kernel 
    printf("Signal: ");
    for (int i = 0; i < siglen; i++)
    {
        printf("%.2f ", signal[i]);
    }
    printf("\nKernel: ");
    for (int i = 0; i < kernlen; i++)
    {
        printf("%.2f ", kernel[i]);
    }
    printf("\n");

    // 1. Test basic tiny_conv_f32
    printf("\n[Test] tiny_conv_f32 basic full convolution...\n");
    memset(convout_ref, 0, sizeof(convout_ref));
    if (tiny_conv_f32(signal, siglen, kernel, kernlen, convout_ref) != TINY_OK)
    {
        printf("[FAIL] tiny_conv_f32 error.\n");
        return;
    }
    printf("[PASS] tiny_conv_f32 full convolution completed.\n");

    // 2. Test tiny_conv_ex_f32 with padding = zero, conv_mode = full
    printf("\n[Test] tiny_conv_ex_f32 padding=zero, mode=full...\n");
    memset(convout_test, 0, sizeof(convout_test));
    if (tiny_conv_ex_f32(signal, siglen, kernel, kernlen, convout_test, TINY_PADDING_ZERO, TINY_CONV_FULL) != TINY_OK)
    {
        printf("[FAIL] tiny_conv_ex_f32 error.\n");
        return;
    }

    if (compare_float_arrays(convout_ref, convout_test, convlen_full, EPSILON))
    {
        printf("[PASS] tiny_conv_ex_f32 matches tiny_conv_f32 (zero padding, full mode).\n");
    }
    else
    {
        printf("[FAIL] tiny_conv_ex_f32 result mismatch!\n");
        return;
    }

    // 3. Test tiny_conv_ex_f32 with different padding modes
    tiny_padding_mode_t paddings[] = {TINY_PADDING_ZERO, TINY_PADDING_SYMMETRIC, TINY_PADDING_PERIODIC};
    const char *padding_names[] = {"ZERO", "SYMMETRIC", "PERIODIC"};

    // 4. Test different convolution modes
    tiny_conv_mode_t modes[] = {TINY_CONV_FULL, TINY_CONV_HEAD, TINY_CONV_CENTER, TINY_CONV_TAIL};
    const char *mode_names[] = {"FULL", "HEAD", "CENTER", "TAIL"};

    for (int p = 0; p < 3; p++)
    {
        for (int m = 0; m < 4; m++)
        {
            printf("\n[Test] tiny_conv_ex_f32 padding=%s, mode=%s...\n", padding_names[p], mode_names[m]);
            memset(convout_test, 0, sizeof(convout_test));

            if (tiny_conv_ex_f32(signal, siglen, kernel, kernlen, convout_test, paddings[p], modes[m]) != TINY_OK)
            {
                printf("[FAIL] tiny_conv_ex_f32 error at padding=%s, mode=%s.\n", padding_names[p], mode_names[m]);
                continue;
            }

            printf("[PASS] tiny_conv_ex_f32 completed (padding=%s, mode=%s).\n", padding_names[p], mode_names[m]);
            printf("Output:\n");

            int out_len = 0;
            switch (modes[m])
            {
            case TINY_CONV_FULL:
                out_len = siglen + kernlen - 1;
                break;
            case TINY_CONV_HEAD:
                out_len = kernlen;
                break;
            case TINY_CONV_CENTER:
                out_len = siglen;
                break;
            case TINY_CONV_TAIL:
                out_len = kernlen;
                break;
            default:
                break;
            }

            for (int i = 0; i < out_len; i++)
            {
                printf("%.5f ", convout_test[i]);
            }
            printf("\n");
        }
    }

    printf("\n===== tiny_conv_f32 and tiny_conv_ex_f32 Test End =====\n");
}
