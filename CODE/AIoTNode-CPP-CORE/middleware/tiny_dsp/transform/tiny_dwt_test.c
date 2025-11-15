/**
 * @file tiny_dwt_test.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_dwt | test | source
 * @version 1.0
 * @date 2025-04-30
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_dwt_test.h" // TinyDWT Test Header

/**
 * @name tiny_dwt_test
 * @brief Unit test for single-level DWT and inverse DWT
 */
void tiny_dwt_test(void)
{
    printf("========== TinyDWT Single-Level Test ==========");

    float input[] = {
        0.0, 0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5,
        0.0, -0.5, -1.0, -1.5, -2.0, -1.5, -1.0, -0.5};
    int input_len = sizeof(input) / sizeof(input[0]);

    float cA[32] = {0}, cD[32] = {0};
    int cA_len = 0, cD_len = 0;

    float output[64] = {0};
    int output_len = 0;

    tiny_error_t err;

    err = tiny_dwt_decompose_f32(input, input_len, TINY_WAVELET_DB4, cA, cD, &cA_len, &cD_len);
    if (err != TINY_OK)
    {
        printf("[FAIL] DWT decomposition error: %d\n", err);
        return;
    }

    err = tiny_dwt_reconstruct_f32(cA, cD, cA_len, TINY_WAVELET_DB4, output, &output_len);
    if (err != TINY_OK)
    {
        printf("[FAIL] DWT reconstruction error: %d\n", err);
        return;
    }

    printf("\nInput:         ");
    for (int i = 0; i < input_len; i++)
        printf("%6.2f ", input[i]);
    printf("\nApprox (cA):   ");
    for (int i = 0; i < cA_len; i++)
        printf("%6.2f ", cA[i]);
    printf("\nDetail (cD):   ");
    for (int i = 0; i < cD_len; i++)
        printf("%6.2f ", cD[i]);
    printf("\nReconstructed: ");
    for (int i = 0; i < output_len; i++)
        printf("%6.2f ", output[i]);
    printf("\n================================================\n");

    int filter_len = TINY_WAVELET_GET_LEN(TINY_WAVELET_DB4);
    int start = filter_len / 2;
    int end = input_len - filter_len / 2;
    int correct = 1;
    for (int i = start; i < end; i++)
    {
        if (fabs(output[i] - input[i]) > 1e-3)
        {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("[PASS] Reconstruction matches original signal (center region).\n");
    else
        printf("[FAIL] Reconstruction differs in center region.\n");
}

/**
 * @name tiny_dwt_test_multilevel
 * @brief Test multi-level DWT decomposition and reconstruction
 */
void tiny_dwt_test_multilevel(void)
{
    printf("========== TinyDWT Multi-Level Test ==========");

    float input[] = {
        0.0, 0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5,
        0.0, -0.5, -1.0, -1.5, -2.0, -1.5, -1.0, -0.5,
        0.0, 0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5,
        0.0, -0.5, -1.0, -1.5, -2.0, -1.5, -1.0, -0.5, -7.0, -2.5};
    int input_len = sizeof(input) / sizeof(input[0]);
    int levels = 3;

    float *cA = NULL;
    float *cD = NULL;
    int cA_len = 0;

    tiny_error_t err = tiny_dwt_multilevel_decompose_f32(input, input_len, TINY_WAVELET_DB4, levels, &cA, &cD, &cA_len);
    if (err != TINY_OK)
    {
        printf("\n[FAIL] Multi-level DWT decomposition error: %d\n", err);
        return;
    }

    tiny_dwt_coeffs_process(cA, cD, cA_len, input_len - cA_len, levels);

    float *output = (float *)malloc(sizeof(float) * input_len);
    if (!output)
    {
        printf("\n[FAIL] Memory alloc error for output\n");
        free(cA);
        free(cD);
        return;
    }

    err = tiny_dwt_multilevel_reconstruct_f32(cA, cD, cA_len, TINY_WAVELET_DB4, levels, output);
    if (err != TINY_OK)
    {
        printf("\n[FAIL] Multi-level DWT reconstruction error: %d\n", err);
        free(cA);
        free(cD);
        free(output);
        return;
    }

    printf("\nInput:        ");
    for (int i = 0; i < input_len; i++)
        printf("%6.2f ", input[i]);
    printf("\nReconstructed:");
    for (int i = 0; i < input_len; i++)
        printf("%6.2f ", output[i]);
    printf("\n================================================\n");

    free(cA);
    free(cD);
    free(output);
}