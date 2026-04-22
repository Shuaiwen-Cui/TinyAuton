/**
 * @file tiny_resample_test.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_resample | test | source
 * @version 1.0
 * @date 2025-04-29
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_resample_test.h" // tiny_resample test header
#include <math.h>               // for fabs()


void tiny_resample_test(void)
{
    printf("========== TinyResample Test ==========\n\n");

    // Original signal
    const float input[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const int input_len = sizeof(input) / sizeof(input[0]);
    
    printf("Original Signal (length=%d):\n", input_len);
    printf("  Input: ");
    for (int i = 0; i < input_len; i++) printf(" %.2f", input[i]);
    printf("\n\n");

    // ============================================================
    // Test 1: Downsampling
    // ============================================================
    printf("Test 1: Downsampling (keep=1, skip=2)\n");
    printf("  Description: Keep every 2nd sample, discard others\n");
    printf("  Input:  ");
    for (int i = 0; i < input_len; i++) printf(" %.2f", input[i]);
    printf("  (length=%d)\n", input_len);
    
    float downsampled[8];
    int down_len = 0;
    tiny_downsample_skip_f32(input, input_len, downsampled, &down_len, 1, 2);
    
    printf("  Output: ");
    for (int i = 0; i < down_len; i++) printf(" %.2f", downsampled[i]);
    printf("  (length=%d)\n", down_len);
    printf("  Mapping: input[0,2,4,6] -> output[0,1,2,3] = [1.00, 3.00, 5.00, 7.00]\n\n");

    // ============================================================
    // Test 2: Upsampling
    // ============================================================
    printf("Test 2: Upsampling (Zero-insertion)\n");
    printf("  Description: Insert zeros between samples to increase length\n");
    printf("  Input:  ");
    for (int i = 0; i < down_len; i++) printf(" %.2f", downsampled[i]);
    printf("  (length=%d)\n", down_len);
    
    float upsampled[16];
    tiny_upsample_zero_f32(downsampled, down_len, upsampled, 16);
    
    printf("  Output: ");
    for (int i = 0; i < 16; i++) printf(" %.2f", upsampled[i]);
    printf("  (length=16)\n");
    printf("  Mapping: input[0,1,2,3] -> output[0,4,8,12] = [1.00, 3.00, 5.00, 7.00]\n");
    printf("           (zeros inserted at positions 1,2,3,5,6,7,9,10,11,13,14,15)\n\n");

    // ============================================================
    // Test 3: Resampling
    // ============================================================
    printf("Test 3: Resampling (Linear Interpolation)\n");
    printf("  Description: Resample from %d to 12 samples using linear interpolation\n", input_len);
    printf("  Input:  ");
    for (int i = 0; i < input_len; i++) printf(" %.2f", input[i]);
    printf("  (length=%d)\n", input_len);
    
    float resampled[12];
    tiny_resample_f32(input, input_len, resampled, 12);
    
    printf("  Output: ");
    for (int i = 0; i < 12; i++) printf(" %.2f", resampled[i]);
    printf("  (length=12)\n");
    printf("  Mapping: Linear interpolation between input samples\n");
    printf("           output[0,2,4,6,8,10] = input[0,1,2,3,4,5] = [1.00, 2.00, 3.00, 4.00, 5.00, 6.00]\n");
    printf("           output[1,3,5,7,9,11] = interpolated midpoints\n\n");

    // ============================================================
    // Test 4: Validation - Verify interpolation correctness
    // ============================================================
    printf("Test 4: Validation - Verify Linear Interpolation Correctness\n");
    printf("  Purpose: Verify that interpolated values are correctly calculated\n");
    printf("           using the formula: output[i] = input[index]*(1-frac) + input[index+1]*frac\n");
    printf("           where pos = i/ratio, index = floor(pos), frac = pos - index\n\n");
    
    float ratio = 12.0f / 8.0f;
    int validation_errors = 0;
    
    printf("  Sample verification (checking a few key points):\n");
    for (int i = 0; i < 12; i++) {
        float pos = i / ratio;
        int index = (int)floorf(pos);
        float frac = pos - index;
        float expected;
        
        if (index >= input_len - 1) {
            expected = input[input_len - 1];
        } else {
            expected = input[index] * (1.0f - frac) + input[index + 1] * frac;
        }
        
        float diff = fabs(resampled[i] - expected);
        if (diff > 0.01f) {
            validation_errors++;
        }
        
        // Only print a few key points to avoid clutter
        if (i == 0 || i == 1 || i == 2 || i == 6 || i == 11) {
            printf("    output[%2d]: pos=%.3f, index=%d, frac=%.3f -> %.2f (expected: %.2f) %s\n",
                   i, pos, index, frac, resampled[i], expected,
                   (diff < 0.01f) ? "✓" : "✗");
        }
    }
    
    if (validation_errors == 0) {
        printf("  ✓ All interpolated values are correct!\n");
    } else {
        printf("  ✗ Found %d interpolation errors\n", validation_errors);
    }

    printf("\n========================================\n");
}