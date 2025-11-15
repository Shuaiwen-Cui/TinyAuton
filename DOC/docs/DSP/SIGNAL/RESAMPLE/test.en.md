# TESTS

## tiny_resample.h

```c

/**
 * @file tiny_resample_test.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_resample | test | header
 * @version 1.0
 * @date 2025-04-29
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
#include "tiny_resample.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tiny_resample_test(void);

#ifdef __cplusplus
}
#endif



```

## tiny_resample.c

```c

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


void tiny_resample_test(void)
{
    printf("========== TinyResample Test ==========\n");

    const float input[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const int input_len = sizeof(input) / sizeof(input[0]);

    // Downsample
    float downsampled[8];
    int down_len = 0;
    tiny_downsample_skip_f32(input, input_len, downsampled, &down_len, 1, 2);
    printf("Downsampled (skip 2): ");
    for (int i = 0; i < down_len; i++) printf(" %.2f", downsampled[i]);
    printf("\n");

    // Upsample
    float upsampled[16];
    tiny_upsample_zero_f32(downsampled, down_len, upsampled, 16);
    printf("Zero-Upsampled:     ");
    for (int i = 0; i < 16; i++) printf(" %.2f", upsampled[i]);
    printf("\n");

    // Resample
    float resampled[12];
    tiny_resample_f32(input, input_len, resampled, 12);
    printf("Interpolated:       ");
    for (int i = 0; i < 12; i++) printf(" %.2f", resampled[i]);
    printf("\n");

    // Validate linear interpolation at midpoints
    printf("Resample validation at midpoints (expected: 1.5, 2.5, ..., 7.5):\n");
    for (int i = 1; i < 12; i += 2) {
        printf("  midpoint[%d] = %.2f\n", i, resampled[i]);
    }

    printf("========================================\n");
}

```

## Test Output

```c
========== TinyResample Test ==========
Downsampled (skip 2):  1.00 3.00 5.00 7.00
Zero-Upsampled:      1.00 0.00 0.00 0.00 3.00 0.00 0.00 0.00 5.00 0.00 0.00 0.00 7.00 0.00 0.00 0.00
Interpolated:        1.00 1.67 2.33 3.00 3.67 4.33 5.00 5.67 6.33 7.00 7.67 8.00
Resample validation at midpoints (expected: 1.5, 2.5, ..., 7.5):
  midpoint[1] = 1.67
  midpoint[3] = 3.00
  midpoint[5] = 4.33
  midpoint[7] = 5.67
  midpoint[9] = 7.00
  midpoint[11] = 8.00
========================================
```