# TESTS

## tiny_view_test.h

```c
/**
 * @file tiny_view_test.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_view | test | header
 * @version 1.0
 * @date 2025-04-29
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
#include "tiny_view.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tiny_view_test(void);

#ifdef __cplusplus
}
#endif


```

## tiny_view_test.c

```c
/**
 * @file tiny_view_test.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_view | test | source
 * @version 1.0
 * @date 2025-04-29
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_view_test.h"
#include <math.h>

/**
 * @brief Generate a test signal (sine wave with noise)
 */
static void generate_test_signal(float *signal, int len, float freq, float sample_rate)
{
    for (int i = 0; i < len; i++)
    {
        float t = (float)i / sample_rate;
        signal[i] = sinf(2.0f * M_PI * freq * t) + 0.1f * sinf(2.0f * M_PI * freq * 3.0f * t);
    }
}

void tiny_view_test(void)
{
    printf("========== TinyView Test ==========\n\n");

    const int signal_len = 64;
    const float sample_rate = 1000.0f;
    float signal[signal_len];

    // Generate test signal
    generate_test_signal(signal, signal_len, 10.0f, sample_rate);

    // Test 1: Signal visualization
    printf("Test 1: Signal Visualization\n");
    printf("  Input: Sine wave signal (length=%d)\n", signal_len);
    tiny_view_signal_f32(signal, signal_len, 64, 16, 0, 0, "Test Signal: 10 Hz Sine Wave");

    // Test 2: Array printing
    printf("Test 2: Array Printing\n");
    printf("  Input: Signal array (length=%d)\n", signal_len);
    tiny_view_array_f32(signal, signal_len, "Test Signal", 3, 8);

    // Test 3: Statistics
    printf("Test 3: Signal Statistics\n");
    printf("  Input: Signal array (length=%d)\n", signal_len);
    tiny_view_statistics_f32(signal, signal_len, "Test Signal");

    // Test 4: Power spectrum visualization
    printf("Test 4: Power Spectrum Visualization\n");
    printf("  Input: Simulated power spectrum (length=128)\n");
    float power_spectrum[128];
    for (int i = 0; i < 128; i++)
    {
        // Simulate power spectrum with peaks at 10 Hz and 30 Hz
        float freq = (float)i * sample_rate / 256.0f;
        if (fabsf(freq - 10.0f) < 2.0f)
        {
            power_spectrum[i] = 100.0f - fabsf(freq - 10.0f) * 10.0f;
        }
        else if (fabsf(freq - 30.0f) < 2.0f)
        {
            power_spectrum[i] = 50.0f - fabsf(freq - 30.0f) * 5.0f;
        }
        else
        {
            power_spectrum[i] = 1.0f + (float)(i % 5);
        }
    }
    tiny_view_spectrum_f32(power_spectrum, 128, sample_rate, "Power Spectrum: Peaks at 10 Hz and 30 Hz");

    printf("========================================\n");
}

```


## TEST RESULTS

```
========== TinyView Test ==========

Test 1: Signal Visualization
  Input: Sine wave signal (length=64)

Test Signal: 10 Hz Sine Wave
Value
  1.07 |                                                                
  0.93 |                 *****************                              
  0.80 |            *****                 *****                         
  0.66 |         ***                           ***                      
  0.53 |       **                                 **                    
  0.39 |     **                                     **                  
  0.26 |   **                                         **                
  0.12 | **                                             **              
 -0.01 |*                                                 **            
 -0.15 |                                                    *           
 -0.28 |                                                     **         
 -0.42 |                                                       **       
 -0.56 |                                                         **     
 -0.69 |                                                           ***  
 -0.83 |                                                              **
 -0.96 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-0.962, 1.069], Length: 64

Test 2: Array Printing
  Input: Signal array (length=64)

Test Signal [64 elements]:
  [  0]   0.000   0.082   0.162   0.241   0.317   0.390   0.459   0.523 
  [  8]   0.582   0.635   0.683   0.725   0.762   0.793   0.819   0.840 
  [ 16]   0.857   0.870   0.880   0.887   0.892   0.896   0.898   0.899 
  [ 24]   0.900   0.900   0.900   0.899   0.898   0.896   0.892   0.887 
  [ 32]   0.880   0.870   0.857   0.840   0.819   0.793   0.762   0.725 
  [ 40]   0.683   0.635   0.582   0.523   0.459   0.390   0.317   0.241 
  [ 48]   0.162   0.082  -0.000  -0.082  -0.162  -0.241  -0.317  -0.390 
  [ 56]  -0.459  -0.523  -0.582  -0.635  -0.683  -0.725  -0.762  -0.793 

Test 3: Signal Statistics
  Input: Signal array (length=64)

=== Statistics: Test Signal ===
  Length:     64 samples
  Min:        -0.792711 (at index 63)
  Max:        0.900000 (at index 25)
  Peak:       0.900000 (at index 25)
  Mean:       0.414478
  Std Dev:    0.530688
  Variance:   0.281630
  Range:      1.692711
========================

Test 4: Power Spectrum Visualization
  Input: Simulated power spectrum (length=128)

Power Spectrum: Peaks at 10 Hz and 30 Hz
Power
 82.81 |   |                                                            
 77.36 |   |                                                            
 71.90 |   |                                                            
 66.45 |   |                                                            
 61.00 |   |                                                            
 55.54 |   |                                                            
 50.09 |   |                                                            
 44.63 |   |    |                                                       
 39.18 |   |    |                                                       
 33.72 |   |    |                                                       
 28.27 |   |    |                                                       
 22.82 |   |    |                                                       
 17.36 |   |    |                                                       
 11.91 |   |    |                                                       
  6.45 |   ||   ||   ||   ||   ||   ||   ||   ||   ||   ||   ||   ||   |
  1.00 |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
       ----------------------------------------------------------------
       0.0    31      62      94      125     156     188     219                          (Hz)
Range: [1.000, 82.812], Nyquist: 500.0 Hz

========================================
```

!!! warning
    As can be seen, the output through serial terminal is not quite accurate due to the limitation of character-based visualization. For more precise and detailed visualization, consider using graphical tools or libraries that support plotting and rendering of signals and spectra.
