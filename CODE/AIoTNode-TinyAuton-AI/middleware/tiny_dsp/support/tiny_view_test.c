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
    // Note: tiny_view_spectrum_f32 plots the whole single-sided spectrum
    // (0 .. Nyquist = sample_rate/2). With sample_rate=1000 Hz the X axis
    // spans 0..500 Hz, so we place test peaks well within that range to
    // get a clear visual demonstration.
    printf("Test 4: Power Spectrum Visualization\n");
    printf("  Input: Simulated power spectrum (length=128)\n");
    float power_spectrum[128];
    const float peak1_hz = 100.0f;
    const float peak2_hz = 300.0f;
    for (int i = 0; i < 128; i++)
    {
        float freq = (float)i * sample_rate / 256.0f;
        if (fabsf(freq - peak1_hz) < 20.0f)
        {
            power_spectrum[i] = 100.0f - fabsf(freq - peak1_hz) * 1.0f;
        }
        else if (fabsf(freq - peak2_hz) < 20.0f)
        {
            power_spectrum[i] = 50.0f - fabsf(freq - peak2_hz) * 0.5f;
        }
        else
        {
            power_spectrum[i] = 1.0f + (float)(i % 5);
        }
    }
    tiny_view_spectrum_f32(power_spectrum, 128, sample_rate,
                           "Power Spectrum: Peaks at 100 Hz and 300 Hz");

    printf("========================================\n");
}

