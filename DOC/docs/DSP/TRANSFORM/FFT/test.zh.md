# TinyFFT — 测试与结果 {#_1}

<!-- Original section links retained for compatibility. -->
<span id="tiny_fft_testc"></span>
<span id="tiny_fft_testh"></span>

!!! info "实现依据与记录"
    本节接口以 `CODE/AIoTNode-TinyAuton-DSP/middleware/` 为依据。源码摘录与串口输出包含历史记录；是否运行某项测试，请核对工程入口和启用开关。

!!! abstract "验证目标"
    本例验证 15 / 70 / 120 Hz 合成信号的 FFT、谱峰搜索与 IFFT 重建。原始结果和完整测试源码均保留在下方。

## 历史结果速览 {#doc-历史结果速览}

记录未注明运行日期、设备和固件版本；源码文件日期不作为运行日期。以下数值摘自原始输出。

| 验证项 | 输入或参考 | 历史输出 | 解释 |
|---|---|---|---|
| 第一峰 | 15 Hz | 15.61 Hz | 绝对差 0.61 Hz |
| 第二峰 | 70 Hz | 70.32 Hz | 绝对差 0.32 Hz |
| 第三峰 | 120 Hz | 120.99 Hz | 绝对差 0.99 Hz |
| IFFT | 原输入信号 | 最大差值 `0.000002` | 本次记录的重建误差 |
| Hann 窗主峰 | 15 Hz | 15.29 Hz | 绝对差 0.29 Hz |
| 测试状态 | 原测试判据 | `PASS` | 仅对应本次运行和所用判据 |

## 配置、判据与结论 {#doc-配置-判据与结论}

采样率 `fs = 1000 Hz`，FFT 长度 `N = 256`，频点间隔 `fs/N = 3.90625 Hz`。实现对相邻谱点进行抛物线插值，所以峰值可以带小数；显示两位小数不等于验证了 0.01 Hz 的测量精度。

测试对无窗主峰使用 `|peak - 15| <= 5 Hz`，对前两个峰检查其频率为正，对 IFFT 使用最大误差 `<= 1e-3`；不能把 `PASS` 理解为三个峰都经过严格误差检验。Hann 窗可减轻泄漏，本次主峰更接近输入；原日志关于“改善频率分辨率”的文字作为历史记录保留，不应扩展为同样点数下必然能分辨更密集的频率。

复现入口为所选工程的 `middleware/tiny_dsp/transform/tiny_fft_test.c` 中 `tiny_fft_test()`。确认入口实际调用此函数，并同时保留配置和完整串口输出。

## 测试结果 {#_2}

```
========== TinyFFT Test ==========

1. FFT Initialization:
  ✓ FFT initialized (max size: 256)

2. Test Signal Generation:
  Input: Signal with frequencies 15 Hz, 70 Hz, and 120 Hz
  Sample rate: 1000.0 Hz
  Signal length: 256 samples
  First 10 samples: 0.400 0.584 0.623 0.505 0.281 0.056 -0.065 -0.016 0.194 0.498 

3. FFT (No Window):
  Input: Test signal (length=256)
  ✓ FFT completed
  Output: FFT result (complex, length=256)
  Magnitude spectrum: First 10 values: 4.801 5.603 8.669 20.017 111.409 16.176 9.060 6.401 4.987 4.095 

4. Peak Frequency Detection:
  Input: Power spectrum (length=256)
  Output: Peak frequency = 15.61 Hz (power = 48.484)
  Expected: strongest peak near ~15 Hz

5. Top Frequencies Detection:
  Input: Power spectrum (length=256)
  Output: Top 3 frequencies:
    [1] 15.61 Hz (power = 48.484)
    [2] 70.32 Hz (power = 10.440)
    [3] 120.99 Hz (power = 1.998)
  Expected: top frequencies should include ~15 Hz and ~70 Hz

6. IFFT (Signal Reconstruction):
  Input: FFT result (complex, length=256)
  Output: Reconstructed signal (length=256)
  First 10 samples: 0.400 0.584 0.623 0.505 0.281 0.056 -0.065 -0.016 0.194 0.498 
  Max difference from original: 0.000002
  ✓ IFFT reconstruction completed

7. FFT with Hanning Window:
  Input: Test signal (length=256) with Hanning window
  Output: Peak frequency = 15.29 Hz (power = 12.439)
  Note: Window reduces spectral leakage, improving frequency resolution

8. FFT Deinitialization:
  ✓ FFT deinitialized
  Result: PASS

========================================
```

## `tiny_fft_test.h` {#tinyffttesth}

```c
/**
 * @file tiny_fft_test.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_fft | test | header
 * @version 1.0
 * @date 2025-11-16
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
#include "tiny_fft.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tiny_fft_test(void);

#ifdef __cplusplus
}
#endif



```

## `tiny_fft_test.c` {#tinyffttestc}

<details class="auton-source" markdown="1">
<summary>展开 <code>tiny_fft_test.c</code> · 241 行</summary>

```c
/**
 * @file tiny_fft_test.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_fft | test | source
 * @version 1.0
 * @date 2025-11-16
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_fft_test.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/**
 * @brief Generate a test signal with known frequency components
 */
static void generate_test_signal(float *signal, int len, float sample_rate)
{
    // Generate signal: 0.9*sin(2*pi*15*t) + 0.4*cos(2*pi*70*t) + 0.2*sin(2*pi*120*t)
    // Frequencies: 15 Hz, 70 Hz, and 120 Hz
    for (int i = 0; i < len; i++)
    {
        float t = (float)i / sample_rate;
        signal[i] = 0.9f * sinf(2.0f * M_PI * 15.0f * t) +
                    0.4f * cosf(2.0f * M_PI * 70.0f * t) +
                    0.2f * sinf(2.0f * M_PI * 120.0f * t);
    }
}

void tiny_fft_test(void)
{
    printf("========== TinyFFT Test ==========\n\n");

    const int fft_size = 256;
    const float sample_rate = 1000.0f; // 1 kHz sampling rate
    const int signal_len = fft_size;
    int fft_test_pass = 1;

    // Initialize FFT
    printf("1. FFT Initialization:\n");
    tiny_error_t ret = tiny_fft_init(fft_size);
    if (ret != TINY_OK)
    {
        printf("  ✗ FFT initialization failed: %d\n", ret);
        return;
    }
    printf("  ✓ FFT initialized (max size: %d)\n\n", fft_size);

    // Generate test signal
    float *input_signal = (float *)malloc(signal_len * sizeof(float));
    float *fft_result = (float *)malloc(signal_len * 2 * sizeof(float)); // Complex output
    float *magnitude = (float *)malloc(signal_len * sizeof(float));
    float *power = (float *)malloc(signal_len * sizeof(float));
    float *reconstructed = (float *)malloc(signal_len * sizeof(float));

    if (!input_signal || !fft_result || !magnitude || !power || !reconstructed)
    {
        printf("  ✗ Memory allocation failed\n");
        goto cleanup;
    }

    generate_test_signal(input_signal, signal_len, sample_rate);

    printf("2. Test Signal Generation:\n");
    printf("  Input: Signal with frequencies 15 Hz, 70 Hz, and 120 Hz\n");
    printf("  Sample rate: %.1f Hz\n", sample_rate);
    printf("  Signal length: %d samples\n", signal_len);
    printf("  First 10 samples: ");
    for (int i = 0; i < 10 && i < signal_len; i++)
    {
        printf("%.3f ", input_signal[i]);
    }
    printf("\n\n");

    // Test FFT without window
    printf("3. FFT (No Window):\n");
    printf("  Input: Test signal (length=%d)\n", signal_len);
    ret = tiny_fft_f32(input_signal, signal_len, fft_result, TINY_FFT_WINDOW_NONE);
    if (ret != TINY_OK)
    {
        printf("  ✗ FFT failed: %d\n", ret);
        goto cleanup;
    }
    printf("  ✓ FFT completed\n");

    // Calculate magnitude
    ret = tiny_fft_magnitude_f32(fft_result, signal_len, magnitude);
    if (ret != TINY_OK)
    {
        printf("  ✗ Magnitude calculation failed\n");
        goto cleanup;
    }

    // Calculate power spectrum
    ret = tiny_fft_power_spectrum_f32(fft_result, signal_len, power);
    if (ret != TINY_OK)
    {
        printf("  ✗ Power spectrum calculation failed\n");
        goto cleanup;
    }

    printf("  Output: FFT result (complex, length=%d)\n", signal_len);
    printf("  Magnitude spectrum: First 10 values: ");
    for (int i = 0; i < 10 && i < signal_len; i++)
    {
        printf("%.3f ", magnitude[i]);
    }
    printf("\n\n");

    // Find peak frequency
    printf("4. Peak Frequency Detection:\n");
    printf("  Input: Power spectrum (length=%d)\n", signal_len);
    float peak_freq, peak_power;
    ret = tiny_fft_find_peak_frequency(power, signal_len, sample_rate, &peak_freq, &peak_power);
    if (ret != TINY_OK)
    {
        printf("  ✗ Peak detection failed\n");
        goto cleanup;
    }
    printf("  Output: Peak frequency = %.2f Hz (power = %.3f)\n", peak_freq, peak_power);
    printf("  Expected: strongest peak near ~15 Hz\n\n");
    if (fabsf(peak_freq - 15.0f) > 5.0f)
    {
        fft_test_pass = 0;
    }

    // Find top frequencies
    printf("5. Top Frequencies Detection:\n");
    printf("  Input: Power spectrum (length=%d)\n", signal_len);
    const int top_n = 3;
    float *top_freqs = (float *)malloc(top_n * sizeof(float));
    float *top_powers = (float *)malloc(top_n * sizeof(float));
    if (!top_freqs || !top_powers)
    {
        printf("  ✗ Memory allocation failed\n");
        goto cleanup;
    }

    ret = tiny_fft_find_top_frequencies(power, signal_len, sample_rate, top_n, top_freqs, top_powers);
    if (ret != TINY_OK)
    {
        printf("  ✗ Top frequencies detection failed\n");
        goto cleanup;
    }
    printf("  Output: Top %d frequencies:\n", top_n);
    for (int i = 0; i < top_n; i++)
    {
        printf("    [%d] %.2f Hz (power = %.3f)\n", i + 1, top_freqs[i], top_powers[i]);
    }
    printf("  Expected: top frequencies should include ~15 Hz and ~70 Hz\n\n");
    if (top_freqs[0] <= 0.0f || top_freqs[1] <= 0.0f)
    {
        fft_test_pass = 0;
    }

    // Test IFFT
    printf("6. IFFT (Signal Reconstruction):\n");
    printf("  Input: FFT result (complex, length=%d)\n", signal_len);
    ret = tiny_fft_ifft_f32(fft_result, signal_len, reconstructed);
    if (ret != TINY_OK)
    {
        printf("  ✗ IFFT failed: %d\n", ret);
        goto cleanup;
    }
    printf("  Output: Reconstructed signal (length=%d)\n", signal_len);
    printf("  First 10 samples: ");
    for (int i = 0; i < 10 && i < signal_len; i++)
    {
        printf("%.3f ", reconstructed[i]);
    }
    printf("\n");

    // Verify reconstruction (should match original, accounting for window effects)
    float max_diff = 0.0f;
    for (int i = 0; i < signal_len; i++)
    {
        float diff = fabsf(reconstructed[i] - input_signal[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
        }
    }
    printf("  Max difference from original: %.6f\n", max_diff);
    printf("  ✓ IFFT reconstruction completed\n\n");
    if (max_diff > 1e-3f)
    {
        fft_test_pass = 0;
    }

    // Test with window
    printf("7. FFT with Hanning Window:\n");
    printf("  Input: Test signal (length=%d) with Hanning window\n", signal_len);
    ret = tiny_fft_f32(input_signal, signal_len, fft_result, TINY_FFT_WINDOW_HANNING);
    if (ret != TINY_OK)
    {
        printf("  ✗ Windowed FFT failed\n");
        goto cleanup;
    }
    ret = tiny_fft_power_spectrum_f32(fft_result, signal_len, power);
    if (ret != TINY_OK)
    {
        printf("  ✗ Power spectrum calculation failed\n");
        goto cleanup;
    }
    ret = tiny_fft_find_peak_frequency(power, signal_len, sample_rate, &peak_freq, &peak_power);
    if (ret != TINY_OK)
    {
        printf("  ✗ Peak detection failed\n");
        goto cleanup;
    }
    printf("  Output: Peak frequency = %.2f Hz (power = %.3f)\n", peak_freq, peak_power);
    printf("  Note: Window reduces spectral leakage, improving frequency resolution\n\n");

    free(top_freqs);
    free(top_powers);

cleanup:
    if (input_signal)
        free(input_signal);
    if (fft_result)
        free(fft_result);
    if (magnitude)
        free(magnitude);
    if (power)
        free(power);
    if (reconstructed)
        free(reconstructed);

    // Deinitialize
    tiny_fft_deinit();
    printf("8. FFT Deinitialization:\n");
    printf("  ✓ FFT deinitialized\n");
    printf("  Result: %s\n", fft_test_pass ? "PASS" : "FAIL");

    printf("\n========================================\n");
}


```

</details>
