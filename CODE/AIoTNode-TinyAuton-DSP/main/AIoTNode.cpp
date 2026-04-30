/**
 * @file AIoTNode.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief AIoTNode application entry point
 *
 * Currently used to drive the staged test of the tiny_dsp middleware.
 * Tests are enabled module-by-module in the order:
 *   support -> signal -> filter -> transform
 * Toggle a single block via the macros below to focus on one module
 * at a time.
 */

/* ============================================================
 * tiny_dsp test selector
 * ============================================================ */

/* Stage 1: support  (visualization, array dump, statistics) */
#define TEST_TINY_DSP_SUPPORT      0

/* Stage 2: signal   (conv / corr / resample) */
#define TEST_TINY_DSP_SIGNAL       0

/* Stage 3: filter   (FIR / IIR) */
#define TEST_TINY_DSP_FILTER       0

/* Stage 4: transform (FFT / DWT / ICA) */
#define TEST_TINY_DSP_TRANSFORM    1

/* ============================================================
 * Includes
 * ============================================================ */

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

/* All tiny_dsp public APIs and *_test() entry points */
#include "tiny_dsp.h"

/* ============================================================
 * Application entry
 * ============================================================ */

extern "C" void app_main(void)
{
    /* Small delay so the serial monitor catches the very first lines */
    vTaskDelay(pdMS_TO_TICKS(500));

    printf("\n");
    printf("##############################################\n");
    printf("#          tiny_dsp module test runner       #\n");
    printf("##############################################\n");

#if TEST_TINY_DSP_SUPPORT
    printf("\n>>> [tiny_dsp / support] tiny_view_test\n");
    tiny_view_test();
#endif

#if TEST_TINY_DSP_SIGNAL
    printf("\n>>> [tiny_dsp / signal] tiny_signal_conv_test\n");
    tiny_signal_conv_test();
    printf("\n>>> [tiny_dsp / signal] tiny_signal_corr_ccorr_test\n");
    tiny_signal_corr_ccorr_test();
    printf("\n>>> [tiny_dsp / signal] tiny_resample_test\n");
    tiny_resample_test();
#endif

#if TEST_TINY_DSP_FILTER
    printf("\n>>> [tiny_dsp / filter] tiny_fir_test\n");
    tiny_fir_test();
    printf("\n>>> [tiny_dsp / filter] tiny_iir_test\n");
    tiny_iir_test();
#endif

#if TEST_TINY_DSP_TRANSFORM
    printf("\n>>> [tiny_dsp / transform] tiny_fft_test\n");
    tiny_fft_test();
    /* tiny_dwt_test_all: single-level, multilevel (cD_lens path), DB1–DB10 round-trip */
    printf("\n>>> [tiny_dsp / transform] tiny_dwt_test_all\n");
    tiny_dwt_test_all();
    printf("\n>>> [tiny_dsp / transform] tiny_ica_test_all\n");
    tiny_ica_test_all();
#endif

    printf("\n##############################################\n");
    printf("#         tiny_dsp test runner finished      #\n");
    printf("##############################################\n");

    /* Idle loop so the task does not exit and reset. */
    while (true)
    {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
