# TinyDSP · SIGNAL · CORRELATION — 实现与源码 {#_1}

!!! info "实现依据与记录"
    本节接口以 `CODE/AIoTNode-TinyAuton-DSP/middleware/` 为依据。源码摘录与串口输出包含历史记录；是否运行某项测试，请核对工程入口和启用开关。

本页保留完整源码摘录，按文件展开阅读。先看[设计说明](notes.md)了解数据流、接口和算法，再核对实现；源码摘录可能属于历史版本，当前实现请以所选工程为准。

## `tiny_corr.h` {#tiny_corrh}

<details class="auton-source" markdown="1">
<summary>展开 <code>tiny_corr.h</code> · 61 行</summary>

```c
/**
 * @file tiny_corr.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_corr | code | header
 * @version 1.0
 * @date 2025-04-27
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
// tiny_dsp configuration file
#include "tiny_dsp_config.h"

// ESP32 DSP Library for Acceleration
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32 // ESP32 DSP library

#include "dsps_corr.h"
#include "dsps_ccorr.h"

#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @name: tiny_corr_f32
     * @brief Correlation function
     *
     * @param Signal: input signal array
     * @param siglen: length of the signal array
     * @param Pattern: input pattern array
     * @param patlen: length of the pattern array
     * @param dest: output array for the correlation result
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_corr_f32(const float *Signal, const int siglen, const float *Pattern, const int patlen, float *dest);

    /**
     * @name: tiny_ccorr_f32
     * @brief Cross-correlation function
     *
     * @param Signal: input signal array
     * @param siglen: length of the signal array
     * @param Kernel: input kernel array
     * @param kernlen: length of the kernel array
     * @param corrvout: output array for the cross-correlation result
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_ccorr_f32(const float *Signal, const int siglen, const float *Kernel, const int kernlen, float *corrvout);

#ifdef __cplusplus
}
#endif

```

</details>

## `tiny_corr.c` {#tiny_corrc}

<details class="auton-source" markdown="1">
<summary>展开 <code>tiny_corr.c</code> · 148 行</summary>

```c
/**
 * @file tiny_corr.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_corr | code | source
 * @version 1.0
 * @date 2025-04-27
 * @copyright Copyright (c) 2025
 *
 */
/* DEPENDENCIES */
#include "tiny_corr.h"

/**
 * @name: tiny_corr_f32
 * @brief Correlation function
 *
 * @param Signal: input signal array
 * @param siglen: length of the signal array
 * @param Pattern: input pattern array
 * @param patlen: length of the pattern array
 * @param dest: output array for the correlation result
 *
 * @return tiny_error_t
 */
tiny_error_t tiny_corr_f32(const float *Signal, const int siglen, const float *Pattern, const int patlen, float *dest)
{
    if (NULL == Signal || NULL == Pattern || NULL == dest)
    {
        return TINY_ERR_DSP_NULL_POINTER;
    }

    if (siglen <= 0 || patlen <= 0)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }

    if (siglen < patlen) /* pattern must fit within the signal */
    {
        return TINY_ERR_DSP_MISMATCH;
    }

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    dsps_corr_f32(Signal, siglen, Pattern, patlen, dest);
#else

    for (size_t n = 0; n <= (siglen - patlen); n++)
    {
        float k_corr = 0;
        for (size_t m = 0; m < patlen; m++)
        {
            k_corr += Signal[n + m] * Pattern[m];
        }
        dest[n] = k_corr;
    }

#endif

    return TINY_OK;
}

/**
 * @name: tiny_ccorr_f32
 * @brief Cross-correlation function
 *
 * @param Signal: input signal array
 * @param siglen: length of the signal array
 * @param Kernel: input kernel array
 * @param kernlen: length of the kernel array
 * @param corrvout: output array for the cross-correlation result
 *
 * @return tiny_error_t
 */
tiny_error_t tiny_ccorr_f32(const float *Signal, const int siglen, const float *Kernel, const int kernlen, float *corrvout)
{
    if (NULL == Signal || NULL == Kernel || NULL == corrvout)
    {
        return TINY_ERR_DSP_NULL_POINTER;
    }

    if (siglen <= 0 || kernlen <= 0)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    dsps_ccorr_f32(Signal, siglen, Kernel, kernlen, corrvout);
#else
    const float *sig  = Signal;
    const float *kern = Kernel;
    int lsig  = siglen;
    int lkern = kernlen;

    /* Cross-correlation is symmetric in its operand lengths, so make
     * the longer one the "signal" to keep the indexing in the three
     * stages below valid for either calling convention. */
    if (siglen < kernlen)
    {
        sig  = Kernel;
        kern = Signal;
        lsig  = kernlen;
        lkern = siglen;
    }
    // stage I
    for (int n = 0; n < lkern; n++)
    {
        size_t k;
        size_t kmin = lkern - 1 - n;
        corrvout[n] = 0;

        for (k = 0; k <= n; k++)
        {
            corrvout[n] += sig[k] * kern[kmin + k];
        }
    }

    // stage II
    for (int n = lkern; n < lsig; n++)
    {
        size_t kmin, kmax, k;

        corrvout[n] = 0;

        kmin = n - lkern + 1;
        kmax = n;
        for (k = kmin; k <= kmax; k++)
        {
            corrvout[n] += sig[k] * kern[k - kmin];
        }
    }

    // stage III
    for (int n = lsig; n < lsig + lkern - 1; n++)
    {
        size_t kmin, kmax, k;

        corrvout[n] = 0;

        kmin = n - lkern + 1;
        kmax = lsig - 1;

        for (k = kmin; k <= kmax; k++)
        {
            corrvout[n] += sig[k] * kern[k - kmin];
        }
    }
#endif
    return TINY_OK;
}
```

</details>
