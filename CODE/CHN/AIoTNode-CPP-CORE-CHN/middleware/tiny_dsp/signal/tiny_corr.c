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

    if (siglen < patlen) // signal length shoudl be greater than pattern length
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

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    dsps_ccorr_f32(Signal, siglen, Kernel, kernlen, corrvout);
#else
    float *sig = (float *)Signal;
    float *kern = (float *)Kernel;
    int lsig = siglen;
    int lkern = kernlen;

    // swap signal and kernel if needed
    if (siglen < kernlen)
    {
        sig = (float *)Kernel;
        kern = (float *)Signal;
        lsig = kernlen;
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