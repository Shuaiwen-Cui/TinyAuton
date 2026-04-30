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
