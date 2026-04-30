/**
 * @file tiny_dwt.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_dwt | code | header
 * @version 1.0
 * @date 2025-04-29
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
// tiny_dsp configuration file
#include "tiny_dsp_config.h"

// tiny_dsp submodules
#include "tiny_conv.h"     // Convolution
#include "tiny_resample.h" // Resampling

// ESP32 DSP Library for Acceleration
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32 // ESP32 DSP library

// Currently No ESP32 DSP Library for DWT

#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /* DWT FILTERS */

    typedef enum
    {
        TINY_WAVELET_DB1 = 0,
        TINY_WAVELET_DB2,
        TINY_WAVELET_DB3,
        TINY_WAVELET_DB4,
        TINY_WAVELET_DB5,
        TINY_WAVELET_DB6,
        TINY_WAVELET_DB7,
        TINY_WAVELET_DB8,
        TINY_WAVELET_DB9,
        TINY_WAVELET_DB10,
        TINY_WAVELET_COUNT
    } tiny_wavelet_type_t;

    extern const float *tiny_lo_d_table[TINY_WAVELET_COUNT];
    extern const float *tiny_hi_d_table[TINY_WAVELET_COUNT];
    extern const float *tiny_lo_r_table[TINY_WAVELET_COUNT];
    extern const float *tiny_hi_r_table[TINY_WAVELET_COUNT];
    extern const int tiny_filter_length_table[TINY_WAVELET_COUNT];

#define TINY_WAVELET_GET_LO_D(w) tiny_lo_d_table[(w)]
#define TINY_WAVELET_GET_HI_D(w) tiny_hi_d_table[(w)]
#define TINY_WAVELET_GET_LO_R(w) tiny_lo_r_table[(w)]
#define TINY_WAVELET_GET_HI_R(w) tiny_hi_r_table[(w)]
#define TINY_WAVELET_GET_LEN(w) tiny_filter_length_table[(w)]

    /* FUNCTION STATEMENTS */
    /**
     * @name tiny_dwt_decompose_f32
     * @brief Perform single-level discrete wavelet decomposition
     */
    tiny_error_t tiny_dwt_decompose_f32(const float *input, int input_len,
                                        tiny_wavelet_type_t wavelet,
                                        float *cA, float *cD,
                                        int *cA_len, int *cD_len);

    /**
     * @name tiny_dwt_reconstruct_f32
     * @brief Perform single-level discrete wavelet reconstruction
     */
    tiny_error_t tiny_dwt_reconstruct_f32(const float *cA, const float *cD, int coeff_len,
                                          tiny_wavelet_type_t wavelet,
                                          float *output, int *output_len);

    /**
     * @name tiny_dwt_multilevel_decompose_f32
     * @brief Perform multi-level DWT decomposition
     */
    tiny_error_t tiny_dwt_multilevel_decompose_f32(const float *input, int input_len,
                                                   tiny_wavelet_type_t wavelet, int levels,
                                                   float **cA_out, float **cD_out, int *len_out);

    /**
     * @name tiny_dwt_coeffs_process
     * @brief Placeholder for user-defined coefficient processing
     */
    void tiny_dwt_coeffs_process(float *cA, float *cD, int cA_len, int cD_len, int levels);

    /**
     * @name tiny_dwt_multilevel_reconstruct_f32
     * @brief Perform multi-level DWT reconstruction
     */
    tiny_error_t tiny_dwt_multilevel_reconstruct_f32(const float *cA_init, const float *cD_all,
                                                     int final_len, tiny_wavelet_type_t wavelet, int levels,
                                                     float *output);

#ifdef __cplusplus
}
#endif
