/**
 * @file tiny_vec.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the header file for the submodule vec of the tiny_math middleware. This module is correspondign to the math & dotprod functions in the ESP-DSP library.
 * @version 1.0
 * @date 2025-04-15
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __TINY__VEC__
#define __TINY__VEC__

/* DEPENDENCIES */
#include "tiny_math_config.h"

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32 // ESP32 DSP library

#include "dsps_math.h" // math operations
#include "dsps_dotprod.h" // dot product

#endif

#ifdef __cplusplus
extern "C"
{
#endif

/* FUNCTION PROTOTYPES */
tiny_error_t tiny_vec_add_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out);
tiny_error_t tiny_vec_addc_f32(const float *input, float *output, int len, float C, int step_in, int step_out);
tiny_error_t tiny_vec_sub_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out);
tiny_error_t tiny_vec_subc_f32(const float *input, float *output, int len, float C, int step_in, int step_out);
tiny_error_t tiny_vec_mul_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out);
tiny_error_t tiny_vec_mulc_f32(const float *input, float *output, int len, float C, int step_in, int step_out);
tiny_error_t tiny_vec_div_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out, bool allow_divide_by_zero);
tiny_error_t tiny_vec_divc_f32(const float *input, float *output, int len, float C, int step_in, int step_out, bool allow_divide_by_zero);
tiny_error_t tiny_vec_sqrt_f32(const float *input, float *output, int len);
tiny_error_t tiny_vec_sqrtf_f32(const float *input, float *output, int len);
tiny_error_t tiny_vec_inv_sqrt_f32(const float *input, float *output, int len);
tiny_error_t tiny_vec_inv_sqrtf_f32(const float *input, float *output, int len);
tiny_error_t tiny_vec_dotprod_f32(const float *src1, const float *src2, float *dest, int len);
tiny_error_t tiny_vec_dotprode_f32(const float *src1, const float *src2, float *dest, int len, int step1, int step2);
#ifdef __cplusplus
}
#endif

#endif /* __TINY__VEC__ */