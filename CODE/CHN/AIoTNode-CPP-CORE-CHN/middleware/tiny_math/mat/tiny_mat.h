/**
 * @file tiny_mat.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the header file for the submodule mat (basic matrix operations) of the tiny_math middleware.
 * @version 1.0
 * @date 2025-04-17
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
#include "tiny_math_config.h"

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
// ESP32 DSP library
#include "dspm_matrix.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

/* FUNCTION PROTOTYPES */
// print matrix
void print_matrix(const char *name, const float *mat, int rows, int cols);
// print matrix padded (row-major)
void print_matrix_padded(const char *name, const float *mat, int rows, int cols, int step);
// addition
tiny_error_t tiny_mat_add_f32(const float *input1, const float *input2, float *output, int rows, int cols, int padd1, int padd2, int padd_out, int step1, int step2, int step_out);
tiny_error_t tiny_mat_addc_f32(const float *input, float *output, float C, int rows, int cols, int padd_in, int padd_out, int step_in, int step_out);
// subtraction
tiny_error_t tiny_mat_sub_f32(const float *input1, const float *input2, float *output, int rows, int cols, int padd1, int padd2, int padd_out, int step1, int step2, int step_out);
tiny_error_t tiny_mat_subc_f32(const float *input, float *output, float C, int rows, int cols, int padd_in, int padd_out, int step_in, int step_out);
// multiplication
tiny_error_t tiny_mat_mult_f32(const float *A, const float *B, float *C, int m, int n, int k);
tiny_error_t tiny_mat_mult_ex_f32(const float *A, const float *B, float *C, int A_rows, int A_cols, int B_cols, int A_padding, int B_padding, int C_padding);
tiny_error_t tiny_mat_multc_f32(const float *input, float *output, float C, int rows, int cols, int padd_in, int padd_out, int step_in, int step_out);

#ifdef __cplusplus
}
#endif
