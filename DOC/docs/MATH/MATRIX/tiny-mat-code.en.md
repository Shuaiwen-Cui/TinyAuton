# CODE

## tiny_mat.h

```c
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

```

## tiny_mat.c

```c
/**
 * @file tiny_mat.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the source file for the submodule mat (basic matrix operations) of the tiny_math middleware.
 * @version 1.0
 * @date 2025-04-17
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_mat.h"

/* SUPPORTIVE FUNCTIONS */

/**
 * @name print_matrix
 * @brief Prints a matrix to the console.
 * @param name Name of the matrix.
 * @param mat Pointer to the matrix data.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 */
void print_matrix(const char *name, const float *mat, int rows, int cols)
{
    printf("%s =\n\r", name);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%10.6f ", mat[i * cols + j]); // padding not considered, row-major order
        }
        printf("\n\r");
    }
    printf("\n\r");
}

// print matrix padded
/**
 * @name print_matrix
 * @brief Prints a matrix to the console.
 * @param name Name of the matrix.
 * @param mat Pointer to the matrix data.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @param step Step size (how many elements in a row) for the matrix data. row-major order.
 */
void print_matrix_padded(const char *name, const float *mat, int rows, int cols, int step)
{
    printf("%s =\n\r", name);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%10.6f ", mat[i * step + j]); // padding considered
        }
        printf("\n\r");
    }
    printf("\n\r");
}

/* ADDITION */

// matrix + matrix | float

/**
 * @name tiny_mat_add_f32
 * @brief Adds two matrices of type float32.
 * @param input1 Pointer to the first input matrix.
 * @param input2 Pointer to the second input matrix.
 * @param output Pointer to the output matrix.
 * @param rows Number of rows in the matrices.
 * @param cols Number of columns in the matrices.
 * @param padd1 Number of padding columns in the first input matrix.
 * @param padd2 Number of padding columns in the second input matrix.
 * @param padd_out Number of padding columns in the output matrix.
 * @param step1 Step size for the first input matrix.
 * @param step2 Step size for the second input matrix.
 * @param step_out Step size for the output matrix.
 * @return Returns TINY_OK on success, or an error code on failure.
 * @note This function performs matrix addition with the specified padding and step sizes.
 * @note The function assumes that the input matrices are in row-major order.
 */
tiny_error_t tiny_mat_add_f32(const float *input1, const float *input2, float *output, int rows, int cols, int padd1, int padd2, int padd_out, int step1, int step2, int step_out)
{
    if (NULL == input1 || NULL == input2 || NULL == output)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }
    // paddings must be non-negative, steps must be at least 1.
    if (rows <= 0 || cols <= 0 || padd1 < 0 || padd2 < 0 || padd_out < 0 || step1 <= 0 || step2 <= 0 || step_out <= 0)
    {
        return TINY_ERR_MATH_INVALID_PARAM;
    }

    // pad refers to the columns that are not used in the matrix operation

    /* Use explicit index math instead of mutating the caller pointers.
       This keeps input pointers const and avoids surprises from pointer
       arithmetic. The storage model is row-major with per-row reserved
       length = cols + padd. Logical column c is at base + c * step. */
    const int in1_row_stride = cols + padd1;
    const int in2_row_stride = cols + padd2;
    const int out_row_stride = cols + padd_out;

    // If we're on ESP32 and all paddings are 0 and all steps are 1 (contiguous),
    // prefer to call the optimized ESP-DSP implementation.
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    if (padd1 == 0 && padd2 == 0 && padd_out == 0 && step1 == 1 && step2 == 1 && step_out == 1) {
        dspm_add_f32(input1, input2, output, rows, cols, 0, 0, 0, 1, 1, 1);
        return TINY_OK;
    }
#endif

    for (int row = 0; row < rows; row++) {
        int base_in1 = row * in1_row_stride;
        int base_in2 = row * in2_row_stride;
        int base_out = row * out_row_stride;

        for (int col = 0; col < cols; col++) {
            int idx_in1 = base_in1 + col * step1;
            int idx_in2 = base_in2 + col * step2;
            int idx_out = base_out + col * step_out;

            /* bounds are the caller's responsibility, but avoid undefined
               behavior by checking indices minimally in debug builds if
               needed (not enforced here for performance). */
            output[idx_out] = input1[idx_in1] + input2[idx_in2];
        }
    }
    return TINY_OK;
}

// matrix + constant | float

/**
 * @name tiny_mat_addc_f32
 * @brief Adds a constant to each element of a matrix of type float32.
 * @param input Pointer to the input matrix.
 * @param output Pointer to the output matrix.
 * @param C Constant value to be added to each element of the matrix.
 * @param rows Number of rows in the matrices.
 * @param cols Number of columns in the matrices.
 * @param padd_in Number of padding columns in the input matrix.
 * @param padd_out Number of padding columns in the output matrix.
 * @param step_in Step size for the input matrix.
 * @param step_out Step size for the output matrix.
 * @return Returns TINY_OK on success, or an error code on failure.
 * @note This function performs matrix addition with a constant with the specified padding and step sizes.
 * @note The function assumes that the input matrix is in row-major order.
 */
tiny_error_t tiny_mat_addc_f32(const float *input, float *output, float C, int rows, int cols, int padd_in, int padd_out, int step_in, int step_out)
{
    if (NULL == input || NULL == output)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }
    // paddings must be non-negative, steps must be at least 1.
    if (rows <= 0 || cols <= 0 || padd_in < 0 || padd_out < 0 || step_in <= 0 || step_out <= 0)
    {
        return TINY_ERR_MATH_INVALID_PARAM;
    }
    // pad refers to the columns that are not used in the matrix operation
    // If running on ESP32 and all paddings are 0 and all steps are 1 (contiguous),
    // prefer the optimized ESP-DSP implementation.
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    if (padd_in == 0 && padd_out == 0 && step_in == 1 && step_out == 1) {
        dspm_addc_f32(input, output, C, rows, cols, 0, 0, 1, 1);
        return TINY_OK;
    }
#endif

    const int in_row_stride = cols + padd_in;
    const int out_row_stride = cols + padd_out;

    for (int row = 0; row < rows; row++) {
        int base_in = row * in_row_stride;
        int base_out = row * out_row_stride;

        for (int col = 0; col < cols; col++) {
            int idx_in = base_in + col * step_in;
            int idx_out = base_out + col * step_out;

            output[idx_out] = input[idx_in] + C;
        }
    }
    return TINY_OK;
}

/* SUBTRACTION */

// matrix - matrix | float

/**
 * @name tiny_mat_sub_f32
 * @brief Subtracts two matrices of type float32.
 * @param input1 Pointer to the first input matrix.
 * @param input2 Pointer to the second input matrix.
 * @param output Pointer to the output matrix.
 * @param rows Number of rows in the matrices.
 * @param cols Number of columns in the matrices.
 * @param padd1 Number of padding columns in the first input matrix.
 * @param padd2 Number of padding columns in the second input matrix.
 * @param padd_out Number of padding columns in the output matrix.
 * @param step1 Step size for the first input matrix.
 * @param step2 Step size for the second input matrix.
 * @param step_out Step size for the output matrix.
 * @return Returns TINY_OK on success, or an error code on failure.
 * @note This function performs matrix subtraction with the specified padding and step sizes.
 * @note The function assumes that the input matrices are in row-major order.
 */
tiny_error_t tiny_mat_sub_f32(const float *input1, const float *input2, float *output, int rows, int cols, int padd1, int padd2, int padd_out, int step1, int step2, int step_out)
{
    if (NULL == input1 || NULL == input2 || NULL == output)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }
    // paddings must be non-negative, steps must be at least 1.
    if (rows <= 0 || cols <= 0 || padd1 < 0 || padd2 < 0 || padd_out < 0 || step1 <= 0 || step2 <= 0 || step_out <= 0)
    {
        return TINY_ERR_MATH_INVALID_PARAM;
    }
    // pad refers to the columns that are not used in the matrix operation
    // Prefer ESP-DSP only when all paddings are 0 and all steps are 1 (contiguous)
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    if (padd1 == 0 && padd2 == 0 && padd_out == 0 && step1 == 1 && step2 == 1 && step_out == 1) {
        dspm_sub_f32(input1, input2, output, rows, cols, 0, 0, 0, 1, 1, 1);
        return TINY_OK;
    }
#endif

    const int in1_row_stride = cols + padd1;
    const int in2_row_stride = cols + padd2;
    const int out_row_stride = cols + padd_out;

    for (int row = 0; row < rows; row++) {
        int base_in1 = row * in1_row_stride;
        int base_in2 = row * in2_row_stride;
        int base_out = row * out_row_stride;

        for (int col = 0; col < cols; col++) {
            int idx_in1 = base_in1 + col * step1;
            int idx_in2 = base_in2 + col * step2;
            int idx_out = base_out + col * step_out;

            output[idx_out] = input1[idx_in1] - input2[idx_in2];
        }
    }
    return TINY_OK;
}

// matrix - constant | float

/**
 * @name tiny_mat_subc_f32
 * @brief Subtracts a constant from each element of a matrix of type float32.
 * @param input Pointer to the input matrix.
 * @param output Pointer to the output matrix.
 * @param C Constant value to be subtracted from each element of the matrix.
 * @param rows Number of rows in the matrices.
 * @param cols Number of columns in the matrices.
 * @param padd_in Number of padding columns in the input matrix.
 * @param padd_out Number of padding columns in the output matrix.
 * @param step_in Step size for the input matrix.
 * @param step_out Step size for the output matrix.
 * @return Returns TINY_OK on success, or an error code on failure.
 * @note This function performs matrix subtraction with a constant with the specified padding and step sizes.
 * @note The function assumes that the input matrix is in row-major order.
 */
tiny_error_t tiny_mat_subc_f32(const float *input, float *output, float C, int rows, int cols, int padd_in, int padd_out, int step_in, int step_out)
{
    if (NULL == input || NULL == output)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }
    // paddings must be non-negative, steps must be at least 1.
    if (rows <= 0 || cols <= 0 || padd_in < 0 || padd_out < 0 || step_in <= 0 || step_out <= 0)
    {
        return TINY_ERR_MATH_INVALID_PARAM;
    }
    // pad refers to the columns that are not used in the matrix operation
    // Prefer ESP-DSP only when all paddings are 0 and all steps are 1 (contiguous)
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    if (padd_in == 0 && padd_out == 0 && step_in == 1 && step_out == 1) {
        // dspm_addc_f32 performs addition; pass -C to implement subtraction-constant
        dspm_addc_f32(input, output, -C, rows, cols, 0, 0, 1, 1);
        return TINY_OK;
    }
#endif

    const int in_row_stride = cols + padd_in;
    const int out_row_stride = cols + padd_out;

    for (int row = 0; row < rows; row++) {
        int base_in = row * in_row_stride;
        int base_out = row * out_row_stride;

        for (int col = 0; col < cols; col++) {
            int idx_in = base_in + col * step_in;
            int idx_out = base_out + col * step_out;

            output[idx_out] = input[idx_in] - C;
        }
    }
    return TINY_OK;
}

/* MULTIPLICATION */

// matrix * matrix | float

/**
 * @name tiny_mat_mult_f32
 * @brief Multiplies two matrices of type float32.
 * @param A Pointer to the first input matrix.
 * @param B Pointer to the second input matrix.
 * @param C Pointer to the output matrix.
 * @param m Number of rows in the first matrix.
 * @param n Number of columns in the first matrix and rows in the second matrix.
 * @param k Number of columns in the second matrix.
 * @return Returns TINY_OK on success, or an error code on failure.
 * @note This function performs matrix multiplication with the specified padding and step sizes.
 * @note The function assumes that the input matrices are in row-major order.
 */
tiny_error_t tiny_mat_mult_f32(const float *A, const float *B, float *C, int m, int n, int k)
{
    if (NULL == A || NULL == B || NULL == C)
        return TINY_ERR_MATH_NULL_POINTER;
    if (m <= 0 || n <= 0 || k <= 0)
        return TINY_ERR_MATH_INVALID_PARAM;

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // Use the ESP-DSP library for optimized matrix multiplication
    dspm_mult_f32(A, B, C, m, n, k);
#else
    // C[i][j] = sum_{s=0}^{n-1} A[i][s] * B[s][j]
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            C[i * k + j] = 0.0f;
            for (int s = 0; s < n; s++)
            {
                C[i * k + j] += A[i * n + s] * B[s * k + j];
            }
        }
    }
#endif
    return TINY_OK;
}

// matrix * matrix | float with padding and step sizes
/**
 * @name tiny_mat_mult_ex_f32
 * @brief Multiplies two matrices of type float32 with padding and step sizes.
 * @param A Pointer to the first input matrix.
 * @param B Pointer to the second input matrix.
 * @param C Pointer to the output matrix.
 * @param A_rows Number of rows in the first matrix.
 * @param A_cols Number of columns in the first matrix and rows in the second matrix.
 * @param B_cols Number of columns in the second matrix.
 * @param A_padding Number of padding columns in the first matrix.
 * @param B_padding Number of padding columns in the second matrix.
 * @param C_padding Number of padding columns in the output matrix.
 * @return Returns TINY_OK on success, or an error code on failure.
 * @note This function performs matrix multiplication with the specified padding and step sizes.
 * @note The function assumes that the input matrices are in row-major order.
 */
tiny_error_t tiny_mat_mult_ex_f32(const float *A, const float *B, float *C, int A_rows, int A_cols, int B_cols, int A_padding, int B_padding, int C_padding)
{
    if (NULL == A || NULL == B || NULL == C)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }
    if (A_rows <= 0 || A_cols <= 0 || B_cols <= 0)
    {
        return TINY_ERR_MATH_INVALID_PARAM;
    }
    if (A_padding < 0 || B_padding < 0 || C_padding < 0)
    {
        return TINY_ERR_MATH_INVALID_PARAM;
    }
    // Prefer ESP-DSP only when paddings are zero (contiguous storage)
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    if (A_padding == 0 && B_padding == 0 && C_padding == 0) {
        dspm_mult_ex_f32(A, B, C, A_rows, A_cols, B_cols, 0, 0, 0);
        return TINY_OK;
    }
#endif

    // Matrix A(m,n), m - amount of rows, n - amount of columns
    // C(m,k) = A(m,n)*B(n,k)
    // C[i][j] = sum_{s=0}^{n-1} A[i][s] * B[s][j]
    const int A_step = A_cols + A_padding;
    const int B_step = B_cols + B_padding;
    const int C_step = B_cols + C_padding;

    for (int i = 0; i < A_rows; i++)
    {
        for (int j = 0; j < B_cols; j++)
        {
            float sum = 0.0f;
            for (int s = 0; s < A_cols; s++)
            {
                sum += A[i * A_step + s] * B[s * B_step + j];
            }
            C[i * C_step + j] = sum;
        }
    }
    return TINY_OK;
}

// matrix * constant | float
/**
 * @name tiny_mat_multc_f32
 * @brief Multiplies a matrix by a constant of type float32.
 * @param input Pointer to the input matrix.
 * @param output Pointer to the output matrix.
 * @param C Constant value to be multiplied with each element of the matrix.
 * @param rows Number of rows in the matrices.
 * @param cols Number of columns in the matrices.
 * @param padd_in Number of padding columns in the input matrix.
 * @param padd_out Number of padding columns in the output matrix.
 * @param step_in Step size for the input matrix.
 * @param step_out Step size for the output matrix.
 * @return Returns TINY_OK on success, or an error code on failure.
 * @note This function performs matrix multiplication with a constant with the specified padding and step sizes.
 * @note The function assumes that the input matrix is in row-major order.
 */
tiny_error_t tiny_mat_multc_f32(const float *input, float *output, float C, int rows, int cols, int padd_in, int padd_out, int step_in, int step_out)
{
    if (NULL == input || NULL == output)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }
    // paddings must be non-negative, steps must be at least 1.
    if (rows <= 0 || cols <= 0 || padd_in < 0 || padd_out < 0 || step_in <= 0 || step_out <= 0)
    {
        return TINY_ERR_MATH_INVALID_PARAM;
    }
    // pad refers to the columns that are not used in the matrix operation
    // Prefer ESP-DSP only when all paddings are 0 and all steps are 1 (contiguous)
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    if (padd_in == 0 && padd_out == 0 && step_in == 1 && step_out == 1) {
        dspm_mulc_f32(input, output, C, rows, cols, 0, 0, 1, 1);
        return TINY_OK;
    }
#endif

    const int in_row_stride = cols + padd_in;
    const int out_row_stride = cols + padd_out;

    for (int row = 0; row < rows; row++) {
        int base_in = row * in_row_stride;
        int base_out = row * out_row_stride;

        for (int col = 0; col < cols; col++) {
            int idx_in = base_in + col * step_in;
            int idx_out = base_out + col * step_out;

            output[idx_out] = input[idx_in] * C;
        }
    }
    return TINY_OK;
}
```