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

#ifndef __TINY_MAT__
#define __TINY_MAT__

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

#endif /* __TINY__MAT__ */
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
            printf("%10.6f ", mat[i + j]); // padding not considered
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
    if (rows <= 0 || cols <= 0 || padd1 < 0 || padd2 < 0 || padd_out < 0 || step1 <= 0 || step2 <= 0 || step_out <= 0)
    {
        return TINY_ERR_MATH_INVALID_PARAM;
    }

    // pad refers to the columns that are not used in the matrix operation
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // Use the ESP-DSP library for optimized matrix addition
    dspm_add_f32(input1, input2, output, rows, cols, padd1, padd2, padd_out, step1, step2, step_out);
#else
    const int ptr_input1_step = cols + padd1;
    const int ptr_input2_step = cols + padd2;
    const int ptr_output_step = cols + padd_out;
    float *ptr_input1 = (float *)input1;
    float *ptr_input2 = (float *)input2;

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            output[col * step_out] = ptr_input1[col * step1] + ptr_input2[col * step2];
        }
        ptr_input1 += ptr_input1_step; // move to the next row of input1
        ptr_input2 += ptr_input2_step; // move to the next row of input2
        output += ptr_output_step;     // move to the next row of output
    }
#endif
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
    if (rows <= 0 || cols <= 0 || padd_in < 0 || padd_out < 0 || step_in <= 0 || step_out <= 0)
    {
        return TINY_ERR_MATH_INVALID_PARAM;
    }
    // pad refers to the columns that are not used in the matrix operation
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // Use the ESP-DSP library for optimized matrix addition with a constant
    dspm_addc_f32(input, output, C, rows, cols, padd_in, padd_out, step_in, step_out);
#else
    const int ptr_input_step = cols + padd_in;
    const int ptr_output_step = cols + padd_out;
    float *ptr_input = (float *)input;

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            output[col * step_out] = ptr_input[col * step_in] + C;
        }
        ptr_input += ptr_input_step; // move to the next row of input
        output += ptr_output_step;   // move to the next row of output
    }
#endif
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
    if (rows <= 0 || cols <= 0 || padd1 < 0 || padd2 < 0 || padd_out < 0 || step1 <= 0 || step2 <= 0 || step_out <= 0)
    {
        return TINY_ERR_MATH_INVALID_PARAM;
    }
    // pad refers to the columns that are not used in the matrix operation
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // Use the ESP-DSP library for optimized matrix subtraction
    dspm_sub_f32(input1, input2, output, rows, cols, padd1, padd2, padd_out, step1, step2, step_out);
#else
    const int ptr_input1_step = cols + padd1;
    const int ptr_input2_step = cols + padd2;
    const int ptr_output_step = cols + padd_out;
    float *ptr_input1 = (float *)input1;
    float *ptr_input2 = (float *)input2;

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            output[col * step_out] = ptr_input1[col * step1] - ptr_input2[col * step2];
        }
        ptr_input1 += ptr_input1_step; // move to the next row of input1
        ptr_input2 += ptr_input2_step; // move to the next row of input2
        output += ptr_output_step;     // move to the next row of output
    }
#endif
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
    if (rows <= 0 || cols <= 0 || padd_in < 0 || padd_out < 0 || step_in <= 0 || step_out <= 0)
    {
        return TINY_ERR_MATH_INVALID_PARAM;
    }
    // pad refers to the columns that are not used in the matrix operation
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // Use the ESP-DSP library for optimized matrix subtraction with a constant
    dspm_addc_f32(input, output, -C, rows, cols, padd_in, padd_out, step_in, step_out);
#else
    const int ptr_input_step = cols + padd_in;
    const int ptr_output_step = cols + padd_out;
    float *ptr_input = (float *)input;

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            output[col * step_out] = ptr_input[col * step_in] - C;
        }
        ptr_input += ptr_input_step; // move to the next row of input
        output += ptr_output_step;   // move to the next row of output
    }
#endif
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
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // Use the ESP-DSP library for optimized matrix multiplication
    dspm_mult_ex_f32(A, B, C, A_rows, A_cols, B_cols, A_padding, B_padding, C_padding);
#else
    // Matrix A(m,n), m - amount or rows, n - amount of columns
    // C(m,k) = A(m,n)*B(n,k)
    // c(i * c_step,j) = sum(a(i * a_step,s)*b(s * b_step,j)) , s=1..n
    const int A_step = A_cols + A_padding;
    const int B_step = B_cols + B_padding;
    const int C_step = B_cols + C_padding;

    for (int i = 0; i < A_rows; i++)
    {
        for (int j = 0; j < B_cols; j++)
        {
            C[i * C_step + j] = A[i * A_step] * B[j];
            for (int s = 1; s < A_cols; s++)
            {
                C[i * C_step + j] += A[i * A_step + s] * B[s * B_step + j];
            }
        }
    }
#endif
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
    if (rows <= 0 || cols <= 0 || padd_in < 0 || padd_out < 0 || step_in <= 0 || step_out <= 0)
    {
        return TINY_ERR_MATH_INVALID_PARAM;
    }
    // pad refers to the columns that are not used in the matrix operation
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // Use the ESP-DSP library for optimized matrix multiplication with a constant
    dspm_mulc_f32(input, output, C, rows, cols, padd_in, padd_out, step_in, step_out);
#else
    const int ptr_input_step = cols + padd_in;
    const int ptr_output_step = cols + padd_out;
    float *ptr_input = (float *)input;

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            output[col * step_out] = ptr_input[col * step_in] * C;
        }
        ptr_input += ptr_input_step; // move to the next row of input
        output += ptr_output_step;   // move to the next row of output
    }
#endif
    return TINY_OK;
}
```