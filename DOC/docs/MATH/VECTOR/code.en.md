# CODE

## tiny_vec.h

```c
/**
 * @file tiny_vec.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the header file for the submodule vec of the tiny_math middleware.
 * @version 1.0
 * @date 2025-04-15
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __TINY__VEC__
#define __TINY__VEC__

/* DEPENDENCIES */
#include "tiny_math_config.h"

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
```

## tiny_vec.c

```c
/**
 * @file tiny_vec.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the source file for the submodule vec of the tiny_math middleware.
 * @version 1.0
 * @date 2025-04-15
 * @copyright Copyright (c) 2025
 *
 */

#include "tiny_vec.h"

// #ifdef __cplusplus

/* ADDITION */

// vector + vector | float

/**
 * @name tiny_vec_add_f32
 * @brief Adds two vectors element-wise.
 * @param input1 Pointer to the first input vector.
 * @param input2 Pointer to the second input vector.
 * @param output Pointer to the output vector.
 * @param len Length of the vectors.
 * @param step1 Step size for the first input vector.
 * @param step2 Step size for the second input vector.
 * @param step_out Step size for the output vector.
 * @return tiny_error_t Error code indicating success or failure.
 * @note This function performs element-wise addition of two vectors with specified step sizes, and the output is also specified with a step size.
 */
tiny_error_t tiny_vec_add_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out)
{
    if (NULL == input1 || NULL == input2 || NULL == output)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }
    if (len <= 0 || step1 <= 0 || step2 <= 0 || step_out <= 0)
    {
        return TINY_ERR_INVALID_ARG;
    }

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // Use the ESP-DSP library for optimized vector addition
    dsps_add_f32(input1, input2, output, len, step1, step2, step_out);
#else
    // Fallback to a simple loop for vector addition
    for (int i = 0; i < len; i++)
    {
        output[i * step_out] = input1[i * step1] + input2[i * step2];
    }

#endif

    return TINY_OK;
}

// vector + constant | float
/**
 * @name tiny_vec_addc_f32
 * @brief Adds a constant to each element of a vector.
 * @param input Pointer to the input vector.
 * @param output Pointer to the output vector.
 * @param len Length of the vectors.
 * @param C Constant value to be added.
 * @param step_in Step size for the input vector.
 * @param step_out Step size for the output vector.
 * @return tiny_error_t Error code indicating success or failure.
 * @note This function adds a constant value to each element of the input vector, with specified step sizes for both input and output vectors.
 */
tiny_error_t tiny_vec_addc_f32(const float *input, float *output, int len, float C, int step_in, int step_out)
{
    if (NULL == input || NULL == output)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }
    if (len <= 0 || step_in <= 0 || step_out <= 0)
    {
        return TINY_ERR_INVALID_ARG;
    }

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // Use the ESP-DSP library for optimized vector addition
    dsps_addc_f32(input, output, len, C, step_in, step_out);
#else
    // Fallback to a simple loop for vector addition
    for (int i = 0; i < len; i++)
    {
        output[i * step_out] = input[i * step_in] + C;
    }
#endif
    return TINY_OK;
}

/* SUBTRACTION */

// vector - vector | float
/**
 * @name tiny_vec_sub_f32
 * @brief Subtracts two vectors element-wise.
 * @param input1 Pointer to the first input vector.
 * @param input2 Pointer to the second input vector.
 * @param output Pointer to the output vector.
 * @param len Length of the vectors.
 * @param step1 Step size for the first input vector.
 * @param step2 Step size for the second input vector.
 * @param step_out Step size for the output vector.
 * @return tiny_error_t Error code indicating success or failure.
 * @note This function performs element-wise subtraction of two vectors with specified step sizes, and the output is also specified with a step size.
 */
tiny_error_t tiny_vec_sub_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out)
{
    if (NULL == input1 || NULL == input2 || NULL == output)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }
    if (len <= 0 || step1 <= 0 || step2 <= 0 || step_out <= 0)
    {
        return TINY_ERR_INVALID_ARG;
    }
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // Use the ESP-DSP library for optimized vector subtraction
    dsps_sub_f32(input1, input2, output, len, step1, step2, step_out);
#else
    // Fallback to a simple loop for vector subtraction
    for (int i = 0; i < len; i++)
    {
        output[i * step_out] = input1[i * step1] - input2[i * step2];
    }
#endif
    return TINY_OK;
}

// vector - constant (add -c) | float
/**
 * @name tiny_vec_subc_f32
 * @brief Subtracts a constant from each element of a vector.
 * @param input Pointer to the input vector.
 * @param output Pointer to the output vector.
 * @param len Length of the vectors.
 * @param C Constant value to be subtracted.
 * @param step_in Step size for the input vector.
 * @param step_out Step size for the output vector.
 * @return tiny_error_t Error code indicating success or failure.
 * @note This function subtracts a constant value from each element of the input vector, with specified step sizes for both input and output vectors.
 */
tiny_error_t tiny_vec_subc_f32(const float *input, float *output, int len, float C, int step_in, int step_out)
{
    if (NULL == input || NULL == output)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }
    if (len <= 0 || step_in <= 0 || step_out <= 0)
    {
        return TINY_ERR_INVALID_ARG;
    }
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // Use the ESP-DSP library for optimized vector subtraction
    dsps_addc_f32(input, output, len, -C, step_in, step_out);
#else
    // Fallback to a simple loop for vector subtraction
    for (int i = 0; i < len; i++)
    {
        output[i * step_out] = input[i * step_in] - C;
    }
#endif
    return TINY_OK;
}

/* MULTIPLICATION */

// vector * vector (elementwise) | float
/**
 * @name tiny_vec_mul_f32
 * @brief Multiplies two vectors element-wise.
 * @param input1 Pointer to the first input vector.
 * @param input2 Pointer to the second input vector.
 * @param output Pointer to the output vector.
 * @param len Length of the vectors.
 * @param step1 Step size for the first input vector.
 * @param step2 Step size for the second input vector.
 * @param step_out Step size for the output vector.
 * @return tiny_error_t Error code indicating success or failure.
 * @note This function performs element-wise multiplication of two vectors with specified step sizes, and the output is also specified with a step size.
 */
tiny_error_t tiny_vec_mul_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out)
{
    if (NULL == input1 || NULL == input2 || NULL == output)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }
    if (len <= 0 || step1 <= 0 || step2 <= 0 || step_out <= 0)
    {
        return TINY_ERR_INVALID_ARG;
    }
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // Use the ESP-DSP library for optimized vector multiplication
    dsps_mul_f32(input1, input2, output, len, step1, step2, step_out);
#else
    // Fallback to a simple loop for vector multiplication
    for (int i = 0; i < len; i++)
    {
        output[i * step_out] = input1[i * step1] * input2[i * step2];
    }
#endif
        return TINY_OK;
}

// vector * constant | float
/**
 * @name tiny_vec_mulc_f32
 * @brief Multiplies each element of a vector by a constant.
 * @param input Pointer to the input vector.
 * @param output Pointer to the output vector.
 * @param len Length of the vectors.
 * @param C Constant value to be multiplied.
 * @param step_in Step size for the input vector.
 * @param step_out Step size for the output vector.
 * @return tiny_error_t Error code indicating success or failure.
 * @note This function multiplies each element of the input vector by a constant value, with specified step sizes for both input and output vectors.
 */
tiny_error_t tiny_vec_mulc_f32(const float *input, float *output, int len, float C, int step_in, int step_out)
{
    if (NULL == input || NULL == output)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }
    if (len <= 0 || step_in <= 0 || step_out <= 0)
    {
        return TINY_ERR_INVALID_ARG;
    }
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // Use the ESP-DSP library for optimized vector multiplication
    dsps_mulc_f32(input, output, len, C, step_in, step_out);
#else
    // Fallback to a simple loop for vector multiplication
    for (int i = 0; i < len; i++)
    {
        output[i * step_out] = input[i * step_in] * C;
    }
#endif
    return TINY_OK;
}

/* DIVISION */

// vector / vector (elementwise) | float
/**
 * @name tiny_vec_div_f32
 * @brief Divides one vector by another element-wise.
 * @param input1 Pointer to the numerator vector.
 * @param input2 Pointer to the denominator vector.
 * @param output Pointer to the output vector.
 * @param len Length of the vectors.
 * @param step1 Step size for the numerator vector.
 * @param step2 Step size for the denominator vector.
 * @param step_out Step size for the output vector.
 * @param allow_divide_by_zero Whether to safely handle zero denominators (true: set output to 0; false: return error if any zero is found).
 * @return tiny_error_t Error code indicating success or failure.
 * @note This function performs element-wise division with specified step sizes.
 *       If allow_divide_by_zero is false, the function will first scan for zero denominators and return an error immediately if any are found.
 */
tiny_error_t tiny_vec_div_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out, bool allow_divide_by_zero)
{
    if (NULL == input1 || NULL == input2 || NULL == output)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }
    if (len <= 0 || step1 <= 0 || step2 <= 0 || step_out <= 0)
    {
        return TINY_ERR_INVALID_ARG;
    }

    const float epsilon = TINY_MATH_MIN_DENOMINATOR;

    // Step 1: Pre-check for zero denominators if not allowed
    if (!allow_divide_by_zero)
    {
        for (int i = 0; i < len; i++)
        {
            if (fabsf(input2[i * step2]) < epsilon)
            {
                return TINY_ERR_MATH_ZERO_DIVISION;
            }
        }
    }

    // Step 2: Perform element-wise division
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    for (int i = 0; i < len; i++)
    {
        float denom = input2[i * step2];
        float numer = input1[i * step1];
        if (fabsf(denom) < epsilon)
        {
            output[i * step_out] = allow_divide_by_zero ? 0.0f : numer / denom; // fallback if user forced it
        }
        else
        {
            output[i * step_out] = numer / denom;
        }
    }
#else
    for (int i = 0; i < len; i++)
    {
        float denom = input2[i * step2];
        float numer = input1[i * step1];
        if (fabsf(denom) < epsilon)
        {
            output[i * step_out] = allow_divide_by_zero ? 0.0f : numer / denom;
        }
        else
        {
            output[i * step_out] = numer / denom;
        }
    }
#endif

    return TINY_OK;
}

// vector / constant | float
/**
 * @name tiny_vec_divc_f32
 * @brief Divides each element of a vector by a constant using multiplication for performance.
 * @param input Pointer to the input vector.
 * @param output Pointer to the output vector.
 * @param len Length of the vectors.
 * @param C Constant value to divide by.
 * @param step_in Step size for the input vector.
 * @param step_out Step size for the output vector.
 * @param allow_divide_by_zero Whether to safely handle zero constant (true: set output to 0; false: return error if C is near zero).
 * @return tiny_error_t Error code indicating success or failure.
 * @note This function divides each element of the input vector by a constant using multiplication for performance.
 *       If allow_divide_by_zero is false and C is near zero, the function returns an error.
 *       Otherwise, 1/C is precomputed and used as a multiplier.
 */
tiny_error_t tiny_vec_divc_f32(const float *input, float *output, int len, float C, int step_in, int step_out, bool allow_divide_by_zero)
{
    if (NULL == input || NULL == output)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }

    if (len <= 0 || step_in <= 0 || step_out <= 0)
    {
        return TINY_ERR_INVALID_ARG;
    }

    const float epsilon = TINY_MATH_MIN_DENOMINATOR;

    // Step 1: Handle constant C
    if (fabsf(C) < epsilon)
    {
        if (!allow_divide_by_zero)
        {
            return TINY_ERR_MATH_ZERO_DIVISION;
        }

        // Safe fallback: set all outputs to 0
        for (int i = 0; i < len; i++)
        {
            output[i * step_out] = 0.0f;
        }
        return TINY_OK;
    }

    // Step 2: Use 1/C for performance
    float invC = 1.0f / C;

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    dsps_mulc_f32(input, output, len, invC, step_in, step_out);
#else
    for (int i = 0; i < len; i++)
    {
        output[i * step_out] = input[i * step_in] * invC;
    }
#endif

    return TINY_OK;
}

/* SQUARE ROOT */

// vector ^ 0.5 (sqrt-based)| float
/**
 * @name tiny_vec_sqrt_f32
 * @brief Computes the square root of each element in a vector using standard library sqrtf.
 * @param input Pointer to the input vector.
 * @param output Pointer to the output vector.
 * @param len Length of the vectors.
 * @return tiny_error_t Error code indicating success or failure.
 * @note This function provides accurate results using math library sqrtf().
 *       It returns TINY_ERR_MATH_NEGATIVE_SQRT immediately if any element is negative.
 */
tiny_error_t tiny_vec_sqrt_f32(const float *input, float *output, int len)
{
    if (NULL == input || NULL == output)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }

    if (len <= 0)
    {
        return TINY_ERR_INVALID_ARG;
    }

    for (int i = 0; i < len; i++)
    {
        float x = input[i];
        if (x < 0.0f)
        {
            return TINY_ERR_MATH_NEGATIVE_SQRT;
        }

        output[i] = sqrtf(x);  // high-precision sqrt
    }

    return TINY_OK;
}

// single value sqrt
/**
 * @name tiny_sqrtf_f32
 * @brief Computes the square root of a single float value using bit manipulation.
 * @param f Input float value.
 * @return Square root of the input value.
 * @note This function uses bit manipulation to compute the square root of a float value.
 *       It returns 0.0f for negative inputs to prevent sqrt of negative values.
 */
inline float tiny_sqrtf_f32(float f)
{
    if (f < 0.0f) {
        return 0.0f;  // Prevent sqrt of negative values
    }

    int result;
    int *f_ptr = (int *)&f;
    result = 0x1fbb4000 + (*f_ptr >> 1);
    const int *p = &result;
    float *f_result = (float *)p;
    return *f_result;
}

// vector ^ 0.5 (sqrtf-based)| float
/**
 * @name tiny_vec_sqrtf_f32
 * @brief Computes the square root of each element in a vector.
 * @param input Pointer to the input vector.
 * @param output Pointer to the output vector.
 * @param len Length of the vectors.
 * @return tiny_error_t Error code indicating success or failure.
 * @note This function computes the square root of each element in the input vector and stores the result in the output vector.
 *       It returns TINY_ERR_MATH_NEGATIVE_SQRT immediately if any element is negative.
 */
tiny_error_t tiny_vec_sqrtf_f32(const float *input, float *output, int len)
{
    if (NULL == input || NULL == output)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }

    if (len <= 0)
    {
        return TINY_ERR_INVALID_ARG;
    }

    for (int i = 0; i < len; i++)
    {
        float x = input[i];
        if (x < 0.0f)
        {
            return TINY_ERR_MATH_NEGATIVE_SQRT;
        }

        output[i] = tiny_sqrtf_f32(x);
    }

    return TINY_OK;
}

// vector ^ -0.5 (sqrt-based) | float
/**
 * @name tiny_vec_inv_sqrt_f32
 * @brief Computes the inverse square root of each element in a vector using standard sqrtf().
 * @param input Pointer to the input vector.
 * @param output Pointer to the output vector.
 * @param len Length of the vectors.
 * @return tiny_error_t Error code indicating success or failure.
 * @note This function provides accurate inverse square root results using 1.0f / sqrtf(x).
 *       It returns TINY_ERR_NOT_ALLOWED immediately if any element is less than TINY_MATH_MIN_POSITIVE_INPUT_F32.
 */
tiny_error_t tiny_vec_inv_sqrt_f32(const float *input, float *output, int len)
{
    if (NULL == input || NULL == output)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }

    if (len <= 0)
    {
        return TINY_ERR_INVALID_ARG;
    }

    for (int i = 0; i < len; i++)
    {
        float x = input[i];
        if (x < TINY_MATH_MIN_POSITIVE_INPUT_F32)
        {
            return TINY_ERR_NOT_ALLOWED;
        }

        output[i] = 1.0f / sqrtf(x);  // Accurate inverse square root
    }

    return TINY_OK;
}


// single value inv sqrt
/**
 * @name tiny_inverted_sqrtf_f32
 * @brief Computes the inverse square root of a single float value using bit manipulation.
 * @param data Input float value.
 * @return Inverse square root of the input value.
 * @note This function uses bit manipulation to compute the inverse square root of a float value.
 *       It returns 0.0f for negative or near-zero inputs to prevent division by zero.
 */
float tiny_inverted_sqrtf_f32(float data)
{
    if (data < TINY_MATH_MIN_POSITIVE_INPUT_F32) {
        return 0.0f;  // Avoid division by near-zero or zero
    }

    const float x2 = data * 0.5F;
    const float threehalfs = 1.5F;

    union {
        float f;
        uint32_t i;
    } conv = {data};

    conv.i  = 0x5f3759df - (conv.i >> 1);
    conv.f  = conv.f * (threehalfs - (x2 * conv.f * conv.f));

    return conv.f;
}

// vector ^ -0.5 (sqrtf-based) | float
/**
 * @name tiny_vec_inv_sqrtf_f32
 * @brief Computes the inverse square root of each element in a vector.
 * @param input Pointer to the input vector.
 * @param output Pointer to the output vector.
 * @param len Length of the vectors.
 * @return tiny_error_t Error code indicating success or failure.
 * @note This function computes the inverse square root of each element in the input vector and stores the result in the output vector.
 *       If any element is less than TINY_MATH_MIN_POSITIVE_INPUT_F32, the function returns TINY_ERR_NOT_ALLOWED.
 */
tiny_error_t tiny_vec_inv_sqrtf_f32(const float *input, float *output, int len)
{
    if (NULL == input || NULL == output)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }

    if (len <= 0)
    {
        return TINY_ERR_INVALID_ARG;
    }

    for (int i = 0; i < len; i++)
    {
        float x = input[i];
        if (x < TINY_MATH_MIN_POSITIVE_INPUT_F32)
        {
            return TINY_ERR_NOT_ALLOWED;
        }

        output[i] = tiny_inverted_sqrtf_f32(x);
    }

    return TINY_OK;
}

/* DOT PRODUCT */

// vector * vector (dot product) | float
/**
 * @name tiny_vec_dotprod_f32
 * @brief Computes the dot product of two vectors.
 * @param src1 Pointer to the first input vector.
 * @param src2 Pointer to the second input vector.
 * @param dest Pointer to the output scalar result.
 * @param len Length of the vectors.
 * @return tiny_error_t Error code indicating success or failure.
 * @note This function computes the dot product of two vectors and stores the result in a single float value.
 *       It returns TINY_ERR_MATH_NULL_POINTER if any pointer is NULL.
 *       The function uses the ESP-DSP library for optimized computation.
 */
tiny_error_t tiny_vec_dotprod_f32(const float *src1, const float *src2, float *dest, int len)
{
    if (NULL == src1 || NULL == src2 || NULL == dest)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }

    if (len <= 0)
    {
        return TINY_ERR_INVALID_ARG;
    }
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // Use the ESP-DSP library for optimized dot product
    dsps_dotprod_f32(src1, src2, dest, len);
#else
    // Fallback to a simple loop for dot product
    float acc = 0.0f;
    for (int i = 0; i < len; i++)
    {
        acc += src1[i] * src2[i];
    }
    *dest = acc;
#endif
    return TINY_OK;
}

// vector * vector (dot product - step support) | float
/**
 * @name tiny_vec_dotprode_f32
 * @brief Computes the dot product of two vectors with step support.
 * @param src1 Pointer to the first input vector.
 * @param src2 Pointer to the second input vector.
 * @param dest Pointer to the output scalar result.
 * @param len Length of the vectors.
 * @param step1 Step size for the first input vector.
 * @param step2 Step size for the second input vector.
 * @return tiny_error_t Error code indicating success or failure.
 * @note This function computes the dot product of two vectors with specified step sizes and stores the result in a single float value.
 */
tiny_error_t tiny_vec_dotprode_f32(const float *src1, const float *src2, float *dest, int len, int step1, int step2)
{
    if (NULL == src1 || NULL == src2 || NULL == dest)
    {
        return TINY_ERR_MATH_NULL_POINTER;
    }

    if (len <= 0 || step1 <= 0 || step2 <= 0)
    {
        return TINY_ERR_INVALID_ARG;
    }
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // Use the ESP-DSP library for optimized dot product with step support
    dsps_dotprode_f32(src1, src2, dest, len, step1, step2);
#else
    // Fallback to a simple loop for dot product with step support
    float acc = 0.0f;
    for (int i = 0; i < len; i++)
    {
        acc += src1[i * step1] * src2[i * step2];
    }
    *dest = acc;
#endif
    return TINY_OK;
}
```