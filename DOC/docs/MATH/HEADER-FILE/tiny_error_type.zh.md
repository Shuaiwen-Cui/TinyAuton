# 错误类型定义

!!! INFO
    该文件定义了一些计算中常见的错误类型，用于辅助判断错误原因。文档更新速度较慢，可能与实际代码不符，请以代码为准。

```c
/**
 * @file tiny_error_type.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief The configuration file for the tiny_math middleware.
 * @version 1.0
 * @date 2025-04-15
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

    /* TYPE DEFINITIONS */
    typedef int tiny_error_t; // Error type for the tiny_math middleware

/* MACROS */
/* Definitions for error constants. */
#define TINY_OK 0    /*!< tiny_err_t value indicating success (no error) */
#define TINY_FAIL -1 /*!< Generic tiny_err_t code indicating failure */

#define TINY_ERR_NO_MEM 0x101           /*!< Out of memory */
#define TINY_ERR_INVALID_ARG 0x102      /*!< Invalid argument */
#define TINY_ERR_INVALID_STATE 0x103    /*!< Invalid state */
#define TINY_ERR_INVALID_SIZE 0x104     /*!< Invalid size */
#define TINY_ERR_NOT_FOUND 0x105        /*!< Requested resource not found */
#define TINY_ERR_NOT_SUPPORTED 0x106    /*!< Operation or feature not supported */
#define TINY_ERR_TIMEOUT 0x107          /*!< Operation timed out */
#define TINY_ERR_INVALID_RESPONSE 0x108 /*!< Received response was invalid */
#define TINY_ERR_INVALID_CRC 0x109      /*!< CRC or checksum was invalid */
#define TINY_ERR_INVALID_VERSION 0x10A  /*!< Version was invalid */
#define TINY_ERR_INVALID_MAC 0x10B      /*!< MAC address was invalid */
#define TINY_ERR_NOT_FINISHED 0x10C     /*!< Operation has not fully completed */
#define TINY_ERR_NOT_ALLOWED 0x10D      /*!< Operation is not allowed */

#define TINY_ERR_WIFI_BASE 0x3000      /*!< Starting number of WiFi error codes */
#define TINY_ERR_MESH_BASE 0x4000      /*!< Starting number of MESH error codes */
#define TINY_ERR_FLASH_BASE 0x6000     /*!< Starting number of flash error codes */
#define TINY_ERR_HW_CRYPTO_BASE 0xc000 /*!< Starting number of HW cryptography module error codes */
#define TINY_ERR_MEMPROT_BASE 0xd000   /*!< Starting number of Memory Protection API error codes */

#define TINY_ERR_MATH_BASE 0x70000
#define TINY_ERR_MATH_INVALID_LENGTH (TINY_ERR_MATH_BASE + 1)
#define TINY_ERR_MATH_INVALID_PARAM (TINY_ERR_MATH_BASE + 2)
#define TINY_ERR_MATH_PARAM_OUTOFRANGE (TINY_ERR_MATH_BASE + 3)
#define TINY_ERR_MATH_UNINITIALIZED (TINY_ERR_MATH_BASE + 4)
#define TINY_ERR_MATH_REINITIALIZED (TINY_ERR_MATH_BASE + 5)
#define TINY_ERR_MATH_ARRAY_NOT_ALIGNED (TINY_ERR_MATH_BASE + 6)
#define TINY_ERR_MATH_NULL_POINTER (TINY_ERR_MATH_BASE + 7)
#define TINY_ERR_MATH_ZERO_DIVISION (TINY_ERR_MATH_BASE + 8)
#define TINY_ERR_MATH_NEGATIVE_SQRT (TINY_ERR_MATH_BASE + 9)

#define TINY_ERR_DSP_BASE 0x80000
#define TINY_ERR_DSP_INVALID_LENGTH (TINY_ERR_DSP_BASE + 1)
#define TINY_ERR_DSP_INVALID_PARAM (TINY_ERR_DSP_BASE + 2)
#define TINY_ERR_DSP_PARAM_OUTOFRANGE (TINY_ERR_DSP_BASE + 3)
#define TINY_ERR_DSP_UNINITIALIZED (TINY_ERR_DSP_BASE + 4)
#define TINY_ERR_DSP_REINITIALIZED (TINY_ERR_DSP_BASE + 5)
#define TINY_ERR_DSP_ARRAY_NOT_ALIGNED (TINY_ERR_DSP_BASE + 6)
#define TINY_ERR_DSP_NULL_POINTER (TINY_ERR_DSP_BASE + 7)
#define TINY_ERR_DSP_ZERO_DIVISION (TINY_ERR_DSP_BASE + 8)
#define TINY_ERR_DSP_NEGATIVE_SQRT (TINY_ERR_DSP_BASE + 9)
#define TINY_ERR_DSP_MISMATCH (TINY_ERR_DSP_BASE + 10)
#define TINY_ERR_DSP_INVALID_MODE (TINY_ERR_DSP_BASE + 11)
#define TINY_ERR_DSP_MEMORY_ALLOC (TINY_ERR_DSP_BASE + 12)

#ifdef __cplusplus
}
#endif

```