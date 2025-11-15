/**
 * @file tiny_dwt_test.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_dwt | test | header
 * @version 1.0
 * @date 2025-04-30
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
#include "tiny_dwt.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tiny_dwt_test(void);

/**
 * @name tiny_dwt_multilevel_test
 * @brief Unit test for multi-level DWT and inverse DWT
 */
void tiny_dwt_test_multilevel(void);

#ifdef __cplusplus
}
#endif
