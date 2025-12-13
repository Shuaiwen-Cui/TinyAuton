/**
 * @file tiny_ai_sine_regression_test.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Sine function regression test for tiny_ai library
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 *
 * @details
 * This module provides a test for sine function regression using neural networks.
 * It generates synthetic data (sine + white noise), trains a model, and compares
 * predictions with ground truth values.
 */

#pragma once

#include "tiny_ai.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Run sine regression test using MLP model
 * 
 * @return tiny_error_t TINY_OK on success, error code on failure
 */
tiny_error_t tiny_ai_test_sine_regression_mlp(void);

/**
 * @brief Run all sine regression tests
 * 
 * @return tiny_error_t TINY_OK on success, error code on failure
 */
tiny_error_t tiny_ai_sine_regression_test_all(void);

#ifdef __cplusplus
}
#endif

