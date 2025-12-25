/**
 * @file tiny_ai_integration_test.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Integration tests for complete training pipeline
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 *
 * @details
 * This module provides end-to-end integration tests for the tiny_ai library.
 * Tests verify the complete training pipeline including:
 * - Model creation (FC layers + activations)
 * - Forward and backward propagation
 * - Loss computation
 * - Optimizer updates
 * - Full training loop
 */

#pragma once

#include "tiny_ai_config.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Run all integration tests
 * @return tiny_error_t TINY_OK if all tests pass
 */
tiny_error_t tiny_ai_integration_test_all(void);

#ifdef __cplusplus
}
#endif

