/**
 * @file tiny_ica_test.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_ica | test | header
 * @version 1.0
 * @date 2025-04-30
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
#include "tiny_ica.hpp"

#ifdef __cplusplus

namespace tiny
{
    /**
     * @name tiny_ica_test_basic
     * @brief Basic test for ICA with simple synthetic signals
     */
    void tiny_ica_test_basic(void);

    /**
     * @name tiny_ica_test_sinusoidal
     * @brief Test ICA with sinusoidal source signals
     */
    void tiny_ica_test_sinusoidal(void);

    /**
     * @name tiny_ica_test_nonlinearity
     * @brief Test different nonlinearity functions
     */
    void tiny_ica_test_nonlinearity(void);

    /**
     * @name tiny_ica_test_reconstruction
     * @brief Test signal reconstruction from separated sources
     */
    void tiny_ica_test_reconstruction(void);

    /**
     * @name tiny_ica_test_all
     * @brief Run all ICA tests
     */
    void tiny_ica_test_all(void);

} // namespace tiny

// C interface wrapper for compatibility
extern "C"
{
    /**
     * @name tiny_ica_test_basic
     * @brief C interface wrapper for basic ICA test
     */
    void tiny_ica_test_basic(void);

    /**
     * @name tiny_ica_test_sinusoidal
     * @brief C interface wrapper for sinusoidal ICA test
     */
    void tiny_ica_test_sinusoidal(void);

    /**
     * @name tiny_ica_test_nonlinearity
     * @brief C interface wrapper for nonlinearity ICA test
     */
    void tiny_ica_test_nonlinearity(void);

    /**
     * @name tiny_ica_test_reconstruction
     * @brief C interface wrapper for reconstruction ICA test
     */
    void tiny_ica_test_reconstruction(void);

    /**
     * @name tiny_ica_test_all
     * @brief C interface wrapper for all ICA tests
     */
    void tiny_ica_test_all(void);
}

#endif // __cplusplus

