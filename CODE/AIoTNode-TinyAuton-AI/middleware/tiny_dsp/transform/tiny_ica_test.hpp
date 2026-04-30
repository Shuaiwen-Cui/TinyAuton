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

#endif // __cplusplus

// C interface wrapper — placed OUTSIDE #ifdef __cplusplus so pure C code can also see these.
#ifdef __cplusplus
extern "C" {
#endif

void tiny_ica_test_basic(void);
void tiny_ica_test_sinusoidal(void);
void tiny_ica_test_nonlinearity(void);
void tiny_ica_test_reconstruction(void);
void tiny_ica_test_all(void);

#ifdef __cplusplus
}
#endif

