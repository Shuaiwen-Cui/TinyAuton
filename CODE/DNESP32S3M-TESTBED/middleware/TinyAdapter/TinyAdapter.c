/**
 * @file TinyAdapter.C
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the source file for the TinyAdapter middleware.
 * @version 1.0
 * @date 2025-03-26
 *
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "TinyAdapter.h"

/* DEFINITIONS */

/* FUNCTIONS */
TinyTimeMark_t tiny_get_time(void)
{
    return esp_timer_get_time();
}
