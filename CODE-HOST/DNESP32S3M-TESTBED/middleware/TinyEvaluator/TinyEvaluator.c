/**
 * @file TinyEvaluator.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the source file for the TinyEvaluator middleware.
 * @version 1.0
 * @date 2025-03-26
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <stdio.h>
#include "TinyEvaluator.h"

TinyTimeMark_t tiny_get_time(void)
{
    return esp_timer_get_time();
}