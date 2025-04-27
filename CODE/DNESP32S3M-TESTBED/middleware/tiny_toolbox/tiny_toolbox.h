/**
 * @file tiny_toolbox.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the header file for the tiny_toolbox middleware.
 * @version 1.0
 * @date 2025-03-26
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __TINY_TOOLBOX__
#define __TINY_TOOLBOX__

#ifdef __cplusplus
extern "C"
{
#endif

/* DEPENDENCIES */
// system
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

// customized drivers
#include "esp_rtc.h"

/* SUBMODULES */
#include "tiny_time.h" // Time

#ifdef __cplusplus
}
#endif

#endif /* __TINY_TOOLBOX__ */