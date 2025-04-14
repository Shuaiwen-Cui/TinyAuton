/**
 * @file TinyToolbox.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the header file for the TinyToolbox middleware.
 * @version 1.0
 * @date 2025-03-26
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __TINYTOOLBOX__
#define __TINYTOOLBOX__

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
#include "esp_dsp.h"

// customized drivers
#include "esp_rtc.h"

// submodules
#include "TinyTime.h" // Time

#ifdef __cplusplus
}
#endif

#endif /* __TINYTOOLBOX__ */