/**
 * @file TinyAdapter.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the header file for the TinyAdapter middleware.
 * @version 1.0
 * @date 2025-03-26
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __TINYADAPTER_H__
#define __TINYADAPTER_H__

/* CONFIGURATIONS */

// MCU type (ESP32, STM32 only one can be defined)
#define ESP32
// #define STM32

/* DEPENDENCIES */
#ifdef ESP32
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#endif

/* DEFINITIONS */
#ifdef ESP32
typedef TickType_t TinyTimeMark_t;
#endif


#endif /* __TINYADAPTER_H__ */