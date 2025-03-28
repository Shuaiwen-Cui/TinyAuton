/**
 * @file TinyAdapter.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the header file for the TinyAdapter middleware.
 * @version 1.0
 * @date 2025-03-26
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __TINYADAPTER_H__
#define __TINYADAPTER_H__

/* CONFIGURATIONS */

/* DEPENDENCIES */
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

/* DEFINITIONS */
typedef TickType_t TinyTimeMark_t;

/* FUNCTIONS */
TinyTimeMark_t tiny_get_time(void);

#endif /* __TINYADAPTER_H__ */