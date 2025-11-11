/**
 * @file node_timer.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file contains the function prototypes for the node_timer component.
 * @version 1.0
 * @date 2025-10-21
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

/* Dependencies */
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "node_led.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* Function Prototypes */

    /**
     * @brief       Initialize a high-precision timer (ESP_TIMER)
     * @param       tps: Timer period in microseconds (μs). For example, to execute the timer interrupt once every second,
     *                   set tps = 1s = 1000000μs.
     * @retval      None
     */
    void esptim_int_init(uint64_t tps);

    /**
     * @brief       Timer callback function
     * @param       arg: No arguments passed
     * @retval      None
     */
    void esptim_callback(void *arg);

#ifdef __cplusplus
}
#endif