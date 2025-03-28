/**
 * @file tim.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file contains the implementation of the TIM component.
 * @version 1.0
 * @date 2024-11-18
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "tim.h"

/**
 * @brief       Initialize a high-precision timer (ESP_TIMER)
 * @param       tps: Timer period in microseconds (μs). For example, to execute the timer interrupt once every second,
 *                   set tps = 1s = 1000000μs.
 * @retval      None
 */
void esptim_int_init(uint64_t tps)
{
    esp_timer_handle_t esp_tim_handle; /* Timer callback function handle */

    /* Define a timer configuration structure */
    esp_timer_create_args_t tim_periodic_arg = {
        .callback = &esptim_callback, /* Set the callback function */
        .arg = NULL,                  /* No arguments passed */
    };

    esp_timer_create(&tim_periodic_arg, &esp_tim_handle); /* Create a timer event */
    esp_timer_start_periodic(esp_tim_handle, tps);        /* Trigger periodically based on the timer period */
}

/**
 * @brief       Timer callback function
 * @param       arg: No arguments passed
 * @retval      None
 */
void esptim_callback(void *arg)
{
    led_toggle();
}