/**
 * @file node_led.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Led initialization and control.
 * @version 1.0
 * @date 2025-10-21
 *
 * @copyright Copyright (c) 2025
 */

#pragma once

/* Dependencies */
#include "driver/gpio.h"

/* GPIO Pin Definition */
#define LED_GPIO_PIN GPIO_NUM_1 /* GPIO port connected to LED */

/* GPIO States */
#define LED_PIN_RESET 0
#define LED_PIN_SET 1

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief       Initialize the LED
     * @param       None
     * @retval      None
     */
    void led_init(void);

    /**
     * @brief       Control the LED
     * @param       x: true for on, false for off
     * @retval      None
     */
    void led(bool x);

    /**
     * @brief       Toggle the LED
     * @param       None
     * @retval      None
     */
    void led_toggle(void);

#ifdef __cplusplus
}
#endif