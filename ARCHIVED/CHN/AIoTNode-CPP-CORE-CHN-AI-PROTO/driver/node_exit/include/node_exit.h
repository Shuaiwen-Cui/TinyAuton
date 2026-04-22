/**
 * @file node_exit.h
 * @author
 * @brief This file is for the external interrupt initialization and configuration.
 * @version 1.0
 * @date 2025-10-21
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#include "esp_system.h"
#include "driver/gpio.h"
#include "node_led.h"

/* Pin definition */
#define BOOT_INT_GPIO_PIN GPIO_NUM_0

/* IO operation */
#define BOOT_EXIT gpio_get_level(BOOT_INT_GPIO_PIN)

#ifdef __cplusplus
extern "C" {
#endif

/* Function declarations */
/**
 * @brief       External interrupt initialization function
 * @param       None
 * @retval      None
 */
void exit_init(void); /* External interrupt initialization function */

#ifdef __cplusplus
}
#endif