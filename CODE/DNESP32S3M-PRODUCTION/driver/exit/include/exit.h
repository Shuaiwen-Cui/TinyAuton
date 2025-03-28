/**
 * @file exit.h
 * @author
 * @brief This file is for the external interrupt initialization and configuration.
 * @version 1.0
 * @date 2024-11-17
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __EXIT_H__
#define __EXIT_H__

#include "esp_system.h"
#include "driver/gpio.h"
#include "led.h"

/* Pin definition */
#define BOOT_INT_GPIO_PIN GPIO_NUM_0

/* IO operation */
#define BOOT_EXIT gpio_get_level(BOOT_INT_GPIO_PIN)

/* Function declarations */
/**
 * @brief       External interrupt initialization function
 * @param       None
 * @retval      None
 */
void exit_init(void); /* External interrupt initialization function */

#endif