/**
 * @file exit.c
 * @author
 * @brief This file is for the external interrupt initialization and configuration.
 * @version 1.0
 * @date 2024-11-17
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "exit.h"

/**
 * @brief       External interrupt service routine
 * @param       arg: Interrupt pin number
 * @note        IRAM_ATTR: The IRAM_ATTR attribute is used to store the interrupt handler in internal RAM to reduce latency
 * @retval      None
 */
static void IRAM_ATTR exit_gpio_isr_handler(void *arg)
{
    uint32_t gpio_num = (uint32_t)arg;

    if (gpio_num == BOOT_INT_GPIO_PIN)
    {
        led_toggle();
    }
}

/**
 * @brief       External interrupt initialization function
 * @param       None
 * @retval      None
 */
void exit_init(void)
{
    gpio_config_t gpio_init_struct;

    /* Configure BOOT pin and external interrupt */
    gpio_init_struct.mode = GPIO_MODE_INPUT;                   /* Set as input mode */
    gpio_init_struct.pull_up_en = GPIO_PULLUP_ENABLE;          /* Enable pull-up */
    gpio_init_struct.pull_down_en = GPIO_PULLDOWN_DISABLE;     /* Disable pull-down */
    gpio_init_struct.intr_type = GPIO_INTR_NEGEDGE;            /* Trigger on falling edge */
    gpio_init_struct.pin_bit_mask = 1ull << BOOT_INT_GPIO_PIN; /* Configure BOOT key pin */
    gpio_config(&gpio_init_struct);                            /* Apply configuration */

    /* Register interrupt service */
    gpio_install_isr_service(0);

    /* Set GPIO interrupt callback function */
    gpio_isr_handler_add(BOOT_INT_GPIO_PIN, exit_gpio_isr_handler, (void *)BOOT_INT_GPIO_PIN);

    /* Enable GPIO interrupt */
    gpio_intr_enable(BOOT_INT_GPIO_PIN);
}