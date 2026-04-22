/**
 * @file node_led.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Led initialization and control.
 * @version 1.1
 * @date 2025-10-21
 *
 * @copyright Copyright (c) 2025
 */

/* Dependencies */
#include "node_led.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief       Initialize the LED
     * @param       None
     * @retval      None
     */
    void led_init(void)
    {
        gpio_config_t gpio_init_struct = {0};

        gpio_init_struct.intr_type = GPIO_INTR_DISABLE;        /* Disable GPIO interrupt */
        gpio_init_struct.mode = GPIO_MODE_INPUT_OUTPUT;        /* Set GPIO mode to input-output */
        gpio_init_struct.pull_up_en = GPIO_PULLUP_ENABLE;      /* Enable pull-up resistor */
        gpio_init_struct.pull_down_en = GPIO_PULLDOWN_DISABLE; /* Disable pull-down resistor */
        gpio_init_struct.pin_bit_mask = 1ull
                                        << LED_GPIO_PIN; /* Set pin bit mask for the configured pin */
        gpio_config(&gpio_init_struct);                  /* Configure GPIO */

        led(true); /* Turn on the LED */
    }

    /**
     * @brief       Control the LED
     * @param       x: true for on, false for off
     * @retval      None
     */
    void led(bool x)
    {
        if (x)
        {
            gpio_set_level(LED_GPIO_PIN, LED_PIN_RESET);
        }
        else
        {
            gpio_set_level(LED_GPIO_PIN, LED_PIN_SET);
        }
    }

    /**
     * @brief       Toggle the LED
     * @param       None
     * @retval      None
     */
    void led_toggle(void)
    {
        gpio_set_level(LED_GPIO_PIN, !gpio_get_level(LED_GPIO_PIN));
    }

#ifdef __cplusplus
}
#endif
