/**
 * @file node_rgb.h
 * @brief SK6812MINI-C (SK6812-class) one-wire RGB: low-level fill, clear, optional chase step.
 *
 * No FreeRTOS tasks or delays here; use vTaskDelay (etc.) in the application for timing and animation.
 */

#pragma once

#include "esp_err.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Data line GPIO */
#define NODE_RGB_GPIO_DEFAULT (3)

typedef struct {
    int gpio_num;
    int num_leds;
    uint8_t brightness; /**< 0–255 scale */
} node_rgb_config_t;

#define NODE_RGB_CONFIG_DEFAULT()           \
    {                                       \
        .gpio_num = NODE_RGB_GPIO_DEFAULT,  \
        .num_leds = 1,                      \
        .brightness = 96,                   \
    }

esp_err_t node_rgb_init(const node_rgb_config_t *cfg);
void node_rgb_deinit(void);
bool node_rgb_is_initialized(void);

void node_rgb_set_brightness(uint8_t v);

/** Solid R,G,B (0–255) on all LEDs, then refresh */
void node_rgb_rgb(uint8_t r, uint8_t g, uint8_t b);

/** Named color string (MQTT/CLI): trims ASCII space; English case-insensitive; UTF-8 Chinese aliases supported. */
esp_err_t node_rgb_str(const char *s);

void node_rgb_clear(void);

/**
 * Advance chase by one step (multi-LED: moving bright spot; single LED: cycles R→G→B internally).
 * Call rate is controlled by the application.
 */
void node_rgb_chase_step(uint8_t r, uint8_t g, uint8_t b);

/** Reset chase position (e.g. restart multi-LED chase from the beginning) */
void node_rgb_chase_reset(void);

#ifdef __cplusplus
}
#endif
