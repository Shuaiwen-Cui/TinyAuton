/**
 * @file measurement.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the source file for the measurement module.
 * @version 1.0
 * @date 2025-04-01
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "measurement.h"
#include <stdio.h>

/* VARIABLES */
static const char *TAG = "TinySHM-Measurement"; // Tag for logging

static uint8_t mpu6050_deviceid;    // MPU6050 Device ID
static mpu6050_acce_value_t acce;   // MPU6050 Accelerometer Value
static mpu6050_gyro_value_t gyro;   // MPU6050 Gyroscope Value
static mpu6050_temp_value_t temp;   // MPU6050 Temperature Value
static complimentary_angle_t angle; // Complimentary Angle

/* DEFINITION */
struct SenseConfig
{
    int sample_rate;
    int sample_duration;
    bool printout; // not recommened for high sample rate as it will slow down the process
};

struct SenseConfig sense_config = {
    .sample_rate = 100,   // Sample rate in Hz
    .sample_duration = 10 // Sample duration in seconds
};



/* FUNCTIONS */
static bool IRAM_ATTR on_timer_callback(gptimer_handle_t timer, const gptimer_alarm_event_data_t *edata, void *user_ctx)
{

    static int counter = 0;
    counter++;
    if (counter % 1000 == 0) {
        printf("call back %d\n", counter);
    }

    return true; // 返回 true 表示自动重新加载，继续定时
}

gptimer_handle_t gptimer = NULL;
gptimer_config_t config = {
    .clk_src = GPTIMER_CLK_SRC_DEFAULT,
    .direction = GPTIMER_COUNT_UP,
    .resolution_hz = 1000000, // resolution：1MHz，即1 tick = 1us
};
gptimer_event_callbacks_t cbs = {
    .on_alarm = on_timer_callback,
};
// set the time interval
gptimer_alarm_config_t alarm_config = {
    .alarm_count = 500, // alarm interval as 100us i.e. 0.1ms
    .reload_count = 0,
    .flags.auto_reload_on_alarm = true,
};

void sense_test(void)
{
    ESP_ERROR_CHECK(gptimer_new_timer(&config, &gptimer));
    ESP_ERROR_CHECK(gptimer_register_event_callbacks(gptimer, &cbs, NULL));
    ESP_ERROR_CHECK(gptimer_set_alarm_action(gptimer, &alarm_config));
    ESP_ERROR_CHECK(gptimer_enable(gptimer));
    ESP_ERROR_CHECK(gptimer_start(gptimer));
    ESP_LOGI(TAG, "Timer started with 100us period");
}