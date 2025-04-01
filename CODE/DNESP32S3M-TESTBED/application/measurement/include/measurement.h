/**
 * @file measurement.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the header file for the measurement module.
 * @version 1.0
 * @date 2025-04-01
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __MEASUREMENT_H__
#define __MEASUREMENT_H__

/* DEPENDENCIES */
// SYSTEM CORE
#include "esp_system.h"    // ESP32 System
// #include "nvs_flash.h"     // ESP32 NVS
// #include "esp_chip_info.h" // ESP32 Chip Info
// #include "esp_psram.h"     // ESP32 PSRAM
// #include "esp_flash.h"     // ESP32 Flash
#include "esp_log.h"       // ESP32 Logging

// RTOS
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "freertos/timers.h"
#include "freertos/semphr.h"

// BSP
#include "driver/gptimer.h"
#include "led.h"
#include "lcd.h"
#include "tim.h"
#include "esp_rtc.h"
#include "spi_sdcard.h"
#include "wifi_wpa2_enterprise.h"
#include "mqtt.h"
#include "mpu6050.h"

/* FUNCTIONS */
void sense_test(void);



#endif /* __MEASUREMENT_H__ */