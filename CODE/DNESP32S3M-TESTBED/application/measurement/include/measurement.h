/**
 * @file measurement.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the header file for the measurement module.
 * @version 1.0
 * @date 2025-04-01
 * @copyright Copyright (c) 2025
 * @note Currently, the sesning is realized by FreeRTOS, and therefore the upper limit sampling rate is 1kHz.
 * @todo Sampling by timer + interrupt
 * @todo Sampling by DMA
 */

#ifndef __MEASUREMENT_H__
#define __MEASUREMENT_H__

#ifdef __cplusplus
extern "C"
{
#endif

/* DEPENDENCIES */
// SYSTEM
#include "esp_system.h" // ESP32 System
#include "esp_log.h"    // ESP32 Logging

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

// TinyAuton/TinySHM
#include "TinyToolbox.h" // TinyToolbox Middleware

/* VARIABLES */
extern struct SenseConfig streamline_config; // Streamline configuration
extern struct SenseConfig sense_config; // Sense configuration

/* FUNCTIONS */
/**
 * @name acc_streamline_task
 * @brief This function is the task for streaming the accelerometer data.
 * @param pvParameters for FreeRTOS, the passed parameters are all void pointers and need to be casted to the correct type.
 * @retval None
 */
void acc_streamline_task(void *pvParameters);

/**
 * @name acc_sense_task
 * @brief This function is the task for sensing the accelerometer data according to the configuration structure.
 * @param pvParameters for FreeRTOS, the passed parameters are all void pointers and need to be casted to the correct type.
 * @retval None
 */
void acc_sense_task(void *pvParameters);

#ifdef __cplusplus
}
#endif

#endif /* __MEASUREMENT_H__ */