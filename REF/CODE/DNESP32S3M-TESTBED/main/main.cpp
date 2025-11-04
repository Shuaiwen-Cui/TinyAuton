/**
 * @file main.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file serves as the entry point for the whole project.
 * @version 1.0
 * @date 2025-04-19
 *
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
// ESP
#include "esp_system.h"    // ESP32 System
#include "nvs_flash.h"     // ESP32 NVS
#include "esp_chip_info.h" // ESP32 Chip Info
#include "esp_psram.h"     // ESP32 PSRAM
#include "esp_flash.h"     // ESP32 Flash
#include "esp_log.h"       // ESP32 Logging

// Standard Libraries
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

// FreeRTOS
#include "freertos/FreeRTOS.h"     // FreeRTOS
#include "freertos/task.h"         // FreeRTOS Task
#include "freertos/event_groups.h" // FreeRTOS Event Groups
#include "freertos/queue.h"        // FreeRTOS Queue
#include "freertos/timers.h"       // FreeRTOS Timers
#include "freertos/semphr.h"       // FreeRTOS Semaphore

// BSP
#include "led.h"
#include "exit.h"
#include "spi.h"
#include "i2c.h"
#include "lcd.h"
#include "tim.h"
#include "esp_rtc.h"
#include "spi_sdcard.h"
#include "wifi_wpa2_enterprise.h"
#include "mqtt.h"
#include "mpu6050.h"

// TinyAuton
#include "tiny_toolbox.h" // TinyToolbox
#include "tiny_math.h"   // TinyMath Header

// #include "tiny_vec.h" // TinyVec Header
// #include "tiny_vec_test.h" // TinyVec Test Header
// #include "tiny_mat.h" // TinyMat Header
// #include "tiny_mat_test.h" // TinyMat Test Header
// #include "tiny_matrix.hpp" // TinyMatrix Header
// #include "tiny_matrix_test.hpp" // TinyMatrix Test Header

// TinySHM
#include "measurement.h" // Measurement Header

extern "C" void app_main();
/**
 * @brief Entry point of the program
 * @param None
 * @retval None
 */
void app_main()
{
    /* RESET REASON CHECK FOR DEBUGGING */

    // Delay for 500ms to allow the system to stabilize
    vTaskDelay(pdMS_TO_TICKS(500)); 

    // Help to determine the reset reason
    esp_reset_reason_t reason = esp_reset_reason();
    ESP_LOGI("RESET", "Reset reason: %d", reason);
    switch (reason)
    {
        case ESP_RST_POWERON:
            ESP_LOGI("RESET", "Power-on reset");
            break;
        case ESP_RST_BROWNOUT:
            ESP_LOGI("RESET", "Brownout reset");
            break;
        case ESP_RST_WDT:
            ESP_LOGI("RESET", "Watchdog reset");
            break;
        case ESP_RST_DEEPSLEEP:
            ESP_LOGI("RESET", "Wake from deep sleep");
            break;
        default:
            ESP_LOGI("RESET", "Other reset reason");
            break;
    }

    /* VARIABLES */
    const char *TAG_NODE = "NODE";

    // Node Initialization Start
    esp_err_t ret;
    uint32_t flash_size;
    esp_chip_info_t chip_info;
    EventBits_t ev = 0; 
    // char mqtt_pub_buff[64];

    /* NVS FLASH INITIALIZATION */
    ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase()); // Erase if needed
        ret = nvs_flash_init();
    }

    /* CHIP INFO */
    esp_flash_get_size(NULL, &flash_size);
    esp_chip_info(&chip_info);
    printf("CPU Cores: %d\n", chip_info.cores);
    printf("Flash size: %ld MB flash\n", flash_size / (1024 * 1024));
    printf("PSRAM size: %d bytes\n", esp_psram_get_size());

    /* BSP INITIALIZATION AND TESTING */

    // BSP Initialization
    led_init();

    exit_init();

    spi2_init();

    lcd_init();

    i2c_bus_init();

    i2c_sensor_mpu6050_init();

    while (sd_card_init()) /* SD card not detected */
    {
        lcd_show_string(0, 0, 200, 16, 16, "SD Card Error!", RED);
        vTaskDelay(500);
        lcd_show_string(0, 20, 200, 16, 16, "Please Check!", RED);
        vTaskDelay(500);
    }
    lcd_clear(WHITE);
    lcd_show_string(0, 0, 200, 16, 16, "SD Initialized!", RED);
    sd_card_test_filesystem(); /* Run SD card test */
    lcd_show_string(0, 0, 200, 16, 16, "SD Tested CSW! ", RED);
    // sd_card_unmount();
    vTaskDelay(1000);

    lcd_show_string(0, 0, lcd_self.width, 16, 16, "WiFi STA Test  ", RED);
    ret = wifi_sta_wpa2_init();
    if (ret == ESP_OK)
    {
        ESP_LOGI(TAG_WIFI, "WiFi STA Init OK");
        lcd_show_string(0, 0, lcd_self.width, 16, 16, "WiFi STA Test OK", RED);
    }
    else
    {
        ESP_LOGE(TAG_WIFI, "WiFi STA Init Failed");
    }

    // time sync and mqtt
    ev = xEventGroupWaitBits(wifi_event_group, CONNECTED_BIT, pdTRUE, pdFALSE, portMAX_DELAY);
    if (ev & CONNECTED_BIT)
    {
        sync_time_with_timezone("CST-8");
        vTaskDelay(500/portTICK_PERIOD_MS);
        mqtt_app_start();
        vTaskDelay(3000 / portTICK_PERIOD_MS); // wait for mqtt to connect
    }

    /* TASK PERFORMING */
    // streamline
    // xTaskCreate(acc_streamline_task, "Accel_Streamline_Task", 4096, &streamline_config, 5, NULL);

    // normal sense
    // xTaskCreate(acc_sense_task, "Accel_Sense_Task", 8192, &sense_config, 5, NULL);

    /* TEST SANDBOX */
    // tiny_vec_test(); // test tiny_vec
    // tiny_mat_test(); // test tiny_mat
    tiny_matrix_test(); // test tiny_matrix



    /* INFINITE LOOP */
    while (1)
    {
        // led test
        led_toggle();
        
        // show time
        // tiny_get_current_time(1); // get the current time

        // time delay
        vTaskDelay(1000 / portTICK_PERIOD_MS); // 1 second delay
    }
}
