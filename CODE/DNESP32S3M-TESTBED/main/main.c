/**
 * @file main.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief
 * @version ?
 * @date 2025-03-18
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

// TinySHM
#include "measurement.h" // Measurement Header

/* Variables */
const char *TAG = "NEXNODE";

/**
 * @brief Entry point of the program
 * @param None
 * @retval None
 */
void app_main(void)
{
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

    // Node Initialization Start
    esp_err_t ret;
    uint32_t flash_size;
    esp_chip_info_t chip_info;

    char mqtt_pub_buff[64];
    int count = 0;

    // Initialize NVS
    ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase()); // Erase if needed
        ret = nvs_flash_init();
    }

    // Get FLASH size
    esp_flash_get_size(NULL, &flash_size);
    esp_chip_info(&chip_info);

    // Display CPU core count
    printf("CPU Cores: %d\n", chip_info.cores);

    // Display FLASH size
    printf("Flash size: %ld MB flash\n", flash_size / (1024 * 1024));

    // Display PSRAM size
    printf("PSRAM size: %d bytes\n", esp_psram_get_size());

    // BSP Initialization
    led_init();
    exit_init();
    spi2_init();
    lcd_init();
    i2c_bus_init();
    i2c_sensor_mpu6050_init();

    // spiffs_test();                                                  /* Run SPIFFS test */
    while (sd_card_init()) /* SD card not detected */
    {
        lcd_show_string(0, 0, 200, 16, 16, "SD Card Error!", RED);
        vTaskDelay(500);
        lcd_show_string(0, 20, 200, 16, 16, "Please Check!", RED);
        vTaskDelay(500);
    }

    // clean the screen
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

    // only when the ip is obtained, start time sync and mqtt
    EventBits_t ev = 0;
    ev = xEventGroupWaitBits(wifi_event_group, CONNECTED_BIT, pdTRUE, pdFALSE, portMAX_DELAY);

    if (ev & CONNECTED_BIT)
    {
        sync_time_with_timezone("CST-8");
        vTaskDelay(500/portTICK_PERIOD_MS);
        mqtt_app_start();
        vTaskDelay(3000 / portTICK_PERIOD_MS); // wait for mqtt to connect
    }

    // streamline
    // xTaskCreate(acc_streamline_task, "Accel_Streamline_Task", 4096, &streamline_config, 5, NULL);

    // normal sense
    // xTaskCreate(acc_sense_task, "Accel_Sense_Task", 8192, &sense_config, 5, NULL);

    while (1)
    {
        // led test
        led_toggle();
        
        // show time
        // tiny_get_current_time(1); // get the current time

        // time delay
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
}
