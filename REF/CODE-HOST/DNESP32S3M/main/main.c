/**
 * @file main.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief
 * @version 1.0
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

/* Variables */
const char *TAG = "NEXNODE";

/**
 * @brief Entry point of the program
 * @param None
 * @retval None
 */
void app_main(void)
{
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

    vTaskDelay(3000);

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

    // only when the ip is obtained, start mqtt
    EventBits_t ev = 0;
    ev = xEventGroupWaitBits(wifi_event_group, CONNECTED_BIT, pdTRUE, pdFALSE, portMAX_DELAY);
    if (ev & CONNECTED_BIT)
    {
        mqtt_app_start();
    }

    uint8_t mpu6050_deviceid;
    mpu6050_acce_value_t acce;
    mpu6050_gyro_value_t gyro;
    mpu6050_temp_value_t temp;
    complimentary_angle_t angle;

    ret = mpu6050_get_deviceid(mpu6050, &mpu6050_deviceid);
    TEST_ASSERT_EQUAL(ESP_OK, ret);
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(MPU6050_WHO_AM_I_VAL, mpu6050_deviceid, "Who Am I register does not contain expected data");

    while (1)
    {
        // led test
        led_toggle();

        // hellow world test
        ESP_LOGI(TAG, "Hello World!");

        // mpu6050 test
        ret = mpu6050_get_acce(mpu6050, &acce);
        // TEST_ASSERT_EQUAL(ESP_OK, ret);
        ESP_LOGI(TAG, "acce_x:%.6f, acce_y:%.6f, acce_z:%.6f\n", acce.acce_x, acce.acce_y, acce.acce_z);

        ret = mpu6050_get_gyro(mpu6050, &gyro);
        // TEST_ASSERT_EQUAL(ESP_OK, ret);
        ESP_LOGI(TAG, "gyro_x:%.6f, gyro_y:%.6f, gyro_z:%.6f\n", gyro.gyro_x, gyro.gyro_y, gyro.gyro_z);

        ret = mpu6050_get_temp(mpu6050, &temp);
        // TEST_ASSERT_EQUAL(ESP_OK, ret);
        ESP_LOGI(TAG, "t:%.6f \n", temp.temp);

        ret = mpu6050_complimentory_filter(mpu6050, &acce, &gyro, &angle);
        // TEST_ASSERT_EQUAL(ESP_OK, ret);
        ESP_LOGI(TAG, "pitch:%.6f roll:%.6f \n", angle.pitch, angle.roll);

        // // mqtt test
        // if(s_is_mqtt_connected)
        // {
        //     snprintf(mqtt_pub_buff,64,"{\"count\":\"%d\"}",count);
        //     esp_mqtt_client_publish(s_mqtt_client, MQTT_PUBLIC_TOPIC,
        //                     mqtt_pub_buff, strlen(mqtt_pub_buff),1, 0);
        //     count++;
        // }

        // mqtt test publish acc data
        if (s_is_mqtt_connected)
        {
            snprintf(mqtt_pub_buff, 64, "{\"acce_x\":\"%.6f\",\"acce_y\":\"%.6f\",\"acce_z\":\"%.6f\"}", acce.acce_x, acce.acce_y, acce.acce_z);
            esp_mqtt_client_publish(s_mqtt_client, MQTT_PUBLIC_TOPIC,
                                    mqtt_pub_buff, strlen(mqtt_pub_buff), 1, 0);
        }

        vTaskDelay(2000 / portTICK_PERIOD_MS);
    }
}
