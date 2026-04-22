/**
 * @file example_usage.c
 * @brief Example usage of the updated node_i2c module with ESP-IDF 6.0 i2c_master driver
 * @author SHUAIWEN CUI
 * @date 2025-01-27
 */

#include "node_i2c.h"
#include "esp_log.h"

static const char *TAG = "I2C_EXAMPLE";

/**
 * @brief Example function showing how to use the updated I2C module
 */
void i2c_example_usage(void)
{
    esp_err_t ret;
    i2c_master_dev_handle_t dev_handle;
    
    // Example device address (replace with your actual device address)
    uint8_t device_addr = 0x68;  // Example: MPU6050 sensor
    uint32_t scl_speed = 400000; // 400kHz I2C speed
    
    ESP_LOGI(TAG, "Starting I2C example...");
    
    // Step 1: Initialize I2C bus
    ret = i2c_bus_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C bus initialization failed: %s", esp_err_to_name(ret));
        return;
    }
    
    // Step 2: Add I2C device to bus
    ret = i2c_add_device(device_addr, scl_speed, &dev_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C device addition failed: %s", esp_err_to_name(ret));
        i2c_bus_deinit();
        return;
    }
    
    // Step 3: Example I2C operations
    
    // Write data to device
    uint8_t write_data[] = {0x75}; // WHO_AM_I register for MPU6050
    ret = i2c_write_data(dev_handle, write_data, sizeof(write_data), 1000);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C write failed: %s", esp_err_to_name(ret));
    } else {
        ESP_LOGI(TAG, "I2C write successful");
    }
    
    // Read data from device
    uint8_t read_data[1];
    ret = i2c_read_data(dev_handle, read_data, sizeof(read_data), 1000);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C read failed: %s", esp_err_to_name(ret));
    } else {
        ESP_LOGI(TAG, "I2C read successful, data: 0x%02X", read_data[0]);
    }
    
    // Combined write-read operation
    uint8_t reg_addr = 0x75;
    uint8_t response[1];
    ret = i2c_write_read_data(dev_handle, &reg_addr, 1, response, 1, 1000);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C write-read failed: %s", esp_err_to_name(ret));
    } else {
        ESP_LOGI(TAG, "I2C write-read successful, response: 0x%02X", response[0]);
    }
    
    // Step 4: Clean up
    ret = i2c_remove_device(dev_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C device removal failed: %s", esp_err_to_name(ret));
    }
    
    ret = i2c_bus_deinit();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C bus deinitialization failed: %s", esp_err_to_name(ret));
    }
    
    ESP_LOGI(TAG, "I2C example completed");
}

/**
 * @brief Example function for multiple I2C devices
 */
void i2c_multiple_devices_example(void)
{
    esp_err_t ret;
    i2c_master_dev_handle_t dev1_handle, dev2_handle;
    
    // Initialize I2C bus
    ret = i2c_bus_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C bus initialization failed");
        return;
    }
    
    // Add multiple devices
    ret = i2c_add_device(0x68, 400000, &dev1_handle); // Device 1
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to add device 1");
        i2c_bus_deinit();
        return;
    }
    
    ret = i2c_add_device(0x69, 400000, &dev2_handle); // Device 2
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to add device 2");
        i2c_remove_device(dev1_handle);
        i2c_bus_deinit();
        return;
    }
    
    // Use both devices...
    uint8_t data1[] = {0x75};
    uint8_t data2[] = {0x00};
    
    i2c_write_data(dev1_handle, data1, 1, 1000);
    i2c_write_data(dev2_handle, data2, 1, 1000);
    
    // Clean up
    i2c_remove_device(dev1_handle);
    i2c_remove_device(dev2_handle);
    i2c_bus_deinit();
}
