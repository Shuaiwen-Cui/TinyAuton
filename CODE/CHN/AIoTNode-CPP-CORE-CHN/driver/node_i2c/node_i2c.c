/**
 * @file node_i2c.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file contains the functions for i2c master initialization using ESP-IDF 6.0 i2c_master driver.
 * @version 1.0
 * @date 2025-10-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "node_i2c.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Global I2C Bus Handle */
i2c_master_bus_handle_t i2c_bus_handle = NULL;

/**
 * @brief i2c master bus initialization using ESP-IDF 6.0 i2c_master driver
 * @return esp_err_t ESP_OK on success, error code on failure
 */
esp_err_t i2c_bus_init(void)
{
    esp_err_t ret;
    
    /* Configure I2C master bus */
    i2c_master_bus_config_t i2c_bus_config = {
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .i2c_port = I2C_MASTER_NUM,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true
    };

    ret = i2c_new_master_bus(&i2c_bus_config, &i2c_bus_handle);
    if (ret != ESP_OK) {
        ESP_LOGE("I2C", "I2C bus creation failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI("I2C", "I2C master bus initialized successfully");
    return ESP_OK;
}

/**
 * @brief i2c master bus deinitialization
 * @return esp_err_t ESP_OK on success, error code on failure
 */
esp_err_t i2c_bus_deinit(void)
{
    esp_err_t ret = ESP_OK;
    
    if (i2c_bus_handle != NULL) {
        ret = i2c_del_master_bus(i2c_bus_handle);
        if (ret != ESP_OK) {
            ESP_LOGE("I2C", "I2C bus deletion failed: %s", esp_err_to_name(ret));
            return ret;
        }
        i2c_bus_handle = NULL;
    }

    ESP_LOGI("I2C", "I2C master bus deinitialized successfully");
    return ESP_OK;
}

/**
 * @brief add i2c device to bus
 * @param device_addr I2C device address
 * @param scl_speed_hz SCL clock speed in Hz
 * @param dev_handle pointer to store device handle
 * @return esp_err_t ESP_OK on success, error code on failure
 */
esp_err_t i2c_add_device(uint8_t device_addr, uint32_t scl_speed_hz, i2c_master_dev_handle_t *dev_handle)
{
    esp_err_t ret;
    
    if (i2c_bus_handle == NULL) {
        ESP_LOGE("I2C", "I2C bus not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = device_addr,
        .scl_speed_hz = scl_speed_hz
    };

    ret = i2c_master_bus_add_device(i2c_bus_handle, &dev_cfg, dev_handle);
    if (ret != ESP_OK) {
        ESP_LOGE("I2C", "I2C device addition failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI("I2C", "I2C device 0x%02X added successfully", device_addr);
    return ESP_OK;
}

/**
 * @brief remove i2c device from bus
 * @param dev_handle device handle to remove
 * @return esp_err_t ESP_OK on success, error code on failure
 */
esp_err_t i2c_remove_device(i2c_master_dev_handle_t dev_handle)
{
    esp_err_t ret = i2c_master_bus_rm_device(dev_handle);
    if (ret != ESP_OK) {
        ESP_LOGE("I2C", "I2C device removal failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI("I2C", "I2C device removed successfully");
    return ESP_OK;
}

/**
 * @brief write data to i2c device
 * @param dev_handle device handle
 * @param data data to write
 * @param data_len length of data
 * @param timeout_ms timeout in milliseconds
 * @return esp_err_t ESP_OK on success, error code on failure
 */
esp_err_t i2c_write_data(i2c_master_dev_handle_t dev_handle, const uint8_t *data, size_t data_len, int timeout_ms)
{
    esp_err_t ret = i2c_master_transmit(dev_handle, data, data_len, timeout_ms);
    if (ret != ESP_OK) {
        ESP_LOGE("I2C", "I2C write failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGD("I2C", "I2C write successful, %d bytes", data_len);
    return ESP_OK;
}

/**
 * @brief read data from i2c device
 * @param dev_handle device handle
 * @param data buffer to store read data
 * @param data_len length of data to read
 * @param timeout_ms timeout in milliseconds
 * @return esp_err_t ESP_OK on success, error code on failure
 */
esp_err_t i2c_read_data(i2c_master_dev_handle_t dev_handle, uint8_t *data, size_t data_len, int timeout_ms)
{
    esp_err_t ret = i2c_master_receive(dev_handle, data, data_len, timeout_ms);
    if (ret != ESP_OK) {
        ESP_LOGE("I2C", "I2C read failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGD("I2C", "I2C read successful, %d bytes", data_len);
    return ESP_OK;
}

/**
 * @brief write then read data from i2c device (combined transaction)
 * @param dev_handle device handle
 * @param write_data data to write
 * @param write_len length of write data
 * @param read_data buffer to store read data
 * @param read_len length of data to read
 * @param timeout_ms timeout in milliseconds
 * @return esp_err_t ESP_OK on success, error code on failure
 */
esp_err_t i2c_write_read_data(i2c_master_dev_handle_t dev_handle, 
                              const uint8_t *write_data, size_t write_len,
                              uint8_t *read_data, size_t read_len, 
                              int timeout_ms)
{
    esp_err_t ret = i2c_master_transmit_receive(dev_handle, write_data, write_len, read_data, read_len, timeout_ms);
    if (ret != ESP_OK) {
        ESP_LOGE("I2C", "I2C write-read failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGD("I2C", "I2C write-read successful, wrote %d bytes, read %d bytes", write_len, read_len);
    return ESP_OK;
}

#ifdef __cplusplus
}
#endif