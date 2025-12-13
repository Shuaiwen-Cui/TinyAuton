/**
 * @file node_i2c.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file contains the function prototypes for i2c master initialization. This is to serve the peripherals that require I2C communication.
 * @version 1.0
 * @date 2025-10-22
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#include "esp_log.h"
#include "driver/i2c_master.h"
#include "esp_system.h"

#define I2C_MASTER_SCL_IO 10       /*!< gpio number for I2C master clock */
#define I2C_MASTER_SDA_IO 11       /*!< gpio number for I2C master data  */
#define I2C_MASTER_NUM I2C_NUM_0  /*!< I2C port number for master dev */
#define I2C_MASTER_FREQ_HZ 100000 /*!< I2C master clock frequency */

/* I2C Bus and Device Handles */
extern i2c_master_bus_handle_t i2c_bus_handle;

   /**
    * @brief i2c master bus initialization
    * @return esp_err_t ESP_OK on success, error code on failure
    */
   esp_err_t i2c_bus_init(void);

   /**
    * @brief i2c master bus deinitialization
    * @return esp_err_t ESP_OK on success, error code on failure
    */
   esp_err_t i2c_bus_deinit(void);

   /**
    * @brief add i2c device to bus
    * @param device_addr I2C device address
    * @param scl_speed_hz SCL clock speed in Hz
    * @param dev_handle pointer to store device handle
    * @return esp_err_t ESP_OK on success, error code on failure
    */
   esp_err_t i2c_add_device(uint8_t device_addr, uint32_t scl_speed_hz, i2c_master_dev_handle_t *dev_handle);

   /**
    * @brief remove i2c device from bus
    * @param dev_handle device handle to remove
    * @return esp_err_t ESP_OK on success, error code on failure
    */
   esp_err_t i2c_remove_device(i2c_master_dev_handle_t dev_handle);

   /**
    * @brief write data to i2c device
    * @param dev_handle device handle
    * @param data data to write
    * @param data_len length of data
    * @param timeout_ms timeout in milliseconds
    * @return esp_err_t ESP_OK on success, error code on failure
    */
   esp_err_t i2c_write_data(i2c_master_dev_handle_t dev_handle, const uint8_t *data, size_t data_len, int timeout_ms);

   /**
    * @brief read data from i2c device
    * @param dev_handle device handle
    * @param data buffer to store read data
    * @param data_len length of data to read
    * @param timeout_ms timeout in milliseconds
    * @return esp_err_t ESP_OK on success, error code on failure
    */
   esp_err_t i2c_read_data(i2c_master_dev_handle_t dev_handle, uint8_t *data, size_t data_len, int timeout_ms);

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
                                 int timeout_ms);

#ifdef __cplusplus
}
#endif
