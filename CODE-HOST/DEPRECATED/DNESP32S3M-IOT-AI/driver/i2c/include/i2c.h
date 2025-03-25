/**
 * @file i2c.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file contains the function prototypes for i2c master initialization. This is to serve the peripherals that require I2C communication.
 * @version 1.0
 * @date 2025-03-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __I2C_H__
#define __I2C_H__

#include <stdio.h>
#include "esp_log.h"
#include "unity.h" // This is for unity testing
#include "driver/i2c.h"
#include "esp_system.h"

#define I2C_MASTER_SCL_IO 4      /*!< gpio number for I2C master clock */
#define I2C_MASTER_SDA_IO 5      /*!< gpio number for I2C master data  */
#define I2C_MASTER_NUM I2C_NUM_0  /*!< I2C port number for master dev */
#define I2C_MASTER_FREQ_HZ 100000 /*!< I2C master clock frequency */

/**
 * @brief i2c master initialization
 */
void i2c_bus_init(void);

#endif /* __I2C_H__ */