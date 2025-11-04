/**
 * @file spi.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief
 * @version 1.0
 * @date 2024-11-18
 * @ref Alientek SPI driver
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __SPI_H__
#define __SPI_H__

#ifdef __cplusplus
extern "C"
{
#endif

/* Dependencies */
#include <string.h>
#include "esp_log.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"

/* GPIO Definitions */
#define SPI2_MOSI_GPIO_PIN GPIO_NUM_11 /* SPI2_MOSI */
#define SPI2_CLK_GPIO_PIN GPIO_NUM_12  /* SPI2_CLK */
#define SPI2_MISO_GPIO_PIN GPIO_NUM_13 /* SPI2_MISO */

/* Function Prototypes */

/**
 * @brief       Initialize SPI
 * @param       None
 * @retval      None
 */
void spi2_init(void);

/**
 * @brief       Send command via SPI
 * @param       handle : SPI handle
 * @param       cmd    : Command to send
 * @retval      None
 */
void spi2_write_cmd(spi_device_handle_t handle, uint8_t cmd);

/**
 * @brief       Send data via SPI
 * @param       handle : SPI handle
 * @param       data   : Data to send
 * @param       len    : Length of data to send
 * @retval      None
 */
void spi2_write_data(spi_device_handle_t handle, const uint8_t *data, int len);

/**
 * @brief       Process data via SPI
 * @param       handle       : SPI handle
 * @param       data         : Data to send
 * @retval      t.rx_data[0] : Received data
 */
uint8_t spi2_transfer_byte(spi_device_handle_t handle, uint8_t byte);

#ifdef __cplusplus
}
#endif

#endif