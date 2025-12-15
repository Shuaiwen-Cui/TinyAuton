/**
 * @file node_spi.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief
 * @version 1.0
 * @date 2025-10-22
 * @ref Alientek SPI driver
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

/* Dependencies */
#include <string.h>
#include "esp_log.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"

/* SPI2 GPIO Definitions */
#define SPI2_MOSI_GPIO_PIN GPIO_NUM_11 /* SPI2_MOSI */
#define SPI2_CLK_GPIO_PIN GPIO_NUM_12  /* SPI2_CLK */
#define SPI2_MISO_GPIO_PIN GPIO_NUM_13 /* SPI2_MISO */

/* SPI3 GPIO Definitions */
#define SPI3_MOSI_GPIO_PIN GPIO_NUM_5 /* SPI3_MOSI */
#define SPI3_CLK_GPIO_PIN GPIO_NUM_7  /* SPI3_SCLK */
#define SPI3_MISO_GPIO_PIN GPIO_NUM_6 /* SPI3_MISO */

    /* Global SPI device handles */

    /* Function Prototypes */

    /**
     * @brief       Initialize SPI2
     * @param       None
     * @retval      None
     */
    void spi2_init(void);

    /**
     * @brief       Initialize SPI3
     * @param       None
     * @retval      None
     */
    void spi3_init(void);

    /**
     * @brief       Send command via SPI2
     * @param       handle : SPI handle
     * @param       cmd    : Command to send
     * @retval      None
     */
    void spi2_write_cmd(spi_device_handle_t handle, uint8_t cmd);

    /**
     * @brief       Send command via SPI3
     * @param       handle : SPI handle
     * @param       cmd    : Command to send
     * @retval      None
     */
    void spi3_write_cmd(spi_device_handle_t handle, uint8_t cmd);

    /**
     * @brief       Send data via SPI2
     * @param       handle : SPI handle
     * @param       data   : Data to send
     * @param       len    : Length of data to send
     * @retval      None
     */
    void spi2_write_data(spi_device_handle_t handle, const uint8_t *data, int len);

    /**
     * @brief       Send data via SPI3
     * @param       handle : SPI handle
     * @param       data   : Data to send
     * @param       len    : Length of data to send
     * @retval      None
     */
    void spi3_write_data(spi_device_handle_t handle, const uint8_t *data, int len);

    /**
     * @brief       Process data via SPI2
     * @param       handle       : SPI handle
     * @param       data         : Data to send
     * @retval      t.rx_data[0] : Received data
     */
    uint8_t spi2_transfer_byte(spi_device_handle_t handle, uint8_t byte);

    /**
     * @brief       Process data via SPI3
     * @param       handle       : SPI handle
     * @param       data         : Data to send
     * @retval      t.rx_data[0] : Received data
     */
    uint8_t spi3_transfer_byte(spi_device_handle_t handle, uint8_t byte);

    /**
     * @brief       Read data via SPI2
     * @param       handle       : SPI handle
     * @param       data         : Buffer to store received data
     * @param       len          : Length of data to read
     * @retval      None
     */
    void spi2_read_data(spi_device_handle_t handle, uint8_t *data, int len);

    /**
     * @brief       Read data via SPI3
     * @param       handle       : SPI handle
     * @param       data         : Buffer to store received data
     * @param       len          : Length of data to read
     * @retval      None
     */
    void spi3_read_data(spi_device_handle_t handle, uint8_t *data, int len);

    /**
     * @brief       Read single byte via SPI2
     * @param       handle       : SPI handle
     * @retval      Received byte
     */
    uint8_t spi2_read_byte(spi_device_handle_t handle);

    /**
     * @brief       Read single byte via SPI3
     * @param       handle       : SPI handle
     * @retval      Received byte
     */
    uint8_t spi3_read_byte(spi_device_handle_t handle);

#ifdef __cplusplus
}
#endif