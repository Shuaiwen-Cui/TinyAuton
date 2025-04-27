/**
 * @file spi_sdcard.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is for SD card initialization and related functions
 * @version 1.0
 * @date 2024-11-19
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __SPI_SDCARD_H__
#define __SPI_SDCARD_H__

#ifdef __cplusplus
extern "C"
{
#endif

/* DEPENDENCIES */
#include "esp_vfs_fat.h" // ESP32 VFS FAT
#include "sdmmc_cmd.h" // ESP32 SDMMC

// SD Card
#define MOUNT_POINT     "/sdcard" 
#define SD_MAX_CHAR_SIZE    64

#define SD_PIN_NUM_CS    GPIO_NUM_2

/* VARIABLES */
extern sdmmc_card_t *card;

/* FUNCTIONS */

/**
 * @brief Initialize the SD card
 * @param None
 * @retval esp_err_t
 */
esp_err_t sd_card_init(void);

/**
 * @brief Test file operations on the SD card
 * @param None
 * @retval esp_err_t
 */
esp_err_t sd_card_test_filesystem(void);

/**
 * @brief Unmount the File System and SPI Bus
 * @param None
 * @retval esp_err_t
 */
esp_err_t sd_card_unmount(void);

#ifdef __cplusplus
}
#endif

#endif