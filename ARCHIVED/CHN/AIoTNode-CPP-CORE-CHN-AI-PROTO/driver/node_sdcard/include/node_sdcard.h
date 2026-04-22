/**
 * @file node_sdcard.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is for SD card initialization and related functions
 * ！Here, we use SPI2, which is already initialized previously for LCD. For a same SPI, there can be many devices using different CS (Chip Select) pins.
 * @version 1.0
 * @date 2025-10-22
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

/* DEPENDENCIES */
#include "esp_vfs_fat.h" // ESP32 VFS FAT
#include "sdmmc_cmd.h"   // ESP32 SDMMC

#ifdef __cplusplus
extern "C"
{
#endif

// SD Card
#define MOUNT_POINT "/sdcard"
#define SD_MAX_CHAR_SIZE 64

#define SD_PIN_NUM_CS GPIO_NUM_2

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