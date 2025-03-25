/**
 * @file spi_sdcard.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is for SD card initialization and related functions
 * @version 1.0
 * @date 2024-11-19
 * @copyright Copyright (c) 2024
 *
 */

/* DEPENDENCIES */
#include "spi_sdcard.h"

/* VARIABLES */
sdmmc_card_t *card;

spi_device_handle_t MY_SD_Handle;

const char *TAG_SD = "SD_CARD";
const char mount_point[] = MOUNT_POINT; // modify the macro above to change the mount point

/* FUNCTIONS */

/**
 * @brief Initialize the SD card
 * ! Here, we assume that the SPI bus has been initialized before calling this function.
 * @param None
 * @retval esp_err_t
 */
esp_err_t sd_card_init(void)
{
    /* Result Indicator */
    esp_err_t ret;

    // Configure parameters for file system mounting
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
#ifdef CONFIG_FORMAT_IF_MOUNT_FAILED
        .format_if_mount_failed = true,   // Format the card if mounting fails
#else
        .format_if_mount_failed = false,  // Do not format the card if mounting fails
#endif // FORMAT_IF_MOUNT_FAILED
        .max_files = 5,                   // Maximum number of files that can be open at the same time
        .allocation_unit_size = 16 * 1024 // FAT allocation unit size
    };

    ESP_LOGI(TAG_SD, "Initializing SD card...");

    ESP_LOGI(TAG_SD, "Configuring SPI Device for SD card...");

    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    /*spi_bus_config_t bus_cfg = {
        .mosi_io_num = PIN_NUM_MOSI,
        .miso_io_num = PIN_NUM_MISO,
        .sclk_io_num = PIN_NUM_CLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4000,
    };*/

    host.slot = SPI2_HOST; // Use SPI2 for SD card

    // In SPI mode, since the SD card's maximum frequency is 25 MHz in SPI mode, the configured value must not exceed this limit. 
    // The default frequency for SD-SPI in IDF is 20 MHz.

    // If SPI2 has not been initialized, initialize it
    /*ret = spi_bus_initialize(host.slot, &bus_cfg, SPI_DMA_CHAN);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize bus.");
        return false;
    }*/

    // This initialization does not include card detection (CD) and write protection (WP) signals.
    // If your board has these signals, modify slot_config.gpio_cd and slot_config.gpio_wp.
    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = SD_PIN_NUM_CS;
    slot_config.host_id = host.slot;

    // Use SPI mode to drive the SD card and mount the FATFS file system
    ret = esp_vfs_fat_sdspi_mount(mount_point, &host, &slot_config, &mount_config, &card);

    if (ret != ESP_OK)
    {
        if (ret == ESP_FAIL)
        {
            ESP_LOGE(TAG_SD, "Failed to mount filesystem. "
                          "If you want the card to be formatted, set the EXAMPLE_FORMAT_IF_MOUNT_FAILED menuconfig option.");
        }
        else
        {
            ESP_LOGE(TAG_SD, "Failed to initialize the card (%s). "
                          "Make sure SD card lines have pull-up resistors in place.",
                     esp_err_to_name(ret));
        }
        return false;
    }

    // The card has been initialized, print its properties
    sdmmc_card_print_info(stdout, card);

    // Progress Report
    ESP_LOGI(TAG_SD, "Filesystem mounted successfully.");

    return ret;
}


/**
 * @brief Test file operations on the SD card
 * @param None
 * @retval esp_err_t
 */
esp_err_t sd_card_test_filesystem(void)
{
    char data[SD_MAX_CHAR_SIZE];

    // Create a file
    char file_path[SD_MAX_CHAR_SIZE];
    snprintf(file_path, sizeof(file_path), "%s/test.txt", mount_point);
    snprintf(data, sizeof(data), "Hello Mr Cui!, %s!\n", card->cid.name);
    FILE *f = fopen(file_path, "w");
    if (!f)
    {
        ESP_LOGE(TAG_SD, "Failed to create file");
        return ESP_FAIL;
    }
    fprintf(f, "%s", data);
    fclose(f);

    ESP_LOGI(TAG_SD, "File created: %s", file_path);

    // Read the file
    f = fopen(file_path, "r");
    if (!f)
    {
        ESP_LOGE(TAG_SD, "Failed to open file for reading");
        return ESP_FAIL;
    }
    char line[SD_MAX_CHAR_SIZE];
    fgets(line, sizeof(line), f);
    fclose(f);

    ESP_LOGI(TAG_SD, "File content: %s", line);

    return ESP_OK;
}

/**
 * @brief Unmount the File System and SPI Bus
 * @param None
 * @retval esp_err_t
 */
esp_err_t sd_card_unmount(void)
{
    esp_err_t ret;

    ret = esp_vfs_fat_sdcard_unmount(mount_point, card);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG_SD, "Failed to unmount filesystem");
        return ret;
    }

    // Unmount SPI Bus
    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    ret = spi_bus_free(host.slot);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG_SD, "Failed to free SPI bus");
        return ret;
    }

    ESP_LOGI(TAG_SD, "Filesystem unmounted and SPI bus freed");

    return ret;
}