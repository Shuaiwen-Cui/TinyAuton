# TIME

## tiny_time.h

```c
/**
 * @file tiny_time.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Submodule for TinyToolbox - header file
 * @version 1.0
 * @date 2025-04-10
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

/* CONFIGURATIONS */

/* ================================ DEPENDENCIES ================================ */
#include <stdbool.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_sntp.h"
// customized drivers
#include "node_rtc.h"

    /* ================================ DEFINITIONS ================================= */
    // Use int64_t to match esp_timer_get_time() return type and avoid overflow
    // esp_timer_get_time() returns microseconds since boot (int64_t)
    typedef int64_t TinyTimeMark_t;

    /**
     * @brief Structure to hold date and time
     */
    typedef struct TinyDateTime_t
    {
        int year;
        int month;
        int day;
        int hour;
        int minute;
        int second;
        int32_t microsecond; // Microseconds (0-999999), using int32_t for portability
    } TinyDateTime_t;

    /* ================================ FUNCTIONS =================================== */
    /* LOCAL RUNNING TIME IN MICROSECONDS */
    /**
     * @brief Get the running time in microseconds
     * @return TinyTimeMark_t
     */
    TinyTimeMark_t tiny_get_running_time(void);

    /* WORLD CURRENT TIME - SNTP */
    /**
     * @brief Obtain the current time with timezone
     * @param timezone_str Timezone string (e.g., "CST-8" or "GMT+8")
     * @note The timezone string format should be compatible with POSIX TZ format (e.g., "CST-8", "GMT+8")
     * @note To use this function, in application, after internet connection, call sync_time_with_timezone("CST-8")
     * @return None
     */
    void sync_time_with_timezone(const char *timezone_str);

    /* WORLD CURRENT TIME - GET TIME */
    /**
     * @name tiny_get_current_datetime
     * @brief Get the current time as a TinyDateTime_t struct
     * @param print_flag Flag to indicate whether to print the time
     * @return TinyDateTime_t structure containing the current date and time
     */
    TinyDateTime_t tiny_get_current_datetime(bool print_flag);

#ifdef __cplusplus
}
#endif
```

## tiny_time.c

```c
/**
 * @file tiny_time.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Submodule for TinyToolbox - source file
 * @version 1.0
 * @date 2025-04-10
 * @copyright Copyright (c) 2025
 *
 */

/* ================================ DEPENDENCIES
 * ================================ */
#include "tiny_time.h" // Time

/* ================================ DEFINITIONS
 * ================================= */
/* CONFIGURATIONS */
#define MIN_VALID_YEAR_OFFSET \
    (2020 - 1900) // Minimum valid year offset (year 2020)

/* TAGS */
static const char *TAG_SNTP = "NTP_SYNC";
static const char *TAG_TIME = "TIME";

/* ================================ FUNCTIONS
 * =================================== */
/* LOCAL RUNNING TIME IN MICROSECONDS */
/**
 * @brief Get the running time in microseconds
 * @return TinyTimeMark_t
 */
TinyTimeMark_t tiny_get_running_time(void) { return esp_timer_get_time(); }

/* WORLD CURRENT TIME - SNTP */
/**
 * @brief Callback function for time synchronization notification
 * @param tv Pointer to the timeval structure containing the synchronized time
 * @return None
 */
static void time_sync_notification_cb(struct timeval *tv)
{
    ESP_LOGI(TAG_SNTP, "Time synchronized!");
}

/**
 * @brief Initialize SNTP
 * @note This function can be called multiple times if needed
 * @return None
 */
static void initialize_sntp(void)
{
    ESP_LOGI(TAG_SNTP, "Initializing SNTP");
    esp_sntp_setoperatingmode(SNTP_OPMODE_POLL);
    esp_sntp_setservername(0, "pool.ntp.org"); // NTP server // pool.ntp.org // ntp.aliyun.com
    esp_sntp_set_time_sync_notification_cb(time_sync_notification_cb);
    esp_sntp_init();
}

/**
 * @brief Obtain the current time with timezone
 * @param timezone_str Timezone string (e.g., "CST-8" or "GMT+8")
 * @note The timezone string format should be compatible with POSIX TZ format
 * (e.g., "CST-8", "GMT+8")
 * @note To use this function, in application, after internet connection, call
 * sync_time_with_timezone("CST-8")
 * @return None
 */
void sync_time_with_timezone(const char *timezone_str)
{
    // Validate input parameter
    if (timezone_str == NULL)
    {
        ESP_LOGE(TAG_SNTP, "timezone_str is NULL");
        return;
    }

    // Set system timezone
    if (setenv("TZ", timezone_str, 1) != 0)
    {
        ESP_LOGE(TAG_SNTP, "Failed to set timezone environment variable");
        return;
    }
    tzset();

    // Initialize SNTP and start time sync
    initialize_sntp();

    // Wait for system time to be set
    time_t now = 0;
    struct tm timeinfo = {0};
    int retry = 0;
    const int retry_count = 15;

    while (timeinfo.tm_year < MIN_VALID_YEAR_OFFSET && ++retry < retry_count)
    {
        ESP_LOGI(TAG_SNTP, "Waiting for system time to be set... (%d/%d)", retry,
                 retry_count);
        vTaskDelay(2000 / portTICK_PERIOD_MS);
        time(&now);
        if (localtime_r(&now, &timeinfo) == NULL)
        {
            ESP_LOGW(TAG_SNTP, "Failed to convert time to local time");
            continue;
        }
    }

    if (timeinfo.tm_year >= MIN_VALID_YEAR_OFFSET)
    {
        rtc_set_time(timeinfo.tm_year + 1900, timeinfo.tm_mon + 1, timeinfo.tm_mday,
                     timeinfo.tm_hour, timeinfo.tm_min,
                     timeinfo.tm_sec); // defined in esp_rtc.c
        ESP_LOGI(TAG_SNTP, "System time is set.");
    }
    else
    {
        ESP_LOGW(TAG_SNTP, "Failed to sync time.");
        return;
    }

    // Log current local time (using thread-safe formatting)
    char time_str[64];
    if (strftime(time_str, sizeof(time_str), "%a %b %d %H:%M:%S %Y", &timeinfo) ==
        0)
    {
        ESP_LOGW(TAG_SNTP, "Failed to format time string");
    }
    else
    {
        ESP_LOGI(TAG_SNTP, "Current time: %s", time_str);
    }

    // vTaskDelay(10000 / portTICK_PERIOD_MS); // Wait for 10 second
    // rtc_get_time(); // uncomment to check the RTC time
    // ESP_LOGI(TAG_SNTP, "Current RTC time: %04d-%02d-%02d %02d:%02d:%02d",
    //          calendar.year, calendar.month, calendar.date,
    //          calendar.hour, calendar.min, calendar.sec); // uncomment to check
    //          the RTC time
}

/* WORLD CURRENT TIME - GET TIME */
/**
 * @name tiny_get_current_datetime
 * @brief Get the current time as a TinyDateTime_t struct
 * @param print_flag Flag to indicate whether to print the time
 * @return TinyDateTime_t structure containing the current date and time
 */
TinyDateTime_t tiny_get_current_datetime(bool print_flag)
{
    TinyDateTime_t result = {0}; // Initialize to zero
    struct timeval tv;

    // Get current time (seconds + microseconds)
    if (gettimeofday(&tv, NULL) != 0)
    {
        ESP_LOGE(TAG_TIME, "Failed to get time of day");
        return result; // Return zero-initialized structure on error
    }

    time_t now = tv.tv_sec;
    struct tm timeinfo;
    if (localtime_r(&now, &timeinfo) == NULL)
    {
        ESP_LOGE(TAG_TIME, "Failed to convert time to local time");
        return result; // Return zero-initialized structure on error
    }

    result.year = timeinfo.tm_year + 1900;
    result.month = timeinfo.tm_mon + 1;
    result.day = timeinfo.tm_mday;
    result.hour = timeinfo.tm_hour;
    result.minute = timeinfo.tm_min;
    result.second = timeinfo.tm_sec;
    result.microsecond = (int32_t)tv.tv_usec; // Explicit cast for portability

    if (print_flag)
    {
        ESP_LOGI(TAG_TIME, "Current Time: %04d-%02d-%02d %02d:%02d:%02d.%06d",
                 result.year, result.month, result.day, result.hour, result.minute,
                 result.second, result.microsecond);
    }

    return result;
}
```

## main.cpp

```cpp
/**
 * @file main.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Main program for testing tiny_time module
 * @version 1.0
 * @date 2025-10-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/* DEPENDENCIES */
// ESP
#include "nvs_flash.h"
#include "esp_log.h"

#ifdef __cplusplus
extern "C" {
#endif

// FreeRTOS (must be inside extern "C" for C++ files)
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"

// ESP Timer (high-precision timer)
#include "esp_timer.h"

// WiFi (required for time sync)
#include "node_wifi.h"

// TinyToolbox
#include "tiny_time.h"

/* Variables */
const char *TAG = "tiny_time_test";

/* Timer precision test variables */
#define TIMESTAMP_COUNT 15
static TinyTimeMark_t s_timestamps[TIMESTAMP_COUNT] = {0};
static int s_timestamp_index = 0;
static bool s_timer_test_complete = false;
static esp_timer_handle_t s_timer_handle = NULL;
static SemaphoreHandle_t s_timer_mutex = NULL;

/**
 * @brief Timer callback function - records timestamp at the very beginning
 * @param arg Timer argument (not used)
 * @return None
 */
static void timer_precision_callback(void *arg)
{
    // CRITICAL: Get timestamp IMMEDIATELY at the start of callback
    // to avoid any execution overhead affecting the measurement
    TinyTimeMark_t timestamp = tiny_get_running_time();
    
    // Store timestamp in array (thread-safe access)
    if (xSemaphoreTake(s_timer_mutex, portMAX_DELAY) == pdTRUE)
    {
        if (s_timestamp_index < TIMESTAMP_COUNT)
        {
            s_timestamps[s_timestamp_index++] = timestamp;
            
            // Stop timer when we have collected all timestamps
            if (s_timestamp_index >= TIMESTAMP_COUNT)
            {
                s_timer_test_complete = true;
                esp_timer_stop(s_timer_handle);
            }
        }
        xSemaphoreGive(s_timer_mutex);
    }
}

/**
 * @brief Entry point of the program - Testing tiny_time module
 * @param None
 * @retval None
 */
void app_main(void)
{
    esp_err_t ret;

    // Initialize NVS (required for WiFi)
    ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  tiny_time Module Test Program");
    ESP_LOGI(TAG, "========================================");

    // Initialize WiFi (required for time synchronization)
    ESP_LOGI(TAG, "Initializing WiFi...");
    ret = wifi_sta_wpa2_init();
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "WiFi initialization failed!");
        return;
    }
    ESP_LOGI(TAG, "WiFi initialized successfully");

    // Wait for WiFi connection
    ESP_LOGI(TAG, "Waiting for WiFi connection...");
    EventBits_t ev = xEventGroupWaitBits(wifi_event_group, CONNECTED_BIT, 
                                         pdTRUE, pdFALSE, portMAX_DELAY);
    if (ev & CONNECTED_BIT)
    {
        ESP_LOGI(TAG, "WiFi connected!");
        
        // ============================================================
        // Test 1: Get running time (microseconds since boot)
        // ============================================================
        ESP_LOGI(TAG, "\n--- Test 1: Get Running Time ---");
        TinyTimeMark_t start_time = tiny_get_running_time();
        ESP_LOGI(TAG, "Running time: %lld microseconds", start_time);
        ESP_LOGI(TAG, "Running time: %.3f seconds", start_time / 1000000.0);

        // ============================================================
        // Test 2: Sync time with timezone
        // ============================================================
        ESP_LOGI(TAG, "\n--- Test 2: Sync Time with Timezone ---");
        ESP_LOGI(TAG, "Syncing time with timezone CST-8...");
        sync_time_with_timezone("CST-8");
        
        // Wait for time synchronization (SNTP may take a few seconds)
        ESP_LOGI(TAG, "Waiting for time synchronization...");
        vTaskDelay(5000 / portTICK_PERIOD_MS);

        // ============================================================
        // Test 3: Get current datetime
        // ============================================================
        ESP_LOGI(TAG, "\n--- Test 3: Get Current DateTime ---");
        (void)tiny_get_current_datetime(true);  // Function prints internally

        // ============================================================
        // Test 4: Measure time elapsed
        // ============================================================
        ESP_LOGI(TAG, "\n--- Test 4: Measure Time Elapsed ---");
        TinyTimeMark_t end_time = tiny_get_running_time();
        TinyTimeMark_t elapsed = end_time - start_time;
        ESP_LOGI(TAG, "Time elapsed: %lld microseconds", elapsed);
        ESP_LOGI(TAG, "Time elapsed: %.3f seconds", elapsed / 1000000.0);

        ESP_LOGI(TAG, "\n========================================");
        ESP_LOGI(TAG, "  Initial Tests Completed");
        ESP_LOGI(TAG, "========================================\n");
    }
    else
    {
        ESP_LOGE(TAG, "WiFi connection failed!");
        return;
    }

    // ============================================================
    // Timer precision test: Record 15 timestamps at 2-second intervals
    // ============================================================
    ESP_LOGI(TAG, "\n========================================");
    ESP_LOGI(TAG, "  Timer Precision Test");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Recording 15 timestamps at 2-second intervals...");
    ESP_LOGI(TAG, "No printing during recording to avoid timing overhead.\n");
    
    // Create mutex for thread-safe access to timestamp array
    s_timer_mutex = xSemaphoreCreateMutex();
    if (s_timer_mutex == NULL)
    {
        ESP_LOGE(TAG, "Failed to create mutex!");
        return;
    }
    
    // Initialize timer
    const uint64_t TIMER_PERIOD_US = 2000000;  // 2 seconds in microseconds
    esp_timer_create_args_t timer_args;
    timer_args.callback = &timer_precision_callback;
    timer_args.arg = NULL;
    timer_args.dispatch_method = ESP_TIMER_TASK;  // Execute callback in timer task
    timer_args.name = "precision_timer";
    timer_args.skip_unhandled_events = false;  // Don't skip events
    
    ret = esp_timer_create(&timer_args, &s_timer_handle);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to create timer: %s", esp_err_to_name(ret));
        vSemaphoreDelete(s_timer_mutex);
        return;
    }
    
    // Start periodic timer
    ret = esp_timer_start_periodic(s_timer_handle, TIMER_PERIOD_US);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to start timer: %s", esp_err_to_name(ret));
        esp_timer_delete(s_timer_handle);
        vSemaphoreDelete(s_timer_mutex);
        return;
    }
    
    // Wait for all timestamps to be collected
    ESP_LOGI(TAG, "Timer started. Waiting for %d timestamps...", TIMESTAMP_COUNT);
    while (!s_timer_test_complete)
    {
        vTaskDelay(100 / portTICK_PERIOD_MS);  // Check every 100ms
    }
    
    // Wait a bit more to ensure timer has stopped
    vTaskDelay(100 / portTICK_PERIOD_MS);
    
    // Print all results
    ESP_LOGI(TAG, "\n========================================");
    ESP_LOGI(TAG, "  Timer Precision Test Results");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Expected interval: 2000000 microseconds (2.000000 seconds)\n");
    
    for (int i = 0; i < TIMESTAMP_COUNT; i++)
    {
        if (i == 0)
        {
            // First timestamp - show absolute time
            ESP_LOGI(TAG, "Timestamp #%2d: %lld microseconds (%.6f seconds) [baseline]",
                     i + 1, s_timestamps[i], s_timestamps[i] / 1000000.0);
        }
        else
        {
            // Calculate interval from previous timestamp
            TinyTimeMark_t interval = s_timestamps[i] - s_timestamps[i - 1];
            int64_t error = interval - 2000000;  // Expected 2 seconds = 2000000 microseconds
            double error_ms = error / 1000.0;
            
            ESP_LOGI(TAG, "Timestamp #%2d: %lld microseconds (%.6f seconds) | "
                     "Interval: %lld us (%.6f s) | Error: %lld us (%.3f ms)",
                     i + 1, 
                     s_timestamps[i], 
                     s_timestamps[i] / 1000000.0,
                     interval,
                     interval / 1000000.0,
                     error,
                     error_ms);
        }
    }
    
    // Calculate statistics
    ESP_LOGI(TAG, "\n--- Statistics ---");
    int64_t total_interval = s_timestamps[TIMESTAMP_COUNT - 1] - s_timestamps[0];
    int64_t expected_total = 2000000 * (TIMESTAMP_COUNT - 1);
    int64_t total_error = total_interval - expected_total;
    
    ESP_LOGI(TAG, "Total time: %lld microseconds (%.6f seconds)", 
             total_interval, total_interval / 1000000.0);
    ESP_LOGI(TAG, "Expected total: %lld microseconds (%.6f seconds)", 
             expected_total, expected_total / 1000000.0);
    ESP_LOGI(TAG, "Total error: %lld microseconds (%.3f milliseconds)", 
             total_error, total_error / 1000.0);
    
    // Calculate average interval
    double avg_interval = (double)total_interval / (TIMESTAMP_COUNT - 1);
    ESP_LOGI(TAG, "Average interval: %.6f seconds (%.3f microseconds)", 
             avg_interval / 1000000.0, avg_interval);
    
    // Cleanup
    esp_timer_delete(s_timer_handle);
    vSemaphoreDelete(s_timer_mutex);
    
    ESP_LOGI(TAG, "\n========================================");
    ESP_LOGI(TAG, "  Test Complete");
    ESP_LOGI(TAG, "========================================\n");
    
    // Main loop: Keep running
    while (1)
    {
        vTaskDelay(10000 / portTICK_PERIOD_MS);
    }
}

#ifdef __cplusplus
}
#endif


```

## 结果

```txt
I (25) boot: ESP-IDF v6.0-dev-1833-g758939caec 2nd stage bootloader
I (25) boot: compile time Nov  4 2025 23:13:16
I (25) boot: Multicore bootloader
I (27) boot: chip revision: v0.2
I (30) boot: efuse block revision: v1.3
I (33) qio_mode: Enabling default flash chip QIO
I (38) boot.esp32s3: Boot SPI Speed : 80MHz
I (41) boot.esp32s3: SPI Mode       : QIO
I (45) boot.esp32s3: SPI Flash Size : 16MB
I (49) boot: Enabling RNG early entropy source...
I (54) boot: Partition Table:
I (56) boot: ## Label            Usage          Type ST Offset   Length
I (62) boot:  0 nvs              WiFi data        01 02 00009000 00006000
I (69) boot:  1 phy_init         RF data          01 01 0000f000 00001000
I (75) boot:  2 factory          factory app      00 00 00010000 001f0000
I (82) boot:  3 vfs              Unknown data     01 81 00200000 00a00000
I (89) boot:  4 storage          Unknown data     01 82 00c00000 00400000
I (95) boot: End of partition table
I (98) esp_image: segment 0: paddr=00010020 vaddr=3c0b0020 size=1df80h (122752) map
I (124) esp_image: segment 1: paddr=0002dfa8 vaddr=3fc99300 size=02070h (  8304) load
I (126) esp_image: segment 2: paddr=00030020 vaddr=42000020 size=a26fch (665340) map
I (227) esp_image: segment 3: paddr=000d2724 vaddr=3fc9b370 size=030e0h ( 12512) load
I (229) esp_image: segment 4: paddr=000d580c vaddr=40374000 size=152ech ( 86764) load
I (247) esp_image: segment 5: paddr=000eab00 vaddr=50000000 size=00020h (    32) load
I (256) boot: Loaded app from partition at offset 0x10000
I (256) boot: Disabling RNG early entropy source...
I (266) octal_psram: vendor id    : 0x0d (AP)
I (267) octal_psram: dev id       : 0x02 (generation 3)
I (267) octal_psram: density      : 0x03 (64 Mbit)
I (269) octal_psram: good-die     : 0x01 (Pass)
I (273) octal_psram: Latency      : 0x01 (Fixed)
I (277) octal_psram: VCC          : 0x01 (3V)
I (281) octal_psram: SRF          : 0x01 (Fast Refresh)
I (286) octal_psram: BurstType    : 0x01 (Hybrid Wrap)
I (291) octal_psram: BurstLen     : 0x01 (32 Byte)
I (296) octal_psram: Readlatency  : 0x02 (10 cycles@Fixed)
I (301) octal_psram: DriveStrength: 0x00 (1/1)
I (306) MSPI Timing: PSRAM timing tuning index: 5
I (310) esp_psram: Found 8MB PSRAM device
I (313) esp_psram: Speed: 80MHz
I (316) cpu_start: Multicore app
I (752) esp_psram: SPI SRAM memory test OK
I (760) cpu_start: GPIO 44 and 43 are used as console UART I/O pins
I (761) cpu_start: Pro cpu start user code
I (761) cpu_start: cpu freq: 240000000 Hz
I (762) app_init: Application information:
I (766) app_init: Project name:     AIoTNode
I (770) app_init: App version:      0a79117-dirty
I (775) app_init: Compile time:     Nov  4 2025 23:13:38
I (780) app_init: ELF file SHA256:  a5e0090b4...
I (784) app_init: ESP-IDF:          v6.0-dev-1833-g758939caec
I (789) efuse_init: Min chip rev:     v0.0
I (793) efuse_init: Max chip rev:     v0.99 
I (797) efuse_init: Chip rev:         v0.2
I (801) heap_init: Initializing. RAM available for dynamic allocation:
I (807) heap_init: At 3FCA2918 len 00046DF8 (283 KiB): RAM
I (812) heap_init: At 3FCE9710 len 00005724 (21 KiB): RAM
I (818) heap_init: At 3FCF0000 len 00008000 (32 KiB): DRAM
I (823) heap_init: At 600FE000 len 00001FE8 (7 KiB): RTCRAM
I (828) esp_psram: Adding pool of 8192K of PSRAM memory to heap allocator
I (835) spi_flash: detected chip: boya
I (838) spi_flash: flash io: qio
I (841) sleep_gpio: Configure to isolate all GPIO pins in sleep state
I (847) sleep_gpio: Enable automatic switching of GPIO sleep configuration
I (854) main_task: Started on CPU0
I (878) esp_psram: Reserving pool of 32K of internal memory for DMA/internal allocations
I (878) main_task: Calling app_main()
I (883) tiny_time_test: ========================================
I (884) tiny_time_test:   tiny_time Module Test Program
I (889) tiny_time_test: ========================================
I (895) tiny_time_test: Initializing WiFi...
I (900) pp: pp rom version: e7ae62f
I (902) net80211: net80211 rom version: e7ae62f
I (907) wifi:wifi driver task: 3fcaf644, prio:23, stack:6656, core=0
I (915) wifi:wifi firmware version: 14da9b7
I (916) wifi:wifi certification version: v7.0
I (920) wifi:config NVS flash: enabled
I (924) wifi:config nano formatting: disabled
I (928) wifi:Init data frame dynamic rx buffer num: 32
I (933) wifi:Init static rx mgmt buffer num: 5
I (937) wifi:Init management short buffer num: 32
I (941) wifi:Init dynamic tx buffer num: 32
I (945) wifi:Init static tx FG buffer num: 2
I (949) wifi:Init static rx buffer size: 1600
I (953) wifi:Init static rx buffer num: 10
I (957) wifi:Init dynamic rx buffer num: 32
I (961) wifi_init: rx ba win: 6
I (964) wifi_init: accept mbox: 6
I (967) wifi_init: tcpip mbox: 32
I (970) wifi_init: udp mbox: 6
I (973) wifi_init: tcp mbox: 6
I (975) wifi_init: tcp tx win: 5760
I (979) wifi_init: tcp rx win: 5760
I (982) wifi_init: tcp mss: 1440
I (985) wifi_init: WiFi IRAM OP enabled
I (988) wifi_init: WiFi RX IRAM OP enabled
I (992) NODE-WIFI: Setting WiFi configuration SSID NTUSECURE...
I (999) phy_init: phy_version 701,f4f1da3a,Mar  3 2025,15:50:10
I (1037) wifi:mode : sta (cc:ba:97:09:a7:50)
I (1038) wifi:enable tsf
I (1039) tiny_time_test: WiFi initialized successfully
I (1040) tiny_time_test: Waiting for WiFi connection...
I (1107) wifi:new:<1,0>, old:<1,0>, ap:<255,255>, sta:<1,0>, prof:1, snd_ch_cfg:0x0
I (1108) wifi:state: init -> auth (0xb0)
I (1111) wifi:state: auth -> assoc (0x0)
I (1115) wifi:state: assoc -> run (0x10)
I (1430) wifi:connected with NTUSECURE, aid = 2, channel 1, BW20, bssid = a8:9d:21:3c:12:b1
I (1430) wifi:security: WPA2-ENT, phy: bgn, rssi: -66
I (1432) wifi:pm start, type: 1

I (1435) wifi:dp: 1, bi: 104448, li: 2, scale listen interval from 307200 us to 208896 us
I (1443) wifi:set rx beacon pti, rx_bcn_pti: 0, bcn_timeout: 25000, mt_pti: 0, mt_time: 10000
I (1459) wifi:<ba-add>idx:0 (ifx:0, a8:9d:21:3c:12:b1), tid:0, ssn:1200, winSize:64
I (1488) wifi:AP's beacon interval = 104448 us, DTIM period = 1
I (2467) esp_netif_handlers: sta ip: 10.91.180.236, mask: 255.255.0.0, gw: 10.91.255.254
I (2467) tiny_time_test: WiFi connected!
I (2467) tiny_time_test: 
--- Test 1: Get Running Time ---
I (2473) tiny_time_test: Running time: 1644833 microseconds
I (2478) tiny_time_test: Running time: 1.645 seconds
I (2483) tiny_time_test: 
--- Test 2: Sync Time with Timezone ---
I (2489) tiny_time_test: Syncing time with timezone CST-8...
I (2494) NTP_SYNC: Initializing SNTP
I (2498) NTP_SYNC: Waiting for system time to be set... (1/15)
I (4503) NTP_SYNC: Waiting for system time to be set... (2/15)
I (4715) NTP_SYNC: Time synchronized!
I (6503) NTP_SYNC: System time is set.
I (6503) NTP_SYNC: Current time: Tue Nov 04 23:15:34 2025
I (6503) tiny_time_test: Waiting for time synchronization...
I (11506) tiny_time_test: 
--- Test 3: Get Current DateTime ---
I (11506) TIME: Current Time: 2025-11-04 23:15:39.003179
I (11506) tiny_time_test: 
--- Test 4: Measure Time Elapsed ---
I (11511) tiny_time_test: Time elapsed: 9038406 microseconds
I (11517) tiny_time_test: Time elapsed: 9.038 seconds
I (11521) tiny_time_test: 
========================================
I (11527) tiny_time_test:   Initial Tests Completed
I (11532) tiny_time_test: ========================================

I (11538) tiny_time_test: 
========================================
I (11544) tiny_time_test:   Timer Precision Test
I (11548) tiny_time_test: ========================================
I (11554) tiny_time_test: Recording 15 timestamps at 2-second intervals...
I (11561) tiny_time_test: No printing during recording to avoid timing overhead.

I (11568) tiny_time_test: Timer started. Waiting for 15 timestamps...
I (41674) tiny_time_test: 
========================================
I (41674) tiny_time_test:   Timer Precision Test Results
I (41674) tiny_time_test: ========================================
I (41680) tiny_time_test: Expected interval: 2000000 microseconds (2.000000 seconds)

I (41687) tiny_time_test: Timestamp # 1: 12740383 microseconds (12.740383 seconds) [baseline]
I (41696) tiny_time_test: Timestamp # 2: 14740381 microseconds (14.740381 seconds) | Interval: 1999998 us (1.999998 s) | Error: -2 us (-0.002 ms)
I (41708) tiny_time_test: Timestamp # 3: 16740383 microseconds (16.740383 seconds) | Interval: 2000002 us (2.000002 s) | Error: 2 us (0.002 ms)
I (41721) tiny_time_test: Timestamp # 4: 18740383 microseconds (18.740383 seconds) | Interval: 2000000 us (2.000000 s) | Error: 0 us (0.000 ms)
I (41733) tiny_time_test: Timestamp # 5: 20740383 microseconds (20.740383 seconds) | Interval: 2000000 us (2.000000 s) | Error: 0 us (0.000 ms)
I (41746) tiny_time_test: Timestamp # 6: 22740383 microseconds (22.740383 seconds) | Interval: 2000000 us (2.000000 s) | Error: 0 us (0.000 ms)
I (41759) tiny_time_test: Timestamp # 7: 24740382 microseconds (24.740382 seconds) | Interval: 1999999 us (1.999999 s) | Error: -1 us (-0.001 ms)
I (41771) tiny_time_test: Timestamp # 8: 26740383 microseconds (26.740383 seconds) | Interval: 2000001 us (2.000001 s) | Error: 1 us (0.001 ms)
I (41784) tiny_time_test: Timestamp # 9: 28740383 microseconds (28.740383 seconds) | Interval: 2000000 us (2.000000 s) | Error: 0 us (0.000 ms)
I (41797) tiny_time_test: Timestamp #10: 30740383 microseconds (30.740383 seconds) | Interval: 2000000 us (2.000000 s) | Error: 0 us (0.000 ms)
I (41809) tiny_time_test: Timestamp #11: 32740383 microseconds (32.740383 seconds) | Interval: 2000000 us (2.000000 s) | Error: 0 us (0.000 ms)
I (41822) tiny_time_test: Timestamp #12: 34740381 microseconds (34.740381 seconds) | Interval: 1999998 us (1.999998 s) | Error: -2 us (-0.002 ms)
I (41834) tiny_time_test: Timestamp #13: 36740383 microseconds (36.740383 seconds) | Interval: 2000002 us (2.000002 s) | Error: 2 us (0.002 ms)
I (41847) tiny_time_test: Timestamp #14: 38740383 microseconds (38.740383 seconds) | Interval: 2000000 us (2.000000 s) | Error: 0 us (0.000 ms)
I (41860) tiny_time_test: Timestamp #15: 40740381 microseconds (40.740381 seconds) | Interval: 1999998 us (1.999998 s) | Error: -2 us (-0.002 ms)
I (41872) tiny_time_test: 
--- Statistics ---
I (41877) tiny_time_test: Total time: 27999998 microseconds (27.999998 seconds)
I (41884) tiny_time_test: Expected total: 28000000 microseconds (28.000000 seconds)
I (41891) tiny_time_test: Total error: -2 microseconds (-0.002 milliseconds)
I (41898) tiny_time_test: Average interval: 2.000000 seconds (1999999.857 microseconds)
I (41906) tiny_time_test: 
========================================
I (41912) tiny_time_test:   Test Complete
I (41915) tiny_time_test: ========================================

```
