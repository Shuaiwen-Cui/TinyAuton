/**
 * @file tiny_time.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Submodule for TinyToolbox - source file
 * @version 1.0
 * @date 2025-04-10
 * @copyright Copyright (c) 2025
 *
 */

/* ================================ DEPENDENCIES ================================ */
#include "tiny_time.h" // Time

/* ================================ DEFINITIONS ================================= */
/* CONFIGURATIONS */

/* TAGS */
static const char *TAG_SNTP = "NTP_SYNC";
static const char *TAG_TIME = "TIME";

/* ================================ FUNCTIONS =================================== */
/* LOCAL RUNNING TIME IN MICROSECONDS */
/**
 * @brief Get the running time in microseconds
 * @return TinyTimeMark_t
 */
TinyTimeMark_t tiny_get_running_time(void)
{
    return esp_timer_get_time();
}

/* WORLD CURRENT TIME - SNTP */
/**
 * @brief Callback function for time synchronization notification
 * @param tv Pointer to the timeval structure containing the synchronized time
 * @return None
 */
void time_sync_notification_cb(struct timeval *tv)
{
    ESP_LOGI(TAG_SNTP, "Time synchronized!");
}

/**
 * @brief Initialize SNTP
 * @return None
 */
void initialize_sntp(void)
{
    ESP_LOGI(TAG_SNTP, "Initializing SNTP");
    sntp_setoperatingmode(SNTP_OPMODE_POLL);
    sntp_setservername(0, "pool.ntp.org"); // NTP server // pool.ntp.org // ntp.aliyun.com
    sntp_set_time_sync_notification_cb(time_sync_notification_cb);
    sntp_init();
}

/**
 * @brief Obtain the current time with timezone
 * @param timezone_str Timezone string (e.g., "CST-8")
 * @note The timezone string should be in the format "TZ=GMT+/-HH:MM"
 * @note To use this function, in application, after internet connection, insert "sync_time_with_timezone("CST-8");" will do
 * @return None
 */
void sync_time_with_timezone(const char *timezone_str)
{
    // Set system timezone
    setenv("TZ", timezone_str, 1);
    tzset();

    // Initialize SNTP and start time sync
    initialize_sntp();

    // Wait for system time to be set
    time_t now = 0;
    struct tm timeinfo = { 0 };
    int retry = 0;
    const int retry_count = 15;

    while (timeinfo.tm_year < (2020 - 1900) && ++retry < retry_count) {
        ESP_LOGI(TAG_SNTP, "Waiting for system time to be set... (%d/%d)", retry, retry_count);
        vTaskDelay(2000 / portTICK_PERIOD_MS);
        time(&now);
        localtime_r(&now, &timeinfo);
    }

    if (timeinfo.tm_year >= (2020 - 1900)) {
        rtc_set_time(timeinfo.tm_year + 1900, timeinfo.tm_mon + 1, timeinfo.tm_mday,
                    timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec); // defined in esp_rtc.c
        ESP_LOGI(TAG_SNTP, "System time is set.");
    } else {
        ESP_LOGW(TAG_SNTP, "Failed to sync time.");
        return;
    }

    // Log current local time
    ESP_LOGI(TAG_SNTP, "Current time: %s", asctime(&timeinfo));

    // vTaskDelay(10000 / portTICK_PERIOD_MS); // Wait for 10 second
    // rtc_get_time(); // uncomment to check the RTC time
    // ESP_LOGI(TAG_SNTP, "Current RTC time: %04d-%02d-%02d %02d:%02d:%02d",
    //          calendar.year, calendar.month, calendar.date,
    //          calendar.hour, calendar.min, calendar.sec); // uncomment to check the RTC time

}

/* WORLD CURRENT TIME - GET TIME */
/**
 * @name tiny_get_current_time
 * @brief Get the current time as a TinyDateTime_t struct
 * @param print_flag Flag to indicate whether to print the time
 * @return TinyDateTime_t structure containing the current date and time
 */
TinyDateTime_t tiny_get_current_time(bool print_flag)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);  // Get current time (seconds + microseconds)

    time_t now = tv.tv_sec;
    struct tm timeinfo;
    localtime_r(&now, &timeinfo);  // Convert to local time

    TinyDateTime_t result;
    result.year   = timeinfo.tm_year + 1900;
    result.month  = timeinfo.tm_mon + 1;
    result.day    = timeinfo.tm_mday;
    result.hour   = timeinfo.tm_hour;
    result.minute = timeinfo.tm_min;
    result.second = timeinfo.tm_sec;
    result.microsecond = tv.tv_usec;

    if (print_flag)
    {
        ESP_LOGI(TAG_TIME, "Current Time: %04d-%02d-%02d %02d:%02d:%02d.%06ld",
                 result.year, result.month, result.day,
                 result.hour, result.minute, result.second, tv.tv_usec);
    }

    return result;
}