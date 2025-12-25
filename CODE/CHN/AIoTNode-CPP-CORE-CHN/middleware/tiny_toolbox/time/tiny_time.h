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