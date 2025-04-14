/**
 * @file TinyTime.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Submodule for TinyToolbox - header file
 * @version 1.0
 * @date 2025-04-10
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __TINYTIME__
#define __TINYTIME__

#ifdef __cplusplus
extern "C"
{
#endif

/* CONFIGURATIONS */

/* ================================ DEPENDENCIES ================================ */
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_sntp.h"
// customized drivers
#include "esp_rtc.h"

/* ================================ DEFINITIONS ================================= */
typedef TickType_t TinyTimeMark_t;

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
    long microsecond;
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
 * @param timezone_str Timezone string (e.g., "CST-8")
 * @note The timezone string should be in the format "TZ=GMT+/-HH:MM"
 * @note To use this function, in application, insert "sync_time_with_timezone("CST-8");" will do
 * @return None
 */
void sync_time_with_timezone(const char *timezone_str);

/* WORLD CURRENT TIME - GET TIME */
/**
 * @name tiny_get_current_time
 * @brief Get the current time as a TinyDateTime_t struct
 * @param print_flag Flag to indicate whether to print the time
 * @return TinyDateTime_t structure containing the current date and time
 */
TinyDateTime_t tiny_get_current_time(bool print_flag);

#ifdef __cplusplus
}
#endif

#endif /* __TINYTIME__ */