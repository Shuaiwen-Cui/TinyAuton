/**
 * @file node_rtc.h
 * @author
 * @brief Header file for the RTC driver
 * @version 1.0
 * @date 2025-10-21
 * @ref Alienteck RTC Driver
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

#include <sys/time.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

    /* Time structure, including year, month, day, week, hour, minute, and second */
    typedef struct
    {
        uint8_t hour; /* Hour */
        uint8_t min;  /* Minute */
        uint8_t sec;  /* Second */
        /* Gregorian year, month, day, and week */
        uint16_t year; /* Year */
        uint8_t month; /* Month */
        uint8_t date;  /* Day */
        uint8_t week;  /* Weekday */
    } _calendar_obj;

    extern _calendar_obj calendar; /* Time structure */

    /* Function declarations */
    /**
     * @brief       Set the RTC time
     * @param       year    : Year
     * @param       mon     : Month
     * @param       mday    : Day
     * @param       hour    : Hour
     * @param       min     : Minute
     * @param       sec     : Second
     * @retval      None
     */
    void rtc_set_time(int year, int mon, int mday, int hour, int min, int sec); /* Set time */

    /**
     * @brief       Get the current time
     * @param       None
     * @retval      None
     */
    void rtc_get_time(void); /* Get time */

    /**
     * @brief       Convert year, month, and day to the day of the week
     * @note        Calculates the weekday based on the Gregorian calendar.
     *              Utilizes the Kim Larson formula for calculation.
     *              For more details, refer to:
     *              https://www.cnblogs.com/fengbohello/p/3264300.html
     * @param       year : Year
     * @param       month: Month
     * @param       day  : Day
     * @retval      0: Sunday; 1 ~ 6: Monday ~ Saturday
     */
    uint8_t rtc_get_week(uint16_t year, uint8_t month, uint8_t day); /* Get the weekday */

#ifdef __cplusplus
}
#endif