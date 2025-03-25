/**
 * @file esp_rtc.c
 * ! Do not name as rtc.c, as it will conflict with the built-in rtc.c
 * @author
 * @brief This file contains the implementation of the RTC driver
 * @version 1.0
 * @date 2024-11-18
 * @ref Alienteck RTC Driver
 *
 */

#include "esp_rtc.h"

_calendar_obj calendar; /* Time structure */

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
void rtc_set_time(int year, int mon, int mday, int hour, int min, int sec)
{
    struct tm datetime;
    /* Set time */
    datetime.tm_year = year - 1900;
    datetime.tm_mon = mon - 1;
    datetime.tm_mday = mday;
    datetime.tm_hour = hour;
    datetime.tm_min = min;
    datetime.tm_sec = sec;
    datetime.tm_isdst = -1;
    /* Get total seconds since 1970-01-01 */
    time_t second = mktime(&datetime);
    struct timeval val = {.tv_sec = second, .tv_usec = 0};
    /* Set current time */
    settimeofday(&val, NULL);
}

/**
 * @brief       Get the current time
 * @param       None
 * @retval      None
 */
void rtc_get_time(void)
{
    struct tm *datetime;
    time_t second;
    /* Get the time elapsed since (1970-01-01 00:00:00 UTC) in seconds */
    time(&second);
    datetime = localtime(&second);

    calendar.hour = datetime->tm_hour; /* Hour */
    calendar.min = datetime->tm_min;   /* Minute */
    calendar.sec = datetime->tm_sec;   /* Second */
    /* Gregorian year, month, day, and week */
    calendar.year = datetime->tm_year + 1900; /* Year */
    calendar.month = datetime->tm_mon + 1;    /* Month */
    calendar.date = datetime->tm_mday;        /* Day */
    /* Weekday */
    calendar.week = rtc_get_week(calendar.year, calendar.month, calendar.date);
}

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
uint8_t rtc_get_week(uint16_t year, uint8_t month, uint8_t day)
{
    uint8_t week = 0;

    if (month < 3)
    {
        month += 12;
        --year;
    }

    week = (day + 1 + 2 * month + 3 * (month + 1) / 5 + year + (year >> 2) - year / 100 + year / 400) % 7;
    return week;
}