/**
 * @file rng.c
 * @author
 * @brief This is the source file for the RNG component.
 * @version 1.0
 * @date 2024-11-19
 * @ref Alientek RNG Driver
 * @copyright Copyright (c) 2024
 *
 */

#include "rng.h"

/**
 * @brief       Get a random number
 * @param       None
 * @retval      Random number (32-bit)
 */
uint32_t rng_get_random_num(void)
{
    uint32_t randomnum;

    randomnum = esp_random();

    return randomnum;
}

/**
 * @brief       Get a random number within a specific range
 * @param       min,max: Minimum and maximum values
 * @retval      Random number (rval), satisfying: min <= rval <= max
 */
int rng_get_random_range(int min, int max)
{
    uint32_t randomnum;

    randomnum = esp_random();

    return randomnum % (max - min + 1) + min;
}