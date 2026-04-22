/**
 * @file node_rng.c
 * @author
 * @brief This is the source file for the node_rng component.
 * @version 1.0
 * @date 2025-10-21
 * @ref Alientek RNG Driver
 * @copyright Copyright (c) 2024
 *
 */

#include "node_rng.h"

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif