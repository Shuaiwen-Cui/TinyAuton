/**
 * @file rng.h
 * @author
 * @brief This is the header file for the RNG component.
 * @version 1.0
 * @date 2024-11-19
 * @ref Alientek RNG Driver
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __RNG_H__
#define __RNG_H__

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <sys/param.h>
#include "esp_attr.h"
#include "esp_cpu.h"
#include "soc/wdev_reg.h"
#include "esp_random.h"
#include "esp_private/esp_clk.h"

    /* Function Declarations */

    /**
     * @brief       Get a random number
     * @param       None
     * @retval      Random number (32-bit)
     */
    uint32_t rng_get_random_num(void); /* Get a random number */

    /**
     * @brief       Get a random number within a specific range
     * @param       min,max: Minimum and maximum values
     * @retval      Random number (rval), satisfying: min <= rval <= max
     */
    int rng_get_random_range(int min, int max); /* Get a random number within a specific range */

#ifdef __cplusplus
}
#endif

#endif /* __RNG_H__ */