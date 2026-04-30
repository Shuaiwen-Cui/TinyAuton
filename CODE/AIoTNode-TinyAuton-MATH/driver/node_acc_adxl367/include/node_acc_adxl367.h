/**
 * @file node_acc_adxl367.h
 * @brief ADXL367 (I2C): init, range/ODR, modes, XYZ/temp, FIFO, INT map, activity/inactivity.
 *
 * I2C via driver/node_i2c. Registers: driver/ref-adxl367. Tests/integration: node_acc_adxl367_test.h.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "driver/i2c_master.h"
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Default 7-bit I2C address when ASEL is high (see datasheet). */
#define NODE_ACC_ADXL367_I2C_ADDR_0X53 0x53u
/** Alternate 7-bit I2C address when ASEL is low. */
#define NODE_ACC_ADXL367_I2C_ADDR_0X1D 0x1Du

/** Measurement range (FILTER_CTL[7:6]). */
typedef enum {
    NODE_ACC_ADXL367_RANGE_2G = 0u,
    NODE_ACC_ADXL367_RANGE_4G = 1u,
    NODE_ACC_ADXL367_RANGE_8G = 2u,
} node_acc_adxl367_range_t;

/** Output data rate (FILTER_CTL[2:0]). Matches ADXL367 ODR encoding. */
typedef enum {
    NODE_ACC_ADXL367_ODR_12_5_HZ = 0u,
    NODE_ACC_ADXL367_ODR_25_HZ = 1u,
    NODE_ACC_ADXL367_ODR_50_HZ = 2u,
    NODE_ACC_ADXL367_ODR_100_HZ = 3u,
    NODE_ACC_ADXL367_ODR_200_HZ = 4u,
    NODE_ACC_ADXL367_ODR_400_HZ = 5u,
} node_acc_adxl367_odr_t;

/** Power/measurement mode (POWER_CTL[1:0]). */
typedef enum {
    NODE_ACC_ADXL367_MODE_STANDBY = 0u,
    NODE_ACC_ADXL367_MODE_MEASURE = 2u,
} node_acc_adxl367_mode_t;

/**
 * Temperature / ADC channel selection (TEMP_CTL / ADC_CTL).
 * adxl367_adc_read_en() in ref clears TEMP when toggling ADC; we only mask ADC_CTL for ADC_EN
 * so TEMP_ONLY does not use that helper verbatim.
 */
typedef enum {
    /** Accel only. */
    NODE_ACC_ADXL367_MEAS_ACCEL_ONLY = 0u,
    /** Temp only (API). */
    NODE_ACC_ADXL367_MEAS_TEMP_ONLY,
    /** Accel + temp. */
    NODE_ACC_ADXL367_MEAS_ACCEL_AND_TEMP,
} node_acc_adxl367_meas_mode_t;

/** FIFO mode (FIFO_CONTROL[1:0]). See adxl367_set_fifo_mode. */
typedef enum {
    NODE_ACC_ADXL367_FIFO_DISABLED = 0u,
    NODE_ACC_ADXL367_FIFO_OLDEST_SAVED = 1u,
    NODE_ACC_ADXL367_FIFO_STREAM = 2u,
    NODE_ACC_ADXL367_FIFO_TRIGGERED = 3u,
} node_acc_adxl367_fifo_mode_t;

/** FIFO channel / format (FIFO_CONTROL[6:3]). See adxl367_set_fifo_format. */
typedef enum {
    NODE_ACC_ADXL367_FIFO_FMT_XYZ = 0u,
    NODE_ACC_ADXL367_FIFO_FMT_X = 1u,
    NODE_ACC_ADXL367_FIFO_FMT_Y = 2u,
    NODE_ACC_ADXL367_FIFO_FMT_Z = 3u,
    NODE_ACC_ADXL367_FIFO_FMT_XYZT = 4u,
} node_acc_adxl367_fifo_format_t;

/**
 * Device instance. @c i2c from @c i2c_add_device() after @c i2c_bus_init().
 */
typedef struct node_acc_adxl367_dev {
    i2c_master_dev_handle_t i2c;
    node_acc_adxl367_range_t range;
    node_acc_adxl367_odr_t odr;
    node_acc_adxl367_mode_t mode;
    node_acc_adxl367_meas_mode_t meas_mode;
    node_acc_adxl367_fifo_mode_t fifo_mode;
    node_acc_adxl367_fifo_format_t fifo_format;
} node_acc_adxl367_dev_t;

/** INTMAP bits; layout matches ref adxl367_int_map. */
typedef struct node_acc_adxl367_int_map {
    uint8_t err_fuse;
    uint8_t err_user_regs;
    uint8_t kpalv_timer;
    uint8_t temp_adc_hi;
    uint8_t temp_adc_low;
    uint8_t tap_two;
    uint8_t tap_one;
    uint8_t int_low;
    uint8_t awake;
    uint8_t inact;
    uint8_t act;
    uint8_t fifo_overrun;
    uint8_t fifo_watermark;
    uint8_t fifo_ready;
    uint8_t data_ready;
} node_acc_adxl367_int_map_t;

/** ACT_INACT_CTL ref mode: ABS fixed vs REL referenced baseline (ref adxl367_setup_activity_detection). */
typedef enum {
    NODE_ACC_ADXL367_AI_REF_ABS = 0u,
    NODE_ACC_ADXL367_AI_REF_REL = 1u,
} node_acc_adxl367_ai_ref_t;

/** STATUS (0x0B). */
#define NODE_ACC_ADXL367_STATUS_DATA_RDY (1u << 0)
#define NODE_ACC_ADXL367_STATUS_FIFO_RDY (1u << 1)
#define NODE_ACC_ADXL367_STATUS_FIFO_WATERMARK (1u << 2)
#define NODE_ACC_ADXL367_STATUS_FIFO_OVERRUN (1u << 3)
#define NODE_ACC_ADXL367_STATUS_ACT (1u << 4)
#define NODE_ACC_ADXL367_STATUS_INACT (1u << 5)
#define NODE_ACC_ADXL367_STATUS_AWAKE (1u << 6)

/** g to THRESH_ACT LSB at +/-2g scale intent (~4000 LSB/g, ref path). */
#define NODE_ACC_ADXL367_THRESH_ACT_LSB_FROM_G(g) ((uint16_t)((float)(g) * 4000.0f))

/**
 * @brief Soft-reset, probe, default 2g / 100 Hz, measure mode, then @c NODE_ACC_ADXL367_MEAS_ACCEL_ONLY.
 */
esp_err_t node_acc_adxl367_init(node_acc_adxl367_dev_t *dev);

esp_err_t node_acc_adxl367_soft_reset(node_acc_adxl367_dev_t *dev);
esp_err_t node_acc_adxl367_probe(node_acc_adxl367_dev_t *dev);

esp_err_t node_acc_adxl367_set_range(node_acc_adxl367_dev_t *dev, node_acc_adxl367_range_t range);
esp_err_t node_acc_adxl367_set_odr(node_acc_adxl367_dev_t *dev, node_acc_adxl367_odr_t odr);
esp_err_t node_acc_adxl367_set_mode(node_acc_adxl367_dev_t *dev, node_acc_adxl367_mode_t mode);

/**
 * @brief Configure TEMP_CTL / ADC_CTL per mode. Stays in current power mode (typically measure).
 */
esp_err_t node_acc_adxl367_set_measurement_mode(node_acc_adxl367_dev_t *dev,
                                                 node_acc_adxl367_meas_mode_t mode);

/** Raw XYZ; ESP_ERR_INVALID_STATE in MEAS_TEMP_ONLY. Same as ref adxl367_get_raw_xyz (DATA_RDY, XDATA_H). */
esp_err_t node_acc_adxl367_read_xyz_raw(node_acc_adxl367_dev_t *dev, int16_t *x, int16_t *y, int16_t *z);

/**
 * @brief Raw temperature (14-bit sign-extended). Requires TEMP_EN; not allowed in @c NODE_ACC_ADXL367_MEAS_ACCEL_ONLY.
 */
esp_err_t node_acc_adxl367_read_temp_raw(node_acc_adxl367_dev_t *dev, int16_t *raw);

/** Celsius using ref-adxl367 adxl367_temp_conv coefficients. */
esp_err_t node_acc_adxl367_read_temp_celsius(node_acc_adxl367_dev_t *dev, float *celsius);

/** Read STATUS register (0x0B). */
esp_err_t node_acc_adxl367_read_status(node_acc_adxl367_dev_t *dev, uint8_t *status);

/**
 * @brief Route interrupt sources to INT1 or INT2 (INTMAP1 / INTMAP2). See @c adxl367_int_map.
 * @param int_pin 1 = INT1, 2 = INT2.
 */
esp_err_t node_acc_adxl367_int_map(node_acc_adxl367_dev_t *dev, uint8_t int_pin,
                                   const node_acc_adxl367_int_map_t *map);

/** Activity: ref adxl367_setup_activity_detection. threshold 13-bit; time_act_samples at ODR. */
esp_err_t node_acc_adxl367_setup_activity_detection(node_acc_adxl367_dev_t *dev,
                                                    node_acc_adxl367_ai_ref_t ref_mode,
                                                    uint16_t threshold,
                                                    uint8_t time_act_samples);

/** Inactivity: ref adxl367_setup_inactivity_detection. */
esp_err_t node_acc_adxl367_setup_inactivity_detection(node_acc_adxl367_dev_t *dev,
                                                      node_acc_adxl367_ai_ref_t ref_mode,
                                                      uint16_t threshold,
                                                      uint16_t time_inact_samples);

/** Clear ACT/INACT enables (ACT_INACT_CTL[3:0]). */
esp_err_t node_acc_adxl367_act_inact_disable(node_acc_adxl367_dev_t *dev);

/** AXIS_MASK bits [2:0]: set bit disables axis. */
esp_err_t node_acc_adxl367_set_activity_axes_enabled(node_acc_adxl367_dev_t *dev, bool enable_x,
                                                   bool enable_y, bool enable_z);

/** ACT_INACT_CTL LINKLOOP 0-3 (datasheet). */
esp_err_t node_acc_adxl367_set_act_inact_linkloop(node_acc_adxl367_dev_t *dev, uint8_t linkloop_mode_0_to_3);

/**
 * @brief FIFO setup: mode, format, watermark sample-sets count, 14-bit+CH ID read mode (adxl367_fifo_setup).
 * @param sample_sets_nb Watermark in sample sets (e.g. 3 entries per set for XYZ).
 */
esp_err_t node_acc_adxl367_fifo_setup(node_acc_adxl367_dev_t *dev,
                                      node_acc_adxl367_fifo_mode_t mode,
                                      node_acc_adxl367_fifo_format_t format,
                                      uint16_t sample_sets_nb);

esp_err_t node_acc_adxl367_fifo_disable(node_acc_adxl367_dev_t *dev);

/** FIFO entry count (10-bit), ref adxl367_get_nb_of_fifo_entries. */
esp_err_t node_acc_adxl367_fifo_get_entry_count(node_acc_adxl367_dev_t *dev, uint16_t *count);

/**
 * @brief Read and parse FIFO for @c NODE_ACC_ADXL367_FIFO_FMT_XYZ (14b+CH ID).
 * Fills up to @p max_triplets complete XYZ samples; @p out_triplets may be less if FIFO is partial.
 */
esp_err_t node_acc_adxl367_fifo_drain_xyz(node_acc_adxl367_dev_t *dev, int16_t *x, int16_t *y, int16_t *z,
                                          size_t max_triplets, size_t *out_triplets);

#ifdef __cplusplus
}
#endif
