/**
 * @file node_acc_adxl355.h
 * @brief ADXL355 (SPI): init, range/ODR, XYZ/temp, DRDY, INT_MAP, FIFO, activity, offset, Phase 5 hardening.
 *
 * SPI framing matches Analog no-OS ref (driver/ref-adxl355). Bus: call @c spi3_init() from
 * node_spi before @ref node_acc_adxl355_init when using default eval wiring (CS=GPIO7, SCK/MOSI/MISO=15/16/17).
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Default SPI clock (Hz); ADXL355 supports up to 10 MHz on SPI. */
#define NODE_ACC_ADXL355_DEFAULT_SPI_HZ 10000000u

/* --- Register addresses (datasheet) --- */

#define NODE_ACC_ADXL355_REG_DEVID_AD 0x00u
#define NODE_ACC_ADXL355_REG_DEVID_MST 0x01u
#define NODE_ACC_ADXL355_REG_PARTID 0x02u
#define NODE_ACC_ADXL355_REG_REVID 0x03u
#define NODE_ACC_ADXL355_REG_STATUS 0x04u
#define NODE_ACC_ADXL355_REG_FIFO_ENTRIES 0x05u
#define NODE_ACC_ADXL355_REG_TEMP2 0x06u
#define NODE_ACC_ADXL355_REG_TEMP1 0x07u
#define NODE_ACC_ADXL355_REG_XDATA3 0x08u
#define NODE_ACC_ADXL355_REG_XDATA2 0x09u
#define NODE_ACC_ADXL355_REG_XDATA1 0x0Au
#define NODE_ACC_ADXL355_REG_YDATA3 0x0Bu
#define NODE_ACC_ADXL355_REG_YDATA2 0x0Cu
#define NODE_ACC_ADXL355_REG_YDATA1 0x0Du
#define NODE_ACC_ADXL355_REG_ZDATA3 0x0Eu
#define NODE_ACC_ADXL355_REG_ZDATA2 0x0Fu
#define NODE_ACC_ADXL355_REG_ZDATA1 0x10u
#define NODE_ACC_ADXL355_REG_FIFO_DATA 0x11u
#define NODE_ACC_ADXL355_REG_OFFSET_X_H 0x1Eu
#define NODE_ACC_ADXL355_REG_OFFSET_X_L 0x1Fu
#define NODE_ACC_ADXL355_REG_OFFSET_Y_H 0x20u
#define NODE_ACC_ADXL355_REG_OFFSET_Y_L 0x21u
#define NODE_ACC_ADXL355_REG_OFFSET_Z_H 0x22u
#define NODE_ACC_ADXL355_REG_OFFSET_Z_L 0x23u
#define NODE_ACC_ADXL355_REG_ACT_EN 0x24u
#define NODE_ACC_ADXL355_REG_ACT_THRESH_H 0x25u
#define NODE_ACC_ADXL355_REG_ACT_THRESH_L 0x26u
#define NODE_ACC_ADXL355_REG_ACT_COUNT 0x27u
#define NODE_ACC_ADXL355_REG_FILTER 0x28u
#define NODE_ACC_ADXL355_REG_FIFO_SAMPLES 0x29u
#define NODE_ACC_ADXL355_REG_INT_MAP 0x2Au
#define NODE_ACC_ADXL355_REG_SYNC 0x2Bu
#define NODE_ACC_ADXL355_REG_RANGE 0x2Cu
#define NODE_ACC_ADXL355_REG_POWER_CTL 0x2Du
#define NODE_ACC_ADXL355_REG_SELF_TEST 0x2Eu
#define NODE_ACC_ADXL355_REG_RESET 0x2Fu

#define NODE_ACC_ADXL355_REG_STATUS_DATA_RDY 0x01u
#define NODE_ACC_ADXL355_REG_STATUS_FIFO_FULL 0x02u
#define NODE_ACC_ADXL355_REG_STATUS_FIFO_OVR 0x04u
#define NODE_ACC_ADXL355_REG_STATUS_ACTIVITY 0x08u
#define NODE_ACC_ADXL355_REG_STATUS_NVM_BUSY 0x10u

#define NODE_ACC_ADXL355_REG_FILTER_HPF_SHIFT 4u
#define NODE_ACC_ADXL355_REG_FILTER_HPF_MASK 0xF0u
#define NODE_ACC_ADXL355_REG_FILTER_ODR_MASK 0x0Fu

#define NODE_ACC_ADXL355_REG_RANGE_RANGE_MASK 0x03u
#define NODE_ACC_ADXL355_REG_RANGE_INT_POL_SHIFT 6u
#define NODE_ACC_ADXL355_REG_RANGE_INT_POL_MASK 0x40u

#define NODE_ACC_ADXL355_REG_POWER_CTL_STANDBY 0x01u
#define NODE_ACC_ADXL355_REG_POWER_CTL_TEMP_OFF 0x02u
#define NODE_ACC_ADXL355_REG_POWER_CTL_DRDY_OFF 0x04u

#define NODE_ACC_ADXL355_REG_RESET_CODE 0x52u

/** Max value for FIFO_SAMPLES (0x29); matches Analog ref ADXL355_MAX_FIFO_SAMPLES_VAL. */
#define NODE_ACC_ADXL355_FIFO_SAMPLES_MAX 0x60u

/** Expected DEVID_AD (register 0x00). */
#define NODE_ACC_ADXL355_EXPECT_DEVID_AD 0xADu
#define NODE_ACC_ADXL355_EXPECT_DEVID_MST 0x1Du
#define NODE_ACC_ADXL355_EXPECT_PARTID 0xEDu

typedef enum {
    NODE_ACC_ADXL355_RANGE_2G = 0x01u,
    NODE_ACC_ADXL355_RANGE_4G = 0x02u,
    NODE_ACC_ADXL355_RANGE_8G = 0x03u,
} node_acc_adxl355_range_t;

/**
 * What to measure / expose (POWER_CTL TEMP_OFF + software policy).
 * Matches Analog @c adxl355_op_mode idea: accel-only clears the temperature path (TEMP_OFF).
 * TEMP_ONLY: chip still runs with temperature enabled (same HW as ACCEL_AND_TEMP); only
 * @ref node_acc_adxl355_read_raw_temp is valid; @ref node_acc_adxl355_read_raw_xyz returns
 * @c ESP_ERR_INVALID_STATE (ADXL355 has no separate “accel off, temp on” power state).
 */
typedef enum {
    NODE_ACC_ADXL355_MEAS_ACCEL_AND_TEMP = 0,
    NODE_ACC_ADXL355_MEAS_ACCEL_ONLY,
    NODE_ACC_ADXL355_MEAS_TEMP_ONLY,
} node_acc_adxl355_meas_mode_t;

typedef enum {
    NODE_ACC_ADXL355_ODR_4000_HZ = 0x00u,
    NODE_ACC_ADXL355_ODR_2000_HZ = 0x01u,
    NODE_ACC_ADXL355_ODR_1000_HZ = 0x02u,
    NODE_ACC_ADXL355_ODR_500_HZ = 0x03u,
    NODE_ACC_ADXL355_ODR_250_HZ = 0x04u,
    NODE_ACC_ADXL355_ODR_125_HZ = 0x05u,
    NODE_ACC_ADXL355_ODR_62_5_HZ = 0x06u,
    NODE_ACC_ADXL355_ODR_31_25_HZ = 0x07u,
    NODE_ACC_ADXL355_ODR_15_625_HZ = 0x08u,
    NODE_ACC_ADXL355_ODR_7_813_HZ = 0x09u,
    NODE_ACC_ADXL355_ODR_3_906_HZ = 0x0Au,
} node_acc_adxl355_odr_t;

typedef enum {
    NODE_ACC_ADXL355_STATUS_DATA_RDY = 0x01u,
    NODE_ACC_ADXL355_STATUS_FIFO_FULL = 0x02u,
    NODE_ACC_ADXL355_STATUS_FIFO_OVR = 0x04u,
    NODE_ACC_ADXL355_STATUS_ACTIVITY = 0x08u,
    NODE_ACC_ADXL355_STATUS_NVM_BUSY = 0x10u,
} node_acc_adxl355_status_t;

typedef struct {
    uint8_t devid_ad;
    uint8_t devid_mst;
    uint8_t partid;
    uint8_t revid;
} node_acc_adxl355_ids_t;

typedef struct {
    int32_t x;
    int32_t y;
    int32_t z;
} node_acc_adxl355_raw_xyz_t;

typedef struct {
    float x;
    float y;
    float z;
} node_acc_adxl355_g_t;

/** INT_MAP (0x2A): route sources to INT1 / INT2 (see datasheet; matches Analog ref bit layout). */
typedef struct {
    bool rdy_int1;
    bool fifo_full_int1;
    bool fifo_ovr_int1;
    bool activity_int1;
    bool rdy_int2;
    bool fifo_full_int2;
    bool fifo_ovr_int2;
    bool activity_int2;
} node_acc_adxl355_int_map_t;

/** RANGE[6] interrupt polarity (INT1/INT2/DRDY pin behavior). */
typedef enum {
    NODE_ACC_ADXL355_INT_POL_ACTIVE_LOW = 0,
    NODE_ACC_ADXL355_INT_POL_ACTIVE_HIGH = 1,
} node_acc_adxl355_int_pol_t;

/** ACT_EN (0x24): enable activity detection per axis (datasheet bit order). */
typedef struct {
    bool x;
    bool y;
    bool z;
} node_acc_adxl355_act_en_t;

/** Offset registers: 16-bit per axis, big-endian in SPI (Analog @c adxl355_set_offset). */
typedef struct {
    uint16_t x;
    uint16_t y;
    uint16_t z;
} node_acc_adxl355_offset_t;

/**
 * SPI device parameters. @a host bus must be initialized (e.g. @c spi3_init()) before
 * @ref node_acc_adxl355_init.
 */
typedef struct {
    spi_host_device_t host;
    gpio_num_t cs_gpio;
    uint32_t clock_hz;
    /** SPI mode 0..3; ADXL355 uses mode 0. */
    uint8_t spi_mode;
    int input_delay_ns;
    /**
     * DRDY MCU input (eval: GPIO6). Use @c GPIO_NUM_NC if @ref node_acc_adxl355_drdy_isr_install
     * is not used.
     */
    gpio_num_t gpio_drdy;
    /** Match RANGE INT_POL / board wiring (often @c GPIO_INTR_POSEDGE for active-high pulse). */
    gpio_int_type_t drdy_intr_type;
    /** INT1 / INT2 MCU inputs (eval: GPIO4 / GPIO5). @c GPIO_NUM_NC if unused. */
    gpio_num_t gpio_int1;
    gpio_num_t gpio_int2;
    /** Edges for INT1/INT2; align with @ref node_acc_adxl355_set_interrupt_polarity. */
    gpio_int_type_t int1_intr_type;
    gpio_int_type_t int2_intr_type;
    /** Initial @ref node_acc_adxl355_meas_mode_t (default eval: ACCEL_AND_TEMP). */
    node_acc_adxl355_meas_mode_t meas_mode;
    /**
     * If true (default), all SPI transfers are serialized with an internal recursive mutex
     * (safe with multi-task access). Set false only if a single task touches the device.
     */
    bool spi_mutex_enable;
    /** If true, log one INFO line after successful init. Default false (quieter production). */
    bool log_info_on_init;
} node_acc_adxl355_config_t;

typedef struct node_acc_adxl355_dev {
    spi_device_handle_t spi;
    node_acc_adxl355_range_t range;
    node_acc_adxl355_odr_t odr;
    float scale_factor;
    node_acc_adxl355_meas_mode_t meas_mode;
    gpio_num_t gpio_drdy;
    gpio_int_type_t drdy_intr_type;
    SemaphoreHandle_t drdy_sem;
    bool drdy_isr_installed;
    gpio_num_t gpio_int1;
    gpio_num_t gpio_int2;
    gpio_int_type_t int1_intr_type;
    gpio_int_type_t int2_intr_type;
    SemaphoreHandle_t int1_sem;
    SemaphoreHandle_t int2_sem;
    bool int1_isr_installed;
    bool int2_isr_installed;
    /** Recursive mutex when @a spi_mutex_enable was set at init; NULL otherwise. */
    SemaphoreHandle_t spi_mutex;
} node_acc_adxl355_dev_t;

/** Eval board defaults: CS GPIO7, SPI3, 10 MHz, mode 0, DRDY GPIO6 posedge. */
void node_acc_adxl355_config_default_eval(node_acc_adxl355_config_t *cfg);

esp_err_t node_acc_adxl355_init(node_acc_adxl355_dev_t *dev, const node_acc_adxl355_config_t *cfg,
                                node_acc_adxl355_range_t range, node_acc_adxl355_odr_t odr);

esp_err_t node_acc_adxl355_set_measurement_mode(node_acc_adxl355_dev_t *dev, node_acc_adxl355_meas_mode_t mode);

esp_err_t node_acc_adxl355_get_measurement_mode(node_acc_adxl355_dev_t *dev, node_acc_adxl355_meas_mode_t *mode);

/**
 * Remove GPIO ISRs, clear INT_MAP, disable DRDY pin and measurement, remove SPI device, delete SPI mutex.
 * Do not call concurrently with other tasks using this @a dev (take mutex first in your app or stop workers).
 */
esp_err_t node_acc_adxl355_deinit(node_acc_adxl355_dev_t *dev);

esp_err_t node_acc_adxl355_reset(node_acc_adxl355_dev_t *dev);

esp_err_t node_acc_adxl355_read_ids(node_acc_adxl355_dev_t *dev, node_acc_adxl355_ids_t *ids);

esp_err_t node_acc_adxl355_read_status(node_acc_adxl355_dev_t *dev, uint8_t *status);

esp_err_t node_acc_adxl355_read_raw_xyz(node_acc_adxl355_dev_t *dev, node_acc_adxl355_raw_xyz_t *out);

esp_err_t node_acc_adxl355_read_g(node_acc_adxl355_dev_t *dev, node_acc_adxl355_g_t *out);

esp_err_t node_acc_adxl355_read_raw_temp(node_acc_adxl355_dev_t *dev, uint16_t *raw_temp);

esp_err_t node_acc_adxl355_read_temp_c(node_acc_adxl355_dev_t *dev, float *temp_c);

esp_err_t node_acc_adxl355_set_range(node_acc_adxl355_dev_t *dev, node_acc_adxl355_range_t range);

esp_err_t node_acc_adxl355_get_range(node_acc_adxl355_dev_t *dev, node_acc_adxl355_range_t *range);

esp_err_t node_acc_adxl355_set_odr(node_acc_adxl355_dev_t *dev, node_acc_adxl355_odr_t odr);

esp_err_t node_acc_adxl355_get_odr(node_acc_adxl355_dev_t *dev, node_acc_adxl355_odr_t *odr);

esp_err_t node_acc_adxl355_enable_measurement(node_acc_adxl355_dev_t *dev);

esp_err_t node_acc_adxl355_disable_measurement(node_acc_adxl355_dev_t *dev);

esp_err_t node_acc_adxl355_enable_temperature(node_acc_adxl355_dev_t *dev);

esp_err_t node_acc_adxl355_disable_temperature(node_acc_adxl355_dev_t *dev);

esp_err_t node_acc_adxl355_enable_data_ready_pin(node_acc_adxl355_dev_t *dev);

esp_err_t node_acc_adxl355_disable_data_ready_pin(node_acc_adxl355_dev_t *dev);

esp_err_t node_acc_adxl355_acceleration_scale_g_per_lsb(node_acc_adxl355_dev_t *dev, float *scale);

/**
 * Install GPIO ISR on @a gpio_drdy from config: binary semaphore released on each DRDY edge.
 * Requires @ref node_acc_adxl355_enable_data_ready_pin (done in @ref node_acc_adxl355_init).
 * Call @ref node_acc_adxl355_drdy_isr_remove before @ref node_acc_adxl355_deinit or let deinit do it.
 */
esp_err_t node_acc_adxl355_drdy_isr_install(node_acc_adxl355_dev_t *dev);

esp_err_t node_acc_adxl355_drdy_isr_remove(node_acc_adxl355_dev_t *dev);

/** Block until next DRDY interrupt or timeout. Install ISR first. */
esp_err_t node_acc_adxl355_drdy_wait(node_acc_adxl355_dev_t *dev, uint32_t timeout_ms);

/** Write INT_MAP (0x2A). */
esp_err_t node_acc_adxl355_int_map_write(node_acc_adxl355_dev_t *dev, const node_acc_adxl355_int_map_t *map);

/** Read INT_MAP. */
esp_err_t node_acc_adxl355_int_map_read(node_acc_adxl355_dev_t *dev, node_acc_adxl355_int_map_t *map);

/** Set RANGE[6] interrupt polarity (INT1/INT2/DRDY). */
esp_err_t node_acc_adxl355_set_interrupt_polarity(node_acc_adxl355_dev_t *dev, node_acc_adxl355_int_pol_t pol);

/**
 * Install GPIO ISR on INT1/INT2 from config (binary semaphore per pin).
 * Map events with @ref node_acc_adxl355_int_map_write before relying on interrupts.
 */
esp_err_t node_acc_adxl355_int_isr_install(node_acc_adxl355_dev_t *dev);

esp_err_t node_acc_adxl355_int_isr_remove(node_acc_adxl355_dev_t *dev);

esp_err_t node_acc_adxl355_int1_wait(node_acc_adxl355_dev_t *dev, uint32_t timeout_ms);

esp_err_t node_acc_adxl355_int2_wait(node_acc_adxl355_dev_t *dev, uint32_t timeout_ms);

/* --- Phase 4: FIFO, activity, offset --- */

/** Write FIFO_SAMPLES (0x29); @a samples must be <= @ref NODE_ACC_ADXL355_FIFO_SAMPLES_MAX. */
esp_err_t node_acc_adxl355_set_fifo_samples(node_acc_adxl355_dev_t *dev, uint8_t samples);

esp_err_t node_acc_adxl355_get_fifo_samples(node_acc_adxl355_dev_t *dev, uint8_t *samples);

/** Read FIFO_ENTRIES (0x05): number of stored 3-byte axis samples (Analog ref semantics). */
esp_err_t node_acc_adxl355_read_fifo_entries(node_acc_adxl355_dev_t *dev, uint8_t *entries);

/**
 * Burst-read FIFO_DATA (0x11), parse XYZ frames (see Analog @c adxl355_get_raw_fifo_data).
 * @a out_frames is how many frames were written (<= @a max_frames). Drains FIFO when @a max_frames is large enough.
 */
esp_err_t node_acc_adxl355_read_fifo_xyz(node_acc_adxl355_dev_t *dev, node_acc_adxl355_raw_xyz_t *out,
                                         size_t max_frames, size_t *out_frames);

esp_err_t node_acc_adxl355_set_offset_xyz(node_acc_adxl355_dev_t *dev, const node_acc_adxl355_offset_t *off);

esp_err_t node_acc_adxl355_get_offset_xyz(node_acc_adxl355_dev_t *dev, node_acc_adxl355_offset_t *off);

esp_err_t node_acc_adxl355_set_activity_enable(node_acc_adxl355_dev_t *dev, const node_acc_adxl355_act_en_t *en);

esp_err_t node_acc_adxl355_get_activity_enable(node_acc_adxl355_dev_t *dev, node_acc_adxl355_act_en_t *en);

esp_err_t node_acc_adxl355_set_activity_threshold(node_acc_adxl355_dev_t *dev, uint16_t threshold);

esp_err_t node_acc_adxl355_get_activity_threshold(node_acc_adxl355_dev_t *dev, uint16_t *threshold);

esp_err_t node_acc_adxl355_set_activity_count(node_acc_adxl355_dev_t *dev, uint8_t count);

esp_err_t node_acc_adxl355_get_activity_count(node_acc_adxl355_dev_t *dev, uint8_t *count);

#ifdef __cplusplus
}
#endif
