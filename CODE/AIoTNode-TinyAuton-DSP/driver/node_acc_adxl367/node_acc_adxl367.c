/**
 * @file node_acc_adxl367.c
 * @brief ADXL367 I2C driver (ref-adxl367/adxl367.c). Tests: node_acc_adxl367_test.h.
 */
#include "node_acc_adxl367.h"

#include <stddef.h>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "node_i2c.h"

static const char *TAG = "node_acc_adxl367";

/* Register map (subset) */
#define REG_DEVID_AD 0x00u
#define REG_DEVID_MST 0x01u
#define REG_PARTID 0x02u
#define REG_REVID 0x03u
#define REG_STATUS 0x0Bu
#define REG_XDATA_H 0x0Eu
#define REG_SOFT_RESET 0x1Fu
#define REG_FILTER_CTL 0x2Cu
#define REG_POWER_CTL 0x2Du
#define REG_TEMP_H 0x14u
#define REG_ADC_CTL 0x3Cu
#define REG_TEMP_CTL 0x3Du
#define REG_FIFO_ENTRIES_L 0x0Cu
#define REG_FIFO_ENTRIES_H 0x0Du
#define REG_FIFO_CONTROL 0x28u
#define REG_FIFO_SAMPLES 0x29u
#define REG_I2C_FIFO_DATA 0x18u
#define REG_INTMAP1_LWR 0x2Au
#define REG_INTMAP2_LWR 0x2Bu
#define REG_INTMAP1_UPPER 0x3Au
#define REG_INTMAP2_UPPER 0x3Bu

#define REG_THRESH_ACT_H 0x20u
#define REG_THRESH_ACT_L 0x21u
#define REG_TIME_ACT 0x22u
#define REG_THRESH_INACT_H 0x23u
#define REG_THRESH_INACT_L 0x24u
#define REG_TIME_INACT_H 0x25u
#define REG_TIME_INACT_L 0x26u
#define REG_ACT_INACT_CTL 0x27u
#define REG_AXIS_MASK 0x43u

#define INTMAP_UPPER_MASK 0xDFu

#define THRESH_H_MASK 0x7Fu
#define THRESH_L_MASK 0xFCu
#define ACT_EN_MSK 0x03u
#define INACT_EN_MSK 0x0Cu
#define LINKLOOP_MSK (0x3u << 4)
#define AXIS_MASK_ACT_BLOCK_MASK 0x07u

#define DEVID_AD_VAL 0xADu
#define DEVID_MST_VAL 0x1Du
#define PARTID_VAL 0xF7u
#define REVID_ADXL367 0x03u

#define RESET_KEY 0x52u

#define STATUS_DATA_RDY (1u << 0)

#define FILTER_CTL_RANGE_MASK 0xC0u
#define FILTER_CTL_ODR_MASK 0x07u

#define POWER_CTL_MEASURE_MASK 0x03u

#define ADC_EN_MASK (1u << 0)
#define TEMP_EN_MASK (1u << 0)

#define FIFO_CTL_MODE_MASK 0x03u
#define FIFO_CTL_CH_MASK 0x78u
#define FIFO_CTL_SAMPLES_BIT (1u << 2)

#define ADC_CTL_FIFO_RD_MODE_MASK 0xC0u
#define FIFO_RD_14B_CHID 3u

#define STATUS_FIFO_RDY (1u << 1)
#define STATUS_FIFO_WATERMARK (1u << 2)
#define STATUS_FIFO_OVERRUN (1u << 3)

#define FIFO_MAX_READ_BYTES 512u

/** adxl367_temp_conv in ref-adxl367/adxl367.c */
#define TEMP_OFFSET 1185
#define TEMP_SCALE_NUM 18518518LL
#define TEMP_SCALE_DEN 1000000000LL

#define I2C_TIMEOUT_MS 1000

/* Low-level helpers: node_i2c only (see node_i2c.h). */

static esp_err_t reg_write(node_acc_adxl367_dev_t *dev, uint8_t reg, uint8_t val)
{
    uint8_t buf[2] = {reg, val};
    return i2c_write_data(dev->i2c, buf, sizeof(buf), I2C_TIMEOUT_MS);
}

static esp_err_t reg_read(node_acc_adxl367_dev_t *dev, uint8_t reg, uint8_t *data, size_t len)
{
    if (len == 0 || data == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    return i2c_write_read_data(dev->i2c, &reg, 1, data, len, I2C_TIMEOUT_MS);
}

static esp_err_t reg_update_mask(node_acc_adxl367_dev_t *dev, uint8_t reg, uint8_t mask, uint8_t set_bits)
{
    uint8_t v = 0;
    esp_err_t err = reg_read(dev, reg, &v, 1);
    if (err != ESP_OK) {
        return err;
    }
    v = (uint8_t)((v & (uint8_t)~mask) | (set_bits & mask));
    return reg_write(dev, reg, v);
}

/** 14-bit pack: ref adxl367_get_raw_xyz. */
static int16_t pack_14bit_xy(uint8_t hi, uint8_t lo)
{
    int16_t v = (int16_t)(((int16_t)hi << 6) | (int16_t)(lo >> 2));
    if (v & (int16_t)(1 << 13)) {
        v |= (int16_t)(3 << 14);
    }
    return v;
}

static esp_err_t adc_en_set_only(node_acc_adxl367_dev_t *dev, bool enable)
{
    return reg_update_mask(dev, REG_ADC_CTL, ADC_EN_MASK, enable ? ADC_EN_MASK : 0u);
}

static esp_err_t temp_en_set(node_acc_adxl367_dev_t *dev, bool enable)
{
    return reg_update_mask(dev, REG_TEMP_CTL, TEMP_EN_MASK, enable ? TEMP_EN_MASK : 0u);
}

/** Poll STATUS until DATA_RDY (ref @c adxl367_get_raw_xyz uses an unbounded while; we cap for RTOS). */
static esp_err_t wait_data_rdy(node_acc_adxl367_dev_t *dev)
{
    const int max_poll = 200;
    for (int i = 0; i < max_poll; i++) {
        uint8_t st = 0;
        esp_err_t err = reg_read(dev, REG_STATUS, &st, 1);
        if (err != ESP_OK) {
            return err;
        }
        if (st & STATUS_DATA_RDY) {
            return ESP_OK;
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }
    return ESP_ERR_TIMEOUT;
}

esp_err_t node_acc_adxl367_set_measurement_mode(node_acc_adxl367_dev_t *dev,
                                                node_acc_adxl367_meas_mode_t mode)
{
    if (dev == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (mode > NODE_ACC_ADXL367_MEAS_ACCEL_AND_TEMP) {
        return ESP_ERR_INVALID_ARG;
    }

    esp_err_t err = adc_en_set_only(dev, false);
    if (err != ESP_OK) {
        return err;
    }

    switch (mode) {
    case NODE_ACC_ADXL367_MEAS_ACCEL_ONLY:
        err = temp_en_set(dev, false);
        break;
    case NODE_ACC_ADXL367_MEAS_TEMP_ONLY:
        err = temp_en_set(dev, true);
        break;
    case NODE_ACC_ADXL367_MEAS_ACCEL_AND_TEMP:
        err = temp_en_set(dev, true);
        break;
    default:
        return ESP_ERR_INVALID_ARG;
    }
    if (err != ESP_OK) {
        return err;
    }
    dev->meas_mode = mode;
    return ESP_OK;
}

esp_err_t node_acc_adxl367_read_temp_raw(node_acc_adxl367_dev_t *dev, int16_t *raw)
{
    if (dev == NULL || raw == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (dev->meas_mode == NODE_ACC_ADXL367_MEAS_ACCEL_ONLY) {
        return ESP_ERR_INVALID_STATE;
    }

    esp_err_t err = wait_data_rdy(dev);
    if (err != ESP_OK) {
        return err;
    }

    uint8_t t[2];
    err = reg_read(dev, REG_TEMP_H, t, sizeof(t));
    if (err != ESP_OK) {
        return err;
    }
    *raw = pack_14bit_xy(t[0], t[1]);
    return ESP_OK;
}

esp_err_t node_acc_adxl367_read_temp_celsius(node_acc_adxl367_dev_t *dev, float *celsius)
{
    if (dev == NULL || celsius == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    int16_t raw = 0;
    esp_err_t err = node_acc_adxl367_read_temp_raw(dev, &raw);
    if (err != ESP_OK) {
        return err;
    }
    int64_t v = (int64_t)raw + (int64_t)TEMP_OFFSET;
    v = v * TEMP_SCALE_NUM;
    *celsius = (float)v / (float)TEMP_SCALE_DEN;
    return ESP_OK;
}

esp_err_t node_acc_adxl367_soft_reset(node_acc_adxl367_dev_t *dev)
{
    if (dev == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    esp_err_t err = reg_write(dev, REG_SOFT_RESET, RESET_KEY);
    if (err != ESP_OK) {
        return err;
    }
    vTaskDelay(pdMS_TO_TICKS(20));
    dev->range = NODE_ACC_ADXL367_RANGE_2G;
    dev->odr = NODE_ACC_ADXL367_ODR_100_HZ;
    dev->mode = NODE_ACC_ADXL367_MODE_STANDBY;
    dev->meas_mode = NODE_ACC_ADXL367_MEAS_ACCEL_ONLY;
    dev->fifo_mode = NODE_ACC_ADXL367_FIFO_DISABLED;
    dev->fifo_format = NODE_ACC_ADXL367_FIFO_FMT_XYZ;
    return ESP_OK;
}

static int16_t pack_fifo_14b_chid(uint8_t hi, uint8_t lo)
{
    int16_t v = (int16_t)(((int16_t)(hi & 0x3Fu) << 8) | lo);
    if (v & (int16_t)(1 << 13)) {
        v |= (int16_t)(3 << 14);
    }
    return v;
}

static esp_err_t fifo_set_sample_sets_nb(node_acc_adxl367_dev_t *dev, uint16_t sets_nb)
{
    esp_err_t err = reg_update_mask(dev, REG_FIFO_CONTROL, FIFO_CTL_SAMPLES_BIT,
                                    (sets_nb & (1u << 9)) ? FIFO_CTL_SAMPLES_BIT : 0u);
    if (err != ESP_OK) {
        return err;
    }
    return reg_write(dev, REG_FIFO_SAMPLES, (uint8_t)(sets_nb & 0xFFu));
}

static esp_err_t fifo_read_payload(node_acc_adxl367_dev_t *dev, uint8_t *data, size_t len)
{
    uint8_t reg = REG_I2C_FIFO_DATA;
    return i2c_write_read_data(dev->i2c, &reg, 1, data, len, I2C_TIMEOUT_MS);
}

esp_err_t node_acc_adxl367_read_status(node_acc_adxl367_dev_t *dev, uint8_t *status)
{
    if (dev == NULL || status == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    return reg_read(dev, REG_STATUS, status, 1);
}

static uint8_t intmap_bit(uint8_t v)
{
    return (uint8_t)(v & 1u);
}

esp_err_t node_acc_adxl367_int_map(node_acc_adxl367_dev_t *dev, uint8_t int_pin,
                                   const node_acc_adxl367_int_map_t *map)
{
    if (dev == NULL || map == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (int_pin != 1u && int_pin != 2u) {
        return ESP_ERR_INVALID_ARG;
    }

    uint8_t upper = (uint8_t)((intmap_bit(map->err_fuse) << 7) | (intmap_bit(map->err_user_regs) << 6) |
                              (intmap_bit(map->kpalv_timer) << 4) | (intmap_bit(map->temp_adc_hi) << 3) |
                              (intmap_bit(map->temp_adc_low) << 2) | (intmap_bit(map->tap_two) << 1) |
                              intmap_bit(map->tap_one));

    uint8_t lower = (uint8_t)((intmap_bit(map->int_low) << 7) | (intmap_bit(map->awake) << 6) |
                             (intmap_bit(map->inact) << 5) | (intmap_bit(map->act) << 4) |
                             (intmap_bit(map->fifo_overrun) << 3) | (intmap_bit(map->fifo_watermark) << 2) |
                             (intmap_bit(map->fifo_ready) << 1) | intmap_bit(map->data_ready));

    uint8_t reg_upper = (int_pin == 1u) ? REG_INTMAP1_UPPER : REG_INTMAP2_UPPER;
    uint8_t reg_lower = (int_pin == 1u) ? REG_INTMAP1_LWR : REG_INTMAP2_LWR;

    esp_err_t err = reg_update_mask(dev, reg_upper, INTMAP_UPPER_MASK, upper);
    if (err != ESP_OK) {
        return err;
    }
    return reg_write(dev, reg_lower, lower);
}

esp_err_t node_acc_adxl367_act_inact_disable(node_acc_adxl367_dev_t *dev)
{
    if (dev == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    return reg_update_mask(dev, REG_ACT_INACT_CTL, ACT_EN_MSK | INACT_EN_MSK, 0u);
}

esp_err_t node_acc_adxl367_set_activity_axes_enabled(node_acc_adxl367_dev_t *dev, bool enable_x,
                                                     bool enable_y, bool enable_z)
{
    if (dev == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (!enable_x && !enable_y && !enable_z) {
        return ESP_ERR_INVALID_ARG;
    }
    uint8_t block = 0u;
    if (!enable_x) {
        block |= (1u << 0);
    }
    if (!enable_y) {
        block |= (1u << 1);
    }
    if (!enable_z) {
        block |= (1u << 2);
    }
    return reg_update_mask(dev, REG_AXIS_MASK, AXIS_MASK_ACT_BLOCK_MASK, block);
}

esp_err_t node_acc_adxl367_set_act_inact_linkloop(node_acc_adxl367_dev_t *dev, uint8_t linkloop_mode_0_to_3)
{
    if (dev == NULL || linkloop_mode_0_to_3 > 3u) {
        return ESP_ERR_INVALID_ARG;
    }
    return reg_update_mask(dev, REG_ACT_INACT_CTL, LINKLOOP_MSK, (uint8_t)(linkloop_mode_0_to_3 << 4));
}

/* Register writes follow driver/ref-adxl367/adxl367.c adxl367_setup_activity_detection(). */
esp_err_t node_acc_adxl367_setup_activity_detection(node_acc_adxl367_dev_t *dev,
                                                    node_acc_adxl367_ai_ref_t ref_mode,
                                                    uint16_t threshold,
                                                    uint8_t time_act_samples)
{
    if (dev == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (threshold > 0x1FFFu || ref_mode > NODE_ACC_ADXL367_AI_REF_REL) {
        return ESP_ERR_INVALID_ARG;
    }

    uint8_t act_val = (ref_mode == NODE_ACC_ADXL367_AI_REF_ABS) ? 1u : 3u;
    esp_err_t err = reg_update_mask(dev, REG_ACT_INACT_CTL, ACT_EN_MSK, act_val);
    if (err != ESP_OK) {
        return err;
    }
    err = reg_update_mask(dev, REG_THRESH_ACT_H, THRESH_H_MASK, (uint8_t)(threshold >> 6));
    if (err != ESP_OK) {
        return err;
    }
    err = reg_update_mask(dev, REG_THRESH_ACT_L, THRESH_L_MASK, (uint8_t)((threshold & 0x3Fu) << 2));
    if (err != ESP_OK) {
        return err;
    }
    return reg_write(dev, REG_TIME_ACT, time_act_samples);
}

/* Register writes follow driver/ref-adxl367/adxl367.c adxl367_setup_inactivity_detection(). */
esp_err_t node_acc_adxl367_setup_inactivity_detection(node_acc_adxl367_dev_t *dev,
                                                      node_acc_adxl367_ai_ref_t ref_mode,
                                                      uint16_t threshold,
                                                      uint16_t time_inact_samples)
{
    if (dev == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (threshold > 0x1FFFu || ref_mode > NODE_ACC_ADXL367_AI_REF_REL) {
        return ESP_ERR_INVALID_ARG;
    }

    uint8_t inact_val = (ref_mode == NODE_ACC_ADXL367_AI_REF_ABS) ? 1u : 3u;
    esp_err_t err = reg_update_mask(dev, REG_ACT_INACT_CTL, INACT_EN_MSK, (uint8_t)(inact_val << 2));
    if (err != ESP_OK) {
        return err;
    }
    err = reg_update_mask(dev, REG_THRESH_INACT_H, THRESH_H_MASK, (uint8_t)(threshold >> 6));
    if (err != ESP_OK) {
        return err;
    }
    err = reg_update_mask(dev, REG_THRESH_INACT_L, THRESH_L_MASK, (uint8_t)((threshold & 0x3Fu) << 2));
    if (err != ESP_OK) {
        return err;
    }
    err = reg_write(dev, REG_TIME_INACT_H, (uint8_t)(time_inact_samples >> 8));
    if (err != ESP_OK) {
        return err;
    }
    return reg_write(dev, REG_TIME_INACT_L, (uint8_t)(time_inact_samples & 0xFFu));
}

esp_err_t node_acc_adxl367_fifo_setup(node_acc_adxl367_dev_t *dev,
                                      node_acc_adxl367_fifo_mode_t mode,
                                      node_acc_adxl367_fifo_format_t format,
                                      uint16_t sample_sets_nb)
{
    if (dev == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (mode > NODE_ACC_ADXL367_FIFO_TRIGGERED || format > NODE_ACC_ADXL367_FIFO_FMT_XYZT) {
        return ESP_ERR_INVALID_ARG;
    }

    esp_err_t err = reg_update_mask(dev, REG_FIFO_CONTROL, FIFO_CTL_MODE_MASK, (uint8_t)mode);
    if (err != ESP_OK) {
        return err;
    }
    err = reg_update_mask(dev, REG_FIFO_CONTROL, FIFO_CTL_CH_MASK, (uint8_t)((uint8_t)format << 3));
    if (err != ESP_OK) {
        return err;
    }
    err = fifo_set_sample_sets_nb(dev, sample_sets_nb);
    if (err != ESP_OK) {
        return err;
    }
    /* adxl367_set_fifo_read_mode: ADXL367_14B_CHID -> ADC_CTL[7:6] = 3 */
    err = reg_update_mask(dev, REG_ADC_CTL, ADC_CTL_FIFO_RD_MODE_MASK, FIFO_RD_14B_CHID << 6);
    if (err != ESP_OK) {
        return err;
    }

    dev->fifo_mode = mode;
    dev->fifo_format = format;
    return ESP_OK;
}

esp_err_t node_acc_adxl367_fifo_disable(node_acc_adxl367_dev_t *dev)
{
    if (dev == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    esp_err_t err = reg_update_mask(dev, REG_FIFO_CONTROL, FIFO_CTL_MODE_MASK, 0u);
    if (err == ESP_OK) {
        dev->fifo_mode = NODE_ACC_ADXL367_FIFO_DISABLED;
    }
    return err;
}

esp_err_t node_acc_adxl367_fifo_get_entry_count(node_acc_adxl367_dev_t *dev, uint16_t *count)
{
    if (dev == NULL || count == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    uint8_t v[2];
    esp_err_t err = reg_read(dev, REG_FIFO_ENTRIES_L, v, sizeof(v));
    if (err != ESP_OK) {
        return err;
    }
    *count = (uint16_t)(((uint16_t)(v[1] & 0x03u) << 8) | v[0]);
    return ESP_OK;
}

esp_err_t node_acc_adxl367_fifo_drain_xyz(node_acc_adxl367_dev_t *dev, int16_t *x, int16_t *y, int16_t *z,
                                          size_t max_triplets, size_t *out_triplets)
{
    if (dev == NULL || x == NULL || y == NULL || z == NULL || out_triplets == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (dev->fifo_format != NODE_ACC_ADXL367_FIFO_FMT_XYZ) {
        return ESP_ERR_INVALID_STATE;
    }

    uint16_t n_ent = 0;
    esp_err_t err = node_acc_adxl367_fifo_get_entry_count(dev, &n_ent);
    if (err != ESP_OK) {
        return err;
    }
    if (n_ent == 0) {
        *out_triplets = 0;
        return ESP_OK;
    }

    size_t n_bytes = (size_t)n_ent * 2u;
    if (n_bytes > FIFO_MAX_READ_BYTES) {
        n_bytes = FIFO_MAX_READ_BYTES;
    }

    uint8_t buf[FIFO_MAX_READ_BYTES];
    err = fifo_read_payload(dev, buf, n_bytes);
    if (err != ESP_OK) {
        return err;
    }

    int16_t *px = x;
    int16_t *py = y;
    int16_t *pz = z;
    size_t nt = 0;

    for (size_t i = 0; i + 1 < n_bytes && nt < max_triplets; i += 2) {
        uint8_t ch = (uint8_t)(buf[i] >> 6);
        int16_t val = pack_fifo_14b_chid(buf[i], buf[i + 1]);
        switch (ch) {
        case 0u:
            if ((size_t)(px - x) >= max_triplets) {
                return ESP_ERR_NO_MEM;
            }
            *px++ = val;
            break;
        case 1u:
            if ((size_t)(py - y) >= max_triplets) {
                return ESP_ERR_NO_MEM;
            }
            *py++ = val;
            break;
        case 2u:
            if ((size_t)(pz - z) >= max_triplets) {
                return ESP_ERR_NO_MEM;
            }
            *pz++ = val;
            nt++;
            break;
        default:
            ESP_LOGE(TAG, "FIFO bad channel id %u", (unsigned)ch);
            return ESP_ERR_INVALID_RESPONSE;
        }
    }

    *out_triplets = nt;
    return ESP_OK;
}

esp_err_t node_acc_adxl367_probe(node_acc_adxl367_dev_t *dev)
{
    if (dev == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    uint8_t v = 0;
    esp_err_t err = reg_read(dev, REG_DEVID_AD, &v, 1);
    if (err != ESP_OK) {
        return err;
    }
    if (v != DEVID_AD_VAL) {
        ESP_LOGE(TAG, "DEVID_AD mismatch: got 0x%02X expect 0x%02X", v, DEVID_AD_VAL);
        return ESP_ERR_NOT_FOUND;
    }
    err = reg_read(dev, REG_DEVID_MST, &v, 1);
    if (err != ESP_OK) {
        return err;
    }
    if (v != DEVID_MST_VAL) {
        ESP_LOGE(TAG, "DEVID_MST mismatch: got 0x%02X expect 0x%02X", v, DEVID_MST_VAL);
        return ESP_ERR_NOT_FOUND;
    }
    err = reg_read(dev, REG_PARTID, &v, 1);
    if (err != ESP_OK) {
        return err;
    }
    if (v != PARTID_VAL) {
        ESP_LOGE(TAG, "PARTID mismatch: got 0x%02X expect 0x%02X", v, PARTID_VAL);
        return ESP_ERR_NOT_FOUND;
    }
    err = reg_read(dev, REG_REVID, &v, 1);
    if (err != ESP_OK) {
        return err;
    }
    if (v != REVID_ADXL367) {
        ESP_LOGE(TAG, "REVID mismatch: got 0x%02X expect 0x%02X (ADXL367)", v, REVID_ADXL367);
        return ESP_ERR_NOT_FOUND;
    }
    return ESP_OK;
}

esp_err_t node_acc_adxl367_set_range(node_acc_adxl367_dev_t *dev, node_acc_adxl367_range_t range)
{
    if (dev == NULL || range > NODE_ACC_ADXL367_RANGE_8G) {
        return ESP_ERR_INVALID_ARG;
    }
    uint8_t shifted = (uint8_t)((uint8_t)range << 6);
    esp_err_t err = reg_update_mask(dev, REG_FILTER_CTL, FILTER_CTL_RANGE_MASK, shifted);
    if (err == ESP_OK) {
        dev->range = range;
    }
    return err;
}

esp_err_t node_acc_adxl367_set_odr(node_acc_adxl367_dev_t *dev, node_acc_adxl367_odr_t odr)
{
    if (dev == NULL || odr > NODE_ACC_ADXL367_ODR_400_HZ) {
        return ESP_ERR_INVALID_ARG;
    }
    esp_err_t err = reg_update_mask(dev, REG_FILTER_CTL, FILTER_CTL_ODR_MASK, (uint8_t)odr);
    if (err == ESP_OK) {
        dev->odr = odr;
    }
    return err;
}

esp_err_t node_acc_adxl367_set_mode(node_acc_adxl367_dev_t *dev, node_acc_adxl367_mode_t mode)
{
    if (dev == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (mode != NODE_ACC_ADXL367_MODE_STANDBY && mode != NODE_ACC_ADXL367_MODE_MEASURE) {
        return ESP_ERR_INVALID_ARG;
    }
    esp_err_t err =
        reg_update_mask(dev, REG_POWER_CTL, POWER_CTL_MEASURE_MASK, (uint8_t)mode);
    if (err != ESP_OK) {
        return err;
    }
    if (mode == NODE_ACC_ADXL367_MODE_MEASURE) {
        vTaskDelay(pdMS_TO_TICKS(100));
    }
    dev->mode = mode;
    return ESP_OK;
}

esp_err_t node_acc_adxl367_read_xyz_raw(node_acc_adxl367_dev_t *dev, int16_t *x, int16_t *y, int16_t *z)
{
    if (dev == NULL || x == NULL || y == NULL || z == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (dev->meas_mode == NODE_ACC_ADXL367_MEAS_TEMP_ONLY) {
        return ESP_ERR_INVALID_STATE;
    }

    /* Same flow as Analog @c adxl367_get_raw_xyz: wait DATA_RDY, then read 6 bytes from XDATA_H (burst). */
    esp_err_t err = wait_data_rdy(dev);
    if (err != ESP_OK) {
        if (err == ESP_ERR_TIMEOUT) {
            ESP_LOGW(TAG, "DATA_RDY timeout");
        }
        return err;
    }

    uint8_t b[6];
    err = reg_read(dev, REG_XDATA_H, b, sizeof(b));
    if (err != ESP_OK) {
        return err;
    }

    *x = pack_14bit_xy(b[0], b[1]);
    *y = pack_14bit_xy(b[2], b[3]);
    *z = pack_14bit_xy(b[4], b[5]);
    return ESP_OK;
}

esp_err_t node_acc_adxl367_init(node_acc_adxl367_dev_t *dev)
{
    if (dev == NULL || dev->i2c == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    esp_err_t err = node_acc_adxl367_soft_reset(dev);
    if (err != ESP_OK) {
        return err;
    }

    err = node_acc_adxl367_probe(dev);
    if (err != ESP_OK) {
        return err;
    }

    err = node_acc_adxl367_set_range(dev, NODE_ACC_ADXL367_RANGE_2G);
    if (err != ESP_OK) {
        return err;
    }

    err = node_acc_adxl367_set_odr(dev, NODE_ACC_ADXL367_ODR_100_HZ);
    if (err != ESP_OK) {
        return err;
    }

    err = node_acc_adxl367_set_mode(dev, NODE_ACC_ADXL367_MODE_MEASURE);
    if (err != ESP_OK) {
        return err;
    }

    err = node_acc_adxl367_set_measurement_mode(dev, NODE_ACC_ADXL367_MEAS_ACCEL_ONLY);
    if (err != ESP_OK) {
        return err;
    }

    ESP_LOGI(TAG, "Init OK (2g, 100 Hz, measure, accel-only)");
    return ESP_OK;
}
