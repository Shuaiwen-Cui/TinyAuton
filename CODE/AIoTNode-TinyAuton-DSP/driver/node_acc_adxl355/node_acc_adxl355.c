/**
 * @file node_acc_adxl355.c
 * @brief ADXL355 SPI driver (phase 1–4: DRDY, INT_MAP, FIFO, activity, offset).
 */
#include "node_acc_adxl355.h"

#include <string.h>

#include "driver/gpio.h"
#include "esp_heap_caps.h"
#include "esp_check.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

static const char *TAG = "node_acc_adxl355";

/* Temperature: datasheet intercept 1885 LSB @ 25 °C, slope -9.05 LSB/°C */
static const float k_temp_intercept_lsb = 1885.0f;
static const float k_temp_slope_lsb_per_c = -9.05f;
static const float k_temp_ref_c = 25.0f;

static const float k_scale_2g = 3.9e-6f;
static const float k_scale_4g = 7.8e-6f;
static const float k_scale_8g = 15.6e-6f;

static esp_err_t dev_spi_add(node_acc_adxl355_dev_t *dev, const node_acc_adxl355_config_t *cfg);
static esp_err_t dev_spi_remove(node_acc_adxl355_dev_t *dev);
static void dev_free_hw(node_acc_adxl355_dev_t *dev);
static void spi_lock(node_acc_adxl355_dev_t *dev);
static void spi_unlock(node_acc_adxl355_dev_t *dev);

static esp_err_t reg_read(node_acc_adxl355_dev_t *dev, uint8_t reg, uint8_t *val);
static esp_err_t reg_write(node_acc_adxl355_dev_t *dev, uint8_t reg, uint8_t val);
static esp_err_t regs_read(node_acc_adxl355_dev_t *dev, uint8_t reg, uint8_t *data, size_t len);
static esp_err_t regs_write(node_acc_adxl355_dev_t *dev, uint8_t reg, const uint8_t *data, size_t len);
static esp_err_t spi_read_after_cmd(node_acc_adxl355_dev_t *dev, uint8_t cmd, uint8_t *data, size_t len);

static void IRAM_ATTR node_acc_adxl355_sem_isr(void *arg)
{
    SemaphoreHandle_t sem = (SemaphoreHandle_t)arg;
    BaseType_t hp = pdFALSE;
    if (sem != NULL) {
        (void)xSemaphoreGiveFromISR(sem, &hp);
    }
    if (hp) {
        portYIELD_FROM_ISR();
    }
}

/** Call once per process; avoids ESP-IDF ERROR log on duplicate gpio_install_isr_service. */
static esp_err_t gpio_isr_service_ensure_installed(void)
{
    esp_err_t e = gpio_install_isr_service(0);
    if (e != ESP_OK && e != ESP_ERR_INVALID_STATE) {
        return e;
    }
    return ESP_OK;
}

void node_acc_adxl355_config_default_eval(node_acc_adxl355_config_t *cfg)
{
    if (cfg == NULL) {
        return;
    }
    memset(cfg, 0, sizeof(*cfg));
    cfg->host = SPI3_HOST;
    cfg->cs_gpio = GPIO_NUM_7;
    cfg->clock_hz = NODE_ACC_ADXL355_DEFAULT_SPI_HZ;
    cfg->spi_mode = 0;
    cfg->input_delay_ns = 30;
    cfg->gpio_drdy = GPIO_NUM_6;
    cfg->drdy_intr_type = GPIO_INTR_POSEDGE;
    cfg->gpio_int1 = GPIO_NUM_4;
    cfg->gpio_int2 = GPIO_NUM_5;
    cfg->int1_intr_type = GPIO_INTR_NEGEDGE;
    cfg->int2_intr_type = GPIO_INTR_NEGEDGE;
    cfg->meas_mode = NODE_ACC_ADXL355_MEAS_ACCEL_AND_TEMP;
    cfg->spi_mutex_enable = true;
    cfg->log_info_on_init = false;
}

static void spi_lock(node_acc_adxl355_dev_t *dev)
{
    if (dev != NULL && dev->spi_mutex != NULL) {
        xSemaphoreTakeRecursive(dev->spi_mutex, portMAX_DELAY);
    }
}

static void spi_unlock(node_acc_adxl355_dev_t *dev)
{
    if (dev != NULL && dev->spi_mutex != NULL) {
        xSemaphoreGiveRecursive(dev->spi_mutex);
    }
}

static void dev_free_hw(node_acc_adxl355_dev_t *dev)
{
    if (dev == NULL) {
        return;
    }
    if (dev->spi_mutex != NULL) {
        vSemaphoreDelete(dev->spi_mutex);
        dev->spi_mutex = NULL;
    }
    if (dev->spi != NULL) {
        (void)dev_spi_remove(dev);
    }
}

static esp_err_t dev_spi_add(node_acc_adxl355_dev_t *dev, const node_acc_adxl355_config_t *cfg)
{
    if (dev == NULL || cfg == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (cfg->cs_gpio < 0) {
        return ESP_ERR_INVALID_ARG;
    }

    /* No address_bits: ADXL355 expects one SPI stream (cmd + data) like Analog no-OS
     * write_and_read(1+size), discarding the first received byte. */
    spi_device_interface_config_t d = {
        .clock_speed_hz = (int)cfg->clock_hz,
        .mode = cfg->spi_mode,
        .spics_io_num = cfg->cs_gpio,
        .queue_size = 4,
        .command_bits = 0,
        .address_bits = 0,
        .dummy_bits = 0,
        .input_delay_ns = cfg->input_delay_ns,
    };

    return spi_bus_add_device(cfg->host, &d, &dev->spi);
}

static esp_err_t dev_spi_remove(node_acc_adxl355_dev_t *dev)
{
    if (dev == NULL || dev->spi == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    esp_err_t e = spi_bus_remove_device(dev->spi);
    dev->spi = NULL;
    return e;
}

/**
 * SPI read: same as Analog adxl355_read_device_data — one transaction of (1+len) bytes,
 * MOSI sends cmd then len zeros; first byte of MISO is discarded (command phase).
 */
static esp_err_t spi_read_after_cmd(node_acc_adxl355_dev_t *dev, uint8_t cmd, uint8_t *data, size_t len)
{
    if (dev == NULL || dev->spi == NULL || data == NULL || len == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    const size_t total = 1u + len;
    if (total > 320u) {
        return ESP_ERR_INVALID_ARG;
    }
    uint8_t tx_stack[33];
    uint8_t rx_stack[33];
    uint8_t *tx_buf = tx_stack;
    uint8_t *rx_buf = rx_stack;
    uint8_t *tx_heap = NULL;
    uint8_t *rx_heap = NULL;

    spi_lock(dev);

    if (total > sizeof(tx_stack)) {
        tx_heap = heap_caps_malloc(total, MALLOC_CAP_DMA);
        rx_heap = heap_caps_malloc(total, MALLOC_CAP_DMA);
        if (tx_heap == NULL || rx_heap == NULL) {
            heap_caps_free(tx_heap);
            heap_caps_free(rx_heap);
            spi_unlock(dev);
            return ESP_ERR_NO_MEM;
        }
        tx_buf = tx_heap;
        rx_buf = rx_heap;
    }

    memset(tx_buf, 0, total);
    tx_buf[0] = cmd;
    spi_transaction_t t = {0};
    t.length = (uint32_t)(total * 8u);
    t.tx_buffer = tx_buf;
    t.rx_buffer = rx_buf;
    esp_err_t err = spi_device_polling_transmit(dev->spi, &t);
    if (err == ESP_OK) {
        memcpy(data, &rx_buf[1], len);
    }
    heap_caps_free(tx_heap);
    heap_caps_free(rx_heap);
    spi_unlock(dev);
    return err;
}

static esp_err_t reg_read(node_acc_adxl355_dev_t *dev, uint8_t reg, uint8_t *val)
{
    uint8_t cmd = (uint8_t)((reg << 1) | 0x01u);
    return spi_read_after_cmd(dev, cmd, val, 1);
}

static esp_err_t reg_write(node_acc_adxl355_dev_t *dev, uint8_t reg, uint8_t val)
{
    if (dev == NULL || dev->spi == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    uint8_t cmd = (uint8_t)(reg << 1);
    uint8_t tx_buf[2] = {cmd, val};
    uint8_t rx_buf[2];

    spi_lock(dev);
    spi_transaction_t t = {0};
    t.length = 16;
    t.tx_buffer = tx_buf;
    t.rx_buffer = rx_buf;
    esp_err_t err = spi_device_polling_transmit(dev->spi, &t);
    spi_unlock(dev);
    return err;
}

static esp_err_t regs_read(node_acc_adxl355_dev_t *dev, uint8_t reg, uint8_t *data, size_t len)
{
    uint8_t cmd = (uint8_t)((reg << 1) | 0x01u);
    return spi_read_after_cmd(dev, cmd, data, len);
}

static esp_err_t regs_write(node_acc_adxl355_dev_t *dev, uint8_t reg, const uint8_t *data, size_t len)
{
    if (dev == NULL || dev->spi == NULL || data == NULL || len == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    const size_t total = 1u + len;
    if (total > 64u) {
        return ESP_ERR_INVALID_ARG;
    }
    uint8_t tx_buf[64];
    uint8_t rx_buf[64];
    tx_buf[0] = (uint8_t)(reg << 1);
    memcpy(&tx_buf[1], data, len);

    spi_lock(dev);
    spi_transaction_t t = {0};
    t.length = (uint32_t)(total * 8u);
    t.tx_buffer = tx_buf;
    t.rx_buffer = rx_buf;
    esp_err_t err = spi_device_polling_transmit(dev->spi, &t);
    spi_unlock(dev);
    return err;
}

static int32_t axis_raw_from_3bytes(const uint8_t data[3])
{
    int32_t v = 0;
    ((uint8_t *)&v)[1] = data[2];
    ((uint8_t *)&v)[2] = data[1];
    ((uint8_t *)&v)[3] = data[0];
    return v / 4096;
}

static bool fifo_frame_is_x_header(const uint8_t frame[9])
{
    return ((frame[2] & 1u) != 0) && ((frame[2] & 2u) == 0);
}

static void fifo_xyz_from_frame(const uint8_t *p, node_acc_adxl355_raw_xyz_t *out)
{
    out->x = axis_raw_from_3bytes(&p[0]);
    out->y = axis_raw_from_3bytes(&p[3]);
    out->z = axis_raw_from_3bytes(&p[6]);
}

static void raw_xyz_from_bytes(const uint8_t data[9], node_acc_adxl355_raw_xyz_t *out)
{
    int32_t x = 0;
    int32_t y = 0;
    int32_t z = 0;
    ((uint8_t *)&x)[1] = data[2];
    ((uint8_t *)&x)[2] = data[1];
    ((uint8_t *)&x)[3] = data[0];
    ((uint8_t *)&y)[1] = data[5];
    ((uint8_t *)&y)[2] = data[4];
    ((uint8_t *)&y)[3] = data[3];
    ((uint8_t *)&z)[1] = data[8];
    ((uint8_t *)&z)[2] = data[7];
    ((uint8_t *)&z)[3] = data[6];
    out->x = x / 4096;
    out->y = y / 4096;
    out->z = z / 4096;
}

static float scale_for_range(node_acc_adxl355_range_t range)
{
    switch (range) {
    case NODE_ACC_ADXL355_RANGE_2G:
        return k_scale_2g;
    case NODE_ACC_ADXL355_RANGE_4G:
        return k_scale_4g;
    case NODE_ACC_ADXL355_RANGE_8G:
        return k_scale_8g;
    default:
        return k_scale_2g;
    }
}

/** POWER_CTL TEMP_OFF: Analog MEAS_TEMP_OFF_* vs MEAS_TEMP_ON_* (driver/ref-adxl355 adxl355_op_mode). */
static esp_err_t apply_meas_mode_hw(node_acc_adxl355_dev_t *dev)
{
    ESP_RETURN_ON_FALSE(dev != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");
    switch (dev->meas_mode) {
    case NODE_ACC_ADXL355_MEAS_ACCEL_AND_TEMP:
        return node_acc_adxl355_enable_temperature(dev);
    case NODE_ACC_ADXL355_MEAS_ACCEL_ONLY:
        return node_acc_adxl355_disable_temperature(dev);
    case NODE_ACC_ADXL355_MEAS_TEMP_ONLY:
        return node_acc_adxl355_enable_temperature(dev);
    default:
        return ESP_ERR_INVALID_ARG;
    }
}

esp_err_t node_acc_adxl355_set_measurement_mode(node_acc_adxl355_dev_t *dev, node_acc_adxl355_meas_mode_t mode)
{
    ESP_RETURN_ON_FALSE(dev != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");
    dev->meas_mode = mode;
    return apply_meas_mode_hw(dev);
}

esp_err_t node_acc_adxl355_get_measurement_mode(node_acc_adxl355_dev_t *dev, node_acc_adxl355_meas_mode_t *mode)
{
    ESP_RETURN_ON_FALSE(dev != NULL && mode != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");
    *mode = dev->meas_mode;
    return ESP_OK;
}

esp_err_t node_acc_adxl355_init(node_acc_adxl355_dev_t *dev, const node_acc_adxl355_config_t *cfg,
                                node_acc_adxl355_range_t range, node_acc_adxl355_odr_t odr)
{
    ESP_RETURN_ON_FALSE(dev != NULL && cfg != NULL, ESP_ERR_INVALID_ARG, TAG, "null arg");

    memset(dev, 0, sizeof(*dev));
    dev->gpio_drdy = cfg->gpio_drdy;
    dev->drdy_intr_type = cfg->drdy_intr_type;
    dev->gpio_int1 = cfg->gpio_int1;
    dev->gpio_int2 = cfg->gpio_int2;
    dev->int1_intr_type = cfg->int1_intr_type;
    dev->int2_intr_type = cfg->int2_intr_type;

    esp_err_t err = dev_spi_add(dev, cfg);
    if (err != ESP_OK) {
        return err;
    }

    if (cfg->spi_mutex_enable) {
        dev->spi_mutex = xSemaphoreCreateRecursiveMutex();
        if (dev->spi_mutex == NULL) {
            (void)dev_spi_remove(dev);
            return ESP_ERR_NO_MEM;
        }
    }

    dev->range = range;
    dev->odr = odr;
    dev->scale_factor = scale_for_range(range);

    err = node_acc_adxl355_reset(dev);
    if (err != ESP_OK) {
        goto fail;
    }
    vTaskDelay(pdMS_TO_TICKS(100));

    node_acc_adxl355_ids_t ids = {0};
    err = node_acc_adxl355_read_ids(dev, &ids);
    if (err != ESP_OK) {
        goto fail;
    }
    if (ids.devid_ad != NODE_ACC_ADXL355_EXPECT_DEVID_AD || ids.devid_mst != NODE_ACC_ADXL355_EXPECT_DEVID_MST ||
        ids.partid != NODE_ACC_ADXL355_EXPECT_PARTID) {
        ESP_LOGE(TAG, "ID mismatch: AD=0x%02X MST=0x%02X PART=0x%02X", ids.devid_ad, ids.devid_mst, ids.partid);
        err = ESP_ERR_NOT_FOUND;
        goto fail;
    }

    err = node_acc_adxl355_set_range(dev, range);
    if (err != ESP_OK) {
        goto fail;
    }
    err = node_acc_adxl355_set_odr(dev, odr);
    if (err != ESP_OK) {
        goto fail;
    }
    err = node_acc_adxl355_enable_measurement(dev);
    if (err != ESP_OK) {
        goto fail;
    }
    dev->meas_mode = cfg->meas_mode;
    err = apply_meas_mode_hw(dev);
    if (err != ESP_OK) {
        goto fail;
    }
    err = node_acc_adxl355_enable_data_ready_pin(dev);
    if (err != ESP_OK) {
        goto fail;
    }

    if (cfg->log_info_on_init) {
        ESP_LOGI(TAG, "init ok: range=%u odr=%u mode=%u", (unsigned)range, (unsigned)odr, (unsigned)dev->meas_mode);
    }
    return ESP_OK;

fail:
    dev_free_hw(dev);
    return err;
}

esp_err_t node_acc_adxl355_deinit(node_acc_adxl355_dev_t *dev)
{
    if (dev == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    (void)node_acc_adxl355_int_isr_remove(dev);
    (void)node_acc_adxl355_drdy_isr_remove(dev);

    if (dev->spi_mutex != NULL) {
        xSemaphoreTakeRecursive(dev->spi_mutex, portMAX_DELAY);
    }

    if (dev->spi != NULL) {
        node_acc_adxl355_int_map_t zmap = {0};
        (void)node_acc_adxl355_int_map_write(dev, &zmap);
        (void)node_acc_adxl355_disable_data_ready_pin(dev);
        (void)node_acc_adxl355_disable_measurement(dev);
        (void)dev_spi_remove(dev);
    }

    if (dev->spi_mutex != NULL) {
        vSemaphoreDelete(dev->spi_mutex);
        dev->spi_mutex = NULL;
    }

    memset(dev, 0, sizeof(*dev));
    return ESP_OK;
}

esp_err_t node_acc_adxl355_reset(node_acc_adxl355_dev_t *dev)
{
    ESP_RETURN_ON_FALSE(dev != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");
    return reg_write(dev, NODE_ACC_ADXL355_REG_RESET, NODE_ACC_ADXL355_REG_RESET_CODE);
}

esp_err_t node_acc_adxl355_read_ids(node_acc_adxl355_dev_t *dev, node_acc_adxl355_ids_t *ids)
{
    ESP_RETURN_ON_FALSE(dev != NULL && ids != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");

    uint8_t b[4] = {0};
    ESP_RETURN_ON_ERROR(regs_read(dev, NODE_ACC_ADXL355_REG_DEVID_AD, b, sizeof(b)), TAG, "regs");
    ids->devid_ad = b[0];
    ids->devid_mst = b[1];
    ids->partid = b[2];
    ids->revid = b[3];
    return ESP_OK;
}

esp_err_t node_acc_adxl355_read_status(node_acc_adxl355_dev_t *dev, uint8_t *status)
{
    ESP_RETURN_ON_FALSE(dev != NULL && status != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");
    return reg_read(dev, NODE_ACC_ADXL355_REG_STATUS, status);
}

esp_err_t node_acc_adxl355_read_raw_xyz(node_acc_adxl355_dev_t *dev, node_acc_adxl355_raw_xyz_t *out)
{
    ESP_RETURN_ON_FALSE(dev != NULL && out != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");
    if (dev->meas_mode == NODE_ACC_ADXL355_MEAS_TEMP_ONLY) {
        return ESP_ERR_INVALID_STATE;
    }

    uint8_t data[9] = {0};
    ESP_RETURN_ON_ERROR(regs_read(dev, NODE_ACC_ADXL355_REG_XDATA3, data, sizeof(data)), TAG, "burst");
    raw_xyz_from_bytes(data, out);
    return ESP_OK;
}

esp_err_t node_acc_adxl355_read_g(node_acc_adxl355_dev_t *dev, node_acc_adxl355_g_t *out)
{
    ESP_RETURN_ON_FALSE(dev != NULL && out != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");

    node_acc_adxl355_raw_xyz_t raw = {0};
    ESP_RETURN_ON_ERROR(node_acc_adxl355_read_raw_xyz(dev, &raw), TAG, "raw");

    float s = 0.0f;
    ESP_RETURN_ON_ERROR(node_acc_adxl355_acceleration_scale_g_per_lsb(dev, &s), TAG, "scale");
    out->x = (float)raw.x * s;
    out->y = (float)raw.y * s;
    out->z = (float)raw.z * s;
    return ESP_OK;
}

esp_err_t node_acc_adxl355_read_raw_temp(node_acc_adxl355_dev_t *dev, uint16_t *raw_temp)
{
    ESP_RETURN_ON_FALSE(dev != NULL && raw_temp != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");
    if (dev->meas_mode == NODE_ACC_ADXL355_MEAS_ACCEL_ONLY) {
        return ESP_ERR_INVALID_STATE;
    }

    uint8_t data[2] = {0};
    ESP_RETURN_ON_ERROR(regs_read(dev, NODE_ACC_ADXL355_REG_TEMP2, data, sizeof(data)), TAG, "temp regs");
    ((uint8_t *)raw_temp)[0] = data[1];
    ((uint8_t *)raw_temp)[1] = data[0];
    return ESP_OK;
}

esp_err_t node_acc_adxl355_read_temp_c(node_acc_adxl355_dev_t *dev, float *temp_c)
{
    ESP_RETURN_ON_FALSE(dev != NULL && temp_c != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");
    if (dev->meas_mode == NODE_ACC_ADXL355_MEAS_ACCEL_ONLY) {
        return ESP_ERR_INVALID_STATE;
    }

    uint16_t raw = 0;
    ESP_RETURN_ON_ERROR(node_acc_adxl355_read_raw_temp(dev, &raw), TAG, "raw temp");

    float v = ((float)(int16_t)raw - k_temp_intercept_lsb) / k_temp_slope_lsb_per_c + k_temp_ref_c;
    *temp_c = v;
    return ESP_OK;
}

esp_err_t node_acc_adxl355_set_range(node_acc_adxl355_dev_t *dev, node_acc_adxl355_range_t range)
{
    ESP_RETURN_ON_FALSE(dev != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");
    if (range != NODE_ACC_ADXL355_RANGE_2G && range != NODE_ACC_ADXL355_RANGE_4G &&
        range != NODE_ACC_ADXL355_RANGE_8G) {
        return ESP_ERR_INVALID_ARG;
    }

    uint8_t r = 0;
    ESP_RETURN_ON_ERROR(reg_read(dev, NODE_ACC_ADXL355_REG_RANGE, &r), TAG, "read range");
    r = (uint8_t)((r & (uint8_t)~NODE_ACC_ADXL355_REG_RANGE_RANGE_MASK) | (range & NODE_ACC_ADXL355_REG_RANGE_RANGE_MASK));
    ESP_RETURN_ON_ERROR(reg_write(dev, NODE_ACC_ADXL355_REG_RANGE, r), TAG, "write range");

    uint8_t verify = 0;
    ESP_RETURN_ON_ERROR(reg_read(dev, NODE_ACC_ADXL355_REG_RANGE, &verify), TAG, "verify range");
    if (verify != r) {
        return ESP_ERR_INVALID_STATE;
    }

    dev->range = range;
    dev->scale_factor = scale_for_range(range);
    return ESP_OK;
}

esp_err_t node_acc_adxl355_get_range(node_acc_adxl355_dev_t *dev, node_acc_adxl355_range_t *range)
{
    ESP_RETURN_ON_FALSE(dev != NULL && range != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");

    uint8_t r = 0;
    ESP_RETURN_ON_ERROR(reg_read(dev, NODE_ACC_ADXL355_REG_RANGE, &r), TAG, "read");
    *range = (node_acc_adxl355_range_t)(r & NODE_ACC_ADXL355_REG_RANGE_RANGE_MASK);
    return ESP_OK;
}

esp_err_t node_acc_adxl355_set_odr(node_acc_adxl355_dev_t *dev, node_acc_adxl355_odr_t odr)
{
    ESP_RETURN_ON_FALSE(dev != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");

    uint8_t f = 0;
    ESP_RETURN_ON_ERROR(reg_read(dev, NODE_ACC_ADXL355_REG_FILTER, &f), TAG, "read filter");
    f = (uint8_t)((f & (uint8_t)~NODE_ACC_ADXL355_REG_FILTER_ODR_MASK) | ((uint8_t)odr & NODE_ACC_ADXL355_REG_FILTER_ODR_MASK));
    ESP_RETURN_ON_ERROR(reg_write(dev, NODE_ACC_ADXL355_REG_FILTER, f), TAG, "write filter");

    uint8_t verify = 0;
    ESP_RETURN_ON_ERROR(reg_read(dev, NODE_ACC_ADXL355_REG_FILTER, &verify), TAG, "verify");
    if (verify != f) {
        return ESP_ERR_INVALID_STATE;
    }
    dev->odr = odr;
    return ESP_OK;
}

esp_err_t node_acc_adxl355_get_odr(node_acc_adxl355_dev_t *dev, node_acc_adxl355_odr_t *odr)
{
    ESP_RETURN_ON_FALSE(dev != NULL && odr != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");

    uint8_t f = 0;
    ESP_RETURN_ON_ERROR(reg_read(dev, NODE_ACC_ADXL355_REG_FILTER, &f), TAG, "read");
    *odr = (node_acc_adxl355_odr_t)(f & NODE_ACC_ADXL355_REG_FILTER_ODR_MASK);
    return ESP_OK;
}

esp_err_t node_acc_adxl355_enable_measurement(node_acc_adxl355_dev_t *dev)
{
    ESP_RETURN_ON_FALSE(dev != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");

    uint8_t p = 0;
    ESP_RETURN_ON_ERROR(reg_read(dev, NODE_ACC_ADXL355_REG_POWER_CTL, &p), TAG, "read");
    p = (uint8_t)(p & (uint8_t)~NODE_ACC_ADXL355_REG_POWER_CTL_STANDBY);
    return reg_write(dev, NODE_ACC_ADXL355_REG_POWER_CTL, p);
}

esp_err_t node_acc_adxl355_disable_measurement(node_acc_adxl355_dev_t *dev)
{
    ESP_RETURN_ON_FALSE(dev != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");

    uint8_t p = 0;
    ESP_RETURN_ON_ERROR(reg_read(dev, NODE_ACC_ADXL355_REG_POWER_CTL, &p), TAG, "read");
    p |= NODE_ACC_ADXL355_REG_POWER_CTL_STANDBY;
    return reg_write(dev, NODE_ACC_ADXL355_REG_POWER_CTL, p);
}

esp_err_t node_acc_adxl355_enable_temperature(node_acc_adxl355_dev_t *dev)
{
    ESP_RETURN_ON_FALSE(dev != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");

    uint8_t p = 0;
    ESP_RETURN_ON_ERROR(reg_read(dev, NODE_ACC_ADXL355_REG_POWER_CTL, &p), TAG, "read");
    p = (uint8_t)(p & (uint8_t)~NODE_ACC_ADXL355_REG_POWER_CTL_TEMP_OFF);
    return reg_write(dev, NODE_ACC_ADXL355_REG_POWER_CTL, p);
}

esp_err_t node_acc_adxl355_disable_temperature(node_acc_adxl355_dev_t *dev)
{
    ESP_RETURN_ON_FALSE(dev != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");

    uint8_t p = 0;
    ESP_RETURN_ON_ERROR(reg_read(dev, NODE_ACC_ADXL355_REG_POWER_CTL, &p), TAG, "read");
    p |= NODE_ACC_ADXL355_REG_POWER_CTL_TEMP_OFF;
    return reg_write(dev, NODE_ACC_ADXL355_REG_POWER_CTL, p);
}

esp_err_t node_acc_adxl355_enable_data_ready_pin(node_acc_adxl355_dev_t *dev)
{
    ESP_RETURN_ON_FALSE(dev != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");

    uint8_t p = 0;
    ESP_RETURN_ON_ERROR(reg_read(dev, NODE_ACC_ADXL355_REG_POWER_CTL, &p), TAG, "read");
    p = (uint8_t)(p & (uint8_t)~NODE_ACC_ADXL355_REG_POWER_CTL_DRDY_OFF);
    return reg_write(dev, NODE_ACC_ADXL355_REG_POWER_CTL, p);
}

esp_err_t node_acc_adxl355_disable_data_ready_pin(node_acc_adxl355_dev_t *dev)
{
    ESP_RETURN_ON_FALSE(dev != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");

    uint8_t p = 0;
    ESP_RETURN_ON_ERROR(reg_read(dev, NODE_ACC_ADXL355_REG_POWER_CTL, &p), TAG, "read");
    p |= NODE_ACC_ADXL355_REG_POWER_CTL_DRDY_OFF;
    return reg_write(dev, NODE_ACC_ADXL355_REG_POWER_CTL, p);
}

esp_err_t node_acc_adxl355_acceleration_scale_g_per_lsb(node_acc_adxl355_dev_t *dev, float *scale)
{
    ESP_RETURN_ON_FALSE(dev != NULL && scale != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");

    node_acc_adxl355_range_t range = NODE_ACC_ADXL355_RANGE_2G;
    ESP_RETURN_ON_ERROR(node_acc_adxl355_get_range(dev, &range), TAG, "range");
    *scale = scale_for_range(range);
    return ESP_OK;
}

esp_err_t node_acc_adxl355_drdy_isr_install(node_acc_adxl355_dev_t *dev)
{
    ESP_RETURN_ON_FALSE(dev != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");
    if (dev->gpio_drdy == GPIO_NUM_NC) {
        return ESP_ERR_NOT_SUPPORTED;
    }
    if (dev->drdy_intr_type == GPIO_INTR_DISABLE) {
        return ESP_ERR_INVALID_ARG;
    }

    if (dev->drdy_sem == NULL) {
        dev->drdy_sem = xSemaphoreCreateBinary();
        ESP_RETURN_ON_FALSE(dev->drdy_sem != NULL, ESP_ERR_NO_MEM, TAG, "sem");
    }
    if (dev->drdy_isr_installed) {
        return ESP_OK;
    }

    ESP_RETURN_ON_ERROR(gpio_isr_service_ensure_installed(), TAG, "isr_service");

    gpio_reset_pin(dev->gpio_drdy);
    gpio_config_t io = {
        .pin_bit_mask = 1ULL << (unsigned)dev->gpio_drdy,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = dev->drdy_intr_type,
    };
    ESP_RETURN_ON_ERROR(gpio_config(&io), TAG, "gpio_config");
    ESP_RETURN_ON_ERROR(gpio_isr_handler_add(dev->gpio_drdy, node_acc_adxl355_sem_isr, dev->drdy_sem), TAG,
                        "isr_add");
    dev->drdy_isr_installed = true;
    gpio_intr_enable(dev->gpio_drdy);
    return ESP_OK;
}

esp_err_t node_acc_adxl355_drdy_isr_remove(node_acc_adxl355_dev_t *dev)
{
    if (dev == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (!dev->drdy_isr_installed) {
        if (dev->drdy_sem != NULL) {
            vSemaphoreDelete(dev->drdy_sem);
            dev->drdy_sem = NULL;
        }
        return ESP_OK;
    }

    gpio_intr_disable(dev->gpio_drdy);
    (void)gpio_isr_handler_remove(dev->gpio_drdy);
    gpio_reset_pin(dev->gpio_drdy);
    dev->drdy_isr_installed = false;

    if (dev->drdy_sem != NULL) {
        vSemaphoreDelete(dev->drdy_sem);
        dev->drdy_sem = NULL;
    }
    return ESP_OK;
}

esp_err_t node_acc_adxl355_drdy_wait(node_acc_adxl355_dev_t *dev, uint32_t timeout_ms)
{
    ESP_RETURN_ON_FALSE(dev != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");
    ESP_RETURN_ON_FALSE(dev->drdy_sem != NULL, ESP_ERR_INVALID_STATE, TAG, "no sem (install ISR first)");

    const TickType_t ticks = pdMS_TO_TICKS(timeout_ms);
    return xSemaphoreTake(dev->drdy_sem, ticks) == pdTRUE ? ESP_OK : ESP_ERR_TIMEOUT;
}

static uint8_t int_map_pack(const node_acc_adxl355_int_map_t *m)
{
    uint8_t v = 0;
    if (m->rdy_int1) {
        v |= 1u << 0;
    }
    if (m->fifo_full_int1) {
        v |= 1u << 1;
    }
    if (m->fifo_ovr_int1) {
        v |= 1u << 2;
    }
    if (m->activity_int1) {
        v |= 1u << 3;
    }
    if (m->rdy_int2) {
        v |= 1u << 4;
    }
    if (m->fifo_full_int2) {
        v |= 1u << 5;
    }
    if (m->fifo_ovr_int2) {
        v |= 1u << 6;
    }
    if (m->activity_int2) {
        v |= 1u << 7;
    }
    return v;
}

static void int_map_unpack(uint8_t v, node_acc_adxl355_int_map_t *m)
{
    m->rdy_int1 = (v & (1u << 0)) != 0;
    m->fifo_full_int1 = (v & (1u << 1)) != 0;
    m->fifo_ovr_int1 = (v & (1u << 2)) != 0;
    m->activity_int1 = (v & (1u << 3)) != 0;
    m->rdy_int2 = (v & (1u << 4)) != 0;
    m->fifo_full_int2 = (v & (1u << 5)) != 0;
    m->fifo_ovr_int2 = (v & (1u << 6)) != 0;
    m->activity_int2 = (v & (1u << 7)) != 0;
}

esp_err_t node_acc_adxl355_int_map_write(node_acc_adxl355_dev_t *dev, const node_acc_adxl355_int_map_t *map)
{
    ESP_RETURN_ON_FALSE(dev != NULL && map != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");
    uint8_t v = int_map_pack(map);
    return reg_write(dev, NODE_ACC_ADXL355_REG_INT_MAP, v);
}

esp_err_t node_acc_adxl355_int_map_read(node_acc_adxl355_dev_t *dev, node_acc_adxl355_int_map_t *map)
{
    ESP_RETURN_ON_FALSE(dev != NULL && map != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");
    uint8_t v = 0;
    ESP_RETURN_ON_ERROR(reg_read(dev, NODE_ACC_ADXL355_REG_INT_MAP, &v), TAG, "read");
    int_map_unpack(v, map);
    return ESP_OK;
}

esp_err_t node_acc_adxl355_set_interrupt_polarity(node_acc_adxl355_dev_t *dev, node_acc_adxl355_int_pol_t pol)
{
    ESP_RETURN_ON_FALSE(dev != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");
    uint8_t r = 0;
    ESP_RETURN_ON_ERROR(reg_read(dev, NODE_ACC_ADXL355_REG_RANGE, &r), TAG, "read");
    r = (uint8_t)((r & (uint8_t)~NODE_ACC_ADXL355_REG_RANGE_INT_POL_MASK) |
                   (((uint8_t)pol << NODE_ACC_ADXL355_REG_RANGE_INT_POL_SHIFT) & NODE_ACC_ADXL355_REG_RANGE_INT_POL_MASK));
    return reg_write(dev, NODE_ACC_ADXL355_REG_RANGE, r);
}

static esp_err_t int_gpio_install_one(gpio_num_t gpio, gpio_int_type_t intr, SemaphoreHandle_t *sem, bool *installed)
{
    if (gpio == GPIO_NUM_NC) {
        return ESP_OK;
    }
    if (intr == GPIO_INTR_DISABLE) {
        return ESP_ERR_INVALID_ARG;
    }
    if (*sem == NULL) {
        *sem = xSemaphoreCreateBinary();
        ESP_RETURN_ON_FALSE(*sem != NULL, ESP_ERR_NO_MEM, TAG, "sem");
    }
    if (*installed) {
        return ESP_OK;
    }

    gpio_reset_pin(gpio);
    gpio_config_t io = {
        .pin_bit_mask = 1ULL << (unsigned)gpio,
        .mode = GPIO_MODE_INPUT,
        /* ADXL355 INT1/INT2 are open-drain active-low; idle high needs a pull (board or MCU). */
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = intr,
    };
    ESP_RETURN_ON_ERROR(gpio_config(&io), TAG, "gpio_config");
    ESP_RETURN_ON_ERROR(gpio_isr_handler_add(gpio, node_acc_adxl355_sem_isr, *sem), TAG, "isr_add");
    *installed = true;
    gpio_intr_enable(gpio);
    return ESP_OK;
}

static void int_gpio_remove_one(gpio_num_t gpio, SemaphoreHandle_t *sem, bool *installed)
{
    if (gpio == GPIO_NUM_NC) {
        return;
    }
    if (!*installed) {
        if (*sem != NULL) {
            vSemaphoreDelete(*sem);
            *sem = NULL;
        }
        return;
    }
    gpio_intr_disable(gpio);
    (void)gpio_isr_handler_remove(gpio);
    gpio_reset_pin(gpio);
    *installed = false;
    if (*sem != NULL) {
        vSemaphoreDelete(*sem);
        *sem = NULL;
    }
}

esp_err_t node_acc_adxl355_int_isr_install(node_acc_adxl355_dev_t *dev)
{
    ESP_RETURN_ON_FALSE(dev != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");
    if (dev->gpio_int1 == GPIO_NUM_NC && dev->gpio_int2 == GPIO_NUM_NC) {
        return ESP_ERR_NOT_SUPPORTED;
    }

    ESP_RETURN_ON_ERROR(gpio_isr_service_ensure_installed(), TAG, "isr_service");

    esp_err_t e = int_gpio_install_one(dev->gpio_int1, dev->int1_intr_type, &dev->int1_sem, &dev->int1_isr_installed);
    if (e != ESP_OK) {
        return e;
    }
    e = int_gpio_install_one(dev->gpio_int2, dev->int2_intr_type, &dev->int2_sem, &dev->int2_isr_installed);
    return e;
}

esp_err_t node_acc_adxl355_int_isr_remove(node_acc_adxl355_dev_t *dev)
{
    if (dev == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    int_gpio_remove_one(dev->gpio_int1, &dev->int1_sem, &dev->int1_isr_installed);
    int_gpio_remove_one(dev->gpio_int2, &dev->int2_sem, &dev->int2_isr_installed);
    return ESP_OK;
}

esp_err_t node_acc_adxl355_int1_wait(node_acc_adxl355_dev_t *dev, uint32_t timeout_ms)
{
    ESP_RETURN_ON_FALSE(dev != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");
    ESP_RETURN_ON_FALSE(dev->int1_sem != NULL, ESP_ERR_INVALID_STATE, TAG, "INT1 ISR not installed");
    const TickType_t ticks = pdMS_TO_TICKS(timeout_ms);
    return xSemaphoreTake(dev->int1_sem, ticks) == pdTRUE ? ESP_OK : ESP_ERR_TIMEOUT;
}

esp_err_t node_acc_adxl355_int2_wait(node_acc_adxl355_dev_t *dev, uint32_t timeout_ms)
{
    ESP_RETURN_ON_FALSE(dev != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");
    ESP_RETURN_ON_FALSE(dev->int2_sem != NULL, ESP_ERR_INVALID_STATE, TAG, "INT2 ISR not installed");
    const TickType_t ticks = pdMS_TO_TICKS(timeout_ms);
    return xSemaphoreTake(dev->int2_sem, ticks) == pdTRUE ? ESP_OK : ESP_ERR_TIMEOUT;
}

esp_err_t node_acc_adxl355_set_fifo_samples(node_acc_adxl355_dev_t *dev, uint8_t samples)
{
    ESP_RETURN_ON_FALSE(dev != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");
    if (samples > NODE_ACC_ADXL355_FIFO_SAMPLES_MAX) {
        return ESP_ERR_INVALID_ARG;
    }
    return reg_write(dev, NODE_ACC_ADXL355_REG_FIFO_SAMPLES, samples);
}

esp_err_t node_acc_adxl355_get_fifo_samples(node_acc_adxl355_dev_t *dev, uint8_t *samples)
{
    ESP_RETURN_ON_FALSE(dev != NULL && samples != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");
    return reg_read(dev, NODE_ACC_ADXL355_REG_FIFO_SAMPLES, samples);
}

esp_err_t node_acc_adxl355_read_fifo_entries(node_acc_adxl355_dev_t *dev, uint8_t *entries)
{
    ESP_RETURN_ON_FALSE(dev != NULL && entries != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");
    return reg_read(dev, NODE_ACC_ADXL355_REG_FIFO_ENTRIES, entries);
}

esp_err_t node_acc_adxl355_read_fifo_xyz(node_acc_adxl355_dev_t *dev, node_acc_adxl355_raw_xyz_t *out,
                                         size_t max_frames, size_t *out_frames)
{
    if (out_frames != NULL) {
        *out_frames = 0;
    }
    ESP_RETURN_ON_FALSE(dev != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");
    ESP_RETURN_ON_FALSE(out != NULL && max_frames > 0, ESP_ERR_INVALID_ARG, TAG, "out");
    if (dev->meas_mode == NODE_ACC_ADXL355_MEAS_TEMP_ONLY) {
        return ESP_ERR_INVALID_STATE;
    }

    uint8_t ent = 0;
    ESP_RETURN_ON_ERROR(node_acc_adxl355_read_fifo_entries(dev, &ent), TAG, "fifo_entries");
    uint8_t n = (uint8_t)(((unsigned)ent / 3u) * 3u);
    if (n == 0) {
        return ESP_OK;
    }

    const size_t byte_len = (size_t)n * 3u;
    uint8_t *buf = heap_caps_malloc(byte_len, MALLOC_CAP_DMA);
    ESP_RETURN_ON_FALSE(buf != NULL, ESP_ERR_NO_MEM, TAG, "fifo buf");

    uint8_t cmd = (uint8_t)((NODE_ACC_ADXL355_REG_FIFO_DATA << 1) | 0x01u);
    esp_err_t err = spi_read_after_cmd(dev, cmd, buf, byte_len);
    if (err != ESP_OK) {
        heap_caps_free(buf);
        return err;
    }

    const size_t num_frames = byte_len / 9u;
    size_t k = 0;
    for (size_t i = 0; i < num_frames && k < max_frames; i++) {
        const uint8_t *p = &buf[i * 9u];
        if (fifo_frame_is_x_header(p)) {
            fifo_xyz_from_frame(p, &out[k]);
            k++;
        }
    }
    if (k == 0 && num_frames >= 1) {
        fifo_xyz_from_frame(buf, &out[0]);
        k = 1;
    }
    if (out_frames != NULL) {
        *out_frames = k;
    }
    heap_caps_free(buf);
    return ESP_OK;
}

esp_err_t node_acc_adxl355_set_offset_xyz(node_acc_adxl355_dev_t *dev, const node_acc_adxl355_offset_t *off)
{
    ESP_RETURN_ON_FALSE(dev != NULL && off != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");
    uint8_t b[6];
    b[0] = (uint8_t)(off->x >> 8);
    b[1] = (uint8_t)(off->x);
    b[2] = (uint8_t)(off->y >> 8);
    b[3] = (uint8_t)(off->y);
    b[4] = (uint8_t)(off->z >> 8);
    b[5] = (uint8_t)(off->z);
    return regs_write(dev, NODE_ACC_ADXL355_REG_OFFSET_X_H, b, sizeof(b));
}

esp_err_t node_acc_adxl355_get_offset_xyz(node_acc_adxl355_dev_t *dev, node_acc_adxl355_offset_t *off)
{
    ESP_RETURN_ON_FALSE(dev != NULL && off != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");
    uint8_t b[6] = {0};
    ESP_RETURN_ON_ERROR(regs_read(dev, NODE_ACC_ADXL355_REG_OFFSET_X_H, b, sizeof(b)), TAG, "offset read");
    off->x = (uint16_t)(((unsigned)b[0] << 8) | b[1]);
    off->y = (uint16_t)(((unsigned)b[2] << 8) | b[3]);
    off->z = (uint16_t)(((unsigned)b[4] << 8) | b[5]);
    return ESP_OK;
}

esp_err_t node_acc_adxl355_set_activity_enable(node_acc_adxl355_dev_t *dev, const node_acc_adxl355_act_en_t *en)
{
    ESP_RETURN_ON_FALSE(dev != NULL && en != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");
    uint8_t v = 0;
    if (en->x) {
        v |= 1u;
    }
    if (en->y) {
        v |= 2u;
    }
    if (en->z) {
        v |= 4u;
    }
    return reg_write(dev, NODE_ACC_ADXL355_REG_ACT_EN, v);
}

esp_err_t node_acc_adxl355_get_activity_enable(node_acc_adxl355_dev_t *dev, node_acc_adxl355_act_en_t *en)
{
    ESP_RETURN_ON_FALSE(dev != NULL && en != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");
    uint8_t v = 0;
    ESP_RETURN_ON_ERROR(reg_read(dev, NODE_ACC_ADXL355_REG_ACT_EN, &v), TAG, "read");
    en->x = (v & 1u) != 0;
    en->y = (v & 2u) != 0;
    en->z = (v & 4u) != 0;
    return ESP_OK;
}

esp_err_t node_acc_adxl355_set_activity_threshold(node_acc_adxl355_dev_t *dev, uint16_t threshold)
{
    ESP_RETURN_ON_FALSE(dev != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");
    uint8_t d[2] = {(uint8_t)(threshold >> 8), (uint8_t)threshold};
    return regs_write(dev, NODE_ACC_ADXL355_REG_ACT_THRESH_H, d, sizeof(d));
}

esp_err_t node_acc_adxl355_get_activity_threshold(node_acc_adxl355_dev_t *dev, uint16_t *threshold)
{
    ESP_RETURN_ON_FALSE(dev != NULL && threshold != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");
    uint8_t d[2] = {0};
    ESP_RETURN_ON_ERROR(regs_read(dev, NODE_ACC_ADXL355_REG_ACT_THRESH_H, d, sizeof(d)), TAG, "read");
    *threshold = (uint16_t)(((unsigned)d[0] << 8) | d[1]);
    return ESP_OK;
}

esp_err_t node_acc_adxl355_set_activity_count(node_acc_adxl355_dev_t *dev, uint8_t count)
{
    ESP_RETURN_ON_FALSE(dev != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "dev");
    return reg_write(dev, NODE_ACC_ADXL355_REG_ACT_COUNT, count);
}

esp_err_t node_acc_adxl355_get_activity_count(node_acc_adxl355_dev_t *dev, uint8_t *count)
{
    ESP_RETURN_ON_FALSE(dev != NULL && count != NULL && dev->spi != NULL, ESP_ERR_INVALID_ARG, TAG, "arg");
    return reg_read(dev, NODE_ACC_ADXL355_REG_ACT_COUNT, count);
}
