/**
 * @file node_acc_adxl367_test.c
 * @brief Phases 1-5 tests; Phase 6: run_phase6_default_app.
 */
#include "node_acc_adxl367_test.h"

#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "node_acc_adxl367.h"
#include "node_i2c.h"

#include <inttypes.h>
#include <stdint.h>

static const char *TAG = "adxl367_test";

/** ~g from raw accel code (same scale intent as @c NODE_ACC_ADXL367_THRESH_ACT_LSB_FROM_G). */
#define ADXL367_STATS_CODE_TO_G(c) ((double)(c) / 4000.0)

/** ADXL367 INT1 on this board (DOC: MORE-PERCEPTION/ADXL367). Override if wiring differs. */
#ifndef NODE_ACC_ADXL367_TEST_INT1_GPIO
#define NODE_ACC_ADXL367_TEST_INT1_GPIO 10
#endif

/** ADXL367 INT2 (DOC: MORE-PERCEPTION/ADXL367). */
#ifndef NODE_ACC_ADXL367_TEST_INT2_GPIO
#define NODE_ACC_ADXL367_TEST_INT2_GPIO 18
#endif

/** Phase 5: monitor window (ms) for linked ACT/INACT demo. */
#ifndef NODE_ACC_ADXL367_TEST_P5_WINDOW_MS
#define NODE_ACC_ADXL367_TEST_P5_WINDOW_MS 60000
#endif
#ifndef NODE_ACC_ADXL367_TEST_P5_POLL_MS
#define NODE_ACC_ADXL367_TEST_P5_POLL_MS 50
#endif
/** Delay before gpio_intr_enable(INT2) after config (baseline settle). */
#ifndef NODE_ACC_ADXL367_TEST_P5_ARM_MS
#define NODE_ACC_ADXL367_TEST_P5_ARM_MS 3000
#endif
/** Phase 5 THRESH_ACT scale (g, +/-2g intent). */
#ifndef NODE_ACC_ADXL367_TEST_P5_THRESH_G
#define NODE_ACC_ADXL367_TEST_P5_THRESH_G 0.22f
#endif
/** TIME_ACT sample count at ODR. */
#ifndef NODE_ACC_ADXL367_TEST_P5_TIME_ACT_SAMPLES
#define NODE_ACC_ADXL367_TEST_P5_TIME_ACT_SAMPLES 32u
#endif
/** THRESH_INACT (g) for referenced inactivity. */
#ifndef NODE_ACC_ADXL367_TEST_P5_THRESH_INACT_G
#define NODE_ACC_ADXL367_TEST_P5_THRESH_INACT_G 0.10f
#endif
/** TIME_INACT sample count. */
#ifndef NODE_ACC_ADXL367_TEST_P5_TIME_INACT_SAMPLES
#define NODE_ACC_ADXL367_TEST_P5_TIME_INACT_SAMPLES 250u
#endif
/** ACT_INACT_CTL[5:4]: 3 = linked + internal STATUS ack (see datasheet). */
#ifndef NODE_ACC_ADXL367_TEST_P5_LINKLOOP_MODE
#define NODE_ACC_ADXL367_TEST_P5_LINKLOOP_MODE 3u
#endif

/** Stats capture duration (ms), one sample per DATA_RDY. */
#ifndef NODE_ACC_ADXL367_TEST_STATS_MS
#define NODE_ACC_ADXL367_TEST_STATS_MS 15000
#endif

/** max(|dx|,|dy|,|dz|) between consecutive samples; above: still_to_motion. */
#ifndef NODE_ACC_ADXL367_MONITOR_DELTA_HIGH_LSB
#define NODE_ACC_ADXL367_MONITOR_DELTA_HIGH_LSB 350
#endif
/** Below this for MONITOR_STILL_COUNT samples: motion_to_still. */
#ifndef NODE_ACC_ADXL367_MONITOR_DELTA_LOW_LSB
#define NODE_ACC_ADXL367_MONITOR_DELTA_LOW_LSB 120
#endif
#ifndef NODE_ACC_ADXL367_MONITOR_STILL_COUNT
#define NODE_ACC_ADXL367_MONITOR_STILL_COUNT 45u
#endif
/** Ignore saturated / invalid samples (|code| above this on any axis). */
#ifndef NODE_ACC_ADXL367_MONITOR_ABS_MAX
#define NODE_ACC_ADXL367_MONITOR_ABS_MAX 7800
#endif

static uint32_t adxl367_uabs_diff(int16_t a, int16_t b)
{
    int32_t d = (int32_t)a - (int32_t)b;
    if (d < 0) {
        d = -d;
    }
    return (uint32_t)d;
}

static uint32_t adxl367_max_u32(uint32_t a, uint32_t b, uint32_t c)
{
    uint32_t m = a;
    if (b > m) {
        m = b;
    }
    if (c > m) {
        m = c;
    }
    return m;
}

static void IRAM_ATTR p4_fifo_int_isr(void *arg)
{
    SemaphoreHandle_t sem = (SemaphoreHandle_t)arg;
    if (sem != NULL) {
        (void)xSemaphoreGiveFromISR(sem, NULL);
    }
}

static void IRAM_ATTR p5_act_int_isr(void *arg)
{
    SemaphoreHandle_t sem = (SemaphoreHandle_t)arg;
    if (sem != NULL) {
        (void)xSemaphoreGiveFromISR(sem, NULL);
    }
}

static esp_err_t probe_i2c_and_init(node_acc_adxl367_dev_t *acc, i2c_master_dev_handle_t *out_dev)
{
    static const uint8_t k_addrs[] = {
        NODE_ACC_ADXL367_I2C_ADDR_0X1D,
        NODE_ACC_ADXL367_I2C_ADDR_0X53,
    };

    esp_err_t ret = ESP_ERR_NOT_FOUND;
    *out_dev = NULL;

    for (size_t a = 0; a < sizeof(k_addrs) / sizeof(k_addrs[0]); a++) {
        uint8_t addr = k_addrs[a];
        ret = i2c_add_device(addr, I2C_MASTER_FREQ_HZ, out_dev);
        if (ret != ESP_OK) {
            continue;
        }
        acc->i2c = *out_dev;
        ret = node_acc_adxl367_init(acc);
        if (ret == ESP_OK) {
            ESP_LOGI(TAG, "ADXL367 at I2C 0x%02X", (unsigned)addr);
            return ESP_OK;
        }
        ESP_LOGD(TAG, "no chip at 0x%02X: %s", (unsigned)addr, esp_err_to_name(ret));
        (void)i2c_remove_device(*out_dev);
        *out_dev = NULL;
    }

    if (*out_dev == NULL) {
        ESP_LOGE(TAG, "No ADXL367 on I2C (tried 0x1D and 0x53).");
    }
    return ret;
}

esp_err_t node_acc_adxl367_run_xyz_stats_capture(void)
{
    i2c_master_dev_handle_t i2c_dev = NULL;
    node_acc_adxl367_dev_t acc = {0};

    esp_err_t ret = probe_i2c_and_init(&acc, &i2c_dev);
    if (ret != ESP_OK) {
        return ret;
    }

    ret = node_acc_adxl367_set_measurement_mode(&acc, NODE_ACC_ADXL367_MEAS_ACCEL_ONLY);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "stats: set_measurement_mode failed: %s", esp_err_to_name(ret));
        goto out;
    }

    int64_t sx = 0, sy = 0, sz = 0;
    int16_t min_x = 0, max_x = 0, min_y = 0, max_y = 0, min_z = 0, max_z = 0;
    uint32_t n = 0;
    bool first = true;

    ESP_LOGI(TAG, "======== XYZ stats capture: %d ms (keep board still for noise floor) ========",
             (int)NODE_ACC_ADXL367_TEST_STATS_MS);
    const TickType_t t_end = xTaskGetTickCount() + pdMS_TO_TICKS(NODE_ACC_ADXL367_TEST_STATS_MS);

    while (xTaskGetTickCount() < t_end) {
        int16_t x = 0, y = 0, z = 0;
        ret = node_acc_adxl367_read_xyz_raw(&acc, &x, &y, &z);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "stats: read_xyz_raw failed: %s", esp_err_to_name(ret));
            break;
        }
        sx += x;
        sy += y;
        sz += z;
        if (first) {
            min_x = max_x = x;
            min_y = max_y = y;
            min_z = max_z = z;
            first = false;
        } else {
            if (x < min_x) {
                min_x = x;
            }
            if (x > max_x) {
                max_x = x;
            }
            if (y < min_y) {
                min_y = y;
            }
            if (y > max_y) {
                max_y = y;
            }
            if (z < min_z) {
                min_z = z;
            }
            if (z > max_z) {
                max_z = z;
            }
        }
        n++;
    }

    if (ret == ESP_OK && n > 0) {
        const int32_t mx = (int32_t)(sx / (int64_t)n);
        const int32_t my = (int32_t)(sy / (int64_t)n);
        const int32_t mz = (int32_t)(sz / (int64_t)n);
        const int32_t span_x = (int32_t)max_x - (int32_t)min_x;
        const int32_t span_y = (int32_t)max_y - (int32_t)min_y;
        const int32_t span_z = (int32_t)max_z - (int32_t)min_z;

        ESP_LOGI(TAG, "stats: samples=%u (~%.0f Hz if steady DATA_RDY)", (unsigned)n,
                 (double)n * 1000.0 / (double)NODE_ACC_ADXL367_TEST_STATS_MS);
        ESP_LOGI(TAG, "stats: X mean=%" PRId32 " (~%.4f g)  min=%d max=%d  span=%" PRId32 " (~%.4f g)", mx,
                 ADXL367_STATS_CODE_TO_G(mx), (int)min_x, (int)max_x, (int32_t)span_x,
                 ADXL367_STATS_CODE_TO_G(span_x));
        ESP_LOGI(TAG, "stats: Y mean=%" PRId32 " (~%.4f g)  min=%d max=%d  span=%" PRId32 " (~%.4f g)", my,
                 ADXL367_STATS_CODE_TO_G(my), (int)min_y, (int)max_y, (int32_t)span_y,
                 ADXL367_STATS_CODE_TO_G(span_y));
        ESP_LOGI(TAG, "stats: Z mean=%" PRId32 " (~%.4f g)  min=%d max=%d  span=%" PRId32 " (~%.4f g)", mz,
                 ADXL367_STATS_CODE_TO_G(mz), (int)min_z, (int)max_z, (int32_t)span_z,
                 ADXL367_STATS_CODE_TO_G(span_z));
        ESP_LOGI(TAG,
                 "stats: THRESH_ACT LSB ~ g*4000 (+/-2g intent)");
    } else if (n == 0) {
        ret = ESP_ERR_INVALID_STATE;
    }

out:
    if (i2c_dev != NULL) {
        (void)i2c_remove_device(i2c_dev);
    }
    return ret;
}

esp_err_t node_acc_adxl367_run_phase1_test(void)
{
    i2c_master_dev_handle_t i2c_dev = NULL;
    node_acc_adxl367_dev_t acc = {0};

    esp_err_t ret = probe_i2c_and_init(&acc, &i2c_dev);
    if (ret != ESP_OK) {
        return ret;
    }

    for (int i = 0; i < 8; i++) {
        int16_t x = 0, y = 0, z = 0;
        ret = node_acc_adxl367_read_xyz_raw(&acc, &x, &y, &z);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "read_xyz_raw failed: %s", esp_err_to_name(ret));
            break;
        }
        ESP_LOGI(TAG, "p1 sample %d: x=%d y=%d z=%d", i, (int)x, (int)y, (int)z);
        vTaskDelay(pdMS_TO_TICKS(100));
    }

    (void)i2c_remove_device(i2c_dev);
    return ret;
}

esp_err_t node_acc_adxl367_run_phase2_test(void)
{
    i2c_master_dev_handle_t i2c_dev = NULL;
    node_acc_adxl367_dev_t acc = {0};

    esp_err_t ret = probe_i2c_and_init(&acc, &i2c_dev);
    if (ret != ESP_OK) {
        return ret;
    }

    /* ACCEL_ONLY: one XYZ sample */
    ret = node_acc_adxl367_set_measurement_mode(&acc, NODE_ACC_ADXL367_MEAS_ACCEL_ONLY);
    if (ret != ESP_OK) {
        goto out;
    }
    {
        int16_t x = 0, y = 0, z = 0;
        ret = node_acc_adxl367_read_xyz_raw(&acc, &x, &y, &z);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "p2 accel-only xyz failed: %s", esp_err_to_name(ret));
            goto out;
        }
        ESP_LOGI(TAG, "p2 accel-only: x=%d y=%d z=%d", (int)x, (int)y, (int)z);
    }

    /* ACCEL_AND_TEMP */
    ret = node_acc_adxl367_set_measurement_mode(&acc, NODE_ACC_ADXL367_MEAS_ACCEL_AND_TEMP);
    if (ret != ESP_OK) {
        goto out;
    }
    {
        int16_t x = 0, y = 0, z = 0;
        float tc = 0;
        ret = node_acc_adxl367_read_xyz_raw(&acc, &x, &y, &z);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "p2 accel+temp xyz failed: %s", esp_err_to_name(ret));
            goto out;
        }
        ret = node_acc_adxl367_read_temp_celsius(&acc, &tc);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "p2 accel+temp temp C failed: %s", esp_err_to_name(ret));
            goto out;
        }
        ESP_LOGI(TAG, "p2 accel+temp: x=%d y=%d z=%d temp=%.2f C", (int)x, (int)y, (int)z, (double)tc);
    }

    /* TEMP_ONLY */
    ret = node_acc_adxl367_set_measurement_mode(&acc, NODE_ACC_ADXL367_MEAS_TEMP_ONLY);
    if (ret != ESP_OK) {
        goto out;
    }
    {
        float tc = 0;
        ret = node_acc_adxl367_read_temp_celsius(&acc, &tc);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "p2 temp-only temp failed: %s", esp_err_to_name(ret));
            goto out;
        }
        ESP_LOGI(TAG, "p2 temp-only: temp=%.2f C", (double)tc);

        int16_t x = 0, y = 0, z = 0;
        ret = node_acc_adxl367_read_xyz_raw(&acc, &x, &y, &z);
        if (ret != ESP_ERR_INVALID_STATE) {
            ESP_LOGE(TAG, "p2 temp-only: expected INVALID_STATE for xyz");
            ret = ESP_FAIL;
            goto out;
        }
        ESP_LOGI(TAG, "p2 temp-only: xyz correctly rejected (INVALID_STATE)");
        ret = ESP_OK;
    }

out:
    if (i2c_dev != NULL) {
        (void)i2c_remove_device(i2c_dev);
    }
    return ret;
}

/** STATUS FIFO watermark bit. */
#define P3_STATUS_FIFO_WATERMARK (1u << 2)

esp_err_t node_acc_adxl367_run_phase3_test(void)
{
    i2c_master_dev_handle_t i2c_dev = NULL;
    node_acc_adxl367_dev_t acc = {0};

    esp_err_t ret = probe_i2c_and_init(&acc, &i2c_dev);
    if (ret != ESP_OK) {
        return ret;
    }

    ret = node_acc_adxl367_set_measurement_mode(&acc, NODE_ACC_ADXL367_MEAS_ACCEL_ONLY);
    if (ret != ESP_OK) {
        goto out;
    }

    /* Stream mode, XYZ, watermark at 8 sample sets (XYZ => 8*3 FIFO entries when full). */
    ret = node_acc_adxl367_fifo_setup(&acc, NODE_ACC_ADXL367_FIFO_STREAM, NODE_ACC_ADXL367_FIFO_FMT_XYZ, 8);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "p3 fifo_setup failed: %s", esp_err_to_name(ret));
        goto out;
    }

    for (int w = 0; w < 50; w++) {
        uint8_t st = 0;
        ret = node_acc_adxl367_read_status(&acc, &st);
        if (ret != ESP_OK) {
            goto out;
        }
        if ((st & P3_STATUS_FIFO_WATERMARK) != 0) {
            ESP_LOGI(TAG, "p3 STATUS 0x%02X (FIFO watermark)", (unsigned)st);
            break;
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }

    uint16_t n_ent = 0;
    ret = node_acc_adxl367_fifo_get_entry_count(&acc, &n_ent);
    if (ret != ESP_OK) {
        goto out;
    }
    ESP_LOGI(TAG, "p3 FIFO entries before drain: %u", (unsigned)n_ent);

    int16_t x[32], y[32], z[32];
    size_t nt = 0;
    ret = node_acc_adxl367_fifo_drain_xyz(&acc, x, y, z, sizeof(x) / sizeof(x[0]), &nt);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "p3 fifo_drain_xyz failed: %s", esp_err_to_name(ret));
        goto out;
    }
    ESP_LOGI(TAG, "p3 drained complete XYZ triplets: %u", (unsigned)nt);
    for (size_t i = 0; i < nt && i < 4u; i++) {
        ESP_LOGI(TAG, "p3 triplet %u: x=%d y=%d z=%d", (unsigned)i, (int)x[i], (int)y[i], (int)z[i]);
    }

    ret = node_acc_adxl367_fifo_disable(&acc);
    if (ret != ESP_OK) {
        goto out;
    }
    ESP_LOGI(TAG, "p3 FIFO disabled OK");

out:
    if (i2c_dev != NULL) {
        (void)i2c_remove_device(i2c_dev);
    }
    return ret;
}

esp_err_t node_acc_adxl367_run_phase4_test(void)
{
    i2c_master_dev_handle_t i2c_dev = NULL;
    node_acc_adxl367_dev_t acc = {0};
    SemaphoreHandle_t sem = NULL;
    bool handler_added = false;

    esp_err_t ret = probe_i2c_and_init(&acc, &i2c_dev);
    if (ret != ESP_OK) {
        return ret;
    }

    sem = xSemaphoreCreateBinary();
    if (sem == NULL) {
        ret = ESP_ERR_NO_MEM;
        goto out;
    }

    ret = node_acc_adxl367_set_measurement_mode(&acc, NODE_ACC_ADXL367_MEAS_ACCEL_ONLY);
    if (ret != ESP_OK) {
        goto out_sem;
    }

    node_acc_adxl367_int_map_t imap = {0};
    imap.fifo_watermark = 1u;
    imap.int_low = 1u;
    ret = node_acc_adxl367_int_map(&acc, 1u, &imap);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "p4 int_map failed: %s", esp_err_to_name(ret));
        goto out_sem;
    }

    ret = node_acc_adxl367_fifo_setup(&acc, NODE_ACC_ADXL367_FIFO_STREAM, NODE_ACC_ADXL367_FIFO_FMT_XYZ, 8);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "p4 fifo_setup failed: %s", esp_err_to_name(ret));
        goto out_sem;
    }

    ret = gpio_install_isr_service(0);
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "p4 gpio_install_isr_service failed: %s", esp_err_to_name(ret));
        goto out_sem;
    }

    gpio_config_t io = {
        .pin_bit_mask = 1ull << NODE_ACC_ADXL367_TEST_INT1_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_NEGEDGE,
    };
    ret = gpio_config(&io);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "p4 gpio_config failed: %s", esp_err_to_name(ret));
        goto out_sem;
    }

    ret = gpio_isr_handler_add((gpio_num_t)NODE_ACC_ADXL367_TEST_INT1_GPIO, p4_fifo_int_isr, sem);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "p4 gpio_isr_handler_add failed: %s", esp_err_to_name(ret));
        goto out_sem;
    }
    handler_added = true;
    gpio_intr_enable((gpio_num_t)NODE_ACC_ADXL367_TEST_INT1_GPIO);

    if (xSemaphoreTake(sem, pdMS_TO_TICKS(3000)) != pdTRUE) {
        ESP_LOGE(TAG, "p4 timeout waiting for INT1 (FIFO watermark)");
        ret = ESP_ERR_TIMEOUT;
        goto out_gpio;
    }

    uint8_t st = 0;
    ret = node_acc_adxl367_read_status(&acc, &st);
    if (ret != ESP_OK) {
        goto out_gpio;
    }
    ESP_LOGI(TAG, "p4 STATUS 0x%02X (after INT)", (unsigned)st);

    uint16_t n_ent = 0;
    ret = node_acc_adxl367_fifo_get_entry_count(&acc, &n_ent);
    if (ret != ESP_OK) {
        goto out_gpio;
    }
    ESP_LOGI(TAG, "p4 FIFO entries: %u", (unsigned)n_ent);

    int16_t x[32], y[32], z[32];
    size_t nt = 0;
    ret = node_acc_adxl367_fifo_drain_xyz(&acc, x, y, z, sizeof(x) / sizeof(x[0]), &nt);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "p4 fifo_drain_xyz failed: %s", esp_err_to_name(ret));
        goto out_gpio;
    }
    ESP_LOGI(TAG, "p4 drained XYZ triplets: %u", (unsigned)nt);
    if (nt > 0) {
        ESP_LOGI(TAG, "p4 first triplet: x=%d y=%d z=%d", (int)x[0], (int)y[0], (int)z[0]);
    }

    ret = node_acc_adxl367_fifo_disable(&acc);
    if (ret != ESP_OK) {
        goto out_gpio;
    }
    ESP_LOGI(TAG, "p4 FIFO disabled OK");

out_gpio:
    if (handler_added) {
        gpio_intr_disable((gpio_num_t)NODE_ACC_ADXL367_TEST_INT1_GPIO);
        gpio_isr_handler_remove((gpio_num_t)NODE_ACC_ADXL367_TEST_INT1_GPIO);
    }
    gpio_reset_pin((gpio_num_t)NODE_ACC_ADXL367_TEST_INT1_GPIO);

out_sem:
    vSemaphoreDelete(sem);

out:
    if (i2c_dev != NULL) {
        (void)i2c_remove_device(i2c_dev);
    }
    return ret;
}

esp_err_t node_acc_adxl367_run_phase5_test(void)
{
    i2c_master_dev_handle_t i2c_dev = NULL;
    node_acc_adxl367_dev_t acc = {0};
    SemaphoreHandle_t sem = NULL;
    bool handler_added = false;

    esp_err_t ret = probe_i2c_and_init(&acc, &i2c_dev);
    if (ret != ESP_OK) {
        return ret;
    }

    sem = xSemaphoreCreateBinary();
    if (sem == NULL) {
        ret = ESP_ERR_NO_MEM;
        goto out;
    }

    ret = node_acc_adxl367_fifo_disable(&acc);
    if (ret != ESP_OK) {
        goto out_sem;
    }

    ret = node_acc_adxl367_act_inact_disable(&acc);
    if (ret != ESP_OK) {
        goto out_sem;
    }

    ret = node_acc_adxl367_set_measurement_mode(&acc, NODE_ACC_ADXL367_MEAS_ACCEL_ONLY);
    if (ret != ESP_OK) {
        goto out_sem;
    }

    /* REF activity/inact, LINKLOOP=3, X axis only. */
    ret = node_acc_adxl367_setup_activity_detection(
        &acc, NODE_ACC_ADXL367_AI_REF_REL, NODE_ACC_ADXL367_THRESH_ACT_LSB_FROM_G(NODE_ACC_ADXL367_TEST_P5_THRESH_G),
        (uint8_t)NODE_ACC_ADXL367_TEST_P5_TIME_ACT_SAMPLES);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "p5 setup_activity failed: %s", esp_err_to_name(ret));
        goto out_sem;
    }
    ret = node_acc_adxl367_setup_inactivity_detection(
        &acc, NODE_ACC_ADXL367_AI_REF_REL,
        NODE_ACC_ADXL367_THRESH_ACT_LSB_FROM_G(NODE_ACC_ADXL367_TEST_P5_THRESH_INACT_G),
        (uint16_t)NODE_ACC_ADXL367_TEST_P5_TIME_INACT_SAMPLES);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "p5 setup_inactivity failed: %s", esp_err_to_name(ret));
        goto out_sem;
    }
    ret = node_acc_adxl367_set_act_inact_linkloop(&acc, (uint8_t)NODE_ACC_ADXL367_TEST_P5_LINKLOOP_MODE);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "p5 linkloop failed: %s", esp_err_to_name(ret));
        goto out_sem;
    }
    /* AXIS_MASK: block Y/Z so ACT/INACT use X only (same mask applies to both). */
    ret = node_acc_adxl367_set_activity_axes_enabled(&acc, true, false, false);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "p5 set_activity_axes failed: %s", esp_err_to_name(ret));
        goto out_sem;
    }
    vTaskDelay(pdMS_TO_TICKS(200));

    node_acc_adxl367_int_map_t imap = {0};
    imap.act = 1u;
    imap.inact = 1u;
    imap.int_low = 1u;
    ret = node_acc_adxl367_int_map(&acc, 2u, &imap);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "p5 int_map failed: %s", esp_err_to_name(ret));
        goto out_sem;
    }

    ret = gpio_install_isr_service(0);
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "p5 gpio_install_isr_service failed: %s", esp_err_to_name(ret));
        goto out_sem;
    }

    gpio_config_t io = {
        .pin_bit_mask = 1ull << NODE_ACC_ADXL367_TEST_INT2_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_NEGEDGE,
    };
    ret = gpio_config(&io);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "p5 gpio_config failed: %s", esp_err_to_name(ret));
        goto out_sem;
    }

    ret = gpio_isr_handler_add((gpio_num_t)NODE_ACC_ADXL367_TEST_INT2_GPIO, p5_act_int_isr, sem);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "p5 gpio_isr_handler_add failed: %s", esp_err_to_name(ret));
        goto out_sem;
    }
    handler_added = true;
    /* INT2 off until arm delay. */

    ESP_LOGI(TAG, "p5 linked ACT/INACT, window %d s", (int)(NODE_ACC_ADXL367_TEST_P5_WINDOW_MS / 1000));
    ESP_LOGI(TAG, "THRESH_ACT ~%.2f g (%u LSB), TIME_ACT=%u; THRESH_INACT ~%.2f g (%u LSB), TIME_INACT=%u",
             (double)NODE_ACC_ADXL367_TEST_P5_THRESH_G,
             (unsigned)NODE_ACC_ADXL367_THRESH_ACT_LSB_FROM_G(NODE_ACC_ADXL367_TEST_P5_THRESH_G),
             (unsigned)NODE_ACC_ADXL367_TEST_P5_TIME_ACT_SAMPLES, (double)NODE_ACC_ADXL367_TEST_P5_THRESH_INACT_G,
             (unsigned)NODE_ACC_ADXL367_THRESH_ACT_LSB_FROM_G(NODE_ACC_ADXL367_TEST_P5_THRESH_INACT_G),
             (unsigned)NODE_ACC_ADXL367_TEST_P5_TIME_INACT_SAMPLES);
    ESP_LOGI(TAG, "LINKLOOP=%u, axes: X only (Y/Z masked)", (unsigned)NODE_ACC_ADXL367_TEST_P5_LINKLOOP_MODE);
    ESP_LOGI(TAG, "Waiting %d ms with INT2 disabled (reference settle; avoids false start)",
             (int)NODE_ACC_ADXL367_TEST_P5_ARM_MS);

    vTaskDelay(pdMS_TO_TICKS(NODE_ACC_ADXL367_TEST_P5_ARM_MS));

    {
        uint8_t st_clr = 0;
        for (int i = 0; i < 3; i++) {
            (void)node_acc_adxl367_read_status(&acc, &st_clr);
        }
        while (xSemaphoreTake(sem, 0) == pdTRUE) {
        }

        gpio_intr_enable((gpio_num_t)NODE_ACC_ADXL367_TEST_INT2_GPIO);
        vTaskDelay(pdMS_TO_TICKS(50));
        while (xSemaphoreTake(sem, 0) == pdTRUE) {
        }

        ESP_LOGI(TAG, "INT2 on; poll STATUS");

        const TickType_t t_end = xTaskGetTickCount() + pdMS_TO_TICKS(NODE_ACC_ADXL367_TEST_P5_WINDOW_MS);

        ret = ESP_OK;
        while (xTaskGetTickCount() < t_end) {
            while (xSemaphoreTake(sem, 0) == pdTRUE) {
                uint8_t st = 0;
                (void)node_acc_adxl367_read_status(&acc, &st);
                if ((st & NODE_ACC_ADXL367_STATUS_ACT) != 0) {
                    ESP_LOGI(TAG, "[activity] motion (STATUS 0x%02X)", (unsigned)st);
                } else if ((st & NODE_ACC_ADXL367_STATUS_INACT) != 0) {
                    ESP_LOGI(TAG, "[inactive] still / below inact threshold (STATUS 0x%02X)", (unsigned)st);
                } else {
                    ESP_LOGD(TAG, "INT2 edge, STATUS 0x%02X (no ACT/INACT latched)", (unsigned)st);
                }
            }

            vTaskDelay(pdMS_TO_TICKS(NODE_ACC_ADXL367_TEST_P5_POLL_MS));
        }

        ESP_LOGI(TAG, "======== Phase 5 end ========");
    }

    if (handler_added) {
        gpio_intr_disable((gpio_num_t)NODE_ACC_ADXL367_TEST_INT2_GPIO);
        gpio_isr_handler_remove((gpio_num_t)NODE_ACC_ADXL367_TEST_INT2_GPIO);
    }
    gpio_reset_pin((gpio_num_t)NODE_ACC_ADXL367_TEST_INT2_GPIO);

    (void)node_acc_adxl367_act_inact_disable(&acc);
    node_acc_adxl367_int_map_t zmap = {0};
    (void)node_acc_adxl367_int_map(&acc, 2u, &zmap);

out_sem:
    vSemaphoreDelete(sem);

out:
    if (i2c_dev != NULL) {
        (void)i2c_remove_device(i2c_dev);
    }
    return ret;
}

void node_acc_adxl367_run_transition_monitor_forever(void)
{
    i2c_master_dev_handle_t i2c_dev = NULL;
    node_acc_adxl367_dev_t acc = {0};

    esp_err_t ret = probe_i2c_and_init(&acc, &i2c_dev);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "transition monitor: init failed");
        while (1) {
            vTaskDelay(pdMS_TO_TICKS(2000));
        }
    }

    ret = node_acc_adxl367_set_measurement_mode(&acc, NODE_ACC_ADXL367_MEAS_ACCEL_ONLY);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "transition monitor: accel-only failed: %s", esp_err_to_name(ret));
        goto hang;
    }
    (void)node_acc_adxl367_fifo_disable(&acc);
    (void)node_acc_adxl367_act_inact_disable(&acc);

    vTaskDelay(pdMS_TO_TICKS(200));

    ESP_LOGI(TAG, "transition monitor: max sample delta, high>%d low<%d still_cnt=%u",
             (int)NODE_ACC_ADXL367_MONITOR_DELTA_HIGH_LSB, (int)NODE_ACC_ADXL367_MONITOR_DELTA_LOW_LSB,
             (unsigned)NODE_ACC_ADXL367_MONITOR_STILL_COUNT);

    bool have_prev = false;
    int16_t px = 0, py = 0, pz = 0;
    bool moving = false;
    uint32_t still_cnt = 0;

    for (;;) {
        int16_t x = 0, y = 0, z = 0;
        ret = node_acc_adxl367_read_xyz_raw(&acc, &x, &y, &z);
        if (ret != ESP_OK) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        if (x > NODE_ACC_ADXL367_MONITOR_ABS_MAX || x < -NODE_ACC_ADXL367_MONITOR_ABS_MAX ||
            y > NODE_ACC_ADXL367_MONITOR_ABS_MAX || y < -NODE_ACC_ADXL367_MONITOR_ABS_MAX ||
            z > NODE_ACC_ADXL367_MONITOR_ABS_MAX || z < -NODE_ACC_ADXL367_MONITOR_ABS_MAX) {
            have_prev = false;
            vTaskDelay(pdMS_TO_TICKS(5));
            continue;
        }

        if (!have_prev) {
            px = x;
            py = y;
            pz = z;
            have_prev = true;
            continue;
        }

        const uint32_t dx = adxl367_uabs_diff(x, px);
        const uint32_t dy = adxl367_uabs_diff(y, py);
        const uint32_t dz = adxl367_uabs_diff(z, pz);
        const uint32_t dmax = adxl367_max_u32(dx, dy, dz);

        px = x;
        py = y;
        pz = z;

        if (dmax > (uint32_t)NODE_ACC_ADXL367_MONITOR_DELTA_HIGH_LSB) {
            still_cnt = 0;
            if (!moving) {
                ESP_LOGI(TAG, "[still_to_motion] max_delta=%u (~%.4f g)", (unsigned)dmax, (double)dmax / 4000.0);
                moving = true;
            }
        } else if (dmax < (uint32_t)NODE_ACC_ADXL367_MONITOR_DELTA_LOW_LSB) {
            still_cnt++;
            if (moving && still_cnt >= NODE_ACC_ADXL367_MONITOR_STILL_COUNT) {
                ESP_LOGI(TAG, "[motion_to_still] max_delta=%u (~%.4f g)", (unsigned)dmax, (double)dmax / 4000.0);
                moving = false;
                still_cnt = 0;
            }
        } else {
            still_cnt = 0;
        }
    }

hang:
    if (i2c_dev != NULL) {
        (void)i2c_remove_device(i2c_dev);
    }
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}

void node_acc_adxl367_run_phase6_default_app(void)
{
    ESP_LOGI(TAG, "phase6 default app");
    node_acc_adxl367_run_transition_monitor_forever();
}
