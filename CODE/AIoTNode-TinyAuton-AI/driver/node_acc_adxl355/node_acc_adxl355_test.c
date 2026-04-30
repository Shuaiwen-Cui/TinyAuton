/**
 * @file node_acc_adxl355_test.c
 * @brief Phase 1 polling; Phase 2 DRDY; Phase 3 INT_MAP; Phase 4 FIFO/activity/offset.
 */
#include "node_acc_adxl355_test.h"

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "node_acc_adxl355.h"
#include "node_spi.h"

static const char *TAG = "adxl355_test";

#ifndef NODE_ACC_ADXL355_TEST_SAMPLES
#define NODE_ACC_ADXL355_TEST_SAMPLES 8
#endif

#ifndef NODE_ACC_ADXL355_TEST_SAMPLE_DELAY_MS
#define NODE_ACC_ADXL355_TEST_SAMPLE_DELAY_MS 100
#endif

#ifndef NODE_ACC_ADXL355_TEST_PHASE2_SAMPLES
#define NODE_ACC_ADXL355_TEST_PHASE2_SAMPLES 16
#endif

/** Per-sample wait; ODR 125 Hz => ~8 ms period, allow margin. */
#ifndef NODE_ACC_ADXL355_TEST_PHASE2_WAIT_MS
#define NODE_ACC_ADXL355_TEST_PHASE2_WAIT_MS 50
#endif

#ifndef NODE_ACC_ADXL355_TEST_PHASE3_SAMPLES
#define NODE_ACC_ADXL355_TEST_PHASE3_SAMPLES 12
#endif

#ifndef NODE_ACC_ADXL355_TEST_PHASE3_WAIT_MS
#define NODE_ACC_ADXL355_TEST_PHASE3_WAIT_MS 100
#endif

/** Override before init if board uses active-high INT (e.g. POSEDGE + ACTIVE_HIGH). */
#ifndef NODE_ACC_ADXL355_PHASE3_INT_POL
#define NODE_ACC_ADXL355_PHASE3_INT_POL NODE_ACC_ADXL355_INT_POL_ACTIVE_LOW
#endif
#ifndef NODE_ACC_ADXL355_PHASE3_INT_INTR
#define NODE_ACC_ADXL355_PHASE3_INT_INTR GPIO_INTR_NEGEDGE
#endif

#ifndef NODE_ACC_ADXL355_TEST_PHASE4_DRDY_WAITS
#define NODE_ACC_ADXL355_TEST_PHASE4_DRDY_WAITS 24
#endif

esp_err_t node_acc_adxl355_run_phase1_test(void)
{
    spi3_init();

    node_acc_adxl355_config_t cfg;
    node_acc_adxl355_config_default_eval(&cfg);

    node_acc_adxl355_dev_t dev = {0};
    esp_err_t ret =
        node_acc_adxl355_init(&dev, &cfg, NODE_ACC_ADXL355_RANGE_2G, NODE_ACC_ADXL355_ODR_125_HZ);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    node_acc_adxl355_ids_t ids = {0};
    ret = node_acc_adxl355_read_ids(&dev, &ids);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "IDs: AD=0x%02X MST=0x%02X PART=0x%02X REV=0x%02X", ids.devid_ad, ids.devid_mst, ids.partid,
                 ids.revid);
    }

    for (int i = 0; i < NODE_ACC_ADXL355_TEST_SAMPLES; i++) {
        node_acc_adxl355_raw_xyz_t raw = {0};
        node_acc_adxl355_g_t g = {0};
        float tc = 0.0f;
        uint8_t st = 0;

        ret = node_acc_adxl355_read_status(&dev, &st);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "status: %s", esp_err_to_name(ret));
            break;
        }
        ret = node_acc_adxl355_read_raw_xyz(&dev, &raw);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "raw xyz: %s", esp_err_to_name(ret));
            break;
        }
        ret = node_acc_adxl355_read_g(&dev, &g);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "g: %s", esp_err_to_name(ret));
            break;
        }
        ret = node_acc_adxl355_read_temp_c(&dev, &tc);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "temp: %s", esp_err_to_name(ret));
            break;
        }

        ESP_LOGI(TAG, "sample %d: ST=0x%02X raw xyz=%ld %ld %ld  g=(%.5f %.5f %.5f)  T=%.2f C", i,
                 (unsigned)st, (long)raw.x, (long)raw.y, (long)raw.z, (double)g.x, (double)g.y, (double)g.z,
                 (double)tc);

        vTaskDelay(pdMS_TO_TICKS(NODE_ACC_ADXL355_TEST_SAMPLE_DELAY_MS));
    }

    esp_err_t de = node_acc_adxl355_deinit(&dev);
    if (de != ESP_OK) {
        ESP_LOGW(TAG, "deinit: %s", esp_err_to_name(de));
    }
    if (ret != ESP_OK) {
        return ret;
    }
    return de;
}

esp_err_t node_acc_adxl355_run_phase2_test(void)
{
    spi3_init();

    node_acc_adxl355_config_t cfg;
    node_acc_adxl355_config_default_eval(&cfg);

    node_acc_adxl355_dev_t dev = {0};
    esp_err_t ret =
        node_acc_adxl355_init(&dev, &cfg, NODE_ACC_ADXL355_RANGE_2G, NODE_ACC_ADXL355_ODR_125_HZ);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "phase2 init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = node_acc_adxl355_drdy_isr_install(&dev);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "phase2 drdy_isr_install failed: %s", esp_err_to_name(ret));
        (void)node_acc_adxl355_deinit(&dev);
        return ret;
    }

    if (dev.drdy_sem != NULL) {
        while (xSemaphoreTake(dev.drdy_sem, 0) == pdTRUE) {
        }
    }

    ESP_LOGI(TAG, "phase2: DRDY GPIO%u %s, %d samples, wait %u ms", (unsigned)dev.gpio_drdy,
             (dev.drdy_intr_type == GPIO_INTR_POSEDGE) ? "posedge" : "edge", NODE_ACC_ADXL355_TEST_PHASE2_SAMPLES,
             (unsigned)NODE_ACC_ADXL355_TEST_PHASE2_WAIT_MS);

    for (int i = 0; i < NODE_ACC_ADXL355_TEST_PHASE2_SAMPLES; i++) {
        ret = node_acc_adxl355_drdy_wait(&dev, NODE_ACC_ADXL355_TEST_PHASE2_WAIT_MS);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "phase2 wait sample %d: %s", i, esp_err_to_name(ret));
            break;
        }

        node_acc_adxl355_raw_xyz_t raw = {0};
        node_acc_adxl355_g_t g = {0};
        ret = node_acc_adxl355_read_raw_xyz(&dev, &raw);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "phase2 raw xyz: %s", esp_err_to_name(ret));
            break;
        }
        ret = node_acc_adxl355_read_g(&dev, &g);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "phase2 g: %s", esp_err_to_name(ret));
            break;
        }
        ESP_LOGI(TAG, "phase2 sample %d: raw %ld %ld %ld  g=(%.5f %.5f %.5f)", i, (long)raw.x, (long)raw.y,
                 (long)raw.z, (double)g.x, (double)g.y, (double)g.z);
    }

    esp_err_t de = node_acc_adxl355_deinit(&dev);
    if (de != ESP_OK) {
        ESP_LOGW(TAG, "phase2 deinit: %s", esp_err_to_name(de));
    }
    if (ret != ESP_OK) {
        return ret;
    }
    return de;
}

esp_err_t node_acc_adxl355_run_phase3_test(void)
{
    spi3_init();

    node_acc_adxl355_config_t cfg;
    node_acc_adxl355_config_default_eval(&cfg);
    cfg.int1_intr_type = NODE_ACC_ADXL355_PHASE3_INT_INTR;

    node_acc_adxl355_dev_t dev = {0};
    esp_err_t ret =
        node_acc_adxl355_init(&dev, &cfg, NODE_ACC_ADXL355_RANGE_2G, NODE_ACC_ADXL355_ODR_125_HZ);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "phase3 init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = node_acc_adxl355_set_interrupt_polarity(&dev, NODE_ACC_ADXL355_PHASE3_INT_POL);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "phase3 set_interrupt_polarity: %s", esp_err_to_name(ret));
        (void)node_acc_adxl355_deinit(&dev);
        return ret;
    }

    /* Arm GPIO ISR before INT_MAP so the first DATA_RDY pulse after mapping is not missed. */
    ret = node_acc_adxl355_int_isr_install(&dev);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "phase3 int_isr_install: %s", esp_err_to_name(ret));
        (void)node_acc_adxl355_deinit(&dev);
        return ret;
    }

    node_acc_adxl355_int_map_t imap = {0};
    imap.rdy_int1 = true;
    ret = node_acc_adxl355_int_map_write(&dev, &imap);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "phase3 int_map_write: %s", esp_err_to_name(ret));
        (void)node_acc_adxl355_deinit(&dev);
        return ret;
    }

    if (dev.int1_sem != NULL) {
        while (xSemaphoreTake(dev.int1_sem, 0) == pdTRUE) {
        }
    }

    /* Clear any pending DATA_RDY so the first int1_wait aligns to the next edge. */
    {
        const TickType_t t_end = xTaskGetTickCount() + pdMS_TO_TICKS(200);
        while (xTaskGetTickCount() < t_end) {
            uint8_t st = 0;
            ret = node_acc_adxl355_read_status(&dev, &st);
            if (ret != ESP_OK) {
                break;
            }
            if (st & NODE_ACC_ADXL355_REG_STATUS_DATA_RDY) {
                node_acc_adxl355_raw_xyz_t raw = {0};
                (void)node_acc_adxl355_read_raw_xyz(&dev, &raw);
                break;
            }
            vTaskDelay(pdMS_TO_TICKS(1));
        }
        while (dev.int1_sem != NULL && xSemaphoreTake(dev.int1_sem, 0) == pdTRUE) {
        }
    }

    ESP_LOGI(TAG, "phase3: INT1 GPIO%u (DATA_RDY), %d samples", (unsigned)dev.gpio_int1,
             NODE_ACC_ADXL355_TEST_PHASE3_SAMPLES);

    for (int i = 0; i < NODE_ACC_ADXL355_TEST_PHASE3_SAMPLES; i++) {
        ret = node_acc_adxl355_int1_wait(&dev, NODE_ACC_ADXL355_TEST_PHASE3_WAIT_MS);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "phase3 int1_wait %d: %s", i, esp_err_to_name(ret));
            break;
        }

        uint8_t st = 0;
        ret = node_acc_adxl355_read_status(&dev, &st);
        if (ret != ESP_OK) {
            break;
        }
        node_acc_adxl355_raw_xyz_t raw = {0};
        ret = node_acc_adxl355_read_raw_xyz(&dev, &raw);
        if (ret != ESP_OK) {
            break;
        }
        ESP_LOGI(TAG, "phase3 sample %d: STATUS=0x%02X (DATA_RDY=%u) raw %ld %ld %ld", i, (unsigned)st,
                 (unsigned)(st & NODE_ACC_ADXL355_REG_STATUS_DATA_RDY), (long)raw.x, (long)raw.y, (long)raw.z);
    }

    node_acc_adxl355_int_map_t zmap = {0};
    (void)node_acc_adxl355_int_map_write(&dev, &zmap);

    esp_err_t de = node_acc_adxl355_deinit(&dev);
    if (de != ESP_OK) {
        ESP_LOGW(TAG, "phase3 deinit: %s", esp_err_to_name(de));
    }
    if (ret != ESP_OK) {
        return ret;
    }
    return de;
}

esp_err_t node_acc_adxl355_run_phase4_test(void)
{
    spi3_init();

    node_acc_adxl355_config_t cfg;
    node_acc_adxl355_config_default_eval(&cfg);

    node_acc_adxl355_dev_t dev = {0};
    esp_err_t ret =
        node_acc_adxl355_init(&dev, &cfg, NODE_ACC_ADXL355_RANGE_2G, NODE_ACC_ADXL355_ODR_125_HZ);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "phase4 init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    node_acc_adxl355_offset_t off0 = {0};
    ret = node_acc_adxl355_get_offset_xyz(&dev, &off0);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "phase4 get_offset: %s", esp_err_to_name(ret));
        (void)node_acc_adxl355_deinit(&dev);
        return ret;
    }
    ESP_LOGI(TAG, "phase4: offset readback X=0x%04X Y=0x%04X Z=0x%04X", (unsigned)off0.x, (unsigned)off0.y,
             (unsigned)off0.z);

    ret = node_acc_adxl355_set_fifo_samples(&dev, 32);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "phase4 set_fifo_samples: %s", esp_err_to_name(ret));
        (void)node_acc_adxl355_deinit(&dev);
        return ret;
    }
    uint8_t fs = 0;
    (void)node_acc_adxl355_get_fifo_samples(&dev, &fs);
    ESP_LOGI(TAG, "phase4: FIFO_SAMPLES reg=%u", (unsigned)fs);

    ret = node_acc_adxl355_drdy_isr_install(&dev);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "phase4 drdy_isr_install: %s", esp_err_to_name(ret));
        (void)node_acc_adxl355_deinit(&dev);
        return ret;
    }

    for (int w = 0; w < NODE_ACC_ADXL355_TEST_PHASE4_DRDY_WAITS; w++) {
        ret = node_acc_adxl355_drdy_wait(&dev, 100);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "phase4 drdy_wait %d: %s", w, esp_err_to_name(ret));
            break;
        }
    }

    uint8_t ent = 0;
    ret = node_acc_adxl355_read_fifo_entries(&dev, &ent);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "phase4 fifo_entries: %s", esp_err_to_name(ret));
        (void)node_acc_adxl355_deinit(&dev);
        return ret;
    }
    ESP_LOGI(TAG, "phase4: FIFO_ENTRIES=%u (after DRDY waits)", (unsigned)ent);

    node_acc_adxl355_raw_xyz_t fifo_xyz[8] = {0};
    size_t nfrm = 0;
    ret = node_acc_adxl355_read_fifo_xyz(&dev, fifo_xyz, 8, &nfrm);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "phase4 read_fifo_xyz: %s", esp_err_to_name(ret));
    } else if (nfrm > 0) {
        ESP_LOGI(TAG, "phase4: FIFO frames=%u  first raw %ld %ld %ld", (unsigned)nfrm, (long)fifo_xyz[0].x,
                 (long)fifo_xyz[0].y, (long)fifo_xyz[0].z);
    } else {
        ESP_LOGW(TAG, "phase4: FIFO parse got 0 frames (entries may still be valid for chip)");
    }

    node_acc_adxl355_act_en_t act = {.x = false, .y = false, .z = false};
    ret = node_acc_adxl355_set_activity_enable(&dev, &act);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "phase4 set_activity_enable: %s", esp_err_to_name(ret));
        (void)node_acc_adxl355_deinit(&dev);
        return ret;
    }
    ret = node_acc_adxl355_set_activity_threshold(&dev, 0x0400);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "phase4 set_activity_threshold: %s", esp_err_to_name(ret));
        (void)node_acc_adxl355_deinit(&dev);
        return ret;
    }
    ret = node_acc_adxl355_set_activity_count(&dev, 0x01);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "phase4 set_activity_count: %s", esp_err_to_name(ret));
        (void)node_acc_adxl355_deinit(&dev);
        return ret;
    }
    node_acc_adxl355_act_en_t ar = {0};
    uint16_t thr = 0;
    uint8_t ac = 0;
    (void)node_acc_adxl355_get_activity_enable(&dev, &ar);
    (void)node_acc_adxl355_get_activity_threshold(&dev, &thr);
    (void)node_acc_adxl355_get_activity_count(&dev, &ac);
    ESP_LOGI(TAG, "phase4: ACT readback en=%d%d%d thresh=0x%04X count=%u", ar.x ? 1 : 0, ar.y ? 1 : 0,
             ar.z ? 1 : 0, (unsigned)thr, (unsigned)ac);

    esp_err_t de = node_acc_adxl355_deinit(&dev);
    if (de != ESP_OK) {
        ESP_LOGW(TAG, "phase4 deinit: %s", esp_err_to_name(de));
    }
    if (ret != ESP_OK) {
        return ret;
    }
    return de;
}

esp_err_t node_acc_adxl355_run_phase5_test(void)
{
    spi3_init();

    node_acc_adxl355_config_t cfg;
    node_acc_adxl355_config_default_eval(&cfg);
    cfg.log_info_on_init = true;

    node_acc_adxl355_dev_t dev = {0};
    esp_err_t ret =
        node_acc_adxl355_init(&dev, &cfg, NODE_ACC_ADXL355_RANGE_2G, NODE_ACC_ADXL355_ODR_125_HZ);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "phase5 init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    node_acc_adxl355_raw_xyz_t raw = {0};
    ret = node_acc_adxl355_read_raw_xyz(&dev, &raw);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "phase5 read_raw_xyz: %s", esp_err_to_name(ret));
        (void)node_acc_adxl355_deinit(&dev);
        return ret;
    }
    ESP_LOGI(TAG, "phase5: sample raw %ld %ld %ld (mutex + SPI ok)", (long)raw.x, (long)raw.y, (long)raw.z);

    ret = node_acc_adxl355_deinit(&dev);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "phase5 deinit: %s", esp_err_to_name(ret));
        return ret;
    }
    ESP_LOGI(TAG, "phase5: deinit ok (device zeroed)");
    return ESP_OK;
}

esp_err_t node_acc_adxl355_run_measurement_mode_test(void)
{
    spi3_init();

    node_acc_adxl355_config_t cfg;
    node_acc_adxl355_config_default_eval(&cfg);

    node_acc_adxl355_dev_t dev = {0};
    esp_err_t ret =
        node_acc_adxl355_init(&dev, &cfg, NODE_ACC_ADXL355_RANGE_2G, NODE_ACC_ADXL355_ODR_125_HZ);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "meas_mode: init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    vTaskDelay(pdMS_TO_TICKS(50));

    ret = node_acc_adxl355_set_measurement_mode(&dev, NODE_ACC_ADXL355_MEAS_ACCEL_AND_TEMP);
    if (ret != ESP_OK) {
        goto out_deinit;
    }
    {
        node_acc_adxl355_raw_xyz_t raw = {0};
        float tc = 0.0f;
        ret = node_acc_adxl355_read_raw_xyz(&dev, &raw);
        if (ret != ESP_OK) {
            goto out_deinit;
        }
        ret = node_acc_adxl355_read_temp_c(&dev, &tc);
        if (ret != ESP_OK) {
            goto out_deinit;
        }
        ESP_LOGI(TAG, "ACCEL_AND_TEMP: raw z=%ld  T=%.2f C", (long)raw.z, (double)tc);
    }

    vTaskDelay(pdMS_TO_TICKS(50));
    ret = node_acc_adxl355_set_measurement_mode(&dev, NODE_ACC_ADXL355_MEAS_ACCEL_ONLY);
    if (ret != ESP_OK) {
        goto out_deinit;
    }
    vTaskDelay(pdMS_TO_TICKS(50));
    {
        node_acc_adxl355_raw_xyz_t raw = {0};
        float tc = 0.0f;
        ret = node_acc_adxl355_read_raw_xyz(&dev, &raw);
        if (ret != ESP_OK) {
            goto out_deinit;
        }
        ret = node_acc_adxl355_read_temp_c(&dev, &tc);
        if (ret != ESP_ERR_INVALID_STATE) {
            ESP_LOGE(TAG, "ACCEL_ONLY: expected temp INVALID_STATE, got %s", esp_err_to_name(ret));
            ret = ESP_FAIL;
            goto out_deinit;
        }
        ESP_LOGI(TAG, "ACCEL_ONLY: raw z=%ld  temp read rejected OK", (long)raw.z);
    }

    vTaskDelay(pdMS_TO_TICKS(50));
    ret = node_acc_adxl355_set_measurement_mode(&dev, NODE_ACC_ADXL355_MEAS_TEMP_ONLY);
    if (ret != ESP_OK) {
        goto out_deinit;
    }
    vTaskDelay(pdMS_TO_TICKS(50));
    {
        float tc = 0.0f;
        ret = node_acc_adxl355_read_temp_c(&dev, &tc);
        if (ret != ESP_OK) {
            goto out_deinit;
        }
        node_acc_adxl355_raw_xyz_t raw = {0};
        ret = node_acc_adxl355_read_raw_xyz(&dev, &raw);
        if (ret != ESP_ERR_INVALID_STATE) {
            ESP_LOGE(TAG, "TEMP_ONLY: expected xyz INVALID_STATE, got %s", esp_err_to_name(ret));
            ret = ESP_FAIL;
            goto out_deinit;
        }
        ESP_LOGI(TAG, "TEMP_ONLY: T=%.2f C  xyz read rejected OK", (double)tc);
    }

    ret = ESP_OK;

out_deinit : {
    esp_err_t de = node_acc_adxl355_deinit(&dev);
    if (de != ESP_OK) {
        ESP_LOGW(TAG, "meas_mode deinit: %s", esp_err_to_name(de));
    }
    if (ret != ESP_OK) {
        return ret;
    }
    return de;
}
}
