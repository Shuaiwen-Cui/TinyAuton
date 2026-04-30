/**
 * @file node_acc_adxl355_test.h
 * @brief Phase 1–4 tests (polling, DRDY, INT_MAP, FIFO/activity/offset). Driver: node_acc_adxl355.h.
 */
#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Phase 1: @c spi3_init() from node_spi, then init ADXL355 on CS GPIO7.
 * Reads chip IDs, prints several XYZ + temperature samples, then deinitializes.
 */
esp_err_t node_acc_adxl355_run_phase1_test(void);

/**
 * Phase 2: DRDY on GPIO6 (default eval), @ref node_acc_adxl355_drdy_isr_install + wait + read XYZ.
 */
esp_err_t node_acc_adxl355_run_phase2_test(void);

/**
 * Phase 3: INT_MAP (DATA_RDY -> INT1), GPIO4, @ref node_acc_adxl355_int_isr_install + int1_wait + STATUS/XYZ.
 */
esp_err_t node_acc_adxl355_run_phase3_test(void);

/** ACCEL+TEMP / ACCEL-only / TEMP-only (software) paths; expects INVALID_STATE where documented. */
esp_err_t node_acc_adxl355_run_measurement_mode_test(void);

/** Phase 4: FIFO_SAMPLES / FIFO read, offset regs, activity regs (readback). */
esp_err_t node_acc_adxl355_run_phase4_test(void);

/** Phase 5: mutex + deinit path (init with optional log, deinit clears device). */
esp_err_t node_acc_adxl355_run_phase5_test(void);

#ifdef __cplusplus
}
#endif
