/**
 * @file node_acc_adxl367_test.h
 * @brief Phases 1-5 tests; Phase 6: run_phase6_default_app. Driver: node_acc_adxl367.h.
 */
#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t node_acc_adxl367_run_phase1_test(void);
esp_err_t node_acc_adxl367_run_phase2_test(void);
esp_err_t node_acc_adxl367_run_phase3_test(void);
esp_err_t node_acc_adxl367_run_phase4_test(void);
esp_err_t node_acc_adxl367_run_phase5_test(void);

/** Phase 6 default: does not return. Calls transition monitor (UART motion/still lines). */
void node_acc_adxl367_run_phase6_default_app(void);

esp_err_t node_acc_adxl367_run_xyz_stats_capture(void);

/** Software motion/still from max sample delta; does not return. */
void node_acc_adxl367_run_transition_monitor_forever(void);

#ifdef __cplusplus
}
#endif
