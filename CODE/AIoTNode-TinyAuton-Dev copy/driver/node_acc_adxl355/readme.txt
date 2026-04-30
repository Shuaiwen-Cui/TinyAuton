ADXL355 (node_acc_adxl355)
===========================

Contents
--------
  include/node_acc_adxl355.h       Public driver API
  include/node_acc_adxl355_test.h  Test entry declarations
  node_acc_adxl355.c               Implementation
  node_acc_adxl355_test.c          Phase 1–5 tests
  CMakeLists.txt                   ESP-IDF component

Architecture
------------
  - SPI only. Bus must be initialized before node_acc_adxl355_init (e.g. spi3_init() from
    driver/node_spi when using default eval pins: SCK/MOSI/MISO = GPIO15/16/17).
  - SPI framing matches Analog no-OS driver/ref-adxl355: one write_and_read stream of
    (1 + N) bytes for reads; discard the first received byte (command byte phase on MISO).
  - Register map and scaling follow the datasheet / ref-adxl355.

Measurement modes (POWER_CTL TEMP_OFF; see ref adxl355_op_mode)
---------------------------------------------------------------
  - ACCEL_AND_TEMP: temperature path on (TEMP_OFF clear). Read XYZ and temperature.
  - ACCEL_ONLY: TEMP_OFF set (Analog MEAS_TEMP_OFF_*). Temperature read returns ESP_ERR_INVALID_STATE.
  - TEMP_ONLY: same HW as ACCEL_AND_TEMP (ADXL355 has no “accel off, temp on” mode); use only
    read_temp; read_raw_xyz returns ESP_ERR_INVALID_STATE.

  Config: node_acc_adxl355_config_t.meas_mode; runtime: set_measurement_mode / get_measurement_mode.
  Test: node_acc_adxl355_run_measurement_mode_test().

Default eval wiring (see project docs)
--------------------------------------
  CS   GPIO7
  MOSI GPIO16   MISO GPIO17   SCK GPIO15
  INT1 GPIO4    INT2 GPIO5    DRDY GPIO6   (defaults in node_acc_adxl355_config_default_eval)

Development plan (phases)
-------------------------
  Phase 0 — Bring-up constraints
    - Config struct for host, CS, clock; optional INT/DRDY GPIO numbers (GPIO_NUM_NC if unused).
    - SPI mode 0; clock within datasheet limit (e.g. <= 10 MHz).
    - Power, ground, and INT polarity (RANGE register) noted for hardware.

  Phase 1 — MVP (current baseline)
    - Soft reset, ID check (0xAD / 0x1D / 0xED), range, ODR, measurement, temperature.
    - Read STATUS, raw XYZ, g, temperature; POWER_CTL for DRDY output enable (pin toggles).
    - Polling only; no GPIO ISR for INT1/INT2/DRDY.
    - Test: node_acc_adxl355_run_phase1_test().

  Phase 2 — DRDY GPIO (implemented)
    - Config: gpio_drdy (eval GPIO6), drdy_intr_type (default GPIO_INTR_POSEDGE).
    - APIs: node_acc_adxl355_drdy_isr_install / node_acc_adxl355_drdy_wait / node_acc_adxl355_drdy_isr_remove
      (deinit calls remove). ISR only gives a binary semaphore; SPI runs in task.
    - Test: node_acc_adxl355_run_phase2_test(). Tuning: NODE_ACC_ADXL355_TEST_PHASE2_* in test .c.
    - If edges do not match RANGE INT_POL / board, change drdy_intr_type or set_int_pol (future helper).

  Phase 3 — INT1 / INT2 + INT_MAP (implemented)
    - Config: gpio_int1 / gpio_int2 (eval 4 / 5), int1_intr_type / int2_intr_type (default NEGEDGE for active-low).
    - APIs: int_map_write / int_map_read, set_interrupt_polarity (RANGE[6]),
      int_isr_install / int_isr_remove, int1_wait / int2_wait. deinit removes INT ISR before DRDY ISR.
    - Test: node_acc_adxl355_run_phase3_test() maps DATA_RDY -> INT1 only, active-low + negedge, then STATUS + XYZ.
      Order: int_isr_install before int_map_write so the first pulse after mapping is not missed; INT GPIO uses MCU pull-up.
    - Clear INT_MAP to 0 at end of test; adjust polarity/edges if board differs (macros NODE_ACC_ADXL355_PHASE3_INT_POL / NODE_ACC_ADXL355_PHASE3_INT_INTR in test .c).
    - If int1_wait returns ESP_ERR_TIMEOUT: INT1 wiring / INT_POL / edge type mismatch — try
      set_interrupt_polarity(ACTIVE_HIGH) + int1_intr_type POSEDGE, or ANYEDGE; confirm GPIO4.
    - gpio_install_isr_service "already installed" was from duplicate calls; driver now installs once per install path.

  Phase 4 — FIFO, activity, calibration (implemented)
    - FIFO: node_acc_adxl355_set_fifo_samples / get_fifo_samples, read_fifo_entries, read_fifo_xyz
      (burst FIFO_DATA; Analog-style X-header check + unconditional first-frame fallback).
    - Offset: node_acc_adxl355_set_offset_xyz / get_offset_xyz (0x1E–0x23, BE 16-bit per axis).
    - Activity: set/get activity_enable, set/get_activity_threshold, set/get_activity_count.
    - Test: node_acc_adxl355_run_phase4_test(); tune NODE_ACC_ADXL355_TEST_PHASE4_DRDY_WAITS if FIFO stays empty.

  Phase 5 — Hardening (implemented)
    - SPI serialized with an internal recursive mutex when config spi_mutex_enable is true (default).
    - reg_write / regs_write / spi_read_after_cmd take the mutex; safe nested calls (e.g. FIFO + registers).
    - deinit: remove INT/DRDY ISR, take mutex, INT_MAP=0, disable DRDY pin, standby, remove SPI device, delete mutex, memset dev.
    - init failure rolls back SPI device and mutex; log_info_on_init (default false) gates the single “init ok” INFO line.
    - Test: node_acc_adxl355_run_phase5_test() enables log_info_on_init, one XYZ read, deinit.

Milestones (shortcut)
---------------------
  M1: Phase 1 stable on hardware.
  M2: Phase 2 DRDY-driven sampling.
  M3: Phase 3 interrupt map + INT pins.
  M4: Phase 4 FIFO / activity / offset (as needed).
  M5: Phase 5 mutex + clean deinit.

Integration
-----------
  main may PRIV_REQUIRES node_acc_adxl355 (and nvs_flash if used). SPI bus init order:
  spi3_init() (or equivalent) before node_acc_adxl355_init.
