ADXL367 (node_acc_adxl367)
==========================

Contents
--------
  include/node_acc_adxl367.h      Public driver API
  include/node_acc_adxl367_test.h Test / app entry declarations
  node_acc_adxl367.c              Implementation
  node_acc_adxl367_test.c         Phase 1-5 tests, stats, Phase 6 default app
  CMakeLists.txt                  ESP-IDF component

Architecture
------------
  - I2C only. All transfers go through driver/node_i2c (not ESP-IDF I2C directly).
  - Register map, reset, 14-bit packing, and behavior align with driver/ref-adxl367/adxl367.c (Analog reference).
  - node_acc_adxl367_dev_t holds the i2c handle plus cached range/ODR/mode/meas/FIFO state.

What the driver does
--------------------
  - Init: soft reset, probe IDs, default +/-2g, 100 Hz, measure, accel-only meas mode.
  - Configure: range, ODR, standby/measure, TEMP_CTL/ADC_CTL meas modes (accel / temp / both).
  - Read: XYZ raw (DATA_RDY + XDATA path), temperature raw and Celsius, STATUS.
  - Interrupts: INTMAP1/2 routing (struct mirrors ref int_map).
  - Activity / inactivity: thresholds, TIME_ACT/TIME_INACT, ABS/REL, axis mask, LINKLOOP, disable.
  - FIFO: mode, format, watermark, entry count, drain XYZ (14b + channel ID read mode).

Tests / integration (node_acc_adxl367_test)
---------------------------------------------
  Phase 1: basic XYZ polling
  Phase 2: meas modes; TEMP_ONLY rejects read_xyz (INVALID_STATE)
  Phase 3: FIFO stream + watermark + drain (polling)
  Phase 4: INT1 + GPIO ISR for FIFO watermark
  Phase 5: linked ACT/INACT on INT2 (chip detection)
  Phase 6 default: software still/motion via max(|dx|,|dy|,|dz|) between samples (not hardware ACT)
  Optional: run_xyz_stats_capture for mean/min/max over a window

Board: INT GPIOs and tuning macros are in node_acc_adxl367_test.c (see DOC for pinout if documented).
