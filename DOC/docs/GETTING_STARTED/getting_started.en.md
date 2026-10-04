# Getting started

Choose an independent ESP-IDF project first. Components present in a directory, functions defined in a source file, and functions actually called at startup are different things.

## 1. Choose a project

| Goal | Project | Current default execution |
|---|---|---|
| Matrices and numerics | `CODE/AIoTNode-TinyAuton-MATH` | `tiny_matrix_test()` with G1/G2/G3 quality checks enabled |
| Signal processing | `CODE/AIoTNode-TinyAuton-DSP` | FFT, DWT and ICA; Support/Signal/Filter selectors are off |
| On-device learning | `CODE/AIoTNode-TinyAuton-AI` | MLP, CNN1D and Attention examples, in sequence |

The entry is `main/AIoTNode.cpp` in each project. See [projects and versions](../PROJECTS/projects.md) for component and copy differences.

## 2. Build and run

The three `sdkconfig` files, the AI dependency lock and CI use ESP-IDF 6.0. In a terminal with that toolchain activated:

```bash
cd CODE/AIoTNode-TinyAuton-DSP
idf.py build
idf.py -p YOUR_SERIAL_PORT flash monitor
```

Replace `YOUR_SERIAL_PORT` with your device port. Existing configurations target ESP32-S3; use `idf.py set-target` when changing targets and save your custom configuration first. Review storage, PSRAM and component settings in `idf.py menuconfig`. The project CMake files, component directories and dependency locks define the build.

## 3. Select a verification run

- **DSP:** set the entry's `TEST_TINY_DSP_*` selectors. The default transform group calls `tiny_fft_test()`, `tiny_dwt_test_all()` and `tiny_ica_test_all()`.
- **Math:** select calls inside `middleware/tiny_math/mat/tiny_matrix_test.cpp`. Most A–F calls are commented out; historical stage outputs do not imply that the current startup runs those stages.
- **AI:** set `TEST_TINY_AI_MLP/CNN/ATTENTION`. Training can occupy the CPU for extended periods. This example entry calls `esp_task_wdt_deinit()`.

[FFT tests and results](../DSP/TRANSFORM/FFT/test.md) explain inputs, criteria and output; the [matrix test overview](../MATH/MATRIX/TESTS/overview.md) indexes historical runs by stage.

## 4. Record reproduction conditions

Record the project, Git commit, hardware, ESP-IDF version, configuration, input, selectors, acceptance criteria and complete serial output. Missing historical dates or devices stay marked as unrecorded. Source dates are not execution dates, and a documentation build is not a hardware test.
