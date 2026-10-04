# 快速开始

先选择一个独立 ESP-IDF 工程。工程目录中包含的模块、源码中定义的测试函数与启动时实际调用的函数，是三个不同的概念。

## 1. 选择工程

| 目标 | 工程 | 当前默认执行 |
|---|---|---|
| 矩阵与数值计算 | `CODE/AIoTNode-TinyAuton-MATH` | `tiny_matrix_test()`，当前启用 G1/G2/G3 质量保证组 |
| 信号处理 | `CODE/AIoTNode-TinyAuton-DSP` | FFT、DWT、ICA；Support/Signal/Filter 组默认关闭 |
| 板载学习 | `CODE/AIoTNode-TinyAuton-AI` | MLP、CNN1D、Attention 三个示例顺序执行 |

每个工程的入口是 `main/AIoTNode.cpp`。详细模块和副本差异见[工程与版本](../PROJECTS/projects.md)。

## 2. 构建与运行

当前三个工程的 `sdkconfig`、AI 工程的锁文件以及 CI 使用 ESP-IDF 6.0。先在已激活该工具链的终端中执行：

```bash
cd CODE/AIoTNode-TinyAuton-DSP
idf.py build
idf.py -p YOUR_SERIAL_PORT flash monitor
```

将 `YOUR_SERIAL_PORT` 替换为实际串口。现有配置目标为 ESP32-S3；更换目标时再使用 `idf.py set-target`，并先保存自己的配置。检查 `idf.py menuconfig` 中的存储、PSRAM 和组件设置。依赖以各工程的 `CMakeLists.txt`、组件目录与锁文件为准。

## 3. 选择一次验证

- **DSP：** 在入口中设置 `TEST_TINY_DSP_*`。默认变换组调用 `tiny_fft_test()`、`tiny_dwt_test_all()`、`tiny_ica_test_all()`。
- **Math：** 在 `middleware/tiny_math/mat/tiny_matrix_test.cpp` 的 `tiny_matrix_test()` 中选择测试。A–F 的多数调用当前已注释；历史阶段输出不代表本次启动覆盖这些阶段。
- **AI：** 在入口中设置 `TEST_TINY_AI_MLP/CNN/ATTENTION`。示例可能长时间占用 CPU；当前 AI runner 调用了 `esp_task_wdt_deinit()`，这是该演示入口的选择。

[FFT 测试与结果](../DSP/TRANSFORM/FFT/test.md)展示输入、判据和输出的阅读方式；[矩阵测试概览](../MATH/MATRIX/TESTS/overview.md)按阶段索引历史结果。

## 4. 记录复现条件

保留工程路径、Git 提交、硬件、ESP-IDF 版本、配置、输入、启用开关、通过条件和完整串口输出。历史记录缺少日期或设备时标为未记录；源码日期不替代运行日期，文档构建也不代表硬件测试通过。
