# 工程与版本

本页描述当前仓库工作区的工程和入口，不将文件夹名称视为正式发布版本。所有工程均以 ESP32-S3 和 ESP-IDF 6.0 配置为起点。

## 工程组成

| 工程 | 中间件 | 默认入口 |
|---|---|---|
| `AIoTNode-TinyAuton-MATH` | Toolbox、Math、DSP | C++ 矩阵测试 |
| `AIoTNode-TinyAuton-DSP` | Toolbox、Math、DSP | FFT、DWT、ICA 测试 |
| `AIoTNode-TinyAuton-AI` | Toolbox、Math、DSP、AI | MLP、CNN1D、Attention 示例 |

工程位于 `CODE/`。MATH 工程也包含 DSP，因此工程名表达的是测试用途，并非严格的组件裁剪。

## 文档的实现依据

| 章节 | 主要依据 |
|---|---|
| Math | MATH 工程；三个工程的 `tiny_math` 文件当前一致 |
| DSP | DSP 工程；它与 AI 工程的 `tiny_dsp` 文件当前一致 |
| AI | AI 工程 |
| Toolbox | AI 工程；三个工程的 `tiny_toolbox` 文件当前一致 |

MATH 工程内的 DSP 副本有 21 个源码、头文件或测试文件与 DSP/AI 工程不同。DWT 多级分解的新版接口增加 `cD_lens_out/cD_total_len`，重构需要 `cD_lens`。在 MATH 工程中使用 DSP 前应核对本地头文件，不能混用两个版本的调用方式。本次文档整理未同步这些固件副本。

## 与 TinySHM 的关系

核对时，AI 工程中的 TinyAI、TinyDSP、TinyToolbox 与 TinySHM Core 对应源码一致，因此复用了部分文档组织和解读。TinySHM 的 TinyMath 增加了独立的 `linalg/cfloat/decomp/eigen/iterative` 等 C 模块；这些目录不属于当前 TinyAuton。TinyAuton 的 C++ `tiny::Mat` 自身仍提供分解、求解和特征值方法。

TinyMeasurement、TinySysid、TinyDamage、TinyBench、TinyOrch 不在当前 TinyAuton 的工程中。矩阵测试中的结构动力学例子是数值算法应用示例，不表示已包含完整 SHM 应用模块。

## 历史与参考目录

`ARCHIVED/` 保存早期工程，`REF/` 保存参考固件与主机脚本。正文 API 以 `CODE/` 中上述工程为依据；历史源码摘录和日志完整保留，并与当前调用说明区分。

[开始构建与验证](../GETTING_STARTED/getting_started.md)
