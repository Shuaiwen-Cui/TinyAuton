# C++ 矩阵测试与结果

**历史记录重复：** 原页“基础操作”小节的输出与“对象基础”完全相同，内容为 A 组测试。它不能作为 B 组基础操作已经验证的证据；下方按原记录保留。

历史记录按 A–G 的八个阶段组织。以下计数仅统计原始输出中的标签，既不代表全部测试覆盖率，也不证明当前固件通过。日期、设备和提交未记录。

**已知历史失败：** [矩阵特性](properties.md)的 C4.4 非方阵伴随矩阵错误路径有一条 `[FAIL]`；原始输出保留。

| 阶段 | 原日志 `[PASS]` 标签 | `[FAIL]` 标签 |
|---|---:|---:|
| [对象基础](foundation.md) | 12 | 0 |
| [基础操作](basic.md) | 12 | 0 |
| [矩阵特性](properties.md) | 39 | 1 |
| [线性系统](systems.md) | 9 | 0 |
| [矩阵分解](decompositions.md) | 53 | 0 |
| [特征值应用](eigen.md) | 58 | 0 |
| [辅助功能](auxiliary.md) | 0 | 0 |
| [质量保证](quality.md) | 1 | 0 |

## 当前复现入口

MATH 工程的 `main/AIoTNode.cpp` 调用 `tiny_matrix_test()`。当前 runner 启用 G1 `test_boundary_conditions()`、G2 `test_performance_benchmarks()`、G3 `test_memory_layout()`，A–F 的主要调用已注释。要复核某阶段，应在 runner 中显式启用对应调用并记录新输出。

[完整源码与历史记录](../tiny-matrix-test.md) · [接口概览](../OVERVIEW/api.md)
