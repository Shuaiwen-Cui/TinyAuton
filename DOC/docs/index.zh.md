# TinyAuton {#tinyauton}

数学、信号处理与板载神经网络训练，为微控制器上的自主计算提供基础。当前实现以 ESP32-S3 / ESP-IDF 为主要载体。

<div class="grid cards auton-entry-grid" markdown>

- :material-compass-outline: **开始使用**

    选择工程，构建固件并追踪一次实际执行。

    [快速开始 →](GETTING_STARTED/getting_started.md)

- :material-package-variant: **工程与版本**

    三个工程的模块、默认测试和副本差异。

    [选择工程 →](PROJECTS/projects.md)

- :material-chart-line: **阅读验证结果**

    先看输入、结果与判据，再展开完整源码。

    [FFT 测试 →](DSP/TRANSFORM/FFT/test.md)

</div>

## 从任务选择模块 {#auton-module-guide}

| 任务 | 起点 |
|---|---|
| 向量与矩阵计算 | [Math](MATH/math.md) |
| 卷积、滤波、FFT、小波、ICA | [DSP](DSP/dsp.md) |
| 神经网络、训练与量化 | [AI](AI/ai.md) |
| 运行计时与世界时间 | [Toolbox](TOOLBOX/toolbox.md) |

当前仓库提供计算库和独立示例工程。平台移植与能力边界见[架构](ARCHITECTURE/architecture.md)。

![封面](cover.jpg){ .auton-cover }

## 关于本项目 {#_1}

这个项目致力于开发一个运行在 MCU 设备上的小型智能体相关的计算库，以服务于多智能体系统，涵盖数学运算、数字信号处理和 TinyML。

!!! info "名字的由来"
    "TinyAuton" 是 "Tiny" 和 "Auton" 的组合。"Tiny" 意味着智能体被设计为运行在 MCU 设备上，而 "Auton" 是 "Autonomous Agent" 的缩写。

## 目标硬件 {#_2}

- MCU 设备（目前以 ESP32 为主要目标）

## 覆盖范围 {#_3}

- 平台适配与各类工具（时间、通讯等）
- 基本数学运算
- 数字信号处理
- TinyML / 边缘人工智能


## 开发载体 {#_4}

!!! TIP 
    以下硬件仅做展示用途，本项目并不局限于此，可以移植到其他类型的硬件上。

- Alientek 的 DNESP32S3M（ESP32-S3）

![DNESP32S3M](DNESP32S3M.png){ .auton-hardware }

![DNESP32S3M-BACK](DNESP32S3M-BACK.png){ .auton-hardware }

- NexNode AIoT节点

![PCB](PCB.png){ .auton-hardware }

![WSN](WSN.jpg){ .auton-hardware }

<div class="grid cards" markdown>

-   :simple-github:{ .lg .middle } __NexNode__

    ---

    [:octicons-arrow-right-24: <a href="https://github.com/Shuaiwen-Cui/NexNode.git" target="_blank"> 代码 </a>](#)

    [:octicons-arrow-right-24: <a href="http://www.cuishuaiwen.com:9100/" target="_blank"> 文档 </a>](#)


</div>

## 项目架构 {#_5}

```txt
+------------------------------+
| 应用层                        |
+------------------------------+
|   - TinyAI                   | <-- AI 函数
|   - TinyDSP                  | <-- DSP 函数
|   - TinyMath                 | <-- 常用数学函数
|   - TinyToolbox              | <-- 平台底层优化 + 各种工具
| 中间件                        |
+------------------------------+
| 驱动层                        |
+------------------------------+
| 硬件层                        |
+------------------------------+

```
