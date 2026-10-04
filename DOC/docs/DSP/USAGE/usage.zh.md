# 使用说明 {#_1}

!!! info "实现依据与记录"
    本节接口以 `CODE/AIoTNode-TinyAuton-DSP/middleware/` 为依据。源码摘录与串口输出包含历史记录；是否运行某项测试，请核对工程入口和启用开关。

!!! info "使用说明"
    该文档是对 `tiny_dsp` 模块的使用说明。

## 整体引入TinyDSP {#tinydsp}

!!! info
    适用于C项目，或者结构较为简单的C++项目。

```c
#include "tiny_dsp.h"
```

## 分模块引入TinyDSP {#tinydsp_1}

!!! info
    适用于需要精确控制引入模块的项目，或者复杂的C++项目。

```c
// 信号处理模块 (signal/)
#include "tiny_conv.h"        // 卷积模块
#include "tiny_corr.h"        // 相关模块
#include "tiny_resample.h"    // 重采样模块

// 滤波器模块 (filter/)
#include "tiny_fir.h"         // FIR滤波器模块
#include "tiny_iir.h"         // IIR滤波器模块

// 变换模块 (transform/)
#include "tiny_fft.h"         // 快速傅里叶变换模块
#include "tiny_dwt.h"         // 离散小波变换模块
#include "tiny_ica.hpp"         // 独立成分分析模块

// 支持功能模块 (support/)
#include "tiny_view.h"        // 信号查看/支持模块
```

!!! tip
    具体的使用方法请参考测试代码。
