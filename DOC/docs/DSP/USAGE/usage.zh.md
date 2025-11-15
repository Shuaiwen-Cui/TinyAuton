# 使用说明

!!! info "使用说明"
    该文档是对 `tiny_dsp` 模块的使用说明。

## 整体引入TinyDSP

!!! info
    适用于C项目，或者结构较为简单的C++项目。

```c
#include "tiny_dsp.h"
```

## 分模块引入TinyDSP

!!! info
    适用于需要精确控制引入模块的项目，或者复杂的C++项目。

```c
#include "tiny_conv.h" // 引入卷积模块
#include "tiny_corr.h" // 引入相关模块
...
```

!!! tip
    具体的使用方法请参考测试代码。