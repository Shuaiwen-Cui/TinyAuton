# TinyAI 头文件

!!! info
    `tiny_ai.h` 是 `tiny_ai` 库的统一入口头文件，按依赖顺序拉入了 quant / core / layers / models / train 子模块全部头文件，并暴露三个供 C 调用的示例入口（`example_mlp` / `example_cnn` / `example_attention`）。完成移植后，在使用 AI 功能的源文件中加入 `#include "tiny_ai.h"` 即可获得整套 API。

## 包含层级

```txt
tiny_ai.h
├── tiny_ai_config.h          # 平台 / 内存 / 错误码
└── (C++) 仅在 __cplusplus 下展开
    ├── 量化
    │   ├── tiny_quant_config.h
    │   ├── tiny_fp8.hpp
    │   ├── tiny_quant.h
    │   └── tiny_quant.hpp
    ├── 核心
    │   ├── tiny_tensor.hpp
    │   ├── tiny_activation.hpp
    │   ├── tiny_loss.hpp
    │   └── tiny_optimizer.hpp
    ├── 层
    │   ├── tiny_layer.hpp
    │   ├── tiny_dense.hpp
    │   ├── tiny_conv.hpp
    │   ├── tiny_pool.hpp
    │   ├── tiny_norm.hpp
    │   └── tiny_attention.hpp
    ├── 模型
    │   ├── tiny_sequential.hpp
    │   ├── tiny_mlp.hpp
    │   └── tiny_cnn.hpp
    └── 训练
        ├── tiny_dataset.hpp
        └── tiny_trainer.hpp
```

!!! note
    所有 C++ 头文件都在 `#ifdef __cplusplus` 块内展开。对于纯 C 项目，`tiny_ai.h` 只暴露 `tiny_ai_config.h` 的宏 / 错误码以及三个 `extern "C"` 示例入口。

## 完整源码

```c
/**
 * @file tiny_ai.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_ai | Unified entry-point header — include this to access all AI
 *        functionality in the tiny_ai middleware.
 *
 * @details
 *  Dependency chain:  tiny_toolbox → tiny_math → tiny_dsp → tiny_ai
 */

#pragma once

#include "tiny_ai_config.h"

#ifdef __cplusplus

/* Quantisation ---------------------------------------------------------------- */
#include "tiny_quant_config.h"
#include "tiny_fp8.hpp"
#include "tiny_quant.h"
#include "tiny_quant.hpp"

/* Core ----------------------------------------------------------------------- */
#include "tiny_tensor.hpp"
#include "tiny_activation.hpp"
#include "tiny_loss.hpp"
#include "tiny_optimizer.hpp"

/* Layers --------------------------------------------------------------------- */
#include "tiny_layer.hpp"
#include "tiny_dense.hpp"
#include "tiny_conv.hpp"
#include "tiny_pool.hpp"
#include "tiny_norm.hpp"
#include "tiny_attention.hpp"

/* Models --------------------------------------------------------------------- */
#include "tiny_sequential.hpp"
#include "tiny_mlp.hpp"
#include "tiny_cnn.hpp"

/* Training utilities --------------------------------------------------------- */
#include "tiny_dataset.hpp"
#include "tiny_trainer.hpp"

#endif /* __cplusplus */

/* ============================================================================
 * Example entry points (callable from C and C++)
 * ============================================================================ */
#ifdef __cplusplus
extern "C" {
#endif

void example_mlp(void);
void example_cnn(void);
void example_attention(void);

#ifdef __cplusplus
}
#endif
```
