# TinyAI Header

!!! info
    `tiny_ai.h` is the unified entry-point header. It pulls in every quant / core / layers / models / train submodule in dependency order and exposes three C-callable example entry points (`example_mlp` / `example_cnn` / `example_attention`). After porting, simply add `#include "tiny_ai.h"` to any source file that uses AI features.

## INCLUDE HIERARCHY

```txt
tiny_ai.h
├── tiny_ai_config.h          # platform / memory / error codes
└── (C++) only expanded when __cplusplus is defined
    ├── Quantisation
    │   ├── tiny_quant_config.h
    │   ├── tiny_fp8.hpp
    │   ├── tiny_quant.h
    │   └── tiny_quant.hpp
    ├── Core
    │   ├── tiny_tensor.hpp
    │   ├── tiny_activation.hpp
    │   ├── tiny_loss.hpp
    │   └── tiny_optimizer.hpp
    ├── Layers
    │   ├── tiny_layer.hpp
    │   ├── tiny_dense.hpp
    │   ├── tiny_conv.hpp
    │   ├── tiny_pool.hpp
    │   ├── tiny_norm.hpp
    │   └── tiny_attention.hpp
    ├── Models
    │   ├── tiny_sequential.hpp
    │   ├── tiny_mlp.hpp
    │   └── tiny_cnn.hpp
    └── Training
        ├── tiny_dataset.hpp
        └── tiny_trainer.hpp
```

!!! note
    All C++ headers live inside an `#ifdef __cplusplus` block. For pure C projects, `tiny_ai.h` only exposes the macros / error codes from `tiny_ai_config.h` plus the three `extern "C"` example entry points.

## FULL SOURCE

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
