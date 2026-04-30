# 说明

!!! note "说明"
    `tiny_quant_config.h` 集中定义了量化子系统的数据类型枚举、量化参数结构以及各种格式的极值常量。所有 INT / FP8 量化函数都依赖这些类型，是整个 quant 模块的基石。

## tiny_dtype_t

```c
typedef enum
{
    TINY_DTYPE_FLOAT32  = 0,  // 32-bit IEEE 754 float（原生计算类型）
    TINY_DTYPE_INT16    = 1,  // 有符号 16-bit
    TINY_DTYPE_INT8     = 2,  // 有符号 8-bit（ESP32-S3 上最硬件友好）
    TINY_DTYPE_FP8_E4M3 = 3,  // 8-bit float E4M3FN：范围 ±448，权重/激活
    TINY_DTYPE_FP8_E5M2 = 4,  // 8-bit float E5M2：范围 ±57344，梯度
} tiny_dtype_t;
```

## tiny_quant_params_t

```c
typedef struct
{
    tiny_dtype_t dtype;
    float        scale;       // float_val = scale * (quant_val - zero_point)
    int          zero_point;  // 对称量化 / FP8 时为 0
} tiny_quant_params_t;
```

`tiny_ai` 默认使用 **对称量化**（`zero_point = 0`）：

\[
\mathrm{quant} = \mathrm{round}(x / \text{scale}),\quad
x = \text{scale} \cdot \mathrm{quant}
\]

`scale` 由 `tiny_quant_calibrate_minmax` 根据张量绝对最大值计算：

\[
\text{scale} = \frac{\max(|x|)}{Q_\text{max}}
\]

其中 \( Q_\text{max} \) 为 INT8 = 127、INT16 = 32767、FP8 E4M3 = 448、FP8 E5M2 = 57344。

## 格式极值

```c
// FP8 E4M3FN（OCP 规范）：无 ±inf，NaN 编码为 0x7F / 0xFF
#define TINY_FP8_E4M3_MAX   448.0f
#define TINY_FP8_E4M3_MIN  (-448.0f)
#define TINY_FP8_E4M3_NAN   0x7Fu

// FP8 E5M2：支持 ±inf 与 NaN
#define TINY_FP8_E5M2_MAX  57344.0f
#define TINY_FP8_E5M2_MIN  (-57344.0f)
#define TINY_FP8_E5M2_INF  0x7Cu
#define TINY_FP8_E5M2_NAN  0x7Fu

// INT8 / INT16 对称范围
#define TINY_INT8_MAX   127
#define TINY_INT8_MIN  (-128)
#define TINY_INT16_MAX  32767
#define TINY_INT16_MIN (-32768)
```

## 选择哪种 dtype

| 场景 | 建议 dtype | 备注 |
| --- | --- | --- |
| 部署纯推理 + 内存敏感 | `INT8` | 与 INT8 dense kernel 配合，4× 压缩 |
| 高精度推理 / 中间统计量 | `INT16` | 2× 压缩，几乎无精度损失 |
| 静态权重，需要 FP 表达 | `FP8_E4M3` | 4× 压缩，比 INT8 表达范围更大 |
| 梯度 / 反向中间结果 | `FP8_E5M2` | 范围更大、精度更低，适合梯度 |
| 训练时全程 | `FLOAT32` | 反向稳定性最高 |

## 命名空间

`tiny_quant_config.h` 中的类型与常量都暴露在全局 / `extern "C"` 段，C 与 C++ 都能直接使用。`tiny_quant.hpp` 在此基础上提供 `tiny::QuantParams`：

```cpp
struct QuantParams
{
    tiny_dtype_t dtype;
    float        scale;
    int          zero_point;

    tiny_quant_params_t to_c() const { return { dtype, scale, zero_point }; }
};
```

`to_c()` 方便在 C++ 调用 C 接口（例如 `tiny_quant_f32_to_int8`）。
