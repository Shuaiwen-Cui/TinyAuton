# Notes

!!! note "Notes"
    `tiny_quant_config.h` centralises the dtype enum, the quantisation parameter struct, and the format-specific limits. Every INT / FP8 quantiser depends on these types, so this header is the foundation of the entire `quant` subsystem.

## tiny_dtype_t

```c
typedef enum
{
    TINY_DTYPE_FLOAT32  = 0,  // 32-bit IEEE 754 float (native compute type)
    TINY_DTYPE_INT16    = 1,  // signed 16-bit integer
    TINY_DTYPE_INT8     = 2,  // signed 8-bit integer (most HW-friendly on ESP32-S3)
    TINY_DTYPE_FP8_E4M3 = 3,  // 8-bit float E4M3FN: range ±448, weights/activations
    TINY_DTYPE_FP8_E5M2 = 4,  // 8-bit float E5M2:   range ±57344, gradients
} tiny_dtype_t;
```

## tiny_quant_params_t

```c
typedef struct
{
    tiny_dtype_t dtype;
    float        scale;       // float_val = scale * (quant_val - zero_point)
    int          zero_point;  // 0 for symmetric / FP8
} tiny_quant_params_t;
```

`tiny_ai` defaults to **symmetric quantisation** (`zero_point = 0`):

\[
\mathrm{quant} = \mathrm{round}(x / \text{scale}),\quad
x = \text{scale} \cdot \mathrm{quant}
\]

`scale` is derived from the tensor's max absolute value:

\[
\text{scale} = \frac{\max(|x|)}{Q_\text{max}}
\]

with \( Q_\text{max} \) = 127 (INT8), 32767 (INT16), 448 (FP8 E4M3), 57344 (FP8 E5M2).

## FORMAT LIMITS

```c
// FP8 E4M3FN (OCP spec): no ±inf, NaN = 0x7F / 0xFF
#define TINY_FP8_E4M3_MAX   448.0f
#define TINY_FP8_E4M3_MIN  (-448.0f)
#define TINY_FP8_E4M3_NAN   0x7Fu

// FP8 E5M2: supports ±inf and NaN
#define TINY_FP8_E5M2_MAX  57344.0f
#define TINY_FP8_E5M2_MIN  (-57344.0f)
#define TINY_FP8_E5M2_INF  0x7Cu
#define TINY_FP8_E5M2_NAN  0x7Fu

// INT8 / INT16 symmetric ranges
#define TINY_INT8_MAX   127
#define TINY_INT8_MIN  (-128)
#define TINY_INT16_MAX  32767
#define TINY_INT16_MIN (-32768)
```

## CHOOSING A dtype

| Scenario | Suggested dtype | Notes |
| --- | --- | --- |
| Inference, memory-bound | `INT8` | Works with the INT8 dense kernel, 4× compression |
| High-precision inference / stats | `INT16` | 2× compression, near-zero loss |
| Static weights needing FP range | `FP8_E4M3` | 4× compression, wider range than INT8 |
| Gradients / backward intermediates | `FP8_E5M2` | Larger range, lower precision, ideal for gradients |
| Training | `FLOAT32` | Maximum numerical stability |

## NAMESPACING

The types and constants in `tiny_quant_config.h` live in the global / `extern "C"` scope and are usable from both C and C++. `tiny_quant.hpp` adds the C++ `tiny::QuantParams` wrapper:

```cpp
struct QuantParams
{
    tiny_dtype_t dtype;
    float        scale;
    int          zero_point;

    tiny_quant_params_t to_c() const { return { dtype, scale, zero_point }; }
};
```

`to_c()` bridges C++ code into the C API (e.g. `tiny_quant_f32_to_int8`).
