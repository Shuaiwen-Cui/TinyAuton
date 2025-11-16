# 说明

!!! note "说明"
    无限脉冲响应（IIR）滤波器是使用反馈的递归数字滤波器，在相同规格下比 FIR 滤波器更高效。但是，如果设计不当，IIR 滤波器可能不稳定。它们广泛应用于音频处理、控制系统和信号调理，其中计算效率很重要。

## IIR 滤波器概述

### 数学原理

IIR 滤波器由其差分方程定义，包括前馈项和反馈项：

\[
y[n] = \sum_{k=0}^{M} b[k] \cdot x[n-k] - \sum_{k=1}^{N} a[k] \cdot y[n-k]
\]

其中：

- \( x[n] \) 是输入信号

- \( y[n] \) 是输出信号

- \( b[k] \) 是前馈（分子）系数

- \( a[k] \) 是反馈（分母）系数

- \( M \) 是分子阶数

- \( N \) 是分母阶数

**传递函数**：

\[
H(z) = \frac{\sum_{k=0}^{M} b[k] \cdot z^{-k}}{1 + \sum_{k=1}^{N} a[k] \cdot z^{-k}}
\]

**关键特性**：

- **递归**：使用反馈（先前的输出）

- **高效**：相同规格下比 FIR 需要更少的系数

- **可能不稳定**：极点必须在单位圆内

- **非线性相位**：通常具有非线性相位响应

## 滤波器类型

库支持四种基本滤波器类型：

- **低通**：通过截止频率以下的频率，衰减以上频率
- **高通**：通过截止频率以上的频率，衰减以下频率
- **带通**：通过频带内的频率，衰减外部频率
- **带阻（陷波）**：衰减频带内的频率，通过外部频率

## 滤波器设计

### 设计方法

库支持 Butterworth 滤波器设计（计划支持 Chebyshev、Elliptic 和 Bessel）：

- **Butterworth**：最大平坦通带，单调阻带
- **Chebyshev Type I**：等波纹通带，单调阻带（未来）
- **Chebyshev Type II**：单调通带，等波纹阻带（未来）
- **Elliptic**：通带和阻带都等波纹（未来）
- **Bessel**：线性相位响应（未来）

### 双线性变换

IIR 滤波器使用双线性变换设计，将模拟 s 平面映射到数字 z 平面：

\[
s = \frac{2}{T} \cdot \frac{1 - z^{-1}}{1 + z^{-1}}
\]

其中 \( T \) 是采样周期。

### 设计参数

- **截止频率**：归一化频率（0.0 到 0.5，其中 0.5 = 奈奎斯特频率）
- **滤波器阶数**：决定过渡的陡度和阻带衰减
- **设计方法**：影响通带/阻带特性

**归一化频率**：

\[
f_{norm} = \frac{f_{cutoff}}{f_s / 2}
\]

其中 \( f_s \) 是采样率。

## 滤波器设计函数

### tiny_iir_design_lowpass

```c
/**
 * @name tiny_iir_design_lowpass
 * @brief Design a low-pass IIR filter
 * @param cutoff_freq Normalized cutoff frequency (0.0 to 0.5)
 * @param order Filter order
 * @param design_method Design method (Butterworth, Chebyshev, etc.)
 * @param ripple_db Passband ripple in dB (for Chebyshev)
 * @param b_coeffs Output numerator coefficients (size: order + 1)
 * @param a_coeffs Output denominator coefficients (size: order + 1)
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_design_lowpass(float cutoff_freq, int order,
                                      tiny_iir_design_method_t design_method,
                                      float ripple_db,
                                      float *b_coeffs, float *a_coeffs);
```

**描述**: 

使用指定的设计方法设计低通 IIR 滤波器。目前支持 1 阶和 2 阶的 Butterworth 设计。

**参数**:

- `cutoff_freq`: 归一化截止频率（0.0 到 0.5，其中 0.5 = 奈奎斯特频率）。

- `order`: 滤波器阶数。目前支持 1 和 2 阶的 Butterworth。

- `design_method`: 来自 `tiny_iir_design_method_t` 枚举的设计方法。目前仅支持 `TINY_IIR_DESIGN_BUTTERWORTH`。

- `ripple_db`: 通带波纹（dB）（用于 Chebyshev 设计，当前未使用）。

- `b_coeffs`: 分子系数的输出数组。大小必须至少为 `order + 1`。

- `a_coeffs`: 分母系数的输出数组。大小必须至少为 `order + 1`。注意：`a[0]` 始终为 1.0（归一化形式）。

**返回值**: 

返回成功或错误代码。

**注意**: 

系数采用归一化形式，其中 `a[0] = 1.0`。更高阶的滤波器需要分解为级联双二阶（二阶节）。

### tiny_iir_design_highpass

```c
/**
 * @name tiny_iir_design_highpass
 * @brief Design a high-pass IIR filter
 * @param cutoff_freq Normalized cutoff frequency (0.0 to 0.5)
 * @param order Filter order
 * @param design_method Design method
 * @param ripple_db Passband ripple in dB
 * @param b_coeffs Output numerator coefficients (size: order + 1)
 * @param a_coeffs Output denominator coefficients (size: order + 1)
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_design_highpass(float cutoff_freq, int order,
                                       tiny_iir_design_method_t design_method,
                                       float ripple_db,
                                       float *b_coeffs, float *a_coeffs);
```

**描述**: 

使用指定的设计方法设计高通 IIR 滤波器。目前支持 1 阶和 2 阶的 Butterworth 设计。

**参数**:

- `cutoff_freq`: 归一化截止频率（0.0 到 0.5）。

- `order`: 滤波器阶数。目前支持 1 和 2 阶。

- `design_method`: 设计方法。目前仅支持 `TINY_IIR_DESIGN_BUTTERWORTH`。

- `ripple_db`: 通带波纹（dB）（当前未使用）。

- `b_coeffs`: 分子系数的输出数组。大小必须至少为 `order + 1`。

- `a_coeffs`: 分母系数的输出数组。大小必须至少为 `order + 1`。

**返回值**: 

返回成功或错误代码。

### tiny_iir_design_bandpass

```c
/**
 * @name tiny_iir_design_bandpass
 * @brief Design a band-pass IIR filter
 * @param low_freq Lower cutoff frequency (normalized, 0.0 to 0.5)
 * @param high_freq Upper cutoff frequency (normalized, 0.0 to 0.5)
 * @param order Filter order
 * @param design_method Design method
 * @param ripple_db Passband ripple in dB
 * @param b_coeffs Output numerator coefficients
 * @param a_coeffs Output denominator coefficients
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_design_bandpass(float low_freq, float high_freq,
                                       int order,
                                       tiny_iir_design_method_t design_method,
                                       float ripple_db,
                                       float *b_coeffs, float *a_coeffs);
```

**描述**: 

设计带通 IIR 滤波器。目前返回 `TINY_ERR_NOT_SUPPORTED`，因为完整的带通设计尚未实现。

**参数**:

- `low_freq`: 下截止频率（归一化，0.0 到 0.5）。必须小于 `high_freq`。

- `high_freq`: 上截止频率（归一化，0.0 到 0.5）。必须大于 `low_freq`。

- `order`: 滤波器阶数。

- `design_method`: 设计方法。

- `ripple_db`: 通带波纹（dB）。

- `b_coeffs`: 分子系数的输出数组。

- `a_coeffs`: 分母系数的输出数组。

**返回值**: 

目前返回 `TINY_ERR_NOT_SUPPORTED`。

### tiny_iir_design_bandstop

```c
/**
 * @name tiny_iir_design_bandstop
 * @brief Design a band-stop (notch) IIR filter
 * @param low_freq Lower cutoff frequency (normalized, 0.0 to 0.5)
 * @param high_freq Upper cutoff frequency (normalized, 0.0 to 0.5)
 * @param order Filter order
 * @param design_method Design method
 * @param ripple_db Passband ripple in dB
 * @param b_coeffs Output numerator coefficients
 * @param a_coeffs Output denominator coefficients
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_design_bandstop(float low_freq, float high_freq,
                                       int order,
                                       tiny_iir_design_method_t design_method,
                                       float ripple_db,
                                       float *b_coeffs, float *a_coeffs);
```

**描述**: 

设计带阻（陷波）IIR 滤波器。目前返回 `TINY_ERR_NOT_SUPPORTED`，因为完整的带阻设计尚未实现。

**参数**:

- `low_freq`: 下截止频率（归一化，0.0 到 0.5）。必须小于 `high_freq`。

- `high_freq`: 上截止频率（归一化，0.0 到 0.5）。必须大于 `low_freq`。

- `order`: 滤波器阶数。

- `design_method`: 设计方法。

- `ripple_db`: 通带波纹（dB）。

- `b_coeffs`: 分子系数的输出数组。

- `a_coeffs`: 分母系数的输出数组。

**返回值**: 

目前返回 `TINY_ERR_NOT_SUPPORTED`。

## 滤波器应用

### 批处理

### tiny_iir_filter_f32

```c
/**
 * @name tiny_iir_filter_f32
 * @brief Apply IIR filter to a signal (batch processing)
 * @param input Input signal array
 * @param input_len Length of input signal
 * @param b_coeffs Numerator coefficients
 * @param num_b Number of b coefficients
 * @param a_coeffs Denominator coefficients
 * @param num_a Number of a coefficients
 * @param output Output filtered signal array (size: input_len)
 * @param initial_state Initial state vector (can be NULL for zero initial conditions)
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_filter_f32(const float *input, int input_len,
                                  const float *b_coeffs, int num_b,
                                  const float *a_coeffs, int num_a,
                                  float *output,
                                  const float *initial_state);
```

**描述**: 

使用直接型 II 转置结构将 IIR 滤波器应用于整个信号。适用于整个信号可用的批处理。

**特点**:

- 使用直接型 II 转置实现（高效）

- 支持初始状态条件

- 输出长度等于输入长度

**参数**:

- `input`: 输入信号数组指针。

- `input_len`: 输入信号长度。

- `b_coeffs`: 分子系数指针。

- `num_b`: 分子系数数量。

- `a_coeffs`: 分母系数指针。注意：`a[0]` 应为 1.0（归一化形式）。

- `num_a`: 分母系数数量。

- `output`: 滤波信号的输出数组指针。大小必须至少为 `input_len`。

- `initial_state`: 初始状态向量。可以为 `NULL` 表示零初始条件。大小应为 `max(num_b, num_a) - 1`。

**返回值**: 

返回成功或错误代码。

**注意**: 

滤波器使用直接型 II 转置结构，计算效率高，需要最少的状态存储。

### 实时处理

### tiny_iir_init

```c
/**
 * @name tiny_iir_init
 * @brief Initialize IIR filter structure for real-time filtering
 * @param filter Pointer to IIR filter structure
 * @param b_coeffs Numerator coefficients (will be copied)
 * @param num_b Number of b coefficients
 * @param a_coeffs Denominator coefficients (will be copied)
 * @param num_a Number of a coefficients
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_init(tiny_iir_filter_t *filter,
                            const float *b_coeffs, int num_b,
                            const float *a_coeffs, int num_a);
```

**描述**: 

初始化 IIR 滤波器结构以进行实时逐样本处理。为系数和状态变量分配内存。

**参数**:

- `filter`: `tiny_iir_filter_t` 结构指针。

- `b_coeffs`: 分子系数指针。将在内部复制。

- `num_b`: 分子系数数量。

- `a_coeffs`: 分母系数指针。将在内部复制。

- `num_a`: 分母系数数量。

**返回值**: 

返回成功或错误代码。

**内存管理**: 

函数在内部分配内存。使用 `tiny_iir_deinit()` 释放它。

### tiny_iir_deinit

```c
/**
 * @name tiny_iir_deinit
 * @brief Deinitialize IIR filter and free allocated memory
 * @param filter Pointer to IIR filter structure
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_deinit(tiny_iir_filter_t *filter);
```

**描述**: 

取消初始化 IIR 滤波器并释放所有分配的内存。

**参数**:

- `filter`: `tiny_iir_filter_t` 结构指针。

**返回值**: 

返回成功或错误代码。

### tiny_iir_process_sample

```c
/**
 * @name tiny_iir_process_sample
 * @brief Process a single sample through IIR filter (real-time)
 * @param filter Pointer to initialized IIR filter structure
 * @param input Input sample value
 * @return Filtered output sample
 */
float tiny_iir_process_sample(tiny_iir_filter_t *filter, float input);
```

**描述**: 

通过 IIR 滤波器处理单个输入样本并返回滤波输出。使用直接型 II 转置结构。

**参数**:

- `filter`: 初始化的 `tiny_iir_filter_t` 结构指针。

- `input`: 输入样本值。

**返回值**: 

返回滤波后的输出样本。

**注意**: 

滤波器在调用之间维护内部状态。使用 `tiny_iir_reset()` 清除状态。

### tiny_iir_reset

```c
/**
 * @name tiny_iir_reset
 * @brief Reset IIR filter state (clear delay line)
 * @param filter Pointer to IIR filter structure
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_reset(tiny_iir_filter_t *filter);
```

**描述**: 

通过清除状态变量来重置 IIR 滤波器状态。在开始新信号或出现不连续后很有用。

**参数**:

- `filter`: 初始化的 `tiny_iir_filter_t` 结构指针。

**返回值**: 

返回成功或错误代码。

## 双二阶滤波器

双二阶（二阶）滤波器是 IIR 滤波器的特殊情况，特别高效且常用。高阶滤波器通常分解为级联双二阶以提高数值稳定性。

### tiny_iir_biquad_init

```c
/**
 * @name tiny_iir_biquad_init
 * @brief Initialize a biquad (second-order) IIR filter
 * @param biquad Pointer to biquad filter structure
 * @param b0 Numerator coefficient b0
 * @param b1 Numerator coefficient b1
 * @param b2 Numerator coefficient b2
 * @param a1 Denominator coefficient a1 (a0 = 1.0)
 * @param a2 Denominator coefficient a2
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_biquad_init(tiny_iir_biquad_t *biquad,
                                    float b0, float b1, float b2,
                                    float a1, float a2);
```

**描述**: 

初始化双二阶（二阶）IIR 滤波器。双二阶是高效且常用的高阶滤波器构建块。

**参数**:

- `biquad`: `tiny_iir_biquad_t` 结构指针。

- `b0`, `b1`, `b2`: 分子系数。

- `a1`, `a2`: 分母系数（a0 = 1.0，归一化形式）。

**返回值**: 

返回成功或错误代码。

### tiny_iir_biquad_process_sample

```c
/**
 * @name tiny_iir_biquad_process_sample
 * @brief Process a single sample through biquad filter (real-time)
 * @param biquad Pointer to initialized biquad filter structure
 * @param input Input sample value
 * @return Filtered output sample
 */
float tiny_iir_biquad_process_sample(tiny_iir_biquad_t *biquad, float input);
```

**描述**: 

通过双二阶滤波器处理单个输入样本并返回滤波输出。

**参数**:

- `biquad`: 初始化的 `tiny_iir_biquad_t` 结构指针。

- `input`: 输入样本值。

**返回值**: 

返回滤波后的输出样本。

### tiny_iir_biquad_reset

```c
/**
 * @name tiny_iir_biquad_reset
 * @brief Reset biquad filter state
 * @param biquad Pointer to biquad filter structure
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_biquad_reset(tiny_iir_biquad_t *biquad);
```

**描述**: 

通过清除内部状态变量来重置双二阶滤波器状态。

**参数**:

- `biquad`: `tiny_iir_biquad_t` 结构指针。

**返回值**: 

返回成功或错误代码。

## 使用流程

### 批处理滤波流程

1. **设计滤波器**:
   ```c
   float b_coeffs[3], a_coeffs[3];
   tiny_iir_design_lowpass(0.1f, 2, TINY_IIR_DESIGN_BUTTERWORTH, 0.0f, b_coeffs, a_coeffs);
   ```

2. **应用滤波器**:
   ```c
   float input[256], output[256];
   tiny_iir_filter_f32(input, 256, b_coeffs, 3, a_coeffs, 3, output, NULL);
   ```

### 实时滤波流程

1. **设计滤波器**:
   ```c
   float b_coeffs[3], a_coeffs[3];
   tiny_iir_design_lowpass(0.1f, 2, TINY_IIR_DESIGN_BUTTERWORTH, 0.0f, b_coeffs, a_coeffs);
   ```

2. **初始化滤波器**:
   ```c
   tiny_iir_filter_t filter;
   tiny_iir_init(&filter, b_coeffs, 3, a_coeffs, 3);
   ```

3. **处理样本**:
   ```c
   for (int i = 0; i < num_samples; i++) {
       float output = tiny_iir_process_sample(&filter, input[i]);
       // 使用输出...
   }
   ```

4. **清理**:
   ```c
   tiny_iir_deinit(&filter);
   ```

### 双二阶流程

1. **设计滤波器**（或使用预设计的系数）:
   ```c
   float b_coeffs[3], a_coeffs[3];
   tiny_iir_design_lowpass(0.1f, 2, TINY_IIR_DESIGN_BUTTERWORTH, 0.0f, b_coeffs, a_coeffs);
   ```

2. **初始化双二阶**:
   ```c
   tiny_iir_biquad_t biquad;
   tiny_iir_biquad_init(&biquad, b_coeffs[0], b_coeffs[1], b_coeffs[2],
                        a_coeffs[1], a_coeffs[2]);
   ```

3. **处理样本**:
   ```c
   for (int i = 0; i < num_samples; i++) {
       float output = tiny_iir_biquad_process_sample(&biquad, input[i]);
       // 使用输出...
   }
   ```

## 应用场景

IIR 滤波器广泛应用于：

- **音频处理**：均衡、音调控制、音频效果
- **控制系统**：信号调理、噪声滤波、反馈控制
- **生物医学**：ECG/EEG 信号调理、伪影去除
- **通信**：信道均衡、降噪
- **传感器信号处理**：降噪、信号调理
- **实时系统**：计算效率至关重要的场景

## 优缺点

### 优点

- **高效**：相同规格下比 FIR 需要更少的系数
- **陡峭过渡**：可以用低阶实现陡峭的频率响应
- **低延迟**：与 FIR 相比群延迟最小
- **内存高效**：比 FIR 需要更少的内存

### 缺点

- **潜在不稳定**：如果极点在单位圆外可能不稳定
- **非线性相位**：通常具有非线性相位响应
- **设计复杂**：设计比 FIR 窗函数法更复杂
- **极限环**：可能出现量化引起的极限环

## 稳定性考虑

对于 IIR 滤波器要稳定，所有极点必须位于 z 平面的单位圆内：

\[
|p_k| < 1 \quad \forall k
\]

其中 \( p_k \) 是传递函数的极点。

**稳定性检查**：

分母多项式 \( A(z) = 1 + \sum_{k=1}^{N} a[k] \cdot z^{-k} \) 的所有根必须在单位圆内。

## 设计考虑

### 滤波器阶数

- **更高阶**：更陡的过渡，更好的阻带衰减，但更复杂
- **更低阶**：更简单，更快，但过渡带更宽
- **Butterworth**：阶数决定 -3 dB 点和阻带衰减

### 归一化频率

记住要归一化频率：

- 在 1 kHz 采样率下 100 Hz 截止：`0.2`（100 / 500）
- 在 10 kHz 采样率下 1 kHz 截止：`0.2`（1000 / 5000）

### 系数归一化

IIR 滤波器使用归一化系数，其中 `a[0] = 1.0`。这是标准做法，简化了实现。

## 注意事项

- IIR 滤波器如果设计不当可能不稳定
- 设计自定义滤波器时始终检查稳定性
- 对于更高阶，分解为级联双二阶以提高数值稳定性
- 使用直接型 II 转置实现高效
- 对于实时应用，使用 `tiny_iir_init()` 和 `tiny_iir_process_sample()`
- 对于批处理，使用 `tiny_iir_filter_f32()`
- 双二阶滤波器推荐用于高阶设计

