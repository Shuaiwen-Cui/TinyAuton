# 说明

!!! note "说明"
    有限脉冲响应（FIR）滤波器是没有反馈的数字滤波器，因此始终稳定。FIR 滤波器可以实现线性相位响应，这对于需要保持相位的应用很重要。它们通过卷积实现，广泛应用于音频处理、通信和信号调理。

## FIR 滤波器概述

### 数学原理

FIR 滤波器由其有限长度的脉冲响应 \( h[n] \) 定义。输出 \( y[n] \) 计算为：

\[
y[n] = \sum_{k=0}^{M-1} h[k] \cdot x[n-k]
\]

其中：

- \( x[n] \) 是输入信号

- \( h[k] \) 是滤波器系数（抽头）

- \( M \) 是滤波器抽头数

- \( y[n] \) 是输出信号

**传递函数**：

\[
H(z) = \sum_{k=0}^{M-1} h[k] \cdot z^{-k}
\]

**关键特性**：

- **始终稳定**：无极点，只有零点

- **线性相位**：使用对称系数可实现

- **有限内存**：仅需要 \( M \) 个过去样本

- **无反馈**：输出仅依赖于输入

## 滤波器类型

库支持四种基本滤波器类型：

- **低通**：通过截止频率以下的频率，衰减以上频率
- **高通**：通过截止频率以上的频率，衰减以下频率
- **带通**：通过频带内的频率，衰减外部频率
- **带阻（陷波）**：衰减频带内的频率，通过外部频率

## 滤波器设计

### 窗函数法

库使用窗函数法进行 FIR 滤波器设计：

1. **生成理想滤波器**：创建理想频率响应
2. **应用窗函数**：乘以窗函数以减少吉布斯现象
3. **截断**：限制为有限数量的抽头

**支持的窗函数**：

- **矩形**：无窗（最快，但可能有振铃）

- **汉明（Hamming）**：主瓣宽度和旁瓣抑制的良好平衡

- **汉宁（Hanning）**：与汉明类似，旁瓣抑制稍好

- **布莱克曼（Blackman）**：最佳旁瓣抑制，主瓣更宽

**窗函数选择指南**：

- **汉明**：通用，良好平衡

- **汉宁**：旁瓣抑制优于汉明

- **布莱克曼**：最适合需要低旁瓣的应用

- **矩形**：仅用于非常简单的应用

### 设计参数

- **截止频率**：归一化频率（0.0 到 0.5，其中 0.5 = 奈奎斯特频率）
- **抽头数**：应为奇数以实现线性相位（I 型滤波器）
- **窗函数类型**：影响过渡带宽和旁瓣水平

**归一化频率**：

\[
f_{norm} = \frac{f_{cutoff}}{f_s / 2}
\]

其中 \( f_s \) 是采样率。

## 滤波器设计函数

### tiny_fir_design_lowpass

```c
/**
 * @name tiny_fir_design_lowpass
 * @brief Design a low-pass FIR filter using window method
 * @param cutoff_freq Normalized cutoff frequency (0.0 to 0.5)
 * @param num_taps Number of filter taps (should be odd)
 * @param window Window function to use
 * @param coefficients Output array for filter coefficients
 * @return tiny_error_t
 */
tiny_error_t tiny_fir_design_lowpass(float cutoff_freq, int num_taps,
                                     tiny_fir_window_t window,
                                     float *coefficients);
```

**描述**: 

使用窗函数法设计低通 FIR 滤波器。生成理想低通滤波器的脉冲响应，然后加窗以减少吉布斯现象。

**参数**:

- `cutoff_freq`: 归一化截止频率（0.0 到 0.5，其中 0.5 = 奈奎斯特频率）。

- `num_taps`: 滤波器抽头数。必须为奇数以实现线性相位响应。

- `window`: 来自 `tiny_fir_window_t` 枚举的窗函数类型。

- `coefficients`: 滤波器系数的输出数组。大小必须至少为 `num_taps`。

**返回值**: 

返回成功或错误代码。

**注意**: 

截止频率已归一化：`cutoff_freq = actual_freq / (sample_rate / 2)`。例如，在 1 kHz 采样率下 100 Hz 截止频率将为 `0.2`（100 / 500）。

### tiny_fir_design_highpass

```c
/**
 * @name tiny_fir_design_highpass
 * @brief Design a high-pass FIR filter using window method
 * @param cutoff_freq Normalized cutoff frequency (0.0 to 0.5)
 * @param num_taps Number of filter taps (should be odd)
 * @param window Window function to use
 * @param coefficients Output array for filter coefficients
 * @return tiny_error_t
 */
tiny_error_t tiny_fir_design_highpass(float cutoff_freq, int num_taps,
                                      tiny_fir_window_t window,
                                      float *coefficients);
```

**描述**: 

使用窗函数法设计高通 FIR 滤波器。生成理想高通滤波器的脉冲响应，然后加窗。

**参数**:

- `cutoff_freq`: 归一化截止频率（0.0 到 0.5）。

- `num_taps`: 滤波器抽头数。必须为奇数。

- `window`: 窗函数类型。

- `coefficients`: 滤波器系数的输出数组。大小必须至少为 `num_taps`。

**返回值**: 

返回成功或错误代码。

### tiny_fir_design_bandpass

```c
/**
 * @name tiny_fir_design_bandpass
 * @brief Design a band-pass FIR filter using window method
 * @param low_freq Lower cutoff frequency (normalized, 0.0 to 0.5)
 * @param high_freq Upper cutoff frequency (normalized, 0.0 to 0.5)
 * @param num_taps Number of filter taps (should be odd)
 * @param window Window function to use
 * @param coefficients Output array for filter coefficients
 * @return tiny_error_t
 */
tiny_error_t tiny_fir_design_bandpass(float low_freq, float high_freq,
                                      int num_taps,
                                      tiny_fir_window_t window,
                                      float *coefficients);
```

**描述**: 

使用窗函数法设计带通 FIR 滤波器。滤波器通过 `low_freq` 和 `high_freq` 之间的频率。

**参数**:

- `low_freq`: 下截止频率（归一化，0.0 到 0.5）。必须小于 `high_freq`。

- `high_freq`: 上截止频率（归一化，0.0 到 0.5）。必须大于 `low_freq`。

- `num_taps`: 滤波器抽头数。必须为奇数。

- `window`: 窗函数类型。

- `coefficients`: 滤波器系数的输出数组。大小必须至少为 `num_taps`。

**返回值**: 

返回成功或错误代码。

### tiny_fir_design_bandstop

```c
/**
 * @name tiny_fir_design_bandstop
 * @brief Design a band-stop (notch) FIR filter using window method
 * @param low_freq Lower cutoff frequency (normalized, 0.0 to 0.5)
 * @param high_freq Upper cutoff frequency (normalized, 0.0 to 0.5)
 * @param num_taps Number of filter taps (should be odd)
 * @param window Window function to use
 * @param coefficients Output array for filter coefficients
 * @return tiny_error_t
 */
tiny_error_t tiny_fir_design_bandstop(float low_freq, float high_freq,
                                      int num_taps,
                                      tiny_fir_window_t window,
                                      float *coefficients);
```

**描述**: 

使用窗函数法设计带阻（陷波）FIR 滤波器。滤波器衰减 `low_freq` 和 `high_freq` 之间的频率。

**参数**:

- `low_freq`: 下截止频率（归一化，0.0 到 0.5）。必须小于 `high_freq`。

- `high_freq`: 上截止频率（归一化，0.0 到 0.5）。必须大于 `low_freq`。

- `num_taps`: 滤波器抽头数。必须为奇数。

- `window`: 窗函数类型。

- `coefficients`: 滤波器系数的输出数组。大小必须至少为 `num_taps`。

**返回值**: 

返回成功或错误代码。

## 滤波器应用

### 批处理

### tiny_fir_filter_f32

```c
/**
 * @name tiny_fir_filter_f32
 * @brief Apply FIR filter to a signal (batch processing)
 * @param input Input signal array
 * @param input_len Length of input signal
 * @param coefficients FIR filter coefficients (taps)
 * @param num_taps Number of filter taps
 * @param output Output filtered signal array
 * @param padding_mode Padding mode for boundary handling
 * @return tiny_error_t
 */
tiny_error_t tiny_fir_filter_f32(const float *input, int input_len,
                                  const float *coefficients, int num_taps,
                                  float *output,
                                  tiny_padding_mode_t padding_mode);
```

**描述**: 

使用卷积将 FIR 滤波器应用于整个信号。适用于整个信号可用的批处理。

**特点**:

- 内部使用卷积

- 支持不同的填充模式以处理边界

- 输出长度等于输入长度

**参数**:

- `input`: 输入信号数组指针。

- `input_len`: 输入信号长度。

- `coefficients`: FIR 滤波器系数（抽头）指针。

- `num_taps`: 滤波器抽头数。

- `output`: 滤波信号的输出数组指针。大小必须至少为 `input_len`。

- `padding_mode`: 边界处理的填充模式（例如，`TINY_PADDING_SYMMETRIC`）。

**返回值**: 

返回成功或错误代码。

### 实时处理

### tiny_fir_init

```c
/**
 * @name tiny_fir_init
 * @brief Initialize FIR filter structure for real-time filtering
 * @param filter Pointer to FIR filter structure
 * @param coefficients Filter coefficients (will be copied internally)
 * @param num_taps Number of filter taps
 * @return tiny_error_t
 */
tiny_error_t tiny_fir_init(tiny_fir_filter_t *filter,
                            const float *coefficients, int num_taps);
```

**描述**: 

初始化 FIR 滤波器结构以进行实时逐样本处理。为系数和延迟线分配内存。

**参数**:

- `filter`: `tiny_fir_filter_t` 结构指针。

- `coefficients`: 滤波器系数指针。将在内部复制。

- `num_taps`: 滤波器抽头数。

**返回值**: 

返回成功或错误代码。

**内存管理**: 

函数在内部分配内存。使用 `tiny_fir_deinit()` 释放它。

### tiny_fir_deinit

```c
/**
 * @name tiny_fir_deinit
 * @brief Deinitialize FIR filter and free allocated memory
 * @param filter Pointer to FIR filter structure
 * @return tiny_error_t
 */
tiny_error_t tiny_fir_deinit(tiny_fir_filter_t *filter);
```

**描述**: 

取消初始化 FIR 滤波器并释放所有分配的内存。

**参数**:

- `filter`: `tiny_fir_filter_t` 结构指针。

**返回值**: 

返回成功或错误代码。

### tiny_fir_process_sample

```c
/**
 * @name tiny_fir_process_sample
 * @brief Process a single sample through FIR filter (real-time)
 * @param filter Pointer to initialized FIR filter structure
 * @param input Input sample value
 * @return Filtered output sample
 */
float tiny_fir_process_sample(tiny_fir_filter_t *filter, float input);
```

**描述**: 

通过 FIR 滤波器处理单个输入样本并返回滤波输出。使用循环缓冲区实现高效的延迟线。

**参数**:

- `filter`: 初始化的 `tiny_fir_filter_t` 结构指针。

- `input`: 输入样本值。

**返回值**: 

返回滤波后的输出样本。

**注意**: 

滤波器在调用之间维护内部状态（延迟线）。使用 `tiny_fir_reset()` 清除状态。

### tiny_fir_reset

```c
/**
 * @name tiny_fir_reset
 * @brief Reset FIR filter state (clear delay line)
 * @param filter Pointer to FIR filter structure
 * @return tiny_error_t
 */
tiny_error_t tiny_fir_reset(tiny_fir_filter_t *filter);
```

**描述**: 

通过清除延迟线来重置 FIR 滤波器状态。在开始新信号或出现不连续后很有用。

**参数**:

- `filter`: 初始化的 `tiny_fir_filter_t` 结构指针。

**返回值**: 

返回成功或错误代码。

## 使用流程

### 批处理滤波流程

1. **设计滤波器**:
   ```c
   float coeffs[51];
   tiny_fir_design_lowpass(0.1f, 51, TINY_FIR_WINDOW_HAMMING, coeffs);
   ```

2. **应用滤波器**:
   ```c
   float input[256], output[256];
   tiny_fir_filter_f32(input, 256, coeffs, 51, output, TINY_PADDING_SYMMETRIC);
   ```

### 实时滤波流程

1. **设计滤波器**:
   ```c
   float coeffs[21];
   tiny_fir_design_lowpass(0.1f, 21, TINY_FIR_WINDOW_HAMMING, coeffs);
   ```

2. **初始化滤波器**:
   ```c
   tiny_fir_filter_t filter;
   tiny_fir_init(&filter, coeffs, 21);
   ```

3. **处理样本**:
   ```c
   for (int i = 0; i < num_samples; i++) {
       float output = tiny_fir_process_sample(&filter, input[i]);
       // 使用输出...
   }
   ```

4. **清理**:
   ```c
   tiny_fir_deinit(&filter);
   ```

## 应用场景

FIR 滤波器广泛应用于：

- **音频处理**：均衡、降噪、抗混叠
- **通信**：脉冲整形、匹配滤波、信道均衡
- **生物医学**：ECG/EEG 信号调理、伪影去除
- **控制系统**：信号调理、噪声滤波
- **图像处理**：边缘检测、平滑、锐化
- **传感器信号处理**：降噪、信号调理

## 优缺点

### 优点

- **始终稳定**：无反馈，保证稳定性
- **线性相位**：可以实现精确的线性相位响应
- **设计简单**：窗函数法简单直接
- **无极限环**：无量化引起的振荡

### 缺点

- **计算成本较高**：相同规格下比 IIR 需要更多抽头
- **延迟更长**：群延迟与滤波器长度成正比
- **内存需求**：需要存储所有滤波器抽头

## 设计考虑

### 抽头数

- **更多抽头**：更陡的过渡，更好的阻带衰减，但计算量更大
- **更少抽头**：计算更快，但过渡带更宽
- **经验法则**：过渡带宽 ≈ 4 / num_taps（对于汉明窗）

### 窗函数选择

- **汉明**：良好的通用选择
- **汉宁**：旁瓣抑制优于汉明
- **布莱克曼**：最佳旁瓣抑制，过渡带更宽
- **矩形**：仅用于非常简单的应用（不推荐）

### 归一化频率

记住要归一化频率：

- 在 1 kHz 采样率下 100 Hz 截止：`0.2`（100 / 500）

- 在 10 kHz 采样率下 1 kHz 截止：`0.2`（1000 / 5000）

## 注意事项

- FIR 滤波器始终稳定（无极点）
- 线性相位需要奇数个抽头和对称系数
- 窗函数法简单，但可能不是所有应用的最佳选择
- 对于实时应用，使用 `tiny_fir_init()` 和 `tiny_fir_process_sample()`
- 对于批处理，使用 `tiny_fir_filter_f32()`

