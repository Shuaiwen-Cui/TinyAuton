# 说明

!!! note "说明"
    快速傅里叶变换（FFT）是信号处理中的基础算法，用于高效计算离散傅里叶变换（DFT）。它将信号从时域转换到频域，实现频率分析、频谱分析和滤波操作。FFT广泛应用于音频处理、通信、结构健康监测和许多其他应用。

## FFT 概述

### 数学原理

长度为 \( N \) 的序列 \( x[n] \) 的离散傅里叶变换（DFT）定义为：

\[
X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j \frac{2\pi kn}{N}}
\]

其中：

- \( x[n] \) 是时域输入信号

- \( X[k] \) 是频域输出

- \( N \) 是信号长度（必须是2的幂）

- \( k \in [0, N-1] \) 是频率箱索引

**频率分辨率**：

\[
\Delta f = \frac{f_s}{N}
\]

其中：

- \( \Delta f \) 是频率分辨率（Hz）

- \( f_s \) 是采样率（Hz）

- \( N \) 是 FFT 大小

**第 k 个频率箱的频率**：

\[
f_k = k \cdot \frac{f_s}{N}
\]

## 窗函数

窗函数在 FFT 之前应用于信号，以减少频谱泄漏。库支持多种窗类型：

- **无（矩形窗）**：不应用窗函数，最快但可能有频谱泄漏

- **汉宁窗（Hanning）**：良好的通用窗，平衡频率分辨率和泄漏减少

- **汉明窗（Hamming）**：与汉宁窗类似，旁瓣抑制稍好

- **布莱克曼窗（Blackman）**：最佳旁瓣抑制，但主瓣更宽

## 初始化和反初始化

### tiny_fft_init

```c
/**
 * @name: tiny_fft_init
 * @brief Initialize FFT tables (required before using FFT functions)
 * @note This function should be called once at startup
 * @param fft_size Maximum FFT size to support (must be power of 2)
 * @return tiny_error_t
 */
tiny_error_t tiny_fft_init(int fft_size);
```

**描述**: 

初始化 FFT 表并准备库以进行 FFT 操作。在使用任何 FFT 操作之前必须调用此函数。

**特点**:

- 支持平台加速（ESP32 使用优化的 DSP 库）。

- 在使用任何 FFT 函数之前必须调用一次。

**参数**:

- `fft_size`: 支持的最大 FFT 大小。必须是2的幂（例如，256、512、1024）。

**返回值**: 

返回成功或错误代码。

**重要说明**:

- FFT 大小必须是2的幂。

- 此函数应在系统启动时调用一次。

- 所有后续的 FFT 操作必须使用大小 ≤ `fft_size`。

### tiny_fft_deinit

```c
/**
 * @name: tiny_fft_deinit
 * @brief Deinitialize FFT tables and free resources
 * @return tiny_error_t
 */
tiny_error_t tiny_fft_deinit(void);
```

**描述**: 

反初始化 FFT 表并释放分配的资源。

**返回值**: 

返回成功或错误代码。

## 正向 FFT

### tiny_fft_f32

```c
/**
 * @name: tiny_fft_f32
 * @brief Perform FFT on real-valued input signal
 * @param input Input signal array (real values)
 * @param input_len Length of input signal (must be power of 2)
 * @param output_fft Output FFT result (complex array: [Re0, Im0, Re1, Im1, ...])
 *                   Size must be at least input_len * 2
 * @param window Window function to apply before FFT (optional)
 * @return tiny_error_t
 */
tiny_error_t tiny_fft_f32(const float *input, int input_len, float *output_fft, tiny_fft_window_t window);
```

**描述**: 

对实值输入信号执行快速傅里叶变换，将其从时域转换到频域。

**特点**:

- 支持平台加速。

- 支持可选的窗函数以减少频谱泄漏。

- 输出为复数格式：`[Re0, Im0, Re1, Im1, ...]`。

**参数**:

- `input`: 输入信号数组指针（实数值）。

- `input_len`: 输入信号长度。必须是2的幂且 ≤ 初始化的 FFT 大小。

- `output_fft`: FFT 结果的输出数组指针。大小必须至少为 `input_len * 2`（复数格式）。

- `window`: 在 FFT 之前应用的窗函数类型。选项：
  - `TINY_FFT_WINDOW_NONE`: 无窗（矩形窗）
  - `TINY_FFT_WINDOW_HANNING`: 汉宁窗
  - `TINY_FFT_WINDOW_HAMMING`: 汉明窗
  - `TINY_FFT_WINDOW_BLACKMAN`: 布莱克曼窗

**返回值**: 

返回成功或错误代码。

**输出格式**:

输出存储为交错的复数数组：

- `output_fft[0]` = 箱0的实部

- `output_fft[1]` = 箱0的虚部

- `output_fft[2]` = 箱1的实部

- `output_fft[3]` = 箱1的虚部

- ...

**频率箱**:

- 箱0：直流分量（0 Hz）

- 箱k：频率 = \( k \cdot \frac{f_s}{N} \) Hz

- 箱N/2：奈奎斯特频率（\( f_s/2 \) Hz）

- 箱N/2+1 到 N-1：箱1 到 N/2-1 的镜像（对于实信号）

## 逆 FFT

### tiny_fft_ifft_f32

```c
/**
 * @name: tiny_fft_ifft_f32
 * @brief Perform inverse FFT to reconstruct time-domain signal
 * @param input_fft Input FFT array (complex: [Re0, Im0, Re1, Im1, ...])
 * @param fft_len Length of FFT (number of complex points)
 * @param output Output reconstructed signal (real values)
 *               Size must be at least fft_len
 * @return tiny_error_t
 */
tiny_error_t tiny_fft_ifft_f32(const float *input_fft, int fft_len, float *output);
```

**描述**: 

执行逆快速傅里叶变换，将频域信号转换回时域。

**特点**:

- 支持平台加速。

- 从 FFT 结果重建原始时域信号。

**参数**:

- `input_fft`: 输入 FFT 数组指针（复数格式：`[Re0, Im0, Re1, Im1, ...]`）。

- `fft_len`: FFT 长度（复数点数）。必须是2的幂。

- `output`: 重建信号的输出数组指针（实数值）。大小必须至少为 `fft_len`。

**返回值**: 

返回成功或错误代码。

**注意**: 

重建的信号应与原始输入信号匹配（在数值精度范围内），假设未对 FFT 结果进行修改。

## 频谱分析

### tiny_fft_magnitude_f32

```c
/**
 * @name: tiny_fft_magnitude_f32
 * @brief Calculate magnitude spectrum from FFT result
 * @param fft_result FFT result (complex array: [Re0, Im0, Re1, Im1, ...])
 * @param fft_len Length of FFT (number of complex points)
 * @param magnitude Output magnitude spectrum (real values)
 *                  Size must be at least fft_len
 * @return tiny_error_t
 */
tiny_error_t tiny_fft_magnitude_f32(const float *fft_result, int fft_len, float *magnitude);
```

**描述**: 

从 FFT 结果计算幅度频谱。幅度表示每个频率分量的振幅。

**数学公式**:

\[
|X[k]| = \sqrt{\text{Re}[X[k]]^2 + \text{Im}[X[k]]^2}
\]

**参数**:

- `fft_result`: FFT 结果数组指针（复数格式）。

- `fft_len`: FFT 长度（复数点数）。

- `magnitude`: 幅度频谱的输出数组指针。大小必须至少为 `fft_len`。

**返回值**: 

返回成功或错误代码。

### tiny_fft_power_spectrum_f32

```c
/**
 * @name: tiny_fft_power_spectrum_f32
 * @brief Calculate power spectrum density (PSD) from FFT result
 * @param fft_result FFT result (complex array: [Re0, Im0, Re1, Im1, ...])
 * @param fft_len Length of FFT (number of complex points)
 * @param power Output power spectrum (real values)
 *              Size must be at least fft_len
 * @return tiny_error_t
 */
tiny_error_t tiny_fft_power_spectrum_f32(const float *fft_result, int fft_len, float *power);
```

**描述**: 

从 FFT 结果计算功率谱密度（PSD）。功率频谱表示每个频率分量的功率，并按 FFT 长度归一化。

**数学公式**:

\[
P[k] = \frac{|X[k]|^2}{N} = \frac{\text{Re}[X[k]]^2 + \text{Im}[X[k]]^2}{N}
\]

**参数**:

- `fft_result`: FFT 结果数组指针（复数格式）。

- `fft_len`: FFT 长度（复数点数）。

- `power`: 功率频谱的输出数组指针。大小必须至少为 `fft_len`。

**返回值**: 

返回成功或错误代码。

## 频率检测

### tiny_fft_find_peak_frequency

```c
/**
 * @name: tiny_fft_find_peak_frequency
 * @brief Find the frequency with maximum power (useful for structural health monitoring)
 * @param power_spectrum Power spectrum array
 * @param fft_len Length of power spectrum
 * @param sample_rate Sampling rate of the original signal (Hz)
 * @param peak_freq Output peak frequency (Hz)
 * @param peak_power Output peak power value
 * @return tiny_error_t
 */
tiny_error_t tiny_fft_find_peak_frequency(const float *power_spectrum, int fft_len, float sample_rate, float *peak_freq, float *peak_power);
```

**描述**: 

在功率频谱中找到功率最大的频率。使用抛物线插值以提高频率估计精度。

**特点**:

- 跳过直流分量（箱0）。

- 使用抛物线插值以提高频率估计精度。

- 用于检测信号中的主导频率。

**参数**:

- `power_spectrum`: 功率频谱数组指针。

- `fft_len`: 功率频谱长度。

- `sample_rate`: 原始信号的采样率（Hz）。

- `peak_freq`: 峰值频率的输出变量指针（Hz）。

- `peak_power`: 峰值功率值的输出变量指针。

**返回值**: 

返回成功或错误代码。

### tiny_fft_find_top_frequencies

```c
/**
 * @name: tiny_fft_find_top_frequencies
 * @brief Find top N frequencies with highest power
 * @param power_spectrum Power spectrum array
 * @param fft_len Length of power spectrum
 * @param sample_rate Sampling rate of the original signal (Hz)
 * @param top_n Number of top frequencies to find
 * @param frequencies Output array for frequencies (Hz), size must be at least top_n
 * @param powers Output array for power values, size must be at least top_n
 * @return tiny_error_t
 */
tiny_error_t tiny_fft_find_top_frequencies(const float *power_spectrum, int fft_len, float sample_rate, int top_n, float *frequencies, float *powers);
```

**描述**: 

在功率频谱中找到功率最高的前 N 个频率。自动检测局部峰值并合并附近的峰值，以避免从同一频率峰值选择多个箱。

**特点**:

- 检测功率频谱中的局部峰值。

- 合并附近的峰值（2个箱内）以避免重复。

- 使用抛物线插值以提高频率精度。

- 过滤掉不显著的峰值（低于最大功率的1%）。

**参数**:

- `power_spectrum`: 功率频谱数组指针。

- `fft_len`: 功率频谱长度。

- `sample_rate`: 原始信号的采样率（Hz）。

- `top_n`: 要查找的前 N 个频率数量。

- `frequencies`: 频率的输出数组指针（Hz）。大小必须至少为 `top_n`。

- `powers`: 功率值的输出数组指针。大小必须至少为 `top_n`。

**返回值**: 

返回成功或错误代码。

**注意**: 

如果找到的峰值少于 `top_n` 个，输出数组中剩余条目将设置为零。

## 使用流程

### 典型的 FFT 分析流程

1. **初始化 FFT**:
   ```c
   tiny_fft_init(256);  // 初始化最大256点 FFT
   ```

2. **执行 FFT**:
   ```c
   float input[256];
   float fft_result[512];  // 复数输出：256 * 2
   tiny_fft_f32(input, 256, fft_result, TINY_FFT_WINDOW_HANNING);
   ```

3. **计算功率频谱**:
   ```c
   float power[256];
   tiny_fft_power_spectrum_f32(fft_result, 256, power);
   ```

4. **查找峰值频率**:
   ```c
   float peak_freq, peak_power;
   tiny_fft_find_peak_frequency(power, 256, 1000.0f, &peak_freq, &peak_power);
   ```

5. **反初始化**（完成后）:
   ```c
   tiny_fft_deinit();
   ```

## 应用场景

FFT 广泛应用于各种应用：

- **音频处理**：频率分析、均衡、音调检测

- **通信**：信号调制、解调、信道分析

- **结构健康监测**：振动分析、共振检测

- **生物医学**：ECG/EEG 分析、心率检测

- **图像处理**：2D FFT 用于图像滤波和分析

- **频谱分析**：识别信号中的频率分量
