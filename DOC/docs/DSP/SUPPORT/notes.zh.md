# 说明

!!! note "说明"
    支持模块为信号处理提供可视化和分析工具。这些函数帮助开发者可视化信号、分析数据并调试 DSP 算法，通过提供基于 ASCII 的图表和格式化输出。这在可能没有图形显示的嵌入式系统中特别有用。

## 概述

支持模块包括四个主要函数：

1. **信号可视化**：以 ASCII 格式绘制信号（类似示波器）
2. **频谱可视化**：可视化功率频谱，带频率轴标签
3. **数组打印**：以格式化表格打印数组
4. **统计信息**：计算并显示信号的统计信息

## 信号可视化

### tiny_view_signal_f32

```c
/**
 * @name: tiny_view_signal_f32
 * @brief Visualize a signal in ASCII format (like oscilloscope)
 * @param data Input signal array
 * @param len Length of the signal
 * @param width Width of the plot in characters (default: 64)
 * @param height Height of the plot in lines (default: 16)
 * @param min Minimum Y-axis value (auto-detect if min == max)
 * @param max Maximum Y-axis value (auto-detect if min == max)
 * @param title Optional title for the plot (NULL for no title)
 * @return tiny_error_t
 */
tiny_error_t tiny_view_signal_f32(const float *data, int len, int width, int height, float min, float max, const char *title);
```

**描述**: 

以 ASCII 格式可视化信号，类似于示波器显示。该函数创建一个基于字符的图表，显示信号波形。

**特点**:

- 高分辨率绘制，数据点之间线性插值

- 当 min == max 时自动检测 Y 轴范围

- 可自定义图表尺寸（宽度和高度）

- 可选的标题显示

- Y 轴标签显示值范围

- X 轴标签显示采样索引

**参数**:

- `data`: 输入信号数组指针。

- `len`: 信号数组长度。

- `width`: 图表宽度（字符数，通常为 64）。

- `height`: 图表高度（行数，通常为 16）。

- `min`: Y 轴最小值。如果 `min == max`，函数将自动检测范围。

- `max`: Y 轴最大值。如果 `min == max`，函数将自动检测范围。

- `title`: 图表的可选标题字符串。传递 `NULL` 表示无标题。

**返回值**: 

返回成功或错误代码。

**输出格式**:

函数打印：

- 标题（如果提供）

- 带值范围的 Y 轴标签

- 用 '*' 字符表示信号的 ASCII 图表

- 带采样索引标签的 X 轴

- 包含值范围和信号长度的摘要行

**示例输出**:

```
Test Signal: 10 Hz Sine Wave
Value
  1.20 |                                        
  1.00 |                                        
  0.80 |                                        
  0.60 |                                        
  0.40 |                                        
  0.20 |                                        
  0.00 |                                        
 -0.20 |                                        
 -0.40 |                                        
 -0.60 |                                        
 -0.80 |                                        
 -1.00 |                                        
 -1.20 |                                        
       ------------------------------------------------------------------------
       0        8       16      24      32      40      48      56 (Sample Index)
Range: [-1.200, 1.200], Length: 64
```

## 频谱可视化

### tiny_view_spectrum_f32

```c
/**
 * @name: tiny_view_spectrum_f32
 * @brief Visualize power spectrum in ASCII format (optimized for frequency domain)
 * @param power_spectrum Power spectrum array
 * @param len Length of the spectrum
 * @param sample_rate Sampling rate (Hz) for frequency axis labels
 * @param title Optional title for the plot (NULL for no title)
 * @return tiny_error_t
 */
tiny_error_t tiny_view_spectrum_f32(const float *power_spectrum, int len, float sample_rate, const char *title);
```

**描述**: 

以 ASCII 格式可视化功率频谱，带频率轴标签。该函数创建一个条形图，显示不同频率处的功率。

**特点**:

- 条形图可视化（垂直条形）

- 基于采样率自动频率轴标签

- 针对频域数据优化（使用频谱的前半部分）

- 频率标签以 Hz 为单位

- 奈奎斯特频率指示

**参数**:

- `power_spectrum`: 功率频谱数组指针。

- `len`: 功率频谱数组长度。

- `sample_rate`: 原始信号的采样率（Hz），用于频率轴标签。

- `title`: 图表的可选标题字符串。传递 `NULL` 表示无标题。

**返回值**: 

返回成功或错误代码。

**输出格式**:

函数打印：

- 标题（如果提供）

- 带功率值的 Y 轴标签

- 用 '|' 字符的 ASCII 条形图

- 带频率标签（Hz）的 X 轴

- 包含值范围和奈奎斯特频率的摘要行

**注意**: 

函数假设功率频谱长度是 FFT 长度的一半（实信号的典型情况）。频率标签计算为：`freq = index * sample_rate / (2 * len)`。

## 数组打印

### tiny_view_array_f32

```c
/**
 * @name: tiny_view_array_f32
 * @brief Print array values in a formatted table
 * @param data Input array
 * @param len Length of the array
 * @param name Name/label for the array
 * @param precision Number of decimal places (default: 3)
 * @param items_per_line Number of items per line (default: 8)
 * @return tiny_error_t
 */
tiny_error_t tiny_view_array_f32(const float *data, int len, const char *name, int precision, int items_per_line);
```

**描述**: 

以格式化表格打印数组值，可自定义精度和每行项目数。

**特点**:

- 带索引标签的格式化表格输出

- 可自定义精度（小数位数）

- 可自定义每行项目数

- 可选的数组名称/标签

**参数**:

- `data`: 输入数组指针。

- `len`: 数组长度。

- `name`: 数组的可选名称/标签。传递 `NULL` 表示默认标签。

- `precision`: 要显示的小数位数。如果为负数，默认为 3。

- `items_per_line`: 每行打印的项目数。如果 ≤ 0，默认为 8。

**返回值**: 

返回成功或错误代码。

**输出格式**:

函数打印：


- 数组名称和长度

- 带索引标签和值的格式化表格

**示例输出**:

```
Test Signal [64 elements]:
  [  0] 0.000  0.063  0.125  0.188  0.250  0.313  0.375  0.438 
  [  8] 0.500  0.563  0.625  0.688  0.750  0.813  0.875  0.938 
  ...
```

## 统计信息

### tiny_view_statistics_f32

```c
/**
 * @name: tiny_view_statistics_f32
 * @brief Print statistical information about a signal
 * @param data Input signal array
 * @param len Length of the signal
 * @param name Name/label for the signal
 * @return tiny_error_t
 */
tiny_error_t tiny_view_statistics_f32(const float *data, int len, const char *name);
```

**描述**: 

计算并打印信号的统计信息，包括最小值、最大值、均值、标准差、方差和峰值。

**特点**:

- 单次遍历计算（高效）

- 全面的统计信息：
  - 带索引的最小值和最大值
  - 峰值（绝对最大值）及索引
  - 均值（平均值）
  - 标准差
  - 方差
  - 范围（最大值 - 最小值）

- 可选的信号名称/标签

**参数**:

- `data`: 输入信号数组指针。

- `len`: 信号数组长度。

- `name`: 信号的可选名称/标签。传递 `NULL` 表示默认标签。

**返回值**: 

返回成功或错误代码。

**输出格式**:

函数打印：

- 带信号名称的统计信息标题

- 格式化表格中的所有计算统计信息

**示例输出**:

```
=== Statistics: Test Signal ===
  Length:     64 samples
  Min:        -1.200000 (at index 48)
  Max:         1.200000 (at index 16)
  Peak:        1.200000 (at index 16)
  Mean:        0.000000
  Std Dev:     0.707107
  Variance:    0.500000
  Range:       2.400000
========================
```

**数学公式**:

- **均值**：\( \mu = \frac{1}{N} \sum_{i=0}^{N-1} x[i] \)

- **方差**：\( \sigma^2 = \frac{1}{N} \sum_{i=0}^{N-1} x[i]^2 - \mu^2 \)

- **标准差**：\( \sigma = \sqrt{\sigma^2} \)

- **范围**：\( \text{range} = \max(x) - \min(x) \)

## 使用流程

### 典型的可视化流程

1. **可视化信号**:
   ```c
   float signal[64];
   // ... 填充信号数据 ...
   tiny_view_signal_f32(signal, 64, 64, 16, 0, 0, "我的信号");
   ```

2. **打印数组**:
   ```c
   tiny_view_array_f32(signal, 64, "信号数据", 3, 8);
   ```

3. **显示统计信息**:
   ```c
   tiny_view_statistics_f32(signal, 64, "信号");
   ```

4. **可视化频谱**:
   ```c
   float power[128];
   // ... 计算功率频谱 ...
   tiny_view_spectrum_f32(power, 128, 1000.0f, "功率频谱");
   ```

## 应用场景

支持模块适用于：

- **调试**：在算法开发过程中可视化信号

- **分析**：快速统计信号分析

- **教育**：演示信号处理概念

- **嵌入式系统**：在没有图形显示的情况下调试 DSP 算法

- **测试**：验证信号处理结果

- **文档**：为文档生成 ASCII 图表

## 注意事项

- 所有可视化函数使用 `printf` 输出到 `stdout`

- ASCII 图表设计用于等宽字体

- 为获得最佳效果，使用至少 80 字符宽度的终端

- 信号可视化使用线性插值以获得平滑图表

- 频谱可视化使用针对频域数据优化的条形图
