# 说明

!!! note "说明"
    离散小波变换（DWT）是一种强大的信号处理技术，可在多个分辨率级别将信号分解为不同的频率分量。与提供全局频率信息的 FFT 不同，DWT 提供时间和频率的局部化，使其非常适合分析非平稳信号、去噪、压缩和特征提取。

## DWT 概述

### 数学原理

离散小波变换使用一对滤波器将信号分解为近似（低频）和细节（高频）系数：低通滤波器（尺度函数）和高通滤波器（小波函数）。

**单级分解**：

\[
cA[n] = \sum_{k} x[k] \cdot h_0[2n - k]
\]

\[
cD[n] = \sum_{k} x[k] \cdot h_1[2n - k]
\]

其中：

- \( x[k] \) 是输入信号

- \( h_0 \) 是低通分解滤波器

- \( h_1 \) 是高通分解滤波器

- \( cA[n] \) 是近似系数（低频）

- \( cD[n] \) 是细节系数（高频）

**输出长度**：

\[
L_{cA} = L_{cD} = \left\lceil \frac{L_{input}}{2} \right\rceil
\]

**重构**：

\[
x[n] = \sum_{k} (cA[k] \cdot g_0[n - 2k] + cD[k] \cdot g_1[n - 2k])
\]

其中：

- \( g_0 \) 是低通重构滤波器

- \( g_1 \) 是高通重构滤波器

## 小波类型

库支持 Daubechies 小波（DB1 到 DB10）：

- **DB1 (Haar)**：最简单的小波，2 抽头滤波器，适合边缘检测
- **DB2**：4 抽头滤波器，频率分辨率优于 DB1
- **DB3**：6 抽头滤波器，比 DB2 更平滑
- **DB4**：8 抽头滤波器，常用，平衡性好
- **DB5-DB10**：更高阶小波，频率分辨率更好但滤波器更长

**滤波器长度**：

\[
L_{filter} = 2 \times N
\]

其中 \( N \) 是小波阶数（DB1: N=1, DB2: N=2, ..., DB10: N=10）。

## 单级 DWT

### tiny_dwt_decompose_f32

```c
/**
 * @name tiny_dwt_decompose_f32
 * @brief Perform single-level discrete wavelet decomposition
 * @param input Input signal array
 * @param input_len Length of the input signal
 * @param wavelet Wavelet type (DB1-DB10)
 * @param cA Output array for approximation coefficients
 * @param cD Output array for detail coefficients
 * @param cA_len Output length of approximation coefficients
 * @param cD_len Output length of detail coefficients
 * @return tiny_error_t
 */
tiny_error_t tiny_dwt_decompose_f32(const float *input, int input_len,
                                    tiny_wavelet_type_t wavelet,
                                    float *cA, float *cD,
                                    int *cA_len, int *cD_len);
```

**描述**: 

执行单级离散小波分解，将输入信号分解为近似（低频）和细节（高频）系数。

**特点**:

- 使用对称填充处理边界

- 执行低通和高通滤波器卷积

- 以 2 为因子下采样以保持临界采样

- 支持所有 Daubechies 小波（DB1-DB10）

**参数**:

- `input`: 输入信号数组指针。

- `input_len`: 输入信号长度。

- `wavelet`: 来自 `tiny_wavelet_type_t` 枚举的小波类型：
  - `TINY_WAVELET_DB1` 到 `TINY_WAVELET_DB10`

- `cA`: 近似系数输出数组指针。大小应至少为 `(input_len + 1) / 2`。

- `cD`: 细节系数输出数组指针。大小应至少为 `(input_len + 1) / 2`。

- `cA_len`: 近似系数长度的输出变量指针。

- `cD_len`: 细节系数长度的输出变量指针。

**返回值**: 

返回成功或错误代码。

**注意**: 

输出系数数组的长度约为输入信号长度的一半。由于卷积操作，信号边缘附近可能出现边界效应。

### tiny_dwt_reconstruct_f32

```c
/**
 * @name tiny_dwt_reconstruct_f32
 * @brief Perform single-level discrete wavelet reconstruction
 * @param cA Approximation coefficients array
 * @param cD Detail coefficients array
 * @param coeff_len Length of coefficient arrays (cA and cD must have same length)
 * @param wavelet Wavelet type (DB1-DB10)
 * @param output Output array for reconstructed signal
 * @param output_len Output length of reconstructed signal
 * @return tiny_error_t
 */
tiny_error_t tiny_dwt_reconstruct_f32(const float *cA, const float *cD, int coeff_len,
                                      tiny_wavelet_type_t wavelet,
                                      float *output, int *output_len);
```

**描述**: 

执行单级离散小波重构，将近似和细节系数组合以重建原始信号。

**特点**:

- 以 2 为因子上采样系数

- 使用重构滤波器执行卷积

- 组合低通和高通重构结果

- 中心区域完美重构（在数值精度范围内）

**参数**:

- `cA`: 近似系数数组指针。

- `cD`: 细节系数数组指针。

- `coeff_len`: 两个系数数组的长度（必须相等）。

- `wavelet`: 用于分解的小波类型。

- `output`: 重构信号的输出数组指针。大小应至少为 `coeff_len * 2`。

- `output_len`: 重构信号长度的输出变量指针。

**返回值**: 

返回成功或错误代码。

**注意**: 

重构信号长度为 `coeff_len * 2`。可能出现边界效应，尤其是在信号边缘附近。中心区域通常具有非常高的重构精度。

## 多级 DWT

### tiny_dwt_multilevel_decompose_f32

```c
/**
 * @name tiny_dwt_multilevel_decompose_f32
 * @brief Perform multi-level DWT decomposition
 * @param input Input signal array
 * @param input_len Length of the input signal
 * @param wavelet Wavelet type (DB1-DB10)
 * @param levels Number of decomposition levels
 * @param cA_out Output pointer for final approximation coefficients
 * @param cD_out Output pointer for all detail coefficients (concatenated)
 * @param len_out Output length of final approximation coefficients
 * @return tiny_error_t
 */
tiny_error_t tiny_dwt_multilevel_decompose_f32(const float *input, int input_len,
                                               tiny_wavelet_type_t wavelet, int levels,
                                               float **cA_out, float **cD_out, int *len_out);
```

**描述**: 

执行多级离散小波分解，递归分解近似系数以创建信号的分层表示。

**特点**:

- 递归分解近似系数

- 存储所有级别的所有细节系数

- 返回最终近似和连接的细节系数

- 内存在内部分配，必须由调用者释放

**参数**:

- `input`: 输入信号数组指针。

- `input_len`: 输入信号长度。

- `wavelet`: 来自 `tiny_wavelet_type_t` 枚举的小波类型。

- `levels`: 分解级别数。必须为正数。

- `cA_out`: 最终近似系数的输出指针指针。内存在内部分配。

- `cD_out`: 所有细节系数的输出指针指针（从所有级别连接）。内存在内部分配。

- `len_out`: 最终近似系数长度的输出变量指针。

**返回值**: 

返回成功或错误代码。

**内存管理**: 

函数为 `cA_out` 和 `cD_out` 分配内存。调用者负责使用 `free()` 释放此内存。

**系数结构**:

对于 N 级分解：
- 级别 1：cA1（长度 ≈ input_len/2），cD1（长度 ≈ input_len/2）

- 级别 2：cA2（长度 ≈ input_len/4），cD2（长度 ≈ input_len/4）

- ...

- 级别 N：cAN（长度 ≈ input_len/2^N），cDN（长度 ≈ input_len/2^N）

`cD_out` 数组包含：连接的 [cD1, cD2, ..., cDN]。

### tiny_dwt_multilevel_reconstruct_f32

```c
/**
 * @name tiny_dwt_multilevel_reconstruct_f32
 * @brief Perform multi-level DWT reconstruction
 * @param cA_init Final approximation coefficients from multi-level decomposition
 * @param cD_all All detail coefficients (concatenated from all levels)
 * @param final_len Length of final approximation coefficients
 * @param wavelet Wavelet type (DB1-DB10)
 * @param levels Number of decomposition levels
 * @param output Output array for reconstructed signal
 * @return tiny_error_t
 */
tiny_error_t tiny_dwt_multilevel_reconstruct_f32(const float *cA_init, const float *cD_all,
                                                 int final_len, tiny_wavelet_type_t wavelet, int levels,
                                                 float *output);
```

**描述**: 

执行多级离散小波重构，从最终近似和所有细节系数递归重构。

**特点**:

- 从最终近似开始递归重构

- 使用相应的细节系数重构每个级别

- 输出长度匹配原始输入长度

- 边界效应随分解级别累积

**参数**:

- `cA_init`: 来自多级分解的最终近似系数指针。

- `cD_all`: 从所有级别连接的所有细节系数指针。

- `final_len`: 最终近似系数长度。

- `wavelet`: 用于分解的小波类型。

- `levels`: 分解级别数。

- `output`: 重构信号的输出数组指针。大小应至少为原始输入长度。

**返回值**: 

返回成功或错误代码。

**注意**: 

`cD_all` 数组应按顺序包含细节系数：[cD_level1, cD_level2, ..., cD_levelN]。边界效应随着分解级别的增加而变得更加明显。

## 系数处理

### tiny_dwt_coeffs_process

```c
/**
 * @name tiny_dwt_coeffs_process
 * @brief Placeholder for user-defined coefficient processing
 * @param cA Approximation coefficients
 * @param cD Detail coefficients
 * @param cA_len Length of approximation coefficients
 * @param cD_len Length of detail coefficients
 * @param levels Number of decomposition levels
 */
void tiny_dwt_coeffs_process(float *cA, float *cD, int cA_len, int cD_len, int levels);
```

**描述**: 

用于用户定义系数处理的占位符函数。用户可以扩展此函数以实现去噪、阈值处理或其他系数操作。

**参数**:

- `cA`: 近似系数指针（可修改）。

- `cD`: 细节系数指针（可修改）。

- `cA_len`: 近似系数长度。

- `cD_len`: 细节系数长度。

- `levels`: 分解级别数。

**注意**: 

目前此函数不执行任何操作。用户可以修改它以实现自定义处理，例如：

- 硬/软阈值处理用于去噪

- 系数选择用于压缩

- 特征提取

- 异常检测

## 使用流程

### 单级 DWT 流程

1. **分解信号**:
   ```c
   float input[64];
   float cA[32], cD[32];
   int cA_len, cD_len;
   tiny_dwt_decompose_f32(input, 64, TINY_WAVELET_DB4, cA, cD, &cA_len, &cD_len);
   ```

2. **处理系数**（可选）:
   ```c
   // 应用阈值处理、去噪等
   ```

3. **重构信号**:
   ```c
   float output[64];
   int output_len;
   tiny_dwt_reconstruct_f32(cA, cD, cA_len, TINY_WAVELET_DB4, output, &output_len);
   ```

### 多级 DWT 流程

1. **多级分解**:
   ```c
   float *cA, *cD;
   int cA_len;
   tiny_dwt_multilevel_decompose_f32(input, 128, TINY_WAVELET_DB4, 3, &cA, &cD, &cA_len);
   ```

2. **处理系数**（可选）:
   ```c
   tiny_dwt_coeffs_process(cA, cD, cA_len, 128 - cA_len, 3);
   ```

3. **多级重构**:
   ```c
   float output[128];
   tiny_dwt_multilevel_reconstruct_f32(cA, cD, cA_len, TINY_WAVELET_DB4, 3, output);
   ```

4. **释放内存**:
   ```c
   free(cA);
   free(cD);
   ```

## 应用场景

DWT 广泛应用于各种应用：

- **信号去噪**：对细节系数进行阈值处理以去除噪声
- **数据压缩**：仅存储显著系
- **特征提取**：分析不同尺度的系数
- **图像处理**：2D DWT 用于图像压缩和分析
- **生物医学信号处理**：ECG/EEG 分析、伪影去除
- **结构健康监测**：振动分析、损伤检测
- **时频分析**：在时间和频率上定位事件

## 边界效应

DWT 操作使用对称填充来处理信号边界。但是，仍可能出现边界效应：

- **单级**：边界效应通常从每个边缘延伸约 filter_length 个样本
- **多级**：边界效应累积并延伸约 filter_length × levels 个样本
- **中心区域**：通常具有非常高的重构精度
- **建议**：使用长度大于 2 × filter_length × levels 的信号以获得最佳结果

## 能量保持

对于完美重构小波（如 Daubechies），能量应该大致保持：

\[
E_{input} \approx E_{cA} + E_{cD}
\]

\[
E_{output} \approx E_{input}
\]

其中能量计算为：

\[
E = \sum_{n} |x[n]|^2
\]

边界效应可能导致轻微的能量差异，但中心区域应保持出色的能量保持。
