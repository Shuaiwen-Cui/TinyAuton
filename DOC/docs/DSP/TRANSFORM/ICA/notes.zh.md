# 说明

!!! note "说明"
    独立成分分析（ICA）是一种盲源分离技术，将混合信号分离为独立的源成分。它假设观测信号是统计独立源的线性混合。ICA 广泛应用于信号处理、神经科学、图像处理和音频源分离应用。

## ICA 概述

### 数学原理

ICA 解决盲源分离问题：

\[
\mathbf{X} = \mathbf{A} \cdot \mathbf{S}
\]

其中：

- \( \mathbf{X} \) 是观测（混合）信号矩阵（num_obs × num_samples）

- \( \mathbf{A} \) 是未知混合矩阵（num_obs × num_sources）

- \( \mathbf{S} \) 是独立源信号矩阵（num_sources × num_samples）

**目标**：找到解混矩阵 \( \mathbf{W} \) 使得：

\[
\mathbf{S} = \mathbf{W} \cdot \mathbf{X}
\]

**关键假设**：

1. **统计独立性**：源信号在统计上独立

2. **非高斯性**：最多一个源可以是高斯分布（用于可识别性）

3. **线性混合**：观测是源的线性组合

4. **方阵或超定**：观测数量 ≥ 源数量

### ICA vs PCA

- **PCA**：找到最大方差的正交方向（二阶统计量）
- **ICA**：找到统计独立的方向（高阶统计量）
- **PCA**：去相关数据（去除线性依赖）
- **ICA**：分离独立源（去除所有依赖）

## 算法

### FastICA

库实现了 FastICA 算法，基于最大化非高斯性：

**目标函数**：最大化 \( \mathbf{w}^T \mathbf{x} \) 的非高斯性

**非线性函数**：

- **tanh**：\( g(u) = \tanh(u) \) — 适用于超高斯源

- **cube**：\( g(u) = u^3 \) — 适用于次高斯源

- **gauss**：高斯型非线性 — 适用于对称分布的源（常用 \( g(u) = u\,e^{-u^2/2} \)）

- **skew**：偏斜敏感的非线性 — 适用于具有偏斜性的源

**算法步骤**：

1. 中心化数据：\( \mathbf{x}_c = \mathbf{x} - \text{mean}(\mathbf{x}) \)

2. 白化数据：\( \mathbf{z} = \mathbf{D}^{-1/2} \mathbf{E}^T \mathbf{x}_c \)

3. 使用定点迭代提取成分

4. 正交化成分（Gram-Schmidt）

## 预处理

### 中心化

从每个观测中减去均值：

\[
\mathbf{x}_c = \mathbf{x} - \bar{\mathbf{x}}
\]

其中 \( \bar{\mathbf{x}} \) 是均值向量。

### 白化

将数据变换为具有单位方差和零相关：

\[
\mathbf{z} = \mathbf{D}^{-1/2} \mathbf{E}^T \mathbf{x}_c
\]

其中：

- \( \mathbf{E} \) 是协方差矩阵的特征向量

- \( \mathbf{D} \) 是协方差矩阵的特征值

**白化矩阵**：

\[
\mathbf{W}_{whiten} = \mathbf{D}^{-1/2} \mathbf{E}^T
\]

## 函数

### tiny_ica_separate_f32

```c
/**
 * @name tiny_ica_separate_f32
 * @brief Perform ICA separation on mixed signals
 * @param mixed_signals Input mixed signals (num_obs x num_samples, row-major)
 * @param num_obs Number of observations (mixed signals)
 * @param num_samples Number of samples per signal
 * @param num_sources Number of independent sources to extract
 * @param separated_sources Output separated sources (num_sources x num_samples, row-major)
 * @param algorithm ICA algorithm to use (default: TINY_ICA_FASTICA)
 * @param nonlinearity Nonlinearity function for FastICA (default: TINY_ICA_NONLINEARITY_TANH)
 * @param max_iter Maximum number of iterations (default: 100)
 * @param tolerance Convergence tolerance (default: 1e-4)
 * @return tiny_error_t
 */
tiny_error_t tiny_ica_separate_f32(const float *mixed_signals,
                                   int num_obs,
                                   int num_samples,
                                   int num_sources,
                                   float *separated_sources,
                                   tiny_ica_algorithm_t algorithm,
                                   tiny_ica_nonlinearity_t nonlinearity,
                                   int max_iter,
                                   float tolerance);
```

**描述**: 

在一次函数调用中执行完整的 ICA 分离。这是 ICA 最简单的接口。

**参数**:

- `mixed_signals`: 输入混合信号数组指针。数据布局为行主序：`mixed_signals[i * num_samples + j]` 是观测 `i` 的样本 `j`。

- `num_obs`: 观测数量（混合信号）。必须 ≥ `num_sources`。

- `num_samples`: 每个信号的样本数。

- `num_sources`: 要提取的独立源数量。必须 ≤ `num_obs`。

- `separated_sources`: 分离源的输出数组指针。数据布局为行主序：`separated_sources[i * num_samples + j]` 是源 `i` 的样本 `j`。大小必须至少为 `num_sources * num_samples`。

- `algorithm`: ICA 算法类型。目前仅支持 `TINY_ICA_FASTICA`。

- `nonlinearity`: FastICA 的非线性函数：
  - `TINY_ICA_NONLINEARITY_TANH`: tanh（默认，适用于超高斯）
  - `TINY_ICA_NONLINEARITY_CUBE`: cube（适用于次高斯）
  - `TINY_ICA_NONLINEARITY_GAUSS`: gauss（适用于对称源）
  - `TINY_ICA_NONLINEARITY_SKEW`: skew（适用于偏斜源）

- `max_iter`: FastICA 的最大迭代次数。如果 ≤ 0，默认值为 100。

- `tolerance`: 收敛容差。当变化 < tolerance 时算法停止。如果 ≤ 0，默认值为 1e-4。

**返回值**: 

成功时返回 `TINY_OK`，失败时返回错误代码。

**处理步骤**:

1. **中心化数据**：从每个观测中减去均值
2. **白化数据**：使用特征值分解去相关并归一化方差
3. **提取成分**：使用 FastICA 找到独立成分
4. **重构源**：将解混矩阵应用于白化数据

**注意**: 

此函数在内部执行所有步骤。对于重复分离，使用基于结构的 API（`tiny_ica_init`、`tiny_ica_fit`、`tiny_ica_transform`）以避免重新计算白化矩阵。

### tiny_ica_init

```c
/**
 * @name tiny_ica_init
 * @brief Initialize ICA structure for repeated use
 * @param ica Pointer to ICA structure
 * @param num_obs Number of observations (mixed signals)
 * @param num_sources Number of sources to extract
 * @return tiny_error_t
 */
tiny_error_t tiny_ica_init(tiny_ica_t *ica, int num_obs, int num_sources);
```

**描述**: 

初始化 ICA 结构以供重复使用。为混合矩阵、解混矩阵、白化矩阵和均值向量分配内存。

**参数**:

- `ica`: `tiny_ica_t` 结构指针。

- `num_obs`: 观测数量（混合信号）。必须 ≥ `num_sources`。

- `num_sources`: 要提取的源数量。必须 ≤ `num_obs`。

**返回值**: 

成功时返回 `TINY_OK`，失败时返回错误代码。

**内存管理**: 

函数在内部分配内存。调用 `tiny_ica_deinit()` 释放它。

### tiny_ica_fit

```c
/**
 * @name tiny_ica_fit
 * @brief Fit ICA model to mixed signals (learn unmixing matrix)
 * @param ica Pointer to initialized ICA structure
 * @param mixed_signals Input mixed signals (num_obs x num_samples, row-major)
 * @param num_samples Number of samples per signal
 * @param algorithm ICA algorithm to use
 * @param nonlinearity Nonlinearity function for FastICA
 * @param max_iter Maximum number of iterations
 * @param tolerance Convergence tolerance
 * @return tiny_error_t
 */
tiny_error_t tiny_ica_fit(tiny_ica_t *ica,
                          const float *mixed_signals,
                          int num_samples,
                          tiny_ica_algorithm_t algorithm,
                          tiny_ica_nonlinearity_t nonlinearity,
                          int max_iter,
                          float tolerance);
```

**描述**: 

将 ICA 模型拟合到训练数据。学习解混矩阵和白化矩阵。拟合后，使用 `tiny_ica_transform()` 分离新信号。

**参数**:

- `ica`: 初始化的 `tiny_ica_t` 结构指针。

- `mixed_signals`: 输入混合信号数组指针（行主序布局）。

- `num_samples`: 每个信号的样本数。

- `algorithm`: ICA 算法类型。目前仅支持 `TINY_ICA_FASTICA`。

- `nonlinearity`: FastICA 的非线性函数。

- `max_iter`: 最大迭代次数。如果 ≤ 0，默认值为 100。

- `tolerance`: 收敛容差。如果 ≤ 0，默认值为 1e-4。

**返回值**: 

成功时返回 `TINY_OK`，失败时返回错误代码。

**注意**: 

拟合后，ICA 结构包含学习到的解混矩阵和白化矩阵。这些可以重复用于变换新数据。

### tiny_ica_transform

```c
/**
 * @name tiny_ica_transform
 * @brief Apply learned ICA model to separate signals
 * @param ica Pointer to fitted ICA structure
 * @param mixed_signals Input mixed signals (num_obs x num_samples, row-major)
 * @param num_samples Number of samples per signal
 * @param separated_sources Output separated sources (num_sources x num_samples, row-major)
 * @return tiny_error_t
 */
tiny_error_t tiny_ica_transform(const tiny_ica_t *ica,
                                const float *mixed_signals,
                                int num_samples,
                                float *separated_sources);
```

**描述**: 

将先前拟合的 ICA 模型应用于分离新信号。比 `tiny_ica_separate_f32()` 快得多，因为它重用学习到的解混矩阵。

**参数**:

- `ica`: 拟合的 `tiny_ica_t` 结构指针。

- `mixed_signals`: 输入混合信号数组指针（行主序布局）。

- `num_samples`: 每个信号的样本数。

- `separated_sources`: 分离源的输出数组指针（行主序布局）。大小必须至少为 `num_sources * num_samples`。

**返回值**: 

成功时返回 `TINY_OK`，失败时返回错误代码。

**注意**: 

需要先使用 `tiny_ica_fit()` 拟合 `ica`。输入信号使用训练数据的均值进行中心化。

### tiny_ica_deinit

```c
/**
 * @name tiny_ica_deinit
 * @brief Deinitialize ICA structure and free memory
 * @param ica Pointer to ICA structure
 * @return tiny_error_t
 */
tiny_error_t tiny_ica_deinit(tiny_ica_t *ica);
```

**描述**: 

取消初始化 ICA 结构并释放所有分配的内存。

**参数**:

- `ica`: `tiny_ica_t` 结构指针。

**返回值**: 

成功时返回 `TINY_OK`，失败时返回错误代码。

## 使用流程

### 简单一次性分离

```c
float mixed_signals[2 * 512];  // 2 个观测，每个 512 个样本
float separated_sources[2 * 512];  // 2 个源，每个 512 个样本

// 执行 ICA 分离
tiny_error_t ret = tiny_ica_separate_f32(
    mixed_signals, 2, 512, 2, separated_sources,
    TINY_ICA_FASTICA, TINY_ICA_NONLINEARITY_TANH, 100, 1e-4f);
```

### 重复分离（结构 API）

```c
tiny_ica_t ica;

// 初始化
tiny_ica_init(&ica, 2, 2);  // 2 个观测，2 个源

// 将模型拟合到训练数据
tiny_ica_fit(&ica, training_mixed, 512,
             TINY_ICA_FASTICA, TINY_ICA_NONLINEARITY_TANH, 100, 1e-4f);

// 变换新数据（可以多次调用）
tiny_ica_transform(&ica, new_mixed, 512, separated);

// 清理
tiny_ica_deinit(&ica);
```

## 应用场景

ICA 广泛应用于：

- **音频源分离**：从混合音频中分离单个乐器或声音
- **生物医学信号处理**：从伪影中分离 EEG/ECG 信号
- **图像处理**：特征提取、去噪
- **通信**：盲信道均衡
- **神经科学**：分析脑信号、fMRI 数据
- **传感器阵列处理**：从多个传感器分离信号

## 优缺点

### 优点

- **盲分离**：不需要混合矩阵的先验知识
- **统计独立性**：找到真正独立的源
- **非高斯源**：适用于非高斯信号
- **灵活**：可以处理不同数量的源和观测

### 缺点

- **模糊性**：分离源的尺度和符号是模糊的
- **顺序模糊性**：分离源的顺序是任意的
- **非高斯要求**：最多一个源可以是高斯分布
- **计算成本**：白化和特征值分解可能很昂贵
- **收敛性**：对于某些信号类型可能不收敛

## 设计考虑

### 源数量 vs 观测数量

- **方阵情况**（num_obs = num_sources）：标准 ICA 问题
- **超定**（num_obs > num_sources）：可以首先使用 PCA 降低维度
- **欠定**（num_obs < num_sources）：不支持（无法提取比观测更多的源）

### 非线性选择

- **tanh**：默认选择，适用于大多数超高斯源（语音、音乐）
- **cube**：用于次高斯源（均匀噪声、某些图像信号）
- **gauss**：适用于对称分布的源
- **skew**：当源具有明显偏斜性时有用

### 收敛参数

- **max_iter**：通常 50-200 次迭代。困难情况下需要更多迭代。
- **tolerance**：通常 1e-4 到 1e-6。更小的容差 = 更准确但更慢。

### 数据要求

- **样本大小**：更多样本 = 更好的分离质量
- **独立性**：源必须在统计上独立
- **非高斯性**：最多一个源可以是高斯分布

## 注意事项

- ICA 只能分离源到缩放因子和排列的程度
- 分离源的顺序可能与原始顺序不匹配
- 当源具有不同统计特性时，ICA 效果最好
- 白化是 ICA 的关键预处理步骤
- FastICA 因其速度和简单性而成为流行算法

