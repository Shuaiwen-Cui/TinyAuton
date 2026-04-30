# 说明

!!! note "说明"
    相关性是信号处理中的重要概念，用于分析信号之间的相似性或依赖性，广泛应用于模式识别、时间序列分析和信号检测。本库提供两个层次的相关功能：滑动相关（`tiny_corr_f32`）和完全互相关（`tiny_ccorr_f32`）。

---

## 1. 算法原理

### 1.1 滑动相关

滑动相关（模式匹配）将短模式在长信号上滑动，在模式完全重叠的每个有效位置计算点积：

\[
\text{Correlation}[n] = \sum_{m=0}^{L_p - 1} S[n + m] \cdot P[m]
\]

其中：

- \(S\) 为输入信号，长度 \(L_s\)
- \(P\) 为模式序列，长度 \(L_p\)
- \(n \in [0, L_s - L_p]\)

**输出长度**：\(L_{out} = L_s - L_p + 1\)

**与卷积不同，模式不翻转** —— 直接原样在信号上滑动。

### 1.2 完全互相关

完全互相关计算两个序列在所有可能移位位置（包括部分重叠）的相关性：

\[
R_{xy}[n] = \sum_{k} x[k] \cdot y[k + n]
\]

其中：

- \(x\) 为较长序列，长度 \(L_x\)
- \(y\) 为较短序列，长度 \(L_y\)
- \(n \in [0, L_x + L_y - 2]\)

**输出长度**：\(L_{out} = L_x + L_y - 1\)

与卷积类似，计算自然分三阶段进行（预热期、稳态期、冷却期）。

---

## 2. 代码设计理念

### 2.1 两种不同的相关语义

`tiny_corr_f32` 和 `tiny_ccorr_f32` 服务于根本不同的目的：

- **滑动相关** 仅返回模式完全在信号内部的位置——适用于模板匹配。
- **完全互相关** 返回所有可能的移位对齐，包括部分重叠——适用于信号对齐、时延估计和完全相关分析。

### 2.2 通用互相关中的长度交换

`tiny_ccorr_f32` **仅在通用回退路径** （`#else`）中交换信号和核以使 `lsig >= lkern`。原因是：

- ESP32 ESP-DSP 后端（`dsps_ccorr_f32`）期望原始顺序。
- 三阶段循环索引假定较长序列为"信号"。
- 在 ESP32 上，调用者需确保 `siglen >= kernlen`（否则后端可能异常）。

### 2.3 const 正确性

在通用路径中，局部指针声明为 `const float *sig`、`const float *kern`，保持输入数组的只读属性。这与只读语义一致，并有助于编译器优化。

---

## 3. API 接口 — 函数

### 3.1 `tiny_corr_f32`

```c
/**
 * @name: tiny_corr_f32
 * @brief Correlation function (pattern matching)
 *
 * @param Signal: input signal array
 * @param siglen: length of the signal array
 * @param Pattern: input pattern array
 * @param patlen: length of the pattern array
 * @param dest: output array for the correlation result
 *              Must hold at least (siglen - patlen + 1) elements
 *
 * @return tiny_error_t
 */
tiny_error_t tiny_corr_f32(const float *Signal, const int siglen,
                            const float *Pattern, const int patlen,
                            float *dest)
{
    if (NULL == Signal || NULL == Pattern || NULL == dest)
        return TINY_ERR_DSP_NULL_POINTER;

    if (siglen <= 0 || patlen <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    if (siglen < patlen)
        return TINY_ERR_DSP_MISMATCH;

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    dsps_corr_f32(Signal, siglen, Pattern, patlen, dest);
#else
    for (size_t n = 0; n <= (siglen - patlen); n++)
    {
        float k_corr = 0;
        for (size_t m = 0; m < patlen; m++)
            k_corr += Signal[n + m] * Pattern[m];
        dest[n] = k_corr;
    }
#endif

    return TINY_OK;
}
```

**描述**：

计算较长信号与较短模式的滑动相关。仅计算模式完全重叠的位置。模式不翻转——每个移位位置直接进行逐元素乘积累加。

**特点**：

- 无翻转的模式匹配
- ESP32 平台加速
- O(siglen × patlen) 时间复杂度

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `Signal` | `const float*` | 输入信号数组指针 |
| `siglen` | `int` | 信号长度（> 0） |
| `Pattern` | `const float*` | 模式数组指针 |
| `patlen` | `int` | 模式长度（> 0） |
| `dest` | `float*` | 输出缓冲区，至少容纳 `siglen - patlen + 1` 个元素 |

**返回值**：

| 码值 | 含义 |
|------|------|
| `TINY_OK` | 相关成功 |
| `TINY_ERR_DSP_NULL_POINTER` | `Signal`、`Pattern` 或 `dest` 为空 |
| `TINY_ERR_DSP_INVALID_PARAM` | `siglen ≤ 0` 或 `patlen ≤ 0` |
| `TINY_ERR_DSP_MISMATCH` | `siglen < patlen`（模式比信号长） |

---

### 3.2 `tiny_ccorr_f32`

```c
/**
 * @name: tiny_ccorr_f32
 * @brief Cross-correlation function (full, all shifts)
 *
 * @param Signal: input signal array
 * @param siglen: length of the signal array
 * @param Kernel: input kernel array
 * @param kernlen: length of the kernel array
 * @param corrvout: output array for the cross-correlation result
 *                  Must hold at least (siglen + kernlen - 1) elements
 *
 * @return tiny_error_t
 */
tiny_error_t tiny_ccorr_f32(const float *Signal, const int siglen,
                             const float *Kernel, const int kernlen,
                             float *corrvout)
{
    if (NULL == Signal || NULL == Kernel || NULL == corrvout)
        return TINY_ERR_DSP_NULL_POINTER;

    if (siglen <= 0 || kernlen <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    dsps_ccorr_f32(Signal, siglen, Kernel, kernlen, corrvout);
#else
    const float *sig  = Signal;
    const float *kern = Kernel;
    int lsig  = siglen;
    int lkern = kernlen;

    /* 将较长者作为"信号"以确保三阶段索引正确 */
    if (siglen < kernlen)
    {
        sig  = Kernel;
        kern = Signal;
        lsig  = kernlen;
        lkern = siglen;
    }

    // 阶段 I — 预热期（核部分覆盖左边缘）
    for (int n = 0; n < lkern; n++)
    {
        size_t kmin = lkern - 1 - n;
        corrvout[n] = 0;
        for (size_t k = 0; k <= n; k++)
            corrvout[n] += sig[k] * kern[kmin + k];
    }

    // 阶段 II — 稳态期（核完全在信号内部）
    for (int n = lkern; n < lsig; n++)
    {
        corrvout[n] = 0;
        size_t kmin = n - lkern + 1;
        for (size_t k = kmin; k <= n; k++)
            corrvout[n] += sig[k] * kern[k - kmin];
    }

    // 阶段 III — 冷却期（核滑过右边缘）
    for (int n = lsig; n < lsig + lkern - 1; n++)
    {
        corrvout[n] = 0;
        size_t kmin = n - lkern + 1;
        size_t kmax = lsig - 1;
        for (size_t k = kmin; k <= kmax; k++)
            corrvout[n] += sig[k] * kern[k - kmin];
    }
#endif
    return TINY_OK;
}
```

**描述**：

计算两个序列的完全互相关，包括所有部分重叠位置。在通用路径中内部交换操作数以确保较长序列始终为"信号"。

**特点**：

- 完全互相关，含部分重叠
- 通用路径自动长度排序
- ESP32 硬件加速

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `Signal` | `const float*` | 输入信号数组指针 |
| `siglen` | `int` | 信号长度（> 0） |
| `Kernel` | `const float*` | 核数组指针 |
| `kernlen` | `int` | 核长度（> 0） |
| `corrvout` | `float*` | 输出缓冲区，至少容纳 `siglen + kernlen - 1` 个元素 |

**返回值**：

| 码值 | 含义 |
|------|------|
| `TINY_OK` | 互相关成功 |
| `TINY_ERR_DSP_NULL_POINTER` | `Signal`、`Kernel` 或 `corrvout` 为空 |
| `TINY_ERR_DSP_INVALID_PARAM` | `siglen ≤ 0` 或 `kernlen ≤ 0` |

---

## 4. 函数对比

| 特性 | `tiny_corr_f32` | `tiny_ccorr_f32` |
|------|----------------|------------------|
| **输出长度** | \(L_s - L_p + 1\) | \(L_x + L_y - 1\) |
| **重叠类型** | 仅完全重叠 | 所有移位（含部分） |
| **长度要求** | `siglen ≥ patlen`（强制） | 任意（通用路径自动交换） |
| **实现方式** | 单层嵌套循环 | 三阶段计算 |
| **ESP32 外部依赖** | `dsps_corr_f32` | `dsps_ccorr_f32` |
| **使用场景** | 模式匹配 / 模板检测 | 时延估计 / 信号对齐 |

### 何时使用 `tiny_corr_f32`

- 在较长信号中搜索已知模式
- 仅关心模式完全在信号内的位置
- 需要最高效的计算（无部分重叠开销）
- 示例：检测接收比特流中的前导码

### 何时使用 `tiny_ccorr_f32`

- 需要包含部分重叠的互相关
- 估计两个信号之间的时间延迟
- 需要对齐可能不同长度的两个信号
- 示例：从两个传感器读数计算到达时间差

---

## 5. ⚠️ 重要注意事项

### 5.1 相关 vs 卷积

相关 **不翻转** 核/模式。卷积会翻转。如果你误以为可以用带反转核的卷积来做相关，请注意：

- 两者的边界处理实现不同。
- 明确使用 `tiny_corr_f32` 或 `tiny_ccorr_f32` 进行相关——不要误用卷积函数。

### 5.2 输出缓冲区最小大小

| 函数 | 最小输出缓冲区大小 |
|------|-------------------|
| `tiny_corr_f32` | `siglen - patlen + 1` |
| `tiny_ccorr_f32` | `siglen + kernlen - 1` |

### 5.3 `siglen < patlen` 的处理

- `tiny_corr_f32`：立即返回 `TINY_ERR_DSP_MISMATCH`（模式必须适配信号）。
- `tiny_ccorr_f32`：通用路径自动交换操作数。在 ESP32 上，`dsps_ccorr_f32` 可能处理或拒绝此情况，取决于其内部实现——为保证可移植性，请确保 `siglen >= kernlen` 或在目标平台上测试。

### 5.4 平台差异

在 ESP32 上，两个函数均委托给 ESP-DSP 库。所有其他平台使用通用回退路径。长度排序的行为保证不同：

- `tiny_ccorr_f32` 通用路径：无论哪个更长均可工作。
- `tiny_ccorr_f32` ESP32 路径：行为取决于 `dsps_ccorr_f32`——请使用具体的 ESP-IDF 版本进行测试。
