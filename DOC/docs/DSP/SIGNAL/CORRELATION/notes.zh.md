# 说明

!!! note "说明"
    相关性是信号处理中的一个重要概念，通常用于分析信号之间的相似性或依赖性。它在许多应用中都很有用，例如模式识别、时间序列分析和信号检测。

## 滑动相关

### 数学原理

相关计算公式为：

\[
\text{Correlation}[n] = \sum_{m=0}^{L_p - 1} S[n + m] \cdot P[m]
\]

其中：

- \( S \) 为输入信号，长度为 \( L_s \)

- \( P \) 为模式序列（Pattern），长度为 \( L_p \)

- \( n \in [0, L_s - L_p] \)

**输出长度计算**：

\[
L_{\text{out}} = L_s - L_p + 1
\]

### tiny_corr_f32

```c
/**
 * @name: tiny_corr_f32
 * @brief Correlation function
 *
 * @param Signal: input signal array
 * @param siglen: length of the signal array
 * @param Pattern: input pattern array
 * @param patlen: length of the pattern array
 * @param dest: output array for the correlation result
 *
 * @return tiny_error_t
 */
tiny_error_t tiny_corr_f32(const float *Signal, const int siglen, const float *Pattern, const int patlen, float *dest)
{
    if (NULL == Signal || NULL == Pattern || NULL == dest)
    {
        return TINY_ERR_DSP_NULL_POINTER;
    }

    if (siglen < patlen) // signal length shoudl be greater than pattern length
    {
        return TINY_ERR_DSP_MISMATCH;
    }

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    dsps_corr_f32(Signal, siglen, Pattern, patlen, dest);
#else

    for (size_t n = 0; n <= (siglen - patlen); n++)
    {
        float k_corr = 0;
        for (size_t m = 0; m < patlen; m++)
        {
            k_corr += Signal[n + m] * Pattern[m];
        }
        dest[n] = k_corr;
    }

#endif

    return TINY_OK;
}
```

**描述**: 

计算信号和模式之间的相关性。

**特点**

- 支持平台加速

**参数**:

- `Signal`: 输入信号数组

- `siglen`: 信号数组的长度

- `Pattern`: 输入模式数组

- `patlen`: 模式数组的长度

- `dest`: 输出数组，用于存储相关性结果

**返回值**: 

返回成功或错误代码。

## 交叉相关函数

### 数学原理

互相关计算公式为：

\[
R_{xy}[n] = \sum_{k} x[k] \cdot y[k + n]
\]

其中：

- \( x \) 为信号序列，长度为 \( L_x \)

- \( y \) 为卷积核（Kernel），长度为 \( L_y \)

- \( n \in [0, L_x + L_y - 2] \)

**输出长度计算**：

\[
L_{\text{out}} = L_x + L_y - 1
\]

### tiny_ccorr_f32

```c
/**
 * @name: tiny_ccorr_f32
 * @brief Cross-correlation function
 *
 * @param Signal: input signal array
 * @param siglen: length of the signal array
 * @param Kernel: input kernel array
 * @param kernlen: length of the kernel array
 * @param corrvout: output array for the cross-correlation result
 *
 * @return tiny_error_t
 */
tiny_error_t tiny_ccorr_f32(const float *Signal, const int siglen, const float *Kernel, const int kernlen, float *corrvout)
{
    if (NULL == Signal || NULL == Kernel || NULL == corrvout)
    {
        return TINY_ERR_DSP_NULL_POINTER;
    }

    float *sig = (float *)Signal;
    float *kern = (float *)Kernel;
    int lsig = siglen;
    int lkern = kernlen;

    // swap signal and kernel if needed
    if (siglen < kernlen)
    {
        sig = (float *)Kernel;
        kern = (float *)Signal;
        lsig = kernlen;
        lkern = siglen;
    }

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    dsps_ccorr_f32(Signal, siglen, Kernel, kernlen, corrvout);
#else
    // stage I
    for (int n = 0; n < lkern; n++)
    {
        size_t k;
        size_t kmin = lkern - 1 - n;
        corrvout[n] = 0;

        for (k = 0; k <= n; k++)
        {
            corrvout[n] += sig[k] * kern[kmin + k];
        }
    }

    // stage II
    for (int n = lkern; n < lsig; n++)
    {
        size_t kmin, kmax, k;

        corrvout[n] = 0;

        kmin = n - lkern + 1;
        kmax = n;
        for (k = kmin; k <= kmax; k++)
        {
            corrvout[n] += sig[k] * kern[k - kmin];
        }
    }

    // stage III
    for (int n = lsig; n < lsig + lkern - 1; n++)
    {
        size_t kmin, kmax, k;

        corrvout[n] = 0;

        kmin = n - lkern + 1;
        kmax = lsig - 1;

        for (k = kmin; k <= kmax; k++)
        {
            corrvout[n] += sig[k] * kern[k - kmin];
        }
    }
#endif
    return TINY_OK;
}
```

**描述**: 

计算信号和卷积核之间的互相关性。

**特点**

- 支持平台加速

**参数**:

- `Signal`: 输入信号数组

- `siglen`: 信号数组的长度

- `Kernel`: 输入卷积核数组

- `kernlen`: 卷积核数组的长度

- `corrvout`: 输出数组，用于存储互相关性结果

**返回值**: 

返回成功或错误代码。



## 对比与总结

### 主要区别

| 特性 | `tiny_corr_f32` | `tiny_ccorr_f32` |
|------|----------------|------------------|
| **输出长度** | \( L_{\text{out}} = L_s - L_p + 1 \) | \( L_{\text{out}} = L_x + L_y - 1 \) |
| **长度要求** | 信号长度必须 ≥ 模式长度 | 任意长度均可（需要时自动交换） |
| **计算类型** | 滑动相关（仅有效区域） | 完全互相关（包含部分重叠） |
| **实现方式** | 单阶段嵌套循环 | 三阶段计算 |
| **应用场景** | 模式匹配、模板检测 | 完全相关分析、信号对齐 |

### 何时使用哪个函数

**使用 `tiny_corr_f32` 当：**

- 需要在较长的信号中查找模式（模式匹配）

- 只关心模式与信号完全重叠的位置

- 信号长度保证大于模式长度

- 需要更高效的模板匹配计算

- 需要在信号中检测已知模式的出现

**典型应用：**

- 图像处理中的模板匹配

- 时间序列中的模式检测

- 信号检测与识别

- 特征匹配

**使用 `tiny_ccorr_f32` 当：**

- 需要包含部分重叠的完全互相关

- 两个序列的长度可以是任意的（任一序列都可能更长）

- 需要分析两个信号之间所有可能的对齐方式

- 需要找到两个信号之间的最佳对齐或时间延迟

- 需要完整的相关函数进行进一步分析

**典型应用：**

- 两个信号之间的时间延迟估计

- 信号对齐与同步

- 完全相关分析

- 信号长度未知或可变的情况

### 总结

- **`tiny_corr_f32`** 针对 **模式匹配** 场景进行了优化，用于在较长的信号中搜索较短的模式。它只计算模式完全重叠时的相关性，输出更短，计算更快。

- **`tiny_ccorr_f32`** 计算任意长度的两个序列之间的 **完全互相关** ，包括所有部分重叠。这提供了完整的相关信息，但需要更多计算并产生更长的输出。

**选择建议：** 

需要高效的模式匹配时选择 `tiny_corr_f32`，需要全面的相关分析或信号长度可变时选择 `tiny_ccorr_f32`。
