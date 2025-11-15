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

**返回值**: 返回成功或错误代码。

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

**描述**: 计算信号和卷积核之间的互相关性。

**特点**

- 支持平台加速

**参数**:

- `Signal`: 输入信号数组

- `siglen`: 信号数组的长度

- `Kernel`: 输入卷积核数组

- `kernlen`: 卷积核数组的长度

- `corrvout`: 输出数组，用于存储互相关性结果

**返回值**: 返回成功或错误代码。
