# 说明

!!! note "说明"
    重采样是信号处理中的一个重要步骤，通常用于改变信号的采样率。它可以用于音频、视频和其他类型的信号处理。

## 信号下采样 - 跳跃  

信号跳跃下采样是指在原始信号中以一定间隔选择样本。它通常用于降低信号的采样率。注意区别于decimation，decimation是先滤波再下采样。

```c
/**
 * @name tiny_downsample_skip_f32
 * @brief Downsample a signal by a given factor using skipping
 *
 * @param input pointer to the input signal array
 * @param input_len length of the input signal array
 * @param output pointer to the output signal array
 * @param output_len pointer to the length of the output signal array
 * @param keep number of samples to keep
 * @param skip number of samples to skip
 *
 * @return tiny_error_t
 */
tiny_error_t tiny_downsample_skip_f32(const float *input, int input_len, float *output, int *output_len, int keep, int skip)
{
    if (!input || !output || !output_len)
        return TINY_ERR_DSP_NULL_POINTER;

    if (input_len <= 0 || keep <= 0 || skip <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    int out_len = input_len / skip;
    *output_len = out_len;

    for (int i = 0; i < out_len; i++)
    {
        output[i] = input[i * skip];
    }

    return TINY_OK;
}

```

**描述**：

信号跳跃下采样函数，通过两个整数参数`keep`和`skip`来控制跳跃下采样的过程。`keep`表示要保留的样本数，`skip`表示要跳过的样本数。

**特点**：

- 整数倍下采样

- 跳跃下采样

**参数**：

- `input`: 输入信号数组的指针

- `input_len`: 输入信号数组的长度

- `output`: 输出信号数组的指针

- `output_len`: 输出信号数组的长度的指针

- `keep`: 要保留的样本数

- `skip`: 要跳过的样本数

## 信号上采样 - 0填充

信号上采样是指在原始信号中插入零以增加采样率。它通常用于提高信号的采样率。注意区别于interpolation，上采样是先填充再插值。

```c
/**
 * @name tiny_upsample_zero_f32
 * @brief Upsample a signal using zero-insertion between samples
 *
 * @param input pointer to the input signal array
 * @param input_len length of the input signal array
 * @param output pointer to the output signal array
 * @param target_len target length for the output signal array
 * @return tiny_error_t
 */
tiny_error_t tiny_upsample_zero_f32(const float *input, int input_len, float *output, int target_len)
{
    if (!input || !output)
        return TINY_ERR_DSP_NULL_POINTER;

    if (input_len <= 0 || target_len <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    int factor = target_len / input_len;
    if (factor <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    for (int i = 0; i < target_len; i++)
    {
        output[i] = (i % factor == 0) ? input[i / factor] : 0.0f;
    }

    return TINY_OK;
}
```

**描述**：

信号上采样函数，通过在原始信号中插入零来增加采样率。它通常用于提高信号的采样率。


**特点**：

- 整数倍上采样

- 0填充

**参数**：

- `input`: 输入信号数组的指针

- `input_len`: 输入信号数组的长度

- `output`: 输出信号数组的指针

- `target_len`: 目标长度，用于输出信号数组的长度


## 信号重采样 - 任意倍数上下采样 - 线性插值

重采样是指将信号的采样率从一个值转换为另一个值。这里我们采用最直接的线性插值法。首先计算新信号中点在老信号中的大概位置，结合起位置平衡左右点数值进行生成。

```c
/**
 * @name: tiny_resample_f32
 * @brief Resample a signal to a target length
 *
 * @param input pointer to the input signal array
 * @param input_len length of the input signal array
 * @param output pointer to the output signal array
 * @param target_len target length for the output signal array
 * @return tiny_error_t
 */
tiny_error_t tiny_resample_f32(const float *input,
                               int input_len,
                               float *output,
                               int target_len)
{
    if (!input || !output)
        return TINY_ERR_DSP_NULL_POINTER;

    if (input_len <= 0 || target_len <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    float ratio = (float)(target_len) / (float)(input_len);

    for (int i = 0; i < target_len; i++)
    {
        float pos = i / ratio;
        int index = (int)floorf(pos);
        float frac = pos - index;

        if (index >= input_len - 1)
            output[i] = input[input_len - 1]; // Clamp at end
        else
            output[i] = input[index] * (1.0f - frac) + input[index + 1] * frac;
    }

    return TINY_OK;
}


```

**描述**：

信号重采样是指将信号的采样率从一个值转换为另一个值。它通常用于音频和视频处理，以适应不同的播放设备或网络带宽。

**特点**：

- 任意倍数上下采样

- 线性插值

**参数**：

- `input`: 输入信号数组的指针

- `input_len`: 输入信号数组的长度

- `output`: 输出信号数组的指针

- `target_len`: 目标长度，用于输出信号数组的长度

