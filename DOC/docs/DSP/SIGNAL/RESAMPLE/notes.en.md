# NOTES

!!! note "Note"
    Resampling is an important step in signal processing, typically used to change the sampling rate of a signal. It can be used in audio, video, and other types of signal processing.

## SIGNAL DOWNSAMPLING - SKIP

    Signal downsampling by skipping samples is a method of selecting samples from the original signal at regular intervals. It is typically used to reduce the sampling rate of a signal. Note that this is different from decimation, which involves filtering before downsampling.

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

***Description**:

Signal downsampling function that uses two integer parameters `keep` and `skip` to control the downsampling process. `keep` indicates the number of samples to keep, while `skip` indicates the number of samples to skip.

**Features**:

- Integer Factor Downsampling

- Skip Downsampling

**Parameters**:

- `input`: Pointer to the input signal array

- `input_len`: Length of the input signal array

- `output`: Pointer to the output signal array

- `output_len`: Pointer to the length of the output signal array

- `keep`: Number of samples to keep

- `skip`: Number of samples to skip

## SIGNAL UPSAMPLING - 0 INSERTION

    Signal upsampling by inserting zeros is a method of increasing the sampling rate of a signal by inserting zeros between the original samples. This is typically used to prepare a signal for further processing, such as filtering.

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

***Description**:

Signal upsampling is the process of increasing the sampling rate of a signal by inserting zeros between the original samples. It is typically used to increase the sampling rate of a signal. Note that this is different from interpolation, which involves filling in gaps between samples.

**Features**:

- Integer Factor Upsampling

- Zero Insertion

**Parameters**:

- `input`: Pointer to the input signal array

- `input_len`: Length of the input signal array

- `output`: Pointer to the output signal array

- `target_len`: Target length for the output signal array



## SIGNAL RESAMPLING - ANY FACTOR - LINEAR INTERPOLATION

    Signal resampling is the process of changing the sampling rate of a signal. We need to first calculate the proportional factor and then use interpolation to fill in the gaps.


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

***Description**:

Signal resampling is the process of changing the sampling rate of a signal. Here, we use the most straightforward linear interpolation method. First, we calculate the approximate position of the new signal in the old signal, and then balance the left and right values to generate the new signal.

**Features**:

- Non-integer Factor

- Linear Interpolation

**Parameters**:

- `input`: Pointer to the input signal array

- `input_len`: Length of the input signal array

- `output`: Pointer to the output signal array

- `target_len`: Target length for the output signal array



