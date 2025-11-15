# NOTES

!!! note "Note"
    Correlation is an important concept in signal processing, often used to analyze similarities or dependencies between signals. It is useful in many applications, such as pattern recognition, time series analysis, and signal detection.

## CORRELATION FUNCTION

### Mathematical Principle

The correlation is computed as:

\[
\text{Correlation}[n] = \sum_{m=0}^{L_p - 1} S[n + m] \cdot P[m]
\]

Where:

- \( S \) is the input signal of length \( L_s \)

- \( P \) is the pattern of length \( L_p \)

- \( n \in [0, L_s - L_p] \)

**Output Length**:

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

**Description**: 

Computes the correlation between a signal and a pattern.

**Features**:

- Platform-specific optimization enabled.

**Parameters**:

- `Signal`: Pointer to the input signal array.

- `siglen`: Length of the signal array.

- `Pattern`: Pointer to the input pattern array.

- `patlen`: Length of the pattern array.

- `dest`: Pointer to the output array for the correlation result.

**Return Value**: Returns success or error code.





## CROSS-CORRELATION FUNCTION

### Mathematical Principle

The cross-correlation is computed as:

\[
R_{xy}[n] = \sum_{k} x[k] \cdot y[k + n]
\]

Where:

- \( x \) is the signal of length \( L_x \)

- \( y \) is the kernel of length \( L_y \)

- \( n \in [0, L_x + L_y - 2] \)

**Output Length**:

\[
L_{\text{out}} = L_x + L_y - 1
\]
