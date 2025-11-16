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

**Description**: 

Computes the cross-correlation between a signal and a kernel.

**Features**:

- Platform-specific optimization enabled.

- Automatically handles length swapping when signal is shorter than kernel.

**Parameters**:

- `Signal`: Pointer to the input signal array.

- `siglen`: Length of the signal array.

- `Kernel`: Pointer to the input kernel array.

- `kernlen`: Length of the kernel array.

- `corrvout`: Pointer to the output array for the cross-correlation result.

**Return Value**: Returns success or error code.



## COMPARISON AND SUMMARY

### Key Differences

| Feature | `tiny_corr_f32` | `tiny_ccorr_f32` |
|---------|----------------|------------------|
| **Output Length** | \( L_{\text{out}} = L_s - L_p + 1 \) | \( L_{\text{out}} = L_x + L_y - 1 \) |
| **Length Requirement** | Signal length must be ≥ pattern length | Works with any length (auto-swaps if needed) |
| **Computation Type** | Sliding correlation (valid region only) | Full cross-correlation (includes partial overlaps) |
| **Implementation** | Single-stage nested loop | Three-stage computation |
| **Use Case** | Pattern matching, template detection | Full correlation analysis, signal alignment |

### When to Use Which Function

**Use `tiny_corr_f32` when:**

- You need to find a pattern within a longer signal (pattern matching)

- You only care about positions where the pattern fully overlaps with the signal

- The signal is guaranteed to be longer than the pattern

- You want a more efficient computation for template matching applications

- You need to detect occurrences of a known pattern in a signal

**Example applications:**

- Template matching in image processing

- Pattern detection in time series

- Signal detection and recognition

- Feature matching

**Use `tiny_ccorr_f32` when:**

- You need the complete cross-correlation including partial overlaps

- The lengths of the two sequences can be arbitrary (either can be longer)

- You need to analyze all possible alignments between two signals

- You want to find the best alignment or time delay between signals

- You need the full correlation function for further analysis

**Example applications:**

- Time delay estimation between two signals

- Signal alignment and synchronization

- Full correlation analysis

- When signal lengths are unknown or variable

### Summary

- **`tiny_corr_f32`** is optimized for **pattern matching** scenarios where you search for a shorter pattern in a longer signal. It only computes correlations where the pattern fully overlaps, resulting in a shorter output and faster computation.

- **`tiny_ccorr_f32`** computes the **full cross-correlation** between two sequences of any length, including all partial overlaps. This provides complete correlation information but requires more computation and produces a longer output.

Choose `tiny_corr_f32` for efficient pattern matching, and `tiny_ccorr_f32` when you need comprehensive correlation analysis or when signal lengths are variable.
