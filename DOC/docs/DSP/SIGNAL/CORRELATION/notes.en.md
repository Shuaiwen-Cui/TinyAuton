# NOTES

!!! note "Note"
    Correlation is an important concept in signal processing, often used to analyze similarities or dependencies between signals. It is useful in many applications, such as pattern recognition, time series analysis, and signal detection. This library provides two levels of correlation: sliding correlation (`tiny_corr_f32`) and full cross-correlation (`tiny_ccorr_f32`).

---

## 1. ALGORITHM PRINCIPLES

### 1.1 Sliding Correlation

The sliding correlation (pattern matching) slides a short pattern across a longer signal, computing the dot product at each valid position where the pattern fully overlaps:

\[
\text{Correlation}[n] = \sum_{m=0}^{L_p - 1} S[n + m] \cdot P[m]
\]

Where:

- \(S\) is the input signal of length \(L_s\)
- \(P\) is the pattern of length \(L_p\)
- \(n \in [0, L_s - L_p]\)

**Output Length**: \(L_{out} = L_s - L_p + 1\)

**Unlike convolution, the pattern is NOT flipped** — it is used as-is, sliding across the signal.

### 1.2 Full Cross-Correlation

Full cross-correlation computes the correlation between two sequences at ALL possible shift positions, including partial overlaps:

\[
R_{xy}[n] = \sum_{k} x[k] \cdot y[k + n]
\]

Where:

- \(x\) is the longer sequence (length \(L_x\))
- \(y\) is the shorter sequence (length \(L_y\))
- \(n \in [0, L_x + L_y - 2]\)

**Output Length**: \(L_{out} = L_x + L_y - 1\)

Like convolution, this computation naturally proceeds in three stages (warm-up, steady-state, cool-down).

---

## 2. CODE DESIGN PHILOSOPHY

### 2.1 Two Different Correlation Semantics

`tiny_corr_f32` and `tiny_ccorr_f32` serve fundamentally different purposes:

- **Sliding correlation** only returns positions where the pattern is fully inside the signal — useful for template matching.
- **Full cross-correlation** returns all possible shift alignments including partial overlaps — useful for signal alignment, time-delay estimation, and full correlation analysis.

### 2.2 Length Swap in Generic Cross-Correlation

`tiny_ccorr_f32` swaps signal and kernel so `lsig >= lkern` in the **generic fallback path only** (`#else`). This is because:

- The ESP32 ESP-DSP backend (`dsps_ccorr_f32`) expects the original order.
- The three-stage loop indexing assumes the longer sequence is the "signal".
- On ESP32, the caller must ensure `siglen >= kernlen` (or the backend may misbehave).

### 2.3 Const Correctness

In the generic paths, local pointers are declared `const float *sig`, `const float *kern`, preserving the constness of the input arrays. This is consistent with the read-only semantics and helps the compiler optimize.

---

## 3. API INTERFACE — METHODS

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

**Description**: 

Computes the sliding correlation between a longer signal and a shorter pattern. Only positions where the pattern fully overlaps are computed. The pattern is NOT flipped — this is direct element-wise multiply-accumulate at each shift position.

**Features**:

- Pattern matching without flipping
- Platform-accelerated on ESP32
- O(siglen × patlen) time complexity

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `Signal` | `const float*` | Pointer to the input signal array |
| `siglen` | `int` | Length of the signal (> 0) |
| `Pattern` | `const float*` | Pointer to the pattern array |
| `patlen` | `int` | Length of the pattern (> 0) |
| `dest` | `float*` | Output buffer, must hold at least `siglen - patlen + 1` elements |

**Return Value**:

| Code | Meaning |
|------|---------|
| `TINY_OK` | Correlation succeeded |
| `TINY_ERR_DSP_NULL_POINTER` | `Signal`, `Pattern`, or `dest` is null |
| `TINY_ERR_DSP_INVALID_PARAM` | `siglen ≤ 0` or `patlen ≤ 0` |
| `TINY_ERR_DSP_MISMATCH` | `siglen < patlen` (pattern longer than signal) |

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

    /* Make the longer one the "signal" for correct three-stage indexing */
    if (siglen < kernlen)
    {
        sig  = Kernel;
        kern = Signal;
        lsig  = kernlen;
        lkern = siglen;
    }

    // Stage I — warm-up (kernel partially overlapping left edge)
    for (int n = 0; n < lkern; n++)
    {
        size_t kmin = lkern - 1 - n;
        corrvout[n] = 0;
        for (size_t k = 0; k <= n; k++)
            corrvout[n] += sig[k] * kern[kmin + k];
    }

    // Stage II — steady-state (kernel fully inside signal)
    for (int n = lkern; n < lsig; n++)
    {
        corrvout[n] = 0;
        size_t kmin = n - lkern + 1;
        for (size_t k = kmin; k <= n; k++)
            corrvout[n] += sig[k] * kern[k - kmin];
    }

    // Stage III — cool-down (kernel sliding past right edge)
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

**Description**: 

Computes the full cross-correlation between two sequences, including all partial overlap positions. Internally swaps operands in the generic path so the longer sequence is always the "signal".

**Features**:

- Full cross-correlation including partial overlaps
- Automatic length ordering in generic path
- ESP32 hardware acceleration

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `Signal` | `const float*` | Pointer to the input signal array |
| `siglen` | `int` | Length of the signal (> 0) |
| `Kernel` | `const float*` | Pointer to the kernel array |
| `kernlen` | `int` | Length of the kernel (> 0) |
| `corrvout` | `float*` | Output buffer, must hold at least `siglen + kernlen - 1` elements |

**Return Value**:

| Code | Meaning |
|------|---------|
| `TINY_OK` | Cross-correlation succeeded |
| `TINY_ERR_DSP_NULL_POINTER` | `Signal`, `Kernel`, or `corrvout` is null |
| `TINY_ERR_DSP_INVALID_PARAM` | `siglen ≤ 0` or `kernlen ≤ 0` |

---

## 4. FUNCTION COMPARISON

| Feature | `tiny_corr_f32` | `tiny_ccorr_f32` |
|---------|----------------|------------------|
| **Output Length** | \(L_s - L_p + 1\) | \(L_x + L_y - 1\) |
| **Overlap Type** | Full overlap only | All shifts (partial included) |
| **Length Requirement** | `siglen ≥ patlen` (enforced) | Any (auto-swap in generic path) |
| **Implementation** | Single nested loop | Three-stage computation |
| **ESP32 External Dep.** | `dsps_corr_f32` | `dsps_ccorr_f32` |
| **Use Case** | Pattern matching / template detection | Time delay estimation / signal alignment |

### When to Use `tiny_corr_f32`

- You are searching for a known pattern within a longer signal
- You only care about positions where the pattern is fully inside
- You want the most efficient computation (no partial overlap overhead)
- Example: detecting a preamble in a received bitstream

### When to Use `tiny_ccorr_f32`

- You need cross-correlation including partial overlaps
- You want to estimate time delay between two signals
- You need to align two signals of potentially different lengths
- Example: computing time-of-flight from two sensor readings

---

## 5. ⚠️ IMPORTANT NOTES

### 5.1 Correlation vs Convolution

Correlation does NOT flip the kernel/pattern. Convolution flips. If you accidentally think you can use convolution with a reversed kernel for correlation, note that:
- Boundary handling differs between the two implementations.
- Use `tiny_corr_f32` or `tiny_ccorr_f32` explicitly for correlation — do not misuse the convolution functions.

### 5.2 Output Buffer Minimum Size

| Function | Minimum Output Buffer Size |
|----------|---------------------------|
| `tiny_corr_f32` | `siglen - patlen + 1` |
| `tiny_ccorr_f32` | `siglen + kernlen - 1` |

### 5.3 `siglen < patlen` Handling

- `tiny_corr_f32`: returns `TINY_ERR_DSP_MISMATCH` immediately (pattern must fit in signal).
- `tiny_ccorr_f32`: in the generic path, automatically swaps operands. On ESP32, `dsps_ccorr_f32` may handle or reject this depending on its internal implementation — for portability, ensure `siglen >= kernlen` or test on your target.

### 5.4 Platform Differences

On ESP32, both functions delegate to the ESP-DSP library. The generic fallback is used on all other platforms. The behavioral guarantee for length ordering differs:

- `tiny_ccorr_f32` generic path: always works regardless of which is longer.
- `tiny_ccorr_f32` ESP32 path: behavior depends on `dsps_ccorr_f32` — test with your specific ESP-IDF version.
