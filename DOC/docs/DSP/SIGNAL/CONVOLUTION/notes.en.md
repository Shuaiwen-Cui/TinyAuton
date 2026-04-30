# NOTES

---

## 1. ALGORITHM PRINCIPLES

Convolution is a fundamental operation in signal processing that combines two signals to produce a third. Mathematically, it is defined as:

$$y(t) = \int_{-\infty}^{\infty} x(\tau) h(t - \tau) d\tau$$

Where $x(t)$ is the input signal, $h(t)$ is the kernel (impulse response), and $y(t)$ is the output. In the discrete domain:

$$y[n] = \sum_{k=-\infty}^{\infty} x[k] \cdot h[n-k]$$

The kernel $h$ is **flipped** (time-reversed) and then **slid** across the signal $x$. At each shift position, the overlapping values are multiplied and summed.

![](https://www.brandonrohrer.com/images/conv1d/aa_copy.gif)

<div class="grid cards" markdown>

-   :fontawesome-brands-bilibili:{ .lg .middle } __Physical Meaning of Convolution （Chinese）__

    ---

    [:octicons-arrow-right-24: <a href="https://www.bilibili.com/video/BV1L3411c7Yt/?spm_id_from=333.337.search-card.all.click&vd_source=5a427660f0337fedc22d4803661d493f" target="_blank"> Portal </a>](#)

</div>

### 1.1 Full Convolution (default output)

The full convolution between a signal of length $L_s$ and a kernel of length $L_k$ produces $L_s + L_k - 1$ output points. The computation naturally proceeds in three stages:

- **Stage I** (ramp-up): the kernel is only partially overlapping the left edge of the signal.
- **Stage II** (steady-state): the kernel is fully inside the signal.
- **Stage III** (ramp-down): the kernel slides past the right edge.

### 1.2 Padding Modes

When the kernel extends beyond the signal boundaries, padding provides synthetic samples:

| Mode | Behaviour |
|------|-----------|
| **Zero** (`TINY_PADDING_ZERO`) | Missing samples are treated as 0.0 |
| **Symmetric** (`TINY_PADDING_SYMMETRIC`) | Signal is mirrored at the edge |
| **Periodic** (`TINY_PADDING_PERIODIC`) | Signal wraps around (circular) |

### 1.3 Output Modes

| Mode | Output Length | Description |
|------|--------------|-------------|
| **Full** (`TINY_CONV_FULL`) | `siglen + kernlen - 1` | Complete convolution result |
| **Head** (`TINY_CONV_HEAD`) | `kernlen` | First `kernlen` points |
| **Center** (`TINY_CONV_CENTER`) | `siglen` | Centered portion matching signal length |
| **Tail** (`TINY_CONV_TAIL`) | `kernlen` | Last `kernlen` points |

---

## 2. CODE DESIGN PHILOSOPHY

### 2.1 Dual-Path Architecture (ESP32 vs Generic)

This is the only convolution module that uses **platform-specific acceleration**. On ESP32, `dsps_conv_f32` from the ESP-DSP library provides hardware-optimized convolution. The generic fallback uses a pure-C three-stage implementation that handles **arbitrary signal/kernel ordering**.

### 2.2 Three-Stage Generic Convolution

The generic convolution is structured as three explicit loops (Stage I/II/III). This form:
- Is slightly more readable than a single-loop with min/max clamping.
- Allows each stage to be optimized independently by the compiler.
- Correctly handles both `siglen >= kernlen` and `siglen < kernlen` (the C standard doesn't restrict relative sizes).

**Important platform asymmetry**: The ESP32 ESP-DSP backend **requires** `siglen >= kernlen` and returns `TINY_ERR_DSP_INVALID_PARAM` if this precondition is violated. The generic fallback has no such restriction.

### 2.3 `const` Correctness

Both signal and kernel pointers are treated as read-only (`const float *`) in the generic path. This prevents accidental modification and allows the compiler to perform better alias analysis.

### 2.4 `memmove` for Result Slicing

`tiny_conv_ex_f32` selects a sub-slice of the full convolution output by shifting data to the front of the output buffer. It uses **`memmove`** rather than a `for` loop or `memcpy` because the source and destination ranges may overlap (e.g., in Center mode where `start_idx > 0`).

### 2.5 Dynamic Allocation in Extended Mode

`tiny_conv_ex_f32` allocates a padded temporary buffer via `calloc`. This is necessary because padding adds `2 * (kernlen - 1)` extra samples. The buffer is freed before return. For real-time or memory-constrained applications, use `tiny_conv_f32` which performs no dynamic allocation.

---

## 3. API INTERFACE — METHODS

### 3.1 `tiny_conv_f32`

```c
/**
 * @name: tiny_conv_f32
 * @brief Convolution function (full mode, zero padding implicit)
 *
 * @param Signal The input signal array
 * @param siglen The length of the input signal array (> 0)
 * @param Kernel The input kernel array
 * @param kernlen The length of the input kernel array (> 0)
 * @param convout The output array for the convolution result.
 *                Caller MUST provide at least (siglen + kernlen - 1) elements.
 *
 * @note On ESP32 the underlying ESP-DSP routine additionally requires
 *       siglen >= kernlen; the generic fallback handles both orderings.
 *
 * @return tiny_error_t
 */
tiny_error_t tiny_conv_f32(const float *Signal, const int siglen,
                            const float *Kernel, const int kernlen,
                            float *convout)
{
    if (NULL == Signal || NULL == Kernel || NULL == convout)
        return TINY_ERR_DSP_NULL_POINTER;

    if (siglen <= 0 || kernlen <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    /* ESP-DSP requires siglen >= kernlen */
    if (siglen < kernlen)
        return TINY_ERR_DSP_INVALID_PARAM;
    dsps_conv_f32(Signal, siglen, Kernel, kernlen, convout);
#else
    const float *sig  = Signal;
    const float *kern = Kernel;
    int lsig  = siglen;
    int lkern = kernlen;

    // Stage I — ramp-up (kernel partially overlapping left edge)
    for (int n = 0; n < lkern; n++)
    {
        convout[n] = 0;
        for (int k = 0; k <= n; k++)
            convout[n] += sig[k] * kern[n - k];
    }

    // Stage II — steady-state (kernel fully inside signal)
    for (int n = lkern; n < lsig; n++)
    {
        convout[n] = 0;
        int kmin = n - lkern + 1;
        int kmax = n;
        for (int k = kmin; k <= kmax; k++)
            convout[n] += sig[k] * kern[n - k];
    }

    // Stage III — ramp-down (kernel sliding past right edge)
    for (int n = lsig; n < lsig + lkern - 1; n++)
    {
        convout[n] = 0;
        int kmin = n - lkern + 1;
        int kmax = lsig - 1;
        for (int k = kmin; k <= kmax; k++)
            convout[n] += sig[k] * kern[n - k];
    }
#endif

    return TINY_OK;
}
```

**Description**:  

Computes the full convolution between an input signal and a kernel. On ESP32, delegates to the hardware-accelerated `dsps_conv_f32`. On all other platforms, uses a three-stage pure-C implementation.

**Features**:

- Zero-copy: no dynamic memory allocation.
- Platform-accelerated on ESP32.
- Generic fallback supports arbitrary signal/kernel length ordering.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `Signal` | `const float*` | Pointer to the input signal array |
| `siglen` | `int` | Length of the input signal (> 0) |
| `Kernel` | `const float*` | Pointer to the kernel array |
| `kernlen` | `int` | Length of the kernel (> 0) |
| `convout` | `float*` | Output buffer for convolution result. Must hold at least `siglen + kernlen - 1` elements |

**Return Value**:

| Code | Meaning |
|------|---------|
| `TINY_OK` | Convolution succeeded |
| `TINY_ERR_DSP_NULL_POINTER` | `Signal`, `Kernel`, or `convout` is null |
| `TINY_ERR_DSP_INVALID_PARAM` | `siglen ≤ 0`, `kernlen ≤ 0`, or (ESP32 only) `siglen < kernlen` |

---

### 3.2 `tiny_conv_ex_f32`

```c
/**
 * @name: tiny_conv_ex_f32
 * @brief Extended convolution function with padding and mode options
 *
 * @param Signal The input signal array
 * @param siglen The length of the input signal array (> 0)
 * @param Kernel The input kernel array
 * @param kernlen The length of the input kernel array (> 0)
 * @param convout The output buffer for the convolution result.
 *                Must hold at least (siglen + kernlen - 1) elements
 * @param padding_mode Padding mode (zero, symmetric, periodic)
 * @param conv_mode Convolution mode (full, head, center, tail)
 *
 * @return tiny_error_t
 */
tiny_error_t tiny_conv_ex_f32(const float *Signal, const int siglen,
                              const float *Kernel, const int kernlen,
                              float *convout,
                              tiny_padding_mode_t padding_mode,
                              tiny_conv_mode_t conv_mode)
{
    if (NULL == Signal || NULL == Kernel || NULL == convout)
        return TINY_ERR_DSP_NULL_POINTER;

    if (siglen <= 0 || kernlen <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    /* Fast path: delegate to hardware only when all three conditions hold */
    if (padding_mode == TINY_PADDING_ZERO &&
        conv_mode == TINY_CONV_FULL &&
        siglen >= kernlen)
    {
        dsps_conv_f32(Signal, siglen, Kernel, kernlen, convout);
        return TINY_OK;
    }
#endif

    int pad_len = kernlen - 1;
    int padded_len = siglen + 2 * pad_len;
    float *padded_signal = (float *)calloc(padded_len, sizeof(float));
    if (padded_signal == NULL)
        return TINY_ERR_DSP_MEMORY_ALLOC;

    switch (padding_mode)
    {
    case TINY_PADDING_ZERO:
        memcpy(padded_signal + pad_len, Signal, sizeof(float) * siglen);
        break;
    case TINY_PADDING_SYMMETRIC:
        for (int i = 0; i < pad_len; i++)
        {
            padded_signal[pad_len - 1 - i] = Signal[i];
            padded_signal[pad_len + siglen + i] = Signal[siglen - 1 - i];
        }
        memcpy(padded_signal + pad_len, Signal, sizeof(float) * siglen);
        break;
    case TINY_PADDING_PERIODIC:
        for (int i = 0; i < pad_len; i++)
        {
            padded_signal[pad_len - 1 - i] = Signal[(siglen - pad_len + i) % siglen];
            padded_signal[pad_len + siglen + i] = Signal[i % siglen];
        }
        memcpy(padded_signal + pad_len, Signal, sizeof(float) * siglen);
        break;
    default:
        free(padded_signal);
        return TINY_ERR_DSP_INVALID_PARAM;
    }

    // Full convolution on padded signal
    int convlen_full = siglen + kernlen - 1;
    for (int n = 0; n < convlen_full; n++)
    {
        float sum = 0.0f;
        for (int k = 0; k < kernlen; k++)
            sum += padded_signal[n + k] * Kernel[kernlen - 1 - k];
        convout[n] = sum;
    }

    free(padded_signal);

    if (conv_mode == TINY_CONV_FULL)
        return TINY_OK;

    int start_idx = 0;
    int out_len = 0;
    switch (conv_mode)
    {
    case TINY_CONV_HEAD:   start_idx = 0;               out_len = kernlen; break;
    case TINY_CONV_CENTER: start_idx = (kernlen-1)/2;   out_len = siglen;  break;
    case TINY_CONV_TAIL:   start_idx = siglen - 1;      out_len = kernlen; break;
    default: return TINY_ERR_DSP_INVALID_MODE;
    }

    /* memmove: source and destination may overlap in-place */
    if (start_idx > 0 && out_len > 0)
        memmove(convout, convout + start_idx, sizeof(float) * (size_t)out_len);

    return TINY_OK;
}
```

**Description**:

Computes convolution with explicit control over padding strategy and output slice selection. The full convolution is always computed first on a padded version of the signal; the requested output mode is then extracted via in-place `memmove`.

**Features**:

- Three padding modes: zero, symmetric, periodic.
- Four output modes: full, head, center, tail.
- ESP32 fast path only when all three conditions are met (zero + full + siglen >= kernlen).
- Uses `memmove` for safe overlapping slice extraction.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `Signal` | `const float*` | Pointer to the input signal array |
| `siglen` | `int` | Length of the input signal (> 0) |
| `Kernel` | `const float*` | Pointer to the kernel array |
| `kernlen` | `int` | Length of the kernel (> 0) |
| `convout` | `float*` | Output buffer. Must hold at least `siglen + kernlen - 1` elements regardless of `conv_mode` |
| `padding_mode` | `tiny_padding_mode_t` | Padding strategy: `TINY_PADDING_ZERO`, `TINY_PADDING_SYMMETRIC`, or `TINY_PADDING_PERIODIC` |
| `conv_mode` | `tiny_conv_mode_t` | Output slice: `TINY_CONV_FULL`, `TINY_CONV_HEAD`, `TINY_CONV_CENTER`, or `TINY_CONV_TAIL` |

**Return Value**:

| Code | Meaning |
|------|---------|
| `TINY_OK` | Convolution succeeded |
| `TINY_ERR_DSP_NULL_POINTER` | `Signal`, `Kernel`, or `convout` is null |
| `TINY_ERR_DSP_INVALID_PARAM` | `siglen ≤ 0` or `kernlen ≤ 0` |
| `TINY_ERR_DSP_MEMORY_ALLOC` | `calloc` for padded signal failed |
| `TINY_ERR_DSP_INVALID_MODE` | Unknown `conv_mode` value |

---

## 4. FUNCTION COMPARISON

| Feature | `tiny_conv_f32` | `tiny_conv_ex_f32` |
|---------|----------------|-------------------|
| **Padding Mode** | Zero padding only (implicit) | Zero, symmetric, or periodic (explicit) |
| **Output Mode** | Full convolution only | Full, head, center, or tail |
| **Output Length** | `siglen + kernlen - 1` | Configurable based on `conv_mode` |
| **Memory Allocation** | None (zero-copy) | `calloc` for padded buffer (`2 × pad_len + siglen` floats) |
| **ESP32 Fast Path** | Always (if `siglen >= kernlen`) | Only when zero + full + `siglen >= kernlen` |
| **Signal/Kernel Order** | Generic path: any; ESP32: `siglen >= kernlen` | Generic path: any; ESP32 fast path: `siglen >= kernlen` |
| **Boundary Handling** | Zero-padding implicit in 3-stage loops | Explicit padding (zero/symmetric/periodic) |
| **Use When** | Simple full convolution, real-time, no allocation | Custom padding or output slice needed |

### When to Use `tiny_conv_f32`

- You need a simple full convolution result.
- Zero padding at boundaries is acceptable.
- You want maximum performance (ESP32 hardware acceleration).
- You want to avoid dynamic memory allocation entirely.
- Real-time or interrupt-context processing.

### When to Use `tiny_conv_ex_f32`

- You need symmetric or periodic padding to reduce boundary artifacts.
- You want to extract only a specific slice (head/center/tail) of the result.
- You need output length equal to input signal length (center mode) without manual slicing.
- You are processing signals where boundary handling matters (image filtering, circular convolution, periodic signals).
- You can tolerate a one-time `calloc`/`free` per call.

---

## 5. ⚠️ IMPORTANT NOTES

### 5.1 Output Buffer Minimum Size

Regardless of `conv_mode`, the output buffer for both functions must be sized for the **full convolution**:

$$\text{buffer\_size} \ge siglen + kernlen - 1$$

In `tiny_conv_ex_f32`, the full result is written first and then the requested slice is shifted to the front. A buffer smaller than `siglen + kernlen - 1` will cause a heap buffer overflow.

### 5.2 ESP32 Signal/Kernel Length Ordering

On ESP32, `dsps_conv_f32` requires `siglen >= kernlen`. If this precondition is violated:
- `tiny_conv_f32` returns `TINY_ERR_DSP_INVALID_PARAM`.
- `tiny_conv_ex_f32` falls through to the generic path (which handles any ordering).

**Recommendation**: always ensure `siglen >= kernlen` for portable code.

### 5.3 Dynamic Allocation in Extended Mode

`tiny_conv_ex_f32` calls `calloc` internally. If the allocation fails:
- Returns `TINY_ERR_DSP_MEMORY_ALLOC`.
- The output buffer is untouched.

In memory-constrained or real-time systems, prefer `tiny_conv_f32` or pre-allocate the padded buffer externally.

### 5.4 `memmove` vs `memcpy`

The slice extraction in `tiny_conv_ex_f32` uses `memmove` (not `memcpy`) because the source and destination ranges overlap. For example, in Center mode, bytes at `convout[start_idx]` are copied to `convout[0]`, and these regions overlap when `start_idx < out_len`.

### 5.5 Padding Modes and Signal Length

- Symmetric and periodic padding both read `Signal[0]` and `Signal[siglen-1]` for mirror/wrap operations. If `siglen == 1`, the left and right padding edges reference the same single sample.
- Periodic mode uses `% siglen` — division by zero if `siglen == 0`, but this is caught by the `siglen <= 0` parameter check.

### 5.6 Convolution is NOT Correlation

Convolution flips the kernel; correlation does not. If you need correlation, use `tiny_corr_f32` or `tiny_ccorr_f32` from the CORRELATION module — do not simply reverse the kernel manually, as the internal implementations differ in boundary handling.

