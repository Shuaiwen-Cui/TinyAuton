# NOTES

!!! note "Note"
    Resampling is an important step in signal processing, typically used to change the sampling rate of a signal. It can be used in audio, video, and other types of signal processing. This library provides three levels of resampling: downsampling via keep/skip patterns, upsampling via zero-insertion, and arbitrary-factor resampling via linear interpolation.

---

## 1. ALGORITHM PRINCIPLES

### 1.1 Downsampling by Keep/Skip

Instead of simple stride-based decimation (keep 1, skip N), the library implements a **keep‑skip pattern**: the user specifies how many consecutive samples to keep (`keep`) and how many to skip (`skip`) in each cycle.

**Pattern**:
```
Input:  [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]
keep=1, skip=1  →  output = [s0, s2, s4, s6, s8]        (stride 2)
keep=2, skip=1  →  output = [s0, s1, s3, s4, s6, s7, s9] (period 3)
keep=3, skip=2  →  output = [s0, s1, s2, s5, s6, s7]     (period 5)
```

This design subsumes simple decimation (`keep=1`) while allowing grouped retention for applications like block averaging pre‑processing.

**Output length**: dynamically determined, approximately

\\[
L_{out} \\approx \\left\\lceil L_{in} \\cdot \\frac{keep}{keep + skip} \\right\\rceil
\\]

### 1.2 Upsampling by Zero-Insertion

Given an integer expansion factor \\(F = target\\_len / input\\_len\\), the output is formed by placing the original samples at positions that are multiples of \\(F\\) and zero elsewhere:

\\[
output[i] = \\begin{cases}
input[i / F] & \\text{if } i \\bmod F = 0 \\text{ and } i/F < input\\_len \\\\
0 & \\text{otherwise}
\\end{cases}
\\]

When `target_len` is not an exact multiple of `input_len`, trailing positions beyond the last valid source sample are zero-filled.

### 1.3 Arbitrary-Factor Resampling via Linear Interpolation

For arbitrary up/down‑sampling (non‑integer ratios), the library uses **linear interpolation**:

1. Compute the ratio: \\(r = \\frac{\\mathrm{target}_{\\mathrm{len}}}{\\mathrm{input}_{\\mathrm{len}}}\\)
2. For each output index \\(i\\), find the corresponding floating‑point position in the input:
   \\[
   pos = i / r
   \\]
3. Split into integer index \\(idx = \\lfloor pos \\rfloor\\) and fractional part \\(frac = pos - idx\\).
4. Linearly blend the two nearest input samples:
   \\[
   output[i] = input[idx] \\cdot (1 - frac) + input[idx+1] \\cdot frac
   \\]
5. Clamp at the end: if \\(idx \\ge \\mathrm{input}_{\\mathrm{len}} - 1\\), use `input[input_len - 1]`.

This is a lightweight, O(N) method. It does **not** include anti‑aliasing filtering — see the notes section for caveats.

---

## 2. CODE DESIGN PHILOSOPHY

### 2.1 Flexible Keep/Skip Downsampling

Simple stride‑based decimation (`keep=1`) drops entire blocks of samples regardless of signal structure. The keep‑skip pattern allows the user to retain consecutive groups, which is useful when:
- Each "chunk" of the signal carries meaning (e.g., packetized data).
- You want to mimic non‑rectangular windowing before decimation.
- You need to control the preservation of short‑duration events.

### 2.2 Zero-Insertion as a Building Block

Zero‑insertion is intentionally separated from interpolation filtering. This gives the user control over:
- Which interpolation filter to apply afterwards (e.g., a low‑pass FIR kernel).
- Whether to cascade with `tiny_conv_f32` for proper interpolation.
- Keeping the upsampling step itself allocation-free and fast.

### 2.3 Linear Interpolation for Simplicity

On resource‑constrained MCUs, full polyphase resampling is expensive. Linear interpolation provides:
- O(target_len) time, O(1) auxiliary memory.
- Acceptable quality when the input is well‑oversampled relative to its bandwidth.
- A predictable performance profile (no dynamic allocation).

### 2.4 Boundary Protection

All three functions include guard logic for under/overflow:
- `tiny_downsample_skip_f32`: `copy_n = min(keep, input_len - in_idx)` prevents over-reading.
- `tiny_upsample_zero_f32`: `src < input_len` prevents out‑of‑bounds access when `target_len` is not a perfect multiple.
- `tiny_resample_f32`: clamping `index >= input_len - 1` prevents reading past the array end.

---

## 3. API INTERFACE — METHODS

### 3.1 `tiny_downsample_skip_f32`

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
tiny_error_t tiny_downsample_skip_f32(const float *input, int input_len,
                                       float *output, int *output_len,
                                       int keep, int skip)
{
    if (!input || !output || !output_len)
        return TINY_ERR_DSP_NULL_POINTER;

    if (input_len <= 0 || keep <= 0 || skip <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    int in_idx = 0;
    int out_idx = 0;

    while (in_idx < input_len)
    {
        // Keep 'keep' samples
        int copy_n = keep;
        if (in_idx + copy_n > input_len)
            copy_n = input_len - in_idx;

        for (int i = 0; i < copy_n; i++)
        {
            output[out_idx++] = input[in_idx++];
        }

        // Skip 'skip' samples
        in_idx += skip;
    }

    *output_len = out_idx;

    return TINY_OK;
}
```

**Description**:

Downsamples a signal by alternately copying `keep` consecutive samples and skipping the next `skip` samples. This repeats until the input is exhausted.

**Features**:

- Generalized keep‑skip pattern (not just stride‑based decimation)
- Handles incomplete final blocks cleanly
- Output length reported back via `output_len`

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `const float*` | Pointer to the input signal array |
| `input_len` | `int` | Length of the input signal (> 0) |
| `output` | `float*` | Pointer to the output buffer. Caller must size it to at least `ceil(input_len * keep / (keep + skip))` |
| `output_len` | `int*` | [out] Number of samples actually written |
| `keep` | `int` | Number of consecutive samples to keep in each cycle (≥ 1) |
| `skip` | `int` | Number of consecutive samples to skip in each cycle (≥ 1) |

**Return Value**:

| Code | Meaning |
|------|---------|
| `TINY_OK` | Downsampling succeeded |
| `TINY_ERR_DSP_NULL_POINTER` | `input`, `output`, or `output_len` is null |
| `TINY_ERR_DSP_INVALID_PARAM` | `input_len ≤ 0`, `keep ≤ 0`, or `skip ≤ 0` |

---

### 3.2 `tiny_upsample_zero_f32`

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
tiny_error_t tiny_upsample_zero_f32(const float *input, int input_len,
                                     float *output, int target_len)
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
        if (i % factor == 0)
        {
            int src = i / factor;
            if (src < input_len)
                output[i] = input[src];
            else
                output[i] = 0.0f;  // Past the end → zero-fill
        }
        else
        {
            output[i] = 0.0f;
        }
    }

    return TINY_OK;
}
```

**Description**:

Upsamples a signal by inserting zeros between the original samples. The expansion factor is `target_len / input_len` (integer division). When `target_len` is not an exact multiple of `input_len`, trailing positions whose source index would fall past the end of the input are zero‑filled.

**Features**:

- Integer‑factor zero‑insertion
- Boundary‑safe: protects against out‑of‑bounds access on non‑exact multiples
- Allocation‑free

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `const float*` | Pointer to the input signal array |
| `input_len` | `int` | Length of the input signal (> 0) |
| `output` | `float*` | Pointer to the output buffer. Size must be at least `target_len` |
| `target_len` | `int` | Target length for the output signal (> 0) |

**Return Value**:

| Code | Meaning |
|------|---------|
| `TINY_OK` | Upsampling succeeded |
| `TINY_ERR_DSP_NULL_POINTER` | `input` or `output` is null |
| `TINY_ERR_DSP_INVALID_PARAM` | `input_len ≤ 0`, `target_len ≤ 0`, or `factor = target_len / input_len ≤ 0` |

---

### 3.3 `tiny_resample_f32`

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

**Description**:

Resamples a signal to a target length using linear interpolation. Supports arbitrary up‑sampling and down‑sampling ratios — non‑integer factors are handled naturally by the interpolation formula.

**Features**:

- Non‑integer factor resampling
- Linear interpolation (computationally lightweight)
- No dynamic memory allocation
- End‑clamping prevents out‑of‑bounds access

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `const float*` | Pointer to the input signal array |
| `input_len` | `int` | Length of the input signal (> 0) |
| `output` | `float*` | Pointer to the output buffer. Size must be at least `target_len` |
| `target_len` | `int` | Target length for the output signal (> 0) |

**Return Value**:

| Code | Meaning |
|------|---------|
| `TINY_OK` | Resampling succeeded |
| `TINY_ERR_DSP_NULL_POINTER` | `input` or `output` is null |
| `TINY_ERR_DSP_INVALID_PARAM` | `input_len ≤ 0` or `target_len ≤ 0` |

---

## 4. FUNCTION COMPARISON

| Feature | `tiny_downsample_skip_f32` | `tiny_upsample_zero_f32` | `tiny_resample_f32` |
|---------|---------------------------|-------------------------|---------------------|
| **Direction** | Reduce sample rate | Increase sample rate | Either direction |
| **Factor Type** | Integer (keep+skip pattern) | Integer (factor = target/input) | Any (non‑integer OK) |
| **Interpolation** | None (skip samples) | None (zero‑insertion) | Linear interpolation |
| **Anti‑Aliasing** | Not included | Not included | Not included |
| **Memory Allocation** | No | No | No |
| **Output Length** | Dynamic, reported back | User‑specified `target_len` | User‑specified `target_len` |
| **Boundary Handling** | Truncates incomplete cycles | Zero‑fills trailing | Clamps at end |
| **Use When** | You know the keep/skip pattern | You need a zero‑filled upsampled version for further filtering | You need a quick resample to any length |

### When to Use `tiny_downsample_skip_f32`

- You need to reduce sample rate while retaining blocks of samples
- You want a non‑uniform decimation pattern (e.g., keep 2 of every 3)
- The keep‑skip pattern aligns with your data's structure

### When to Use `tiny_upsample_zero_f32`

- You need to increase sample rate as a first step in interpolation
- You plan to apply a reconstruction filter afterwards
- The expansion factor is an integer

### When to Use `tiny_resample_f32`

- The ratio is non‑integer
- You need a one‑step resample without external filtering
- The signal is well‑oversampled (to avoid aliasing artifacts)

---

## 5. ⚠️ IMPORTANT NOTES

### 5.1 Downsampling — Aliasing Risk

`tiny_downsample_skip_f32` performs **pure selection** without anti‑aliasing filtering. If the input signal contains frequency components above the new Nyquist frequency (\\(f_s' / 2 = f_s / (2 \\cdot stride)\\)), aliasing will occur. **Pre‑filter the signal with a low‑pass filter** (e.g., `tiny_fir_filter_f32`) before calling this function if your signal has significant high‑frequency content.

### 5.2 Upsampling — Zero‑Filled Spectrum

`tiny_upsample_zero_f32` inserts zeros, creating spectral images. A reconstruction low‑pass filter (interpolation filter) is typically required afterwards. This function is intended as a building block; pair it with `tiny_conv_f32` or `tiny_fir_filter_f32` for complete interpolation.

### 5.3 Resampling — Linear Interpolation Limits

`tiny_resample_f32` uses pure linear interpolation:

- **No anti‑aliasing**: downsampling may introduce aliasing artifacts.
- **No anti‑imaging**: upsampling may have stair‑step artifacts.
- Quality is acceptable when the signal is heavily oversampled (e.g., 10× the Nyquist rate).
- For high‑quality resampling, pre‑filter (for down) or post‑filter (for up) with a proper FIR/IIR low‑pass filter.

### 5.4 Output Buffer Sizing

| Function | Minimum Output Buffer Size |
|----------|---------------------------|
| `tiny_downsample_skip_f32` | `ceil(input_len * keep / (keep + skip))` |
| `tiny_upsample_zero_f32` | `target_len` |
| `tiny_resample_f32` | `target_len` |

For `tiny_downsample_skip_f32`, the exact output length is returned via `*output_len`.

### 5.5 Factor Validity

- `tiny_downsample_skip_f32`: both `keep ≥ 1` and `skip ≥ 1`.
- `tiny_upsample_zero_f32`: `target_len / input_len ≥ 1` (must be ≥ 1; zero‑insertion only, shrinking is not supported here).
- `tiny_resample_f32`: any `target_len > 0` (up or down).

### 5.6 Platform Independence

All three functions are **platform‑independent** — they are pure C implementations with no `#if ESP32` branches. They operate identically on all supported MCU platforms.



