# NOTES

!!! note "Note"
    Discrete Wavelet Transform (DWT) is a powerful signal processing technique that decomposes signals into different frequency components at multiple resolution levels. Unlike FFT which provides global frequency information, DWT provides both time and frequency localization, making it ideal for analyzing non-stationary signals, denoising, compression, and feature extraction.

## DWT OVERVIEW

### Mathematical Principle

The Discrete Wavelet Transform decomposes a signal into approximation (low-frequency) and detail (high-frequency) coefficients using a pair of filters: a low-pass filter (scaling function) and a high-pass filter (wavelet function).

**Single-Level Decomposition**:

\[
cA[n] = \sum_{k} x[k] \cdot h_0[2n - k]
\]

\[
cD[n] = \sum_{k} x[k] \cdot h_1[2n - k]
\]

Where:

- \( x[k] \) is the input signal

- \( h_0 \) is the low-pass decomposition filter

- \( h_1 \) is the high-pass decomposition filter

- \( cA[n] \) are approximation coefficients (low-frequency)

- \( cD[n] \) are detail coefficients (high-frequency)

**Output Length**:

\[
L_{cA} = L_{cD} = \left\lceil \frac{L_{input}}{2} \right\rceil
\]

**Reconstruction**:

\[
x[n] = \sum_{k} (cA[k] \cdot g_0[n - 2k] + cD[k] \cdot g_1[n - 2k])
\]

Where:

- \( g_0 \) is the low-pass reconstruction filter

- \( g_1 \) is the high-pass reconstruction filter

## WAVELET TYPES

The library supports Daubechies wavelets (DB1 through DB10):

- **DB1 (Haar)**: Simplest wavelet, 2-tap filter, good for edge detection
- **DB2**: 4-tap filter, better frequency resolution than DB1
- **DB3**: 6-tap filter, smoother than DB2
- **DB4**: 8-tap filter, commonly used, good balance
- **DB5-DB10**: Higher order wavelets with better frequency resolution but longer filters

**Filter Length**:

\[
L_{filter} = 2 \times N
\]

Where \( N \) is the wavelet order (DB1: N=1, DB2: N=2, ..., DB10: N=10).

## SINGLE-LEVEL DWT

### tiny_dwt_decompose_f32

```c
/**
 * @name tiny_dwt_decompose_f32
 * @brief Perform single-level discrete wavelet decomposition
 * @param input Input signal array
 * @param input_len Length of the input signal
 * @param wavelet Wavelet type (DB1-DB10)
 * @param cA Output array for approximation coefficients
 * @param cD Output array for detail coefficients
 * @param cA_len Output length of approximation coefficients
 * @param cD_len Output length of detail coefficients
 * @return tiny_error_t
 */
tiny_error_t tiny_dwt_decompose_f32(const float *input, int input_len,
                                    tiny_wavelet_type_t wavelet,
                                    float *cA, float *cD,
                                    int *cA_len, int *cD_len);
```

**Description**: 

Performs single-level discrete wavelet decomposition, splitting the input signal into approximation (low-frequency) and detail (high-frequency) coefficients.

**Features**:

- Uses symmetric padding to handle boundaries

- Performs convolution with low-pass and high-pass filters

- Downsamples by factor of 2 to maintain critical sampling

- Supports all Daubechies wavelets (DB1-DB10)

**Parameters**:

- `input`: Pointer to the input signal array.

- `input_len`: Length of the input signal.

- `wavelet`: Wavelet type from `tiny_wavelet_type_t` enum:
  - `TINY_WAVELET_DB1` through `TINY_WAVELET_DB10`

- `cA`: Pointer to output array for approximation coefficients. Size should be at least `(input_len + 1) / 2`.

- `cD`: Pointer to output array for detail coefficients. Size should be at least `(input_len + 1) / 2`.

- `cA_len`: Pointer to output variable for approximation coefficients length.

- `cD_len`: Pointer to output variable for detail coefficients length.

**Return Value**: 

Returns success or error code.

**Note**: 

The output coefficient arrays are approximately half the length of the input signal. Boundary effects may occur near the signal edges due to convolution operations.

### tiny_dwt_reconstruct_f32

```c
/**
 * @name tiny_dwt_reconstruct_f32
 * @brief Perform single-level discrete wavelet reconstruction
 * @param cA Approximation coefficients array
 * @param cD Detail coefficients array
 * @param coeff_len Length of coefficient arrays (cA and cD must have same length)
 * @param wavelet Wavelet type (DB1-DB10)
 * @param output Output array for reconstructed signal
 * @param output_len Output length of reconstructed signal
 * @return tiny_error_t
 */
tiny_error_t tiny_dwt_reconstruct_f32(const float *cA, const float *cD, int coeff_len,
                                      tiny_wavelet_type_t wavelet,
                                      float *output, int *output_len);
```

**Description**: 

Performs single-level discrete wavelet reconstruction, combining approximation and detail coefficients to reconstruct the original signal.

**Features**:

- Upsamples coefficients by factor of 2

- Performs convolution with reconstruction filters

- Combines low-pass and high-pass reconstruction results

- Perfect reconstruction (within numerical precision) for center region

**Parameters**:

- `cA`: Pointer to approximation coefficients array.

- `cD`: Pointer to detail coefficients array.

- `coeff_len`: Length of both coefficient arrays (must be equal).

- `wavelet`: Wavelet type used for decomposition.

- `output`: Pointer to output array for reconstructed signal. Size should be at least `coeff_len * 2`.

- `output_len`: Pointer to output variable for reconstructed signal length.

**Return Value**: 

Returns success or error code.

**Note**: 

The reconstructed signal length is `coeff_len * 2`. Boundary effects may occur, especially near signal edges. The center region typically has very high reconstruction accuracy.

## MULTI-LEVEL DWT

### tiny_dwt_multilevel_decompose_f32

```c
/**
 * @name tiny_dwt_multilevel_decompose_f32
 * @brief Perform multi-level DWT decomposition
 * @param input Input signal array
 * @param input_len Length of the input signal
 * @param wavelet Wavelet type (DB1-DB10)
 * @param levels Number of decomposition levels
 * @param cA_out Output pointer for final approximation coefficients
 * @param cD_out Output pointer for all detail coefficients (concatenated)
 * @param len_out Output length of final approximation coefficients
 * @return tiny_error_t
 */
tiny_error_t tiny_dwt_multilevel_decompose_f32(const float *input, int input_len,
                                               tiny_wavelet_type_t wavelet, int levels,
                                               float **cA_out, float **cD_out, int *len_out);
```

**Description**: 

Performs multi-level discrete wavelet decomposition, recursively decomposing the approximation coefficients to create a hierarchical representation of the signal.

**Features**:

- Recursive decomposition of approximation coefficients

- Stores all detail coefficients from all levels

- Returns final approximation and concatenated detail coefficients

- Memory is allocated internally and must be freed by caller

**Parameters**:

- `input`: Pointer to the input signal array.

- `input_len`: Length of the input signal.

- `wavelet`: Wavelet type from `tiny_wavelet_type_t` enum.

- `levels`: Number of decomposition levels. Must be positive.

- `cA_out`: Pointer to output pointer for final approximation coefficients. Memory is allocated internally.

- `cD_out`: Pointer to output pointer for all detail coefficients (concatenated from all levels). Memory is allocated internally.

- `len_out`: Pointer to output variable for final approximation coefficients length.

**Return Value**: 

Returns success or error code.

**Memory Management**: 

The function allocates memory for `cA_out` and `cD_out`. The caller is responsible for freeing this memory using `free()`.

**Coefficient Structure**:

For N-level decomposition:

- Level 1: cA1 (length ≈ input_len/2), cD1 (length ≈ input_len/2)

- Level 2: cA2 (length ≈ input_len/4), cD2 (length ≈ input_len/4)

- ...

- Level N: cAN (length ≈ input_len/2^N), cDN (length ≈ input_len/2^N)

The `cD_out` array contains: [cD1, cD2, ..., cDN] concatenated.

### tiny_dwt_multilevel_reconstruct_f32

```c
/**
 * @name tiny_dwt_multilevel_reconstruct_f32
 * @brief Perform multi-level DWT reconstruction
 * @param cA_init Final approximation coefficients from multi-level decomposition
 * @param cD_all All detail coefficients (concatenated from all levels)
 * @param final_len Length of final approximation coefficients
 * @param wavelet Wavelet type (DB1-DB10)
 * @param levels Number of decomposition levels
 * @param output Output array for reconstructed signal
 * @return tiny_error_t
 */
tiny_error_t tiny_dwt_multilevel_reconstruct_f32(const float *cA_init, const float *cD_all,
                                                 int final_len, tiny_wavelet_type_t wavelet, int levels,
                                                 float *output);
```

**Description**: 

Performs multi-level discrete wavelet reconstruction, recursively reconstructing from the final approximation and all detail coefficients.

**Features**:

- Recursive reconstruction starting from final approximation

- Reconstructs each level using corresponding detail coefficients

- Output length matches original input length

- Boundary effects accumulate with decomposition levels

**Parameters**:

- `cA_init`: Pointer to final approximation coefficients from multi-level decomposition.

- `cD_all`: Pointer to all detail coefficients concatenated from all levels.

- `final_len`: Length of final approximation coefficients.

- `wavelet`: Wavelet type used for decomposition.

- `levels`: Number of decomposition levels.

- `output`: Pointer to output array for reconstructed signal. Size should be at least original input length.

**Return Value**: 

Returns success or error code.

**Note**: 

The `cD_all` array should contain detail coefficients in order: [cD_level1, cD_level2, ..., cD_levelN]. Boundary effects become more pronounced with increasing decomposition levels.

## COEFFICIENT PROCESSING

### tiny_dwt_coeffs_process

```c
/**
 * @name tiny_dwt_coeffs_process
 * @brief Placeholder for user-defined coefficient processing
 * @param cA Approximation coefficients
 * @param cD Detail coefficients
 * @param cA_len Length of approximation coefficients
 * @param cD_len Length of detail coefficients
 * @param levels Number of decomposition levels
 */
void tiny_dwt_coeffs_process(float *cA, float *cD, int cA_len, int cD_len, int levels);
```

**Description**: 

Placeholder function for user-defined coefficient processing. This function can be extended by users to implement denoising, thresholding, or other coefficient manipulation operations.

**Parameters**:

- `cA`: Pointer to approximation coefficients (can be modified).

- `cD`: Pointer to detail coefficients (can be modified).

- `cA_len`: Length of approximation coefficients.

- `cD_len`: Length of detail coefficients.

- `levels`: Number of decomposition levels.

**Note**: 

Currently this function does nothing. Users can modify it to implement custom processing such as:

- Hard/soft thresholding for denoising

- Coefficient selection for compression

- Feature extraction

- Anomaly detection

## USAGE WORKFLOW

### Single-Level DWT Workflow

1. **Decompose Signal**:
   ```c
   float input[64];
   float cA[32], cD[32];
   int cA_len, cD_len;
   tiny_dwt_decompose_f32(input, 64, TINY_WAVELET_DB4, cA, cD, &cA_len, &cD_len);
   ```

2. **Process Coefficients** (optional):
   ```c
   // Apply thresholding, denoising, etc.
   ```

3. **Reconstruct Signal**:
   ```c
   float output[64];
   int output_len;
   tiny_dwt_reconstruct_f32(cA, cD, cA_len, TINY_WAVELET_DB4, output, &output_len);
   ```

### Multi-Level DWT Workflow

1. **Multi-Level Decomposition**:
   ```c
   float *cA, *cD;
   int cA_len;
   tiny_dwt_multilevel_decompose_f32(input, 128, TINY_WAVELET_DB4, 3, &cA, &cD, &cA_len);
   ```

2. **Process Coefficients** (optional):
   ```c
   tiny_dwt_coeffs_process(cA, cD, cA_len, 128 - cA_len, 3);
   ```

3. **Multi-Level Reconstruction**:
   ```c
   float output[128];
   tiny_dwt_multilevel_reconstruct_f32(cA, cD, cA_len, TINY_WAVELET_DB4, 3, output);
   ```

4. **Free Memory**:
   ```c
   free(cA);
   free(cD);
   ```

## APPLICATIONS

DWT is widely used in various applications:

- **Signal Denoising**: Threshold detail coefficients to remove noise
- **Data Compression**: Store only significant coefficients
- **Feature Extraction**: Analyze coefficients at different scales
- **Image Processing**: 2D DWT for image compression and analysis
- **Biomedical Signal Processing**: ECG/EEG analysis, artifact removal
- **Structural Health Monitoring**: Vibration analysis, damage detection
- **Time-Frequency Analysis**: Localize events in both time and frequency

## BOUNDARY EFFECTS

DWT operations use symmetric padding to handle signal boundaries. However, boundary effects may still occur:

- **Single-Level**: Boundary effects typically extend ~filter_length samples from each edge
- **Multi-Level**: Boundary effects accumulate and extend ~filter_length × levels samples
- **Center Region**: Typically has very high reconstruction accuracy
- **Recommendation**: Use signals longer than 2 × filter_length × levels for best results

## ENERGY PRESERVATION

For perfect reconstruction wavelets (like Daubechies), energy should be approximately preserved:

\[
E_{input} \approx E_{cA} + E_{cD}
\]

\[
E_{output} \approx E_{input}
\]

Where energy is calculated as:

\[
E = \sum_{n} |x[n]|^2
\]

Boundary effects may cause slight energy differences, but the center region should maintain excellent energy preservation.
