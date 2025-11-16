# NOTES

!!! note "Note"
    Fast Fourier Transform (FFT) is a fundamental algorithm in signal processing that efficiently computes the Discrete Fourier Transform (DFT). It converts signals from the time domain to the frequency domain, enabling frequency analysis, spectral analysis, and filtering operations. FFT is widely used in audio processing, communications, structural health monitoring, and many other applications.

## FFT OVERVIEW

### Mathematical Principle

The Discrete Fourier Transform (DFT) of a sequence \( x[n] \) of length \( N \) is defined as:

\[
X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j \frac{2\pi kn}{N}}
\]

Where:

- \( x[n] \) is the input signal in the time domain

- \( X[k] \) is the output in the frequency domain

- \( N \) is the length of the signal (must be a power of 2)

- \( k \in [0, N-1] \) is the frequency bin index

**Frequency Resolution**:

\[
\Delta f = \frac{f_s}{N}
\]

Where:

- \( \Delta f \) is the frequency resolution (Hz)

- \( f_s \) is the sampling rate (Hz)

- \( N \) is the FFT size

**Frequency of bin k**:

\[
f_k = k \cdot \frac{f_s}{N}
\]

## WINDOW FUNCTIONS

Window functions are applied to signals before FFT to reduce spectral leakage. The library supports several window types:

- **None (Rectangular)**: No window applied, fastest but may have spectral leakage

- **Hanning**: Good general-purpose window, balances frequency resolution and leakage reduction

- **Hamming**: Similar to Hanning, slightly better sidelobe suppression

- **Blackman**: Best sidelobe suppression, but wider main lobe

## INITIALIZATION AND DEINITIALIZATION

### tiny_fft_init

```c
/**
 * @name: tiny_fft_init
 * @brief Initialize FFT tables (required before using FFT functions)
 * @note This function should be called once at startup
 * @param fft_size Maximum FFT size to support (must be power of 2)
 * @return tiny_error_t
 */
tiny_error_t tiny_fft_init(int fft_size);
```

**Description**: 

Initializes FFT tables and prepares the library for FFT operations. This function must be called before any FFT operations.

**Features**:

- Platform-specific optimization enabled (ESP32 uses optimized DSP library).

- Must be called once before using any FFT functions.

**Parameters**:

- `fft_size`: Maximum FFT size to support. Must be a power of 2 (e.g., 256, 512, 1024).

**Return Value**: 

Returns success or error code.

**Important Notes**:

- FFT size must be a power of 2.

- This function should be called once at system startup.

- All subsequent FFT operations must use sizes ≤ `fft_size`.

### tiny_fft_deinit

```c
/**
 * @name: tiny_fft_deinit
 * @brief Deinitialize FFT tables and free resources
 * @return tiny_error_t
 */
tiny_error_t tiny_fft_deinit(void);
```

**Description**: 

Deinitializes FFT tables and frees allocated resources.

**Return Value**: 

Returns success or error code.

## FORWARD FFT

### tiny_fft_f32

```c
/**
 * @name: tiny_fft_f32
 * @brief Perform FFT on real-valued input signal
 * @param input Input signal array (real values)
 * @param input_len Length of input signal (must be power of 2)
 * @param output_fft Output FFT result (complex array: [Re0, Im0, Re1, Im1, ...])
 *                   Size must be at least input_len * 2
 * @param window Window function to apply before FFT (optional)
 * @return tiny_error_t
 */
tiny_error_t tiny_fft_f32(const float *input, int input_len, float *output_fft, tiny_fft_window_t window);
```

**Description**: 

Performs Fast Fourier Transform on a real-valued input signal, converting it from time domain to frequency domain.

**Features**:

- Platform-specific optimization enabled.

- Supports optional window functions to reduce spectral leakage.

- Output is in complex format: `[Re0, Im0, Re1, Im1, ...]`.

**Parameters**:

- `input`: Pointer to the input signal array (real values).

- `input_len`: Length of the input signal. Must be a power of 2 and ≤ initialized FFT size.

- `output_fft`: Pointer to the output array for FFT result. Size must be at least `input_len * 2` (complex format).

- `window`: Window function type to apply before FFT. Options:
  - `TINY_FFT_WINDOW_NONE`: No window (rectangular)
  - `TINY_FFT_WINDOW_HANNING`: Hanning window
  - `TINY_FFT_WINDOW_HAMMING`: Hamming window
  - `TINY_FFT_WINDOW_BLACKMAN`: Blackman window

**Return Value**: 

Returns success or error code.

**Output Format**:

The output is stored as an interleaved complex array:

- `output_fft[0]` = Real part of bin 0

- `output_fft[1]` = Imaginary part of bin 0

- `output_fft[2]` = Real part of bin 1

- `output_fft[3]` = Imaginary part of bin 1

- ...

**Frequency Bins**:

- Bin 0: DC component (0 Hz)

- Bin k: Frequency = \( k \cdot \frac{f_s}{N} \) Hz

- Bin N/2: Nyquist frequency (\( f_s/2 \) Hz)

- Bins N/2+1 to N-1: Mirror of bins 1 to N/2-1 (for real signals)

## INVERSE FFT

### tiny_fft_ifft_f32

```c
/**
 * @name: tiny_fft_ifft_f32
 * @brief Perform inverse FFT to reconstruct time-domain signal
 * @param input_fft Input FFT array (complex: [Re0, Im0, Re1, Im1, ...])
 * @param fft_len Length of FFT (number of complex points)
 * @param output Output reconstructed signal (real values)
 *               Size must be at least fft_len
 * @return tiny_error_t
 */
tiny_error_t tiny_fft_ifft_f32(const float *input_fft, int fft_len, float *output);
```

**Description**: 

Performs Inverse Fast Fourier Transform, converting a frequency-domain signal back to the time domain.

**Features**:

- Platform-specific optimization enabled.

- Reconstructs the original time-domain signal from FFT result.

**Parameters**:

- `input_fft`: Pointer to the input FFT array (complex format: `[Re0, Im0, Re1, Im1, ...]`).

- `fft_len`: Length of the FFT (number of complex points). Must be a power of 2.

- `output`: Pointer to the output array for reconstructed signal (real values). Size must be at least `fft_len`.

**Return Value**: 

Returns success or error code.

**Note**: 

The reconstructed signal should match the original input signal (within numerical precision), assuming no modifications were made to the FFT result.

## SPECTRUM ANALYSIS

### tiny_fft_magnitude_f32

```c
/**
 * @name: tiny_fft_magnitude_f32
 * @brief Calculate magnitude spectrum from FFT result
 * @param fft_result FFT result (complex array: [Re0, Im0, Re1, Im1, ...])
 * @param fft_len Length of FFT (number of complex points)
 * @param magnitude Output magnitude spectrum (real values)
 *                  Size must be at least fft_len
 * @return tiny_error_t
 */
tiny_error_t tiny_fft_magnitude_f32(const float *fft_result, int fft_len, float *magnitude);
```

**Description**: 

Calculates the magnitude spectrum from FFT result. The magnitude represents the amplitude of each frequency component.

**Mathematical Formula**:

\[
|X[k]| = \sqrt{\text{Re}[X[k]]^2 + \text{Im}[X[k]]^2}
\]

**Parameters**:

- `fft_result`: Pointer to the FFT result array (complex format).

- `fft_len`: Length of the FFT (number of complex points).

- `magnitude`: Pointer to the output array for magnitude spectrum. Size must be at least `fft_len`.

**Return Value**: 

Returns success or error code.

### tiny_fft_power_spectrum_f32

```c
/**
 * @name: tiny_fft_power_spectrum_f32
 * @brief Calculate power spectrum density (PSD) from FFT result
 * @param fft_result FFT result (complex array: [Re0, Im0, Re1, Im1, ...])
 * @param fft_len Length of FFT (number of complex points)
 * @param power Output power spectrum (real values)
 *              Size must be at least fft_len
 * @return tiny_error_t
 */
tiny_error_t tiny_fft_power_spectrum_f32(const float *fft_result, int fft_len, float *power);
```

**Description**: 

Calculates the power spectrum density (PSD) from FFT result. Power spectrum represents the power of each frequency component and is normalized by FFT length.

**Mathematical Formula**:

\[
P[k] = \frac{|X[k]|^2}{N} = \frac{\text{Re}[X[k]]^2 + \text{Im}[X[k]]^2}{N}
\]

**Parameters**:

- `fft_result`: Pointer to the FFT result array (complex format).

- `fft_len`: Length of the FFT (number of complex points).

- `power`: Pointer to the output array for power spectrum. Size must be at least `fft_len`.

**Return Value**: 

Returns success or error code.

## FREQUENCY DETECTION

### tiny_fft_find_peak_frequency

```c
/**
 * @name: tiny_fft_find_peak_frequency
 * @brief Find the frequency with maximum power (useful for structural health monitoring)
 * @param power_spectrum Power spectrum array
 * @param fft_len Length of power spectrum
 * @param sample_rate Sampling rate of the original signal (Hz)
 * @param peak_freq Output peak frequency (Hz)
 * @param peak_power Output peak power value
 * @return tiny_error_t
 */
tiny_error_t tiny_fft_find_peak_frequency(const float *power_spectrum, int fft_len, float sample_rate, float *peak_freq, float *peak_power);
```

**Description**: 

Finds the frequency with the maximum power in the power spectrum. Uses parabolic interpolation for sub-bin accuracy.

**Features**:

- Skips DC component (bin 0).

- Uses parabolic interpolation to improve frequency estimation accuracy.

- Useful for detecting dominant frequencies in signals.

**Parameters**:

- `power_spectrum`: Pointer to the power spectrum array.

- `fft_len`: Length of the power spectrum.

- `sample_rate`: Sampling rate of the original signal in Hz.

- `peak_freq`: Pointer to output variable for peak frequency (Hz).

- `peak_power`: Pointer to output variable for peak power value.

**Return Value**: 

Returns success or error code.

### tiny_fft_find_top_frequencies

```c
/**
 * @name: tiny_fft_find_top_frequencies
 * @brief Find top N frequencies with highest power
 * @param power_spectrum Power spectrum array
 * @param fft_len Length of power spectrum
 * @param sample_rate Sampling rate of the original signal (Hz)
 * @param top_n Number of top frequencies to find
 * @param frequencies Output array for frequencies (Hz), size must be at least top_n
 * @param powers Output array for power values, size must be at least top_n
 * @return tiny_error_t
 */
tiny_error_t tiny_fft_find_top_frequencies(const float *power_spectrum, int fft_len, float sample_rate, int top_n, float *frequencies, float *powers);
```

**Description**: 

Finds the top N frequencies with the highest power in the power spectrum. Automatically detects local peaks and merges nearby peaks to avoid selecting multiple bins from the same frequency peak.

**Features**:

- Detects local peaks in the power spectrum.

- Merges nearby peaks (within 2 bins) to avoid duplicates.

- Uses parabolic interpolation for improved frequency accuracy.

- Filters out insignificant peaks (below 1% of maximum power).

**Parameters**:

- `power_spectrum`: Pointer to the power spectrum array.

- `fft_len`: Length of the power spectrum.

- `sample_rate`: Sampling rate of the original signal in Hz.

- `top_n`: Number of top frequencies to find.

- `frequencies`: Pointer to output array for frequencies (Hz). Size must be at least `top_n`.

- `powers`: Pointer to output array for power values. Size must be at least `top_n`.

**Return Value**: 

Returns success or error code.

**Note**: 

If fewer than `top_n` peaks are found, remaining entries in the output arrays are set to zero.

## USAGE WORKFLOW

### Typical FFT Analysis Workflow

1. **Initialize FFT**:
   ```c
   tiny_fft_init(256);  // Initialize for max 256-point FFT
   ```

2. **Perform FFT**:
   ```c
   float input[256];
   float fft_result[512];  // Complex output: 256 * 2
   tiny_fft_f32(input, 256, fft_result, TINY_FFT_WINDOW_HANNING);
   ```

3. **Calculate Power Spectrum**:
   ```c
   float power[256];
   tiny_fft_power_spectrum_f32(fft_result, 256, power);
   ```

4. **Find Peak Frequency**:
   ```c
   float peak_freq, peak_power;
   tiny_fft_find_peak_frequency(power, 256, 1000.0f, &peak_freq, &peak_power);
   ```

5. **Deinitialize** (when done):
   ```c
   tiny_fft_deinit();
   ```

## APPLICATIONS

FFT is widely used in various applications:

- **Audio Processing**: Frequency analysis, equalization, pitch detection

- **Communications**: Signal modulation, demodulation, channel analysis

- **Structural Health Monitoring**: Vibration analysis, resonance detection

- **Biomedical**: ECG/EEG analysis, heart rate detection

- **Image Processing**: 2D FFT for image filtering and analysis

- **Spectral Analysis**: Identifying frequency components in signals
