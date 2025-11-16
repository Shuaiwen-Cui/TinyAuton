# NOTES

!!! note "Note"
    Finite Impulse Response (FIR) filters are digital filters with no feedback, making them always stable. FIR filters can achieve linear phase response, which is important for applications requiring phase preservation. They are implemented via convolution and are widely used in audio processing, communications, and signal conditioning.

## FIR FILTER OVERVIEW

### Mathematical Principle

An FIR filter is defined by its impulse response \( h[n] \), which has finite length. The output \( y[n] \) is computed as:

\[
y[n] = \sum_{k=0}^{M-1} h[k] \cdot x[n-k]
\]

Where:

- \( x[n] \) is the input signal

- \( h[k] \) are the filter coefficients (taps)

- \( M \) is the number of filter taps

- \( y[n] \) is the output signal

**Transfer Function**:

\[
H(z) = \sum_{k=0}^{M-1} h[k] \cdot z^{-k}
\]

**Key Properties**:

- **Always Stable**: No poles, only zeros

- **Linear Phase**: Possible with symmetric coefficients

- **Finite Memory**: Only requires \( M \) past samples

- **No Feedback**: Output depends only on input

## FILTER TYPES

The library supports four basic filter types:

- **Low-Pass**: Passes frequencies below cutoff, attenuates above
- **High-Pass**: Passes frequencies above cutoff, attenuates below
- **Band-Pass**: Passes frequencies within a band, attenuates outside
- **Band-Stop (Notch)**: Attenuates frequencies within a band, passes outside

## FILTER DESIGN

### Window Method

The library uses the window method for FIR filter design:

1. **Generate Ideal Filter**: Create ideal frequency response
2. **Apply Window**: Multiply by window function to reduce Gibbs phenomenon
3. **Truncate**: Limit to finite number of taps

**Supported Windows**:

- **Rectangular**: No window (fastest, but may have ringing)

- **Hamming**: Good balance of main lobe width and side lobe suppression

- **Hanning**: Similar to Hamming, slightly better side lobe suppression

- **Blackman**: Best side lobe suppression, wider main lobe

**Window Selection Guidelines**:

- **Hamming**: General purpose, good balance

- **Hanning**: Better side lobe suppression than Hamming

- **Blackman**: Best for applications requiring low side lobes

- **Rectangular**: Only for very simple applications

### Design Parameters

- **Cutoff Frequency**: Normalized frequency (0.0 to 0.5, where 0.5 = Nyquist)
- **Number of Taps**: Should be odd for linear phase (Type I filter)
- **Window Type**: Affects transition bandwidth and side lobe levels

**Normalized Frequency**:

\[
f_{norm} = \frac{f_{cutoff}}{f_s / 2}
\]

Where \( f_s \) is the sampling rate.

## FILTER DESIGN FUNCTIONS

### tiny_fir_design_lowpass

```c
/**
 * @name tiny_fir_design_lowpass
 * @brief Design a low-pass FIR filter using window method
 * @param cutoff_freq Normalized cutoff frequency (0.0 to 0.5)
 * @param num_taps Number of filter taps (should be odd)
 * @param window Window function to use
 * @param coefficients Output array for filter coefficients
 * @return tiny_error_t
 */
tiny_error_t tiny_fir_design_lowpass(float cutoff_freq, int num_taps,
                                     tiny_fir_window_t window,
                                     float *coefficients);
```

**Description**: 

Designs a low-pass FIR filter using the window method. The ideal low-pass filter impulse response is generated and then windowed to reduce Gibbs phenomenon.

**Parameters**:

- `cutoff_freq`: Normalized cutoff frequency (0.0 to 0.5, where 0.5 = Nyquist frequency).

- `num_taps`: Number of filter taps. Must be odd for linear phase response.

- `window`: Window function type from `tiny_fir_window_t` enum.

- `coefficients`: Output array for filter coefficients. Size must be at least `num_taps`.

**Return Value**: 

Returns success or error code.

**Note**: 

The cutoff frequency is normalized: `cutoff_freq = actual_freq / (sample_rate / 2)`. For example, a 100 Hz cutoff at 1 kHz sample rate would be `0.2` (100 / 500).

### tiny_fir_design_highpass

```c
/**
 * @name tiny_fir_design_highpass
 * @brief Design a high-pass FIR filter using window method
 * @param cutoff_freq Normalized cutoff frequency (0.0 to 0.5)
 * @param num_taps Number of filter taps (should be odd)
 * @param window Window function to use
 * @param coefficients Output array for filter coefficients
 * @return tiny_error_t
 */
tiny_error_t tiny_fir_design_highpass(float cutoff_freq, int num_taps,
                                      tiny_fir_window_t window,
                                      float *coefficients);
```

**Description**: 

Designs a high-pass FIR filter using the window method. The ideal high-pass filter impulse response is generated and then windowed.

**Parameters**:

- `cutoff_freq`: Normalized cutoff frequency (0.0 to 0.5).

- `num_taps`: Number of filter taps. Must be odd.

- `window`: Window function type.

- `coefficients`: Output array for filter coefficients. Size must be at least `num_taps`.

**Return Value**: 

Returns success or error code.

### tiny_fir_design_bandpass

```c
/**
 * @name tiny_fir_design_bandpass
 * @brief Design a band-pass FIR filter using window method
 * @param low_freq Lower cutoff frequency (normalized, 0.0 to 0.5)
 * @param high_freq Upper cutoff frequency (normalized, 0.0 to 0.5)
 * @param num_taps Number of filter taps (should be odd)
 * @param window Window function to use
 * @param coefficients Output array for filter coefficients
 * @return tiny_error_t
 */
tiny_error_t tiny_fir_design_bandpass(float low_freq, float high_freq,
                                      int num_taps,
                                      tiny_fir_window_t window,
                                      float *coefficients);
```

**Description**: 

Designs a band-pass FIR filter using the window method. The filter passes frequencies between `low_freq` and `high_freq`.

**Parameters**:

- `low_freq`: Lower cutoff frequency (normalized, 0.0 to 0.5). Must be less than `high_freq`.

- `high_freq`: Upper cutoff frequency (normalized, 0.0 to 0.5). Must be greater than `low_freq`.

- `num_taps`: Number of filter taps. Must be odd.

- `window`: Window function type.

- `coefficients`: Output array for filter coefficients. Size must be at least `num_taps`.

**Return Value**: 

Returns success or error code.

### tiny_fir_design_bandstop

```c
/**
 * @name tiny_fir_design_bandstop
 * @brief Design a band-stop (notch) FIR filter using window method
 * @param low_freq Lower cutoff frequency (normalized, 0.0 to 0.5)
 * @param high_freq Upper cutoff frequency (normalized, 0.0 to 0.5)
 * @param num_taps Number of filter taps (should be odd)
 * @param window Window function to use
 * @param coefficients Output array for filter coefficients
 * @return tiny_error_t
 */
tiny_error_t tiny_fir_design_bandstop(float low_freq, float high_freq,
                                      int num_taps,
                                      tiny_fir_window_t window,
                                      float *coefficients);
```

**Description**: 

Designs a band-stop (notch) FIR filter using the window method. The filter attenuates frequencies between `low_freq` and `high_freq`.

**Parameters**:

- `low_freq`: Lower cutoff frequency (normalized, 0.0 to 0.5). Must be less than `high_freq`.

- `high_freq`: Upper cutoff frequency (normalized, 0.0 to 0.5). Must be greater than `low_freq`.

- `num_taps`: Number of filter taps. Must be odd.

- `window`: Window function type.

- `coefficients`: Output array for filter coefficients. Size must be at least `num_taps`.

**Return Value**: 

Returns success or error code.

## FILTER APPLICATION

### Batch Processing

### tiny_fir_filter_f32

```c
/**
 * @name tiny_fir_filter_f32
 * @brief Apply FIR filter to a signal (batch processing)
 * @param input Input signal array
 * @param input_len Length of input signal
 * @param coefficients FIR filter coefficients (taps)
 * @param num_taps Number of filter taps
 * @param output Output filtered signal array
 * @param padding_mode Padding mode for boundary handling
 * @return tiny_error_t
 */
tiny_error_t tiny_fir_filter_f32(const float *input, int input_len,
                                  const float *coefficients, int num_taps,
                                  float *output,
                                  tiny_padding_mode_t padding_mode);
```

**Description**: 

Applies an FIR filter to an entire signal using convolution. This is suitable for batch processing when the entire signal is available.

**Features**:

- Uses convolution internally

- Supports different padding modes for boundary handling

- Output length equals input length

**Parameters**:

- `input`: Pointer to input signal array.

- `input_len`: Length of input signal.

- `coefficients`: Pointer to FIR filter coefficients (taps).

- `num_taps`: Number of filter taps.

- `output`: Pointer to output array for filtered signal. Size must be at least `input_len`.

- `padding_mode`: Padding mode for boundary handling (e.g., `TINY_PADDING_SYMMETRIC`).

**Return Value**: 

Returns success or error code.

### Real-Time Processing

### tiny_fir_init

```c
/**
 * @name tiny_fir_init
 * @brief Initialize FIR filter structure for real-time filtering
 * @param filter Pointer to FIR filter structure
 * @param coefficients Filter coefficients (will be copied internally)
 * @param num_taps Number of filter taps
 * @return tiny_error_t
 */
tiny_error_t tiny_fir_init(tiny_fir_filter_t *filter,
                            const float *coefficients, int num_taps);
```

**Description**: 

Initializes an FIR filter structure for real-time sample-by-sample processing. Allocates memory for coefficients and delay line.

**Parameters**:

- `filter`: Pointer to `tiny_fir_filter_t` structure.

- `coefficients`: Pointer to filter coefficients. Will be copied internally.

- `num_taps`: Number of filter taps.

**Return Value**: 

Returns success or error code.

**Memory Management**: 

The function allocates memory internally. Use `tiny_fir_deinit()` to free it.

### tiny_fir_deinit

```c
/**
 * @name tiny_fir_deinit
 * @brief Deinitialize FIR filter and free allocated memory
 * @param filter Pointer to FIR filter structure
 * @return tiny_error_t
 */
tiny_error_t tiny_fir_deinit(tiny_fir_filter_t *filter);
```

**Description**: 

Deinitializes an FIR filter and frees all allocated memory.

**Parameters**:

- `filter`: Pointer to `tiny_fir_filter_t` structure.

**Return Value**: 

Returns success or error code.

### tiny_fir_process_sample

```c
/**
 * @name tiny_fir_process_sample
 * @brief Process a single sample through FIR filter (real-time)
 * @param filter Pointer to initialized FIR filter structure
 * @param input Input sample value
 * @return Filtered output sample
 */
float tiny_fir_process_sample(tiny_fir_filter_t *filter, float input);
```

**Description**: 

Processes a single input sample through the FIR filter and returns the filtered output. Uses a circular buffer for efficient delay line implementation.

**Parameters**:

- `filter`: Pointer to initialized `tiny_fir_filter_t` structure.

- `input`: Input sample value.

**Return Value**: 

Returns filtered output sample.

**Note**: 

The filter maintains internal state (delay line) between calls. Use `tiny_fir_reset()` to clear the state.

### tiny_fir_reset

```c
/**
 * @name tiny_fir_reset
 * @brief Reset FIR filter state (clear delay line)
 * @param filter Pointer to FIR filter structure
 * @return tiny_error_t
 */
tiny_error_t tiny_fir_reset(tiny_fir_filter_t *filter);
```

**Description**: 

Resets the FIR filter state by clearing the delay line. Useful when starting a new signal or after a discontinuity.

**Parameters**:

- `filter`: Pointer to initialized `tiny_fir_filter_t` structure.

**Return Value**: 

Returns success or error code.

## USAGE WORKFLOW

### Batch Filtering Workflow

1. **Design Filter**:
   ```c
   float coeffs[51];
   tiny_fir_design_lowpass(0.1f, 51, TINY_FIR_WINDOW_HAMMING, coeffs);
   ```

2. **Apply Filter**:
   ```c
   float input[256], output[256];
   tiny_fir_filter_f32(input, 256, coeffs, 51, output, TINY_PADDING_SYMMETRIC);
   ```

### Real-Time Filtering Workflow

1. **Design Filter**:
   ```c
   float coeffs[21];
   tiny_fir_design_lowpass(0.1f, 21, TINY_FIR_WINDOW_HAMMING, coeffs);
   ```

2. **Initialize Filter**:
   ```c
   tiny_fir_filter_t filter;
   tiny_fir_init(&filter, coeffs, 21);
   ```

3. **Process Samples**:
   ```c
   for (int i = 0; i < num_samples; i++) {
       float output = tiny_fir_process_sample(&filter, input[i]);
       // Use output...
   }
   ```

4. **Cleanup**:
   ```c
   tiny_fir_deinit(&filter);
   ```

## APPLICATIONS

FIR filters are widely used in:

- **Audio Processing**: Equalization, noise reduction, anti-aliasing
- **Communications**: Pulse shaping, matched filtering, channel equalization
- **Biomedical**: ECG/EEG signal conditioning, artifact removal
- **Control Systems**: Signal conditioning, noise filtering
- **Image Processing**: Edge detection, smoothing, sharpening
- **Sensor Signal Processing**: Noise reduction, signal conditioning

## ADVANTAGES AND DISADVANTAGES

### Advantages

- **Always Stable**: No feedback, guaranteed stability
- **Linear Phase**: Can achieve exact linear phase response
- **Simple Design**: Window method is straightforward
- **No Limit Cycles**: No quantization-induced oscillations

### Disadvantages

- **Higher Computational Cost**: Requires more taps than IIR for same specifications
- **Longer Delay**: Group delay proportional to filter length
- **Memory Requirements**: Needs storage for all filter taps

## DESIGN CONSIDERATIONS

### Number of Taps

- **More Taps**: Sharper transition, better stopband attenuation, but higher computation
- **Fewer Taps**: Faster computation, but wider transition band
- **Rule of Thumb**: Transition bandwidth ≈ 4 / num_taps (for Hamming window)

### Window Selection

- **Hamming**: Good general-purpose choice
- **Hanning**: Better side lobe suppression
- **Blackman**: Best side lobe suppression, wider transition
- **Rectangular**: Only for very simple cases (not recommended)

### Normalized Frequency

Remember to normalize frequencies:

- Cutoff at 100 Hz with 1 kHz sample rate: `0.2` (100 / 500)
- Cutoff at 1 kHz with 10 kHz sample rate: `0.2` (1000 / 5000)

## NOTES

- FIR filters are always stable (no poles)
- Linear phase requires odd number of taps and symmetric coefficients
- Window method is simple but may not be optimal for all applications
- For real-time applications, use `tiny_fir_init()` and `tiny_fir_process_sample()`
- For batch processing, use `tiny_fir_filter_f32()`

