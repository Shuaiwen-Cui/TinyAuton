# NOTES

!!! note "Note"
    Infinite Impulse Response (IIR) filters are recursive digital filters that use feedback, making them more efficient than FIR filters for the same specifications. However, IIR filters can be unstable if not designed carefully. They are widely used in audio processing, control systems, and signal conditioning where computational efficiency is important.

## IIR FILTER OVERVIEW

### Mathematical Principle

An IIR filter is defined by its difference equation, which includes both feedforward and feedback terms:

\[
y[n] = \sum_{k=0}^{M} b[k] \cdot x[n-k] - \sum_{k=1}^{N} a[k] \cdot y[n-k]
\]

Where:

- \( x[n] \) is the input signal

- \( y[n] \) is the output signal

- \( b[k] \) are feedforward (numerator) coefficients

- \( a[k] \) are feedback (denominator) coefficients

- \( M \) is the order of the numerator

- \( N \) is the order of the denominator

**Transfer Function**:

\[
H(z) = \frac{\sum_{k=0}^{M} b[k] \cdot z^{-k}}{1 + \sum_{k=1}^{N} a[k] \cdot z^{-k}}
\]

**Key Properties**:

- **Recursive**: Uses feedback (previous outputs)

- **Efficient**: Fewer coefficients than FIR for same specifications

- **Can be Unstable**: Poles must be inside unit circle

- **Non-Linear Phase**: Generally has non-linear phase response

## FILTER TYPES

The library supports four basic filter types:

- **Low-Pass**: Passes frequencies below cutoff, attenuates above
- **High-Pass**: Passes frequencies above cutoff, attenuates below
- **Band-Pass**: Passes frequencies within a band, attenuates outside
- **Band-Stop (Notch)**: Attenuates frequencies within a band, passes outside

## FILTER DESIGN

### Design Methods

The library supports Butterworth filter design (with plans for Chebyshev, Elliptic, and Bessel):

- **Butterworth**: Maximally flat passband, monotonic stopband
- **Chebyshev Type I**: Equiripple passband, monotonic stopband (future)
- **Chebyshev Type II**: Monotonic passband, equiripple stopband (future)
- **Elliptic**: Equiripple in both passband and stopband (future)
- **Bessel**: Linear phase response (future)

### Bilinear Transform

IIR filters are designed using the bilinear transform, which maps the analog s-plane to the digital z-plane:

\[
s = \frac{2}{T} \cdot \frac{1 - z^{-1}}{1 + z^{-1}}
\]

Where \( T \) is the sampling period.

### Design Parameters

- **Cutoff Frequency**: Normalized frequency (0.0 to 0.5, where 0.5 = Nyquist)
- **Filter Order**: Determines sharpness of transition and stopband attenuation
- **Design Method**: Affects passband/stopband characteristics

**Normalized Frequency**:

\[
f_{norm} = \frac{f_{cutoff}}{f_s / 2}
\]

Where \( f_s \) is the sampling rate.

## FILTER DESIGN FUNCTIONS

### tiny_iir_design_lowpass

```c
/**
 * @name tiny_iir_design_lowpass
 * @brief Design a low-pass IIR filter
 * @param cutoff_freq Normalized cutoff frequency (0.0 to 0.5)
 * @param order Filter order
 * @param design_method Design method (Butterworth, Chebyshev, etc.)
 * @param ripple_db Passband ripple in dB (for Chebyshev)
 * @param b_coeffs Output numerator coefficients (size: order + 1)
 * @param a_coeffs Output denominator coefficients (size: order + 1)
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_design_lowpass(float cutoff_freq, int order,
                                      tiny_iir_design_method_t design_method,
                                      float ripple_db,
                                      float *b_coeffs, float *a_coeffs);
```

**Description**: 

Designs a low-pass IIR filter using the specified design method. Currently supports Butterworth design for orders 1 and 2.

**Parameters**:

- `cutoff_freq`: Normalized cutoff frequency (0.0 to 0.5, where 0.5 = Nyquist frequency).

- `order`: Filter order. Currently supports 1 and 2 for Butterworth.

- `design_method`: Design method from `tiny_iir_design_method_t` enum. Currently only `TINY_IIR_DESIGN_BUTTERWORTH` is supported.

- `ripple_db`: Passband ripple in dB (for Chebyshev designs, currently unused).

- `b_coeffs`: Output array for numerator coefficients. Size must be at least `order + 1`.

- `a_coeffs`: Output array for denominator coefficients. Size must be at least `order + 1`. Note: `a[0]` is always 1.0 (normalized form).

**Return Value**: 

Returns success or error code.

**Note**: 

The coefficients are in normalized form where `a[0] = 1.0`. Higher order filters would need to be decomposed into cascaded biquads (second-order sections).

### tiny_iir_design_highpass

```c
/**
 * @name tiny_iir_design_highpass
 * @brief Design a high-pass IIR filter
 * @param cutoff_freq Normalized cutoff frequency (0.0 to 0.5)
 * @param order Filter order
 * @param design_method Design method
 * @param ripple_db Passband ripple in dB
 * @param b_coeffs Output numerator coefficients (size: order + 1)
 * @param a_coeffs Output denominator coefficients (size: order + 1)
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_design_highpass(float cutoff_freq, int order,
                                       tiny_iir_design_method_t design_method,
                                       float ripple_db,
                                       float *b_coeffs, float *a_coeffs);
```

**Description**: 

Designs a high-pass IIR filter using the specified design method. Currently supports Butterworth design for orders 1 and 2.

**Parameters**:

- `cutoff_freq`: Normalized cutoff frequency (0.0 to 0.5).

- `order`: Filter order. Currently supports 1 and 2.

- `design_method`: Design method. Currently only `TINY_IIR_DESIGN_BUTTERWORTH` is supported.

- `ripple_db`: Passband ripple in dB (currently unused).

- `b_coeffs`: Output array for numerator coefficients. Size must be at least `order + 1`.

- `a_coeffs`: Output array for denominator coefficients. Size must be at least `order + 1`.

**Return Value**: 

Returns success or error code.

### tiny_iir_design_bandpass

```c
/**
 * @name tiny_iir_design_bandpass
 * @brief Design a band-pass IIR filter
 * @param low_freq Lower cutoff frequency (normalized, 0.0 to 0.5)
 * @param high_freq Upper cutoff frequency (normalized, 0.0 to 0.5)
 * @param order Filter order
 * @param design_method Design method
 * @param ripple_db Passband ripple in dB
 * @param b_coeffs Output numerator coefficients
 * @param a_coeffs Output denominator coefficients
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_design_bandpass(float low_freq, float high_freq,
                                       int order,
                                       tiny_iir_design_method_t design_method,
                                       float ripple_db,
                                       float *b_coeffs, float *a_coeffs);
```

**Description**: 

Designs a band-pass IIR filter. Currently returns `TINY_ERR_NOT_SUPPORTED` as full band-pass design is not yet implemented.

**Parameters**:

- `low_freq`: Lower cutoff frequency (normalized, 0.0 to 0.5). Must be less than `high_freq`.

- `high_freq`: Upper cutoff frequency (normalized, 0.0 to 0.5). Must be greater than `low_freq`.

- `order`: Filter order.

- `design_method`: Design method.

- `ripple_db`: Passband ripple in dB.

- `b_coeffs`: Output array for numerator coefficients.

- `a_coeffs`: Output array for denominator coefficients.

**Return Value**: 

Currently returns `TINY_ERR_NOT_SUPPORTED`.

### tiny_iir_design_bandstop

```c
/**
 * @name tiny_iir_design_bandstop
 * @brief Design a band-stop (notch) IIR filter
 * @param low_freq Lower cutoff frequency (normalized, 0.0 to 0.5)
 * @param high_freq Upper cutoff frequency (normalized, 0.0 to 0.5)
 * @param order Filter order
 * @param design_method Design method
 * @param ripple_db Passband ripple in dB
 * @param b_coeffs Output numerator coefficients
 * @param a_coeffs Output denominator coefficients
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_design_bandstop(float low_freq, float high_freq,
                                       int order,
                                       tiny_iir_design_method_t design_method,
                                       float ripple_db,
                                       float *b_coeffs, float *a_coeffs);
```

**Description**: 

Designs a band-stop (notch) IIR filter. Currently returns `TINY_ERR_NOT_SUPPORTED` as full band-stop design is not yet implemented.

**Parameters**:

- `low_freq`: Lower cutoff frequency (normalized, 0.0 to 0.5). Must be less than `high_freq`.

- `high_freq`: Upper cutoff frequency (normalized, 0.0 to 0.5). Must be greater than `low_freq`.

- `order`: Filter order.

- `design_method`: Design method.

- `ripple_db`: Passband ripple in dB.

- `b_coeffs`: Output array for numerator coefficients.

- `a_coeffs`: Output array for denominator coefficients.

**Return Value**: 

Currently returns `TINY_ERR_NOT_SUPPORTED`.

## FILTER APPLICATION

### Batch Processing

### tiny_iir_filter_f32

```c
/**
 * @name tiny_iir_filter_f32
 * @brief Apply IIR filter to a signal (batch processing)
 * @param input Input signal array
 * @param input_len Length of input signal
 * @param b_coeffs Numerator coefficients
 * @param num_b Number of b coefficients
 * @param a_coeffs Denominator coefficients
 * @param num_a Number of a coefficients
 * @param output Output filtered signal array (size: input_len)
 * @param initial_state Initial state vector (can be NULL for zero initial conditions)
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_filter_f32(const float *input, int input_len,
                                  const float *b_coeffs, int num_b,
                                  const float *a_coeffs, int num_a,
                                  float *output,
                                  const float *initial_state);
```

**Description**: 

Applies an IIR filter to an entire signal using Direct Form II Transposed structure. This is suitable for batch processing when the entire signal is available.

**Features**:

- Uses Direct Form II Transposed implementation (efficient)

- Supports initial state conditions

- Output length equals input length

**Parameters**:

- `input`: Pointer to input signal array.

- `input_len`: Length of input signal.

- `b_coeffs`: Pointer to numerator coefficients.

- `num_b`: Number of numerator coefficients.

- `a_coeffs`: Pointer to denominator coefficients. Note: `a[0]` should be 1.0 (normalized form).

- `num_a`: Number of denominator coefficients.

- `output`: Pointer to output array for filtered signal. Size must be at least `input_len`.

- `initial_state`: Initial state vector. Can be `NULL` for zero initial conditions. Size should be `max(num_b, num_a) - 1`.

**Return Value**: 

Returns success or error code.

**Note**: 

The filter uses Direct Form II Transposed structure, which is computationally efficient and requires minimal state storage.

### Real-Time Processing

### tiny_iir_init

```c
/**
 * @name tiny_iir_init
 * @brief Initialize IIR filter structure for real-time filtering
 * @param filter Pointer to IIR filter structure
 * @param b_coeffs Numerator coefficients (will be copied)
 * @param num_b Number of b coefficients
 * @param a_coeffs Denominator coefficients (will be copied)
 * @param num_a Number of a coefficients
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_init(tiny_iir_filter_t *filter,
                            const float *b_coeffs, int num_b,
                            const float *a_coeffs, int num_a);
```

**Description**: 

Initializes an IIR filter structure for real-time sample-by-sample processing. Allocates memory for coefficients and state variables.

**Parameters**:

- `filter`: Pointer to `tiny_iir_filter_t` structure.

- `b_coeffs`: Pointer to numerator coefficients. Will be copied internally.

- `num_b`: Number of numerator coefficients.

- `a_coeffs`: Pointer to denominator coefficients. Will be copied internally.

- `num_a`: Number of denominator coefficients.

**Return Value**: 

Returns success or error code.

**Memory Management**: 

The function allocates memory internally. Use `tiny_iir_deinit()` to free it.

### tiny_iir_deinit

```c
/**
 * @name tiny_iir_deinit
 * @brief Deinitialize IIR filter and free allocated memory
 * @param filter Pointer to IIR filter structure
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_deinit(tiny_iir_filter_t *filter);
```

**Description**: 

Deinitializes an IIR filter and frees all allocated memory.

**Parameters**:

- `filter`: Pointer to `tiny_iir_filter_t` structure.

**Return Value**: 

Returns success or error code.

### tiny_iir_process_sample

```c
/**
 * @name tiny_iir_process_sample
 * @brief Process a single sample through IIR filter (real-time)
 * @param filter Pointer to initialized IIR filter structure
 * @param input Input sample value
 * @return Filtered output sample
 */
float tiny_iir_process_sample(tiny_iir_filter_t *filter, float input);
```

**Description**: 

Processes a single input sample through the IIR filter and returns the filtered output. Uses Direct Form II Transposed structure.

**Parameters**:

- `filter`: Pointer to initialized `tiny_iir_filter_t` structure.

- `input`: Input sample value.

**Return Value**: 

Returns filtered output sample.

**Note**: 

The filter maintains internal state between calls. Use `tiny_iir_reset()` to clear the state.

### tiny_iir_reset

```c
/**
 * @name tiny_iir_reset
 * @brief Reset IIR filter state (clear delay line)
 * @param filter Pointer to IIR filter structure
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_reset(tiny_iir_filter_t *filter);
```

**Description**: 

Resets the IIR filter state by clearing the state variables. Useful when starting a new signal or after a discontinuity.

**Parameters**:

- `filter`: Pointer to initialized `tiny_iir_filter_t` structure.

**Return Value**: 

Returns success or error code.

## BIQUAD FILTERS

Biquad (second-order) filters are a special case of IIR filters that are particularly efficient and commonly used. Higher-order filters are often decomposed into cascaded biquads for numerical stability.

### tiny_iir_biquad_init

```c
/**
 * @name tiny_iir_biquad_init
 * @brief Initialize a biquad (second-order) IIR filter
 * @param biquad Pointer to biquad filter structure
 * @param b0 Numerator coefficient b0
 * @param b1 Numerator coefficient b1
 * @param b2 Numerator coefficient b2
 * @param a1 Denominator coefficient a1 (a0 = 1.0)
 * @param a2 Denominator coefficient a2
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_biquad_init(tiny_iir_biquad_t *biquad,
                                    float b0, float b1, float b2,
                                    float a1, float a2);
```

**Description**: 

Initializes a biquad (second-order) IIR filter. Biquads are efficient and commonly used building blocks for higher-order filters.

**Parameters**:

- `biquad`: Pointer to `tiny_iir_biquad_t` structure.

- `b0`, `b1`, `b2`: Numerator coefficients.

- `a1`, `a2`: Denominator coefficients (a0 = 1.0 in normalized form).

**Return Value**: 

Returns success or error code.

### tiny_iir_biquad_process_sample

```c
/**
 * @name tiny_iir_biquad_process_sample
 * @brief Process a single sample through biquad filter (real-time)
 * @param biquad Pointer to initialized biquad filter structure
 * @param input Input sample value
 * @return Filtered output sample
 */
float tiny_iir_biquad_process_sample(tiny_iir_biquad_t *biquad, float input);
```

**Description**: 

Processes a single input sample through the biquad filter and returns the filtered output.

**Parameters**:

- `biquad`: Pointer to initialized `tiny_iir_biquad_t` structure.

- `input`: Input sample value.

**Return Value**: 

Returns filtered output sample.

### tiny_iir_biquad_reset

```c
/**
 * @name tiny_iir_biquad_reset
 * @brief Reset biquad filter state
 * @param biquad Pointer to biquad filter structure
 * @return tiny_error_t
 */
tiny_error_t tiny_iir_biquad_reset(tiny_iir_biquad_t *biquad);
```

**Description**: 

Resets the biquad filter state by clearing internal state variables.

**Parameters**:

- `biquad`: Pointer to `tiny_iir_biquad_t` structure.

**Return Value**: 

Returns success or error code.

## USAGE WORKFLOW

### Batch Filtering Workflow

1. **Design Filter**:
   ```c
   float b_coeffs[3], a_coeffs[3];
   tiny_iir_design_lowpass(0.1f, 2, TINY_IIR_DESIGN_BUTTERWORTH, 0.0f, b_coeffs, a_coeffs);
   ```

2. **Apply Filter**:
   ```c
   float input[256], output[256];
   tiny_iir_filter_f32(input, 256, b_coeffs, 3, a_coeffs, 3, output, NULL);
   ```

### Real-Time Filtering Workflow

1. **Design Filter**:
   ```c
   float b_coeffs[3], a_coeffs[3];
   tiny_iir_design_lowpass(0.1f, 2, TINY_IIR_DESIGN_BUTTERWORTH, 0.0f, b_coeffs, a_coeffs);
   ```

2. **Initialize Filter**:
   ```c
   tiny_iir_filter_t filter;
   tiny_iir_init(&filter, b_coeffs, 3, a_coeffs, 3);
   ```

3. **Process Samples**:
   ```c
   for (int i = 0; i < num_samples; i++) {
       float output = tiny_iir_process_sample(&filter, input[i]);
       // Use output...
   }
   ```

4. **Cleanup**:
   ```c
   tiny_iir_deinit(&filter);
   ```

### Biquad Workflow

1. **Design Filter** (or use pre-designed coefficients):
   ```c
   float b_coeffs[3], a_coeffs[3];
   tiny_iir_design_lowpass(0.1f, 2, TINY_IIR_DESIGN_BUTTERWORTH, 0.0f, b_coeffs, a_coeffs);
   ```

2. **Initialize Biquad**:
   ```c
   tiny_iir_biquad_t biquad;
   tiny_iir_biquad_init(&biquad, b_coeffs[0], b_coeffs[1], b_coeffs[2],
                        a_coeffs[1], a_coeffs[2]);
   ```

3. **Process Samples**:
   ```c
   for (int i = 0; i < num_samples; i++) {
       float output = tiny_iir_biquad_process_sample(&biquad, input[i]);
       // Use output...
   }
   ```

## APPLICATIONS

IIR filters are widely used in:

- **Audio Processing**: Equalization, tone control, audio effects
- **Control Systems**: Signal conditioning, noise filtering, feedback control
- **Biomedical**: ECG/EEG signal conditioning, artifact removal
- **Communications**: Channel equalization, noise reduction
- **Sensor Signal Processing**: Noise reduction, signal conditioning
- **Real-Time Systems**: Where computational efficiency is critical

## ADVANTAGES AND DISADVANTAGES

### Advantages

- **Efficient**: Fewer coefficients than FIR for same specifications
- **Sharp Transition**: Can achieve sharp frequency response with low order
- **Low Latency**: Minimal group delay compared to FIR
- **Memory Efficient**: Requires less memory than FIR

### Disadvantages

- **Potential Instability**: Can be unstable if poles are outside unit circle
- **Non-Linear Phase**: Generally has non-linear phase response
- **Design Complexity**: More complex design than FIR window method
- **Limit Cycles**: Can exhibit quantization-induced limit cycles

## STABILITY CONSIDERATIONS

For an IIR filter to be stable, all poles must lie inside the unit circle in the z-plane:

\[
|p_k| < 1 \quad \forall k
\]

Where \( p_k \) are the poles of the transfer function.

**Stability Check**:

The denominator polynomial \( A(z) = 1 + \sum_{k=1}^{N} a[k] \cdot z^{-k} \) must have all roots inside the unit circle.

## DESIGN CONSIDERATIONS

### Filter Order

- **Higher Order**: Sharper transition, better stopband attenuation, but more complex
- **Lower Order**: Simpler, faster, but wider transition band
- **Butterworth**: Order determines -3 dB point and stopband attenuation

### Normalized Frequency

Remember to normalize frequencies:

- Cutoff at 100 Hz with 1 kHz sample rate: `0.2` (100 / 500)
- Cutoff at 1 kHz with 10 kHz sample rate: `0.2` (1000 / 5000)

### Coefficient Normalization

IIR filters use normalized coefficients where `a[0] = 1.0`. This is standard practice and simplifies implementation.

## NOTES

- IIR filters can be unstable if not designed properly
- Always check stability when designing custom filters
- For higher orders, decompose into cascaded biquads for numerical stability
- Direct Form II Transposed is used for efficient implementation
- For real-time applications, use `tiny_iir_init()` and `tiny_iir_process_sample()`
- For batch processing, use `tiny_iir_filter_f32()`
- Biquad filters are recommended for higher-order designs

