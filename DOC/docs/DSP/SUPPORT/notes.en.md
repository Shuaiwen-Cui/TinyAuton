# NOTES

!!! note "Note"
    The support module provides visualization and analysis utilities for signal processing. These functions help developers visualize signals, analyze data, and debug DSP algorithms by providing ASCII-based plots and formatted output. This is particularly useful in embedded systems where graphical displays may not be available.

## OVERVIEW

The support module includes four main functions:

1. **Signal Visualization**: Plot signals in ASCII format (like an oscilloscope)
2. **Spectrum Visualization**: Visualize power spectra with frequency axis labels
3. **Array Printing**: Print arrays in formatted tables
4. **Statistics**: Calculate and display statistical information about signals

## SIGNAL VISUALIZATION

### tiny_view_signal_f32

```c
/**
 * @name: tiny_view_signal_f32
 * @brief Visualize a signal in ASCII format (like oscilloscope)
 * @param data Input signal array
 * @param len Length of the signal
 * @param width Width of the plot in characters (default: 64)
 * @param height Height of the plot in lines (default: 16)
 * @param min Minimum Y-axis value (auto-detect if min == max)
 * @param max Maximum Y-axis value (auto-detect if min == max)
 * @param title Optional title for the plot (NULL for no title)
 * @return tiny_error_t
 */
tiny_error_t tiny_view_signal_f32(const float *data, int len, int width, int height, float min, float max, const char *title);
```

**Description**: 

Visualizes a signal in ASCII format, similar to an oscilloscope display. The function creates a character-based plot showing the signal waveform.

**Features**:

- High-resolution drawing with linear interpolation between data points

- Automatic Y-axis range detection when min == max

- Customizable plot dimensions (width and height)

- Optional title display

- Y-axis labels showing value range

- X-axis labels showing sample indices

**Parameters**:

- `data`: Pointer to the input signal array.

- `len`: Length of the signal array.

- `width`: Width of the plot in characters (typically 64).

- `height`: Height of the plot in lines (typically 16).

- `min`: Minimum Y-axis value. If `min == max`, the function will auto-detect the range.

- `max`: Maximum Y-axis value. If `min == max`, the function will auto-detect the range.
- 
- `title`: Optional title string for the plot. Pass `NULL` for no title.


**Return Value**: 

Returns success or error code.

**Output Format**:

The function prints:

- Title (if provided)

- Y-axis labels with value range

- ASCII plot with '*' characters representing the signal

- X-axis with sample index labels

- Summary line with value range and signal length

**Example Output**:

```
Test Signal: 10 Hz Sine Wave
Value
  1.20 |                                        
  1.00 |                                        
  0.80 |                                        
  0.60 |                                        
  0.40 |                                        
  0.20 |                                        
  0.00 |                                        
 -0.20 |                                        
 -0.40 |                                        
 -0.60 |                                        
 -0.80 |                                        
 -1.00 |                                        
 -1.20 |                                        
       ------------------------------------------------------------------------
       0        8       16      24      32      40      48      56 (Sample Index)
Range: [-1.200, 1.200], Length: 64
```

## SPECTRUM VISUALIZATION

### tiny_view_spectrum_f32

```c
/**
 * @name: tiny_view_spectrum_f32
 * @brief Visualize power spectrum in ASCII format (optimized for frequency domain)
 * @param power_spectrum Power spectrum array
 * @param len Length of the spectrum
 * @param sample_rate Sampling rate (Hz) for frequency axis labels
 * @param title Optional title for the plot (NULL for no title)
 * @return tiny_error_t
 */
tiny_error_t tiny_view_spectrum_f32(const float *power_spectrum, int len, float sample_rate, const char *title);
```

**Description**: 

Visualizes a power spectrum in ASCII format with frequency axis labels. The function creates a bar chart showing the power at different frequencies.

**Features**:

- Bar chart visualization (vertical bars)

- Automatic frequency axis labeling based on sample rate

- Optimized for frequency domain data (uses first half of spectrum)

- Frequency labels in Hz

- Nyquist frequency indication

**Parameters**:

- `power_spectrum`: Pointer to the power spectrum array.

- `len`: Length of the power spectrum array.

- `sample_rate`: Sampling rate of the original signal in Hz (used for frequency axis labels).

- `title`: Optional title string for the plot. Pass `NULL` for no title.

**Return Value**: 

Returns success or error code.

**Output Format**:

The function prints:

- Title (if provided)

- Y-axis labels with power values

- ASCII bar chart with '|' characters

- X-axis with frequency labels in Hz

- Summary line with value range and Nyquist frequency

**Note**: 

The function assumes the power spectrum length is half of the FFT length (typical for real signals). Frequency labels are calculated as: `freq = index * sample_rate / (2 * len)`.

## ARRAY PRINTING

### tiny_view_array_f32

```c
/**
 * @name: tiny_view_array_f32
 * @brief Print array values in a formatted table
 * @param data Input array
 * @param len Length of the array
 * @param name Name/label for the array
 * @param precision Number of decimal places (default: 3)
 * @param items_per_line Number of items per line (default: 8)
 * @return tiny_error_t
 */
tiny_error_t tiny_view_array_f32(const float *data, int len, const char *name, int precision, int items_per_line);
```

**Description**: 

Prints array values in a formatted table with customizable precision and items per line.

**Features**:

- Formatted table output with index labels

- Customizable precision (decimal places)

- Customizable items per line

- Optional array name/label

**Parameters**:

- `data`: Pointer to the input array.

- `len`: Length of the array.

- `name`: Optional name/label for the array. Pass `NULL` for default label.

- `precision`: Number of decimal places to display. If negative, defaults to 3.

- `items_per_line`: Number of items to print per line. If ≤ 0, defaults to 8.


**Return Value**: 

Returns success or error code.

**Output Format**:

The function prints:


- Array name and length

- Formatted table with index labels and values

**Example Output**:

```
Test Signal [64 elements]:
  [  0] 0.000  0.063  0.125  0.188  0.250  0.313  0.375  0.438 
  [  8] 0.500  0.563  0.625  0.688  0.750  0.813  0.875  0.938 
  ...
```

## STATISTICS

### tiny_view_statistics_f32

```c
/**
 * @name: tiny_view_statistics_f32
 * @brief Print statistical information about a signal
 * @param data Input signal array
 * @param len Length of the signal
 * @param name Name/label for the signal
 * @return tiny_error_t
 */
tiny_error_t tiny_view_statistics_f32(const float *data, int len, const char *name);
```

**Description**: 

Calculates and prints statistical information about a signal, including min, max, mean, standard deviation, variance, and peak values.

**Features**:

- Single-pass calculation (efficient)

- Comprehensive statistics:
  - Minimum and maximum values with indices
  - Peak value (absolute maximum) with index
  - Mean (average)
  - Standard deviation
  - Variance
  - Range (max - min)

- Optional signal name/label

**Parameters**:

- `data`: Pointer to the input signal array.

- `len`: Length of the signal array.

- `name`: Optional name/label for the signal. Pass `NULL` for default label.

**Return Value**: Returns success or error code.

**Output Format**:

The function prints:

- Statistics header with signal name

- All calculated statistics in a formatted table

**Example Output**:

```
=== Statistics: Test Signal ===
  Length:     64 samples
  Min:        -1.200000 (at index 48)
  Max:         1.200000 (at index 16)
  Peak:        1.200000 (at index 16)
  Mean:        0.000000
  Std Dev:     0.707107
  Variance:    0.500000
  Range:       2.400000
========================
```

**Mathematical Formulas**:

- **Mean**: \( \mu = \frac{1}{N} \sum_{i=0}^{N-1} x[i] \)

- **Variance**: \( \sigma^2 = \frac{1}{N} \sum_{i=0}^{N-1} x[i]^2 - \mu^2 \)

- **Standard Deviation**: \( \sigma = \sqrt{\sigma^2} \)

- **Range**: \( \text{range} = \max(x) - \min(x) \)

## USAGE WORKFLOW

### Typical Visualization Workflow

1. **Visualize Signal**:
   ```c
   float signal[64];
   // ... fill signal with data ...
   tiny_view_signal_f32(signal, 64, 64, 16, 0, 0, "My Signal");
   ```

2. **Print Array**:
   ```c
   tiny_view_array_f32(signal, 64, "Signal Data", 3, 8);
   ```

3. **Show Statistics**:
   ```c
   tiny_view_statistics_f32(signal, 64, "Signal");
   ```

4. **Visualize Spectrum**:
   ```c
   float power[128];
   // ... calculate power spectrum ...
   tiny_view_spectrum_f32(power, 128, 1000.0f, "Power Spectrum");
   ```

## APPLICATIONS

The support module is useful for:

- **Debugging**: Visualize signals during algorithm development

- **Analysis**: Quick statistical analysis of signals

- **Education**: Demonstrate signal processing concepts

- **Embedded Systems**: Debug DSP algorithms without graphical displays

- **Testing**: Verify signal processing results

- **Documentation**: Generate ASCII plots for documentation

## NOTES

- All visualization functions output to `stdout` using `printf`

- The ASCII plots are designed for monospace fonts

- For best results, use a terminal with at least 80 characters width

- Signal visualization uses linear interpolation for smooth plots

- Spectrum visualization uses bar charts optimized for frequency domain data
