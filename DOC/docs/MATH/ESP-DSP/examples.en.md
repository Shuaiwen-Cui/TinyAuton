# ESP-DSP EXAMPLES

## List of esp-dsp Examples

Signal processing APIs use dsps prefix. The following modules are available:

- Basic math - the example shows how to use basic vector math operations

- Dot-product - the example demonstrates how to use dot product functions

- FFT - the example demonstrates how to use FFT functionality

- FFT Window - the example demonstrates how to use Window and FFT functionality

- FFT 4 Real - the example demonstrates how to use FFT functionality for real input signals

- IIR - the example demonstrates how to use IIR filters functionality

- FIR - the example demonstrates how to use FIR filter functionality

- Kalman Filter - Extended Kalman Filter (EKF) example

- Matrix - example demonstrates how to use Mat class functionality

## Basic math

This example demonstrates how to use basic math functions from esp-dsp library. The example does the following steps:

- Initialize the library

- Initialize input signals with 1024 samples

- Apply window to input signal by standard C loop.

- Calculate FFT for 1024 complex samples and show the result

- Show results on the plots

- Apply window to input signal by basic math functions dsps_mul_f32 and dsps_mulc_f32.

- Calculate FFT for 1024 complex samples

- Show results on the plots

For more details please look to the examples/basic_math/README.md

### Dot-product

The example demonstrates how to use dotprod dsps_dotprod_f32 from esp-dsp library. Example does the following steps:

- Initialize the input arrays

- Calculate dot product of two arrays

- Compare results and calculate execution time in cycles.

For more details please look to the examples/dotprod/README.md

### FFT

This example demonstrates how to use FFT functionality from esp-dsp library. Example does the following steps:

- Initialize the library

- Initialize input signals with 1024 samples: one 0 dB, second with -20 dB

- Combine two signals as one complex input signal and apply window to input signals paar.

- Calculate FFT for 1024 complex samples

- Apply bit reverse operation for output complex vector

- Split one complex FFT output spectrum to two real signal spectrums

- Show results on the plots

- Show execution time of FFT

For more details please look to the examples/fft/README.md

### FFT Window

This example demonstrates how to use Window and FFT functionality from esp-dsp library. Example does the following steps:

- Initialize the library

- Initialize input signals with 1024 samples

- Apply window to input signal.

- Calculate FFT for 1024 complex samples

- Apply bit reverse operation for output complex vector

- Split one complex FFT output spectrum to two real signal spectrums

- Show results on the plots

For more details please look to the examples/fft_window/README.md

### FFT 4 Real

This example demonstrates how to use FFT functionality from esp-dsp library. Example does the following steps:

- Initialize the library

- Initialize input signals with 1024 samples: one 0 dB, second with -20 dB

- Calculate FFT Radix-2 for 1024 complex samples

- Calculate FFT Radix-4 for 1024 complex samples

- Apply bit reverse operation for output complex vectors

- Show results on the plots

- Show execution time of FFTs

For more details please look to the examples/fft4real/README.md

### IIR

This example demonstrates how to use IIR filters functionality from esp-dsp library. Example does the following steps:

- Initialize the library

- Initialize input signal

- Show LPF filter with Q factor 1
  
    - Calculate iir filter coefficients

    - Filter the input test signal (delta function)

    - Shows impulse response on the plot

    - Shows frequency response on the plot

- Calculate execution performance

- The same for LPF filter with Q factor 10

For more details please look to the examples/fir/README.md

### FIR

This example demonstrates how to use FIR filter functionality from esp-dsp library. Example does the following steps:

- Initialize the FFT library

- Initialize input signal

- Show input signal

- Show filtered signal

For more details please look to the examples/fir/README.md

### Kalman Filter

This example emulate system with IMU sensors and show how to use Extended Kalman Filter (EKF), with 13 values states vector, to estimate gyroscope errors and calculate system attitude. Also, this example show how to use esp-dsp library to operate with matrices and vectors.

In real system, the emulated sensors values should be replace by the real sensors values. Then, in real system, a calibration phase should be implemented and after the calibration phase the state vector X and covariance matrix P should be saved and restored next time, when filter called. It will save time for initial phase.

For more details please look to the examples/kalman/README.md


### Matrix

This example demonstrates how to use Mat class functionality from esp-dsp library. Example does the following steps:

- Initialize a matrix A and matirx x

- Calculate matrix b: b = A*x

- Find roots x1: A*x1 = b, with different methods

- Print result