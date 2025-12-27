# TinyDSP 头文件

!!! INFO
    这是TinyDSP库的主头文件。它包含所有必要的头文件，并提供了一个统一的接口来使用库的功能。在项目中完成该库的移植后，在需要使用相关函数的地方插入该头文件即可使用库内的所有函数。文档更新速度较慢，可能与实际代码不一致，请以实际代码为准。

```c

/**
 * @file tiny_dsp.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_dsp | Main header file - Unified entry point for all DSP functionality
 * @version 1.0
 * @date 2025-04-28
 * @copyright Copyright (c) 2025
 *
 * @details
 * This header file provides a unified interface to all DSP (Digital Signal Processing)
 * functionality in the tiny_dsp middleware. It includes:
 *
 * - Signal Processing: Convolution, Correlation, Resampling
 * - Filters: FIR (Finite Impulse Response), IIR (Infinite Impulse Response)
 * - Transforms: FFT (Fast Fourier Transform), DWT (Discrete Wavelet Transform), ICA (Independent Component Analysis)
 * - Support: Signal visualization and analysis tools
 *
 * Usage:
 *   Simply include this header to access all DSP functions:
 *   @code
 *   #include "tiny_dsp.h"
 *   @endcode
 */

#pragma once

/* ============================================================================
 * DEPENDENCIES
 * ============================================================================ */

// Core configuration
#include "tiny_dsp_config.h"

/* ============================================================================
 * SIGNAL PROCESSING MODULES
 * ============================================================================ */

/**
 * @name Signal Processing - Convolution
 * @brief Convolution operations with various padding and output modes
 * @details
 * - Full, center, head, and tail convolution modes
 * - Zero, symmetric, and periodic padding options
 * - Platform-optimized for ESP32
 */
#include "tiny_conv.h"
#include "tiny_conv_test.h"

/**
 * @name Signal Processing - Correlation
 * @brief Correlation and cross-correlation functions
 * @details
 * - Auto-correlation: Pattern matching, template matching
 * - Cross-correlation: Signal alignment, delay estimation
 * - Platform-optimized for ESP32
 */
#include "tiny_corr.h"
#include "tiny_corr_test.h"

/**
 * @name Signal Processing - Resampling
 * @brief Signal resampling, upsampling, and downsampling
 * @details
 * - Linear interpolation resampling
 * - Zero-insertion upsampling
 * - Skip-based downsampling
 */
#include "tiny_resample.h"
#include "tiny_resample_test.h"

/* ============================================================================
 * FILTER MODULES
 * ============================================================================ */

/**
 * @name Filter - FIR (Finite Impulse Response)
 * @brief FIR filter design and application
 * @details
 * - Always stable (no poles, only zeros)
 * - Linear phase response possible
 * - Window-based design methods (Hamming, Hanning, Blackman)
 * - Support for low-pass, high-pass, band-pass, band-stop
 * - Real-time and batch processing modes
 * - Platform-optimized for ESP32
 */
#include "tiny_fir.h"
#include "tiny_fir_test.h"

/**
 * @name Filter - IIR (Infinite Impulse Response)
 * @brief IIR filter design and application
 * @details
 * - Recursive filter with feedback
 * - More efficient than FIR for same specifications
 * - Butterworth, Chebyshev design methods
 * - Support for low-pass, high-pass, band-pass, band-stop
 * - Direct Form II transposed structure
 * - Biquad (second-order) cascade support
 * - Real-time and batch processing modes
 * - Platform-optimized for ESP32
 */
#include "tiny_iir.h"
#include "tiny_iir_test.h"

/* ============================================================================
 * TRANSFORM MODULES
 * ============================================================================ */

/**
 * @name Transform - Discrete Wavelet Transform (DWT)
 * @brief Multi-level wavelet decomposition and reconstruction
 * @details
 * - Support for Daubechies wavelets (DB1-DB10)
 * - Single-level and multi-level decomposition
 * - Perfect reconstruction capability
 * - Energy preservation analysis
 */
#include "tiny_dwt.h"
#include "tiny_dwt_test.h"

/**
 * @name Transform - Fast Fourier Transform (FFT)
 * @brief FFT/IFFT and frequency domain analysis
 * @details
 * - Forward and inverse FFT
 * - Power spectrum density calculation
 * - Peak frequency detection with parabolic interpolation
 * - Top N frequencies detection with peak merging
 * - Window functions: Hanning, Hamming, Blackman
 * - Platform-optimized for ESP32
 */
#include "tiny_fft.h"
#include "tiny_fft_test.h"

/**
 * @name Transform - Independent Component Analysis (ICA)
 * @brief Blind source separation using ICA
 * @details
 * Independent Component Analysis for blind source separation from mixed observations.
 * 
 * Algorithm:
 * - FastICA algorithm implementation (default)
 * - Blind source separation model: X = A * S
 *   where X is mixed signals, A is mixing matrix, S is independent sources
 * 
 * Preprocessing:
 * - Centering: Subtract mean from each observation
 * - Whitening: Decorrelate and normalize variance using eigenvalue decomposition
 * 
 * Features:
 * - Multiple nonlinearity functions:
 *   - tanh: Good for super-Gaussian sources (default)
 *   - cube: Good for sub-Gaussian sources
 *   - gauss: Alternative for symmetric sources
 *   - skew: Good for skewed sources
 * - Orthogonalization: Ensures extracted components are independent
 * - Iterative convergence with configurable tolerance
 * 
 * API Modes:
 * - C++ class-based API: ICA class with decompose() method
 * - Batch processing: Direct separation via decompose()
 * - Reconstruction: Reconstruct mixed signals from sources
 * 
 * Requirements:
 * - num_sources <= num_sensors (cannot extract more sources than sensors)
 * - Sufficient samples for stable statistics (recommended: > 100 samples)
 * - Sources must be statistically independent and non-Gaussian
 * 
 * Dependencies:
 * - Uses tiny_math matrix operations (Mat class)
 * - Implements eigenvalue decomposition for whitening
 * 
 * Applications:
 * - Audio source separation (cocktail party problem)
 * - Signal denoising and artifact removal
 * - Feature extraction from sensor arrays
 * - Biomedical signal processing (EEG, ECG artifact removal)
 */
#ifdef __cplusplus
#include "tiny_ica.hpp"
#include "tiny_ica_test.hpp"
#endif

/* ============================================================================
 * SUPPORT MODULES
 * ============================================================================ */

/**
 * @name Support - Signal Visualization
 * @brief ASCII-based signal and spectrum visualization tools
 * @details
 * - Signal plotting with configurable resolution
 * - Spectrum visualization
 * - Array formatting and statistics
 * - Console-based output (similar to ESP-DSP built-in features)
 */
#include "tiny_view.h"
#include "tiny_view_test.h"

/* ============================================================================
 * C++ COMPATIBILITY
 * ============================================================================ */

#ifdef __cplusplus
extern "C"
{
#endif

    // All DSP functions are C-compatible and can be called from C++

#ifdef __cplusplus
}
#endif
```