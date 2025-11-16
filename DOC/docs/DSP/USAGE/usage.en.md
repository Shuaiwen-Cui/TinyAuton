# USAGE INSTRUCTIONS

!!! info "Usage Instructions"
    This document provides usage instructions for the `tiny_dsp` module. 

## Import TinyDSP as a Whole

!!! info
    Suitable for C projects or projects with a simple structure in C++.

```c
#include "tiny_dsp.h"
```

## Import TinyDSP by Module
!!! info
    Suitable for projects that require precise control over module imports or complex C++ projects.

```c
// Signal processing modules (signal/)
#include "tiny_conv.h"        // convolution module
#include "tiny_corr.h"        // correlation module
#include "tiny_resample.h"    // resampling module

// Filter modules (filter/)
#include "tiny_fir.h"         // FIR filter module
#include "tiny_iir.h"         // IIR filter module

// Transform modules (transform/)
#include "tiny_fft.h"         // fast fourier transform module
#include "tiny_dwt.h"         // discrete wavelet transform module
#include "tiny_ica.h"         // independent component analysis module

// Support modules (support/)
#include "tiny_view.h"        // signal view/support module
```

!!! tip
    For specific usage methods, please refer to the test code.