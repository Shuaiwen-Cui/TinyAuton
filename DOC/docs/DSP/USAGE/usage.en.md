# USAGE INSTRUCTIONS {#usage-instructions}

!!! info "Implementation and records"
    APIs in this section are based on `CODE/AIoTNode-TinyAuton-DSP/middleware/`. Source excerpts and serial output include historical records; check the project entry and enabled selectors before reproducing a test.

!!! info "Usage Instructions"
    This document provides usage instructions for the `tiny_dsp` module. 

## Import TinyDSP as a Whole {#import-tinydsp-as-a-whole}

!!! info
    Suitable for C projects or projects with a simple structure in C++.

```c
#include "tiny_dsp.h"
```

## Import TinyDSP by Module {#import-tinydsp-by-module}
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
#include "tiny_ica.hpp"         // independent component analysis module

// Support modules (support/)
#include "tiny_view.h"        // signal view/support module
```

!!! tip
    For specific usage methods, please refer to the test code.
