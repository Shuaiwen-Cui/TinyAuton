# DIGITAL SIGNAL PROCESSING

!!! note
    This component provides a set of functions designed for signal processing on edge devices, with a focus on lightweight and efficient implementations of commonly used signal processing algorithms.

!!! note
    This component is a wrapper and extension of the official ESP32 digital signal processing library [ESP-DSP](https://docs.espressif.com/projects/esp-dsp/en/latest/esp32/index.html), providing higher-level API interfaces. In simple terms, the TinyMath library corresponds to the Math, Matrix, and DotProduct modules in ESP-DSP, while the other modules in ESP-DSP correspond to the TinyDSP library. Additionally, TinyDSP provides some functionalities not available in ESP-DSP, focusing on scenarios such as structural health monitoring.

## COMPONENT DEPENDENCIES

```c
set(src_dirs
    .
    signal
    filter
    transform
    support
)

set(include_dirs
    .
    include
    signal
    filter
    transform
    support
)

set(requires
    tiny_math
)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})


```

## ARCHITECTURE AND DIRECTORY

### Dependency Diagram

![](tiny_dsp.png)

### Code Tree

```txt
tiny_dsp/
├── include/                     
│   ├── tiny_dsp.h               # entrance header file
│   └── tiny_dsp_config.h        # dsp module configuration file
│
├── signal/
│   ├── tiny_conv.h              # convolution - header file
│   ├── tiny_conv.c              # convolution - source file
|   ├── tiny_conv_test.h         # convolution - test header file
|   ├── tiny_conv_test.c         # convolution - test source file
│   ├── tiny_corr.h              # correlation - header file
│   ├── tiny_corr.c              # correlation - source file
|   ├── tiny_corr_test.h         # correlation - test header file
|   ├── tiny_corr_test.c         # correlation - test source file
|   ├── tiny_resample.h          # resampling - header file
|   ├── tiny_resample.c          # resampling - source file
|   ├── tiny_resample_test.h     # resampling - test header file
|   └── tiny_resample_test.c     # resampling - test source file
│
├── filter/
│
├── transform/
│   ├── tiny_dwt.h               # discrete wavelet transform - header file
│   ├── tiny_dwt.c               # discrete wavelet transform - source file
│   ├── tiny_dwt_test.h          # discrete wavelet transform - test header file
│   └── tiny_dwt_test.c          # discrete wavelet transform - test source
│
└── support/
```