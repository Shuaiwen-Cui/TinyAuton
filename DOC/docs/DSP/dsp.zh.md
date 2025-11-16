# 数字信号处理

!!! note
    该组件用于旨在为边缘设备提供信号处理的一系列函数，设计主旨为轻量高效，范围为常用重要的信号处理算法。

!!! note
    该组件基于ESP32官方数字信号处理库 [ESP-DSP](https://docs.espressif.com/projects/esp-dsp/en/latest/esp32/index.html) 进行封装和扩展，提供了更高层次的API接口。先前TinyMath已经对应了ESP-DSP中的Math, Matrix, DotProduct模块，ESP-DSP中的其余模块对应本组件TinyDSP库。除此以外，TinyDSP还提供了ESP-DSP中未曾提供的一些功能，重点覆盖结构健康监测等场景。


## 组件依赖

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

## 架构与功能目录

### 依赖关系示意图

![](tiny_dsp.png)

### 代码树

```txt
tiny_dsp/
├── include/                     
│   ├── tiny_dsp.h               # entrance header file
│   └── tiny_dsp_config.h        # dsp module configuration file
│
├── signal/
│   ├── tiny_conv.h              # convolution - header file
│   ├── tiny_conv.c              # convolution - source file
│   ├── tiny_conv_test.h         # convolution - test header file
│   ├── tiny_conv_test.c         # convolution - test source file
│   ├── tiny_corr.h              # correlation - header file
│   ├── tiny_corr.c              # correlation - source file
│   ├── tiny_corr_test.h         # correlation - test header file
│   ├── tiny_corr_test.c         # correlation - test source file
│   ├── tiny_resample.h          # resampling - header file
│   ├── tiny_resample.c          # resampling - source file
│   ├── tiny_resample_test.h     # resampling - test header file
│   └── tiny_resample_test.c     # resampling - test source file
│
├── filter/
│   ├── tiny_fir.h               # FIR filter - header file
│   ├── tiny_fir.c               # FIR filter - source file
│   ├── tiny_fir_test.h          # FIR filter - test header file
│   ├── tiny_fir_test.c          # FIR filter - test source file
│   ├── tiny_iir.h               # IIR filter - header file
│   ├── tiny_iir.c               # IIR filter - source file
│   ├── tiny_iir_test.h          # IIR filter - test header file
│   └── tiny_iir_test.c          # IIR filter - test source file
│
├── transform/
│   ├── tiny_fft.h               # fast fourier transform - header file
│   ├── tiny_fft.c               # fast fourier transform - source file
│   ├── tiny_fft_test.h          # fast fourier transform - test header file
│   ├── tiny_fft_test.c          # fast fourier transform - test source file
│   ├── tiny_dwt.h               # discrete wavelet transform - header file
│   ├── tiny_dwt.c               # discrete wavelet transform - source file
│   ├── tiny_dwt_test.h          # discrete wavelet transform - test header file
│   ├── tiny_dwt_test.c          # discrete wavelet transform - test source file
│   ├── tiny_ica.h               # independent component analysis - header file
│   ├── tiny_ica.c               # independent component analysis - source file
│   ├── tiny_ica_test.h          # independent component analysis - test header file
│   └── tiny_ica_test.c          # independent component analysis - test source file
│
└── support/
    ├── tiny_view.h              # signal view/support - header file
    ├── tiny_view.c              # signal view/support - source file
    ├── tiny_view_test.h         # signal view/support - test header file
    └── tiny_view_test.c         # signal view/support - test source file
```