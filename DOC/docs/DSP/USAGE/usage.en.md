# USAGE INSTRUCTIONS

!!! info "Usage Instructions"
    This document provides usage instructions for the `tiny_dsp` module. 

## Import TinyMath as a Whole

!!! info
    Suitable for C projects or projects with a simple structure in C++.

```c
#include "tiny_dsp.h"
```

## Import TinyMath by Module
!!! info
    Suitable for projects that require precise control over module imports or complex C++ projects.

```c
#include "tiny_conv.h" // import convolution module
#include "tiny_corr.h" // import correlation module
...
```

!!! tip
    For specific usage methods, please refer to the test code.