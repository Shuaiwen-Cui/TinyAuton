# USAGE INSTRUCTIONS {#usage-instructions}

!!! info "Implementation and records"
    APIs in this section are based on `CODE/AIoTNode-TinyAuton-MATH/middleware/`. Source excerpts and serial output include historical records; check the project entry and enabled selectors before reproducing a test.

!!! info "Usage Instructions"
    This document provides usage instructions for the `math` module in Python. 
    It includes examples and explanations of various functions and methods available in the module.

## Import TinyMath as a Whole {#import-tinymath-as-a-whole}

!!! info
    Suitable for C projects or projects with a simple structure in C++.

```c
#include "tiny_math.h"
```

## Import TinyMath by Module {#import-tinymath-by-module}
!!! info
    Suitable for projects that require precise control over module imports or complex C++ projects.

```c
#include "tiny_vec.h" // Import vector module
#include "tiny_mat.h" // Import matrix module
```

```cpp
#include "tiny_matrix.hpp" // Import advanced matrix module
```

!!! tip "Note"

    - `tiny_vec.h` and `tiny_mat.h` are header files for the C language version, suitable for C programming.

    - `tiny_matrix.hpp` is a header file for the C++ language version, suitable for C++ programming.

    In simple terms, C language projects can only use `tiny_vec.h` and `tiny_mat.h`, while C++ projects can use `tiny_vec.h`, `tiny_mat.h`, and `tiny_matrix.hpp`.

!!! tip
    For specific usage methods, please refer to the test code.
