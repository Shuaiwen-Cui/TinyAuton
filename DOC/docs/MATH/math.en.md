# MATH OPERATIONS {#math-operations}

## Choose an API by task {#auton-math-guide}

| Need | API and entry |
|---|---|
| Vectors and dot products | [tiny_vec](VECTOR/api.md) |
| C array matrix operations | [tiny_mat](MATRIX/tiny-mat-api.md) |
| C++ objects, ROI and arithmetic | [tiny::Mat](MATRIX/OVERVIEW/api.md) |
| Solving, decomposition and eigenvalues | [Linear systems](MATRIX/LINALG/api.md) · [Decomposition](MATRIX/DECOMP/api.md) · [Eigenvalues](MATRIX/EIGEN/api.md) |
| Historical verification | [Matrix tests](MATRIX/TESTS/overview.md) |

Current directories are `include/vec/mat`. C++ `tiny::Mat` provides its own advanced linear algebra; TinySHM's newer standalone C modules are absent here.

!!! info "Implementation and records"
    APIs in this section are based on `CODE/AIoTNode-TinyAuton-MATH/middleware/`. Source excerpts and serial output include historical records; check the project entry and enabled selectors before reproducing a test.

!!! note
    This component is designed for **mathematical operations**. It is a lightweight library that provides basic mathematical functions to facilitate onboard computation and AI model inference. The library is designed to be **lightweight and efficient**, making it suitable for edge computing applications.

!!! note
    This component is a wrapper and extension of the official ESP32 digital signal processing library [ESP-DSP](https://docs.espressif.com/projects/esp-dsp/en/latest/esp32/index.html), providing higher-level API interfaces. In simple terms, the TinyMath library corresponds to the Math, Matrix, and DotProduct modules in ESP-DSP, while the other modules in ESP-DSP correspond to the TinyDSP library.

## COMPONENT DEPENDENCIES {#component-dependencies}

```c
set(src_dirs
    .
    vec
    mat
)

set(include_dirs
    .
    include
    vec
    mat
)

set(requires
    tiny_toolbox
)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})

```

## ARCHITECTURE AND DIRECTORY {#architecture-and-directory}

### Dependency Diagram {#dependency-diagram}
![](tiny_math.png)

### Code Tree {#code-tree}

```txt
TinyMath
    ├── CMakeLists.txt
    ├── include
    |   ├── tiny_error_type.h // error type header file
    |   ├── tiny_constant.h // constant header file
    |   ├── tiny_math_config.h // configuration header file
    |   └── tiny_math.h // main header file, include this file where you want to use the library
    ├── vec
    |   ├── tiny_vec.h // vector header file
    |   ├── tiny_vec.c // vector source file
    |   ├── tiny_vec_test.c // vector test file
    |   └── tiny_vec_test.h // vector test header file
    ├── mat
    |   ├── tiny_mat.h // matrix header file - c
    |   ├── tiny_mat.c // matrix source file - c
    |   ├── tiny_mat_test.c // matrix test file - c 
    |   ├── tiny_mat_test.h // matrix test header file - c
    |   ├── tiny_matrix.hpp // matrix header file - cpp
    |   ├── tiny_matrix.cpp // matrix source file - cpp
    |   ├── tiny_matrix_test.cpp // matrix test file - cpp
    |   └── tiny_matrix_test.hpp // matrix test header file - cpp
    └── ...
```
