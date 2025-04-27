# 矩阵操作 - TINY_MAT

!!! INFO "关于tiny_mat库"
    tiny_mat是一个C语言实现的矩阵库，提供了基本的矩阵操作函数。它支持浮点数矩阵的加法、减法和乘法等操作。该库适用于需要进行矩阵计算的嵌入式系统和实时应用。该库基于ANSIC C标准，具有良好的可移植性和性能,同时又支持在配置文件中进行配置从而支持平台加速（ESP32）。

!!! TIP "关于tiny_mat库的使用"
    tiny_mat的功能被tiny_matrix完全覆盖，也就是说在tiny_matrix中的功能包含了tiny_mat的所有功能。对于简单的矩阵操作，可以仅引入tiny_mat库；对于复杂的矩阵操作，建议使用tiny_matrix库。tiny_matrix库是一个C++实现的矩阵库，提供了更丰富的功能和更好的性能。它支持浮点数和整数矩阵的加法、减法、乘法、转置、求逆等操作。


## 目录

```c
TinyMath
    ├──Vector
    └──Matrix
        ├── tiny_mat (c) <---
        └── tiny_matrix (c++)
```

```c
// print matrix
void print_matrix(const char *name, const float *mat, int rows, int cols);
// print matrix padded (row-major)
void print_matrix_padded(const char *name, const float *mat, int rows, int cols, int step);
// addition
tiny_error_t tiny_mat_add_f32(const float *input1, const float *input2, float *output, int rows, int cols, int padd1, int padd2, int padd_out, int step1, int step2, int step_out);
tiny_error_t tiny_mat_addc_f32(const float *input, float *output, float C, int rows, int cols, int padd_in, int padd_out, int step_in, int step_out);
// subtraction
tiny_error_t tiny_mat_sub_f32(const float *input1, const float *input2, float *output, int rows, int cols, int padd1, int padd2, int padd_out, int step1, int step2, int step_out);
tiny_error_t tiny_mat_subc_f32(const float *input, float *output, float C, int rows, int cols, int padd_in, int padd_out, int step_in, int step_out);
// multiplication
tiny_error_t tiny_mat_mult_f32(const float *A, const float *B, float *C, int m, int n, int k);
tiny_error_t tiny_mat_mult_ex_f32(const float *A, const float *B, float *C, int A_rows, int A_cols, int B_cols, int A_padding, int B_padding, int C_padding);
tiny_error_t tiny_mat_multc_f32(const float *input, float *output, float C, int rows, int cols, int padd_in, int padd_out, int step_in, int step_out);
```

## 工具函数

### 打印矩阵
```c
void print_matrix(const char *name, const float *mat, int rows, int cols);
```

**函数:** 以行主序打印矩阵。

**参数:**

- `name`: 矩阵名称。

- `mat`: 矩阵数据指针。

- `rows`: 矩阵行数。

- `cols`: 矩阵列数。

**返回:** 无。

### 打印带填充的矩阵
```c
void print_matrix_padded(const char *name, const float *mat, int rows, int cols, int step);
```

**函数:** 以行主序打印带填充的矩阵。

**参数:**

- `name`: 矩阵名称。

- `mat`: 矩阵数据指针。

- `rows`: 矩阵行数。

- `cols`: 矩阵列数。

- `step`: 步长。

**返回:** 无。

### 矩阵加法
```c
tiny_error_t tiny_mat_add_f32(const float *input1, const float *input2, float *output, int rows, int cols, int padd1, int padd2, int padd_out, int step1, int step2, int step_out);
```

**函数:** 矩阵加法。

**参数:**

- `input1`: 输入矩阵1。

- `input2`: 输入矩阵2。

- `output`: 输出矩阵。

- `rows`: 矩阵行数。

- `cols`: 矩阵列数。

- `padd1`: 输入矩阵1的填充。

- `padd2`: 输入矩阵2的填充。

- `padd_out`: 输出矩阵的填充。

- `step1`: 输入矩阵1的步长。

- `step2`: 输入矩阵2的步长。

- `step_out`: 输出矩阵的步长。

**返回:** 错误码。

### 矩阵加常数
```c
tiny_error_t tiny_mat_addc_f32(const float *input, float *output, float C, int rows, int cols, int padd_in, int padd_out, int step_in, int step_out);
```

**函数:** 矩阵加常数。

**参数:**

- `input`: 输入矩阵。

- `output`: 输出矩阵。

- `C`: 常数。

- `rows`: 矩阵行数。

- `cols`: 矩阵列数。

- `padd_in`: 输入矩阵的填充。

- `padd_out`: 输出矩阵的填充。

- `step_in`: 输入矩阵的步长。

- `step_out`: 输出矩阵的步长。

**返回:** 错误码。

### 矩阵减法
```c
tiny_error_t tiny_mat_sub_f32(const float *input1, const float *input2, float *output, int rows, int cols, int padd1, int padd2, int padd_out, int step1, int step2, int step_out);
```

**函数:** 矩阵减法。

**参数:**

- `input1`: 输入矩阵1。

- `input2`: 输入矩阵2。

- `output`: 输出矩阵。

- `rows`: 矩阵行数。

- `cols`: 矩阵列数。

- `padd1`: 输入矩阵1的填充。

- `padd2`: 输入矩阵2的填充。

- `padd_out`: 输出矩阵的填充。

- `step1`: 输入矩阵1的步长。

- `step2`: 输入矩阵2的步长。

- `step_out`: 输出矩阵的步长。

**返回:** 错误码。

### 矩阵减常数
```c
tiny_error_t tiny_mat_subc_f32(const float *input, float *output, float C, int rows, int cols, int padd_in, int padd_out, int step_in, int step_out);
```

**函数:** 矩阵减常数。

**参数:**

- `input`: 输入矩阵。

- `output`: 输出矩阵。

- `C`: 常数。

- `rows`: 矩阵行数。

- `cols`: 矩阵列数。

- `padd_in`: 输入矩阵的填充。

- `padd_out`: 输出矩阵的填充。

- `step_in`: 输入矩阵的步长。

- `step_out`: 输出矩阵的步长。

**返回:** 错误码。

### 矩阵乘法
```c
tiny_error_t tiny_mat_mult_f32(const float *A, const float *B, float *C, int m, int n, int k);
```

**函数:** 矩阵乘法。

**参数:**

- `A`: 输入矩阵A。

- `B`: 输入矩阵B。

- `C`: 输出矩阵C。

- `m`: 矩阵A的行数。

- `n`: 矩阵A的列数。

- `k`: 矩阵B的列数。

**返回:** 错误码。


### 扩展矩阵乘法
```c
tiny_error_t tiny_mat_mult_ex_f32(const float *A, const float *B, float *C, int A_rows, int A_cols, int B_cols, int A_padding, int B_padding, int C_padding);
```

**函数:** 扩展矩阵乘法。

**参数:**

- `A`: 输入矩阵A。

- `B`: 输入矩阵B。

- `C`: 输出矩阵C。

- `A_rows`: 矩阵A的行数。

- `A_cols`: 矩阵A的列数。

- `B_cols`: 矩阵B的列数。

- `A_padding`: 矩阵A的填充。

- `B_padding`: 矩阵B的填充。

- `C_padding`: 矩阵C的填充。

**返回:** 错误码。

### 矩阵乘常数
```c
tiny_error_t tiny_mat_multc_f32(const float *input, float *output, float C, int rows, int cols, int padd_in, int padd_out, int step_in, int step_out);
```

**函数:** 矩阵乘常数。

**参数:**

- `input`: 输入矩阵。

- `output`: 输出矩阵。

- `C`: 常数。

- `rows`: 矩阵行数。

- `cols`: 矩阵列数。

- `padd_in`: 输入矩阵的填充。

- `padd_out`: 输出矩阵的填充。

- `step_in`: 输入矩阵的步长。

- `step_out`: 输出矩阵的步长。

**返回:** 错误码。
