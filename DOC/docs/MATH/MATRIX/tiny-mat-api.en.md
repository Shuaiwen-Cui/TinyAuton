# MATRIX OPERATIONS - TINY_MAT

!!! INFO "About tiny_mat library"
    tiny_mat is a C implementation of a matrix library that provides basic matrix operation functions. It supports operations such as addition, subtraction, and multiplication of floating-point matrices. This library is suitable for embedded systems and real-time applications that require matrix calculations. The library is based on the ANSI C standard, ensuring good portability and performance, while also supporting platform acceleration through configuration files (ESP32).

!!! TIP "About the usage of tiny_mat library"
    The functionality of tiny_mat is completely covered by tiny_matrix, which means that the functions in tiny_matrix include all the functions of tiny_mat. For simple matrix operations, you can only include the tiny_mat library; for complex matrix operations, it is recommended to use the tiny_matrix library. The tiny_matrix library is a C++ implementation of a matrix library that provides richer functionality and better performance. It supports operations such as addition, subtraction, multiplication, transposition, and inversion of floating-point and integer matrices.



## LIST OF FUNCTIONS

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

## UTILITY FUNCTIONS

### Print Matrix
```c
void print_matrix(const char *name, const float *mat, int rows, int cols);
```
**Function:** Print a matrix in row-major order.

**Parameters:**

- `name`: Name of the matrix.

- `mat`: Pointer to the matrix data.

- `rows`: Number of rows in the matrix.

- `cols`: Number of columns in the matrix.

**Returns:** None.

### Print Padded Matrix
```c
void print_matrix_padded(const char *name, const float *mat, int rows, int cols, int step);
```
**Function:** Print a matrix in row-major order with padding.

**Parameters:**
- `name`: Name of the matrix.

- `mat`: Pointer to the matrix data.

- `rows`: Number of rows in the matrix.

- `cols`: Number of columns in the matrix.

- `step`: Step size for the matrix data.

**Returns:** None.

## ADDITION

### Matrix Addition
```c
tiny_error_t tiny_mat_add_f32(const float *input1, const float *input2, float *output, int rows, int cols, int padd1, int padd2, int padd_out, int step1, int step2, int step_out);
```

**Function:** Add two matrices.

**Parameters:**

- `input1`: Pointer to the first input matrix.

- `input2`: Pointer to the second input matrix.

- `output`: Pointer to the output matrix.

- `rows`: Number of rows in the matrices.

- `cols`: Number of columns in the matrices.

- `padd1`: Padding for the first input matrix.

- `padd2`: Padding for the second input matrix.

- `padd_out`: Padding for the output matrix.

- `step1`: Step size for the first input matrix.

- `step2`: Step size for the second input matrix.

- `step_out`: Step size for the output matrix.

**Returns:** `tiny_error_t` indicating success or failure.

### Matrix Addition with Constant
```c
tiny_error_t tiny_mat_addc_f32(const float *input, float *output, float C, int rows, int cols, int padd_in, int padd_out, int step_in, int step_out);
```

**Function:** Add a constant to a matrix.

**Parameters:**

- `input`: Pointer to the input matrix.

- `output`: Pointer to the output matrix.

- `C`: Constant to add.

- `rows`: Number of rows in the matrix.

- `cols`: Number of columns in the matrix.

- `padd_in`: Padding for the input matrix.

- `padd_out`: Padding for the output matrix.

- `step_in`: Step size for the input matrix.

- `step_out`: Step size for the output matrix.

**Returns:** `tiny_error_t` indicating success or failure.

## SUBTRACTION

### Matrix Subtraction
```c
tiny_error_t tiny_mat_sub_f32(const float *input1, const float *input2, float *output, int rows, int cols, int padd1, int padd2, int padd_out, int step1, int step2, int step_out);
```

**Function:** Subtract two matrices.

**Parameters:**

- `input1`: Pointer to the first input matrix.

- `input2`: Pointer to the second input matrix.

- `output`: Pointer to the output matrix.

- `rows`: Number of rows in the matrices.

- `cols`: Number of columns in the matrices.

- `padd1`: Padding for the first input matrix.

- `padd2`: Padding for the second input matrix.

- `padd_out`: Padding for the output matrix.

- `step1`: Step size for the first input matrix.

- `step2`: Step size for the second input matrix.

- `step_out`: Step size for the output matrix.

**Returns:** `tiny_error_t` indicating success or failure.

### Matrix Subtraction with Constant
```c
tiny_error_t tiny_mat_subc_f32(const float *input, float *output, float C, int rows, int cols, int padd_in, int padd_out, int step_in, int step_out);
```

**Function:** Subtract a constant from a matrix.

**Parameters:**

- `input`: Pointer to the input matrix.

- `output`: Pointer to the output matrix.

- `C`: Constant to subtract.

- `rows`: Number of rows in the matrix.

- `cols`: Number of columns in the matrix.

- `padd_in`: Padding for the input matrix.

- `padd_out`: Padding for the output matrix.

- `step_in`: Step size for the input matrix.

- `step_out`: Step size for the output matrix.

**Returns:** `tiny_error_t` indicating success or failure.

## MULTIPLICATION

### Matrix Multiplication
```c
tiny_error_t tiny_mat_mult_f32(const float *A, const float *B, float *C, int m, int n, int k);
```

**Function:** Multiply two matrices.

**Parameters:**

- `A`: Pointer to the first input matrix.

- `B`: Pointer to the second input matrix.

- `C`: Pointer to the output matrix.

- `m`: Number of rows in the first matrix.

- `n`: Number of columns in the first matrix (and rows in the second matrix).

- `k`: Number of columns in the second matrix.

**Returns:** `tiny_error_t` indicating success or failure.

### Extended Matrix Multiplication
```c
tiny_error_t tiny_mat_mult_ex_f32(const float *A, const float *B, float *C, int A_rows, int A_cols, int B_cols, int A_padding, int B_padding, int C_padding);
```

**Function:** Multiply two matrices with extended parameters.

**Parameters:**

- `A`: Pointer to the first input matrix.

- `B`: Pointer to the second input matrix.

- `C`: Pointer to the output matrix.

- `A_rows`: Number of rows in the first matrix.

- `A_cols`: Number of columns in the first matrix.

- `B_cols`: Number of columns in the second matrix.

- `A_padding`: Padding for the first matrix.

- `B_padding`: Padding for the second matrix.

- `C_padding`: Padding for the output matrix.

**Returns:** `tiny_error_t` indicating success or failure.

### Matrix Multiplication with Constant
```c
tiny_error_t tiny_mat_multc_f32(const float *input, float *output, float C, int rows, int cols, int padd_in, int padd_out, int step_in, int step_out);
```

**Function:** Multiply a matrix by a constant.

**Parameters:**

- `input`: Pointer to the input matrix.

- `output`: Pointer to the output matrix.

- `C`: Constant to multiply.

- `rows`: Number of rows in the matrix.

- `cols`: Number of columns in the matrix.

- `padd_in`: Padding for the input matrix.

- `padd_out`: Padding for the output matrix.

- `step_in`: Step size for the input matrix.

- `step_out`: Step size for the output matrix.

**Returns:** `tiny_error_t` indicating success or failure.
```