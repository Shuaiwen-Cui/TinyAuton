# VECTOR OPERATIONS

## LIST OF FUNCTIONS

```c
// Addition
tiny_error_t tiny_vec_add_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out);
tiny_error_t tiny_vec_addc_f32(const float *input, float *output, int len, float C, int step_in, int step_out);
// Subtraction
tiny_error_t tiny_vec_sub_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out);
tiny_error_t tiny_vec_subc_f32(const float *input, float *output, int len, float C, int step_in, int step_out);
// Multiplication
tiny_error_t tiny_vec_mul_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out);
tiny_error_t tiny_vec_mulc_f32(const float *input, float *output, int len, float C, int step_in, int step_out);
// Division
tiny_error_t tiny_vec_div_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out, bool allow_divide_by_zero);
tiny_error_t tiny_vec_divc_f32(const float *input, float *output, int len, float C, int step_in, int step_out, bool allow_divide_by_zero);
// Square root
tiny_error_t tiny_vec_sqrt_f32(const float *input, float *output, int len);
tiny_error_t tiny_vec_sqrtf_f32(const float *input, float *output, int len);
tiny_error_t tiny_vec_inv_sqrt_f32(const float *input, float *output, int len);
tiny_error_t tiny_vec_inv_sqrtf_f32(const float *input, float *output, int len);
// Dot product
tiny_error_t tiny_vec_dotprod_f32(const float *src1, const float *src2, float *dest, int len);
tiny_error_t tiny_vec_dotprode_f32(const float *src1, const float *src2, float *dest, int len, int step1, int step2);
```

## ADDITION

### Addition of Two Vectors

```c
tiny_error_t tiny_vec_add_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out);
```

**Function:** Computes the element-wise addition of two vectors.

**Parameters:**

- `input1`: Pointer to the first input vector.
- `input2`: Pointer to the second input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vectors.
- `step1`: Step size for the first input vector.
- `step2`: Step size for the second input vector.
- `step_out`: Step size for the output vector.
  
**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

### Addition of a Vector and a Constant

```c
tiny_error_t tiny_vec_addc_f32(const float *input, float *output, int len, float C, int step_in, int step_out);
```

**Function:** Computes the element-wise addition of a vector and a constant.

**Parameters:**

- `input`: Pointer to the input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vector.
- `C`: Constant value to be added.
- `step_in`: Step size for the input vector.
- `step_out`: Step size for the output vector.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

## SUBTRACTION

### Subtraction of Two Vectors

```c
tiny_error_t tiny_vec_sub_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out);
```

**Function:** Computes the element-wise subtraction of two vectors.

**Parameters:**

- `input1`: Pointer to the first input vector.
- `input2`: Pointer to the second input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vectors.
- `step1`: Step size for the first input vector.
- `step2`: Step size for the second input vector.
- `step_out`: Step size for the output vector.
  
**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

### Subtraction of a Vector and a Constant

```c
tiny_error_t tiny_vec_subc_f32(const float *input, float *output, int len, float C, int step_in, int step_out);
```

**Function:** Computes the element-wise subtraction of a vector and a constant.

**Parameters:**

- `input`: Pointer to the input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vector.
- `C`: Constant value to be subtracted.
- `step_in`: Step size for the input vector.
- `step_out`: Step size for the output vector.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

## MULTIPLICATION

### Multiplication of Two Vectors

```c
tiny_error_t tiny_vec_mul_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out);
```

**Function:** Computes the element-wise multiplication of two vectors.

**Parameters:**

- `input1`: Pointer to the first input vector.
- `input2`: Pointer to the second input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vectors.
- `step1`: Step size for the first input vector.
- `step2`: Step size for the second input vector.
- `step_out`: Step size for the output vector.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

### Multiplication of a Vector and a Constant

```c
tiny_error_t tiny_vec_mulc_f32(const float *input, float *output, int len, float C, int step_in, int step_out);
```

**Function:** Computes the element-wise multiplication of a vector and a constant.

**Parameters:**

- `input`: Pointer to the input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vector.
- `C`: Constant value to be multiplied.
- `step_in`: Step size for the input vector.
- `step_out`: Step size for the output vector.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

## DIVISION

### Division of Two Vectors

```c
tiny_error_t tiny_vec_div_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out, bool allow_divide_by_zero);
```

**Function:** Computes the element-wise division of two vectors.

**Parameters:**

- `input1`: Pointer to the first input vector.
- `input2`: Pointer to the second input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vectors.
- `step1`: Step size for the first input vector.
- `step2`: Step size for the second input vector.
- `step_out`: Step size for the output vector.
- `allow_divide_by_zero`: Flag to allow division by zero (true or false).

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

### Division of a Vector and a Constant

```c
tiny_error_t tiny_vec_divc_f32(const float *input, float *output, int len, float C, int step_in, int step_out, bool allow_divide_by_zero);
```

**Function:** Computes the element-wise division of a vector and a constant.

**Parameters:**

- `input`: Pointer to the input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vector.
- `C`: Constant value to be divided.
- `step_in`: Step size for the input vector.
- `step_out`: Step size for the output vector.
- `allow_divide_by_zero`: Flag to allow division by zero (true or false).

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

## SQUARE ROOT

### Square Root of a Vector

```c
tiny_error_t tiny_vec_sqrt_f32(const float *input, float *output, int len);
```

**Function:** Computes the element-wise square root of a vector.

**Parameters:**

- `input`: Pointer to the input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vector.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

### Square Root of a Vector (Fast)

```c
tiny_error_t tiny_vec_sqrtf_f32(const float *input, float *output, int len);
```

**Function:** Computes the element-wise square root of a vector (fast version).

**Parameters:**

- `input`: Pointer to the input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vector.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

### Inverse Square Root of a Vector

```c
tiny_error_t tiny_vec_inv_sqrt_f32(const float *input, float *output, int len);
```

**Function:** Computes the element-wise inverse square root of a vector.

**Parameters:**

- `input`: Pointer to the input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vector.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

### Inverse Square Root of a Vector (Fast)

```c
tiny_error_t tiny_vec_inv_sqrtf_f32(const float *input, float *output, int len);
```

**Function:** Computes the element-wise inverse square root of a vector (fast version).

**Parameters:**

- `input`: Pointer to the input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vector.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

## DOT PRODUCT

### Dot Product of Two Vectors

```c
tiny_error_t tiny_vec_dotprod_f32(const float *src1, const float *src2, float *dest, int len);
```

**Function:** Computes the dot product of two vectors.

**Parameters:**

- `src1`: Pointer to the first input vector.
- `src2`: Pointer to the second input vector.
- `dest`: Pointer to the output scalar value.
- `len`: Length of the vectors.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.


### Dot Product of Two Vectors with Different Steps

```c
tiny_error_t tiny_vec_dotprode_f32(const float *src1, const float *src2, float *dest, int len, int step1, int step2);
```

**Function:** Computes the dot product of two vectors with different step sizes.

**Parameters:**

- `src1`: Pointer to the first input vector.
- `src2`: Pointer to the second input vector.
- `dest`: Pointer to the output scalar value.
- `len`: Length of the vectors.
- `step1`: Step size for the first input vector.
- `step2`: Step size for the second input vector.
- `step_out`: Step size for the output vector.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.
