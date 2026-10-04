# VECTOR OPERATIONS {#vector-operations}

!!! info "Implementation and records"
    APIs in this section are based on `CODE/AIoTNode-TinyAuton-MATH/middleware/`. Source excerpts and serial output include historical records; check the project entry and enabled selectors before reproducing a test.

## LIST OF FUNCTIONS {#list-of-functions}

```c
tiny_error_t tiny_vec_add_f32(const float *input1, const float *input2, float *output, int len, int stride1, int stride2, int stride_out);
tiny_error_t tiny_vec_addc_f32(const float *input, float *output, int len, float C, int stride_in, int stride_out);
tiny_error_t tiny_vec_sub_f32(const float *input1, const float *input2, float *output, int len, int stride1, int stride2, int stride_out);
tiny_error_t tiny_vec_subc_f32(const float *input, float *output, int len, float C, int stride_in, int stride_out);
tiny_error_t tiny_vec_mul_f32(const float *input1, const float *input2, float *output, int len, int stride1, int stride2, int stride_out);
tiny_error_t tiny_vec_mulc_f32(const float *input, float *output, int len, float C, int stride_in, int stride_out);
tiny_error_t tiny_vec_div_f32(const float *input1, const float *input2, float *output, int len, int stride1, int stride2, int stride_out, bool allow_divide_by_zero);
tiny_error_t tiny_vec_divc_f32(const float *input, float *output, int len, float C, int stride_in, int stride_out, bool allow_divide_by_zero);
tiny_error_t tiny_vec_sqrt_f32(const float *input, float *output, int len);
tiny_error_t tiny_vec_sqrtf_f32(const float *input, float *output, int len);
tiny_error_t tiny_vec_inv_sqrt_f32(const float *input, float *output, int len);
tiny_error_t tiny_vec_inv_sqrtf_f32(const float *input, float *output, int len);
tiny_error_t tiny_vec_dotprod_f32(const float *src1, const float *src2, float *dest, int len);
tiny_error_t tiny_vec_dotprode_f32(const float *src1, const float *src2, float *dest, int len, int stride1, int stride2);
```

## ADDITION {#addition}

### Addition of Two Vectors {#addition-of-two-vectors}

```c
tiny_error_t tiny_vec_add_f32(const float *input1, const float *input2, float *output, int len, int stride1, int stride2, int stride_out);
```

**Function:** Computes the element-wise addition of two vectors.

**Parameters:**

- `input1`: Pointer to the first input vector.
- `input2`: Pointer to the second input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vectors.
- `stride1`: Step size for the first input vector.
- `stride2`: Step size for the second input vector.
- `stride_out`: Step size for the output vector.
  
**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

### Addition of a Vector and a Constant {#addition-of-a-vector-and-a-constant}

```c
tiny_error_t tiny_vec_addc_f32(const float *input, float *output, int len, float C, int stride_in, int stride_out);
```

**Function:** Computes the element-wise addition of a vector and a constant.

**Parameters:**

- `input`: Pointer to the input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vector.
- `C`: Constant value to be added.
- `stride_in`: Step size for the input vector.
- `stride_out`: Step size for the output vector.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

## SUBTRACTION {#subtraction}

### Subtraction of Two Vectors {#subtraction-of-two-vectors}

```c
tiny_error_t tiny_vec_sub_f32(const float *input1, const float *input2, float *output, int len, int stride1, int stride2, int stride_out);
```

**Function:** Computes the element-wise subtraction of two vectors.

**Parameters:**

- `input1`: Pointer to the first input vector.
- `input2`: Pointer to the second input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vectors.
- `stride1`: Step size for the first input vector.
- `stride2`: Step size for the second input vector.
- `stride_out`: Step size for the output vector.
  
**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

### Subtraction of a Vector and a Constant {#subtraction-of-a-vector-and-a-constant}

```c
tiny_error_t tiny_vec_subc_f32(const float *input, float *output, int len, float C, int stride_in, int stride_out);
```

**Function:** Computes the element-wise subtraction of a vector and a constant.

**Parameters:**

- `input`: Pointer to the input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vector.
- `C`: Constant value to be subtracted.
- `stride_in`: Step size for the input vector.
- `stride_out`: Step size for the output vector.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

## MULTIPLICATION {#multiplication}

### Multiplication of Two Vectors {#multiplication-of-two-vectors}

```c
tiny_error_t tiny_vec_mul_f32(const float *input1, const float *input2, float *output, int len, int stride1, int stride2, int stride_out);
```

**Function:** Computes the element-wise multiplication of two vectors.

**Parameters:**

- `input1`: Pointer to the first input vector.
- `input2`: Pointer to the second input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vectors.
- `stride1`: Step size for the first input vector.
- `stride2`: Step size for the second input vector.
- `stride_out`: Step size for the output vector.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

### Multiplication of a Vector and a Constant {#multiplication-of-a-vector-and-a-constant}

```c
tiny_error_t tiny_vec_mulc_f32(const float *input, float *output, int len, float C, int stride_in, int stride_out);
```

**Function:** Computes the element-wise multiplication of a vector and a constant.

**Parameters:**

- `input`: Pointer to the input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vector.
- `C`: Constant value to be multiplied.
- `stride_in`: Step size for the input vector.
- `stride_out`: Step size for the output vector.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

## DIVISION {#division}

### Division of Two Vectors {#division-of-two-vectors}

```c
tiny_error_t tiny_vec_div_f32(const float *input1, const float *input2, float *output, int len, int stride1, int stride2, int stride_out, bool allow_divide_by_zero);
```

**Function:** Computes the element-wise division of two vectors.

**Parameters:**

- `input1`: Pointer to the first input vector.
- `input2`: Pointer to the second input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vectors.
- `stride1`: Step size for the first input vector.
- `stride2`: Step size for the second input vector.
- `stride_out`: Step size for the output vector.
- `allow_divide_by_zero`: Flag to allow division by zero (true or false).

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

### Division of a Vector and a Constant {#division-of-a-vector-and-a-constant}

```c
tiny_error_t tiny_vec_divc_f32(const float *input, float *output, int len, float C, int stride_in, int stride_out, bool allow_divide_by_zero);
```

**Function:** Computes the element-wise division of a vector and a constant.

**Parameters:**

- `input`: Pointer to the input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vector.
- `C`: Constant value to be divided.
- `stride_in`: Step size for the input vector.
- `stride_out`: Step size for the output vector.
- `allow_divide_by_zero`: Flag to allow division by zero (true or false).

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

## SQUARE ROOT {#square-root}

### Square Root of a Vector {#square-root-of-a-vector}

```c
tiny_error_t tiny_vec_sqrt_f32(const float *input, float *output, int len);
```

**Function:** Computes the element-wise square root of a vector.

**Parameters:**

- `input`: Pointer to the input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vector.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

### Square Root of a Vector (Fast) {#square-root-of-a-vector-fast}

```c
tiny_error_t tiny_vec_sqrtf_f32(const float *input, float *output, int len);
```

**Function:** Computes the element-wise square root of a vector (fast version).

**Parameters:**

- `input`: Pointer to the input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vector.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

### Inverse Square Root of a Vector {#inverse-square-root-of-a-vector}

```c
tiny_error_t tiny_vec_inv_sqrt_f32(const float *input, float *output, int len);
```

**Function:** Computes the element-wise inverse square root of a vector.

**Parameters:**

- `input`: Pointer to the input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vector.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

### Inverse Square Root of a Vector (Fast) {#inverse-square-root-of-a-vector-fast}

```c
tiny_error_t tiny_vec_inv_sqrtf_f32(const float *input, float *output, int len);
```

**Function:** Computes the element-wise inverse square root of a vector (fast version).

**Parameters:**

- `input`: Pointer to the input vector.
- `output`: Pointer to the output vector.
- `len`: Length of the vector.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.

## DOT PRODUCT {#dot-product}

### Dot Product of Two Vectors {#dot-product-of-two-vectors}

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


### Dot Product of Two Vectors with Different Strides {#dot-product-of-two-vectors-with-different-strides}

```c
tiny_error_t tiny_vec_dotprode_f32(const float *src1, const float *src2, float *dest, int len, int stride1, int stride2);
```

**Function:** Computes the dot product of two vectors with different stride sizes.

**Parameters:**

- `src1`: Pointer to the first input vector.
- `src2`: Pointer to the second input vector.
- `dest`: Pointer to the output scalar value.
- `len`: Length of the vectors.
- `stride1`: Step size for the first input vector.
- `stride2`: Step size for the second input vector.

**Returns:** Returns a `tiny_error_t` type error code indicating whether the operation was successful.
