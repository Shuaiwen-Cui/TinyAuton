# 向量操作

## 目录

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

## 加法

### 两个向量的加法

```c
tiny_error_t tiny_vec_add_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out);
```
**功能：** 计算两个向量的逐元素加法。

**参数：**

- `input1`：指向第一个输入向量的指针。
- `input2`：指向第二个输入向量的指针。
- `output`：指向输出向量的指针。
- `len`：向量的长度。
- `step1`：第一个输入向量的步长。
- `step2`：第二个输入向量的步长。
- `step_out`：输出向量的步长。

**返回值：** 返回 `tiny_error_t` 类型的错误码，表示操作是否成功。

### 向量与常数的加法

```c
tiny_error_t tiny_vec_addc_f32(const float *input, float *output, int len, float C, int step_in, int step_out);
```
**功能：** 计算向量与常数的逐元素加法。

**参数：**

- `input`：指向输入向量的指针。
- `output`：指向输出向量的指针。
- `len`：向量的长度。
- `C`：常数值。
- `step_in`：输入向量的步长。
- `step_out`：输出向量的步长。

**返回值：** 返回 `tiny_error_t` 类型的错误码，表示操作是否成功。

## 减法

### 两个向量的减法

```c
tiny_error_t tiny_vec_sub_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out);
```

**功能：** 计算两个向量的逐元素减法。

**参数：**

- `input1`：指向第一个输入向量的指针。
- `input2`：指向第二个输入向量的指针。
- `output`：指向输出向量的指针。
- `len`：向量的长度。
- `step1`：第一个输入向量的步长。
- `step2`：第二个输入向量的步长。
- `step_out`：输出向量的步长。

**返回值：** 返回 `tiny_error_t` 类型的错误码，表示操作是否成功。

### 向量与常数的减法

```c
tiny_error_t tiny_vec_subc_f32(const float *input, float *output, int len, float C, int step_in, int step_out);
```

**功能：** 计算向量与常数的逐元素减法。

**参数：**

- `input`：指向输入向量的指针。
- `output`：指向输出向量的指针。
- `len`：向量的长度。
- `C`：常数值。
- `step_in`：输入向量的步长。
- `step_out`：输出向量的步长。
  
**返回值：** 返回 `tiny_error_t` 类型的错误码，表示操作是否成功。

## 乘法

### 两个向量的乘法

```c
tiny_error_t tiny_vec_mul_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out);
```

**功能：** 计算两个向量的逐元素乘法。

**参数：**

- `input1`：指向第一个输入向量的指针。
- `input2`：指向第二个输入向量的指针。
- `output`：指向输出向量的指针。
- `len`：向量的长度。
- `step1`：第一个输入向量的步长。
- `step2`：第二个输入向量的步长。
- `step_out`：输出向量的步长。
  
**返回值：** 返回 `tiny_error_t` 类型的错误码，表示操作是否成功。

### 向量与常数的乘法

```c
tiny_error_t tiny_vec_mulc_f32(const float *input, float *output, int len, float C, int step_in, int step_out);
```

**功能：** 计算向量与常数的逐元素乘法。

**参数：**

- `input`：指向输入向量的指针。
- `output`：指向输出向量的指针。
- `len`：向量的长度。
- `C`：常数值。
- `step_in`：输入向量的步长。
- `step_out`：输出向量的步长。

**返回值：** 返回 `tiny_error_t` 类型的错误码，表示操作是否成功。

## 除法

### 两个向量的除法

```c
tiny_error_t tiny_vec_div_f32(const float *input1, const float *input2, float *output, int len, int step1, int step2, int step_out, bool allow_divide_by_zero);
```

**功能：** 计算两个向量的逐元素除法。

**参数：**

- `input1`：指向第一个输入向量的指针。
- `input2`：指向第二个输入向量的指针。
- `output`：指向输出向量的指针。
- `len`：向量的长度。
- `step1`：第一个输入向量的步长。
- `step2`：第二个输入向量的步长。
- `step_out`：输出向量的步长。
- `allow_divide_by_zero`：布尔值，指示是否允许除以零的操作。

### 向量与常数的除法

```c
tiny_error_t tiny_vec_divc_f32(const float *input, float *output, int len, float C, int step_in, int step_out, bool allow_divide_by_zero);
```

**功能：** 计算向量与常数的逐元素除法。

**参数：**

- `input`：指向输入向量的指针。
- `output`：指向输出向量的指针。
- `len`：向量的长度。
- `C`：常数值。
- `step_in`：输入向量的步长。
- `step_out`：输出向量的步长。
- `allow_divide_by_zero`：布尔值，指示是否允许除以零的操作。

**返回值：** 返回 `tiny_error_t` 类型的错误码，表示操作是否成功。

## 平方根

### 向量的平方根

```c
tiny_error_t tiny_vec_sqrt_f32(const float *input, float *output, int len);
```

**功能：** 计算向量的逐元素平方根。

**参数：**

- `input`：指向输入向量的指针。
- `output`：指向输出向量的指针。
- `len`：向量的长度。

**返回值：** 返回 `tiny_error_t` 类型的错误码，表示操作是否成功。

### 向量的平方根（快速）

```c
tiny_error_t tiny_vec_sqrtf_f32(const float *input, float *output, int len);
```

**功能：** 计算向量的逐元素平方根（快速版本）。

**参数：**

- `input`：指向输入向量的指针。
- `output`：指向输出向量的指针。
- `len`：向量的长度。

**返回值：** 返回 `tiny_error_t` 类型的错误码，表示操作是否成功。

### 向量的平方根倒数

```c
tiny_error_t tiny_vec_inv_sqrt_f32(const float *input, float *output, int len);
```

**功能：** 计算向量的逐元素平方根倒数。

**参数：**

- `input`：指向输入向量的指针。
- `output`：指向输出向量的指针。
- `len`：向量的长度。

**返回值：** 返回 `tiny_error_t` 类型的错误码，表示操作是否成功。

### 向量的平方根倒数（快速）

```c
tiny_error_t tiny_vec_inv_sqrtf_f32(const float *input, float *output, int len);
```

**功能：** 计算向量的逐元素平方根倒数（快速版本）。

**参数：**

- `input`：指向输入向量的指针。
- `output`：指向输出向量的指针。
- `len`：向量的长度。

**返回值：** 返回 `tiny_error_t` 类型的错误码，表示操作是否成功。

## 点积

### 向量的点积

```c
tiny_error_t tiny_vec_dotprod_f32(const float *src1, const float *src2, float *dest, int len);
```

**功能：** 计算两个向量的点积。

**参数：**

- `src1`：指向第一个输入向量的指针。
- `src2`：指向第二个输入向量的指针。
- `dest`：指向输出结果的指针。
- `len`：向量的长度。

**返回值：** 返回 `tiny_error_t` 类型的错误码，表示操作是否成功。

### 向量的点积（带步长）

```c
tiny_error_t tiny_vec_dotprode_f32(const float *src1, const float *src2, float *dest, int len, int step1, int step2);
```

**功能：** 计算两个向量的点积（带步长）。

**参数：**

- `src1`：指向第一个输入向量的指针。
- `src2`：指向第二个输入向量的指针。
- `dest`：指向输出结果的指针。
- `len`：向量的长度。
- `step1`：第一个输入向量的步长。
- `step2`：第二个输入向量的步长。

**返回值：** 返回 `tiny_error_t` 类型的错误码，表示操作是否成功。