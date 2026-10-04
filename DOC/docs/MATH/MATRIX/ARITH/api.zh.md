# 数据操作与运算符

接口以 MATH 工程的 `middleware/tiny_math/mat/tiny_matrix.hpp` 为依据。矩阵元素为 `float`；先检查尺寸、布局和返回状态，再使用结果。

[返回概览](../OVERVIEW/api.md) · [历史测试](../TESTS/overview.md)

## 数据操作 {#_17}

### 复制其他矩阵到当前矩阵 {#_18}

```cpp
tiny_error_t copy_paste(const Mat &src, int row_pos, int col_pos);
```

**描述**:

将源矩阵的元素复制到当前矩阵的指定位置。

**参数**:

- `const Mat &src`: 源矩阵对象

- `int row_pos`: 目标矩阵的起始行索引

- `int col_pos`: 目标矩阵的起始列索引

**返回值**:

错误代码

### 复制矩阵头部 {#_19}

```cpp
tiny_error_t copy_head(const Mat &src);
```

**描述**:

将源矩阵的头部信息复制到当前矩阵。

**参数**:

- `const Mat &src`: 源矩阵对象

**返回值**:

错误代码

### 获取子矩阵视图 {#_20}

```cpp
Mat view_roi(int start_row, int start_col, int roi_rows, int roi_cols) const;
```

**描述**:

获取当前矩阵的子矩阵视图。

**参数**:

- `int start_row`: 起始行索引

- `int start_col`: 起始列索引

- `int roi_rows`: 子矩阵的行数

- `int roi_cols`: 子矩阵的列数

**返回值**:

子矩阵对象

### 获取子矩阵视图 - 使用 ROI 结构 {#-roi}

```cpp
Mat view_roi(const Mat::ROI &roi) const;
```

**描述**:

获取当前矩阵的子矩阵视图，使用 ROI 结构。

**参数**:

- `const Mat::ROI &roi`: ROI 结构对象

**返回值**:

子矩阵对象

!!! 警告
    与 ESP-DSP 不同，view_roi 不允许设置步长，因为它会根据列数和填充数自动计算步长。该函数还会拒绝非法请求，即超出范围的请求。


### 获取子矩阵副本 {#_21}

```cpp
Mat copy_roi(int start_row, int start_col, int height, int width);
```

**描述**:

获取当前矩阵的子矩阵副本。

**参数**:

- `int start_row`: 起始行索引

- `int start_col`: 起始列索引

- `int height`: 子矩阵的行数

- `int width`: 子矩阵的列数

**返回值**:

子矩阵对象

### 获取子矩阵副本 - 使用 ROI 结构 {#-roi_1}

```cpp
Mat copy_roi(const Mat::ROI &roi);
```

**描述**:

获取当前矩阵的子矩阵副本，使用 ROI 结构。

**参数**:

- `const Mat::ROI &roi`: ROI 结构对象

**返回值**:

子矩阵对象

### 获取矩阵块 {#_22}

```cpp
Mat block(int start_row, int start_col, int block_rows, int block_cols);
```

**描述**:

获取当前矩阵的块。

**参数**:

- `int start_row`: 起始行索引

- `int start_col`: 起始列索引

- `int block_rows`: 块的行数

- `int block_cols`: 块的列数

**返回值**:

块对象

!!! tip "view_roi | copy_roi | block 之间的区别"

    - `view_roi` : 从该矩阵浅拷贝子矩阵 (ROI)。

    - `copy_roi` : 从该矩阵深拷贝子矩阵 (ROI)。复制内存拷贝，速度更快。

    - `block` : 从该矩阵深拷贝块。逐个元素拷贝，速度更慢。
### 交换行 {#_23}

```cpp
void Mat::swap_rows(int row1, int row2);
```

**描述**:

交换当前矩阵的两行。

**参数**:

- `int row1`: 第一行索引

- `int row2`: 第二行索引

**返回值**:

void

### 交换列 {#_24}

```cpp
void Mat::swap_cols(int col1, int col2);
```

**描述**:

交换当前矩阵的两列。

**参数**:

- `int col1`: 第一列索引

- `int col2`: 第二列索引

**返回值**:

void

### 清除矩阵 {#_25}

```cpp
void Mat::clear(void);
```

**描述**:

通过将所有元素设置为零来清除矩阵。

**参数**:

void

**返回值**:

void

## 算术运算符 {#_26}

!!! INFO "就地操作"
    本节定义了作用于当前矩阵本身的算术运算符（就地操作）。这些运算符修改矩阵并返回其引用，支持链式操作如`A += B += C`。这些运算符经过优化以处理填充，并在可用时使用DSP加速函数。

### 拷贝赋值 {#_27}

```cpp
Mat &operator=(const Mat &src);
```

**描述**:

矩阵的拷贝赋值运算符。将源矩阵的元素复制到当前矩阵。必要时通过重新分配内存来处理维度变化。为防止意外数据损坏，禁止对子矩阵视图进行赋值。

**数学原理**:

创建源矩阵的独立副本。与拷贝构造函数不同，这用于现有矩阵。

**参数**:

- `const Mat &src` : 源矩阵。

**返回值**:

Mat& - 对当前矩阵的引用（支持链式操作）。

**使用建议**:

- **内存管理**: 如果维度不同，自动重新分配内存。如果内存是内部分配的，释放旧内存。

- **子矩阵保护**: 禁止对子矩阵视图进行赋值，以防止意外数据损坏。

- **自赋值**: 安全处理自赋值 (A = A)。

- **性能**: 对于n×n矩阵为O(n²)。对于大矩阵，考虑视图是否足够。

### 加法运算符 {#_28}

```cpp
Mat &operator+=(const Mat &A);
```

**描述**:

加法运算符，将源矩阵的元素加到当前矩阵。

**参数**:

- `const Mat &A`: 源矩阵对象

### 加法运算符 - 常量 {#-}

```cpp
Mat &operator+=(float C);
```

**描述**:

将常量按元素加到当前矩阵。

**参数**:

- `float C`: 要加的常量

**返回值**:

Mat& - 当前矩阵的引用

### 减法运算符 {#_29}

```cpp
Mat &operator-=(const Mat &A);
```

**描述**:

从当前矩阵中按元素减去源矩阵。

**参数**:

- `const Mat &A`: 源矩阵对象

**返回值**:

Mat& - 当前矩阵的引用

### 减法运算符 - 常量 {#-_1}

```cpp
Mat &operator-=(float C);
```

**描述**:

从当前矩阵中按元素减去常量。

**参数**:

- `float C`: 要减的常量

**返回值**:

Mat& - 当前矩阵的引用

### 矩阵乘法 {#_30}

```cpp
Mat &operator*=(const Mat &A);
```

**描述**:

矩阵乘法：this = this * A。执行标准矩阵乘法（非逐元素）。当前矩阵的列数必须等于A的行数。

**数学原理**:

矩阵乘法 C = A * B，其中 Cᵢⱼ = Σₖ Aᵢₖ * Bₖⱼ。这是标准矩阵乘积，不是逐元素乘法。

**维度要求**: 
- 当前矩阵: m × n

- 矩阵 A: n × p

- 结果: m × p

**参数**:

- `const Mat &A` : 要乘的矩阵（必须有n行，其中n = 当前矩阵的列数）。

**返回值**:

Mat& - 对当前矩阵的引用。

**使用建议**:

- **内存效率**: 创建临时副本以避免在计算期间覆盖数据，然后更新当前矩阵。

- **填充支持**: 在可用时使用专用DSP函数处理带填充的矩阵。

- **性能**: 对于m×n * n×p乘法为O(mnp)。在ESP32平台上使用优化的DSP函数。

- **常见错误**: 这是矩阵乘法，不是逐元素的。对于逐元素，使用带有`operator()()`的循环。

### 乘法运算符 - 常量 {#-_2}

```cpp
Mat &operator*=(float C);
```

**描述**:

按元素乘以常量。

**参数**:

- `float C`: 常量乘数

**返回值**:

Mat& - 当前矩阵的引用

### 除法运算符 {#_31}

```cpp
Mat &operator/=(const Mat &B);
```

**描述**:

按元素除法：this = this / B

**参数**:

- `const Mat &B`: 除数矩阵

**返回值**:

Mat& - 当前矩阵的引用

### 除法运算符 - 常量 {#-_3}

```cpp
Mat &operator/=(float C);
```

**描述**:

将当前矩阵按元素除以常量。

**参数**:

- `float C`: 常量除数

**返回值**:

Mat& - 当前矩阵的引用

### 幂运算符 {#_32}

```cpp
Mat operator^(int C);
```

**描述**:

按元素整数幂运算。返回一个新矩阵，其中每个元素都提升到给定幂次。

**参数**:

- `int C`: 指数（整数）

**返回值**:

Mat - 幂运算后的新矩阵

## 流操作符 {#_61}

### 矩阵输出流操作符 {#_62}

```cpp
std::ostream &operator<<(std::ostream &os, const Mat &m);
```

**描述**:

矩阵的重载输出流操作符。

**参数**:

- `std::ostream &os` : 输出流。

- `const Mat &m` : 要输出的矩阵。

### ROI输出流操作符 {#roi_5}

```cpp
std::ostream &operator<<(std::ostream &os, const Mat::ROI &roi);
```

**描述**:

ROI结构体的重载输出流操作符。

**参数**:

- `std::ostream &os` : 输出流。

- `const Mat::ROI &roi` : ROI结构。

### 矩阵输入流操作符 {#_63}

```cpp
std::istream &operator>>(std::istream &is, Mat &m);
```

**描述**:

矩阵的重载输入流操作符。

**参数**:

- `std::istream &is` : 输入流。

- `Mat &m` : 要输入的矩阵。

!!! tip 
    本节实际上在显示矩阵方面与打印函数有些重叠。

## 全局算术运算符 {#_64}

!!! INFO "非修改操作"
    本节中的运算符返回一个新的矩阵对象，作为运算结果。原始矩阵保持不变。这些是函数式操作，不修改其操作数，使其可以安全地与const引用和临时对象一起使用。
    
!!! TIP "何时使用"
    - 使用全局运算符 (A + B) 当您想保留原始矩阵时
    - 使用成员运算符 (A += B) 当您想就地修改矩阵时（更节省内存）

### 加法运算符 {#_65}

```cpp
Mat operator+(const Mat &A, const Mat &B);
```

**描述**:

按元素将两个矩阵相加。

**参数**:

- `const Mat &A`: 第一个矩阵

- `const Mat &B`: 第二个矩阵

**返回值**:

Mat - 结果矩阵 A+B

### 加法运算符 - 常量 {#-_9}

```cpp
Mat operator+(const Mat &A, float C);
```

**描述**:

按元素将常量加到矩阵。

**参数**:

- `const Mat &A`: 输入矩阵 A

- `float C`: 输入常量

**返回值**:

Mat - 结果矩阵 A+C

### 减法运算符 {#_66}

```cpp
Mat operator-(const Mat &A, const Mat &B);
```

**描述**:

按元素将两个矩阵相减。

**参数**:

- `const Mat &A`: 第一个矩阵

- `const Mat &B`: 第二个矩阵

**返回值**:

Mat - 结果矩阵 A-B

### 减法运算符 - 常量 {#-_10}

```cpp
Mat operator-(const Mat &A, float C);
```

**描述**:

按元素从矩阵中减去常量。

**参数**:

- `const Mat &A`: 输入矩阵 A

- `float C`: 输入常量

**返回值**:

Mat - 结果矩阵 A-C

### 乘法运算符 {#_67}

```cpp
Mat operator*(const Mat &A, const Mat &B);
```

**描述**:

将两个矩阵相乘（矩阵乘法）。

**参数**:

- `const Mat &A`: 第一个矩阵

- `const Mat &B`: 第二个矩阵

**返回值**:

Mat - 结果矩阵 A*B

### 乘法运算符 - 常量 {#-_11}

```cpp
Mat operator*(const Mat &A, float C);
```

**描述**:

按元素将矩阵乘以常量。

**参数**:

- `const Mat &A`: 输入矩阵 A

- `float C`: 浮点数值

**返回值**:

Mat - 结果矩阵 A*C

### 乘法运算符 - 常量（左侧） {#-_12}

```cpp
Mat operator*(float C, const Mat &A);
```

**描述**:

按元素将常量乘以矩阵。

**参数**:

- `float C`: 浮点数值

- `const Mat &A`: 输入矩阵 A

**返回值**:

Mat - 结果矩阵 C*A

### 除法运算符 {#_68}

```cpp
Mat operator/(const Mat &A, float C);
```

**描述**:

按元素将矩阵除以常量。

**参数**:

- `const Mat &A`: 输入矩阵 A

- `float C`: 浮点数值

**返回值**:

Mat - 结果矩阵 A/C

### 除法运算符 - 矩阵 {#-_13}

```cpp
Mat operator/(const Mat &A, const Mat &B);
```

**描述**:

按元素将矩阵 A 除以矩阵 B。

**参数**:

- `const Mat &A`: 输入矩阵 A

- `const Mat &B`: 输入矩阵 B

**返回值**:

Mat - 结果矩阵 C，其中 C[i,j] = A[i,j]/B[i,j]

### 等于运算符 {#_69}

```cpp
bool operator==(const Mat &A, const Mat &B);
```

**描述**:

等于运算符，检查两个矩阵是否相等。

**参数**:

- `const Mat &A`: 第一个矩阵对象

- `const Mat &B`: 第二个矩阵对象

**返回值**:

布尔值，表示两个矩阵是否相等
