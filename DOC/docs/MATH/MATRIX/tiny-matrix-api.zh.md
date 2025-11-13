# 矩阵操作 - TINY_MATRIX

!!! INFO "TINY_MATRIX库"
    - 该库是一个轻量级的矩阵运算库，基于C++实现，提供了基本的矩阵操作和线性代数功能。
    - 该库的设计目标是提供简单易用的矩阵操作接口，适合于嵌入式系统和资源受限的环境。

!!! TIP "使用场景"
    相对于TINY_MAT库而言，TINY_MATRIX库提供了更丰富的功能和更高的灵活性，适合于需要进行复杂矩阵运算的应用场景。但是请注意，该库基于C++编写，

## 目录

```c
TinyMath
    ├──Vector
    └──Matrix
        ├── tiny_mat (c)
        └── tiny_matrix (c++) <---
```

```cpp
/**
 * @file tiny_matrix.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the header file for the submodule matrix (advanced matrix operations) of the tiny_math middleware.
 * @version 1.0
 * @date 2025-04-17
 * @note This file is built on top of the mat.h file from the ESP-DSP library.
 *
 */

#pragma once

/* DEPENDENCIES */
// TinyMath
#include "tiny_math_config.h"
#include "tiny_vec.h"
#include "tiny_mat.h"

// Standard Libraries
#include <iostream>
#include <stdint.h>

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
// ESP32 DSP C++ Matrix library
#include "mat.h"
#endif

/* STATEMENTS */
namespace tiny
{
    class Mat
    {
    public:
        /* === Matrix Metadata === */
        int row;         //< number of rows
        int col;         //< number of columns
        int pad;         //< number of paddings between 2 rows
        int stride;      //< stride = (number of elements in a row) + padding
        int element;     //< number of elements = rows * cols
        int memory;      //< size of the data buffer = rows * stride
        float *data;     //< pointer to the data buffer
        float *temp;     //< pointer to the temporary data buffer
        bool ext_buff;   //< flag indicates that matrix use external buffer
        bool sub_matrix; //< flag indicates that matrix is a subset of another matrix

        /* === Rectangular ROI Structure === */
        /**
         * @name Region of Interest (ROI) Structure
         * @brief This is the structure for ROI
         * 
         */
        struct ROI
        {
            int pos_x;  ///< starting column index
            int pos_y;  ///< starting row index
            int width;  ///< width of ROI (columns)
            int height; ///< height of ROI (rows)

            // ROI constructor
            ROI(int pos_x = 0, int pos_y = 0, int width = 0, int height = 0);

            // resize ROI
            void resize_roi(int pos_x, int pos_y, int width, int height);

            // calculate area of ROI
            int area_roi(void) const;
        };
        
        /* === Printing Functions === */
        // print matrix info
        void print_info() const;

        // print matrix elements, paddings optional
        void print_matrix(bool show_padding);

        /* === Constructors & Destructor === */
        // memory allocation
        void alloc_mem(); // Allocate internal memory

        // constructor
        Mat();
        Mat(int rows, int cols);
        Mat(int rows, int cols, int stride);
        Mat(float *data, int rows, int cols);
        Mat(float *data, int rows, int cols, int stride);
        Mat(const Mat &src);

        // destructor
        ~Mat();

        /* === Element Access === */
        // access matrix elements - non const
        inline float &operator()(int row, int col) { return data[row * stride + col]; }

        // access matrix elements - const             
        inline const float &operator()(int row, int col) const { return data[row * stride + col]; }

        /* === Data Manipulation === */
        // copy other matrix into this matrix as a sub-matrix
        tiny_error_t copy_paste(const Mat &src, int row_pos, int col_pos);

        // copy header of other matrix to this matrix
        tiny_error_t copy_head(const Mat &src);

        // get a view (shallow copy) of sub-matrix (ROI) from this matrix
        Mat view_roi(int start_row, int start_col, int roi_rows, int roi_cols) const;

        // get a view (shallow copy) of sub-matrix (ROI) from this matrix using ROI structure
        Mat view_roi(const Mat::ROI &roi) const;

        // get a replica (deep copy) of sub-matrix (ROI) 
        Mat copy_roi(int start_row, int start_col, int roi_rows, int roi_cols);

        // get a replica (deep copy) of sub-matrix (ROI) using ROI structure
        Mat copy_roi(const Mat::ROI &roi);

        // get a block of matrix
        Mat block(int start_row, int start_col, int block_rows, int block_cols);

        // swap rows
        void swap_rows(int row1, int row2);

        // clear matrix
        void clear(void);

        /* === Arithmetic Operators === */
        Mat &operator=(const Mat &src);    // Copy assignment
        Mat &operator+=(const Mat &A);     // Add matrix
        Mat &operator+=(float C);          // Add constant
        Mat &operator-=(const Mat &A);     // Subtract matrix
        Mat &operator-=(float C);          // Subtract constant 
        Mat &operator*=(const Mat &A);     // Multiply matrix
        Mat &operator*=(float C);          // Multiply constant
        Mat &operator/=(const Mat &B);     // Divide matrix
        Mat &operator/=(float C);          // Divide constant
        Mat operator^(int C);              // Exponentiation

        /* === Linear Algebra === */
        Mat transpose();                   // Transpose matrix
        Mat cofactor(int row, int col);    // cofactor matrix extraction
        float determinant();
        Mat adjoint(); 
        void normalize();
        float norm() const;
        Mat inverse_adjoint();
        static Mat eye(int size);
        static Mat augment(const Mat &A, const Mat &B);
        static Mat ones(int rows, int cols);
        static Mat ones(int size);
        Mat gaussian_eliminate() const;
        Mat row_reduce_from_gaussian();
        Mat inverse_gje(); // Inverse using Gaussian-Jordan elimination
        float dotprod(const Mat &A, const Mat &B);
        Mat solve(const Mat &A, const Mat &b);
        Mat band_solve(Mat A, Mat b, int k);
        Mat roots(Mat A, Mat y);
        
        /* === Eigenvalue & Eigenvector Decomposition === */
        // Forward declarations (structures defined after class)
        struct EigenPair;
        struct EigenDecomposition;
        
        // Check if matrix is symmetric (within tolerance)
        bool is_symmetric(float tolerance = 1e-6f) const;
        
        // Power iteration method: compute dominant eigenvalue and eigenvector
        // Fast method suitable for real-time SHM applications
        EigenPair power_iteration(int max_iter = 1000, float tolerance = 1e-6f) const;
        
        // Jacobi method: complete eigendecomposition for symmetric matrices
        // Robust and accurate, ideal for structural dynamics matrices
        EigenDecomposition eigendecompose_jacobi(float tolerance = 1e-6f, int max_iter = 100) const;
        
        // QR algorithm: complete eigendecomposition for general matrices
        // Supports non-symmetric matrices, may have complex eigenvalues
        EigenDecomposition eigendecompose_qr(int max_iter = 100, float tolerance = 1e-6f) const;
        
        // Automatic method selection: uses Jacobi for symmetric, QR for general
        // Convenient interface for edge computing applications
        EigenDecomposition eigendecompose(float tolerance = 1e-6f) const;
        
    protected:

    private:

    };

    /* === Eigenvalue & Eigenvector Decomposition Structures === */
    /**
     * @brief Structure to hold a single eigenvalue-eigenvector pair
     * @note Used primarily for power iteration method
     */
    struct Mat::EigenPair
    {
        float eigenvalue;      ///< Eigenvalue (real part)
        Mat eigenvector;       ///< Corresponding eigenvector (column vector)
        int iterations;        ///< Number of iterations performed
        tiny_error_t status;   ///< Computation status
        
        EigenPair();
    };
    
    /**
     * @brief Structure to hold complete eigenvalue decomposition results
     * @note Contains all eigenvalues and eigenvectors
     */
    struct Mat::EigenDecomposition
    {
        Mat eigenvalues;       ///< Eigenvalues (diagonal matrix or vector)
        Mat eigenvectors;      ///< Eigenvector matrix (each column is an eigenvector)
        int iterations;        ///< Number of iterations performed
        tiny_error_t status;   ///< Computation status
        
        EigenDecomposition();
    };

    /* === Stream Operators === */
    std::ostream &operator<<(std::ostream &os, const Mat &m);
    std::ostream &operator<<(std::ostream &os, const Mat::ROI &roi);
    std::istream &operator>>(std::istream &is, Mat &m);

    /* === Global Arithmetic Operators === */
    Mat operator+(const Mat &A, const Mat &B);
    Mat operator+(const Mat &A, float C);
    Mat operator-(const Mat &A, const Mat &B);
    Mat operator-(const Mat &A, float C);
    Mat operator*(const Mat &A, const Mat &B);
    Mat operator*(const Mat &A, float C);
    Mat operator*(float C, const Mat &A);
    Mat operator/(const Mat &A, float C);
    Mat operator/(const Mat &A, const Mat &B);
    bool operator==(const Mat &A, const Mat &B);

}
```

## 元数据

- `int row`: 行数

- `int col`: 列数

- `int pad`: 行间填充数

- `int stride`: 行内元素数 + 填充数

- `int element`: 元素数

- `int memory`: 数据缓冲区大小 = 行数 * 步幅

- `float *data`: 数据缓冲区指针

- `float *temp`: 临时数据缓冲区指针

- `bool ext_buff`: 标志矩阵是否使用外部缓冲区

- `bool sub_matrix`: 标志矩阵是否为另一个矩阵的子集

## ROI 结构

### 元数据

- `int pos_x`: 起始列索引

- `int pos_y`: 起始行索引

- `int width`: ROI 的宽度（列数）

- `int height`: ROI 的高度（行数）

### ROI 构造函数

```cpp
Mat::ROI::ROI(int pos_x = 0, int pos_y = 0, int width = 0, int height = 0);
```

**描述**: 构造一个 ROI 对象，默认值为 (0, 0, 0, 0)。

**参数**:

- `int pos_x`: 起始列索引

- `int pos_y`: 起始行索引

- `int width`: ROI 的宽度（列数）

- `int height`: ROI 的高度（行数）

### ROI 重置函数

```cpp
void Mat::ROI::resize_roi(int pos_x, int pos_y, int width, int height);
```

**描述**: 重置 ROI 的位置和大小。

**参数**:

- `int pos_x`: 起始列索引

- `int pos_y`: 起始行索引

- `int width`: ROI 的宽度（列数）

- `int height`: ROI 的高度（行数）

**返回值**: void

### ROI 面积函数

```cpp
int Mat::ROI::area_roi(void) const;
```

**描述**: 计算 ROI 的面积。

**参数**: void

**返回值**: 整数类型 ROI 的面积

## 打印函数

### 打印矩阵信息

```cpp
void Mat::print_info() const
```

**描述**: 打印矩阵的基本信息，包括行数、列数、元素数、填充数、步幅数、内存大小、数据缓冲区指针、临时数据缓冲区指针、外部缓冲区标志、子矩阵标志。

**参数**: void

**返回值**: void

### 打印矩阵元素

```cpp
void Mat::print_matrix(bool show_padding);
```

**描述**: 打印矩阵的元素。

**参数**:

- `bool show_padding`: 是否显示填充区元素， true 显示，false 不显示

**返回值**: void

## 构造与析构函数

### 默认构造函数

```cpp
Mat::Mat();
```

**描述**: 默认构造函数将使用默认值初始化一个矩阵对象。这个函数会创建一个一行一列的矩阵，唯一的元素是0。

**参数**: void

### 构造函数 - Mat(int rows, int cols)

```cpp
Mat::Mat(int rows, int cols);
```

**描述**: 构造一个指定行数和列数的矩阵对象。

**参数**:

- `int rows`: 行数

- `int cols`: 列数

### 构造函数 - Mat(int rows, int cols, int stride)

```cpp
Mat::Mat(int rows, int cols, int stride);
```

**描述**: 构造一个指定行数、列数和步幅的矩阵对象。

**参数**:

- `int rows`: 行数

- `int cols`: 列数

- `int stride`: 步幅

### 构造函数 - Mat(float *data, int rows, int cols)

```cpp
Mat::Mat(float *data, int rows, int cols);
```

**描述**: 构造一个指定行数和列数的矩阵对象，并使用给定的数据缓冲区。

**参数**:

- `float *data`: 数据缓冲区指针

- `int rows`: 行数

- `int cols`: 列数

### 构造函数 - Mat(float *data, int rows, int cols, int stride)

```cpp
Mat::Mat(float *data, int rows, int cols, int stride);
```

**描述**: 构造一个指定行数、列数和步幅的矩阵对象，并使用给定的数据缓冲区。

**参数**:

- `float *data`: 数据缓冲区指针

- `int rows`: 行数

- `int cols`: 列数

- `int stride`: 步幅

### 构造函数 - Mat(const Mat &src)

```cpp
Mat::Mat(const Mat &src);
```

**描述**: 构造一个矩阵对象，并使用给定的矩阵对象的头部信息。

**参数**:

- `const Mat &src`: 源矩阵对象

### 析构函数

```cpp
Mat::~Mat();
```

**描述**: 析构函数释放矩阵对象的内存。

**参数**: void

!!! note
    对于构造函数，其名称必须与类名相同，并且不能有返回类型。如上所述，对于 C++，只要参数的排列顺序不同，就可以通过更改参数的数量和顺序来重新加载函数名称。当对象超出范围时，析构函数将自动调用。


## 元素访问

### 非常量访问

```cpp
inline float &operator()(int row, int col);
```

**描述**: 访问矩阵元素，返回对指定行和列的引用。

**参数**:

- `int row`: 行索引

- `int col`: 列索引

**返回值**: 对应位置的元素 float类型

### 常量访问

```cpp
inline const float &operator()(int row, int col) const;
```

**描述**: 访问矩阵元素，返回对指定行和列的常量引用。

**参数**:

- `int row`: 行索引

- `int col`: 列索引

**返回值**: 对应位置的元素 float类型

!!! 注意
    这两个函数实际上是重新定义了 `()` 运算符，它允许你使用 `matrix(row, col)` 语法访问矩阵的元素。

## 数据操作

### 复制其他矩阵到当前矩阵

```cpp
tiny_error_t copy_paste(const Mat &src, int row_pos, int col_pos);
```

**描述**: 将源矩阵的元素复制到当前矩阵的指定位置。

**参数**:

- `const Mat &src`: 源矩阵对象

- `int row_pos`: 目标矩阵的起始行索引

- `int col_pos`: 目标矩阵的起始列索引

**返回值**: 错误代码

### 复制矩阵头部

```cpp
tiny_error_t copy_head(const Mat &src);
```

**描述**: 将源矩阵的头部信息复制到当前矩阵。

**参数**:

- `const Mat &src`: 源矩阵对象

**返回值**: 错误代码

### 获取子矩阵视图

```cpp
Mat view_roi(int start_row, int start_col, int roi_rows, int roi_cols) const;
```

**描述**: 获取当前矩阵的子矩阵视图。

**参数**:

- `int start_row`: 起始行索引

- `int start_col`: 起始列索引

- `int roi_rows`: 子矩阵的行数

- `int roi_cols`: 子矩阵的列数

**返回值**: 子矩阵对象

### 获取子矩阵视图 - 使用 ROI 结构

```cpp
Mat view_roi(const Mat::ROI &roi) const;
```

**描述**: 获取当前矩阵的子矩阵视图，使用 ROI 结构。

**参数**:

- `const Mat::ROI &roi`: ROI 结构对象

**返回值**: 子矩阵对象

!!! 警告
    与 ESP-DSP 不同，view_roi 不允许设置步长，因为它会根据列数和填充数自动计算步长。该函数还会拒绝非法请求，即超出范围的请求。


### 获取子矩阵副本

```cpp
Mat copy_roi(int start_row, int start_col, int roi_rows, int roi_cols);
```

**描述**: 获取当前矩阵的子矩阵副本。

**参数**:

- `int start_row`: 起始行索引

- `int start_col`: 起始列索引

- `int roi_rows`: 子矩阵的行数

- `int roi_cols`: 子矩阵的列数

**返回值**: 子矩阵对象

### 获取子矩阵副本 - 使用 ROI 结构

```cpp
Mat copy_roi(const Mat::ROI &roi);
```

**描述**: 获取当前矩阵的子矩阵副本，使用 ROI 结构。

**参数**:

- `const Mat::ROI &roi`: ROI 结构对象

**返回值**: 子矩阵对象

### 获取矩阵块

```cpp
Mat block(int start_row, int start_col, int block_rows, int block_cols);
```

**描述**: 获取当前矩阵的块。

**参数**:

- `int start_row`: 起始行索引

- `int start_col`: 起始列索引

- `int block_rows`: 块的行数

- `int block_cols`: 块的列数

**返回值**: 块对象

!!! tip “view_roi | copy_roi | block 之间的区别”

  - `view_roi`：从该矩阵浅拷贝子矩阵 (ROI)。

  - `copy_roi`：从该矩阵深拷贝子矩阵 (ROI)。刚性拷贝，速度更快。

  - `block`：从该矩阵深拷贝块。柔性拷贝，速度更慢。

### 交换行

```cpp
void Mat::swap_rows(int row1, int row2);
```

**描述**: 交换当前矩阵的两行。

**参数**:

- `int row1`: 第一行索引

- `int row2`: 第二行索引

**返回值**: void

### 交换列

```cpp
void Mat::swap_cols(int col1, int col2);
```

**描述**: 交换当前矩阵的两列。在需要列主元的高斯消元等算法中很有用。

**参数**:

- `int col1`: 第一列索引

- `int col2`: 第二列索引

**返回值**: void

### 清除矩阵

```cpp
void Mat::clear(void);
```

**描述**: 通过将所有元素设置为零来清除矩阵。

**参数**: void

**返回值**: void

## 算术运算符

!!! note "注意"
    本节定义了作用于当前矩阵本身的算术运算符。这些运算符已被重载以执行矩阵运算。

### 赋值运算符

```cpp
Mat &operator=(const Mat &src);
```

**描述**: 赋值运算符，将源矩阵的元素复制到当前矩阵。

**参数**:

- `const Mat &src`: 源矩阵对象

### 加法运算符

```cpp
Mat &operator+=(const Mat &A);
```

**描述**: 加法运算符，将源矩阵的元素加到当前矩阵。

**参数**:

- `const Mat &A`: 源矩阵对象

### 加法运算符 - 常量

```cpp
Mat &operator+=(float C);
```

**描述**: 将常量按元素加到当前矩阵。

**参数**:

- `float C`: 要加的常量

**返回值**: Mat& - 当前矩阵的引用

### 减法运算符

```cpp
Mat &operator-=(const Mat &A);
```

**描述**: 从当前矩阵中按元素减去源矩阵。

**参数**:

- `const Mat &A`: 源矩阵对象

**返回值**: Mat& - 当前矩阵的引用

### 减法运算符 - 常量

```cpp
Mat &operator-=(float C);
```

**描述**: 从当前矩阵中按元素减去常量。

**参数**:

- `float C`: 要减的常量

**返回值**: Mat& - 当前矩阵的引用

### 乘法运算符

```cpp
Mat &operator*=(const Mat &A);
```

**描述**: 矩阵乘法：this = this * A

**参数**:

- `const Mat &A`: 要乘的矩阵

**返回值**: Mat& - 当前矩阵的引用

### 乘法运算符 - 常量

```cpp
Mat &operator*=(float C);
```

**描述**: 按元素乘以常量。

**参数**:

- `float C`: 常量乘数

**返回值**: Mat& - 当前矩阵的引用

### 除法运算符

```cpp
Mat &operator/=(const Mat &B);
```

**描述**: 按元素除法：this = this / B

**参数**:

- `const Mat &B`: 除数矩阵

**返回值**: Mat& - 当前矩阵的引用

### 除法运算符 - 常量

```cpp
Mat &operator/=(float C);
```

**描述**: 将当前矩阵按元素除以常量。

**参数**:

- `float C`: 常量除数

**返回值**: Mat& - 当前矩阵的引用

### 幂运算符

```cpp
Mat operator^(int C);
```

**描述**: 按元素整数幂运算。返回一个新矩阵，其中每个元素都提升到给定幂次。

**参数**:

- `int C`: 指数（整数）

**返回值**: Mat - 幂运算后的新矩阵

## 线性代数

### 转置矩阵

```cpp
Mat Mat::transpose();
```

**描述**: 计算矩阵的转置，返回新矩阵。

**参数**: void

**返回值**: Mat - 转置后的矩阵对象

### 余子式矩阵

```cpp
Mat Mat::minor(int row, int col);
```

**描述**: 通过移除指定的行和列来计算余子式矩阵。余子式是移除一行一列后得到的子矩阵。

**参数**:

- `int row`: 要移除的行索引

- `int col`: 要移除的列索引

**返回值**: Mat - (n-1)x(n-1) 的余子式矩阵

### 代数余子式矩阵

```cpp
Mat Mat::cofactor(int row, int col);
```

**描述**: 计算代数余子式矩阵（与余子式矩阵相同）。代数余子式矩阵与余子式矩阵相同。符号 (-1)^(i+j) 在计算代数余子式值时应用，而不是应用到矩阵元素本身。

**参数**:

- `int row`: 要移除的行索引

- `int col`: 要移除的列索引

**返回值**: Mat - (n-1)x(n-1) 的代数余子式矩阵（与余子式矩阵相同）


### 行列式

```cpp
float Mat::determinant();
```

**描述**: 使用拉普拉斯展开计算方阵的行列式。效率较低，仅适用于小矩阵！！！

**参数**: void

**返回值**: float - 行列式的值


### 伴随矩阵

```cpp
Mat Mat::adjoint();
```

**描述**: 计算方阵的伴随（或伴随转置）矩阵。

**参数**: void

**返回值**: Mat - 伴随矩阵对象


### 归一化

```cpp
void Mat::normalize();
```

**描述**: 使用L2范数（Frobenius范数）归一化矩阵。归一化后，||Matrix|| = 1。

**参数**: void

**返回值**: void

### 范数

```cpp
float Mat::norm() const;
```

**描述**: 计算矩阵的Frobenius范数（L2范数）。

**参数**: void

**返回值**: float - 计算得到的矩阵范数

### 矩阵求逆 -- 基于伴随矩阵
```cpp
Mat Mat::inverse_adjoint();
```

**描述**: 使用伴随矩阵法计算方阵的逆矩阵。如果矩阵是奇异的，返回零矩阵。

**参数**: void

**返回值**: Mat - 逆矩阵对象。如果矩阵是奇异的，返回零矩阵

### 单位矩阵

```cpp
static Mat Mat::eye(int size);
```

**描述**: 生成指定大小的单位矩阵。

**参数**:

- `int size`: 方阵的维度

**返回值**: Mat - 单位矩阵 (size x size)

### 增广矩阵（水平连接）

```cpp
static Mat Mat::augment(const Mat &A, const Mat &B);
```

**描述**: 通过水平连接两个矩阵创建增广矩阵 [A | B]。A和B的行数必须匹配。

**参数**:

- `const Mat &A`: 左侧矩阵

- `const Mat &B`: 右侧矩阵

**返回值**: Mat - 增广矩阵 [A B]

### 垂直堆叠

```cpp
static Mat Mat::vstack(const Mat &A, const Mat &B);
```

**描述**: 垂直堆叠两个矩阵 [A; B]。A和B的列数必须匹配。

**参数**:

- `const Mat &A`: 顶部矩阵

- `const Mat &B`: 底部矩阵

**返回值**: Mat - 垂直堆叠的矩阵 [A; B]

### 全1矩阵（矩形）

```cpp
static Mat Mat::ones(int rows, int cols);
```

**描述**: 创建指定大小的全1矩阵。

**参数**:

- `int rows`: 行数

- `int cols`: 列数

**返回值**: Mat - 矩阵 [rows x cols]，所有元素 = 1

### 全1矩阵（方阵）

```cpp
static Mat Mat::ones(int size);
```

**描述**: 创建指定大小的方阵，所有元素为1。

**参数**:

- `int size`: 方阵的大小（行数 = 列数）

**返回值**: Mat - 方阵 [size x size]，所有元素 = 1

### 高斯消元法

```cpp
Mat Mat::gaussian_eliminate() const;
```

**描述**: 执行高斯消元法，将矩阵转换为行阶梯形式（REF）。

**参数**: void

**返回值**: Mat - 上三角矩阵（REF形式）

### 从高斯消元到行最简形式

```cpp
Mat Mat::row_reduce_from_gaussian();
```

**描述**: 将矩阵（假设已为行阶梯形式）转换为简化行阶梯形式（RREF）。

**参数**: void

**返回值**: Mat - RREF形式的矩阵

### 高斯-约旦消元法求逆

```cpp
Mat Mat::inverse_gje();
```

**描述**: 使用高斯-约旦消元法计算方阵的逆矩阵。

**参数**: void

**返回值**: Mat - 如果矩阵可逆则返回逆矩阵，否则返回空矩阵

### 点积

```cpp
float Mat::dotprod(const Mat &A, const Mat &B);
```

**描述**: 计算两个向量（Nx1）的点积。

**参数**:

- `const Mat &A`: 输入向量 A (Nx1)

- `const Mat &B`: 输入向量 B (Nx1)

**返回值**: float - 计算得到的点积值

### 解线性方程组

```cpp
Mat Mat::solve(const Mat &A, const Mat &b);
```

**描述**: 使用高斯消元法求解线性方程组 Ax = b。

**参数**:

- `const Mat &A`: 系数矩阵 (NxN)

- `const Mat &b`: 结果向量 (Nx1)

**返回值**: Mat - 解向量 (Nx1)，包含方程 Ax = b 的根

### 带状矩阵求解

```cpp
Mat Mat::band_solve(Mat A, Mat b, int k);
```

**描述**: 使用优化的高斯消元法求解带状矩阵方程组 Ax = b。

**参数**:

- `Mat A`: 系数矩阵 (NxN) - 带状矩阵

- `Mat b`: 结果向量 (Nx1)

- `int k`: 矩阵的带宽（非零带的宽度）

**返回值**: Mat - 解向量 (Nx1)，包含方程 Ax = b 的根

### 线性系统求根

```cpp
Mat Mat::roots(Mat A, Mat y);
```

**描述**: 使用不同方法求解矩阵。这是 'solve' 函数的另一种实现，原理上没有区别。此方法使用高斯消元法求解线性系统 A * x = y。

**参数**:

- `Mat A`: 矩阵 [N]x[N]，包含输入系数

- `Mat y`: 向量 [N]x[1]，包含结果值

**返回值**: Mat - 矩阵 [N]x[1]，包含根

## 线性代数 - 特征值与特征向量 (Eigen)

### 结构体：`Mat::EigenPair`

```cpp
Mat::EigenPair::EigenPair();
// fields:
// float eigenvalue;      // 支配（最大模）特征值
// Mat eigenvector;       // 对应的特征向量（n x 1）
// int iterations;        // 迭代次数（若为迭代法返回）
// tiny_error_t status;   // 计算状态（TINY_OK / 错误码）
```

**描述**: 用于保存单一特征值/特征向量对及其计算信息。常由 `power_iteration` 返回。

### 结构体：`Mat::EigenDecomposition`

```cpp
Mat::EigenDecomposition::EigenDecomposition();
// fields:
// Mat eigenvalues;    // n x 1 矩阵，存放特征值
// Mat eigenvectors;   // n x n 矩阵，通常每一列为对应的特征向量
// int iterations;     // 迭代次数（若为迭代法返回）
// tiny_error_t status; // 计算状态（TINY_OK / 错误码）
```

**描述**: 用于保存完整的特征值分解结果，包括全部特征值和对应的特征向量矩阵。

### 判断对称性

```cpp
bool Mat::is_symmetric(float tolerance) const;
```

**描述**: 检查矩阵是否为对称矩阵（A(i,j) 与 A(j,i) 的差值小于 `tolerance`）。

**参数**:
- `float tolerance`: 允许的最大差值（例如 1e-6）。

**返回值**: `true`（近似对称）或 `false`。

### 幂迭代（求主特征值/向量）

```cpp
Mat::EigenPair Mat::power_iteration(int max_iter, float tolerance) const;
```

**描述**: 使用幂迭代法计算矩阵的主特征值（绝对值最大）及对应特征向量。

**参数**:
- `int max_iter`：最大迭代次数（代码注释中常用默认值 `1000`）。
- `float tolerance`：收敛容差（例如 `1e-6`）。

**返回值**: 一个 `EigenPair`，包含 `eigenvalue`、`eigenvector`、`iterations` 与 `status`。

**注意**:
- 要求矩阵为方阵且数据指针非空；若矩阵非方阵或数据为空，返回带错误状态的 `EigenPair`。
- 单次幂迭代只给出主特征值/向量；若需全部特征值/向量请使用下面的分解函数。

### Jacobi 特征分解（对称矩阵）

```cpp
Mat::EigenDecomposition Mat::eigendecompose_jacobi(float tolerance, int max_iter) const;
```

**描述**: 对称矩阵优先使用 Jacobi 方法求全部特征值与特征向量，收敛性与稳定性好，适合结构动力学场景（SHM）。

**参数**:
- `float tolerance`：收敛阈值（例如 `1e-6`）。
- `int max_iter`：最大迭代次数（例如 `100`）。

**返回值**: `EigenDecomposition`（包含 `eigenvalues` 与 `eigenvectors`；计算状态可查看 `status` 字段）。

**注意**: 若矩阵不是近似对称，函数会发出警告，但仍可运行；对于非对称矩阵推荐使用 QR 方法。

### QR 特征分解（一般矩阵）

```cpp
Mat::EigenDecomposition Mat::eigendecompose_qr(int max_iter, float tolerance) const;
```

**描述**: 使用 QR 算法对一般矩阵进行特征值分解（不要求对称）。对于非对称矩阵可能产生复数特征值——当前实现仅返回实部（会存在精度或收敛性限制）。

**参数**:
- `int max_iter`：最大 QR 迭代次数（默认示例值 `100`）。
- `float tolerance`：收敛容差（例如 `1e-6`）。

**返回值**: `EigenDecomposition`（`eigenvalues`、`eigenvectors`、`iterations`、`status`）。

**注意**: 算法使用 Gram–Schmidt 构造 Q/R，可能在数值病态情况下不稳定；对于严格对称矩阵优先使用 Jacobi。

### 自动特征分解（根据矩阵特性选择方法）

```cpp
Mat::EigenDecomposition Mat::eigendecompose(float tolerance) const;
```

**描述**: 简便接口，会先调用 `is_symmetric(tolerance * 10)` 判断矩阵是否近似对称：
- 若为对称，使用 `eigendecompose_jacobi`；
- 否则使用 `eigendecompose_qr`。

**参数**:
- `float tolerance`：用于对称性检测与分解收敛判断（建议 `1e-6`）。

**返回值**: `EigenDecomposition`。

**使用建议**:
- 若明确知道矩阵为对称矩阵（如刚度矩阵、质量矩阵），直接使用 `eigendecompose_jacobi`；对非对称或不确定情形，可使用 `eigendecompose`。
- 对于较大矩阵或嵌入式场景，Jacobi/QR 计算成本较高，请权衡数值稳定性与性能。


## 流操作符

### 矩阵输出到流

```cpp
std::ostream &operator<<(std::ostream &os, const Mat &m);
```

**描述**: 将矩阵输出到流。

**参数**:

- `std::ostream &os`: 输出流对象

- `const Mat &m`: 矩阵对象

### 子矩阵输出到流

```cpp
std::ostream &operator<<(std::ostream &os, const Mat::ROI &roi);
```

**描述**: 将子矩阵输出到流。

**参数**:

- `std::ostream &os`: 输出流对象

- `const Mat::ROI &roi`: 子矩阵对象

### 矩阵输入流

```cpp
std::istream &operator>>(std::istream &is, Mat &m);
```

**描述**: 从流中读取矩阵。

**参数**:

- `std::istream &is`: 输入流对象

- `Mat &m`: 矩阵对象

## 全局算数运算符

!!! 提示
    本节中的运算符返回一个新的矩阵对象，作为运算结果。原始矩阵保持不变。与上一节不同，这些运算符旨在对当前矩阵本身执行运算。

### 加法运算符

```cpp
Mat operator+(const Mat &A, const Mat &B);
```

**描述**: 按元素将两个矩阵相加。

**参数**:

- `const Mat &A`: 第一个矩阵

- `const Mat &B`: 第二个矩阵

**返回值**: Mat - 结果矩阵 A+B

### 加法运算符 - 常量

```cpp
Mat operator+(const Mat &A, float C);
```

**描述**: 按元素将常量加到矩阵。

**参数**:

- `const Mat &A`: 输入矩阵 A

- `float C`: 输入常量

**返回值**: Mat - 结果矩阵 A+C

### 减法运算符

```cpp
Mat operator-(const Mat &A, const Mat &B);
```

**描述**: 按元素将两个矩阵相减。

**参数**:

- `const Mat &A`: 第一个矩阵

- `const Mat &B`: 第二个矩阵

**返回值**: Mat - 结果矩阵 A-B

### 减法运算符 - 常量

```cpp
Mat operator-(const Mat &A, float C);
```

**描述**: 按元素从矩阵中减去常量。

**参数**:

- `const Mat &A`: 输入矩阵 A

- `float C`: 输入常量

**返回值**: Mat - 结果矩阵 A-C

### 乘法运算符

```cpp
Mat operator*(const Mat &A, const Mat &B);
```

**描述**: 将两个矩阵相乘（矩阵乘法）。

**参数**:

- `const Mat &A`: 第一个矩阵

- `const Mat &B`: 第二个矩阵

**返回值**: Mat - 结果矩阵 A*B

### 乘法运算符 - 常量

```cpp
Mat operator*(const Mat &A, float C);
```

**描述**: 按元素将矩阵乘以常量。

**参数**:

- `const Mat &A`: 输入矩阵 A

- `float C`: 浮点数值

**返回值**: Mat - 结果矩阵 A*C

### 乘法运算符 - 常量（左侧）

```cpp
Mat operator*(float C, const Mat &A);
```

**描述**: 按元素将常量乘以矩阵。

**参数**:

- `float C`: 浮点数值

- `const Mat &A`: 输入矩阵 A

**返回值**: Mat - 结果矩阵 C*A

### 除法运算符

```cpp
Mat operator/(const Mat &A, float C);
```

**描述**: 按元素将矩阵除以常量。

**参数**:

- `const Mat &A`: 输入矩阵 A

- `float C`: 浮点数值

**返回值**: Mat - 结果矩阵 A/C

### 除法运算符 - 矩阵

```cpp
Mat operator/(const Mat &A, const Mat &B);
```

**描述**: 按元素将矩阵 A 除以矩阵 B。

**参数**:

- `const Mat &A`: 输入矩阵 A

- `const Mat &B`: 输入矩阵 B

**返回值**: Mat - 结果矩阵 C，其中 C[i,j] = A[i,j]/B[i,j]

### 等于运算符

```cpp
bool operator==(const Mat &A, const Mat &B);
```

**描述**: 等于运算符，检查两个矩阵是否相等。

**参数**:

- `const Mat &A`: 第一个矩阵对象

- `const Mat &B`: 第二个矩阵对象

**返回值**: 布尔值，表示两个矩阵是否相等






