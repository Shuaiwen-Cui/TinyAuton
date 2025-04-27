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
        
    protected:

    private:

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
void swap_rows(int row1, int row2);
```

**描述**: 交换当前矩阵的两行。

**参数**:

- `int row1`: 第一行索引

- `int row2`: 第二行索引

**返回值**: void


### 清除矩阵

```cpp
void clear(void);
```

**描述**: 清除当前矩阵的元素。

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

**描述**: 加法运算符，将常量加到当前矩阵。

**参数**:

- `float C`: 常量

### 减法运算符

```cpp
Mat &operator-=(const Mat &A);
```

**描述**: 减法运算符，将源矩阵的元素从当前矩阵中减去。

**参数**:

- `const Mat &A`: 源矩阵对象

### 减法运算符 - 常量

```cpp
Mat &operator-=(float C);
```

**描述**: 减法运算符，将常量从当前矩阵中减去。

**参数**:

- `float C`: 常量

### 乘法运算符

```cpp
Mat &operator*=(const Mat &A);
```

**描述**: 乘法运算符，将源矩阵的元素乘到当前矩阵。

**参数**:

- `const Mat &A`: 源矩阵对象

### 乘法运算符 - 常量

```cpp
Mat &operator*=(float C);
```

**描述**: 乘法运算符，将常量乘到当前矩阵。

**参数**:

- `float C`: 常量

### 除法运算符

```cpp
Mat &operator/=(const Mat &B);
```

**描述**: 除法运算符，将当前矩阵除以源矩阵。

**参数**:

- `const Mat &B`: 源矩阵对象

### 除法运算符 - 常量

```cpp
Mat &operator/=(float C);
```

**描述**: 除法运算符，将当前矩阵除以常量。

**参数**:

- `float C`: 常量

### 幂运算符

```cpp
Mat operator^(int C);
```

**描述**: 幂运算符，将当前矩阵的元素进行幂运算。

**参数**:

- `int C`: 幂指数

## 线性代数

### 转置矩阵

```cpp
Mat::transpose();
```

**描述**: 转置当前矩阵。

**参数**: void


**返回值**: 转置后的矩阵对象


### 余子式矩阵

```cpp
Mat::cofactor(int row, int col);
```

**描述**: 提取当前矩阵的余子式矩阵。

**参数**:

- `int row`: 行索引

- `int col`: 列索引

**返回值**: 余子式矩阵对象


### 行列式

```cpp
float Mat::determinant();
```

**描述**: 计算当前矩阵的行列式。

**参数**: void

**返回值**: 行列式的值


### 伴随矩阵

```cpp
Mat::adjoint();
```

**描述**: 计算当前矩阵的伴随矩阵。

**参数**: void

**返回值**: 伴随矩阵对象


### 归一化

```cpp
void Mat::normalize();
```

**描述**: 归一化当前矩阵的元素。

**参数**: void

**返回值**: void

### 范数

```cpp
float Mat::norm() const;
```

**描述**: 计算当前矩阵的范数。

**参数**: void

**返回值**: 范数的值

### 矩阵求逆 -- 基于伴随矩阵
```cpp
Mat::inverse_adjoint();
```

**描述**: 计算当前矩阵的逆矩阵，基于伴随矩阵法。

**参数**: void

**返回值**: 逆矩阵对象

### 单位矩阵

```cpp
Mat::eye(int size);
```

**描述**: 创建一个单位矩阵。

**参数**:

- `int size`: 矩阵的大小

**返回值**: 单位矩阵对象

### 增广矩阵

```cpp
Mat::augment(const Mat &A, const Mat &B);
```

**描述**: 创建一个增广矩阵，将两个矩阵连接在一起。

**参数**:

- `const Mat &A`: 第一个矩阵对象

- `const Mat &B`: 第二个矩阵对象

### 单位矩阵

```cpp
Mat::ones(int rows, int cols);
```

**描述**: 创建一个全为1的矩阵。

**参数**:

- `int rows`: 行数

- `int cols`: 列数

### 全1的矩阵

```cpp
Mat::ones(int size);
```

**描述**: 创建一个全为1的矩阵。

**参数**:

- `int size`: 矩阵的大小

### 高斯消元法

```cpp
Mat::gaussian_eliminate() const;
```

**描述**: 使用高斯消元法对当前矩阵进行变换。

**参数**: void

### 高斯消元法下行最简形式

```cpp
Mat::row_reduce_from_gaussian();
```

**描述**: 使用高斯消元法将当前矩阵转换为下行最简形式。

**参数**: void

**返回值**: 下行最简形式的矩阵对象

### 高斯-约旦消元法求逆

```cpp
Mat::inverse_gje();
```

**描述**: 使用高斯-约旦消元法计算当前矩阵的逆矩阵。

**参数**: void

**返回值**: 逆矩阵对象

### 点积

```cpp
float Mat::dotprod(const Mat &A, const Mat &B);
```

**描述**: 计算两个矩阵的点积。

**参数**:

- `const Mat &A`: 第一个矩阵对象

- `const Mat &B`: 第二个矩阵对象

**返回值**: 点积的值

### 解线性方程组

```cpp
Mat::solve(const Mat &A, const Mat &b);
```

**描述**: 解线性方程组 Ax = b。

**参数**:

- `const Mat &A`: 系数矩阵

- `const Mat &b`: 常数矩阵

**返回值**: 解矩阵 x


### 带状矩阵求解

```cpp
Mat::band_solve(Mat A, Mat b, int k);
```

**描述**: 解带状矩阵方程 Ax = b。

**参数**:

- `Mat A`: 带状矩阵

- `Mat b`: 常数矩阵

- `int k`: 带宽

**返回值**: 解矩阵 x

### 线性系统求根

```cpp
Mat::roots(Mat A, Mat y);
```

**描述**: 求解线性系统的根。

**参数**:

- `Mat A`: 系数矩阵

- `Mat y`: 常数矩阵

**返回值**: 根矩阵 x

## 线性代数

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

**描述**: 加法运算符，将两个矩阵相加。

**参数**:

- `const Mat &A`: 第一个矩阵对象

- `const Mat &B`: 第二个矩阵对象

### 加法运算符 - 常量

```cpp
Mat operator+(const Mat &A, float C);
```

**描述**: 加法运算符，将矩阵与常量相加。

**参数**:

- `const Mat &A`: 矩阵对象

- `float C`: 常量

### 减法运算符

```cpp
Mat operator-(const Mat &A, const Mat &B);
```

**描述**: 减法运算符，将第一个矩阵减去第二个矩阵。

**参数**:

- `const Mat &A`: 第一个矩阵对象

- `const Mat &B`: 第二个矩阵对象

### 减法运算符 - 常量

```cpp
Mat operator-(const Mat &A, float C);
```

**描述**: 减法运算符，将矩阵减去常量。

**参数**:

- `const Mat &A`: 矩阵对象

- `float C`: 常量   

### 乘法运算符

```cpp
Mat operator*(const Mat &A, const Mat &B);
```

**描述**: 乘法运算符，将两个矩阵相乘。

**参数**:

- `const Mat &A`: 第一个矩阵对象

- `const Mat &B`: 第二个矩阵对象

### 乘法运算符 - 常量

```cpp
Mat operator*(const Mat &A, float C);
```

**描述**: 乘法运算符，将矩阵与常量相乘。

**参数**:

- `const Mat &A`: 矩阵对象

- `float C`: 常量

### 乘法运算符 - 常量 - 反向

```cpp
Mat operator*(float C, const Mat &A);
```

**描述**: 乘法运算符，将常量与矩阵相乘。

**参数**:

- `float C`: 常量

- `const Mat &A`: 矩阵对象

### 除法运算符

```cpp
Mat operator/(const Mat &A, float C);
```

**描述**: 除法运算符，将矩阵除以常量。

**参数**:

- `const Mat &A`: 矩阵对象

- `float C`: 常量

### 除法运算符 - 矩阵

```cpp
Mat operator/(const Mat &A, const Mat &B);
```

**描述**: 除法运算符，将第一个矩阵除以第二个矩阵。

**参数**:

- `const Mat &A`: 第一个矩阵对象

- `const Mat &B`: 第二个矩阵对象

### 等于运算符

```cpp
bool operator==(const Mat &A, const Mat &B);
```

**描述**: 等于运算符，检查两个矩阵是否相等。

**参数**:

- `const Mat &A`: 第一个矩阵对象

- `const Mat &B`: 第二个矩阵对象

**返回值**: 布尔值，表示两个矩阵是否相等






