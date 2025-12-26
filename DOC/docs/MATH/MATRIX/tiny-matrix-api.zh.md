# 矩阵操作 - TINY_MATRIX

!!! INFO "TINY_MATRIX库"
    - 该库是一个轻量级的矩阵运算库，基于C++实现，提供了基本的矩阵操作和线性代数功能。
    - 该库的设计目标是提供简单易用的矩阵操作接口，适合于嵌入式系统和资源受限的环境。

!!! TIP "使用场景"
    相对于TINY_MAT库而言，TINY_MATRIX库提供了更丰富的功能和更高的灵活性，适合于需要进行复杂矩阵运算的应用场景。但是请注意，该库基于C++编写。

## 函数列表

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
        // ============================================================================
        // Matrix Metadata
        // ============================================================================
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

        // ============================================================================
        // Rectangular ROI Structure
        // ============================================================================
        /**
         * @name Region of Interest (ROI) Structure
         * @brief This is the structure for ROI
         */
        struct ROI
        {
            int pos_x;  ///< starting column index
            int pos_y;  ///< starting row index
            int width;  ///< width of ROI (columns)
            int height; ///< height of ROI (rows)

            ROI(int pos_x = 0, int pos_y = 0, int width = 0, int height = 0);
            void resize_roi(int pos_x, int pos_y, int width, int height);
            int area_roi(void) const;
        };
        
        // ============================================================================
        // Printing Functions
        // ============================================================================
        void print_info() const;
        void print_matrix(bool show_padding) const;

        // ============================================================================
        // Constructors & Destructor
        // ============================================================================
        /**
         * @brief Allocate memory for the matrix according to the memory required.
         * @note For ESP32, it will automatically determine if using RAM or PSRAM based on the size of the matrix.
         * @note This function sets ext_buff to false and allocates memory based on row * stride.
         *       If allocation fails or parameters are invalid, data will be set to nullptr.
         */
        void alloc_mem();
        
        /**
         * @brief Default constructor: create a 1x1 matrix with only a zero element.
         * @note If memory allocation fails, the object will be in an invalid state (data = nullptr).
         *       Caller should check the data pointer before using the matrix.
         */
        Mat();
        
        /**
         * @brief Constructor - create a matrix with the specified number of rows and columns.
         * @param rows Number of rows
         * @param cols Number of columns
         */
        Mat(int rows, int cols);
        
        /**
         * @brief Constructor - create a matrix with the specified number of rows, columns and stride.
         * @param rows Number of rows
         * @param cols Number of columns
         * @param stride Stride (number of elements in a row)
         */
        Mat(int rows, int cols, int stride);
        
        /**
         * @brief Constructor - create a matrix with the specified number of rows, columns and external data.
         * @param data Pointer to external data buffer
         * @param rows Number of rows
         * @param cols Number of columns
         */
        Mat(float *data, int rows, int cols);
        
        /**
         * @brief Constructor - create a matrix with the specified number of rows, columns and external data.
         * @param data Pointer to external data buffer
         * @param rows Number of rows
         * @param cols Number of columns
         * @param stride Stride (number of elements in a row)
         */
        Mat(float *data, int rows, int cols, int stride);
        
        /**
         * @brief Copy constructor - create a matrix with the same properties as the source matrix.
         * @param src Source matrix
         */
        Mat(const Mat &src);
        
        /**
         * @brief Destructor - free the memory allocated for the matrix.
         */
        ~Mat();

        // ============================================================================
        // Element Access
        // ============================================================================
        inline float &operator()(int row, int col) { return data[row * stride + col]; }
        inline const float &operator()(int row, int col) const { return data[row * stride + col]; }

        // ============================================================================
        // Data Manipulation
        // ============================================================================
        tiny_error_t copy_paste(const Mat &src, int row_pos, int col_pos);
        tiny_error_t copy_head(const Mat &src);
        Mat view_roi(int start_row, int start_col, int roi_rows, int roi_cols) const;
        Mat view_roi(const Mat::ROI &roi) const;
        Mat copy_roi(int start_row, int start_col, int height, int width);
        Mat copy_roi(const Mat::ROI &roi);
        Mat block(int start_row, int start_col, int block_rows, int block_cols);
        void swap_rows(int row1, int row2);
        void swap_cols(int col1, int col2);
        void clear(void);

        // ============================================================================
        // Arithmetic Operators
        // ============================================================================
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

        // ============================================================================
        // Linear Algebra - Basic Operations
        // ============================================================================
        Mat transpose();                   // Transpose matrix
        float determinant();               // Compute determinant (auto-selects method based on size)
        float determinant_laplace();        // Compute determinant using Laplace expansion (O(n!), for small matrices)
        float determinant_lu();            // Compute determinant using LU decomposition (O(n³), efficient for large matrices)
        float determinant_gaussian();      // Compute determinant using Gaussian elimination (O(n³), efficient for large matrices)
        Mat adjoint();                     // Compute adjoint matrix
        Mat inverse_adjoint();            // Compute inverse using adjoint method
        void normalize();                  // Normalize matrix
        float norm() const;                // Compute matrix norm
        float dotprod(const Mat &A, const Mat &B);  // Dot product

        // ============================================================================
        // Linear Algebra - Matrix Utilities
        // ============================================================================
        static Mat eye(int size);          // Create identity matrix
        static Mat ones(int rows, int cols);  // Create matrix filled with ones
        static Mat ones(int size);         // Create square matrix filled with ones
        static Mat augment(const Mat &A, const Mat &B);  // Horizontal concatenation [A | B]
        static Mat vstack(const Mat &A, const Mat &B);   // Vertical concatenation [A; B]
        
        /**
         * @brief Gram-Schmidt orthogonalization process
         * @note Orthogonalizes a set of vectors using the Gram-Schmidt process
         * @param vectors Input matrix where each column is a vector to be orthogonalized
         * @param orthogonal_vectors Output matrix for orthogonalized vectors (each column is orthogonal)
         * @param coefficients Output matrix for projection coefficients (R matrix in QR decomposition)
         * @param tolerance Minimum norm threshold for linear independence check
         * @return true if successful, false if input is invalid
         */
        static bool gram_schmidt_orthogonalize(const Mat &vectors, Mat &orthogonal_vectors, 
                                               Mat &coefficients, float tolerance = 1e-6f);

        // ============================================================================
        // Linear Algebra - Matrix Operations
        // ============================================================================
        Mat minor(int target_row, int target_col);       // Minor matrix (submatrix after removing row and col)
        Mat cofactor(int target_row, int target_col);    // Cofactor matrix
        Mat gaussian_eliminate() const;    // Gaussian elimination
        Mat row_reduce_from_gaussian();   // Row reduction from Gaussian form
        Mat inverse_gje();                 // Inverse using Gaussian-Jordan elimination

        // ============================================================================
        // Linear Algebra - Linear System Solving
        // ============================================================================
        Mat solve(const Mat &A, const Mat &b) const;  // Solve Ax = b using Gaussian elimination
        Mat band_solve(Mat A, Mat b, int k);          // Solve banded system
        Mat roots(Mat A, Mat y);                      // Alternative solve method

        // ============================================================================
        // Matrix Decomposition
        // ============================================================================
        // Forward declarations (structures defined after class)
        struct LUDecomposition;
        struct CholeskyDecomposition;
        struct QRDecomposition;
        struct SVDDecomposition;
        
        // Matrix property checks
        /**
         * @brief Check if the matrix is symmetric within a given tolerance.
         * @param tolerance Maximum allowed difference between A(i,j) and A(j,i) (must be >= 0)
         * @return true if matrix is symmetric, false otherwise
         */
        bool is_symmetric(float tolerance = 1e-6f) const;
        
        /**
         * @brief Check if matrix is positive definite using Sylvester's criterion.
         * @param tolerance Tolerance for numerical checks (must be >= 0)
         * @param max_minors_to_check Maximum number of leading principal minors to check.
         *                            - If -1: check all minors (complete Sylvester's criterion)
         *                            - If > 0: check first max_minors_to_check minors
         * @return true if matrix is positive definite, false otherwise
         */
        bool is_positive_definite(float tolerance = 1e-6f, int max_minors_to_check = -1) const;
        
        // Decomposition methods
        LUDecomposition lu_decompose(bool use_pivoting = true) const;
        CholeskyDecomposition cholesky_decompose() const;
        QRDecomposition qr_decompose() const;
        SVDDecomposition svd_decompose(int max_iter = 100, float tolerance = 1e-6f) const;
        
        // Solve using decomposition (more efficient for multiple RHS)
        static Mat solve_lu(const LUDecomposition &lu, const Mat &b);
        static Mat solve_cholesky(const CholeskyDecomposition &chol, const Mat &b);
        static Mat solve_qr(const QRDecomposition &qr, const Mat &b);  // Least squares solution
        
        // Pseudo-inverse using SVD (for rank-deficient or non-square matrices)
        static Mat pseudo_inverse(const SVDDecomposition &svd, float tolerance = 1e-6f);

        // ============================================================================
        // Eigenvalue & Eigenvector Decomposition
        // ============================================================================
        // Forward declarations (structures defined after class)
        struct EigenPair;
        struct EigenDecomposition;
        
        // Single eigenvalue methods (fast, for real-time applications)
        /**
         * @brief Compute the dominant (largest magnitude) eigenvalue and eigenvector using power iteration.
         * @param max_iter Maximum number of iterations (must be > 0)
         * @param tolerance Convergence tolerance (must be >= 0). Convergence when |λ_k - λ_{k-1}| < tolerance * |λ_k|
         * @return EigenPair containing the dominant eigenvalue, eigenvector, and status
         */
        EigenPair power_iteration(int max_iter = 1000, float tolerance = 1e-6f) const;
        
        /**
         * @brief Compute the smallest (minimum magnitude) eigenvalue and eigenvector using inverse power iteration.
         * @param max_iter Maximum number of iterations (must be > 0)
         * @param tolerance Convergence tolerance (must be >= 0). Convergence when |λ_k - λ_{k-1}| < tolerance * max(|λ_k|, 1)
         * @return EigenPair containing the smallest eigenvalue, eigenvector, and status
         * @note The matrix must be invertible (non-singular) for this method to work.
         */
        EigenPair inverse_power_iteration(int max_iter = 1000, float tolerance = 1e-6f) const;
        
        // Complete eigendecomposition methods
        /**
         * @brief Compute complete eigenvalue decomposition using Jacobi method for symmetric matrices.
         * @param tolerance Convergence tolerance (must be >= 0). Convergence when max off-diagonal < tolerance
         * @param max_iter Maximum number of iterations (must be > 0)
         * @return EigenDecomposition containing all eigenvalues, eigenvectors, and status
         * @note Best for symmetric matrices. Matrix should be symmetric for best results.
         */
        EigenDecomposition eigendecompose_jacobi(float tolerance = 1e-6f, int max_iter = 100) const;
        
        /**
         * @brief Compute complete eigenvalue decomposition using QR algorithm for general matrices.
         * @param max_iter Maximum number of QR iterations (must be > 0)
         * @param tolerance Convergence tolerance (must be >= 0). Convergence when subdiagonal < tolerance
         * @return EigenDecomposition containing eigenvalues, eigenvectors, and status
         * @note Supports non-symmetric matrices, but may have complex eigenvalues (only real part returned).
         */
        EigenDecomposition eigendecompose_qr(int max_iter = 100, float tolerance = 1e-6f) const;
        
        /**
         * @brief Automatic eigenvalue decomposition with method selection.
         * @param tolerance Convergence tolerance (must be >= 0)
         * @param max_iter Maximum number of iterations (must be > 0, default = 100)
         * @return EigenDecomposition containing eigenvalues, eigenvectors, and status
         * @note Automatically selects Jacobi method for symmetric matrices, QR algorithm for general matrices.
         */
        EigenDecomposition eigendecompose(float tolerance = 1e-6f, int max_iter = 100) const;

    protected:

    private:

    };

    // ============================================================================
    // Matrix Decomposition Structures
    // ============================================================================
    /**
     * @brief Structure to hold LU decomposition results
     * @note A = L * U, where L is lower triangular and U is upper triangular
     */
    struct Mat::LUDecomposition
    {
        Mat L;                 ///< Lower triangular matrix (with unit diagonal)
        Mat U;                 ///< Upper triangular matrix
        Mat P;                 ///< Permutation matrix (if pivoting used)
        bool pivoted;          ///< Whether pivoting was used
        tiny_error_t status;   ///< Computation status
        
        LUDecomposition();
    };
    
    /**
     * @brief Structure to hold Cholesky decomposition results
     * @note A = L * L^T, where L is lower triangular (for symmetric positive definite matrices)
     */
    struct Mat::CholeskyDecomposition
    {
        Mat L;                 ///< Lower triangular matrix
        tiny_error_t status;   ///< Computation status
        
        CholeskyDecomposition();
    };
    
    /**
     * @brief Structure to hold QR decomposition results
     * @note A = Q * R, where Q is orthogonal and R is upper triangular
     */
    struct Mat::QRDecomposition
    {
        Mat Q;                 ///< Orthogonal matrix (Q^T * Q = I)
        Mat R;                 ///< Upper triangular matrix
        tiny_error_t status;   ///< Computation status
        
        QRDecomposition();
    };
    
    /**
     * @brief Structure to hold SVD decomposition results
     * @note A = U * S * V^T, where U and V are orthogonal, S is diagonal (singular values)
     */
    struct Mat::SVDDecomposition
    {
        Mat U;                 ///< Left singular vectors (orthogonal matrix)
        Mat S;                 ///< Singular values (diagonal matrix or vector)
        Mat V;                 ///< Right singular vectors (orthogonal matrix, V^T)
        int rank;              ///< Numerical rank of the matrix
        int iterations;        ///< Number of iterations performed
        tiny_error_t status;   ///< Computation status
        
        SVDDecomposition();
    };
    
    // ============================================================================
    // Eigenvalue & Eigenvector Decomposition Structures
    // ============================================================================
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

    // ============================================================================
    // Stream Operators
    // ============================================================================
    std::ostream &operator<<(std::ostream &os, const Mat &m);
    std::ostream &operator<<(std::ostream &os, const Mat::ROI &roi);
    std::istream &operator>>(std::istream &is, Mat &m);

    // ============================================================================
    // Global Arithmetic Operators
    // ============================================================================
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

## 矩阵元数据

!!! INFO "矩阵结构"
    Mat类使用行主序存储布局，支持填充和步幅。这种设计实现了高效的内存访问模式，并与DSP库兼容。

### 核心维度

- **`int row`** : 矩阵的行数。

- **`int col`** : 矩阵的列数。

- **`int element`** : 元素总数 = 行数 × 列数。

### 内存布局

- **`int stride`** : 步幅 = (每行元素数) + 填充数。步幅决定了在内存中移动到下一行需要跳过的元素数。

- **`int pad`** : 两行之间的填充元素数。填充用于内存对齐和DSP优化。

- **`int memory`** : 数据缓冲区大小 = 行数 × 步幅（以float元素为单位）。

### 数据指针

- **`float *data`** : 指向包含矩阵元素的数据缓冲区的指针。元素按行主序存储：位置(i, j)的元素位于 `data[i * stride + j]`。

- **`float *temp`** : 指向临时数据缓冲区的指针（如果已分配）。某些操作内部使用。

### 内存管理标志

- **`bool ext_buff`** : 标志矩阵是否使用外部缓冲区。当为`true`时，析构函数不会释放内存（调用者负责管理）。

- **`bool sub_matrix`** : 标志矩阵是否为另一个矩阵的子集/视图。当为`true`时，矩阵与父矩阵共享数据。

!!! TIP "内存布局示例"
    对于3×4矩阵，步幅=4（无填充）：
    ```
    [a b c d]   行 0: data[0*4+0] 到 data[0*4+3]
    [e f g h]   行 1: data[1*4+0] 到 data[1*4+3]
    [i j k l]   行 2: data[2*4+0] 到 data[2*4+3]
    ```
    
    对于3×4矩阵，步幅=6（填充=2）：
    ```
    [a b c d _ _]   行 0: data[0*6+0] 到 data[0*6+3]，填充在 data[0*6+4,5]
    [e f g h _ _]   行 1: data[1*6+0] 到 data[1*6+3]，填充在 data[1*6+4,5]
    [i j k l _ _]   行 2: data[2*6+0] 到 data[2*6+3]，填充在 data[2*6+4,5]
    ```

## ROI 结构

!!! INFO "感兴趣区域"
    ROI（感兴趣区域）结构表示矩阵的矩形子区域。它与`view_roi()`和`copy_roi()`函数一起使用，以高效地提取或引用子矩阵。

### ROI 元数据

- **`int pos_x`** : 起始列索引（左上角的x坐标）。

- **`int pos_y`** : 起始行索引（左上角的y坐标）。

- **`int width`** : ROI的宽度（列数）。

- **`int height`** : ROI的高度（行数）。

### ROI 构造函数

```cpp
Mat::ROI::ROI(int pos_x = 0, int pos_y = 0, int width = 0, int height = 0);
```

**描述**: 

构造一个 ROI 对象，默认值为 (0, 0, 0, 0)。

**参数**:

- `int pos_x`: 起始列索引

- `int pos_y`: 起始行索引

- `int width`: ROI 的宽度（列数）

- `int height`: ROI 的高度（行数）

### ROI 重置函数

```cpp
void Mat::ROI::resize_roi(int pos_x, int pos_y, int width, int height);
```

**描述**: 

重置 ROI 的位置和大小。

**参数**:

- `int pos_x`: 起始列索引

- `int pos_y`: 起始行索引

- `int width`: ROI 的宽度（列数）

- `int height`: ROI 的高度（行数）

**返回值**:

void

### ROI 面积函数

```cpp
int Mat::ROI::area_roi(void) const;
```

**描述**: 

计算 ROI 的面积。

**参数**:

void

**返回值**:

整数类型 ROI 的面积

## 打印函数

!!! TIP "调试工具"
    这些函数对于调试和理解矩阵状态至关重要。使用它们来验证矩阵维度、内存布局和数据值。

### 打印矩阵信息

```cpp
void Mat::print_info() const;
```

**描述**: 

打印全面的矩阵信息，包括：

- 维度：行数、列数、元素数

- 内存布局：填充数、步幅、内存大小

- 指针：数据缓冲区地址、临时缓冲区地址

- 标志：外部缓冲区使用情况、子矩阵状态

- 警告：维度不匹配、无效状态

**参数**:

void

**返回值**:

void

**使用建议**:

- **调试**: 对于验证矩阵状态和检测内存问题至关重要。

- **内存分析**: 显示实际内存使用情况与逻辑大小的对比，帮助识别内存效率问题。

- **子矩阵检测**: 清楚地指示矩阵是否为视图，这会影响内存管理。

### 打印矩阵元素

```cpp
   void Mat::print_matrix(bool show_padding) const;
```

**描述**: 

以格式化表格形式打印矩阵元素。可选择显示由视觉分隔符分隔的填充元素。

**参数**:

- `bool show_padding` : 如果为`true`，显示填充值并用分隔符`|`分隔。如果为`false`，仅显示实际矩阵元素。

**返回值**:

void

**使用建议**:

- **格式化**: 元素以固定宽度（12个字符）格式化以保持对齐。

- **填充可视化**: `show_padding`选项有助于理解内存布局和验证填充值。

- **大矩阵**: 对于非常大的矩阵，考虑使用`view_roi()`打印特定区域。

## 构造与析构函数

!!! INFO "内存管理"
    构造函数自动处理内存分配。析构函数仅在内存是内部分配的情况下安全释放内存（不包括外部缓冲区或视图）。构造后始终检查`data`指针以确保分配成功。

### 内存分配

```cpp
void Mat::alloc_mem();
```

**描述**: 

根据计算的内存需求为矩阵分配内存的内部函数。设置`ext_buff = false`并分配`row * stride`个float元素。

**参数**:

void

**返回值**:

void

**使用建议**:

- **自动调用**: 由构造函数自动调用。很少需要手动调用。

- **内存计算**: 分配`row * stride`个元素，可能包括填充。

- **错误处理**: 如果分配失败，`data`保持为`nullptr`。构造后始终检查`data`。

### 默认构造函数

```cpp
Mat::Mat();
```

**描述**: 

默认构造函数创建一个1×1的零矩阵。这对于初始化和作为错误情况的返回值很有用。

**数学原理**:

在某些上下文中创建矩阵运算的恒等元素，尽管通常您会希望指定维度。

**参数**:

void

**返回值**:

Mat - 一个元素为0的1×1矩阵。

### 构造函数 - Mat(int rows, int cols)

```cpp
Mat::Mat(int rows, int cols);
```

**描述**: 

构造函数创建具有指定维度的矩阵。所有元素初始化为零。这是最常用的构造函数。

**参数**:

- `int rows` : 行数（必须 > 0）。

- `int cols` : 列数（必须 > 0）。

**返回值**:

Mat - 一个rows×cols的矩阵，所有元素初始化为0。

**使用建议**:

- **零初始化**: 所有元素使用`memset`设置为零，确保干净的状态。

- **内存布局**: 创建连续的内存布局，无填充（stride = cols）。

- **错误处理**: 如果内存分配失败，`data`将为`nullptr`。始终验证分配成功。

### 构造函数 - Mat(int rows, int cols, int stride)

```cpp
Mat::Mat(int rows, int cols, int stride);
```

**描述**: 

构造函数创建具有指定维度、列数和步幅的矩阵。当需要填充以进行内存对齐或DSP优化时很有用。

**参数**:

- `int rows` : 行数。

- `int cols` : 列数。

- `int stride` : 步幅（必须 ≥ cols）。填充 = stride - cols。

**返回值**:

Mat - 一个具有步幅的rows×cols矩阵，所有元素初始化为0。

**使用建议**:

- **DSP优化**: 某些DSP库需要对齐的内存。使用步幅确保适当的对齐。

- **内存效率**: 填充允许在对齐边界上进行高效的向量化操作。

- **兼容性**: 实现与使用步幅内存布局的外部库的兼容性。

### 构造函数 - Mat(float *data, int rows, int cols)

```cpp
Mat::Mat(float *data, int rows, int cols);
```

**描述**: 

构造函数在外部数据缓冲区上创建矩阵视图。矩阵不拥有内存；调用者负责管理它。对于与现有数据数组接口很有用。

**参数**:

- `float *data` : 指向外部数据缓冲区的指针（在矩阵生命周期内必须保持有效）。

- `int rows` : 行数。

- `int cols` : 列数。

**返回值**:

Mat - 一个`ext_buff = true`的矩阵视图。

**使用建议**:

- **零拷贝**: 不发生内存拷贝；矩阵直接引用外部数据。

- **生命周期管理**: 外部缓冲区在矩阵存在期间必须保持有效。析构函数不会释放此内存。

- **数据布局**: 假设行主序布局，无填充（stride = cols）。

- **使用场景**: 

  - 包装C数组

  - 与其他库接口

  - 避免不必要的拷贝

### 构造函数 - Mat(float *data, int rows, int cols, int stride)

```cpp
Mat::Mat(float *data, int rows, int cols, int stride);
```

**描述**: 

构造函数在具有指定步幅的外部数据缓冲区上创建矩阵视图。支持步幅内存布局以实现DSP兼容性。

**参数**:

- `float *data` : 指向外部数据缓冲区的指针（在矩阵生命周期内必须保持有效）。

- `int rows` : 行数。

- `int cols` : 列数。

- `int stride` : 步幅（必须 ≥ cols）。

**返回值**:

Mat - 一个`ext_buff = true`并具有指定步幅的矩阵视图。

**使用建议**:

- **步幅布局**: 对于使用步幅内存布局的DSP库至关重要。

- **内存安全**: 与上一个构造函数相同的生命周期要求 - 外部缓冲区必须保持有效。

- **填充支持**: 可以处理行间有填充的缓冲区。

### 拷贝构造函数 - Mat(const Mat &src)

```cpp
Mat::Mat(const Mat &src);
```

**描述**: 

拷贝构造函数从源矩阵创建新矩阵。使用智能拷贝策略：常规矩阵进行深拷贝，子矩阵视图进行浅拷贝。

**拷贝策略**:

- **常规矩阵**: 深拷贝 - 分配新内存并拷贝所有数据

- **子矩阵视图**: 浅拷贝 - 与源共享数据（创建另一个视图）

**参数**:

- `const Mat &src` : 源矩阵。

**返回值**:

Mat - 根据源类型具有拷贝或共享数据的新矩阵。

**使用建议**:

- **自动选择**: 根据源矩阵类型自动选择深拷贝或浅拷贝。

- **内存效率**: 子矩阵视图进行浅拷贝以避免不必要的内存分配。

- **独立性**: 深拷贝是独立的；修改不会影响源。

### 析构函数

```cpp
Mat::~Mat();
```

**描述**: 

析构函数安全地释放分配的内存。仅释放内部分配的内存（`ext_buff = false`）。外部缓冲区和视图不会被释放。

**内存管理**:

- 如果`ext_buff = false`，释放`data`缓冲区

- 如果已分配，释放`temp`缓冲区

- 对外部缓冲区或视图不执行任何操作

**参数**:

void

**返回值**:

void

!!! note "构造函数和析构函数规则"
    - 构造函数函数必须与类名相同且无返回类型
    - C++允许通过更改参数数量/顺序进行函数重载
    - 当对象超出作用域时，析构函数会自动调用
    - 构造后始终检查`data != nullptr`以验证分配成功


## 元素访问

!!! INFO "矩阵索引"
    Mat类使用运算符重载提供直观的矩阵元素访问。`operator()`允许使用`A(i, j)`这样的自然语法，而不是`A.data[i * stride + j]`。实现自动处理步幅和填充。

### 访问矩阵元素（非常量）

```cpp
inline float &operator()(int row, int col);
```

**描述**:

以读写方式访问矩阵元素。返回元素的引用，允许读取和修改。

**数学原理**:

位置(row, col)的元素访问为`data[row * stride + col]`，其中步幅考虑了填充。

**参数**：

- `int row` : 行索引（基于0，必须在范围[0, row-1]内）。

- `int col` : 列索引（基于0，必须在范围[0, col-1]内）。

**返回值**:

`float&` - 矩阵元素的引用，允许修改。

**使用建议**:

- **边界检查**: 为了性能，不进行自动边界检查。确保索引有效。

- **步幅处理**: 自动考虑步幅，因此可以正确处理带填充的矩阵。

- **性能**: 内联函数，开销最小，适用于紧密循环。

- **示例**: `A(2, 3) = 5.0f;` 将第2行第3列的元素设置为5.0。

### 访问矩阵元素（常量）

```cpp
inline const float &operator()(int row, int col) const;
```

**描述**:

以只读方式访问矩阵元素。返回常量引用，防止修改。当矩阵为const时使用。

**参数**：

- `int row` : 行索引（基于0）。

- `int col` : 列索引（基于0）。

**返回值**:

`const float&` - 矩阵元素的常量引用（只读）。

**使用建议**:

- **常量正确性**: 实现正确的const-correct代码。在const成员函数中使用此版本。

- **安全性**: 防止意外修改const矩阵。

!!! note "运算符重载"
    这些函数重载了`()`运算符，实现了自然的矩阵索引语法：
    ```cpp
    Mat A(3, 4);
    A(1, 2) = 3.14f;        // 写访问
    float val = A(1, 2);    // 读访问
    const Mat& B = A;
    float val2 = B(1, 2);   // 只读访问（使用const版本）
    ```

## 数据操作

### 复制其他矩阵到当前矩阵

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

### 复制矩阵头部

```cpp
tiny_error_t copy_head(const Mat &src);
```

**描述**:

将源矩阵的头部信息复制到当前矩阵。

**参数**:

- `const Mat &src`: 源矩阵对象

**返回值**:

错误代码

### 获取子矩阵视图

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

### 获取子矩阵视图 - 使用 ROI 结构

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


### 获取子矩阵副本

```cpp
Mat copy_roi(int start_row, int start_col, int roi_rows, int roi_cols);
```

**描述**:

获取当前矩阵的子矩阵副本。

**参数**:

- `int start_row`: 起始行索引

- `int start_col`: 起始列索引

- `int roi_rows`: 子矩阵的行数

- `int roi_cols`: 子矩阵的列数

**返回值**:

子矩阵对象

### 获取子矩阵副本 - 使用 ROI 结构

```cpp
Mat copy_roi(const Mat::ROI &roi);
```

**描述**:

获取当前矩阵的子矩阵副本，使用 ROI 结构。

**参数**:

- `const Mat::ROI &roi`: ROI 结构对象

**返回值**:

子矩阵对象

### 获取矩阵块

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
### 交换行

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

### 交换列

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

### 清除矩阵

```cpp
void Mat::clear(void);
```

**描述**:

通过将所有元素设置为零来清除矩阵。

**参数**:

void

**返回值**:

void

## 算术运算符

!!! INFO "就地操作"
    本节定义了作用于当前矩阵本身的算术运算符（就地操作）。这些运算符修改矩阵并返回其引用，支持链式操作如`A += B += C`。这些运算符经过优化以处理填充，并在可用时使用DSP加速函数。

### 拷贝赋值

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

### 加法运算符

```cpp
Mat &operator+=(const Mat &A);
```

**描述**:

加法运算符，将源矩阵的元素加到当前矩阵。

**参数**:

- `const Mat &A`: 源矩阵对象

### 加法运算符 - 常量

```cpp
Mat &operator+=(float C);
```

**描述**:

将常量按元素加到当前矩阵。

**参数**:

- `float C`: 要加的常量

**返回值**:

Mat& - 当前矩阵的引用

### 减法运算符

```cpp
Mat &operator-=(const Mat &A);
```

**描述**:

从当前矩阵中按元素减去源矩阵。

**参数**:

- `const Mat &A`: 源矩阵对象

**返回值**:

Mat& - 当前矩阵的引用

### 减法运算符 - 常量

```cpp
Mat &operator-=(float C);
```

**描述**:

从当前矩阵中按元素减去常量。

**参数**:

- `float C`: 要减的常量

**返回值**:

Mat& - 当前矩阵的引用

### 矩阵乘法

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

### 乘法运算符 - 常量

```cpp
Mat &operator*=(float C);
```

**描述**:

按元素乘以常量。

**参数**:

- `float C`: 常量乘数

**返回值**:

Mat& - 当前矩阵的引用

### 除法运算符

```cpp
Mat &operator/=(const Mat &B);
```

**描述**:

按元素除法：this = this / B

**参数**:

- `const Mat &B`: 除数矩阵

**返回值**:

Mat& - 当前矩阵的引用

### 除法运算符 - 常量

```cpp
Mat &operator/=(float C);
```

**描述**:

将当前矩阵按元素除以常量。

**参数**:

- `float C`: 常量除数

**返回值**:

Mat& - 当前矩阵的引用

### 幂运算符

```cpp
Mat operator^(int C);
```

**描述**:

按元素整数幂运算。返回一个新矩阵，其中每个元素都提升到给定幂次。

**参数**:

- `int C`: 指数（整数）

**返回值**:

Mat - 幂运算后的新矩阵

## 线性代数

### 转置矩阵

```cpp
Mat Mat::transpose();
```

**描述**:

计算矩阵的转置，返回新矩阵。矩阵A的转置A^T 通过交换行和列得到：(A^T)ᵢⱼ = Aⱼᵢ。

**数学原理**: 

- 对于任何矩阵A，(A^T) ^T = A

- (A + B)^T = A^T + B^T

- (AB)^T = B^T * A^T

- 对于方阵，det(A) = det(A^T)

**参数**:

无。

**返回值**:

Mat - 转置后的矩阵 (col × row)。

**使用建议**:

- **内存布局**: 创建新矩阵，因此内存使用量暂时翻倍。对于大矩阵，考虑内存限制。

- **对称矩阵**: 如果A = A^T，矩阵是对称的。使用`is_symmetric()`检查。

- **应用**: 

  - 内积: u^T * v

  - 二次型: x^T * A * x

  - 矩阵方程: A^T * A（正规方程）

### 余子式矩阵

```cpp
Mat Mat::minor(int row, int col);
```

**描述**:

通过移除指定的行和列来计算余子式矩阵。余子式是移除一行一列后得到的子矩阵。

**参数**:

- `int row`: 要移除的行索引

- `int col`: 要移除的列索引

**返回值**:

Mat - (n-1)x(n-1) 的余子式矩阵

### 代数余子式矩阵

```cpp
Mat Mat::cofactor(int row, int col);
```

**描述**:

计算代数余子式矩阵（与余子式矩阵相同）。代数余子式矩阵与余子式矩阵相同。符号 (-1)^(i+j) 在计算代数余子式值时应用，而不是应用到矩阵元素本身。

**参数**:

- `int row`: 要移除的行索引

- `int col`: 要移除的列索引

**返回值**:

Mat - (n-1)x(n-1) 的代数余子式矩阵（与余子式矩阵相同）


### 行列式（自动选择方法）

```cpp
float Mat::determinant();
```

**描述**:

计算方阵的行列式，根据矩阵大小自动选择最优方法。对于小矩阵（n ≤ 4），使用拉普拉斯展开法；对于大矩阵（n > 4），使用LU分解法以提高效率。

**数学原理**: 

行列式是方阵的一个重要数值特征，具有以下性质：

- det(AB) = det(A) * det(B)

- det(A^T) = det(A)

- det(A^(-1)) = 1 / det(A)

- 如果矩阵是奇异的，det(A) = 0

**方法选择**:

- **小矩阵 (n ≤ 4)**: 使用 `determinant_laplace()` - 拉普拉斯展开法，时间复杂度 O(n!)，对小矩阵更准确

- **大矩阵 (n > 4)**: 使用 `determinant_lu()` - LU分解法，时间复杂度 O(n³)，效率更高

**参数**:

void

**返回值**:

float - 行列式的值

**使用建议**:

- **自动选择**: 对于大多数应用，直接使用 `determinant()` 即可，函数会自动选择最优方法

- **性能优化**: 如果需要多次计算相同大小的矩阵行列式，可以考虑直接调用 `determinant_lu()` 或 `determinant_gaussian()`

- **精度要求**: 对于小矩阵，`determinant_laplace()` 可能提供更好的数值精度

### 行列式 - 拉普拉斯展开法

```cpp
float Mat::determinant_laplace();
```

**描述**:

使用拉普拉斯展开（余子式展开）计算方阵的行列式。时间复杂度为 O(n!)，仅适用于小矩阵（n ≤ 4）。

**数学原理**: 

拉普拉斯展开是行列式的递归定义方法：

- 对于 1×1 矩阵: det([a]) = a

- 对于 2×2 矩阵: det([[a,b],[c,d]]) = ad - bc

- 对于 n×n 矩阵: det(A) = Σⱼ₌₁ⁿ (-1)ⁱ⁺ʲ aᵢⱼ * det(Mᵢⱼ)，其中 Mᵢⱼ 是余子式矩阵

本实现使用第一行展开，递归计算余子式的行列式。

**参数**:

void

**返回值**:

float - 行列式的值

!!! warning "性能警告"
    时间复杂度为 O(n!)，仅适用于小矩阵（n ≤ 4）。对于大矩阵，请使用 `determinant_lu()` 或 `determinant_gaussian()`。

### 行列式 - LU分解法

```cpp
float Mat::determinant_lu();
```

**描述**:

使用LU分解计算方阵的行列式。时间复杂度为 O(n³)，适用于大矩阵。

**数学原理**: 

LU分解将矩阵分解为 A = P * L * U，其中：

- P 是置换矩阵（如果使用主元）

- L 是单位对角线的下三角矩阵

- U 是上三角矩阵

行列式计算公式：det(A) = det(P) * det(L) * det(U)

其中：

- det(P) = (-1)^(置换的符号)，由行交换次数决定

- det(L) = 1（因为L是单位对角线的下三角矩阵）

- det(U) = ∏ᵢ Uᵢᵢ（U的对角线元素的乘积）

**算法步骤**:

1. 执行LU分解（带主元以提高数值稳定性）
2. 计算置换矩阵的行列式 det(P)
3. 计算上三角矩阵U的对角线元素乘积 det(U)
4. 返回 det(P) * det(U)

**参数**:

void

**返回值**:

float - 行列式的值。如果矩阵是奇异的或接近奇异的，返回 0.0

**使用建议**:

- **效率**: 对于 n > 4 的矩阵，比拉普拉斯展开法快得多

- **数值稳定性**: 使用主元（pivoting）提高数值稳定性

- **奇异矩阵**: 如果矩阵是奇异的，LU分解会失败，函数返回 0.0

### 行列式 - 高斯消元法

```cpp
float Mat::determinant_gaussian();
```

**描述**:

使用高斯消元法计算方阵的行列式。时间复杂度为 O(n³)，适用于大矩阵。

**数学原理**: 

高斯消元法将矩阵转换为上三角形式，然后计算对角线元素的乘积。行列式的值等于上三角矩阵对角线元素的乘积，并根据行交换次数调整符号。

**算法步骤**:

1. 使用部分主元法进行高斯消元，将矩阵转换为上三角形式
2. 跟踪行交换次数
3. 计算上三角矩阵对角线元素的乘积
4. 根据行交换次数调整符号：每次行交换使行列式乘以 -1

**参数**:

void

**返回值**:

float - 行列式的值。如果矩阵是奇异的，返回 0.0

**使用建议**:

- **效率**: 对于大矩阵，时间复杂度为 O(n³)，与LU分解法相当

- **数值稳定性**: 使用部分主元法提高数值稳定性

- **实现简单**: 相比LU分解，实现更直观，但功能较少（不能用于求解线性系统）


### 伴随矩阵

```cpp
Mat Mat::adjoint();
```

**描述**:

计算方阵的伴随（或伴随转置）矩阵。

**参数**:

void

**返回值**:

Mat - 伴随矩阵对象


### 归一化

```cpp
void Mat::normalize();
```

**描述**:

使用L2范数（Frobenius范数）归一化矩阵。归一化后，||Matrix|| = 1。

**参数**:

void

**返回值**:

void

### 范数

```cpp
float Mat::norm() const;
```

**描述**:

计算矩阵的Frobenius范数（L2范数）。

**参数**:

void

**返回值**:

float - 计算得到的矩阵范数

### 矩阵求逆 -- 基于伴随矩阵
```cpp
Mat Mat::inverse_adjoint();
```

**描述**:

使用伴随矩阵法计算方阵的逆矩阵。如果矩阵是奇异的，返回零矩阵。

**参数**:

void

**返回值**:

Mat - 逆矩阵对象。如果矩阵是奇异的，返回零矩阵

### 单位矩阵

```cpp
static Mat Mat::eye(int size);
```

**描述**:

生成指定大小的单位矩阵。

**参数**:

- `int size`: 方阵的维度

**返回值**:

Mat - 单位矩阵 (size x size)

### 增广矩阵（水平连接）

```cpp
static Mat Mat::augment(const Mat &A, const Mat &B);
```

**描述**:

通过水平连接两个矩阵创建增广矩阵 [A | B]。A和B的行数必须匹配。

**参数**:

- `const Mat &A`: 左侧矩阵

- `const Mat &B`: 右侧矩阵

**返回值**:

Mat - 增广矩阵 [A B]

### 垂直堆叠

```cpp
static Mat Mat::vstack(const Mat &A, const Mat &B);
```

**描述**:

垂直堆叠两个矩阵 [A; B]。A和B的列数必须匹配。

**参数**:

- `const Mat &A`: 顶部矩阵

- `const Mat &B`: 底部矩阵

**返回值**:

Mat - 垂直堆叠的矩阵 [A; B]

### Gram-Schmidt正交化

```cpp
static bool Mat::gram_schmidt_orthogonalize(const Mat &vectors, Mat &orthogonal_vectors, 
                                            Mat &coefficients, float tolerance = 1e-6f);
```

**描述**:

使用Gram-Schmidt过程对一组向量进行正交化。这是一个通用的正交化函数，可重复用于QR分解和其他需要正交基的应用。使用改进的Gram-Schmidt算法和重新正交化以提高数值稳定性。

**数学原理**:

给定一组向量 {v₁, v₂, ..., vₙ}，Gram-Schmidt过程产生正交集合 {q₁, q₂, ..., qₙ}，其中：

- q₁ = v₁ / ||v₁||

- qⱼ = (vⱼ - Σᵢ₌₁ʲ⁻¹⟨vⱼ, qᵢ⟩qᵢ) / ||vⱼ - Σᵢ₌₁ʲ⁻¹⟨vⱼ, qᵢ⟩qᵢ||

改进版本立即减去投影，这提高了数值稳定性。

**参数**:

- `const Mat &vectors` : 输入矩阵，其中每列是要正交化的向量 (m × n)。

- `Mat &orthogonal_vectors` : 输出矩阵，用于正交化向量 (m × n)，每列都是正交且归一化的。

- `Mat &coefficients` : 输出矩阵，用于投影系数 (n × n，上三角)，类似于QR分解中的R矩阵。

- `float tolerance` : 线性独立性检查的最小范数阈值（默认：1e-6）。

**返回值**:

`bool` - 成功返回`true`，输入无效返回`false`。

**使用建议**:

- **数值稳定性**: 实现使用改进的Gram-Schmidt和重新正交化，这显著提高了近线性相关向量的稳定性。

- **QR分解**: 此函数由`qr_decompose()`内部使用。对于QR分解，系数矩阵对应于R矩阵。

- **基构造**: 用于从一组向量构造正交基，这在许多线性代数应用中都是基础。

- **性能**: 对于大矩阵，考虑计算成本。对于m维向量和n个向量，复杂度为O(mn²)。

### 全1矩阵（矩形）

```cpp
static Mat Mat::ones(int rows, int cols);
```

**描述**:

创建指定大小的全1矩阵。

**参数**:

- `int rows`: 行数

- `int cols`: 列数

**返回值**:

Mat - 矩阵 [rows x cols]，所有元素 = 1

### 全1矩阵（方阵）

```cpp
static Mat Mat::ones(int size);
```

**描述**:

创建指定大小的方阵，所有元素为1。

**参数**:

- `int size`: 方阵的大小（行数 = 列数）

**返回值**:

Mat - 方阵 [size x size]，所有元素 = 1

### 高斯消元法

```cpp
Mat Mat::gaussian_eliminate() const;
```

**描述**:

执行高斯消元法，将矩阵转换为行阶梯形式（REF）。这是求解线性系统和计算矩阵秩的第一步。

**数学原理**:

高斯消元通过初等行变换将矩阵转换为行阶梯形式：

1. **行交换**: 交换两行

2. **行缩放**: 将一行乘以非零标量

3. **行加法**: 将一行的倍数加到另一行

**行阶梯形式（REF）的性质**:

- 所有零行在底部

- 每个非零行的前导系数（主元）位于上一行主元的右侧

- 主元下方的所有条目为零

**参数**:

无。

**返回值**:

Mat - 上三角矩阵（REF形式）。

**使用建议**:

- **线性系统求解**: 求解Ax = b的第一步。REF之后，使用回代。

- **秩计算**: 秩等于REF中非零行的数量。

- **行列式**: 可以从REF计算行列式（对角元素的乘积，根据行交换进行调整）。

- **数值稳定性**: 实现使用部分主元以提高数值稳定性。

- **性能**: 对于n×n矩阵为O(n³)。对于多个系统，优先使用LU分解。

### 从高斯消元到行最简形式

```cpp
Mat Mat::row_reduce_from_gaussian();
```

**描述**:

将矩阵（假设已为行阶梯形式）转换为简化行阶梯形式（RREF）。

**参数**:

void

**返回值**:

Mat - RREF形式的矩阵

### 高斯-约旦消元法求逆

```cpp
Mat Mat::inverse_gje();
```

**描述**:

使用高斯-约旦消元法计算方阵的逆矩阵。

**参数**:

void

**返回值**:

Mat - 如果矩阵可逆则返回逆矩阵，否则返回空矩阵

### 点积

```cpp
float Mat::dotprod(const Mat &A, const Mat &B);
```

**描述**:

计算两个向量（Nx1）的点积。

**参数**:

- `const Mat &A`: 输入向量 A (Nx1)

- `const Mat &B`: 输入向量 B (Nx1)

**返回值**:

float - 计算得到的点积值

### 解线性方程组

```cpp
Mat Mat::solve(const Mat &A, const Mat &b) const;
```

**描述**:

使用高斯消元法和回代求解线性系统Ax = b。这是适用于良条件系统的直接方法。

**数学原理**:

该方法包括两个阶段：

1. **前向消元**: 将增广矩阵[A|b]转换为上三角形式

2. **回代**: 从下到上求解Ux = y

**算法**:

- 创建增广矩阵 [A | b]

- 应用高斯消元得到 [U | y]，其中U是上三角矩阵

- 使用回代求解Ux = y：xᵢ = (yᵢ - Σⱼ₌ᵢ₊₁ⁿ Uᵢⱼxⱼ) / Uᵢᵢ

**参数**:

- `const Mat &A` : 系数矩阵 (N×N)，必须是方阵且非奇异。

- `const Mat &b` : 右端项向量 (N×1)。

**返回值**:

Mat - 解向量 (N×1)，包含方程Ax = b的根。如果系统是奇异的或不兼容的，返回空矩阵。

**使用建议**:

- **单个系统**: 对于求解一个系统很高效。对于具有相同A的多个系统，使用LU分解 + `solve_lu()`。

- **条件数**: 对于病态矩阵，性能会下降。如果结果不准确，检查条件数。

- **奇异系统**: 如果A是奇异的（det(A) = 0），返回空矩阵。对于秩亏系统，使用SVD + 伪逆。

- **性能**: 消元为O(n³)，回代为O(n²)。总计O(n³)。

- **替代方法**:

  - 对于SPD矩阵：使用Cholesky分解 + `solve_cholesky()`（更快）

  - 对于多个右端项：使用LU分解 + `solve_lu()`（更高效）

  - 对于超定系统：使用QR分解 + `solve_qr()`（最小二乘）

### 带状矩阵求解

```cpp
Mat Mat::band_solve(Mat A, Mat b, int k);
```

**描述**:

使用优化的高斯消元法求解带状矩阵方程组 Ax = b。

**参数**:

- `Mat A`: 系数矩阵 (NxN) - 带状矩阵

- `Mat b`: 结果向量 (Nx1)

- `int k`: 矩阵的带宽（非零带的宽度）

**返回值**:

Mat - 解向量 (Nx1)，包含方程 Ax = b 的根

### 线性系统求根

```cpp
Mat Mat::roots(Mat A, Mat y);
```

**描述**:

使用不同方法求解矩阵。这是 'solve' 函数的另一种实现，原理上没有区别。此方法使用高斯消元法求解线性系统 A * x = y。

**参数**:

- `Mat A`: 矩阵 [N]x[N]，包含输入系数

- `Mat y`: 向量 [N]x[1]，包含结果值

**返回值**:

Mat - 矩阵 [N]x[1]，包含根

## 矩阵属性与分解

!!! INFO "矩阵分解概述"
    矩阵分解是数值线性代数中的基本工具。它们将矩阵分解为更简单的组件，揭示其结构并实现高效计算。不同的分解适用于不同类型的矩阵和应用。

### 矩阵属性检查

#### 检查对称性

```cpp
bool Mat::is_symmetric(float tolerance = 1e-6f) const;
```

**描述**:

检查矩阵在给定容差内是否对称。矩阵A是对称的，如果A = A^T，即对于所有i, j，A(i,j) = A(j,i)。

**数学原理**:

对于对称矩阵，所有特征值都是实数，并且可以选择特征向量使其正交。对称矩阵在许多应用中都是基础的，特别是在结构动力学和优化中。

**参数**:

- `float tolerance` : 允许的最大差值 |A(i,j) - A(j,i)|（默认：1e-6）。

**返回值**:

`bool` - 如果近似对称返回`true`，否则返回`false`。

**使用建议**:

- **特征分解**: 对称矩阵可以使用更高效和稳定的特征分解方法（例如Jacobi方法）。

- **Cholesky分解**: 只有对称正定矩阵可以使用Cholesky分解进行分解。

- **结构动力学**: 结构分析中的刚度和质量矩阵通常是对称的。

#### 检查正定性

```cpp
bool Mat::is_positive_definite(float tolerance = 1e-6f) const;
```

**描述**:

使用Sylvester准则检查矩阵是否正定。对称矩阵A是正定的，如果对于所有非零向量x，x^T A x > 0，或者等价地，所有特征值都是正的。

**数学原理**:

Sylvester准则指出，对称矩阵是正定的当且仅当所有前导主子式都是正的。为了效率，函数检查前几个前导主子式和对角元素。

**参数**:

- `float tolerance` : 数值检查的容差（默认：1e-6）。

**返回值**:

`bool` - 如果矩阵是正定的返回`true`，否则返回`false`。

**使用建议**:

- **Cholesky分解**: 正定矩阵可以使用Cholesky分解进行分解，这比LU分解更快、更稳定。

- **优化**: 正定Hessian矩阵在优化问题中表示局部最小值。

- **稳定性分析**: 在控制系统中，某些矩阵的正定性确保系统稳定性。

### 矩阵分解结构

#### LU分解结构

```cpp
struct Mat::LUDecomposition
{
    Mat L;                 // 下三角矩阵（单位对角线）
    Mat U;                 // 上三角矩阵
    Mat P;                 // 置换矩阵（如果使用主元）
    bool pivoted;          // 是否使用主元
    tiny_error_t status;   // 计算状态
    
    LUDecomposition();
};
```

**描述**:

LU分解结果的容器。分解A = P * L * U（带主元）或A = L * U（不带主元），其中L是单位对角线的下三角矩阵，U是上三角矩阵，P是置换矩阵。

**数学原理**:

LU分解将矩阵分解为下三角和上三角矩阵，实现线性系统的高效求解。使用主元时，可以更好地处理近奇异矩阵。

#### Cholesky分解结构

```cpp
struct Mat::CholeskyDecomposition
{
    Mat L;                 // 下三角矩阵
    tiny_error_t status;   // 计算状态
    
    CholeskyDecomposition();
};
```

**描述**:

Cholesky分解结果的容器。对于对称正定矩阵，A = L * L^T，其中L是下三角矩阵。

**数学原理**:

Cholesky分解是用于对称正定矩阵的专用LU分解。它只需要LU分解的一半存储和计算。

#### QR分解结构

```cpp
struct Mat::QRDecomposition
{
    Mat Q;                 // 正交矩阵 (Q^T * Q = I)
    Mat R;                 // 上三角矩阵
    tiny_error_t status;   // 计算状态
    
    QRDecomposition();
};
```

**描述**:

QR分解结果的容器。A = Q * R，其中Q是正交的（Q^T * Q = I），R是上三角矩阵。

**数学原理**:

QR分解将矩阵表示为正交矩阵和上三角矩阵的乘积。它在数值上稳定，是最小二乘问题的基础。

#### SVD分解结构

```cpp
struct Mat::SVDDecomposition
{
    Mat U;                 // 左奇异向量（正交矩阵）
    Mat S;                 // 奇异值（对角矩阵或向量）
    Mat V;                 // 右奇异向量（正交矩阵，V^T）
    int rank;              // 矩阵的数值秩
    int iterations;        // 执行的迭代次数
    tiny_error_t status;   // 计算状态
    
    SVDDecomposition();
};
```

**描述**:

SVD分解结果的容器。A = U * S * V^T，其中U和V是正交矩阵，S在对角线上包含奇异值。

**数学原理**:

SVD是最通用的矩阵分解。奇异值揭示矩阵的秩、条件数，并能够计算秩亏矩阵的伪逆。

### 矩阵分解方法

#### LU分解

```cpp
Mat::LUDecomposition Mat::lu_decompose(bool use_pivoting = true) const;
```

**描述**:

计算LU分解：A = P * L * U（带主元）或A = L * U（不带主元）。对于求解具有相同系数矩阵的多个系统很高效。

**数学原理**: 

- **不带主元**: A = L * U，其中L具有单位对角线

- **带主元**: P * A = L * U，其中P是置换矩阵

分解通过求解Ly = Pb（前向替换）然后Ux = y（后向替换）来求解Ax = b。

**参数**:

- `bool use_pivoting` : 是否使用部分主元以提高数值稳定性（默认：true）。

**返回值**:

`LUDecomposition`，包含L、U、P矩阵和状态。

**使用建议**:

- **多个右端项**: 一旦分解，使用`solve_lu()`高效求解具有不同右端项的多个系统。

- **行列式**: det(A) = det(P) * det(L) * det(U) = det(P) * det(U)（因为det(L) = 1）。

- **逆矩阵**: 可以通过为每个单位向量eᵢ求解LUx = eᵢ来计算A^(-1)。

- **性能**: 分解为O(n³)，分解后每次求解为O(n²)。

#### Cholesky分解

```cpp
Mat::CholeskyDecomposition Mat::cholesky_decompose() const;
```

**描述**:

计算对称正定矩阵的Cholesky分解：A = L * L^T。对于SPD矩阵比LU更快，常用于结构动力学。

**数学原理**:

对于对称正定矩阵A，存在唯一的具有正对角元素的下三角矩阵L，使得A = L * L^T。这本质上是利用对称性的专用LU分解。

**参数**:

无（矩阵必须是对称正定的）。

**返回值**:

`CholeskyDecomposition`，包含L矩阵和状态。

**使用建议**:

- **效率**: 需要大约LU分解的一半计算和存储。

- **稳定性**: 对于对称正定矩阵比LU更稳定。

- **应用**: 

  - 结构动力学：质量和刚度矩阵通常是SPD

  - 优化：Newton方法中的Hessian矩阵

  - 统计：协方差矩阵

- **错误处理**: 如果矩阵不对称或不是正定的，返回错误。

#### QR分解

```cpp
Mat::QRDecomposition Mat::qr_decompose() const;
```

**描述**:

计算QR分解：A = Q * R，其中Q是正交的，R是上三角的。数值稳定，用于最小二乘和正交化。

**数学原理**:

QR分解将任何矩阵表示为正交矩阵Q（Q^T * Q = I）和上三角矩阵R的乘积。使用改进的Gram-Schmidt过程和重新正交化计算分解。

**参数**:

无。

**返回值**:

`QRDecomposition`，包含Q和R矩阵和状态。

**使用建议**:

- **最小二乘**: 对于超定系统Ax ≈ b，最小化||Ax - b||₂的解是x = R^(-1) * Q^T * b。

- **数值稳定性**: QR分解对于最小二乘问题比正规方程更稳定。

- **特征分解**: QR算法迭代使用QR分解来查找特征值。

- **秩揭示**: A的秩等于R的非零对角元素数。

#### SVD分解

```cpp
Mat::SVDDecomposition Mat::svd_decompose(int max_iter = 100, float tolerance = 1e-6f) const;
```

**描述**:

计算奇异值分解：A = U * S * V^T。最通用的分解，用于秩估计、伪逆、降维。使用基于特征分解的迭代方法。

**数学原理**:

SVD将任何m × n矩阵A分解为：
- U: m × m正交矩阵（左奇异向量）

- S: m × n对角矩阵（奇异值σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0）

- V: n × n正交矩阵（右奇异向量）

奇异值揭示矩阵的基本属性：秩、条件数和数值行为。

**参数**:

- `int max_iter` : 最大迭代次数（默认：100）。

- `float tolerance` : 收敛容差（默认：1e-6）。

**返回值**:

`SVDDecomposition`，包含U、S、V矩阵、秩和状态。

**使用建议**:

- **秩估计**: 数值秩是高于容差阈值的奇异值数量。

- **伪逆**: A⁺ = V * S⁺ * U^T，其中S⁺对于非零σᵢ有1/σᵢ。

- **降维**: 截断SVD（仅保留最大奇异值）提供低秩近似。

- **条件数**: κ(A) = σ₁ / σᵣ，其中σᵣ是最小的非零奇异值。

- **应用**: 

  - 秩亏系统的最小二乘

  - 主成分分析（PCA）

  - 图像压缩

  - 降噪

### 使用分解求解线性系统

#### 使用LU分解求解

```cpp
static Mat Mat::solve_lu(const LUDecomposition &lu, const Mat &b);
```

**描述**:

使用预计算的LU分解求解线性系统Ax = b。当求解具有相同系数矩阵的多个系统时，比`solve()`更高效。

**数学原理**:

给定A = P * L * U，通过以下方式求解Ax = b：
1. 求解Ly = Pb（前向替换）
2. 求解Ux = y（后向替换）

**参数**:

- `const LUDecomposition &lu` : 预计算的LU分解。

- `const Mat &b` : 右端项向量 (N×1)。

**返回值**:

Mat - 解向量 (N×1)。

**使用建议**:

- **多个右端项**: 计算一次LU分解后，高效求解多个系统。

- **性能**: 每次求解O(n²) vs 完整求解O(n³)，对于多个右端项节省显著。

- **内存**: 重用分解，避免重复计算。

#### 使用Cholesky分解求解

```cpp
static Mat Mat::solve_cholesky(const CholeskyDecomposition &chol, const Mat &b);
```

**描述**:

使用预计算的Cholesky分解求解线性系统Ax = b。对于对称正定矩阵比LU更高效。

**数学原理**:

给定A = L * L^T，通过以下方式求解Ax = b：
1. 求解Ly = b（前向替换）
2. 求解L^T x = y（后向替换）

**参数**:

- `const CholeskyDecomposition &chol` : 预计算的Cholesky分解。

- `const Mat &b` : 右端项向量 (N×1)。

**返回值**:

Mat - 解向量 (N×1)。

**使用建议**:

- **效率**: 对于SPD矩阵，在分解和求解方面都比LU更快。

- **稳定性**: 对于SPD矩阵在数值上更稳定。

- **应用**: 结构动力学、优化、统计。

#### 使用QR分解求解（最小二乘）

```cpp
static Mat Mat::solve_qr(const QRDecomposition &qr, const Mat &b);
```

**描述**:

使用QR分解求解线性系统。为超定系统（方程数多于未知数）提供最小二乘解。

**数学原理**:

对于Ax ≈ b（超定），最小二乘解最小化||Ax - b||₂。使用A = Q * R：
- x = R^(-1) * Q^T * b

这避免了数值不稳定的正规方程A^T * A * x = A^T * b。

**参数**:

- `const QRDecomposition &qr` : 预计算的QR分解。

- `const Mat &b` : 右端项向量 (M×1，其中M ≥ N)。

**返回值**:

Mat - 最小二乘解向量 (N×1)。

**使用建议**:

- **超定系统**: 处理方程数多于未知数的情况。

- **数值稳定性**: 比直接求解正规方程更稳定。

- **应用**: 

  - 曲线拟合

  - 数据回归

  - 信号处理

### 伪逆

```cpp
static Mat Mat::pseudo_inverse(const SVDDecomposition &svd, float tolerance = 1e-6f);
```

**描述**:

使用SVD分解计算Moore-Penrose伪逆A⁺。适用于秩亏或非方矩阵，其中常规逆不存在。

**数学原理**:

对于A = U * S * V^T，伪逆为A⁺ = V * S⁺ * U^T，其中S⁺对于奇异值σᵢ > tolerance有1/σᵢ，否则为0。

**伪逆的性质**:

- A * A⁺ * A = A

- A⁺ * A * A⁺ = A⁺

- (A * A⁺)^T = A * A⁺

- (A⁺ * A)^T = A⁺ * A

**参数**:

- `const SVDDecomposition &svd` : 预计算的SVD分解。

- `float tolerance` : 奇异值阈值（默认：1e-6）。低于此值的奇异值被视为零。

**返回值**:

Mat - 伪逆矩阵。

**使用建议**:

- **秩亏系统**: 为A不是满秩的系统提供解。

- **最小范数解**: 对于欠定系统，给出具有最小||x||₂的解。

- **最小二乘**: 对于超定系统，给出最小二乘解。

- **应用**:

  - 控制系统

  - 信号处理

  - 机器学习（正则化）

## 线性代数 - 特征值与特征向量

### 结构体：`Mat::EigenPair`

```cpp
Mat::EigenPair::EigenPair();
// fields:
// float eigenvalue;      // 特征值（power_iteration 为最大模，inverse_power_iteration 为最小模）
// Mat eigenvector;       // 对应的特征向量（n x 1）
// int iterations;        // 迭代次数（若为迭代法返回）
// tiny_error_t status;   // 计算状态（TINY_OK / 错误码）
```

**描述**:

用于保存单一特征值/特征向量对及其计算信息。常由 `power_iteration` 或 `inverse_power_iteration` 返回。

### 结构体：`Mat::EigenDecomposition`

```cpp
Mat::EigenDecomposition::EigenDecomposition();
// fields:
// Mat eigenvalues;    // n x 1 矩阵，存放特征值
// Mat eigenvectors;   // n x n 矩阵，通常每一列为对应的特征向量
// int iterations;     // 迭代次数（若为迭代法返回）
// tiny_error_t status; // 计算状态（TINY_OK / 错误码）
```

**描述**:

用于保存完整的特征值分解结果，包括全部特征值和对应的特征向量矩阵。

### 幂迭代（求主特征值/向量）

```cpp
Mat::EigenPair Mat::power_iteration(int max_iter, float tolerance) const;
```

**描述**:

使用幂迭代法计算矩阵的主特征值（绝对值最大）及对应特征向量。快速方法，适用于实时SHM应用，可快速识别主频率。

**数学原理**:

幂迭代通过迭代地将矩阵应用于向量来找到绝对值最大的特征值：

1. 从随机向量v₀开始

2. 迭代：vₖ₊₁ = A * vₖ / ||A * vₖ||

3. 特征值估计：λₖ = (vₖ^T * A * vₖ) / (vₖ^T * vₖ) (Rayleigh商)

**收敛性**:

方法收敛到主特征值，如果：

- 主特征值是唯一的 (|λ₁| > |λ₂| ≥ ... ≥ |λₙ|)

- 初始向量在主特征向量方向上有非零分量

**参数**:

- `int max_iter` : 最大迭代次数（典型默认值：1000）。

- `float tolerance` : 收敛容差（例如 1e-6）。收敛性通过 |λₖ - λₖ₋₁| < tolerance * |λₖ| 检查。

**返回值**:

`EigenPair`，包含 `eigenvalue`、`eigenvector`、`iterations` 和 `status`。

**使用建议**:

- **实时应用**: 对于分离良好的特征值快速收敛，适用于实时结构健康监测。

- **初始化**: 实现使用智能初始化策略（列绝对值之和）以避免收敛到较小的特征值。

- **收敛速度**: 收敛是线性的，速度为 |λ₂|/|λ₁|。当特征值接近时较慢。

- **局限性**: 

  - 只找到一个特征值-特征向量对

  - 需要 |λ₁| > |λ₂|（主特征值必须唯一）

  - 如果特征值接近，可能收敛缓慢

- **应用**:

  - 主成分分析（第一主成分）

  - PageRank算法

  - 结构动力学（基频）

### 反幂迭代（求最小特征值/向量）

```cpp
Mat::EigenPair Mat::inverse_power_iteration(int max_iter, float tolerance) const;
```

**描述**:

使用反幂迭代法计算矩阵的最小（最小模）特征值及其对应特征向量。对于系统识别至关重要——在结构动力学中找到基频/最低模态。该方法对于SHM应用至关重要，其中最小特征值对应于系统的基本频率。

**数学原理**:

反幂迭代将幂迭代应用于A^(-1)， 其特征值为1/λᵢ。由于1/λₙ是A^(-1) 的最大特征值，该方法收敛到A的最小特征值：

1. 从向量v₀开始

2. 迭代：求解A * yₖ = vₖ，然后vₖ₊₁ = yₖ / ||yₖ||

3. 特征值估计：λₖ = (vₖ^T * A * vₖ) / (vₖ^T * vₖ) (Rayleigh商)

**收敛性**:

收敛到最小特征值，如果：

- 最小特征值是唯一的 (|λₙ| < |λₙ₋₁| ≤ ... ≤ |λ₁|)

- 矩阵A是可逆的（非奇异）

- 初始向量在最小特征向量方向上有分量

**参数**:

- `int max_iter` : 最大迭代次数（默认：1000）。

- `float tolerance` : 收敛容差（默认：1e-6）。使用相对容差：|λₖ - λₖ₋₁| < tolerance * max(|λₖ|, 1.0)。

**返回值**:

`EigenPair`，包含最小特征值、特征向量、迭代次数和状态。

**算法步骤**:

1. 初始化归一化特征向量v（使用交替符号以避免与主特征向量对齐）

2. 迭代：使用`solve()`求解A * y = v（等价于y = A^(-1) * v）

3. 归一化y得到新的v

4. 使用Rayleigh商计算特征值估计：λ = (v^T * A * v) / (v^T * v)

5. 使用相对容差检查收敛性

**使用建议**:

- **系统识别**: 对于在结构动力学中查找基频至关重要，其中最小特征值对应于最低固有频率。

- **数值稳定性**: 实现包括对奇异矩阵的检查，并优雅地处理近奇异情况。

- **初始化策略**: 使用交替符号模式以避免收敛到较大的特征值，确保收敛到最小特征值。

- **性能**: 每次迭代需要求解线性系统（对于稠密矩阵为O(n³)），但通常比幂迭代收敛更快。

- **与幂迭代互补**: 

  - 幂迭代：找到λ_max（最高频率）

  - 反幂迭代：找到λ_min（基频）

  - 两者一起提供系统的频率范围

- **应用**:

  - 结构健康监测（基频检测）

  - 模态分析（最低模态形状）

  - 系统识别

  - 稳定性分析（最小特征值表示稳定性裕度）

**注意**:

- 要求矩阵为方阵且数据指针非空；否则返回错误状态。

- 矩阵必须是可逆的（非奇异）才能使此方法工作。如果矩阵是奇异的或接近奇异的，该方法将优雅地失败。

- 反幂迭代只返回最小特征值/向量对。若需全部特征值/向量请使用下面的分解函数。

- 该方法与幂迭代互补：幂迭代找到最大特征值，而反幂迭代找到最小特征值。

### Jacobi 特征分解（对称矩阵）

```cpp
Mat::EigenDecomposition Mat::eigendecompose_jacobi(float tolerance, int max_iter) const;
```

**描述**:

使用Jacobi方法计算完整的特征值分解。推荐用于对称矩阵（良好的精度和稳定性，适用于结构动力学应用）。稳健且准确，是结构动力学矩阵的理想选择。

**数学原理**:

Jacobi方法通过一系列正交相似变换（Givens旋转）对角化对称矩阵：
1. 找到最大的非对角元素aₚq
2. 计算旋转角θ以将该元素置零
3. 应用旋转：A' = J^T * A * J，其中J是旋转矩阵
4. 重复直到所有非对角元素低于容差

**收敛性**:

当最大非对角元素低于容差时方法收敛。每次旋转将一个非对角元素置零，过程持续直到矩阵对角化。

**参数**:

- `float tolerance` : 收敛阈值（例如 1e-6）。允许的非对角元素最大幅度。

- `int max_iter` : 最大迭代次数（例如 100）。对于n×n矩阵，通常需要O(n²)次迭代收敛。

**返回值**:

`EigenDecomposition`，包含 `eigenvalues`、`eigenvectors`、`iterations` 和 `status`。

**使用建议**:

- **对称矩阵**: 专为对称矩阵设计。对于非对称矩阵，使用QR方法。

- **数值稳定性**: 对于对称矩阵非常稳定，具有良好的正交性保持。

- **精度**: 高精度，适用于需要精确特征值/特征向量对的应用。

- **性能**: 每次迭代O(n³)，但对于对称矩阵通常比QR需要更少的迭代。

- **应用**:

  - 结构动力学：刚度和质量矩阵是对称的

  - 主成分分析（PCA）

  - 谱聚类

  - 二次型优化

**注意**:

如果矩阵不是近似对称，函数会发出警告，但仍可能运行。对于非对称矩阵，推荐使用QR方法。

### QR 特征分解（一般矩阵）

```cpp
Mat::EigenDecomposition Mat::eigendecompose_qr(int max_iter, float tolerance) const;
```

**描述**:

使用QR算法计算特征值分解。适用于一般（可能非对称）矩阵。支持非对称矩阵，但可能产生复数特征值（仅返回实部）。

**数学原理**:

QR算法迭代应用QR分解：

1. 从A₀ = A开始

2. 对于k = 0, 1, 2, ...： 计算QR分解：Aₖ = Qₖ * Rₖ，然后更新：Aₖ₊₁ = Rₖ * Qₖ

3. Aₖ收敛到上三角形式（Schur形式），特征值在对角线上

**收敛性**:

当Aₖ近似上三角（次对角元素 < 容差）时算法收敛。特征值出现在对角线上，特征向量从Q矩阵累积。

**参数**:

- `int max_iter` : 最大QR迭代次数（默认：100）。

- `float tolerance` : 收敛容差（例如 1e-6）。使用相对容差，比较次对角元素与对角元素。

**返回值**:

`EigenDecomposition`，包含特征值、特征向量、迭代次数和状态。

**使用建议**:

- **一般矩阵**: 可以处理非对称矩阵，与Jacobi方法不同。

- **复数特征值**: 非对称矩阵可能具有复数特征值；当前实现仅返回实部。

- **数值稳定性**: 使用改进的Gram-Schmidt和重新正交化以提高稳定性。

- **性能**: 每次迭代O(n³)。可能需要多次迭代才能收敛，特别是对于病态矩阵。

- **收敛加速**: 实现可以从移位（Wilkinson移位）中受益以加快收敛，但当前版本使用基本QR迭代。

- **应用**:

  - 一般矩阵特征值问题

  - 动力系统分析

  - 控制理论（系统极点）

**注意**:

QR在此实现中使用Gram–Schmidt构造Q/R；对于病态矩阵可能不太稳定。对于对称矩阵，由于更好的稳定性和精度，推荐使用Jacobi。

### 自动特征分解（根据矩阵特性选择方法）

```cpp
Mat::EigenDecomposition Mat::eigendecompose(float tolerance = 1e-6f, int max_iter = 100) const;
```

**描述**:

简便接口，会自动根据矩阵特性选择最优算法。先调用 `is_symmetric(tolerance * 10.0f)` 判断矩阵是否近似对称：

- 若为对称，使用 `eigendecompose_jacobi(tolerance, max_iter)`（更稳定和精确）；
- 否则使用 `eigendecompose_qr(max_iter, tolerance)`（处理一般矩阵）。

**算法选择流程**:

1. 测试矩阵是否对称：`is_symmetric(tolerance * 10.0f)`
2. 若对称 → 使用 `eigendecompose_jacobi(tolerance, max_iter)`
3. 若非对称 → 使用 `eigendecompose_qr(max_iter, tolerance)`

**参数**:

- `float tolerance`：用于对称性检测与分解收敛判断（默认 `1e-6f`）。
- `int max_iter`：最大迭代次数（必须 > 0，默认值 = 100）。

**返回值**:

`EigenDecomposition` 结构，包含所有特征值和特征向量。

**使用建议**:

- **自动优化**：无需手动选择算法，同时仍能获得最佳性能。

- **边缘计算**：非常适合嵌入式系统，无需手动调优即可获得良好性能。

- **鲁棒性**：对称性测试使用放宽的容差（10×），以处理数值误差，确保正确识别对称矩阵。

**实用技巧**:

- **已知对称性**：若明确知道矩阵为对称矩阵（如刚度矩阵、质量矩阵），直接使用 `eigendecompose_jacobi` 可获得最佳稳定性和略好的性能。

- **未知特性**：对于一般矩阵或对称性未知的情况，使用 `eigendecompose` 进行自动选择。

- **性能考虑**：
  - 在嵌入式平台上，对大型矩阵进行特征分解的计算成本很高
  - 对于 n > 20，当只需要少数特征值时，考虑使用降阶方法或迭代方法（幂迭代）
  - 对于实时应用，使用 `power_iteration()` 或 `inverse_power_iteration()` 计算单个特征值

- **内存使用**：完全特征分解需要存储所有特征向量（n×n 矩阵），对于大型矩阵可能占用大量内存。


## 流操作符

### 矩阵输出流操作符

```cpp
std::ostream &operator<<(std::ostream &os, const Mat &m);
```

**描述**:

矩阵的重载输出流操作符。

**参数**:

- `std::ostream &os` : 输出流。

- `const Mat &m` : 要输出的矩阵。

### ROI输出流操作符

```cpp
std::ostream &operator<<(std::ostream &os, const Mat::ROI &roi);
```

**描述**:

ROI结构体的重载输出流操作符。

**参数**:

- `std::ostream &os` : 输出流。

- `const Mat::ROI &roi` : ROI结构。

### 矩阵输入流操作符

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

## 全局算术运算符

!!! INFO "非修改操作"
    本节中的运算符返回一个新的矩阵对象，作为运算结果。原始矩阵保持不变。这些是函数式操作，不修改其操作数，使其可以安全地与const引用和临时对象一起使用。
    
!!! TIP "何时使用"
    - 使用全局运算符 (A + B) 当您想保留原始矩阵时
    - 使用成员运算符 (A += B) 当您想就地修改矩阵时（更节省内存）

### 加法运算符

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

### 加法运算符 - 常量

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

### 减法运算符

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

### 减法运算符 - 常量

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

### 乘法运算符

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

### 乘法运算符 - 常量

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

### 乘法运算符 - 常量（左侧）

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

### 除法运算符

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

### 除法运算符 - 矩阵

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

### 等于运算符

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






