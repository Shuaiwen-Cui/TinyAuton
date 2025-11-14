# MATRIX OPERATIONS - TINY_MATRIX

!!! INFO "TINY_MATRIX Library"
    - This library is a lightweight matrix computation library implemented in C++, providing basic matrix operations and linear algebra functions.
    - The design goal of this library is to provide a simple and easy-to-use matrix operation interface, suitable for embedded systems and resource-constrained environments.

!!! TIP "Usage Scenario"
    Compared to the TINY_MAT library, the TINY_MATRIX library offers richer functionality and higher flexibility, suitable for applications that require complex matrix computations. However, please note that this library is written in C++.

## LIST OF FUNCTIONS

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
        void print_matrix(bool show_padding);

        // ============================================================================
        // Constructors & Destructor
        // ============================================================================
        void alloc_mem(); // Allocate internal memory
        Mat();
        Mat(int rows, int cols);
        Mat(int rows, int cols, int stride);
        Mat(float *data, int rows, int cols);
        Mat(float *data, int rows, int cols, int stride);
        Mat(const Mat &src);
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
        Mat copy_roi(int start_row, int start_col, int roi_rows, int roi_cols);
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
        Mat minor(int row, int col);       // Minor matrix (submatrix after removing row and col)
        Mat cofactor(int row, int col);    // Cofactor matrix
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
        bool is_symmetric(float tolerance = 1e-6f) const;
        bool is_positive_definite(float tolerance = 1e-6f) const;
        
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
        EigenPair power_iteration(int max_iter = 1000, float tolerance = 1e-6f) const;
        EigenPair inverse_power_iteration(int max_iter = 1000, float tolerance = 1e-6f) const;
        
        // Complete eigendecomposition methods
        EigenDecomposition eigendecompose_jacobi(float tolerance = 1e-6f, int max_iter = 100) const;
        EigenDecomposition eigendecompose_qr(int max_iter = 100, float tolerance = 1e-6f) const;
        EigenDecomposition eigendecompose(float tolerance = 1e-6f) const;  // Auto-select method

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
## MATRIX METADATA

!!! INFO "Matrix Structure"
    The Mat class uses a row-major storage layout with support for padding and stride. This design enables efficient memory access patterns and compatibility with DSP libraries.

### Core Dimensions

- **`int row`** : Number of rows in the matrix.

- **`int col`** : Number of columns in the matrix.

- **`int element`** : Total number of elements = rows × cols.

### Memory Layout

- **`int stride`** : Stride = (number of elements in a row) + padding. The stride determines how many elements to skip to move to the next row in memory.

- **`int pad`** : Number of padding elements between two rows. Padding is used for memory alignment and DSP optimization.

- **`int memory`** : Size of the data buffer = rows × stride (in number of float elements).

### Data Pointers

- **`float *data`** : Pointer to the data buffer containing matrix elements. Elements are stored in row-major order: element at (i, j) is at `data[i * stride + j]`.

- **`float *temp`** : Pointer to the temporary data buffer (if allocated). Used internally for certain operations.

### Memory Management Flags

- **`bool ext_buff`** : Flag indicating that the matrix uses an external buffer. When `true`, the destructor will not free the memory (caller is responsible).

- **`bool sub_matrix`** : Flag indicating that the matrix is a subset/view of another matrix. When `true`, the matrix shares data with the parent matrix.

!!! TIP "Memory Layout Example"
    For a 3×4 matrix with stride=4 (no padding):
    ```
    [a b c d]   row 0: data[0*4+0] to data[0*4+3]
    [e f g h]   row 1: data[1*4+0] to data[1*4+3]
    [i j k l]   row 2: data[2*4+0] to data[2*4+3]
    ```
    
    For a 3×4 matrix with stride=6 (padding=2):
    ```
    [a b c d _ _]   row 0: data[0*6+0] to data[0*6+3], padding at data[0*6+4,5]
    [e f g h _ _]   row 1: data[1*6+0] to data[1*6+3], padding at data[1*6+4,5]
    [i j k l _ _]   row 2: data[2*6+0] to data[2*6+3], padding at data[2*6+4,5]
    ```

## ROI STRUCTURE

!!! INFO "Region of Interest"
    The ROI (Region of Interest) structure represents a rectangular subregion of a matrix. It's used with `view_roi()` and `copy_roi()` functions to extract or reference submatrices efficiently.

### ROI Metadata

- **`int pos_x`** : Starting column index (x-coordinate of the top-left corner).

- **`int pos_y`** : Starting row index (y-coordinate of the top-left corner).

- **`int width`** : Width of the ROI in columns.

- **`int height`** : Height of the ROI in rows.

### ROI Constructor

```cpp
Mat::ROI::ROI(int pos_x = 0, int pos_y = 0, int width = 0, int height = 0);
```

**Description**: 

ROI constructor initializes the ROI with the specified position and size.

**Parameters**:

- `int pos_x` : Starting column index.

- `int pos_y` : Starting row index.

- `int width` : Width of the ROI (columns).

- `int height` : Height of the ROI (rows).

### ROI RESIZE

```cpp
void Mat::ROI::resize_roi(int pos_x, int pos_y, int width, int height);
```

**Description**: 

Resizes the ROI to the specified position and size.

**Parameters**:

- `int pos_x` : Starting column index.

- `int pos_y` : Starting row index.

- `int width` : Width of the ROI (columns).

- `int height` : Height of the ROI (rows).

**Returns**:

void

### AREA ROI

```cpp
int Mat::ROI::area_roi(void) const;
```

**Description**: 

Calculates the area of the ROI.

**Parameters**:

void

**Returns**:

int - Area of the ROI.

## PRINTING FUNCTIONS

!!! TIP "Debugging Tools"
    These functions are essential for debugging and understanding matrix state. Use them to verify matrix dimensions, memory layout, and data values during development.

### Print Matrix Information

```cpp
void Mat::print_info() const;
```

**Description**: 

Prints comprehensive matrix information including:

- Dimensions: rows, columns, elements

- Memory layout: paddings, stride, memory size

- Pointers: data buffer address, temporary buffer address

- Flags: external buffer usage, sub-matrix status

- Warnings: dimension mismatches, invalid states

**Parameters**:

void

**Returns**:

void

**Usage Insights**:

- **Debugging**: Essential for verifying matrix state and detecting memory issues.

- **Memory Analysis**: Shows actual memory usage vs. logical size, helping identify memory inefficiencies.

- **Sub-Matrix Detection**: Clearly indicates if a matrix is a view, which affects memory management.

### Print Matrix Elements

```cpp
void Mat::print_matrix(bool show_padding);
```

**Description**: 

Prints the matrix elements in a formatted table. Optionally displays padding elements separated by a visual separator.

**Parameters**: 

- `bool show_padding` : If `true`, displays padding values with a separator `|`. If `false`, only shows actual matrix elements.

**Returns**:

void

**Usage Insights**:

- **Formatting**: Elements are formatted with fixed width (12 characters) for alignment.

- **Padding Visualization**: The `show_padding` option helps understand memory layout and verify padding values.

- **Large Matrices**: For very large matrices, consider using `view_roi()` to print specific regions.

## CONSTRUCTORS & DESTRUCTOR

!!! INFO "Memory Management"
    Constructors handle memory allocation automatically. The destructor safely frees memory only if it was internally allocated (not external buffers or views). Always check the `data` pointer after construction to ensure successful allocation.

### Memory Allocation

```cpp
void Mat::alloc_mem();
```

**Description**: 

Internal function that allocates memory for the matrix according to the computed memory requirements. Sets `ext_buff = false` and allocates `row * stride` float elements.

**Parameters**:

void

**Returns**:

void

**Usage Insights**:

- **Automatic Call**: Called automatically by constructors. Rarely needs manual invocation.

- **Memory Calculation**: Allocates `row * stride` elements, which may include padding.

- **Error Handling**: If allocation fails, `data` remains `nullptr`. Always check `data` after construction.

### Default Constructor

```cpp
Mat::Mat();
```

**Description**: 

Default constructor creates a 1×1 zero matrix. This is useful for initialization and as a return value for error cases.

**Mathematical Principle**:

Creates the identity element for matrix operations in some contexts, though typically you'll want to specify dimensions.

**Parameters**:

void

**Returns**:

Mat - A 1×1 matrix with element = 0.

### Constructor - Mat(int rows, int cols)

```cpp
Mat::Mat(int rows, int cols);
```

**Description**: 

Constructor creates a matrix with specified dimensions. All elements are initialized to zero. This is the most commonly used constructor.

**Parameters**:

- `int rows` : Number of rows (must be > 0).

- `int cols` : Number of columns (must be > 0).

**Returns**:

Mat - A rows×cols matrix with all elements initialized to 0.

**Usage Insights**:

- **Zero Initialization**: All elements are set to zero using `memset`, ensuring clean state.

- **Memory Layout**: Creates a contiguous memory layout with no padding (stride = cols).

- **Error Handling**: If memory allocation fails, `data` will be `nullptr`. Always verify allocation success.

### Constructor - Mat(int rows, int cols, int stride)

```cpp
Mat::Mat(int rows, int cols, int stride);
```

**Description**: 

Constructor creates a matrix with specified dimensions and stride. Useful when you need padding for memory alignment or DSP optimization.

**Parameters**:

- `int rows` : Number of rows.

- `int cols` : Number of columns.

- `int stride` : Stride (must be ≥ cols). Padding = stride - cols.

**Returns**:

Mat - A rows×cols matrix with stride, all elements initialized to 0.

**Usage Insights**:

- **DSP Optimization**: Some DSP libraries require aligned memory. Use stride to ensure proper alignment.

- **Memory Efficiency**: Padding allows efficient vectorized operations on aligned boundaries.

- **Compatibility**: Enables compatibility with external libraries that use strided memory layouts.

### Constructor - Mat(float *data, int rows, int cols)

```cpp
Mat::Mat(float *data, int rows, int cols);
```

**Description**: 

Constructor creates a matrix view over an external data buffer. The matrix does not own the memory; the caller is responsible for managing it. Useful for interfacing with existing data arrays.

**Parameters**:

- `float *data` : Pointer to external data buffer (must remain valid for matrix lifetime).

- `int rows` : Number of rows.

- `int cols` : Number of columns.

**Returns**:

Mat - A matrix view with `ext_buff = true`.

**Usage Insights**:

- **Zero-Copy**: No memory copy occurs; the matrix directly references external data.

- **Lifetime Management**: The external buffer must remain valid while the matrix exists. The destructor will not free this memory.

- **Data Layout**: Assumes row-major layout with no padding (stride = cols).

- **Use Cases**: 

  - Wrapping C arrays

  - Interfacing with other libraries

  - Avoiding unnecessary copies

### Constructor - Mat(float *data, int rows, int cols, int stride)

```cpp
Mat::Mat(float *data, int rows, int cols, int stride);
```

**Description**: 

Constructor creates a matrix view over an external data buffer with specified stride. Supports strided memory layouts for DSP compatibility.

**Parameters**:

- `float *data` : Pointer to external data buffer (must remain valid for matrix lifetime).

- `int rows` : Number of rows.

- `int cols` : Number of columns.

- `int stride` : Stride (must be ≥ cols).

**Returns**:

Mat - A matrix view with `ext_buff = true` and specified stride.

**Usage Insights**:

- **Strided Layouts**: Essential for working with DSP libraries that use strided memory layouts.

- **Memory Safety**: Same lifetime requirements as the previous constructor - external buffer must remain valid.

- **Padding Support**: Can handle buffers with padding between rows.

### Copy Constructor - Mat(const Mat &src)

```cpp
Mat::Mat(const Mat &src);
```

**Description**: 

Copy constructor creates a new matrix from a source matrix. Uses intelligent copying: deep copy for regular matrices, shallow copy for sub-matrix views.

**Copy Strategy**:

- **Regular matrices**: Deep copy - allocates new memory and copies all data

- **Sub-matrix views**: Shallow copy - shares data with source (creates another view)

**Parameters**:

- `const Mat &src` : Source matrix.

**Returns**:

Mat - A new matrix with copied or shared data depending on source type.

**Usage Insights**:

- **Automatic Selection**: Automatically chooses deep or shallow copy based on source matrix type.

- **Memory Efficiency**: Sub-matrix views are copied shallowly to avoid unnecessary memory allocation.

- **Independence**: Deep copies are independent; modifications don't affect the source.

### Destructor

```cpp
Mat::~Mat();
```

**Description**: 

Destructor safely releases allocated memory. Only frees memory if it was internally allocated (`ext_buff = false`). External buffers and views are not freed.

**Memory Management**:

- Frees `data` buffer if `ext_buff = false`

- Frees `temp` buffer if allocated

- Does nothing for external buffers or views

**Parameters**:

void

**Returns**:

void

!!! note "Constructor and Destructor Rules"
    - Constructor functions must have the same name as the class and no return type
    - C++ allows function overloading by changing parameter number/order
    - The destructor is automatically called when the object goes out of scope
    - Always check `data != nullptr` after construction to verify successful allocation

## ELEMENT ACCESS

!!! INFO "Matrix Indexing"
    The Mat class uses operator overloading to provide intuitive matrix element access. The `operator()` allows natural syntax like `A(i, j)` instead of `A.data[i * stride + j]`. The implementation automatically handles stride and padding.

### Access Matrix Elements (Non-Const)

```cpp
inline float &operator()(int row, int col);
```

**Description**:

Accesses matrix elements with read-write capability. Returns a reference to the element, allowing both reading and modification.

**Mathematical Principle**:

Element at position (row, col) is accessed as `data[row * stride + col]`, where stride accounts for padding.

**Parameters**：

- `int row` : Row index (0-based, must be in range [0, row-1]).

- `int col` : Column index (0-based, must be in range [0, col-1]).

**Returns**:

`float&` - Reference to the matrix element, enabling modification.

**Usage Insights**:

- **Bounds Checking**: No automatic bounds checking for performance. Ensure indices are valid.

- **Stride Handling**: Automatically accounts for stride, so it works correctly with padded matrices.

- **Performance**: Inline function with minimal overhead, suitable for tight loops.

- **Example**: `A(2, 3) = 5.0f;` sets element at row 2, column 3 to 5.0.

### Access Matrix Elements (Const)

```cpp
inline const float &operator()(int row, int col) const;
```

**Description**:

Accesses matrix elements in read-only mode. Returns a const reference, preventing modification. Used when the matrix is const.

**Parameters**：

- `int row` : Row index (0-based).

- `int col` : Column index (0-based).

**Returns**:

`const float&` - Const reference to the matrix element (read-only).

**Usage Insights**:

- **Const Correctness**: Enables proper const-correct code. Use this version in const member functions.

- **Safety**: Prevents accidental modification of const matrices.

!!! note "Operator Overloading"
    These functions overload the `()` operator, enabling natural matrix indexing syntax:
    ```cpp
    Mat A(3, 4);
    A(1, 2) = 3.14f;        // Write access
    float val = A(1, 2);    // Read access
    const Mat& B = A;
    float val2 = B(1, 2);   // Read-only access (uses const version)
    ```

## DATA MANIPULATION

### Copy other matrix into this matrix as a sub-matrix
```cpp
tiny_error_t Mat::copy_paste(const Mat &src, int row_pos, int col_pos);
```

**Description**:

Copies the specified source matrix into this matrix as a sub-matrix starting from the specified row and column positions, not sharing the data buffer.

**Parameters**:

- `const Mat &src` : Source matrix.

- `int row_pos` : Starting row position.

- `int col_pos` : Starting column position.

**Returns**:

tiny_error_t - Error code (TINY_OK on success).

### Copy header of other matrix to this matrix
```cpp
tiny_error_t Mat::copy_head(const Mat &src);
```

**Description**:

Copies the header of the specified source matrix to this matrix, sharing the data buffer. All items copy the source matrix.

**Parameters**:

- `const Mat &src` : Source matrix.

**Returns**:

tiny_error_t - Error code.

### Get a view (shallow copy) of sub-matrix (ROI) from this matrix
```cpp
Mat Mat::view_roi(int start_row, int start_col, int roi_rows, int roi_cols) const;
```

**Description**:

Gets a view (shallow copy) of the sub-matrix (ROI) from this matrix starting from the specified row and column positions.

**Parameters**:

- `int start_row` : Starting row position.

- `int start_col` : Starting column position.

- `int roi_rows` : Number of rows in the ROI.

- `int roi_cols` : Number of columns in the ROI.

!!! warning
    Unlike ESP-DSP, view_roi does not allow to setup stride as it will automatically calculate the stride based on the number of columns and paddings. The function will also refuse illegal requests, i.e., out of bound requests. 

### Get a view (shallow copy) of sub-matrix (ROI) from this matrix using ROI structure
```cpp
Mat Mat::view_roi(const Mat::ROI &roi) const;
```

**Description**:

Gets a view (shallow copy) of the sub-matrix (ROI) from this matrix using the specified ROI structure. This function will call the previous function in low level by passing the ROI structure to the parameters.

**Parameters**:

- `const Mat::ROI &roi` : ROI structure.

### Get a replica (deep copy) of sub-matrix (ROI)
```cpp
Mat Mat::copy_roi(int start_row, int start_col, int roi_rows, int roi_cols);
```

**Description**:

Gets a replica (deep copy) of the sub-matrix (ROI) from this matrix starting from the specified row and column positions. This function will return a new matrix object that does not share the data buffer with the original matrix.

**Parameters**:

- `int start_row` : Starting row position.

- `int start_col` : Starting column position.

- `int roi_rows` : Number of rows in the ROI.

- `int roi_cols` : Number of columns in the ROI.

### Get a replica (deep copy) of sub-matrix (ROI) using ROI structure
```cpp
Mat Mat::copy_roi(const Mat::ROI &roi);
```

**Description**:

Gets a replica (deep copy) of the sub-matrix (ROI) from this matrix using the specified ROI structure. This function will call the previous function in low level by passing the ROI structure to the parameters.

**Parameters**:

- `const Mat::ROI &roi` : ROI structure.

### Get a block of matrix
```cpp
Mat Mat::block(int start_row, int start_col, int block_rows, int block_cols);
```

**Description**:

Gets a block of the matrix starting from the specified row and column positions.

**Parameters**:

- `int start_row` : Starting row position.

- `int start_col` : Starting column position.

- `int block_rows` : Number of rows in the block.

- `int block_cols` : Number of columns in the block.

!!! tip "Differences between view_roi | copy_roi | block"

    - `view_roi` : Shallow copy of the sub-matrix (ROI) from this matrix.

    - `copy_roi` : Deep copy of the sub-matrix (ROI) from this matrix. Rigid and faster.

    - `block` : Deep copy of the block from this matrix. Flexible and slower.

### Swap rows

```cpp
void Mat::swap_rows(int row1, int row2);
```

**Description**:

Swaps the specified rows in the matrix.

**Parameters**:

- `int row1` : First row index.

- `int row2` : Second row index.

**Returns**:

void

### Swap columns

```cpp
void Mat::swap_cols(int col1, int col2);
```

**Description**:

Swaps the specified columns in the matrix. 

**Parameters**:

- `int col1` : First column index.

- `int col2` : Second column index.

**Returns**:

void

### Clear matrix

```cpp
void Mat::clear(void);
```

**Description**:

Clears the matrix by setting all elements to zero.

**Parameters**:

void

**Returns**:

void

## ARITHMETIC OPERATORS

!!! INFO "In-Place Operations"
    This section defines the arithmetic operators that act on the current matrix itself (in-place operations). These operators modify the matrix and return a reference to it, enabling chained operations like `A += B += C`. The operators are optimized to handle padding and use DSP-accelerated functions when available.

### Copy assignment
```cpp
Mat &operator=(const Mat &src);
```

**Description**:

Copy assignment operator for the matrix. Copies elements from source matrix to current matrix. Handles dimension changes by reallocating memory if necessary. Prevents assignment to sub-matrix views for safety.

**Mathematical Principle**:

Creates an independent copy of the source matrix. Unlike copy constructor, this is used for existing matrices.

**Parameters**:

- `const Mat &src` : Source matrix.

**Returns**:

Mat& - Reference to the current matrix (enables chaining).

**Usage Insights**:

- **Memory Management**: Automatically reallocates memory if dimensions differ. Frees old memory if it was internally allocated.

- **Sub-Matrix Protection**: Assignment to sub-matrix views is forbidden to prevent accidental data corruption.

- **Self-Assignment**: Handles self-assignment safely (A = A).

- **Performance**: O(n²) for n×n matrices. For large matrices, consider if a view would suffice.

### Add matrix
```cpp
Mat &operator+=(const Mat &A);
```

**Description**:

Adds the specified matrix to this matrix.

**Parameters**:

- `const Mat &A` : Matrix to be added.

### Add constant
```cpp
Mat &operator+=(float C);
```

**Description**:

Element-wise addition of a constant to this matrix.

**Parameters**:

- `float C` : The constant to add.

**Returns**:

Mat& - Reference to the current matrix.

### Subtract matrix
```cpp
Mat &operator-=(const Mat &A);
```

**Description**:

Subtracts the specified matrix from this matrix.

**Parameters**:

- `const Mat &A` : Matrix to be subtracted.

### Subtract constant
```cpp
Mat &operator-=(float C);
```

**Description**:

Element-wise subtraction of a constant from this matrix.

**Parameters**:

- `float C` : The constant to subtract.

**Returns**:

Mat& - Reference to the current matrix.

### Multiply matrix
```cpp
Mat &operator*=(const Mat &A);
```

**Description**:

Matrix multiplication: this = this * A. Performs standard matrix multiplication (not element-wise). The number of columns of the current matrix must equal the number of rows of A.

**Mathematical Principle**:

Matrix multiplication C = A * B where Cᵢⱼ = Σₖ Aᵢₖ * Bₖⱼ. This is the standard matrix product, not element-wise multiplication.

**Dimension Requirements**: 
- Current matrix: m × n

- Matrix A: n × p

- Result: m × p

**Parameters**:

- `const Mat &A` : Matrix to be multiplied (must have n rows, where n = current matrix columns).

**Returns**:

Mat& - Reference to the current matrix.

**Usage Insights**:

- **Memory Efficiency**: Creates a temporary copy to avoid overwriting data during computation, then updates the current matrix.

- **Padding Support**: Handles matrices with padding using specialized DSP functions when available.

- **Performance**: O(mnp) for m×n * n×p multiplication. Uses optimized DSP functions on ESP32 platform.

- **Common Mistake**: This is matrix multiplication, not element-wise. For element-wise, use a loop with `operator()()`.

### Multiply constant
```cpp
Mat &operator*=(float C);
```

**Description**:

Element-wise multiplication by a constant.

**Parameters**:

- `float C` : The constant multiplier.

**Returns**:

Mat& - Reference to the current matrix.

### Divide matrix (element-wise)
```cpp
Mat &operator/=(const Mat &B);
```

**Description**:

Element-wise division: this = this / B.

**Parameters**:

- `const Mat &B` : The matrix divisor.

**Returns**:

Mat& - Reference to the current matrix.

### Divide constant
```cpp
Mat &operator/=(float C);
```

**Description**:

Element-wise division of this matrix by a constant.

**Parameters**:

- `float C` : The constant divisor.

**Returns**:

Mat& - Reference to the current matrix.

### Exponentiation
```cpp
Mat operator^(int C);
```

**Description**:

Element-wise integer exponentiation. Returns a new matrix where each element is raised to the given power.

**Parameters**:

- `int C` : The exponent (integer).

**Returns**:

Mat - New matrix after exponentiation.


## LINEAR ALGEBRA

### Transpose

```cpp
Mat Mat::transpose();
```

**Description**:

Calculates the transpose of the matrix, returning a new matrix. The transpose A^T of a matrix A is obtained by interchanging rows and columns: (A^T)ᵢⱼ = Aⱼᵢ.

**Mathematical Principle**: 
- For any matrix A, (A^T)^T = A

- (A + B)^T = A^T + B^T

- (AB)^T = B^T * A^T

- For square matrices, det(A) = det(A^T)

**Parameters**:

None.

**Returns**:

Mat - Transposed matrix (col × row).

**Usage Insights**:

- **Memory Layout**: Creates a new matrix, so memory usage doubles temporarily. For large matrices, consider memory constraints.

- **Symmetric Matrices**: If A = A^T, the matrix is symmetric. Use `is_symmetric()` to check.

- **Applications**: 

  - Inner products: u^T * v

  - Quadratic forms: x^T * A * x

  - Matrix equations: A^T * A (normal equations)

### Minor matrix

```cpp
Mat Mat::minor(int row, int col);
```

**Description**:

Calculates the minor matrix by removing the specified row and column. The minor is the submatrix obtained by removing one row and one column.

**Parameters**: 

- `int row`: Row index to remove.

- `int col`: Column index to remove.

**Returns**:

Mat - The (n-1)x(n-1) minor matrix.

### Cofactor matrix

```cpp
Mat Mat::cofactor(int row, int col);
```

**Description**:

Calculates the cofactor matrix (same as minor matrix). The cofactor matrix is the same as the minor matrix. The sign (-1)^(i+j) is applied when computing the cofactor value, not to the matrix elements themselves.

**Parameters**: 

- `int row`: Row index to remove.

- `int col`: Column index to remove.

**Returns**:

Mat - The (n-1)x(n-1) cofactor matrix (same as minor matrix).

### Determinant (Auto-select Method)

```cpp
float Mat::determinant();
```

**Description**: 

Computes the determinant of a square matrix, automatically selecting the optimal method based on matrix size. For small matrices (n ≤ 4), uses Laplace expansion; for larger matrices (n > 4), uses LU decomposition for better efficiency.

**Mathematical Principle**: 

The determinant is an important numerical characteristic of a square matrix with the following properties:

- det(AB) = det(A) * det(B)

- det(A^T) = det(A)

- det(A^(-1)) = 1 / det(A)

- If A is singular, det(A) = 0

**Method Selection**:

- **Small matrices (n ≤ 4)**: Uses `determinant_laplace()` - Laplace expansion method, time complexity O(n!), more accurate for small matrices

- **Large matrices (n > 4)**: Uses `determinant_lu()` - LU decomposition method, time complexity O(n³), more efficient

**Parameters**:

None.

**Returns**: 

float - The determinant value.

**Usage Insights**:

- **Automatic Selection**: For most applications, simply use `determinant()` and the function will automatically select the optimal method

- **Performance Optimization**: If you need to compute determinants of matrices of the same size multiple times, consider directly calling `determinant_lu()` or `determinant_gaussian()`

- **Precision Requirements**: For small matrices, `determinant_laplace()` may provide better numerical precision

### Determinant - Laplace Expansion

```cpp
float Mat::determinant_laplace();
```

**Description**: 

Computes the determinant of a square matrix using Laplace expansion (cofactor expansion). Time complexity is O(n!), suitable only for small matrices (n ≤ 4).

**Mathematical Principle**: 

Laplace expansion is the recursive definition of the determinant:

- For 1×1 matrix: det([a]) = a

- For 2×2 matrix: det([[a,b],[c,d]]) = ad - bc

- For n×n matrix: det(A) = Σⱼ₌₁ⁿ (-1)ⁱ⁺ʲ aᵢⱼ * det(Mᵢⱼ), where Mᵢⱼ is the minor matrix

This implementation uses first-row expansion, recursively computing the determinant of minors.

**Parameters**:

None.

**Returns**: 

float - The determinant value.

!!! warning "Performance Warning"
    Time complexity is O(n!), suitable only for small matrices (n ≤ 4). For large matrices, use `determinant_lu()` or `determinant_gaussian()`.

### Determinant - LU Decomposition

```cpp
float Mat::determinant_lu();
```

**Description**: 

Computes the determinant of a square matrix using LU decomposition. Time complexity is O(n³), suitable for large matrices.

**Mathematical Principle**: 

LU decomposition factorizes the matrix as A = P * L * U, where:

- P is a permutation matrix (if pivoting is used)

- L is a lower triangular matrix with unit diagonal

- U is an upper triangular matrix

Determinant formula: det(A) = det(P) * det(L) * det(U)

Where:

- det(P) = (-1)^(permutation signature), determined by the number of row swaps

- det(L) = 1 (since L has unit diagonal)

- det(U) = ∏ᵢ Uᵢᵢ (product of diagonal elements of U)

**Algorithm Steps**:

1. Perform LU decomposition (with pivoting for numerical stability)
2. Compute the determinant of the permutation matrix det(P)
3. Compute the product of diagonal elements of U: det(U)
4. Return det(P) * det(U)

**Parameters**:

None.

**Returns**: 

float - The determinant value. Returns 0.0 if the matrix is singular or near-singular.

**Usage Insights**:

- **Efficiency**: Much faster than Laplace expansion for matrices with n > 4

- **Numerical Stability**: Uses pivoting to improve numerical stability

- **Singular Matrices**: If the matrix is singular, LU decomposition fails and the function returns 0.0

### Determinant - Gaussian Elimination

```cpp
float Mat::determinant_gaussian();
```

**Description**: 

Computes the determinant of a square matrix using Gaussian elimination. Time complexity is O(n³), suitable for large matrices.

**Mathematical Principle**: 

Gaussian elimination converts the matrix to upper triangular form, then computes the product of diagonal elements. The determinant value equals the product of diagonal elements of the upper triangular matrix, adjusted for the sign based on the number of row swaps.

**Algorithm Steps**:

1. Use partial pivoting Gaussian elimination to convert matrix to upper triangular form
2. Track the number of row swaps
3. Compute the product of diagonal elements of the upper triangular matrix
4. Adjust the sign based on row swaps: each row swap multiplies the determinant by -1

**Parameters**:

None.

**Returns**: 

float - The determinant value. Returns 0.0 if the matrix is singular.

**Usage Insights**:

- **Efficiency**: Time complexity O(n³) for large matrices, comparable to LU decomposition

- **Numerical Stability**: Uses partial pivoting to improve numerical stability

- **Implementation Simplicity**: More intuitive than LU decomposition, but less versatile (cannot be used for solving linear systems)

- **Applications**:

  - Check invertibility: det(A) ≠ 0 means A is invertible

  - Volume scaling: |det(A)| is the scaling factor of the linear transformation

  - System solvability: det(A) = 0 indicates singular system

### Adjoint

```cpp
Mat Mat::adjoint();
```

**Description**:

Calculates the adjoint (adjugate) matrix of a square matrix.

**Parameters**:

None.

**Returns**:

Mat - Adjoint matrix.

### Normalize

```cpp
void Mat::normalize();
```

**Description**:

Normalizes the matrix using L2 norm (Frobenius norm). After normalization, ||Matrix|| = 1.

**Parameters**:

None.

**Returns**:

void

### Norm

```cpp
float Mat::norm() const;
```

**Description**:

Calculates the Frobenius norm (also called Euclidean norm or L2 norm) of the matrix. The Frobenius norm is the square root of the sum of squares of all matrix elements.

**Mathematical Principle**: 
- Frobenius norm: ||A||_F = √(Σᵢ Σⱼ |aᵢⱼ|²) = √(trace(A^T * A))
- For vectors, this reduces to the standard L2 norm
- Properties:
  - ||A + B||_F ≤ ||A||_F + ||B||_F (triangle inequality)
  - ||AB||_F ≤ ||A||_F * ||B||_F
  - ||A||_F = ||A^T||_F

**Parameters**:

None.

**Returns**:

float - The computed matrix norm.

**Usage Insights**:

- **Error Measurement**: Useful for measuring the "size" of a matrix or error in numerical computations.

- **Normalization**: Used in `normalize()` to scale matrices to unit norm.

- **Convergence**: Often used as a convergence criterion in iterative algorithms.

- **Comparison**: For vectors, this is equivalent to the standard Euclidean norm ||v||₂.

### Inverse using Adjoint

```cpp
Mat Mat::inverse_adjoint();
```

**Description**:

Computes the inverse of a square matrix using adjoint method. If the matrix is singular, returns a zero matrix.

**Parameters**:

None.

**Returns**:

Mat - The inverse matrix. If singular, returns a zero matrix.

### Identity Matrix

```cpp
static Mat Mat::eye(int size);
```

**Description**:

Generates an identity matrix of given size.

**Parameters**: 

- `int size` : Dimension of the square identity matrix.

**Returns**:

Mat - Identity matrix (size x size).


### Augmentation Matrix (Horizontal Concatenation)

```cpp
static Mat Mat::augment(const Mat &A, const Mat &B);
```

**Description**:

Creates an augmented matrix by horizontally concatenating two matrices [A | B]. The row counts of A and B must match.

**Parameters**:

- `const Mat &A` : Left matrix.

- `const Mat &B` : Right matrix.

**Returns**:

Mat - Augmented matrix [A B].

### Vertical Stack

```cpp
static Mat Mat::vstack(const Mat &A, const Mat &B);
```

**Description**:

Vertically stacks two matrices [A; B]. The column counts of A and B must match.

**Parameters**:

- `const Mat &A` : Top matrix.

- `const Mat &B` : Bottom matrix.

**Returns**:

Mat - Vertically stacked matrix [A; B].

### Gram-Schmidt Orthogonalization

```cpp
static bool Mat::gram_schmidt_orthogonalize(const Mat &vectors, Mat &orthogonal_vectors, 
                                            Mat &coefficients, float tolerance = 1e-6f);
```

**Description**:

Orthogonalizes a set of vectors using the Gram-Schmidt process. This is a general-purpose orthogonalization function that can be reused for QR decomposition and other applications requiring orthogonal bases. Uses the modified Gram-Schmidt algorithm with re-orthogonalization for improved numerical stability.

**Mathematical Principle**:

Given a set of vectors {v₁, v₂, ..., vₙ}, the Gram-Schmidt process produces an orthogonal set {q₁, q₂, ..., qₙ} where:

- q₁ = v₁ / ||v₁||

- qⱼ = (vⱼ - Σᵢ₌₁ʲ⁻¹⟨vⱼ, qᵢ⟩qᵢ) / ||vⱼ - Σᵢ₌₁ʲ⁻¹⟨vⱼ, qᵢ⟩qᵢ||

The modified version subtracts projections immediately, which improves numerical stability.

**Parameters**:

- `const Mat &vectors` : Input matrix where each column is a vector to be orthogonalized (m × n).

- `Mat &orthogonal_vectors` : Output matrix for orthogonalized vectors (m × n), each column is orthogonal and normalized.

- `Mat &coefficients` : Output matrix for projection coefficients (n × n, upper triangular), similar to R in QR decomposition.

- `float tolerance` : Minimum norm threshold for linear independence check (default: 1e-6).

**Returns**:

`bool` - `true` if successful, `false` if input is invalid.

**Usage Insights**:

- **Numerical Stability**: The implementation uses modified Gram-Schmidt with re-orthogonalization, which significantly improves stability for near-linearly-dependent vectors.

- **QR Decomposition**: This function is internally used by `qr_decompose()`. For QR decomposition, the coefficients matrix corresponds to the R matrix.

- **Basis Construction**: Useful for constructing orthogonal bases from a set of vectors, which is fundamental in many linear algebra applications.

- **Performance**: For large matrices, consider the computational cost. The complexity is O(mn²) for m-dimensional vectors and n vectors.

### All-Ones Matrix (Rectangular)

```cpp
static Mat Mat::ones(int rows, int cols);
```

**Description**:

Creates a matrix of specified size filled with ones.

**Parameters**:

- `int rows` : Number of rows.

- `int cols` : Number of columns.

**Returns**:

Mat - Matrix [rows x cols] with all elements = 1.

### All-Ones Matrix (Square)

```cpp
static Mat Mat::ones(int size);
```

**Description**:

Creates a square matrix filled with ones of the specified size.

**Parameters**:

- `int size` : Size of the square matrix (rows = cols).

**Returns**:

Mat - Square matrix [size x size] with all elements = 1.


### Gaussian Elimination

```cpp
Mat Mat::gaussian_eliminate() const;
```

**Description**:

Performs Gaussian Elimination to convert matrix to Row Echelon Form (REF). This is the first step in solving linear systems and computing matrix rank.

**Mathematical Principle**:

Gaussian elimination transforms a matrix into row echelon form through elementary row operations:

1. **Row swapping**: Exchange two rows

2. **Row scaling**: Multiply a row by a non-zero scalar

3. **Row addition**: Add a multiple of one row to another

**Row Echelon Form (REF) properties**:

- All zero rows are at the bottom

- The leading coefficient (pivot) of each non-zero row is to the right of the pivot in the row above

- All entries below a pivot are zero

**Parameters**:

None.

**Returns**:

Mat - The upper triangular matrix (REF form).

**Usage Insights**:

- **Linear System Solving**: First step in solving Ax = b. After REF, use back substitution.

- **Rank Computation**: The rank equals the number of non-zero rows in REF.

- **Determinant**: Can compute determinant from REF (product of diagonal elements, adjusted for row swaps).

- **Numerical Stability**: The implementation uses partial pivoting to improve numerical stability.

- **Performance**: O(n³) for n×n matrices. For multiple systems, prefer LU decomposition.

### Row Reduce from Gaussian

```cpp
Mat Mat::row_reduce_from_gaussian();
```

**Description**:

Converts a matrix (assumed in row echelon form) to Reduced Row Echelon Form (RREF).

**Parameters**:

None.

**Returns**:

Mat - The matrix in RREF form.

### Inverse using Gaussian-Jordan Elimination

```cpp
Mat Mat::inverse_gje();
```

**Description**:

Computes the inverse of a square matrix using Gauss-Jordan elimination.

**Parameters**:

None.

**Returns**:

Mat - The inverse matrix if invertible, otherwise returns empty matrix.

### Dot Product

```cpp
float Mat::dotprod(const Mat &A, const Mat &B);
```

**Description**:

Calculates the dot product of two vectors (Nx1).

**Parameters**:

- `const Mat &A` : Input vector A (Nx1).

- `const Mat &B` : Input vector B (Nx1).

**Returns**:

float - The computed dot product value.

### Solve Linear System

```cpp
Mat Mat::solve(const Mat &A, const Mat &b) const;
```

**Description**:

Solves the linear system Ax = b using Gaussian elimination with back substitution. This is a direct method suitable for well-conditioned systems.

**Mathematical Principle**:

The method consists of two phases:

1. **Forward elimination**: Transform augmented matrix [A|b] to upper triangular form

2. **Back substitution**: Solve Ux = y from bottom to top

**Algorithm**:

- Create augmented matrix [A | b]

- Apply Gaussian elimination to get [U | y] where U is upper triangular

- Solve Ux = y using back substitution: xᵢ = (yᵢ - Σⱼ₌ᵢ₊₁ⁿ Uᵢⱼxⱼ) / Uᵢᵢ

**Parameters**:

- `const Mat &A` : Coefficient matrix (N×N), must be square and non-singular.

- `const Mat &b` : Right-hand side vector (N×1).

**Returns**:

Mat - Solution vector (N×1) containing the roots of the equation Ax = b. Returns empty matrix if system is singular or incompatible.

**Usage Insights**:

- **Single System**: Efficient for solving one system. For multiple systems with same A, use LU decomposition + `solve_lu()`.

- **Condition Number**: Performance degrades for ill-conditioned matrices. Check condition number if results are inaccurate.

- **Singular Systems**: Returns empty matrix if A is singular (det(A) = 0). Use SVD + pseudo-inverse for rank-deficient systems.

- **Performance**: O(n³) for elimination, O(n²) for back substitution. Total O(n³).

- **Alternative Methods**:

  - For SPD matrices: Use Cholesky decomposition + `solve_cholesky()` (faster)

  - For multiple RHS: Use LU decomposition + `solve_lu()` (more efficient)

  - For overdetermined: Use QR decomposition + `solve_qr()` (least squares)

### Band Solve

```cpp
Mat Mat::band_solve(Mat A, Mat b, int k);
```

**Description**:

Solves the system of equations Ax = b using optimized Gaussian elimination for banded matrices.

**Parameters**:

- `Mat A` : Coefficient matrix (NxN) - banded matrix.

- `Mat b` : Result vector (Nx1).

- `int k` : Bandwidth of the matrix (the width of the non-zero bands).

**Returns**:

Mat - Solution vector (Nx1) containing the roots of the equation Ax = b.



### Roots

```cpp
Mat Mat::roots(Mat A, Mat y);
```

**Description**:

Solves the matrix using a different method. Another implementation of the 'solve' function, no difference in principle. This method solves the linear system A * x = y using Gaussian elimination.

**Parameters**:

- `Mat A` : Matrix [N]x[N] with input coefficients.

- `Mat y` : Vector [N]x[1] with result values.

**Returns**:

Mat - Matrix [N]x[1] with roots.

## MATRIX PROPERTIES & DECOMPOSITIONS

!!! INFO "Matrix Decompositions Overview"
    Matrix decompositions are fundamental tools in numerical linear algebra. They break down a matrix into simpler components that reveal its structure and enable efficient computations. Different decompositions are suited for different types of matrices and applications.

### Matrix Property Checks

#### Check Symmetry

```cpp
bool Mat::is_symmetric(float tolerance = 1e-6f) const;
```

**Description**:

Check whether a matrix is symmetric within the given tolerance. A matrix A is symmetric if A = A^T, i.e., A(i,j) = A(j,i) for all i, j.

**Mathematical Principle**:

For a symmetric matrix, all eigenvalues are real, and eigenvectors can be chosen to be orthogonal. Symmetric matrices are fundamental in many applications, especially in structural dynamics and optimization.

**Parameters**:

- `float tolerance` : Maximum allowed difference |A(i,j) - A(j,i)| (default: 1e-6).

**Returns**:

`bool` - `true` if approximately symmetric, `false` otherwise.

**Usage Insights**:

- **Eigendecomposition**: Symmetric matrices can use more efficient and stable eigendecomposition methods (e.g., Jacobi method).

- **Cholesky Decomposition**: Only symmetric positive definite matrices can be decomposed using Cholesky decomposition.

- **Structural Dynamics**: Stiffness and mass matrices in structural analysis are typically symmetric.

#### Check Positive Definiteness

```cpp
bool Mat::is_positive_definite(float tolerance = 1e-6f) const;
```

**Description**:

Check if a matrix is positive definite using Sylvester's criterion. A symmetric matrix A is positive definite if x^T A x > 0 for all non-zero vectors x, or equivalently, all eigenvalues are positive.

**Mathematical Principle**:

Sylvester's criterion states that a symmetric matrix is positive definite if and only if all leading principal minors are positive. The function checks the first few leading minors and diagonal elements for efficiency.

**Parameters**:

- `float tolerance` : Tolerance for numerical checks (default: 1e-6).

**Returns**:

`bool` - `true` if matrix is positive definite, `false` otherwise.

**Usage Insights**:

- **Cholesky Decomposition**: Positive definite matrices can be decomposed using Cholesky decomposition, which is faster and more stable than LU decomposition.

- **Optimization**: Positive definite Hessian matrices indicate local minima in optimization problems.

- **Stability Analysis**: In control systems, positive definiteness of certain matrices ensures system stability.

### Matrix Decomposition Structures

#### LU Decomposition Structure

```cpp
struct Mat::LUDecomposition
{
    Mat L;                 // Lower triangular matrix (with unit diagonal)
    Mat U;                 // Upper triangular matrix
    Mat P;                 // Permutation matrix (if pivoting used)
    bool pivoted;          // Whether pivoting was used
    tiny_error_t status;   // Computation status
    
    LUDecomposition();
};
```

**Description**:

Container for LU decomposition results. The decomposition A = P * L * U (with pivoting) or A = L * U (without pivoting), where L is lower triangular with unit diagonal, U is upper triangular, and P is a permutation matrix.

**Mathematical Principle**:

LU decomposition factors a matrix into lower and upper triangular matrices, enabling efficient solution of linear systems. With pivoting, it handles near-singular matrices better.

#### Cholesky Decomposition Structure

```cpp
struct Mat::CholeskyDecomposition
{
    Mat L;                 // Lower triangular matrix
    tiny_error_t status;   // Computation status
    
    CholeskyDecomposition();
};
```

**Description**:

Container for Cholesky decomposition results. For symmetric positive definite matrices, A = L * L^T, where L is lower triangular.

**Mathematical Principle**:

Cholesky decomposition is a specialized LU decomposition for symmetric positive definite matrices. It requires only half the storage and computation of LU decomposition.

#### QR Decomposition Structure

```cpp
struct Mat::QRDecomposition
{
    Mat Q;                 // Orthogonal matrix (Q^T * Q = I)
    Mat R;                 // Upper triangular matrix
    tiny_error_t status;   // Computation status
    
    QRDecomposition();
};
```

**Description**:

Container for QR decomposition results. A = Q * R, where Q is orthogonal (Q^T * Q = I) and R is upper triangular.

**Mathematical Principle**:

QR decomposition expresses a matrix as the product of an orthogonal matrix and an upper triangular matrix. It's numerically stable and fundamental for least squares problems.

#### SVD Decomposition Structure

```cpp
struct Mat::SVDDecomposition
{
    Mat U;                 // Left singular vectors (orthogonal matrix)
    Mat S;                 // Singular values (diagonal matrix or vector)
    Mat V;                 // Right singular vectors (orthogonal matrix, V^T)
    int rank;              // Numerical rank of the matrix
    int iterations;        // Number of iterations performed
    tiny_error_t status;   // Computation status
    
    SVDDecomposition();
};
```

**Description**:

Container for SVD decomposition results. A = U * S * V^T, where U and V are orthogonal matrices, and S contains singular values on the diagonal.

**Mathematical Principle**:

SVD is the most general matrix decomposition. The singular values reveal the matrix's rank, condition number, and enable computation of pseudo-inverse for rank-deficient matrices.

### Matrix Decomposition Methods

#### LU Decomposition

```cpp
Mat::LUDecomposition Mat::lu_decompose(bool use_pivoting = true) const;
```

**Description**:

Compute LU decomposition: A = P * L * U (with pivoting) or A = L * U (without pivoting). Efficient for solving multiple systems with the same coefficient matrix.

**Mathematical Principle**: 

- **Without pivoting**: A = L * U, where L has unit diagonal

- **With pivoting**: P * A = L * U, where P is a permutation matrix

The decomposition enables solving Ax = b by solving Ly = Pb (forward substitution) then Ux = y (back substitution).

**Parameters**:

- `bool use_pivoting` : Whether to use partial pivoting for numerical stability (default: true).

**Returns**:

`LUDecomposition` containing L, U, P matrices and status.

**Usage Insights**:

- **Multiple RHS**: Once decomposed, solve multiple systems with different right-hand sides efficiently using `solve_lu()`.

- **Determinant**: det(A) = det(P) * det(L) * det(U) = det(P) * det(U) (since det(L) = 1).

- **Inverse**: Can compute A^(-1) by solving LUx = eᵢ for each unit vector eᵢ.

- **Performance**: O(n³) for decomposition, O(n²) for each solve after decomposition.

#### Cholesky Decomposition

```cpp
Mat::CholeskyDecomposition Mat::cholesky_decompose() const;
```

**Description**:

Compute Cholesky decomposition: A = L * L^T for symmetric positive definite matrices. Faster than LU for SPD matrices, commonly used in structural dynamics.

**Mathematical Principle**:

For a symmetric positive definite matrix A, there exists a unique lower triangular matrix L with positive diagonal elements such that A = L * L^T. This is essentially a specialized LU decomposition that takes advantage of symmetry.

**Parameters**:

None (matrix must be symmetric positive definite).

**Returns**:

`CholeskyDecomposition` containing L matrix and status.

**Usage Insights**:

- **Efficiency**: Requires approximately half the computation and storage of LU decomposition.

- **Stability**: More stable than LU for symmetric positive definite matrices.

- **Applications**: 

  - Structural dynamics: Mass and stiffness matrices are often SPD

  - Optimization: Hessian matrices in Newton's method

  - Statistics: Covariance matrices

- **Error Handling**: Returns error if matrix is not symmetric or not positive definite.

#### QR Decomposition

```cpp
Mat::QRDecomposition Mat::qr_decompose() const;
```

**Description**:

Compute QR decomposition: A = Q * R, where Q is orthogonal and R is upper triangular. Numerically stable, used for least squares and orthogonalization.

**Mathematical Principle**:

QR decomposition expresses any matrix as the product of an orthogonal matrix Q (Q^T * Q = I) and an upper triangular matrix R. The decomposition is computed using the modified Gram-Schmidt process with re-orthogonalization.

**Parameters**:

None.

**Returns**:

`QRDecomposition` containing Q and R matrices and status.

**Usage Insights**:

- **Least Squares**: For overdetermined system Ax ≈ b, the solution minimizes ||Ax - b||₂ is x = R^(-1) * Q^T * b.

- **Numerical Stability**: QR decomposition is more stable than normal equations for least squares problems.

- **Eigendecomposition**: QR algorithm uses QR decomposition iteratively to find eigenvalues.

- **Rank Revealing**: The rank of A equals the number of non-zero diagonal elements of R.

#### SVD Decomposition

```cpp
Mat::SVDDecomposition Mat::svd_decompose(int max_iter = 100, float tolerance = 1e-6f) const;
```

**Description**:

Compute Singular Value Decomposition: A = U * S * V^T. Most general decomposition, used for rank estimation, pseudo-inverse, dimension reduction. Uses iterative method based on eigendecomposition.

**Mathematical Principle**:

SVD decomposes any m × n matrix A into:
- U: m × m orthogonal matrix (left singular vectors)

- S: m × n diagonal matrix (singular values σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0)

- V: n × n orthogonal matrix (right singular vectors)

The singular values reveal the matrix's fundamental properties: rank, condition number, and numerical behavior.

**Parameters**:

- `int max_iter` : Maximum number of iterations (default: 100).

- `float tolerance` : Convergence tolerance (default: 1e-6).

**Returns**:

`SVDDecomposition` containing U, S, V matrices, rank, and status.

**Usage Insights**:

- **Rank Estimation**: The numerical rank is the number of singular values above the tolerance threshold.

- **Pseudo-Inverse**: A⁺ = V * S⁺ * U^T, where S⁺ has 1/σᵢ for non-zero σᵢ.

- **Dimension Reduction**: Truncated SVD (keeping only largest singular values) provides low-rank approximation.

- **Condition Number**: κ(A) = σ₁ / σᵣ, where σᵣ is the smallest non-zero singular value.

- **Applications**: 

  - Least squares for rank-deficient systems

  - Principal Component Analysis (PCA)

  - Image compression

  - Noise reduction

### Solving Linear Systems Using Decompositions

#### Solve using LU Decomposition

```cpp
static Mat Mat::solve_lu(const LUDecomposition &lu, const Mat &b);
```

**Description**:

Solve linear system Ax = b using precomputed LU decomposition. More efficient than `solve()` when solving multiple systems with the same coefficient matrix.

**Mathematical Principle**:

Given A = P * L * U, solve Ax = b by:
1. Solve Ly = Pb (forward substitution)
2. Solve Ux = y (back substitution)

**Parameters**:

- `const LUDecomposition &lu` : Precomputed LU decomposition.

- `const Mat &b` : Right-hand side vector (N×1).

**Returns**:

Mat - Solution vector (N×1).

**Usage Insights**:

- **Multiple RHS**: After computing LU decomposition once, solve multiple systems efficiently.

- **Performance**: O(n²) per solve vs O(n³) for full solve, significant savings for multiple RHS.

- **Memory**: Reuses the decomposition, avoiding repeated computation.

#### Solve using Cholesky Decomposition

```cpp
static Mat Mat::solve_cholesky(const CholeskyDecomposition &chol, const Mat &b);
```

**Description**:

Solve linear system Ax = b using precomputed Cholesky decomposition. More efficient than LU for symmetric positive definite matrices.

**Mathematical Principle**:

Given A = L * L^T, solve Ax = b by:
1. Solve Ly = b (forward substitution)
2. Solve L^T x = y (back substitution)

**Parameters**:

- `const CholeskyDecomposition &chol` : Precomputed Cholesky decomposition.

- `const Mat &b` : Right-hand side vector (N×1).

**Returns**:

Mat - Solution vector (N×1).

**Usage Insights**:

- **Efficiency**: Faster than LU for SPD matrices, both in decomposition and solving.

- **Stability**: More numerically stable for SPD matrices.

- **Applications**: Structural dynamics, optimization, statistics.

#### Solve using QR Decomposition (Least Squares)

```cpp
static Mat Mat::solve_qr(const QRDecomposition &qr, const Mat &b);
```

**Description**:

Solve linear system using QR decomposition. Provides least squares solution for overdetermined systems (more equations than unknowns).

**Mathematical Principle**:

For Ax ≈ b (overdetermined), the least squares solution minimizes ||Ax - b||₂. Using A = Q * R:
- x = R^(-1) * Q^T * b

This avoids the numerically unstable normal equations A^T * A * x = A^T * b.

**Parameters**:

- `const QRDecomposition &qr` : Precomputed QR decomposition.

- `const Mat &b` : Right-hand side vector (M×1, where M ≥ N).

**Returns**:

Mat - Least squares solution vector (N×1).

**Usage Insights**:

- **Overdetermined Systems**: Handles cases where there are more equations than unknowns.

- **Numerical Stability**: More stable than solving normal equations directly.

- **Applications**: 

  - Curve fitting

  - Data regression

  - Signal processing

### Pseudo-Inverse

```cpp
static Mat Mat::pseudo_inverse(const SVDDecomposition &svd, float tolerance = 1e-6f);
```

**Description**:

Compute the Moore-Penrose pseudo-inverse A⁺ using SVD decomposition. Works for rank-deficient or non-square matrices where the regular inverse doesn't exist.

**Mathematical Principle**:

For A = U * S * V^T, the pseudo-inverse is A⁺ = V * S⁺ * U^T, where S⁺ has 1/σᵢ for singular values σᵢ > tolerance, and 0 otherwise.

**Properties of Pseudo-Inverse**:

- A * A⁺ * A = A

- A⁺ * A * A⁺ = A⁺

- (A * A⁺)^T = A * A⁺

- (A⁺ * A)^T = A⁺ * A

**Parameters**:

- `const SVDDecomposition &svd` : Precomputed SVD decomposition.

- `float tolerance` : Threshold for singular values (default: 1e-6). Singular values below this are treated as zero.

**Returns**:

Mat - Pseudo-inverse matrix.

**Usage Insights**:

- **Rank-Deficient Systems**: Provides solution for systems where A is not full rank.

- **Minimum Norm Solution**: For underdetermined systems, gives the solution with minimum ||x||₂.

- **Least Squares**: For overdetermined systems, gives the least squares solution.

- **Applications**:

  - Control systems

  - Signal processing

  - Machine learning (regularization)

## LINEAR ALGEBRA - Eigenvalues & Eigenvectors

### Struct: `Mat::EigenPair`

```cpp
Mat::EigenPair::EigenPair();
// fields:
// float eigenvalue;      // eigenvalue (largest-magnitude for power_iteration, smallest for inverse_power_iteration)
// Mat eigenvector;       // corresponding eigenvector (n x 1)
// int iterations;        // number of iterations (for iterative methods)
// tiny_error_t status;   // computation status (TINY_OK / error code)
```

**Description**:

Container for a single eigenvalue/eigenvector result and related metadata. Typically returned by `power_iteration` or `inverse_power_iteration`.

### Struct: `Mat::EigenDecomposition`

```cpp
Mat::EigenDecomposition::EigenDecomposition();
// fields:
// Mat eigenvalues;    // n x 1 matrix storing eigenvalues
// Mat eigenvectors;   // n x n matrix, columns are eigenvectors
// int iterations;     // iterations used by the algorithm
// tiny_error_t status; // computation status
```

**Description**:

Container for a full eigendecomposition result (all eigenvalues and eigenvectors).

### Power Iteration (dominant eigenpair)

```cpp
Mat::EigenPair Mat::power_iteration(int max_iter, float tolerance) const;
```

**Description**:

Compute the dominant (largest-magnitude) eigenvalue and its eigenvector using the power iteration method. Fast method suitable for real-time SHM applications to quickly identify primary frequency.

**Mathematical Principle**:

Power iteration finds the eigenvalue with the largest absolute value by iteratively applying the matrix to a vector:

1. Start with random vector v₀

2. Iterate: vₖ₊₁ = A * vₖ / ||A * vₖ||

3. Eigenvalue estimate: λₖ = (vₖ^T * A * vₖ) / (vₖ^T * vₖ) (Rayleigh quotient)

**Convergence**:

The method converges to the dominant eigenvalue if:

- The dominant eigenvalue is unique (|λ₁| > |λ₂| ≥ ... ≥ |λₙ|)

- The initial vector has a non-zero component in the direction of the dominant eigenvector

**Parameters**:

- `int max_iter` : Maximum number of iterations (typical default: 1000).

- `float tolerance` : Convergence tolerance (e.g. 1e-6). Convergence is checked by |λₖ - λₖ₋₁| < tolerance * |λₖ|.

**Returns**:

`EigenPair` containing `eigenvalue`, `eigenvector`, `iterations`, and `status`.

**Usage Insights**:

- **Real-Time Applications**: Fast convergence for well-separated eigenvalues, suitable for real-time structural health monitoring.

- **Initialization**: The implementation uses a smart initialization strategy (sum of column absolute values) to avoid convergence to smaller eigenvalues.

- **Convergence Rate**: Convergence is linear with rate |λ₂|/|λ₁|. Slower when eigenvalues are close.

- **Limitations**: 

  - Only finds one eigenvalue-eigenvector pair

  - Requires |λ₁| > |λ₂| (dominant eigenvalue must be unique)

  - May converge slowly if eigenvalues are close

- **Applications**:

  - Principal component analysis (first principal component)

  - PageRank algorithm

  - Structural dynamics (fundamental frequency)

### Inverse Power Iteration (smallest eigenpair)

```cpp
Mat::EigenPair Mat::inverse_power_iteration(int max_iter, float tolerance) const;
```

**Description**:

Compute the smallest (minimum magnitude) eigenvalue and its eigenvector using the inverse power iteration method. Critical for system identification - finds fundamental frequency/lowest mode in structural dynamics. This method is essential for SHM applications where the smallest eigenvalue corresponds to the fundamental frequency of the system.

**Mathematical Principle**:

Inverse power iteration applies power iteration to A^(-1), which has eigenvalues 1/λᵢ. Since 1/λₙ is the largest eigenvalue of A^(-1), the method converges to the smallest eigenvalue of A:

1. Start with vector v₀

2. Iterate: Solve A * yₖ = vₖ, then vₖ₊₁ = yₖ / ||yₖ||

3. Eigenvalue estimate: λₖ = (vₖ^T * A * vₖ) / (vₖ^T * vₖ) (Rayleigh quotient)

**Convergence**:

Converges to the smallest eigenvalue if:

- The smallest eigenvalue is unique (|λₙ| < |λₙ₋₁| ≤ ... ≤ |λ₁|)

- Matrix A is invertible (non-singular)

- Initial vector has component in direction of smallest eigenvector

**Parameters**:

- `int max_iter` : Maximum number of iterations (default: 1000).

- `float tolerance` : Convergence tolerance (default: 1e-6). Uses relative tolerance: |λₖ - λₖ₋₁| < tolerance * max(|λₖ|, 1.0).

**Returns**:

`EigenPair` containing the smallest eigenvalue, eigenvector, iterations, and status.

**Algorithm Steps**:

1. Initialize normalized eigenvector v (with alternating signs to avoid alignment with dominant eigenvector)
2. Iterate: Solve A * y = v (equivalent to y = A^(-1) * v) using `solve()`
3. Normalize y to get new v
4. Compute eigenvalue estimate using Rayleigh quotient: λ = (v^T * A * v) / (v^T * v)
5. Check convergence using relative tolerance

**Usage Insights**:

- **System Identification**: Essential for finding fundamental frequencies in structural dynamics, where the smallest eigenvalue corresponds to the lowest natural frequency.

- **Numerical Stability**: The implementation includes checks for singular matrices and handles near-singular cases gracefully.

- **Initialization Strategy**: Uses alternating sign pattern to avoid convergence to larger eigenvalues, ensuring convergence to the smallest eigenvalue.

- **Performance**: Each iteration requires solving a linear system (O(n³) for dense matrices), but typically converges in fewer iterations than power iteration.

- **Complementary to Power Iteration**: 

  - Power iteration: finds λ_max (highest frequency)

  - Inverse power iteration: finds λ_min (fundamental frequency)

  - Together they provide the frequency range of the system

- **Applications**:

  - Structural health monitoring (fundamental frequency detection)

  - Modal analysis (lowest mode shape)

  - System identification

  - Stability analysis (smallest eigenvalue indicates stability margin)

**Notes**:

- Requires a square matrix and non-null data pointer; returns an error status otherwise.

- The matrix must be invertible (non-singular) for this method to work. If the matrix is singular or near-singular, the method will fail gracefully.

- Inverse power iteration only returns the smallest eigenpair. For full spectrum, use eigendecomposition functions below.

- This method is complementary to power iteration: power iteration finds the largest eigenvalue, while inverse power iteration finds the smallest eigenvalue.

### Jacobi Eigendecomposition (symmetric matrices)

```cpp
Mat::EigenDecomposition Mat::eigendecompose_jacobi(float tolerance, int max_iter) const;
```

**Description**:

Compute full eigendecomposition using the Jacobi method. Recommended for symmetric matrices (good accuracy and stability for structural dynamics applications). Robust and accurate, ideal for structural dynamics matrices in SHM.

**Mathematical Principle**:

The Jacobi method diagonalizes a symmetric matrix through a series of orthogonal similarity transformations (Givens rotations):

1. Find largest off-diagonal element aₚq

2. Compute rotation angle θ to zero this element

3. Apply rotation: A' = J^T * A * J, where J is the rotation matrix

4. Repeat until all off-diagonal elements are below tolerance

**Convergence**:

The method converges when the maximum off-diagonal element is below tolerance. Each rotation zeros one off-diagonal element, and the process continues until the matrix is diagonal.

**Parameters**:

- `float tolerance` : Convergence threshold (e.g. 1e-6). Maximum allowed magnitude of off-diagonal elements.

- `int max_iter` : Maximum iterations (e.g. 100). Typically converges in O(n²) iterations for n×n matrices.

**Returns**:

`EigenDecomposition` with `eigenvalues`, `eigenvectors`, `iterations`, and `status`.

**Usage Insights**:

- **Symmetric Matrices**: Designed for symmetric matrices. For non-symmetric matrices, use QR method.

- **Numerical Stability**: Very stable for symmetric matrices, with good preservation of orthogonality.

- **Accuracy**: High accuracy, suitable for applications requiring precise eigenvalue/eigenvector pairs.

- **Performance**: O(n³) per iteration, but typically requires fewer iterations than QR for symmetric matrices.

- **Applications**:

  - Structural dynamics: Stiffness and mass matrices are symmetric

  - Principal Component Analysis (PCA)

  - Spectral clustering

  - Quadratic forms optimization

**Notes**:

If the matrix is not approximately symmetric the function will warn, though it may still run. For non-symmetric matrices prefer the QR method.

### QR Eigendecomposition (general matrices)

```cpp
Mat::EigenDecomposition Mat::eigendecompose_qr(int max_iter, float tolerance) const;
```

**Description**:

Compute eigendecomposition using the QR algorithm. Works for general (possibly non-symmetric) matrices. Supports non-symmetric matrices, but may have complex eigenvalues (only real part returned).

**Mathematical Principle**:

The QR algorithm iteratively applies QR decomposition:

1. Start with A₀ = A

2. For k = 0, 1, 2, ...: Compute QR decomposition: Aₖ = Qₖ * Rₖ, then update: Aₖ₊₁ = Rₖ * Qₖ

3. Aₖ converges to upper triangular form (Schur form), with eigenvalues on the diagonal

**Convergence**:

The algorithm converges when Aₖ is approximately upper triangular (sub-diagonal elements < tolerance). The eigenvalues appear on the diagonal, and eigenvectors are accumulated from Q matrices.

**Parameters**:

- `int max_iter` : Maximum number of QR iterations (default: 100).

- `float tolerance` : Convergence tolerance (e.g. 1e-6). Uses relative tolerance comparing sub-diagonal elements to diagonal elements.

**Returns**:

`EigenDecomposition` containing eigenvalues, eigenvectors, iterations and status.

**Usage Insights**:

- **General Matrices**: Can handle non-symmetric matrices, unlike Jacobi method.

- **Complex Eigenvalues**: Non-symmetric matrices may have complex eigenvalues; current implementation returns real parts only.

- **Numerical Stability**: Uses modified Gram-Schmidt with re-orthogonalization for improved stability.

- **Performance**: O(n³) per iteration. May require many iterations for convergence, especially for ill-conditioned matrices.

- **Convergence Acceleration**: The implementation could benefit from shifts (Wilkinson shift) for faster convergence, but current version uses basic QR iteration.

- **Applications**:

  - General matrix eigenvalue problems

  - Dynamical systems analysis

  - Control theory (system poles)

**Notes**:

QR uses Gram–Schmidt for Q/R in this implementation; it can be less stable for ill-conditioned matrices. For symmetric matrices, Jacobi is preferred due to better stability and accuracy.

### Automatic Eigendecomposition

```cpp
Mat::EigenDecomposition Mat::eigendecompose(float tolerance) const;
```

**Description**:

Convenience interface that automatically selects the optimal algorithm based on matrix properties. It tests symmetry with `is_symmetric(tolerance * 10.0f)`. If approximately symmetric, it uses Jacobi; otherwise it runs QR. Convenient interface for edge computing applications.

**Algorithm Selection**:

1. Test if matrix is symmetric: `is_symmetric(tolerance * 10.0f)`
2. If symmetric → use `eigendecompose_jacobi(tolerance, 100)` (more stable and accurate)
3. If not symmetric → use `eigendecompose_qr(100, tolerance)` (handles general matrices)

**Parameters**:

- `float tolerance` : Used for symmetry test and decomposition convergence (recommended 1e-6).

**Returns**:

`EigenDecomposition` containing all eigenvalues and eigenvectors.

**Usage Insights**:

- **Automatic Optimization**: Saves the user from manually choosing the algorithm, while still providing optimal performance.

- **Edge Computing**: Ideal for embedded systems where you want good performance without manual tuning.

- **Robustness**: The symmetry test uses a relaxed tolerance (10×) to handle numerical errors, ensuring symmetric matrices are correctly identified.

**Usage Tips**:

- **Known Symmetry**: If the matrix is known to be symmetric (e.g. stiffness or mass matrices), call `eigendecompose_jacobi` directly for best stability and slightly better performance.

- **Unknown Properties**: For general matrices or unknown symmetry, use `eigendecompose` for automatic selection.

- **Performance Considerations**: 
  - Eigendecomposition is computationally expensive for large matrices on embedded platforms
  - For n > 20, consider reduced-order methods or iterative methods (power iteration) when only a few eigenvalues are needed
  - For real-time applications, use `power_iteration()` or `inverse_power_iteration()` for single eigenvalues

- **Memory Usage**: Full eigendecomposition requires storing all eigenvectors (n×n matrix), which can be memory-intensive for large matrices.

## STREAM OPERATORS

### Matrix output stream operator
```cpp
std::ostream &operator<<(std::ostream &os, const Mat &m);
```

**Description**:

Overloaded output stream operator for the matrix.

**Parameters**:

- `std::ostream &os` : Output stream.

- `const Mat &m` : Matrix to be output.

### ROI output stream operator
```cpp
std::ostream &operator<<(std::ostream &os, const Mat::ROI &roi);
```

**Description**:

Overloaded output stream operator for the ROI structure.

**Parameters**:

- `std::ostream &os` : Output stream.

- `const Mat::ROI &roi` : ROI structure.

### Matrix input stream operator
```cpp
std::istream &operator>>(std::istream &is, Mat &m);
```

**Description**:

Overloaded input stream operator for the matrix.

**Parameters**:

- `std::istream &is` : Input stream.

- `Mat &m` : Matrix to be input.

!!! tip 
    This section is actually kind of overlapping with print function in terms of showing the matrix.

## GLOBAL ARITHMETIC OPERATORS

!!! INFO "Non-Modifying Operations"
    The operators in this section return a new matrix object, which is the result of the operation. The original matrices remain unchanged. These are functional-style operations that don't modify their operands, making them safe for use with const references and temporary objects.
    
!!! TIP "When to Use"
    - Use global operators (A + B) when you want to preserve original matrices
    - Use member operators (A += B) when you want to modify the matrix in-place (more memory efficient)


### Add matrix
```cpp
Mat operator+(const Mat &A, const Mat &B);
```

**Description**:

Adds two matrices element-wise.

**Parameters**:

- `const Mat &A` : First matrix.

- `const Mat &B` : Second matrix.

**Returns**:

Mat - Result matrix A+B.

### Add constant
```cpp
Mat operator+(const Mat &A, float C);
```

**Description**:

Adds a constant to a matrix element-wise.

**Parameters**:

- `const Mat &A` : Input matrix A.

- `float C` : Input constant.

**Returns**:

Mat - Result matrix A+C.

### Subtract matrix
```cpp
Mat operator-(const Mat &A, const Mat &B);
```

**Description**:

Subtracts two matrices element-wise.

**Parameters**:

- `const Mat &A` : First matrix.

- `const Mat &B` : Second matrix.

**Returns**:

Mat - Result matrix A-B.

### Subtract constant
```cpp
Mat operator-(const Mat &A, float C);
```

**Description**:

Subtracts a constant from a matrix element-wise.

**Parameters**:

- `const Mat &A` : Input matrix A.

- `float C` : Input constant.

**Returns**:

Mat - Result matrix A-C.

### Multiply matrix
```cpp
Mat operator*(const Mat &A, const Mat &B);
```

**Description**:

Multiplies two matrices (matrix multiplication).

**Parameters**:

- `const Mat &A` : First matrix.

- `const Mat &B` : Second matrix.

**Returns**:

Mat - Result matrix A*B.

### Multiply constant
```cpp
Mat operator*(const Mat &A, float C);
```

**Description**:

Multiplies a matrix by a constant element-wise.

**Parameters**:

- `const Mat &A` : Input matrix A.

- `float C` : Floating point value.

**Returns**:

Mat - Result matrix A*C.

### Multiply constant (left side)
```cpp
Mat operator*(float C, const Mat &A);
```

**Description**:

Multiplies a constant by a matrix element-wise.

**Parameters**:

- `float C` : Floating point value.

- `const Mat &A` : Input matrix A.

**Returns**:

Mat - Result matrix C*A.


### Divide matrix (by constant)
```cpp
Mat operator/(const Mat &A, float C);
```

**Description**:

Divides a matrix by a constant element-wise.

**Parameters**:

- `const Mat &A` : Input matrix A.

- `float C` : Floating point value.

**Returns**:

Mat - Result matrix A/C.

### Divide matrix (element-wise)
```cpp
Mat operator/(const Mat &A, const Mat &B);
```

**Description**:

Divides matrix A by matrix B element-wise.

**Parameters**:

- `const Mat &A` : Input matrix A.

- `const Mat &B` : Input matrix B.

**Returns**:

Mat - Result matrix C, where C[i,j] = A[i,j]/B[i,j].

### Equality check
```cpp
bool operator==(const Mat &A, const Mat &B);
```

**Description**:

Checks if the specified matrices are equal.

**Parameters**:

- `const Mat &A` : First matrix.

- `const Mat &B` : Second matrix.

**Returns**:

bool - true if equal, false otherwise.

