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
## META DATA

- `int row` : Number of rows in the matrix.

- `int col` : Number of columns in the matrix.

- `int pad` : Number of paddings between two rows.

- `int stride` : Stride = (number of elements in a row) + padding.

- `int element` : Number of elements = rows * cols.

- `int memory` : Size of the data buffer = rows * stride.

- `float *data` : Pointer to the data buffer.

- `float *temp` : Pointer to the temporary data buffer.

- `bool ext_buff` : Flag indicating that the matrix uses an external buffer.

- `bool sub_matrix` : Flag indicating that the matrix is a subset of another matrix.

## ROI STRUCTURE

### Metadata

- `int pos_x` : Starting column index.

- `int pos_y` : Starting row index.

- `int width` : Width of the ROI (columns).

- `int height` : Height of the ROI (rows).

### ROI Constructor

```cpp
Mat::ROI::ROI(int pos_x = 0, int pos_y = 0, int width = 0, int height = 0);
```

**Description**: ROI constructor initializes the ROI with the specified position and size.

**Parameters**:

- `int pos_x` : Starting column index.

- `int pos_y` : Starting row index.

- `int width` : Width of the ROI (columns).

- `int height` : Height of the ROI (rows).

### ROI RESIZE

```cpp
void Mat::ROI::resize_roi(int pos_x, int pos_y, int width, int height);
```

**Description**: Resizes the ROI to the specified position and size.

**Parameters**:

- `int pos_x` : Starting column index.

- `int pos_y` : Starting row index.

- `int width` : Width of the ROI (columns).

- `int height` : Height of the ROI (rows).

**Returns**: void

### AREA ROI

```cpp
int Mat::ROI::area_roi(void) const;
```

**Description**: Calculates the area of the ROI.

**Parameters**: void

**Returns**: int - Area of the ROI.

## PRINT FUNCTION

### Print matrix information

```cpp
void print_info() const;
```

**Description** : Prints the matrix information including number of rows, columns, elements, paddings, stride, memory size (size of float), data buffer address, temporary buffer address, indicators whether the matrix uses an external buffer, and whether it is a sub-matrix.

**Parameters**: void

**Returns**: void

### Print matrix elements

```cpp
void Mat::print_matrix(bool show_padding);
```

**Description**: Prints the matrix elements. If `show_padding` is true, it will also print the padding values.

**Parameters**: 

- `bool show_padding` - If true, show padding values.

**Returns**: void

## CONSTRUCTORS & DESTRUCTOR

### Default Constructor

```cpp
Mat::Mat();
```

**Description**: Default constructor initializes the matrix with default values. This function will create a matrix with only one row and one column, and the only element is set to 0.

**Parameters**: void

### Constructor - Mat(int rows, int cols)

```cpp
Mat::Mat(int rows, int cols);
```

**Description**: Constructor initializes the matrix with the specified number of rows and columns.

**Parameters**:

- `int rows` : Number of rows.

- `int cols` : Number of columns.

### Constructor - Mat(int rows, int cols, int stride)

```cpp
Mat::Mat(int rows, int cols, int stride);
```

**Description**: Constructor initializes the matrix with the specified number of rows, columns, and stride.

**Parameters**:

- `int rows` : Number of rows.

- `int cols` : Number of columns.

- `int stride` : Stride.

### Constructor - Mat(float *data, int rows, int cols)

```cpp
Mat::Mat(float *data, int rows, int cols);
```

**Description**: Constructor initializes the matrix with the specified data buffer, number of rows, and columns.

**Parameters**:

- `float *data` : Pointer to the data buffer.

- `int rows` : Number of rows.

- `int cols` : Number of columns.

### Constructor - Mat(float *data, int rows, int cols, int stride)

```cpp
Mat(float *data, int rows, int cols, int stride);
```

**Description**: Constructor initializes the matrix with the specified data buffer, number of rows, columns, and stride.

**Parameters**:

- `float *data` : Pointer to the data buffer.

- `int rows` : Number of rows.

- `int cols` : Number of columns.

- `int stride` : Stride.

### Constructor - Mat(const Mat &src)

```cpp
Mat::Mat(const Mat &src);
```

**Description**: Copy constructor initializes the matrix with the specified source matrix.

**Parameters**:

- `const Mat &src` : Source matrix.

### Destructor

```cpp
Mat::~Mat();
```

**Description**: Destructor releases the allocated memory for the matrix.

**Parameters**: void

!!! note
    For constructor functions, it must has the same name as the class name, and it must not have a return type. As shown, for C++, the function name can be reloaded by changing the number and order of the parameters as long as the permutation of the parameters is different. The destructor will be automatically called when the object goes out of scope.

## ELEMENT ACCESS

### Access matrix elements - non const

```cpp
inline float &operator()(int row, int col);
```

**Description**: Accesses the matrix elements using the specified row and column indices.

**Parameters**：

- `int row` : Row index.

- `int col` : Column index.

### Access matrix elements - const

```cpp
inline const float &operator()(int row, int col) const;
```

**Description**: Accesses the matrix elements using the specified row and column indices (const version).

**Parameters**：

- `int row` : Row index.

- `int col` : Column index.

!!! note
    These two functions are in fact redefining the `()` operator, which allows you to access the elements of the matrix using the syntax `matrix(row, col)`.

## DATA MANIPULATION

### Copy other matrix into this matrix as a sub-matrix
```cpp
tiny_error_t Mat::copy_paste(const Mat &src, int row_pos, int col_pos);
```

**Description**: Copies the specified source matrix into this matrix as a sub-matrix starting from the specified row and column positions, not sharing the data buffer.

**Parameters**:

- `const Mat &src` : Source matrix.

- `int row_pos` : Starting row position.

- `int col_pos` : Starting column position.

***Returns**: tiny_error_t - Error code.

### Copy header of other matrix to this matrix
```cpp
tiny_error_t Mat::copy_head(const Mat &src);
```

**Description**: Copies the header of the specified source matrix to this matrix, sharing the data buffer. All items copy the source matrix.

**Parameters**:

- `const Mat &src` : Source matrix.

**Returns**: tiny_error_t - Error code.

### Get a view (shallow copy) of sub-matrix (ROI) from this matrix
```cpp
Mat Mat::view_roi(int start_row, int start_col, int roi_rows, int roi_cols) const;
```

**Description**: Gets a view (shallow copy) of the sub-matrix (ROI) from this matrix starting from the specified row and column positions.

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

**Description**: Gets a view (shallow copy) of the sub-matrix (ROI) from this matrix using the specified ROI structure. This function will call the previous function in low level by passing the ROI structure to the parameters.

**Parameters**:

- `const Mat::ROI &roi` : ROI structure.

### Get a replica (deep copy) of sub-matrix (ROI)
```cpp
Mat Mat::copy_roi(int start_row, int start_col, int roi_rows, int roi_cols);
```

**Description**: Gets a replica (deep copy) of the sub-matrix (ROI) from this matrix starting from the specified row and column positions. This function will return a new matrix object that does not share the data buffer with the original matrix.

**Parameters**:

- `int start_row` : Starting row position.

- `int start_col` : Starting column position.

- `int roi_rows` : Number of rows in the ROI.

- `int roi_cols` : Number of columns in the ROI.

### Get a replica (deep copy) of sub-matrix (ROI) using ROI structure
```cpp
Mat Mat::copy_roi(const Mat::ROI &roi);
```

**Description**: Gets a replica (deep copy) of the sub-matrix (ROI) from this matrix using the specified ROI structure. This function will call the previous function in low level by passing the ROI structure to the parameters.

**Parameters**:

- `const Mat::ROI &roi` : ROI structure.

### Get a block of matrix
```cpp
Mat Mat::block(int start_row, int start_col, int block_rows, int block_cols);
```

**Description**: Gets a block of the matrix starting from the specified row and column positions.

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

**Description**: Swaps the specified rows in the matrix.

**Parameters**:

- `int row1` : First row index.

- `int row2` : Second row index.

**Returns**: void

### Clear matrix

```cpp
void Mat::clear(void);
```

**Description**: Clears the matrix by setting all elements to zero.

**Parameters**: void

**Returns**: void

## ARITHMETIC OPERATORS

!!! note
    This section defines the arithmetic operators that act on the current matrix itself. The operators are overloaded to perform matrix operations.

### Copy assignment
```cpp
Mat &operator=(const Mat &src);
```

**Description**: Copy assignment operator for the matrix.

**Parameters**:

- `const Mat &src` : Source matrix.

### Add matrix
```cpp
Mat &operator+=(const Mat &A);
```

**Description**: Adds the specified matrix to this matrix.

**Parameters**:

- `const Mat &A` : Matrix to be added.

### Add constant
```cpp
Mat &operator+=(float C);
```

### Sbtract matrix
```cpp
Mat &operator-=(const Mat &A);
```

**Description**: Subtracts the specified matrix from this matrix.

**Parameters**:

- `const Mat &A` : Matrix to be subtracted.

### Subtract constant
```cpp
Mat &operator-=(float C);
```

**Description**: Subtracts the specified constant from this matrix.

***Parameters**:

- `float C` : Constant to be subtracted.

### Multiply matrix
```cpp
Mat &operator*=(const Mat &A);
```

**Description**: Multiplies this matrix by the specified matrix.

**Parameters**:

- `const Mat &A` : Matrix to be multiplied.

### Multiply constant
```cpp
Mat &operator*=(float C);
```

**Description**: Multiplies this matrix by the specified constant.

**Parameters**:

- `float C` : Constant to be multiplied.

### Divide matrix (element-wise)
```cpp
Mat &operator/=(const Mat &B);
```

**Description**: Divides this matrix by the specified matrix element-wise.

**Parameters**:

- `const Mat &B` : Matrix to be divided by.

### Divide constant
```cpp
Mat &operator/=(float C);
```

**Description**: Divides this matrix by the specified constant.

**Parameters**:

- `float C` : Constant to be divided by.

### Exponentiation
```cpp
Mat operator^(int C);
```

**Description**: Raises this matrix to the specified power.

**Parameters**:

- `int C` : Exponent.


## LINEAR ALGEBRA

### Transpose

```cpp
Mat::transpose();
```

**Description**: Calculates the transpose of the matrix, returning a new matrix.


**Parameters**: None.

### cofactor 

```cpp
Mat::cofactor(int row, int col);
```

**Description**: Extracts the cofactor matrix from the specified row and column.

**Parameters**: 

- `int row`: Number of the row to be excluded.

- `int col`: Number of the column to be excluded.

### Determinant

```cpp
float Mat::determinant();
```

**Description**: Calculates the determinant of the matrix. It is based on cofactor and adjoint matrices.

**Parameters**: None.

**Returns**: float - Determinant value.

### Adjoint

```cpp
Mat::adjoint();
```

**Description**: Calculates the adjoint of the matrix.

**Parameters**: None.

**Returns**: Mat - Adjoint matrix.

### Normalize

```cpp
void Mat::normalize();
```

**Description**: Normalizes the matrix.

**Parameters**: None.

**Returns**: void

### Norm

```cpp
float Mat::norm() const;
```

**Description**: Calculates the norm of the matrix.

**Parameters**: None.

**Returns**: float - Norm value.

### Inverse using Adjoint

```cpp
Mat::inverse_adjoint();
```

**Description**: Calculates the inverse of the matrix using the adjoint method.

**Parameters**: None.

**Returns**: Mat - Inverse matrix.

### Identity matrix

```cpp
static Mat::eye(int size);
```

**Description**: Creates an identity matrix of the specified size.

**Parameters**: 

- `int size` : Size of the identity matrix.

**Returns**: Mat - Identity matrix.


### Augmentation Matrix

```cpp
static Mat::augment(const Mat &A, const Mat &B);
```

**Description**: Creates an augmented matrix by combining two matrices.

**Parameters**:

- `const Mat &A` : First matrix.

### identity matrix

```cpp
static Mat::ones(int rows, int cols);
```

**Description**: Creates a matrix filled with ones of the specified size.

**Parameters**:

- `int rows` : Number of rows.

- `int cols` : Number of columns.

**Returns**: Mat - Matrix filled with ones.

### All-Ones matrix

```cpp
static Mat::ones(int rows, int cols);
```

**Description**: Creates a matrix filled with ones of the specified size.

**Parameters**:

- `int rows` : Number of rows.

- `int cols` : Number of columns.

**Returns**: Mat - Matrix filled with ones.


### All-Ones matrix

```cpp
static Mat::ones(int size);
```

**Description**: Creates a square matrix filled with ones of the specified size.

**Parameters**:

- `int size` : Size of the square matrix.

**Returns**: Mat - Square matrix filled with ones.


### Gaussian Elimination

```cpp
Mat::gaussian_eliminate() const;
```

**Description**: Performs Gaussian elimination on the matrix.


**Parameters**: None.

### row reduce from Gaussian elimination

```cpp
Mat::row_reduce_from_gaussian();
```

**Description**: Performs row reduction from Gaussian elimination on the matrix.

**Parameters**: None.

**Returns**: Mat - Row reduced matrix.

### Inverse using Gaussian-Jordan elimination

```cpp
Mat::inverse_gje();
```

**Description**: Calculates the inverse of the matrix using Gaussian-Jordan elimination.

**Parameters**: None.

**Returns**: Mat - Inverse matrix.

### Dot Product

```cpp
float Mat::dotprod(const Mat &A, const Mat &B);
```

**Description**: Calculates the dot product of two matrices.

**Parameters**:

- `const Mat &A` : First matrix.

- `const Mat &B` : Second matrix.

**Returns**: float - Dot product value.

### Solve Linear System

```cpp
Mat Mat::solve(const Mat &A, const Mat &b);
```

**Description**: Solves the linear system Ax = b.

**Parameters**:

- `const Mat &A` : Coefficient matrix.

- `const Mat &b` : Right-hand side matrix.

**Returns**: Mat - Solution matrix.

### Band Solve

```cpp
Mat Mat::band_solve(Mat A, Mat b, int k);
```

**Description**: Solves a banded linear system.

**Parameters**:

- `Mat A` : Coefficient matrix.

- `Mat b` : Right-hand side matrix.


### Band Solve

```cpp
Mat Mat::band_solve(Mat A, Mat b, int k);
```

**Description**: Solves a banded linear system.

**Parameters**:

- `Mat A` : Coefficient matrix.

- `Mat b` : Right-hand side matrix.

- `int k` : Bandwidth.

**Returns**: Mat - Solution matrix.



### Roots

```cpp
Mat Mat::roots(Mat A, Mat y);
```

**Description**: Calculates the roots of a polynomial represented by the matrix A.

**Parameters**:

- `Mat A` : Coefficient matrix.

- `Mat y` : Right-hand side matrix.


**Returns**: Mat - Roots matrix.



## LINEAR ALGEBRA - Eigenvalues & Eigenvectors

### Struct: `Mat::EigenPair`

```cpp
Mat::EigenPair::EigenPair();
// fields:
// float eigenvalue;      // dominant (largest-magnitude) eigenvalue
// Mat eigenvector;       // corresponding eigenvector (n x 1)
// int iterations;        // number of iterations (for iterative methods)
// tiny_error_t status;   // computation status (TINY_OK / error code)
```

**Description**: Container for a single eigenvalue/eigenvector result and related metadata. Typically returned by `power_iteration`.

### Struct: `Mat::EigenDecomposition`

```cpp
Mat::EigenDecomposition::EigenDecomposition();
// fields:
// Mat eigenvalues;    // n x 1 matrix storing eigenvalues
// Mat eigenvectors;   // n x n matrix, columns are eigenvectors
// int iterations;     // iterations used by the algorithm
// tiny_error_t status; // computation status
```

**Description**: Container for a full eigendecomposition result (all eigenvalues and eigenvectors).

### Check Symmetry

```cpp
bool Mat::is_symmetric(float tolerance) const;
```

**Description**: Check whether a matrix is symmetric within the given `tolerance` (|A(i,j)-A(j,i)| < tolerance).

**Parameters**:
- `float tolerance` : Maximum allowed difference (e.g. 1e-6).

**Returns**: `true` if approximately symmetric, otherwise `false`.

### Power Iteration (dominant eigenpair)

```cpp
Mat::EigenPair Mat::power_iteration(int max_iter, float tolerance) const;
```

**Description**: Compute the dominant (largest-magnitude) eigenvalue and its eigenvector using the power iteration method.

**Parameters**:
- `int max_iter` : Maximum number of iterations (typical default: 1000).
- `float tolerance` : Convergence tolerance (e.g. 1e-6).

**Returns**: `EigenPair` containing `eigenvalue`, `eigenvector`, `iterations`, and `status`.

**Notes**:
- Requires a square matrix and non-null data pointer; returns an error status otherwise.
- Power iteration only returns the dominant eigenpair. For full spectrum, use eigendecomposition functions below.

### Jacobi Eigendecomposition (symmetric matrices)

```cpp
Mat::EigenDecomposition Mat::eigendecompose_jacobi(float tolerance, int max_iter) const;
```

**Description**: Compute full eigendecomposition using the Jacobi method. Recommended for symmetric matrices (good accuracy and stability for structural dynamics applications).

**Parameters**:
- `float tolerance` : Convergence threshold (e.g. 1e-6).
- `int max_iter` : Maximum iterations (e.g. 100).

**Returns**: `EigenDecomposition` with `eigenvalues`, `eigenvectors`, `iterations`, and `status`.

**Notes**: If the matrix is not approximately symmetric the function will warn, though it may still run. For non-symmetric matrices prefer the QR method.

### QR Eigendecomposition (general matrices)

```cpp
Mat::EigenDecomposition Mat::eigendecompose_qr(int max_iter, float tolerance) const;
```

**Description**: Compute eigendecomposition using the QR algorithm. Works for general (possibly non-symmetric) matrices. Complex eigenvalues may arise; current implementation returns real parts.

**Parameters**:
- `int max_iter` : Maximum number of QR iterations (default example: 100).
- `float tolerance` : Convergence tolerance (e.g. 1e-6).

**Returns**: `EigenDecomposition` containing eigenvalues, eigenvectors, iterations and status.

**Notes**: QR uses Gram–Schmidt for Q/R in this implementation; it can be less stable for ill-conditioned matrices. For symmetric matrices, Jacobi is preferred.

### Automatic Eigendecomposition

```cpp
Mat::EigenDecomposition Mat::eigendecompose(float tolerance) const;
```

**Description**: Convenience interface that automatically selects the algorithm: it tests symmetry with `is_symmetric(tolerance * 10.0f)`. If approximately symmetric, it uses Jacobi; otherwise it runs QR.

**Parameters**:
- `float tolerance` : Used for symmetry test and decomposition convergence (recommended 1e-6).

**Returns**: `EigenDecomposition`.

**Usage Tips**:
- If the matrix is known to be symmetric (e.g. stiffness or mass matrices), call `eigendecompose_jacobi` for best stability.
- For general matrices or unknown symmetry use `eigendecompose`.
- Eigendecomposition is computationally expensive for large matrices on embedded platforms; consider reduced-order or iterative methods when possible.

## STREAM OPERATORS

### Matrix output stream operator
```cpp
std::ostream &operator<<(std::ostream &os, const Mat &m);
```

**Description**: Overloaded output stream operator for the matrix.

**Parameters**:

- `std::ostream &os` : Output stream.

- `const Mat &m` : Matrix to be output.

### ROI output stream operator
```cpp
std::ostream &operator<<(std::ostream &os, const Mat::ROI &roi);
```

**Description**: Overloaded output stream operator for the ROI structure.

**Parameters**:

- `std::ostream &os` : Output stream.

- `const Mat::ROI &roi` : ROI structure.

### Matrix input stream operator
```cpp
std::istream &operator>>(std::istream &is, Mat &m);
```

**Description**: Overloaded input stream operator for the matrix.

**Parameters**:

- `std::istream &is` : Input stream.

- `Mat &m` : Matrix to be input.

!!! tip 
    This section is actually kind of overlapping with print function in terms of showing the matrix.

## GLOBAL ARITHMETIC OPERATORS

!!! tip
    The operators in this section return a new matrix object, which is the result of the operation. The original matrices remain unchanged. Unlike the previous section, the operators are designed to perform operation acting on the current matrix itself.


### Add matrix
```cpp
Mat operator+(const Mat &A, const Mat &B);
```

**Description**: Adds the specified matrices.

**Parameters**:

- `const Mat &A` : First matrix.

- `const Mat &B` : Second matrix.

### Add constant
```cpp
Mat operator+(const Mat &A, float C);
```

**Description**: Adds the specified constant to the matrix.

**Parameters**:

- `const Mat &A` : Matrix to be added.

- `float C` : Constant to be added.

### Subtract matrix
```cpp
Mat operator-(const Mat &A, const Mat &B);
```

**Description**: Subtracts the specified matrices.

**Parameters**:

- `const Mat &A` : First matrix.

- `const Mat &B` : Second matrix.

### Subtract constant
```cpp
Mat operator-(const Mat &A, float C);
```

**Description**: Subtracts the specified constant from the matrix.

**Parameters**:

- `const Mat &A` : Matrix to be subtracted.

- `float C` : Constant to be subtracted.

### Multiply matrix
```cpp
Mat operator*(const Mat &A, const Mat &B);
```

**Description**: Multiplies the specified matrices.

**Parameters**:

- `const Mat &A` : First matrix.

- `const Mat &B` : Second matrix.

### Multiply constant
```cpp
Mat operator*(const Mat &A, float C);
```

**Description**: Multiplies the specified matrix by the constant.

**Parameters**:

- `const Mat &A` : Matrix to be multiplied.

- `float C` : Constant to be multiplied.

### Multiply constant
```cpp
Mat operator*(float C, const Mat &A);
```

**Description**: Multiplies the specified matrix by the constant.

**Parameters**:

- `float C` : Constant to be multiplied.

- `const Mat &A` : Matrix to be multiplied.

### Multiply constant with constant on the left

```cpp
Mat operator*(float C, const Mat &A);
```

**Description**: Multiplies the specified matrix by the constant.

**Parameters**:

- `float C` : Constant to be multiplied.

- `const Mat &A` : Matrix to be multiplied.


### Divide matrix (by constant)
```cpp
Mat operator/(const Mat &A, float C);
```

**Description**: Divides the specified matrix by the constant element-wise.

**Parameters**:

- `const Mat &A` : Matrix to be divided.

- `float C` : Constant to divide by.

### Divide matrix (element-wise)
```cpp
Mat operator/(const Mat &A, const Mat &B);
```

**Description**: Divides the specified matrices element-wise.

**Parameters**:

- `const Mat &A` : First matrix.

- `const Mat &B` : Second matrix.

### Equality check
```cpp
bool operator==(const Mat &A, const Mat &B);
```

**Description**: Checks if the specified matrices are equal.

**Parameters**:

- `const Mat &A` : First matrix.

- `const Mat &B` : Second matrix.

**Returns**: bool - true if equal, false otherwise.

