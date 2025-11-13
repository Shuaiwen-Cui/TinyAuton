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

### Swap columns

```cpp
void Mat::swap_cols(int col1, int col2);
```

**Description**: Swaps the specified columns in the matrix. Useful for column pivoting in algorithms like Gaussian elimination with column pivoting.

**Parameters**:

- `int col1` : First column index.

- `int col2` : Second column index.

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

**Description**: Element-wise addition of a constant to this matrix.

**Parameters**:

- `float C` : The constant to add.

**Returns**: Mat& - Reference to the current matrix.

### Subtract matrix
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

**Description**: Element-wise subtraction of a constant from this matrix.

**Parameters**:

- `float C` : The constant to subtract.

**Returns**: Mat& - Reference to the current matrix.

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

**Description**: Element-wise multiplication by a constant.

**Parameters**:

- `float C` : The constant multiplier.

**Returns**: Mat& - Reference to the current matrix.

### Divide matrix (element-wise)
```cpp
Mat &operator/=(const Mat &B);
```

**Description**: Element-wise division: this = this / B.

**Parameters**:

- `const Mat &B` : The matrix divisor.

**Returns**: Mat& - Reference to the current matrix.

### Divide constant
```cpp
Mat &operator/=(float C);
```

**Description**: Element-wise division of this matrix by a constant.

**Parameters**:

- `float C` : The constant divisor.

**Returns**: Mat& - Reference to the current matrix.

### Exponentiation
```cpp
Mat operator^(int C);
```

**Description**: Element-wise integer exponentiation. Returns a new matrix where each element is raised to the given power.

**Parameters**:

- `int C` : The exponent (integer).

**Returns**: Mat - New matrix after exponentiation.


## LINEAR ALGEBRA

### Transpose

```cpp
Mat Mat::transpose();
```

**Description**: Calculates the transpose of the matrix, returning a new matrix.

**Parameters**: None.

**Returns**: Mat - Transposed matrix.

### Minor matrix

```cpp
Mat Mat::minor(int row, int col);
```

**Description**: Calculates the minor matrix by removing the specified row and column. The minor is the submatrix obtained by removing one row and one column.

**Parameters**: 

- `int row`: Row index to remove.

- `int col`: Column index to remove.

**Returns**: Mat - The (n-1)x(n-1) minor matrix.

### Cofactor matrix

```cpp
Mat Mat::cofactor(int row, int col);
```

**Description**: Calculates the cofactor matrix (same as minor matrix). The cofactor matrix is the same as the minor matrix. The sign (-1)^(i+j) is applied when computing the cofactor value, not to the matrix elements themselves.

**Parameters**: 

- `int row`: Row index to remove.

- `int col`: Column index to remove.

**Returns**: Mat - The (n-1)x(n-1) cofactor matrix (same as minor matrix).

### Determinant

```cpp
float Mat::determinant();
```

**Description**: Calculates the determinant of a square matrix using Laplace Expansion. Low efficiency, only suitable for small matrices!!!

**Parameters**: None.

**Returns**: float - Determinant value.

### Adjoint

```cpp
Mat Mat::adjoint();
```

**Description**: Calculates the adjoint (adjugate) matrix of a square matrix.

**Parameters**: None.

**Returns**: Mat - Adjoint matrix.

### Normalize

```cpp
void Mat::normalize();
```

**Description**: Normalizes the matrix using L2 norm (Frobenius norm). After normalization, ||Matrix|| = 1.

**Parameters**: None.

**Returns**: void

### Norm

```cpp
float Mat::norm() const;
```

**Description**: Calculates the Frobenius norm (L2 norm) of the matrix.

**Parameters**: None.

**Returns**: float - The computed matrix norm.

### Inverse using Adjoint

```cpp
Mat Mat::inverse_adjoint();
```

**Description**: Computes the inverse of a square matrix using adjoint method. If the matrix is singular, returns a zero matrix.

**Parameters**: None.

**Returns**: Mat - The inverse matrix. If singular, returns a zero matrix.

### Identity Matrix

```cpp
static Mat Mat::eye(int size);
```

**Description**: Generates an identity matrix of given size.

**Parameters**: 

- `int size` : Dimension of the square identity matrix.

**Returns**: Mat - Identity matrix (size x size).


### Augmentation Matrix (Horizontal Concatenation)

```cpp
static Mat Mat::augment(const Mat &A, const Mat &B);
```

**Description**: Creates an augmented matrix by horizontally concatenating two matrices [A | B]. The row counts of A and B must match.

**Parameters**:

- `const Mat &A` : Left matrix.

- `const Mat &B` : Right matrix.

**Returns**: Mat - Augmented matrix [A B].

### Vertical Stack

```cpp
static Mat Mat::vstack(const Mat &A, const Mat &B);
```

**Description**: Vertically stacks two matrices [A; B]. The column counts of A and B must match.

**Parameters**:

- `const Mat &A` : Top matrix.

- `const Mat &B` : Bottom matrix.

**Returns**: Mat - Vertically stacked matrix [A; B].

### All-Ones Matrix (Rectangular)

```cpp
static Mat Mat::ones(int rows, int cols);
```

**Description**: Creates a matrix of specified size filled with ones.

**Parameters**:

- `int rows` : Number of rows.

- `int cols` : Number of columns.

**Returns**: Mat - Matrix [rows x cols] with all elements = 1.

### All-Ones Matrix (Square)

```cpp
static Mat Mat::ones(int size);
```

**Description**: Creates a square matrix filled with ones of the specified size.

**Parameters**:

- `int size` : Size of the square matrix (rows = cols).

**Returns**: Mat - Square matrix [size x size] with all elements = 1.


### Gaussian Elimination

```cpp
Mat Mat::gaussian_eliminate() const;
```

**Description**: Performs Gaussian Elimination to convert matrix to Row Echelon Form (REF).

**Parameters**: None.

**Returns**: Mat - The upper triangular matrix (REF form).

### Row Reduce from Gaussian

```cpp
Mat Mat::row_reduce_from_gaussian();
```

**Description**: Converts a matrix (assumed in row echelon form) to Reduced Row Echelon Form (RREF).

**Parameters**: None.

**Returns**: Mat - The matrix in RREF form.

### Inverse using Gaussian-Jordan Elimination

```cpp
Mat Mat::inverse_gje();
```

**Description**: Computes the inverse of a square matrix using Gauss-Jordan elimination.

**Parameters**: None.

**Returns**: Mat - The inverse matrix if invertible, otherwise returns empty matrix.

### Dot Product

```cpp
float Mat::dotprod(const Mat &A, const Mat &B);
```

**Description**: Calculates the dot product of two vectors (Nx1).

**Parameters**:

- `const Mat &A` : Input vector A (Nx1).

- `const Mat &B` : Input vector B (Nx1).

**Returns**: float - The computed dot product value.

### Solve Linear System

```cpp
Mat Mat::solve(const Mat &A, const Mat &b);
```

**Description**: Solves the linear system Ax = b using Gaussian elimination.

**Parameters**:

- `const Mat &A` : Coefficient matrix (NxN).

- `const Mat &b` : Result vector (Nx1).

**Returns**: Mat - Solution vector (Nx1) containing the roots of the equation Ax = b.

### Band Solve

```cpp
Mat Mat::band_solve(Mat A, Mat b, int k);
```

**Description**: Solves the system of equations Ax = b using optimized Gaussian elimination for banded matrices.

**Parameters**:

- `Mat A` : Coefficient matrix (NxN) - banded matrix.

- `Mat b` : Result vector (Nx1).

- `int k` : Bandwidth of the matrix (the width of the non-zero bands).

**Returns**: Mat - Solution vector (Nx1) containing the roots of the equation Ax = b.



### Roots

```cpp
Mat Mat::roots(Mat A, Mat y);
```

**Description**: Solves the matrix using a different method. Another implementation of the 'solve' function, no difference in principle. This method solves the linear system A * x = y using Gaussian elimination.

**Parameters**:

- `Mat A` : Matrix [N]x[N] with input coefficients.

- `Mat y` : Vector [N]x[1] with result values.

**Returns**: Mat - Matrix [N]x[1] with roots.



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

**Description**: Adds two matrices element-wise.

**Parameters**:

- `const Mat &A` : First matrix.

- `const Mat &B` : Second matrix.

**Returns**: Mat - Result matrix A+B.

### Add constant
```cpp
Mat operator+(const Mat &A, float C);
```

**Description**: Adds a constant to a matrix element-wise.

**Parameters**:

- `const Mat &A` : Input matrix A.

- `float C` : Input constant.

**Returns**: Mat - Result matrix A+C.

### Subtract matrix
```cpp
Mat operator-(const Mat &A, const Mat &B);
```

**Description**: Subtracts two matrices element-wise.

**Parameters**:

- `const Mat &A` : First matrix.

- `const Mat &B` : Second matrix.

**Returns**: Mat - Result matrix A-B.

### Subtract constant
```cpp
Mat operator-(const Mat &A, float C);
```

**Description**: Subtracts a constant from a matrix element-wise.

**Parameters**:

- `const Mat &A` : Input matrix A.

- `float C` : Input constant.

**Returns**: Mat - Result matrix A-C.

### Multiply matrix
```cpp
Mat operator*(const Mat &A, const Mat &B);
```

**Description**: Multiplies two matrices (matrix multiplication).

**Parameters**:

- `const Mat &A` : First matrix.

- `const Mat &B` : Second matrix.

**Returns**: Mat - Result matrix A*B.

### Multiply constant
```cpp
Mat operator*(const Mat &A, float C);
```

**Description**: Multiplies a matrix by a constant element-wise.

**Parameters**:

- `const Mat &A` : Input matrix A.

- `float C` : Floating point value.

**Returns**: Mat - Result matrix A*C.

### Multiply constant (left side)
```cpp
Mat operator*(float C, const Mat &A);
```

**Description**: Multiplies a constant by a matrix element-wise.

**Parameters**:

- `float C` : Floating point value.

- `const Mat &A` : Input matrix A.

**Returns**: Mat - Result matrix C*A.


### Divide matrix (by constant)
```cpp
Mat operator/(const Mat &A, float C);
```

**Description**: Divides a matrix by a constant element-wise.

**Parameters**:

- `const Mat &A` : Input matrix A.

- `float C` : Floating point value.

**Returns**: Mat - Result matrix A/C.

### Divide matrix (element-wise)
```cpp
Mat operator/(const Mat &A, const Mat &B);
```

**Description**: Divides matrix A by matrix B element-wise.

**Parameters**:

- `const Mat &A` : Input matrix A.

- `const Mat &B` : Input matrix B.

**Returns**: Mat - Result matrix C, where C[i,j] = A[i,j]/B[i,j].

### Equality check
```cpp
bool operator==(const Mat &A, const Mat &B);
```

**Description**: Checks if the specified matrices are equal.

**Parameters**:

- `const Mat &A` : First matrix.

- `const Mat &B` : Second matrix.

**Returns**: bool - true if equal, false otherwise.

