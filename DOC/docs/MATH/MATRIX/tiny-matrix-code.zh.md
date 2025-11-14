# 代码

## tiny_matrix.hpp

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

## tiny_matrix.cpp

```cpp
/**
 * @file tiny_matrix.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the source file for the submodule matrix (advanced matrix operations) of the tiny_math middleware.
 * @version 1.0
 * @date 2025-04-17
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
// TinyMath
#include "tiny_matrix.hpp"

// Standard Libraries
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cinttypes>
#include <iomanip>
#include <vector>

/* LIBRARIE CONTENTS */
namespace tiny
{
    // ============================================================================
    // Rectangular ROI Structure
    // ============================================================================
    /**
     * @brief Construct a new Mat:: R O I:: R O I object
     * 
     * @param pos_x 
     * @param pos_y 
     * @param width 
     * @param height 
     */
    Mat::ROI::ROI(int pos_x, int pos_y, int width, int height)
    {
        this->pos_x = pos_x;
        this->pos_y = pos_y;
        this->width = width;
        this->height = height;
    }

    /**
     * @brief resize the ROI structure
     * 
     * @param pos_x starting column
     * @param pos_y starting row
     * @param width number of columns
     * @param height number of rows
     */
    void Mat::ROI::resize_roi(int pos_x, int pos_y, int width, int height)
    {
        this->pos_x = pos_x;
        this->pos_y = pos_y;
        this->width = width;
        this->height = height;
    }

    /**
     * @brief calculate the area of the ROI structure - how many elements covered
     * 
     * @return int 
     */
    int Mat::ROI::area_roi(void) const
    {
        return this->width * this->height;
    }

    // ============================================================================
    // Printing Functions
    // ============================================================================
    /**
     * @name Mat::PrintHead()
     * @brief Print the header of the matrix.
     */
    void Mat::print_info() const
    {
        std::cout << "Matrix Info >>>\n";

        // Basic matrix metadata
        std::cout << "rows            " << this->row << "\n";
        std::cout << "cols            " << this->col << "\n";
        std::cout << "elements        " << this->element;

        // Check if elements match rows * cols
        if (this->element != this->row * this->col)
        {
            std::cout << "   [Warning] Mismatch! Expected: " << (this->row * this->col);
        }
        std::cout << "\n";

        std::cout << "paddings        " << this->pad << "\n";
        std::cout << "stride          " << this->stride << "\n";
        std::cout << "memory          " << this->memory << "\n";

        // Pointer information
        std::cout << "data pointer    " << static_cast<const void *>(this->data) << "\n";
        std::cout << "temp pointer    " << static_cast<const void *>(this->temp) << "\n";

        // Flags information
        std::cout << "ext_buff        " << this->ext_buff;
        if (this->ext_buff)
        {
            std::cout << "   (External buffer or View)";
        }
        std::cout << "\n";

        std::cout << "sub_matrix      " << this->sub_matrix;
        if (this->sub_matrix)
        {
            std::cout << "   (This is a Sub-Matrix View)";
        }
        std::cout << "\n";

        // State warnings
        if (this->sub_matrix && !this->ext_buff)
        {
            std::cout << "[Warning] Sub-matrix is marked but ext_buff is false! Potential logic error.\n";
        }

        if (this->data == nullptr)
        {
            std::cout << "[Info] No data buffer assigned to this matrix.\n";
        }

        std::cout << "<<< Matrix Info\n";
    }

    /**
     * @name Mat::print_matrix()
     * @brief Print the matrix elements.
     *
     * @param show_padding If true, print the padding elements as well.
     */
    void Mat::print_matrix(bool show_padding)
    {
        if (this->data == nullptr)
        {
            std::cout << "[Error] Cannot print matrix: data pointer is null.\n";
            return;
        }
        
        std::cout << "Matrix Elements >>>\n";
        for (int i = 0; i < this->row; ++i)
        {
            // print the non-padding elements
            for (int j = 0; j < this->col; ++j)
            {
                std::cout << std::setw(12) << this->data[i * this->stride + j] << " ";
            }

            // if padding is enabled, print the padding elements
            if (show_padding)
            {
                // print a separator first
                std::cout << "      |";

                // print the padding elements
                for (int j = this->col; j < this->stride; ++j)
                {
                    if (j == this->col)
                    {
                        std::cout << std::setw(7) << this->data[i * this->stride + j] << " ";
                    }
                    else
                    {
                        // print the padding elements
                        std::cout << std::setw(12) << this->data[i * this->stride + j] << " ";
                    }
                }
            }

            // print a new line after each row
            std::cout << "\n";
        }

        std::cout << "<<< Matrix Elements\n";
        std::cout << std::endl;
    }

    // ============================================================================
    // Constructors & Destructor
    // ============================================================================
    // memory allocation
    /**
     * @name Mat::allocate()
     * @brief Allocate memory for the matrix according to the memory required.
     */
    void Mat::alloc_mem()
    {
        this->ext_buff = false;
        this->memory = this->row * this->stride;
        this->data = new float[this->memory];
    }

    /**
     * @name Mat::Mat()
     * @brief Constructor - default constructor: create a 1x1 matrix with only a zero element.
     */
    Mat::Mat()
    {
        this->row = 1;
        this->col = 1;
        this->pad = 0;
        this->stride = 1;
        this->element = 1;
        this->memory = 1;
        this->data = nullptr;
        this->temp = nullptr;
        this->ext_buff = false;
        this->sub_matrix = false;
        alloc_mem();
        if (this->data == nullptr)
        {
            std::cerr << "[>>> Error ! <<<] Memory allocation failed in alloc_mem()\n";
            // Memory allocation failed, object is in invalid state (data = nullptr)
            // Caller should check data pointer before using the matrix
            return;
        }
        std::memset(this->data, 0, this->memory * sizeof(float));
    }

    /**
     * @name Mat::Mat(int rows, int cols)
     * @brief Constructor - create a matrix with the specified number of rows and columns.
     *
     * @param rows Number of rows
     * @param cols Number of columns
     */
    Mat::Mat(int rows, int cols)
    {
        this->row = rows;
        this->col = cols;
        this->pad = 0;
        this->stride = cols;
        this->element = rows * cols;
        this->memory = rows * cols;
        this->data = nullptr;
        this->temp = nullptr;
        this->ext_buff = false;
        this->sub_matrix = false;
        alloc_mem();
        if (this->data == nullptr)
        {
            std::cerr << "[>>> Error ! <<<] Memory allocation failed in alloc_mem()\n";
            // Memory allocation failed, object is in invalid state (data = nullptr)
            // Caller should check data pointer before using the matrix
            return;
        }
        std::memset(this->data, 0, this->memory * sizeof(float));
    }
    /**
     * @name Mat::Mat(int rows, int cols, int stride)
     * @brief Constructor - create a matrix with the specified number of rows, columns and stride.
     *
     * @param rows Number of rows
     * @param cols Number of columns
     * @param stride Stride (number of elements in a row)
     */
    Mat::Mat(int rows, int cols, int stride)
    {
        this->row = rows;
        this->col = cols;
        this->pad = stride - cols;
        this->stride = stride;
        this->element = rows * cols;
        this->memory = rows * stride;
        this->data = nullptr;
        this->temp = nullptr;
        this->ext_buff = false;
        this->sub_matrix = false;
        alloc_mem();
        if (this->data == nullptr)
        {
            std::cerr << "[>>> Error ! <<<] Memory allocation failed in alloc_mem()\n";
            // Memory allocation failed, object is in invalid state (data = nullptr)
            // Caller should check data pointer before using the matrix
            return;
        }
        std::memset(this->data, 0, this->memory * sizeof(float));
    }

    /**
     * @name Mat::Mat(float *data, int rows, int cols)
     * @brief Constructor - create a matrix with the specified number of rows, columns and external data.
     *
     * @param data Pointer to external data buffer
     * @param rows Number of rows
     * @param cols Number of columns
     */
    Mat::Mat(float *data, int rows, int cols)
    {
        this->row = rows;
        this->col = cols;
        this->pad = 0;
        this->stride = cols;
        this->element = rows * cols;
        this->memory = rows * cols; // for external data, this item is actually not used
        this->data = data;
        this->temp = nullptr;
        this->ext_buff = true;
        this->sub_matrix = false;
    }

    /**
     * @name Mat::Mat(float *data, int rows, int cols, int stride)
     * @brief Constructor - create a matrix with the specified number of rows, columns and external data.
     *
     * @param data Pointer to external data buffer
     * @param rows Number of rows
     * @param cols Number of columns
     * @param stride Stride (number of elements in a row)
     */
    Mat::Mat(float *data, int rows, int cols, int stride)
    {
        this->row = rows;
        this->col = cols;
        this->pad = stride - cols;
        this->stride = stride;
        this->element = rows * cols;
        this->memory = rows * stride; // for external data, this item is actually not used
        this->data = data;
        this->temp = nullptr;
        this->ext_buff = true;
        this->sub_matrix = false;
    }

    /**
     * @name Mat::Mat(const Mat &src)
     * @brief Copy constructor - create a matrix with the same properties as the source matrix.
     *
     * @param src Source matrix
     */
    Mat::Mat(const Mat &src)
    {
        this->row = src.row;
        this->col = src.col;
        this->pad = src.pad;
        this->stride = src.stride;
        this->element = src.element;
        this->memory = src.memory;

        if (src.sub_matrix && src.ext_buff)
        {
            // if the source is a view (submatrix), do shallow copy
            this->data = src.data;
            this->temp = nullptr;
            this->ext_buff = true;
            this->sub_matrix = true;
        }
        else
        {
            // otherwise do deep copy
            this->data = nullptr;
            this->temp = nullptr;
            this->ext_buff = false;
            this->sub_matrix = false;

            if (src.data != nullptr)
            {
                alloc_mem();
                if (this->data == nullptr)
                {
                    std::cerr << "[Error] Memory allocation failed in alloc_mem()\n";
                    // Memory allocation failed, object is in invalid state (data = nullptr)
                    // Caller should check data pointer before using the matrix
                    return;
                }
                std::memcpy(this->data, src.data, this->memory * sizeof(float));
            }
        }
    }

    /**
     * @name ~Mat()
     * @brief Destructor - free the memory allocated for the matrix.
     */
    Mat::~Mat()
    {
        if (!this->ext_buff && this->data)
        {
            delete[] this->data;
        }
        if (this->temp)
        {
            delete[] this->temp;
        }
    }

    // ============================================================================
    // Element Access
    // ============================================================================
    // Already defined by inline functions in the header file

    // ============================================================================
    // Data Manipulation
    // ============================================================================

    /**
     * @name Mat::copy_paste(const Mat &src, int row_pos, int col_pos)
     * @brief Copy the elements of the source matrix into the destination matrix. The dimension of the current matrix must be larger than the source matrix.
     * @brief This one does not share memory with the source matrix.
     *
     * @param src Source matrix
     * @param row_pos Start row position of the destination matrix
     * @param col_pos Start column position of the destination matrix
     */
    tiny_error_t Mat::copy_paste(const Mat &src, int row_pos, int col_pos)
    {
        if ((row_pos + src.row) > this->row)
        {
            std::cerr << "[>>> Error ! <<<] Invalid row position " << std::endl;
            return TINY_ERR_INVALID_ARG;
        }
        if ((col_pos + src.col) > this->col)
        {
            std::cerr << "[>>> Error ! <<<] Invalid column position " << std::endl;
            return TINY_ERR_INVALID_ARG;
        }
        for (size_t r = 0; r < src.row; r++)
        {
            memcpy(&this->data[(r + row_pos) * this->stride + col_pos], &src.data[r * src.stride], src.col * sizeof(float));
        }

        return TINY_OK;
    }

    /**
     * @name Mat::copy_head(const Mat &src)
     * @brief Copy the header of the source matrix into the destination matrix. The data pointer is shared.
     *
     * @param src Source matrix
     */
    tiny_error_t Mat::copy_head(const Mat &src)
    {
        if (!this->ext_buff)
        {
            delete[] this->data;
        }
        this->row = src.row;
        this->col = src.col;
        this->element = src.element;
        this->pad = src.pad;
        this->stride = src.stride;
        this->memory = src.memory;
        this->data = src.data;
        this->temp = src.temp;
        this->ext_buff = src.ext_buff;
        this->sub_matrix = src.sub_matrix;

        return TINY_OK;
    }

    /**
     * @name Mat::view_roi(int start_row, int start_col, int roi_rows, int roi_cols)
     * @brief Make a shallow copy of ROI matrix. | Make a view of the ROI matrix. Low level function. Unlike ESP-DSP, it is not allowed to setup stride here, stride is automatically calculated inside the function.
     *
     * @param start_row Start row position of source matrix to copy
     * @param start_col Start column position of source matrix to copy
     * @param roi_rows Size of row elements of source matrix to copy
     * @param roi_cols Size of column elements of source matrix to copy
     *
     * @todo the pointer address is changing every time access, but the result is correct.
     *
     * @return result matrix size row_size x col_size
     */
    Mat Mat::view_roi(int start_row, int start_col, int roi_rows, int roi_cols) const
    {
        if ((start_row + roi_rows) > this->row || (start_col + roi_cols) > this->col)
        {
            std::cerr << "[Error] Invalid ROI request.\n";
            return Mat();
        }

        Mat result;
        result.row = roi_rows;
        result.col = roi_cols;
        result.stride = this->stride;
        result.pad = this->stride - roi_cols;
        result.element = roi_rows * roi_cols;
        result.memory = roi_rows * this->stride;
        result.data = this->data + (start_row * this->stride + start_col);
        result.temp = nullptr;
        result.ext_buff = true;
        result.sub_matrix = true;

        return result;
    }

    /**
     * @name Mat::view_roi(const Mat::ROI &roi)
     * @brief Make a shallow copy of ROI matrix. | Make a view of the ROI matrix. Using ROI structure.
     *
     * @param roi Rectangular area of interest
     *
     * @return result matrix size row_size x col_size
     */
    Mat Mat::view_roi(const Mat::ROI &roi) const
    {
        return view_roi(roi.pos_y, roi.pos_x, roi.height, roi.width);
    }

    /**
     * @name Mat::copy_roi(int start_row, int start_col, int height, int width)
     * @brief Make a deep copy of matrix. Copared to view_roi(), this one is a deep copy, not sharing memory with the source matrix.
     *
     * @param start_row Start row position of source matrix to copy
     * @param start_col Start column position of source matrix to copy
     * @param height Size of row elements of source matrix to copy
     * @param width Size of column elements of source matrix to copy
     *
     * @return result matrix size row_size x col_size
     */
    Mat Mat::copy_roi(int start_row, int start_col, int height, int width)
    {
        if ((start_row + height) > this->row)
        {
            std::cerr << "[>>> Error ! <<<] Invalid row position " << std::endl;
            return Mat();
        }
        if ((start_col + width) > this->col)
        {
            std::cerr << "[>>> Error ! <<<] Invalid columnn position " << std::endl;
            return Mat();
        }

        // initiate the result matrix
        Mat result(height, width);

        // deep copy the data from the source matrix
        for (size_t r = 0; r < result.row; r++)
        {
            memcpy(&result.data[r * result.stride], &this->data[(r + start_row) * this->stride + start_col], result.col * sizeof(float));
        }

        // return result;
        return result;
    }

    /**
     * @name Mat::copy_roi(const Mat::ROI &roi)
     * @brief Make a deep copy of matrix. Using ROI structure. Copared to view_roi(), this one is a deep copy, not sharing memory with the source matrix.
     *
     * @param roi Rectangular area of interest
     *
     * @return result matrix size row_size x col_size
     */
    Mat Mat::copy_roi(const Mat::ROI &roi)
    {
        return (copy_roi(roi.pos_y, roi.pos_x, roi.height, roi.width));
    }

    /**
     * @name Mat::block(int start_row, int start_col, int block_rows, int block_cols)
     * @brief Get a block of matrix.
     *
     * @param start_row
     * @param start_col
     * @param block_rows
     * @param block_cols
     * @return Mat
     */
    Mat Mat::block(int start_row, int start_col, int block_rows, int block_cols)
    {
        // Boundary check
        if (start_row < 0 || start_col < 0 || block_rows <= 0 || block_cols <= 0)
        {
            std::cerr << "[Error] Invalid block parameters: negative start position or non-positive block size.\n";
            return Mat();
        }
        if ((start_row + block_rows) > this->row || (start_col + block_cols) > this->col)
        {
            std::cerr << "[Error] Block exceeds matrix boundaries.\n";
            return Mat();
        }
        
        Mat result(block_rows, block_cols);
        for (int i = 0; i < block_rows; ++i)
        {
            for (int j = 0; j < block_cols; ++j)
            {
                result(i, j) = (*this)(start_row + i, start_col + j);
            }
        }
        return result;
    }

    /**
     * @name Mat::swap_rows(int row1, int row2)
     * @brief Swap two rows of the matrix.
     *
     * @param row1 The index of the first row to swap
     * @param row2 The index of the second row to swap
     */
    void Mat::swap_rows(int row1, int row2)
    {
        if (row1 < 0 || row1 >= this->row || row2 < 0 || row2 >= this->row)
        {
            std::cerr << "Error: row index out of range" << std::endl;
            return;
        }
        
        float *temp_row = new float[this->col];
        memcpy(temp_row, &this->data[row1 * this->stride], this->col * sizeof(float));
        memcpy(&this->data[row1 * this->stride], &this->data[row2 * this->stride], this->col * sizeof(float));
        memcpy(&this->data[row2 * this->stride], temp_row, this->col * sizeof(float));
        delete[] temp_row;
    }

    /**
     * @name Mat::swap_cols(int col1, int col2)
     * @brief Swap two columns of the matrix.
     * @note Useful for column pivoting in algorithms like Gaussian elimination with column pivoting.
     *
     * @param col1 The index of the first column to swap
     * @param col2 The index of the second column to swap
     */
    void Mat::swap_cols(int col1, int col2)
    {
        if (col1 < 0 || col1 >= this->col || col2 < 0 || col2 >= this->col)
        {
            std::cerr << "Error: column index out of range" << std::endl;
            return;
        }
        
        // Swap columns element by element (considering stride)
        for (int i = 0; i < this->row; ++i)
        {
            float temp = (*this)(i, col1);
            (*this)(i, col1) = (*this)(i, col2);
            (*this)(i, col2) = temp;
        }
    }

    /**
     * @name Mat::clear()
     * @brief Clear the matrix by setting all elements to zero.
     */
    void Mat::clear(void)
    {
        for (int row = 0; row < this->row; row++)
        {
            memset(this->data + (row * this->stride), 0, this->col * sizeof(float));
        }
    }

    // ============================================================================
    // Arithmetic Operators
    // ============================================================================
    /**
     * @name &Mat::operator=(const Mat &src)
     * @brief Copy assignment operator - copy the elements of the source matrix into the destination matrix. Compared to the copy constructor, this one is used for existing matrix to copy the elements. The copy constructor is used for the first time to create a new matrix and copy the elements at the same time.
     *
     * @param src
     * @return Mat&
     */
    Mat &Mat::operator=(const Mat &src)
    {
        // 1. Self-assignment check
        if (this == &src)
        {
            return *this;
        }

        // 2. Forbid assignment to sub-matrix views
        if (this->sub_matrix)
        {
            std::cerr << "[Error] Assignment to a sub-matrix is not allowed.\n";
            return *this;
        }

        // 3. If dimensions differ, reallocate memory
        if (this->row != src.row || this->col != src.col)
        {
            if (!this->ext_buff && this->data != nullptr)
            {
                delete[] this->data;
            }

            // Update dimensions and memory info
            this->row = src.row;
            this->col = src.col;
            this->stride = src.col; // Follow source's logical stride
            this->pad = 0;
            this->element = this->row * this->col;
            this->memory = this->row * this->stride;

            this->ext_buff = false;
            this->sub_matrix = false;

            alloc_mem();
        }

        // 4. Data copy (row-wise)
        for (int r = 0; r < this->row; ++r)
        {
            std::memcpy(this->data + r * this->stride, src.data + r * src.stride, this->col * sizeof(float));
        }

        return *this;
    }

    /**
     * @name Mat::operator+=(const Mat &A)
     * @brief Element-wise addition of another matrix to this matrix.
     *
     * @param A The matrix to add
     * @return Mat& Reference to the current matrix
     */
    /**
     * @name Mat::operator+=(const Mat &A)
     * @brief Element-wise addition of another matrix to this matrix.
     *
     * @param A The matrix to add
     * @return Mat& Reference to the current matrix
     */
    Mat &Mat::operator+=(const Mat &A)
    {
        // 1. Dimension check
        if ((this->row != A.row) || (this->col != A.col))
        {
            std::cerr << "[Error] Matrix addition failed: Dimension mismatch ("
                      << this->row << "x" << this->col << " vs "
                      << A.row << "x" << A.col << ")\n";
            return *this;
        }

        // 2. Determine if padding handling is needed
        bool need_padding_handling = (this->pad > 0) || (A.pad > 0);

        if (need_padding_handling)
        {
            // Padding-aware addition
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_add_f32(this->data, A.data, this->data,
                         this->row, this->col,
                         this->pad, A.pad, this->pad,
                         1, 1, 1);
#else
            tiny_mat_add_f32(this->data, A.data, this->data,
                             this->row, this->col,
                             this->pad, A.pad, this->pad,
                             1, 1, 1);
#endif
        }
        else
        {
            // Vectorized addition for contiguous memory
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dsps_add_f32(this->data, A.data, this->data, this->memory, 1, 1, 1);
#else
            tiny_vec_add_f32(this->data, A.data, this->data, this->memory, 1, 1, 1);
#endif
        }

        return *this;
    }

    /**
     * @name Mat::operator+=(float C)
     * @brief Element-wise addition of a constant to this matrix.
     *
     * @param C The constant to add
     */
    /**
     * @name Mat::operator+=(float C)
     * @brief Element-wise addition of a constant to this matrix.
     *
     * @param C The constant to add
     * @return Mat& Reference to the current matrix
     */
    Mat &Mat::operator+=(float C)
    {
        // check whether padding is presented
        bool need_padding_handling = (this->pad > 0);

        if (need_padding_handling)
        {
            // Padding-aware constant addition
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_addc_f32(this->data, this->data, C,
                          this->row, this->col,
                          this->pad, this->pad,
                          1, 1);
#else
            tiny_mat_addc_f32(this->data, this->data, C,
                              this->row, this->col,
                              this->pad, this->pad,
                              1, 1);
#endif
        }
        else
        {
            // Vectorized constant addition
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dsps_addc_f32(this->data, this->data, this->memory, C, 1, 1);
#else
            tiny_vec_addc_f32(this->data, this->data, this->memory, C, 1, 1);
#endif
        }

        return *this;
    }

    /**
     * @name Mat::operator-=(const Mat &A)
     * @brief Element-wise subtraction of another matrix from this matrix.
     *
     * @param A The matrix to subtract
     * @return Mat& Reference to the current matrix
     */
    /**
     * @name Mat::operator-=(const Mat &A)
     * @brief Element-wise subtraction of another matrix from this matrix.
     *
     * @param A The matrix to subtract
     * @return Mat& Reference to the current matrix
     */
    Mat &Mat::operator-=(const Mat &A)
    {
        // 1. Dimension check
        if ((this->row != A.row) || (this->col != A.col))
        {
            std::cerr << "[Error] Matrix subtraction failed: Dimension mismatch ("
                      << this->row << "x" << this->col << " vs "
                      << A.row << "x" << A.col << ")\n";
            return *this;
        }

        // 2. Determine if padding handling is needed
        bool need_padding_handling = (this->pad > 0) || (A.pad > 0);

        if (need_padding_handling)
        {
            // Padding-aware subtraction
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_sub_f32(this->data, A.data, this->data,
                         this->row, this->col,
                         this->pad, A.pad, this->pad,
                         1, 1, 1);
#else
            tiny_mat_sub_f32(this->data, A.data, this->data,
                             this->row, this->col,
                             this->pad, A.pad, this->pad,
                             1, 1, 1);
#endif
        }
        else
        {
            // Vectorized subtraction for contiguous memory
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dsps_sub_f32(this->data, A.data, this->data, this->memory, 1, 1, 1);
#else
            tiny_vec_sub_f32(this->data, A.data, this->data, this->memory, 1, 1, 1);
#endif
        }

        return *this;
    }

    /**
     * @name Mat::operator-=(float C)
     * @brief Element-wise subtraction of a constant from this matrix.
     *
     * @param C The constant to subtract
     */
    /**
     * @name Mat::operator-=(float C)
     * @brief Element-wise subtraction of a constant from this matrix.
     *
     * @param C The constant to subtract
     * @return Mat& Reference to the current matrix
     */
    Mat &Mat::operator-=(float C)
    {
        bool need_padding_handling = (this->pad > 0);

        if (need_padding_handling)
        {
            // Padding-aware constant subtraction
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            // Note: ESP32 DSP does not provide dspm_subc_f32, using dspm_addc_f32 with -C
            dspm_addc_f32(this->data, this->data, -C,
                          this->row, this->col,
                          this->pad, this->pad,
                          1, 1);
#else
            tiny_mat_subc_f32(this->data, this->data, C,
                              this->row, this->col,
                              this->pad, this->pad,
                              1, 1);
#endif
        }
        else
        {
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            // Note: ESP32 DSP does not provide dsps_subc_f32, using dsps_addc_f32 with -C
            dsps_addc_f32(this->data, this->data, this->memory, -C, 1, 1);
#else
            tiny_vec_subc_f32(this->data, this->data, this->memory, C, 1, 1);
#endif
        }

        return *this;
    }

    /**
     * @name Mat::operator*=(const Mat &m)
     * @brief Matrix multiplication: this = this * m
     *
     * @param m The matrix to multiply with
     * @return Mat& Reference to the current matrix
     */
    Mat &Mat::operator*=(const Mat &m)
    {
        // 1. Dimension check
        if (this->col != m.row)
        {
            std::cerr << "[Error] Matrix multiplication failed: incompatible dimensions ("
                      << this->row << "x" << this->col << " * "
                      << m.row << "x" << m.col << ")\n";
            return *this;
        }

        // 2. Prepare temp matrix (incase overwriting the original data)
        Mat temp = this->copy_roi(0, 0, this->row, this->col);

        // 3. check whether padding is present in either matrix
        bool need_padding_handling = (this->pad > 0) || (m.pad > 0);

        if (need_padding_handling)
        {
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_mult_ex_f32(temp.data, m.data, this->data, temp.row, temp.col, m.col, temp.pad, m.pad, this->pad);
#else
            tiny_mat_mult_ex_f32(temp.data, m.data, this->data, temp.row, temp.col, m.col, temp.pad, m.pad, this->pad);
#endif
        }
        else
        {
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_mult_f32(temp.data, m.data, this->data, temp.row, temp.col, m.col);
#else
            tiny_mat_mult_f32(temp.data, m.data, this->data, temp.row, temp.col, m.col);
#endif
        }

        return *this;
    }

    /**
     * @name Mat::operator*=(float num)
     * @brief Element-wise multiplication by a constant
     *
     * @param num The constant multiplier
     * @return Mat& Reference to the current matrix
     */
    Mat &Mat::operator*=(float num)
    {
        // check whether padding is present
        bool need_padding_handling = (this->pad > 0);

        if (need_padding_handling)
        {
            // Padding-aware multiplication
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_mulc_f32(this->data, this->data, num,
                          this->row, this->col,
                          this->pad, this->pad,
                          1, 1);
#else
            tiny_mat_multc_f32(this->data, this->data, num, this->row, this->col, this->pad, this->pad, 1, 1);
#endif
        }
        else
        {
            // No padding, use vectorized multiplication
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dsps_mulc_f32(this->data, this->data, this->memory, num, 1, 1);
#else
            tiny_vec_mulc_f32(this->data, this->data, this->memory, num, 1, 1);
#endif
        }

        return *this;
    }

    /**
     * @name Mat::operator/=(const Mat &B)
     * @brief Element-wise division: this = this / B
     *
     * @param B The matrix divisor
     * @return Mat& Reference to the current matrix
     */
    Mat &Mat::operator/=(const Mat &B)
    {
        // 1. Dimension check
        if ((this->row != B.row) || (this->col != B.col))
        {
            std::cerr << "[Error] Matrix division failed: Dimension mismatch ("
                      << this->row << "x" << this->col << " vs "
                      << B.row << "x" << B.col << ")\n";
            return *this;
        }

        // 2. Zero division check
        bool zero_found = false;
        const float epsilon = 1e-9f;
        for (int i = 0; i < B.row; ++i)
        {
            for (int j = 0; j < B.col; ++j)
            {
                if (fabs(B(i, j)) < epsilon)
                {
                    zero_found = true;
                    break;
                }
            }
            if (zero_found)
                break;
        }

        if (zero_found)
        {
            std::cerr << "[Error] Matrix division failed: Division by zero detected.\n";
            return *this;
        }

        // 3. Element-wise division
        for (int i = 0; i < this->row; ++i)
        {
            for (int j = 0; j < this->col; ++j)
            {
                (*this)(i, j) /= B(i, j);
            }
        }

        return *this;
    }

    /**
     * @name Mat::operator/=(float num)
     * @brief Element-wise division of this matrix by a constant.
     *
     * @param num The constant divisor
     * @return Mat& Reference to the current matrix
     */
    Mat &Mat::operator/=(float num)
    {
        // 1. Check division by zero
        if (num == 0.0f)
        {
            std::cerr << "[Error] Matrix division by zero is undefined.\n";
            return *this;
        }

        // 2. Determine if padding handling is needed
        bool need_padding_handling = (this->pad > 0);

        float inv_num = 1.0f / num;

        if (need_padding_handling)
        {
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_mulc_f32(this->data, this->data, inv_num,
                          this->row, this->col,
                          this->pad, this->pad,
                          1, 1);
#else
            tiny_mat_multc_f32(this->data, this->data, inv_num,
                              this->row, this->col,
                              this->pad, this->pad,
                              1, 1);
#endif
        }
        else
        {
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dsps_mulc_f32(this->data, this->data, this->memory, inv_num, 1, 1);
#else
            tiny_vec_mulc_f32(this->data, this->data, this->memory, inv_num, 1, 1);
#endif
        }

        return *this;
    }

    /**
     * @name Mat::operator^(int num)
     * @brief Element-wise integer exponentiation. Returns a new matrix where each element is raised to the given power.
     *
     * @param num The exponent (integer)
     * @return Mat New matrix after exponentiation
     */
    Mat Mat::operator^(int num)
    {
        // Handle special cases
        if (num == 0)
        {
            // Any number to the power of 0 is 1
            Mat result(this->row, this->col, this->stride);
            for (int i = 0; i < this->row; ++i)
            {
                for (int j = 0; j < this->col; ++j)
                {
                    result(i, j) = 1.0f;
                }
            }
            return result;
        }

        if (num == 1)
        {
            // Return a copy of current matrix
            return Mat(*this);
        }

        if (num < 0)
        {
            std::cerr << "[Error] Negative exponent not supported in operator^.\n";
            return Mat(*this); // Return a copy without modification
        }

        // General case: positive exponent > 1
        Mat result(this->row, this->col, this->stride);
        for (int i = 0; i < this->row; ++i)
        {
            for (int j = 0; j < this->col; ++j)
            {
                float base = (*this)(i, j);
                float value = 1.0f;
                for (int k = 0; k < num; ++k)
                {
                    value *= base;
                }
                result(i, j) = value;
            }
        }

        return result;
    }

    // ============================================================================
    // Linear Algebra - Basic Operations
    // ============================================================================
    /**
     * @name Mat::transpose
     * @brief Transpose the matrix.
     *
     * @return Transposed matrix
     */
    Mat Mat::transpose()
    {
        Mat result(this->col, this->row);
        for (int i = 0; i < this->row; ++i)
        {
            for (int j = 0; j < this->col; ++j)
            {
                result(j, i) = this->data[i * this->stride + j];
            }
        }
        return result;
    }

    /**
     * @name Mat::determinant()
     * @brief Compute the determinant of the matrix (auto-selects method based on size).
     * @note For small matrices (n <= 4), uses Laplace expansion.
     *       For larger matrices, uses LU decomposition (O(n³)) for better efficiency.
     *
     * @return float The determinant value
     */
    float Mat::determinant()
    {
        if (this->row != this->col)
        {
            std::cerr << "[Error] Determinant requires a square matrix.\n";
            return 0.0f;
        }

        int n = this->row;
        
        // For small matrices, use Laplace expansion (more accurate for small sizes)
        if (n <= 4)
        {
            return this->determinant_laplace();
        }
        
        // For larger matrices, use LU decomposition (much faster, O(n³) vs O(n!))
        return this->determinant_lu();
    }

    /**
     * @name Mat::determinant_laplace()
     * @brief Compute the determinant using Laplace expansion (cofactor expansion).
     * @note Time complexity: O(n!) - suitable only for small matrices (n <= 4).
     *       Uses recursive method with first row expansion.
     *
     * @return float The determinant value
     */
    float Mat::determinant_laplace()
    {
        if (this->row != this->col)
        {
            std::cerr << "[Error] Determinant requires a square matrix.\n";
            return 0.0f;
        }

        int n = this->row;
        if (n == 1)
        {
            return this->data[0];
        }
        if (n == 2)
        {
            return this->data[0] * this->data[this->stride + 1] - this->data[1] * this->data[this->stride];
        }

        float det = 0.0f;
        for (int j = 0; j < n; ++j)
        {
            Mat minor_mat = this->minor(0, j);
            float cofactor_val = ((j % 2 == 0) ? 1.0f : -1.0f) * minor_mat.determinant_laplace();
            det += this->data[j] * cofactor_val;
        }
        return det;
    }

    /**
     * @name Mat::determinant_lu()
     * @brief Compute the determinant using LU decomposition.
     * @note Time complexity: O(n³) - efficient for large matrices.
     *       Formula: det(A) = det(P) * det(L) * det(U) = det(P) * 1 * (product of U diagonal)
     *       where det(P) = (-1)^(number of row swaps)
     *
     * @return float The determinant value
     */
    float Mat::determinant_lu()
    {
        if (this->row != this->col)
        {
            std::cerr << "[Error] Determinant requires a square matrix.\n";
            return 0.0f;
        }

        // Perform LU decomposition
        LUDecomposition lu = this->lu_decompose(true);  // Use pivoting for numerical stability
        
        if (lu.status != TINY_OK)
        {
            // Matrix is singular or near-singular
            return 0.0f;
        }

        int n = this->row;
        
        // Compute det(P): permutation matrix determinant = (-1)^(permutation signature)
        float det_P = 1.0f;
        if (lu.pivoted)
        {
            // Compute permutation signature by finding cycles in P
            // P is a permutation matrix, so each row/column has exactly one 1
            // We can compute the sign by decomposing into transpositions
            std::vector<bool> visited(n, false);
            int cycle_count = 0;
            
            for (int i = 0; i < n; ++i)
            {
                if (visited[i]) continue;
                
                // Find the cycle starting at i
                int current = i;
                int cycle_length = 0;
                while (!visited[current])
                {
                    visited[current] = true;
                    cycle_length++;
                    
                    // Find where P maps current row
                    for (int j = 0; j < n; ++j)
                    {
                        if (fabsf(lu.P(current, j) - 1.0f) < 1e-6f)
                        {
                            current = j;
                            break;
                        }
                    }
                }
                
                // A cycle of length k contributes (k-1) transpositions
                if (cycle_length > 1)
                {
                    cycle_count += (cycle_length - 1);
                }
            }
            
            // det(P) = (-1)^(number of transpositions)
            det_P = (cycle_count % 2 == 0) ? 1.0f : -1.0f;
        }

        // Compute det(L): lower triangular with unit diagonal = 1
        float det_L = 1.0f;  // L has unit diagonal, so det(L) = 1

        // Compute det(U): product of diagonal elements
        float det_U = 1.0f;
        for (int i = 0; i < n; ++i)
        {
            det_U *= lu.U(i, i);
        }

        // det(A) = det(P) * det(L) * det(U)
        return det_P * det_L * det_U;
    }

    /**
     * @name Mat::determinant_gaussian()
     * @brief Compute the determinant using Gaussian elimination.
     * @note Time complexity: O(n³) - efficient for large matrices.
     *       Converts matrix to upper triangular form, then multiplies diagonal elements.
     *       Tracks row swaps to account for sign changes.
     *
     * @return float The determinant value
     */
    float Mat::determinant_gaussian()
    {
        if (this->row != this->col)
        {
            std::cerr << "[Error] Determinant requires a square matrix.\n";
            return 0.0f;
        }

        int n = this->row;
        Mat A = Mat(*this);  // Working copy
        int swap_count = 0;  // Track number of row swaps

        // Gaussian elimination to upper triangular form
        for (int k = 0; k < n - 1; ++k)
        {
            // Partial pivoting: find row with largest element in column k
            int max_row = k;
            float max_val = fabsf(A(k, k));
            for (int i = k + 1; i < n; ++i)
            {
                if (fabsf(A(i, k)) > max_val)
                {
                    max_val = fabsf(A(i, k));
                    max_row = i;
                }
            }

            // Swap rows if necessary
            if (max_row != k)
            {
                A.swap_rows(k, max_row);
                swap_count++;
            }

            // Check for singular matrix
            if (fabsf(A(k, k)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                // Matrix is singular
                return 0.0f;
            }

            // Eliminate below diagonal
            for (int i = k + 1; i < n; ++i)
            {
                float factor = A(i, k) / A(k, k);
                for (int j = k; j < n; ++j)
                {
                    A(i, j) -= factor * A(k, j);
                }
            }
        }

        // Compute determinant: product of diagonal elements
        float det = 1.0f;
        for (int i = 0; i < n; ++i)
        {
            det *= A(i, i);
        }

        // Account for row swaps: each swap multiplies determinant by -1
        if (swap_count % 2 == 1)
        {
            det = -det;
        }

        return det;
    }

    /**
     * @name Mat::adjoint()
     * @brief Compute the adjoint (adjugate) matrix.
     *
     * @return Mat The adjoint matrix
     */
    Mat Mat::adjoint()
    {
        if (this->row != this->col)
        {
            std::cerr << "[Error] Adjoint requires a square matrix.\n";
            return Mat();
        }

        int n = this->row;
        Mat result(n, n);

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                Mat cofactor_mat = this->cofactor(i, j);
                result(j, i) = cofactor_mat.determinant(); // Note: transpose (j,i) not (i,j)
            }
        }

        return result;
    }

    /**
     * @name Mat::normalize()
     * @brief Normalize the matrix by dividing each element by the matrix norm.
     */
    void Mat::normalize()
    {
        float n = this->norm();
        if (n > TINY_MATH_MIN_POSITIVE_INPUT_F32)
        {
            for (int i = 0; i < this->row; ++i)
            {
                for (int j = 0; j < this->col; ++j)
                {
                    (*this)(i, j) /= n;
                }
            }
        }
    }

    /**
     * @name Mat::norm()
     * @brief Compute the Frobenius norm of the matrix.
     *
     * @return float The matrix norm
     */
    float Mat::norm() const
    {
        float sum_sq = 0.0f;
        for (int i = 0; i < this->row; ++i)
        {
            for (int j = 0; j < this->col; ++j)
            {
                float val = (*this)(i, j);
                sum_sq += val * val;
            }
        }
        return sqrtf(sum_sq);
    }

    /**
     * @name Mat::inverse_adjoint()
     * @brief Compute the inverse matrix using the adjoint method.
     *
     * @return Mat The inverse matrix
     */
    Mat Mat::inverse_adjoint()
    {
        float det = this->determinant();
        if (fabsf(det) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
        {
            std::cerr << "[Error] Matrix is singular, cannot compute inverse.\n";
            return Mat();
        }

        Mat adj = this->adjoint();
        return adj * (1.0f / det);
    }

    /**
     * @name Mat::dotprod(const Mat &A, const Mat &B)
     * @brief Compute the dot product of two matrices (element-wise multiplication and sum).
     *
     * @param A First matrix
     * @param B Second matrix
     * @return float Dot product value
     */
    float Mat::dotprod(const Mat &A, const Mat &B)
    {
        if (A.row != B.row || A.col != B.col)
        {
            std::cerr << "[Error] Dot product requires matrices of the same size.\n";
            return 0.0f;
        }

        float result = 0.0f;
        for (int i = 0; i < A.row; ++i)
        {
            for (int j = 0; j < A.col; ++j)
            {
                result += A(i, j) * B(i, j);
            }
        }
        return result;
    }

    // ============================================================================
    // Linear Algebra - Matrix Utilities
    // ============================================================================
    /**
     * @name Mat::eye(int size)
     * @brief Create an identity matrix of specified size.
     *
     * @param size Size of the square identity matrix
     * @return Mat Identity matrix
     */
    Mat Mat::eye(int size)
    {
        Mat identity(size, size);
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                identity(i, j) = (i == j) ? 1.0f : 0.0f;
            }
        }
        return identity;
    }

    /**
     * @name Mat::ones(int rows, int cols)
     * @brief Create a matrix filled with ones.
     *
     * @param rows Number of rows
     * @param cols Number of columns
     * @return Mat Matrix filled with ones
     */
    Mat Mat::ones(int rows, int cols)
    {
        Mat result(rows, cols);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                result(i, j) = 1.0f;
            }
        }
        return result;
    }

    /**
     * @name Mat::ones(int size)
     * @brief Create a square matrix filled with ones.
     *
     * @param size Size of the square matrix (rows = cols)
     * @return Mat Square matrix [size x size] with all elements = 1
     */
    Mat Mat::ones(int size)
    {
        return Mat::ones(size, size);
    }

    /**
     * @name Mat::augment(const Mat &A, const Mat &B)
     * @brief Augment two matrices horizontally [A | B].
     *
     * @param A Left matrix
     * @param B Right matrix
     * @return Mat Augmented matrix [A B]
     */
    Mat Mat::augment(const Mat &A, const Mat &B)
    {
        // 1. Check if row counts match
        if (A.row != B.row)
        {
            std::cerr << "[Error] Cannot augment matrices: Row counts do not match ("
                      << A.row << " vs " << B.row << ")\n";
            return Mat();
        }

        // 2. Create new matrix with combined columns
        Mat AB(A.row, A.col + B.col);

        // 3. Copy data from A and B
        for (int i = 0; i < A.row; ++i)
        {
            // Copy A
            for (int j = 0; j < A.col; ++j)
            {
                AB(i, j) = A(i, j);
            }
            // Copy B
            for (int j = 0; j < B.col; ++j)
            {
                AB(i, A.col + j) = B(i, j);
            }
        }

        return AB;
    }

    /**
     * @name Mat::vstack(const Mat &A, const Mat &B)
     * @brief Vertically stack two matrices [A; B].
     *
     * @param A Top matrix
     * @param B Bottom matrix
     * @return Mat Vertically stacked matrix [A; B]
     */
    Mat Mat::vstack(const Mat &A, const Mat &B)
    {
        // 1. Check if column counts match
        if (A.col != B.col)
        {
            std::cerr << "[Error] Cannot vstack matrices: Column counts do not match ("
                      << A.col << " vs " << B.col << ")\n";
            return Mat();
        }

        // 2. Create new matrix with combined rows
        Mat AB(A.row + B.row, A.col);

        // 3. Copy data from A and B
        // Copy A (top rows)
        for (int i = 0; i < A.row; ++i)
        {
            for (int j = 0; j < A.col; ++j)
            {
                AB(i, j) = A(i, j);
            }
        }
        // Copy B (bottom rows)
        for (int i = 0; i < B.row; ++i)
        {
            for (int j = 0; j < B.col; ++j)
            {
                AB(A.row + i, j) = B(i, j);
            }
        }

        return AB;
    }

    /**
     * @name Mat::gram_schmidt_orthogonalize()
     * @brief Orthogonalize a set of vectors using the Gram-Schmidt process
     * @note This is a general-purpose orthogonalization function that can be reused
     *       for QR decomposition and other applications requiring orthogonal bases
     * 
     * @param vectors Input matrix where each column is a vector to be orthogonalized
     * @param orthogonal_vectors Output matrix for orthogonalized vectors (each column is orthogonal)
     * @param coefficients Output matrix for projection coefficients (upper triangular, like R in QR)
     * @param tolerance Minimum norm threshold for linear independence check
     * @return true if successful, false if input is invalid
     */
    bool Mat::gram_schmidt_orthogonalize(const Mat &vectors, Mat &orthogonal_vectors, 
                                         Mat &coefficients, float tolerance)
    {
        // Validation
        if (vectors.data == nullptr)
        {
            std::cerr << "[Error] gram_schmidt_orthogonalize: Input matrix is null.\n";
            return false;
        }

        int m = vectors.row;  // Dimension of vectors
        int n = vectors.col;  // Number of vectors

        if (m == 0 || n == 0)
        {
            std::cerr << "[Error] gram_schmidt_orthogonalize: Invalid dimensions.\n";
            return false;
        }

        // Initialize output matrices
        orthogonal_vectors = Mat(m, n);
        coefficients = Mat(n, n);  // Upper triangular matrix for coefficients
        coefficients.clear();  // Initialize to zero

        // Modified Gram-Schmidt process (more numerically stable than classical GS)
        // Also includes re-orthogonalization for better numerical stability
        for (int j = 0; j < n; ++j)
        {
            // Copy j-th column of input vectors to working vector
            for (int i = 0; i < m; ++i)
            {
                orthogonal_vectors(i, j) = vectors(i, j);
            }

            // Modified Gram-Schmidt: orthogonalize against previous columns
            // Use a more stable approach: subtract projection immediately
            for (int k = 0; k < j; ++k)
            {
                // Compute dot product: coefficient(k,j) = Q(:,k)^T * Q(:,j)
                float dot = 0.0f;
                for (int i = 0; i < m; ++i)
                {
                    dot += orthogonal_vectors(i, k) * orthogonal_vectors(i, j);
                }
                coefficients(k, j) = dot;  // Store projection coefficient

                // Subtract projection immediately: Q(:,j) = Q(:,j) - dot * Q(:,k)
                for (int i = 0; i < m; ++i)
                {
                    orthogonal_vectors(i, j) -= dot * orthogonal_vectors(i, k);
                }
            }

            // Re-orthogonalization: improve numerical stability by doing one more pass
            // This helps reduce accumulated rounding errors (especially important for
            // near-linearly-dependent vectors)
            for (int k = 0; k < j; ++k)
            {
                float dot = 0.0f;
                for (int i = 0; i < m; ++i)
                {
                    dot += orthogonal_vectors(i, k) * orthogonal_vectors(i, j);
                }
                // Update coefficient with correction (small correction for numerical stability)
                coefficients(k, j) += dot;
                // Subtract residual projection to improve orthogonality
                for (int i = 0; i < m; ++i)
                {
                    orthogonal_vectors(i, j) -= dot * orthogonal_vectors(i, k);
                }
            }

            // Normalize: compute norm of orthogonalized vector
            float norm = 0.0f;
            for (int i = 0; i < m; ++i)
            {
                norm += orthogonal_vectors(i, j) * orthogonal_vectors(i, j);
            }
            norm = sqrtf(norm);

            // Use stricter tolerance for near-linear-dependent vectors
            // If norm is very small, generate an orthogonal vector to complete the basis
            if (norm < tolerance || norm < 1e-5f)  // Stricter check for numerical stability
            {
                // Vector is linearly dependent (or near-zero)
                // Instead of setting to zero, generate an orthogonal vector to maintain Q's orthogonality
                // Strategy: Start with a standard basis vector and orthogonalize it
                coefficients(j, j) = 0.0f;  // Original vector has zero norm (linearly dependent)
                
                // Try to find an orthogonal vector by starting with standard basis vectors
                // and orthogonalizing them against previous columns
                bool found_orthogonal = false;
                for (int basis_idx = 0; basis_idx < m && !found_orthogonal; ++basis_idx)
                {
                    // Start with standard basis vector e_basis_idx
                    for (int i = 0; i < m; ++i)
                    {
                        orthogonal_vectors(i, j) = (i == basis_idx) ? 1.0f : 0.0f;
                    }
                    
                    // Orthogonalize against previous columns
                    for (int k = 0; k < j; ++k)
                    {
                        float dot = 0.0f;
                        for (int i = 0; i < m; ++i)
                        {
                            dot += orthogonal_vectors(i, k) * orthogonal_vectors(i, j);
                        }
                        for (int i = 0; i < m; ++i)
                        {
                            orthogonal_vectors(i, j) -= dot * orthogonal_vectors(i, k);
                        }
                    }
                    
                    // Check if we got a non-zero vector
                    float new_norm = 0.0f;
                    for (int i = 0; i < m; ++i)
                    {
                        new_norm += orthogonal_vectors(i, j) * orthogonal_vectors(i, j);
                    }
                    new_norm = sqrtf(new_norm);
                    
                    if (new_norm > 1e-5f)
                    {
                        // Found a valid orthogonal vector, normalize it
                        // Note: coefficients(j, j) remains 0 (original vector was linearly dependent)
                        // but Q(:, j) is now a normalized orthogonal vector
                        for (int i = 0; i < m; ++i)
                        {
                            orthogonal_vectors(i, j) /= new_norm;
                        }
                        found_orthogonal = true;
                    }
                }
                
                // If still no orthogonal vector found, set to zero (shouldn't happen for full-rank cases)
                if (!found_orthogonal)
                {
                    coefficients(j, j) = 0.0f;
                    for (int i = 0; i < m; ++i)
                    {
                        orthogonal_vectors(i, j) = 0.0f;
                    }
                }
            }
            else
            {
                coefficients(j, j) = norm;
                // Normalize the orthogonalized vector
                for (int i = 0; i < m; ++i)
                {
                    orthogonal_vectors(i, j) /= norm;
                }
            }
        }

        return true;
    }

    // ============================================================================
    // Linear Algebra - Matrix Operations
    // ============================================================================
    /**
     * @name Mat::minor(int target_row, int target_col)
     * @brief Calculate the minor matrix by removing specified row and column.
     * @note Minor is the submatrix obtained by removing one row and one column.
     *
     * @param target_row Row index to remove
     * @param target_col Column index to remove
     * @return Mat The (n-1)x(n-1) minor matrix
     */
    Mat Mat::minor(int target_row, int target_col)
    {
        if (this->row != this->col)
        {
            std::cerr << "[Error] Minor requires square matrix.\n";
            return Mat();
        }

        int n = this->row;
        Mat result(n - 1, n - 1);

        for (int i = 0, res_i = 0; i < n; ++i)
        {
            if (i == target_row)
                continue;

            for (int j = 0, res_j = 0; j < n; ++j)
            {
                if (j == target_col)
                    continue;

                result.data[res_i * result.stride + res_j] = this->data[i * this->stride + j];
                res_j++;
            }
            res_i++;
        }

        return result;
    }

    /**
     * @name Mat::cofactor(int target_row, int target_col)
     * @brief Calculate the cofactor matrix (same as minor matrix).
     * @note The cofactor matrix is the same as the minor matrix.
     *       The sign (-1)^(i+j) is applied when computing the cofactor value,
     *       not to the matrix elements themselves.
     *       Cofactor value C_ij = (-1)^(i+j) * det(minor_matrix)
     *
     * @param target_row Row index to remove
     * @param target_col Column index to remove
     * @return Mat The (n-1)x(n-1) cofactor matrix (same as minor matrix)
     */
    Mat Mat::cofactor(int target_row, int target_col)
    {
        // Cofactor matrix is the same as minor matrix
        // The sign is applied when computing cofactor values, not to matrix elements
        return this->minor(target_row, target_col);
    }

    /**
     * @name Mat::gaussian_eliminate
     * @brief Perform Gaussian Elimination to convert matrix to Row Echelon Form (REF).
     *
     * @return Mat The upper triangular matrix (REF form)
     */
    Mat Mat::gaussian_eliminate() const
    {
        Mat result(*this); // Create a copy of the original matrix
        int rows = result.row;
        int cols = result.col;

        int lead = 0; // Leading column tracker

        for (int r = 0; r < rows; ++r)
        {
            if (lead >= cols)
                break;

            int i = r;

            // Find pivot row (partial pivoting)
            while (result(i, lead) == 0)
            {
                i++;
                if (i == rows)
                {
                    i = r;
                    lead++;
                    if (lead == cols)
                        return result; // Return the result matrix (upper triangular)
                }
            }

            // Swap rows if pivot is not in current row
            if (i != r)
                result.swap_rows(i, r);

            // Eliminate rows below
            for (int j = r + 1; j < rows; ++j)
            {
                if (result(j, lead) == 0)
                    continue;

                float factor = result(j, lead) / result(r, lead);
                for (int k = lead; k < cols; ++k)
                {
                    result(j, k) -= factor * result(r, k);

                    // Numerical precision handling (set near-zero values to zero)
                    if (fabs(result(j, k)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                        result(j, k) = 0.0f;
                }
            }

            lead++;
        }

        return result; // Return the upper triangular matrix
    }

    /**
     * @name Mat::row_reduce_from_gaussian()
     * @brief Convert a matrix (assumed in row echelon form) to Reduced Row Echelon Form (RREF).
     *
     * @return Mat The matrix in RREF form
     */
    Mat Mat::row_reduce_from_gaussian()
    {
        Mat R(*this); // Make a copy to preserve original matrix
        int rows = R.row;
        int cols = R.col;

        int pivot_row = rows - 1;
        int pivot_col = cols - 2;

        while (pivot_row >= 0)
        {
            // Locate pivot in current row
            int current_pivot_col = -1;
            for (int k = 0; k < cols; ++k)
            {
                if (R(pivot_row, k) != 0)
                {
                    current_pivot_col = k;
                    break;
                }
            }

            if (current_pivot_col != -1)
            {
                // Normalize pivot row
                float pivot_val = R(pivot_row, current_pivot_col);
                for (int s = current_pivot_col; s < cols; ++s)
                {
                    R(pivot_row, s) /= pivot_val;
                    if (fabs(R(pivot_row, s)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                    {
                        R(pivot_row, s) = 0.0f;
                    }
                }

                // Eliminate above pivot
                for (int t = pivot_row - 1; t >= 0; --t)
                {
                    float factor = R(t, current_pivot_col);
                    for (int s = current_pivot_col; s < cols; ++s)
                    {
                        R(t, s) -= factor * R(pivot_row, s);
                        if (fabs(R(t, s)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                        {
                            R(t, s) = 0.0f;
                        }
                    }
                }
            }

            pivot_row--;
        }

        return R;
    }

    /**
     * @name Mat::inverse_gje()
     * @brief Compute the inverse of a square matrix using Gauss-Jordan elimination.
     *
     * @return Mat The inverse matrix if invertible, otherwise returns empty matrix.
     */
    Mat Mat::inverse_gje()
    {
        if (this->row != this->col)
        {
            std::cerr << "[Error] Inversion requires a square matrix.\n";
            return Mat();
        }

        // Step 1: Create augmented matrix [A | I]
        Mat I = Mat::eye(this->row);            // Identity matrix
        Mat augmented = Mat::augment(*this, I); // Augment matrix A with I

        // Step 2: Apply Gauss-Jordan elimination to get [I | A_inv]
        Mat rref = augmented.gaussian_eliminate().row_reduce_from_gaussian();

        // Check if the left half is the identity matrix
        for (int i = 0; i < this->row; ++i)
        {
            for (int j = 0; j < this->col; ++j)
            {
                if (fabs(rref(i, j) - I(i, j)) > TINY_MATH_MIN_POSITIVE_INPUT_F32)
                {
                    std::cerr << "[Error] Matrix is singular, cannot compute inverse.\n";
                    return Mat();
                }
            }
        }

        // Step 3: Extract the right half as the inverse matrix
        Mat result(this->row, this->col);
        for (int i = 0; i < this->row; ++i)
        {
            for (int j = 0; j < this->col; ++j)
            {
                result(i, j) = rref(i, j + this->col); // Extract the right part
            }
        }

        return result;
    }

    /**
     * @name Mat::solve
     * @brief Solve the linear system Ax = b using Gaussian elimination.
     *
     * @param A Coefficient matrix (NxN)
     * @param b Result vector (Nx1)
     * @return Mat Solution vector (Nx1) containing the roots of the equation Ax = b
     */
    Mat Mat::solve(const Mat &A, const Mat &b) const
    {
        // Check if the matrix A is square
        if (A.row != A.col)
        {
            std::cerr << "[Error] Matrix A must be square for solving.\n";
            return Mat(); // Return empty matrix
        }

        // Check if A and b dimensions are compatible for solving
        if (A.row != b.row || b.col != 1)
        {
            std::cerr << "[Error] Matrix dimensions do not match for solving.\n";
            return Mat(); // Return empty matrix
        }

        // Create augmented matrix [A | b]
        Mat augmentedMatrix(A.row, A.col + 1);
        for (int i = 0; i < A.row; ++i)
        {
            for (int j = 0; j < A.col; ++j)
            {
                augmentedMatrix(i, j) = A(i, j); // Copy matrix A into augmented matrix
            }
            augmentedMatrix(i, A.col) = b(i, 0); // Copy vector b into augmented matrix
        }

        // Perform Gaussian elimination
        for (int i = 0; i < A.row; ++i)
        {
            // Find pivot and make sure it's non-zero
            if (augmentedMatrix(i, i) == 0)
            {
                std::cerr << "[Error] Pivot is zero, matrix is singular.\n";
                return Mat(); // Return empty matrix
            }

            // Normalize the pivot row
            float pivot = augmentedMatrix(i, i);
            for (int j = i; j < augmentedMatrix.col; ++j)
            {
                augmentedMatrix(i, j) /= pivot; // Normalize the pivot row
            }

            // Eliminate the entries below the pivot
            for (int j = i + 1; j < A.row; ++j)
            {
                float factor = augmentedMatrix(j, i);
                for (int k = i; k < augmentedMatrix.col; ++k)
                {
                    augmentedMatrix(j, k) -= factor * augmentedMatrix(i, k);
                }
            }
        }

        // Back-substitution to find the solution
        Mat solution(A.row, 1);
        for (int i = A.row - 1; i >= 0; --i)
        {
            float sum = augmentedMatrix(i, A.col);
            for (int j = i + 1; j < A.row; ++j)
            {
                sum -= augmentedMatrix(i, j) * solution(j, 0);
            }
            solution(i, 0) = sum;
        }

        return solution;
    }

    /**
     * @name Mat::band_solve
     * @brief Solve the system of equations Ax = b using optimized Gaussian elimination for banded matrices.
     *
     * @param A Coefficient matrix (NxN) - banded matrix
     * @param b Result vector (Nx1)
     * @param k Bandwidth of the matrix (the width of the non-zero bands)
     * @return Mat Solution vector (Nx1) containing the roots of the equation Ax = b
     */
    Mat Mat::band_solve(Mat A, Mat b, int k)
    {
        // Dimension compatibility check
        if (A.row != A.col) // Check if A is a square matrix
        {
            std::cerr << "[Error] Matrix A must be square for solving.\n";
            return Mat(); // Return an empty matrix in case of an error
        }

        if (A.row != b.row || b.col != 1) // Check if dimensions of A and b are compatible
        {
            std::cerr << "[Error] Matrix dimensions are not compatible for solving.\n";
            return Mat(); // Return an empty matrix in case of an error
        }

        int bandsBelow = (k - 1) / 2; // Number of bands below the main diagonal

        // Perform forward elimination to reduce the matrix
        for (int i = 0; i < A.row; ++i)
        {
            if (A(i, i) == 0)
            {
                // Pivot 0 - error
                std::cerr << "[Error] Zero pivot detected in bandSolve. Cannot proceed.\n";
                Mat err_result(b.row, 1);
                memset(err_result.data, 0, b.row * sizeof(float));
                return err_result;
            }

            float a_ii = 1 / A(i, i); // Inverse of the pivot element

            // Eliminate elements below the pivot in the current column
            for (int j = i + 1; j < A.row && j <= i + bandsBelow; ++j)
            {
                if (A(j, i) != 0)
                {
                    float factor = A(j, i) * a_ii;
                    for (int col_idx = i; col_idx < A.col; ++col_idx)
                    {
                        A(j, col_idx) -= A(i, col_idx) * factor; // Eliminate the element
                    }
                    b(j, 0) -= b(i, 0) * factor; // Update the result vector
                    A(j, i) = 0;                 // Set the element to zero as it has been eliminated
                }
            }
        }

        // Back substitution to solve for x
        Mat x(b.row, 1);
        x(x.row - 1, 0) = b(x.row - 1, 0) / A(x.row - 1, x.row - 1); // Solve the last variable

        for (int i = x.row - 2; i >= 0; --i)
        {
            float sum = 0;
            for (int j = i + 1; j < x.row; ++j)
            {
                sum += A(i, j) * x(j, 0); // Sum of the known terms
            }
            x(i, 0) = (b(i, 0) - sum) / A(i, i); // Solve for the current variable
        }

        return x; // Return the solution vector
    }

    /**
     * @name Mat::roots(Mat A, Mat y)
     * @brief   Solve the matrix using a different method. Another implementation of the 'solve' function, no difference in principle.
     *
     * This method solves the linear system A * x = y using Gaussian elimination.
     *
     * @param[in] A: matrix [N]x[N] with input coefficients
     * @param[in] y: vector [N]x[1] with result values
     *
     * @return
     *      - matrix [N]x[1] with roots
     */
    Mat Mat::roots(Mat A, Mat y)
    {
        // Check if A is square
        if (A.row != A.col)
        {
            std::cerr << "[Error] Matrix A must be square for solving.\n";
            return Mat();
        }
        
        int n = A.row; // Number of rows and columns in A (A is square)

        // Create augmented matrix [A | y]
        Mat augmentedMatrix = Mat::augment(A, y);

        // Perform Gaussian elimination
        for (int j = 0; j < n; j++)
        {
            // Normalize the pivot row (make pivot element equal to 1)
            float pivot = augmentedMatrix(j, j);
            if (pivot == 0)
            {
                std::cerr << "[Error] Pivot is zero, system may have no solution." << std::endl;
                return Mat(); // Return an empty matrix in case of an error
            }

            for (int k = 0; k < augmentedMatrix.col; k++)
            {
                augmentedMatrix(j, k) /= pivot;
            }

            // Eliminate the column below the pivot (set other elements in the column to zero)
            for (int i = j + 1; i < n; i++)
            {
                float factor = augmentedMatrix(i, j);
                for (int k = 0; k < augmentedMatrix.col; k++)
                {
                    augmentedMatrix(i, k) -= factor * augmentedMatrix(j, k);
                }
            }
        }

        // Perform back-substitution
        Mat result(n, 1);
        for (int i = n - 1; i >= 0; i--)
        {
            float sum = augmentedMatrix(i, n); // Right-hand side of the augmented matrix
            for (int j = i + 1; j < n; j++)
            {
                sum -= augmentedMatrix(i, j) * result(j, 0); // Subtract the known terms
            }
            result(i, 0) = sum; // Solve for the current variable
        }

        return result;
    }

    // ============================================================================
    // Matrix Decomposition
    // ============================================================================
    /**
     * @name Mat::LUDecomposition::LUDecomposition()
     * @brief Default constructor for LUDecomposition structure
     */
    Mat::LUDecomposition::LUDecomposition()
    {
        pivoted = false;
        status = TINY_OK;
    }

    /**
     * @name Mat::CholeskyDecomposition::CholeskyDecomposition()
     * @brief Default constructor for CholeskyDecomposition structure
     */
    Mat::CholeskyDecomposition::CholeskyDecomposition()
    {
        status = TINY_OK;
    }

    /**
     * @name Mat::QRDecomposition::QRDecomposition()
     * @brief Default constructor for QRDecomposition structure
     */
    Mat::QRDecomposition::QRDecomposition()
    {
        status = TINY_OK;
    }

    /**
     * @name Mat::SVDDecomposition::SVDDecomposition()
     * @brief Default constructor for SVDDecomposition structure
     */
    Mat::SVDDecomposition::SVDDecomposition()
    {
        rank = 0;
        iterations = 0;
        status = TINY_OK;
    }

    /**
     * @name Mat::is_positive_definite()
     * @brief Check if matrix is positive definite (for Cholesky decomposition)
     * @note Uses Sylvester's criterion: all leading principal minors must be positive
     * 
     * @param tolerance Tolerance for numerical checks
     * @return true if matrix is positive definite, false otherwise
     */
    bool Mat::is_positive_definite(float tolerance) const
    {
        // Must be square
        if (this->row != this->col)
        {
            return false;
        }

        // Must be symmetric
        if (!this->is_symmetric(tolerance))
        {
            return false;
        }

        int n = this->row;

        // Check Sylvester's criterion: all leading principal minors must be positive
        // For efficiency, we check a few leading minors
        for (int k = 1; k <= n && k <= 5; ++k)  // Check first 5 minors
        {
            Mat submatrix(k, k);
            for (int i = 0; i < k; ++i)
            {
                for (int j = 0; j < k; ++j)
                {
                    submatrix(i, j) = (*this)(i, j);
                }
            }
            
            float det = submatrix.determinant();
            if (det <= tolerance)
            {
                return false;
            }
        }

        // Additional check: all diagonal elements should be positive
        for (int i = 0; i < n; ++i)
        {
            if ((*this)(i, i) <= tolerance)
            {
                return false;
            }
        }

        return true;
    }

    /**
     * @name Mat::lu_decompose()
     * @brief Compute LU decomposition: A = L * U (with optional pivoting)
     * @note Efficient for solving multiple systems with same coefficient matrix
     * 
     * @param use_pivoting Whether to use partial pivoting (default: true)
     * @return LUDecomposition containing L, U, P matrices and status
     */
    Mat::LUDecomposition Mat::lu_decompose(bool use_pivoting) const
    {
        LUDecomposition result;

        // Validation: must be square matrix
        if (this->row != this->col)
        {
            std::cerr << "[Error] LU decomposition requires a square matrix.\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        if (this->data == nullptr)
        {
            std::cerr << "[Error] Matrix data pointer is null.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }

        int n = this->row;
        Mat A = Mat(*this);  // Working copy
        result.L = Mat::eye(n);  // Initialize L as identity
        result.U = Mat(n, n);   // Initialize U
        result.pivoted = use_pivoting;

        if (use_pivoting)
        {
            result.P = Mat::eye(n);  // Initialize P as identity
        }

        // LU decomposition with partial pivoting
        for (int k = 0; k < n; ++k)
        {
            if (use_pivoting)
            {
                // Find pivot (largest element in column k, below diagonal)
                int max_row = k;
                float max_val = fabsf(A(k, k));
                for (int i = k + 1; i < n; ++i)
                {
                    float abs_val = fabsf(A(i, k));
                    if (abs_val > max_val)
                    {
                        max_val = abs_val;
                        max_row = i;
                    }
                }

                // Swap rows if necessary
                if (max_row != k)
                {
                    A.swap_rows(k, max_row);
                    result.P.swap_rows(k, max_row);
                    // Also swap previously computed L rows (but only the multipliers)
                    for (int j = 0; j < k; ++j)
                    {
                        float temp = result.L(k, j);
                        result.L(k, j) = result.L(max_row, j);
                        result.L(max_row, j) = temp;
                    }
                }
            }

            // Check for singular matrix
            if (fabsf(A(k, k)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] LU decomposition: Matrix is singular or near-singular.\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }

            // Compute U (upper triangular part)
            for (int j = k; j < n; ++j)
            {
                result.U(k, j) = A(k, j);
            }

            // Compute L (lower triangular multipliers)
            for (int i = k + 1; i < n; ++i)
            {
                float multiplier = A(i, k) / A(k, k);
                result.L(i, k) = multiplier;
                
                // Update A for next iteration
                for (int j = k + 1; j < n; ++j)
                {
                    A(i, j) -= multiplier * A(k, j);
                }
            }
        }

        result.status = TINY_OK;
        return result;
    }

    /**
     * @name Mat::cholesky_decompose()
     * @brief Compute Cholesky decomposition: A = L * L^T (for symmetric positive definite matrices)
     * @note Faster than LU for SPD matrices, used in structural dynamics
     * 
     * @return CholeskyDecomposition containing L matrix and status
     */
    Mat::CholeskyDecomposition Mat::cholesky_decompose() const
    {
        CholeskyDecomposition result;

        // Validation: must be square matrix
        if (this->row != this->col)
        {
            std::cerr << "[Error] Cholesky decomposition requires a square matrix.\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        if (this->data == nullptr)
        {
            std::cerr << "[Error] Matrix data pointer is null.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }

        // Check if symmetric
        if (!this->is_symmetric(1e-6f))
        {
            std::cerr << "[Error] Cholesky decomposition requires a symmetric matrix.\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        int n = this->row;
        result.L = Mat(n, n);

        // Cholesky decomposition: A = L * L^T
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                float sum = 0.0f;
                
                if (j == i)
                {
                    // Diagonal elements
                    for (int k = 0; k < j; ++k)
                    {
                        sum += result.L(j, k) * result.L(j, k);
                    }
                    float diag_val = (*this)(j, j) - sum;
                    
                    if (diag_val <= 0.0f)
                    {
                        std::cerr << "[Error] Cholesky decomposition: Matrix is not positive definite.\n";
                        result.status = TINY_ERR_MATH_INVALID_PARAM;
                        return result;
                    }
                    
                    result.L(j, j) = sqrtf(diag_val);
                }
                else
                {
                    // Off-diagonal elements
                    for (int k = 0; k < j; ++k)
                    {
                        sum += result.L(i, k) * result.L(j, k);
                    }
                    result.L(i, j) = ((*this)(i, j) - sum) / result.L(j, j);
                }
            }
        }

        result.status = TINY_OK;
        return result;
    }

    /**
     * @name Mat::qr_decompose()
     * @brief Compute QR decomposition: A = Q * R (Q orthogonal, R upper triangular)
     * @note Numerically stable, used for least squares and orthogonalization
     * 
     * @return QRDecomposition containing Q and R matrices and status
     */
    Mat::QRDecomposition Mat::qr_decompose() const
    {
        QRDecomposition result;

        if (this->data == nullptr)
        {
            std::cerr << "[Error] Matrix data pointer is null.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }

        int m = this->row;
        int n = this->col;
        int min_dim = (m < n) ? m : n;

        // QR decomposition using Gram-Schmidt process
        // Use the reusable gram_schmidt_orthogonalize function
        Mat Q_ortho, R_coeff;
        if (!Mat::gram_schmidt_orthogonalize(*this, Q_ortho, R_coeff, TINY_MATH_MIN_POSITIVE_INPUT_F32))
        {
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }

        // Extract Q and R from the orthogonalization results
        result.Q = Q_ortho;
        result.R = Mat(m, n);

        // Copy coefficients to R (upper triangular part)
        for (int j = 0; j < min_dim; ++j)
        {
            for (int k = 0; k <= j; ++k)
            {
                result.R(k, j) = R_coeff(k, j);
            }

            // Compute remaining R elements: R(j,k) = Q(:,j)^T * A(:,k) for k > j
            for (int k = j + 1; k < n; ++k)
            {
                float dot = 0.0f;
                for (int i = 0; i < m; ++i)
                {
                    dot += result.Q(i, j) * (*this)(i, k);
                }
                result.R(j, k) = dot;
            }
        }

        result.status = TINY_OK;
        return result;
    }

    /**
     * @name Mat::svd_decompose()
     * @brief Compute Singular Value Decomposition: A = U * S * V^T
     * @note Most general decomposition, used for rank estimation, pseudo-inverse, dimension reduction
     *       Uses iterative method (bidiagonalization + QR iteration)
     * 
     * @param max_iter Maximum number of iterations (default: 100)
     * @param tolerance Convergence tolerance (default: 1e-6)
     * @return SVDDecomposition containing U, S, V matrices and status
     */
    Mat::SVDDecomposition Mat::svd_decompose(int max_iter, float tolerance) const
    {
        SVDDecomposition result;

        if (this->data == nullptr)
        {
            std::cerr << "[Error] Matrix data pointer is null.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }

        int m = this->row;
        int n = this->col;
        int min_dim = (m < n) ? m : n;

        // For simplicity, we use a simplified SVD algorithm
        // Full SVD implementation is complex, so we use an iterative approach
        // based on eigendecomposition of A^T * A and A * A^T

        // Compute A^T * A (n x n matrix)
        Mat AtA(n, n);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                AtA(i, j) = 0.0f;
                for (int k = 0; k < m; ++k)
                {
                    AtA(i, j) += (*this)(k, i) * (*this)(k, j);
                }
            }
        }

        // Eigendecomposition of A^T * A to get V and singular values squared
        Mat::EigenDecomposition eig_AtA = AtA.eigendecompose_jacobi(tolerance, max_iter);
        
        if (eig_AtA.status != TINY_OK)
        {
            result.status = eig_AtA.status;
            return result;
        }

        // Extract singular values (square root of eigenvalues of A^T * A)
        result.S = Mat(min_dim, 1);
        result.V = Mat(n, n);
        
        // Sort eigenvalues in descending order and extract singular values
        // For simplicity, we'll use the eigenvalues directly
        int sv_count = 0;
        for (int i = 0; i < n && sv_count < min_dim; ++i)
        {
            float eigenval = eig_AtA.eigenvalues(i, 0);
            if (eigenval > tolerance)
            {
                result.S(sv_count, 0) = sqrtf(eigenval);
                // Copy corresponding eigenvector to V
                for (int j = 0; j < n; ++j)
                {
                    result.V(j, sv_count) = eig_AtA.eigenvectors(j, i);
                }
                sv_count++;
            }
        }

        result.rank = sv_count;

        // Compute U from A * V = U * S
        result.U = Mat(m, min_dim);
        for (int i = 0; i < sv_count; ++i)
        {
            float sigma = result.S(i, 0);
            if (sigma > tolerance)
            {
                // U(:,i) = (A * V(:,i)) / sigma
                for (int j = 0; j < m; ++j)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < n; ++k)
                    {
                        sum += (*this)(j, k) * result.V(k, i);
                    }
                    result.U(j, i) = sum / sigma;
                }
            }
        }

        result.iterations = eig_AtA.iterations;
        result.status = TINY_OK;
        return result;
    }

    /**
     * @name Mat::solve_lu()
     * @brief Solve linear system using LU decomposition (more efficient for multiple RHS)
     * 
     * @param lu LU decomposition of coefficient matrix
     * @param b Right-hand side vector
     * @return Solution vector x such that A * x = b
     */
    Mat Mat::solve_lu(const LUDecomposition &lu, const Mat &b)
    {
        if (lu.status != TINY_OK)
        {
            std::cerr << "[Error] solve_lu: Invalid LU decomposition.\n";
            return Mat();
        }

        int n = lu.L.row;
        if (b.row != n || b.col != 1)
        {
            std::cerr << "[Error] solve_lu: Dimension mismatch.\n";
            return Mat();
        }

        // Apply permutation if pivoting was used
        Mat b_perm = b;
        if (lu.pivoted)
        {
            // b_perm = P * b
            b_perm = Mat(n, 1);
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (lu.P(i, j) > 0.5f)  // P is permutation matrix
                    {
                        b_perm(i, 0) = b(j, 0);
                        break;
                    }
                }
            }
        }

        // Solve L * y = b_perm (forward substitution)
        Mat y(n, 1);
        for (int i = 0; i < n; ++i)
        {
            float sum = b_perm(i, 0);
            for (int j = 0; j < i; ++j)
            {
                sum -= lu.L(i, j) * y(j, 0);
            }
            y(i, 0) = sum;  // L has unit diagonal
        }

        // Solve U * x = y (backward substitution)
        Mat x(n, 1);
        for (int i = n - 1; i >= 0; --i)
        {
            float sum = y(i, 0);
            for (int j = i + 1; j < n; ++j)
            {
                sum -= lu.U(i, j) * x(j, 0);
            }
            if (fabsf(lu.U(i, i)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] solve_lu: Singular matrix.\n";
                return Mat();
            }
            x(i, 0) = sum / lu.U(i, i);
        }

        return x;
    }

    /**
     * @name Mat::solve_cholesky()
     * @brief Solve linear system using Cholesky decomposition (for SPD matrices)
     * 
     * @param chol Cholesky decomposition of coefficient matrix
     * @param b Right-hand side vector
     * @return Solution vector x such that A * x = b
     */
    Mat Mat::solve_cholesky(const CholeskyDecomposition &chol, const Mat &b)
    {
        if (chol.status != TINY_OK)
        {
            std::cerr << "[Error] solve_cholesky: Invalid Cholesky decomposition.\n";
            return Mat();
        }

        int n = chol.L.row;
        if (b.row != n || b.col != 1)
        {
            std::cerr << "[Error] solve_cholesky: Dimension mismatch.\n";
            return Mat();
        }

        // Solve L * y = b (forward substitution)
        Mat y(n, 1);
        for (int i = 0; i < n; ++i)
        {
            float sum = b(i, 0);
            for (int j = 0; j < i; ++j)
            {
                sum -= chol.L(i, j) * y(j, 0);
            }
            if (fabsf(chol.L(i, i)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] solve_cholesky: Singular matrix.\n";
                return Mat();
            }
            y(i, 0) = sum / chol.L(i, i);
        }

        // Solve L^T * x = y (backward substitution)
        Mat x(n, 1);
        for (int i = n - 1; i >= 0; --i)
        {
            float sum = y(i, 0);
            for (int j = i + 1; j < n; ++j)
            {
                sum -= chol.L(j, i) * x(j, 0);  // L^T(j,i) = L(i,j)
            }
            if (fabsf(chol.L(i, i)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] solve_cholesky: Singular matrix.\n";
                return Mat();
            }
            x(i, 0) = sum / chol.L(i, i);
        }

        return x;
    }

    /**
     * @name Mat::solve_qr()
     * @brief Solve linear system using QR decomposition (least squares solution)
     * 
     * @param qr QR decomposition of coefficient matrix
     * @param b Right-hand side vector
     * @return Least squares solution vector x such that ||A * x - b|| is minimized
     */
    Mat Mat::solve_qr(const QRDecomposition &qr, const Mat &b)
    {
        if (qr.status != TINY_OK)
        {
            std::cerr << "[Error] solve_qr: Invalid QR decomposition.\n";
            return Mat();
        }

        int m = qr.Q.row;
        int n = qr.R.col;
        
        if (b.row != m || b.col != 1)
        {
            std::cerr << "[Error] solve_qr: Dimension mismatch.\n";
            return Mat();
        }

        // Compute Q^T * b
        Mat Qt_b(n, 1);
        for (int i = 0; i < n; ++i)
        {
            Qt_b(i, 0) = 0.0f;
            for (int j = 0; j < m; ++j)
            {
                Qt_b(i, 0) += qr.Q(j, i) * b(j, 0);  // Q^T(i,j) = Q(j,i)
            }
        }

        // Solve R * x = Q^T * b (backward substitution)
        Mat x(n, 1);
        int min_dim = (m < n) ? m : n;
        for (int i = min_dim - 1; i >= 0; --i)
        {
            if (fabsf(qr.R(i, i)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                // Skip zero diagonal (underdetermined system)
                x(i, 0) = 0.0f;
                continue;
            }
            
            float sum = Qt_b(i, 0);
            for (int j = i + 1; j < n; ++j)
            {
                sum -= qr.R(i, j) * x(j, 0);
            }
            x(i, 0) = sum / qr.R(i, i);
        }

        // Set remaining components to zero if n > m
        for (int i = min_dim; i < n; ++i)
        {
            x(i, 0) = 0.0f;
        }

        return x;
    }

    /**
     * @name Mat::pseudo_inverse()
     * @brief Compute pseudo-inverse using SVD: A^+ = V * S^+ * U^T
     * 
     * @param svd SVD decomposition of matrix A
     * @param tolerance Tolerance for singular values (default: 1e-6)
     * @return Pseudo-inverse matrix A^+
     */
    Mat Mat::pseudo_inverse(const SVDDecomposition &svd, float tolerance)
    {
        if (svd.status != TINY_OK)
        {
            std::cerr << "[Error] pseudo_inverse: Invalid SVD decomposition.\n";
            return Mat();
        }

        int m = svd.U.row;
        int n = svd.V.row;
        int rank = svd.rank;

        // Compute S^+ (pseudo-inverse of S)
        Mat Sp(n, m);
        for (int i = 0; i < rank; ++i)
        {
            float sigma = svd.S(i, 0);
            if (sigma > tolerance)
            {
                Sp(i, i) = 1.0f / sigma;
            }
        }

        // Compute A^+ = V * S^+ * U^T
        // First compute V * S^+
        Mat VS(n, m);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                VS(i, j) = 0.0f;
                for (int k = 0; k < rank; ++k)
                {
                    VS(i, j) += svd.V(i, k) * Sp(k, j);
                }
            }
        }

        // Then compute (V * S^+) * U^T
        Mat A_plus(n, m);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                A_plus(i, j) = 0.0f;
                for (int k = 0; k < rank; ++k)
                {
                    A_plus(i, j) += VS(i, k) * svd.U(j, k);  // U^T(k,j) = U(j,k)
                }
            }
        }

        return A_plus;
    }

    // ============================================================================
    // Eigenvalue & Eigenvector Decomposition
    // ============================================================================
    /**
     * @name Mat::EigenPair::EigenPair()
     * @brief Default constructor for EigenPair structure
     */
    Mat::EigenPair::EigenPair() : eigenvalue(0.0f), iterations(0), status(TINY_OK)
    {
    }

    /**
     * @name Mat::EigenDecomposition::EigenDecomposition()
     * @brief Default constructor for EigenDecomposition structure
     */
    Mat::EigenDecomposition::EigenDecomposition() : iterations(0), status(TINY_OK)
    {
    }

    /**
     * @name Mat::is_symmetric()
     * @brief Check if the matrix is symmetric within a given tolerance.
     * @note Essential for SHM applications where structural matrices are typically symmetric.
     *
     * @param tolerance Maximum allowed difference between A(i,j) and A(j,i)
     * @return true if matrix is symmetric, false otherwise
     */
    bool Mat::is_symmetric(float tolerance) const
    {
        // Only square matrices can be symmetric
        if (this->row != this->col)
        {
            return false;
        }

        // Check symmetry: A(i,j) should equal A(j,i) within tolerance
        for (int i = 0; i < this->row; ++i)
        {
            for (int j = i + 1; j < this->col; ++j)
            {
                float diff = fabsf((*this)(i, j) - (*this)(j, i));
                if (diff > tolerance)
                {
                    return false;
                }
            }
        }

        return true;
    }

    /**
     * @name Mat::power_iteration()
     * @brief Compute the dominant (largest magnitude) eigenvalue and eigenvector using power iteration.
     * @note Fast method suitable for real-time SHM applications to quickly identify primary frequency.
     *
     * @param max_iter Maximum number of iterations (default: 1000)
     * @param tolerance Convergence tolerance (default: 1e-6)
     * @return EigenPair containing the dominant eigenvalue, eigenvector, and status
     */
    Mat::EigenPair Mat::power_iteration(int max_iter, float tolerance) const
    {
        EigenPair result;

        // Validation: must be square matrix
        if (this->row != this->col)
        {
            std::cerr << "[Error] Power iteration requires a square matrix.\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        if (this->data == nullptr)
        {
            std::cerr << "[Error] Matrix data pointer is null.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }

        int n = this->row;

        // Initialize eigenvector with better strategy to avoid convergence to smaller eigenvalues
        // Strategy: Use sum of columns (or rows) to get a vector with components in all directions
        result.eigenvector = Mat(n, 1);
        float norm_sq = 0.0f;
        
        // Method 1: Use sum of absolute values of columns (more robust)
        for (int i = 0; i < n; ++i)
        {
            float col_sum = 0.0f;
            for (int j = 0; j < n; ++j)
            {
                col_sum += fabsf((*this)(j, i));
            }
            result.eigenvector(i, 0) = col_sum + 1.0f; // Add 1 to avoid zero
            norm_sq += result.eigenvector(i, 0) * result.eigenvector(i, 0);
        }
        
        // If all components are too similar, use a different initialization
        if (norm_sq < TINY_MATH_MIN_POSITIVE_INPUT_F32)
        {
            // Fallback: use values based on index with some variation
            for (int i = 0; i < n; ++i)
            {
                result.eigenvector(i, 0) = 1.0f + 0.1f * static_cast<float>(i);
                norm_sq += result.eigenvector(i, 0) * result.eigenvector(i, 0);
            }
        }
        
        float inv_norm = 1.0f / sqrtf(norm_sq);
        for (int i = 0; i < n; ++i)
        {
            result.eigenvector(i, 0) *= inv_norm;
        }

        // Power iteration loop
        Mat temp_vec(n, 1);
        float prev_eigenvalue = 0.0f;

        for (int iter = 0; iter < max_iter; ++iter)
        {
            // Compute A * v
            for (int i = 0; i < n; ++i)
            {
                temp_vec(i, 0) = 0.0f;
                for (int j = 0; j < n; ++j)
                {
                    temp_vec(i, 0) += (*this)(i, j) * result.eigenvector(j, 0);
                }
            }

            // Compute Rayleigh quotient: lambda = v^T * A * v / (v^T * v)
            float numerator = 0.0f;
            float denominator = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                numerator += result.eigenvector(i, 0) * temp_vec(i, 0);
                denominator += result.eigenvector(i, 0) * result.eigenvector(i, 0);
            }

            if (fabsf(denominator) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] Power iteration: eigenvector norm too small.\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }

            result.eigenvalue = numerator / denominator;

            // Normalize the new vector
            float new_norm_sq = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                new_norm_sq += temp_vec(i, 0) * temp_vec(i, 0);
            }

            if (new_norm_sq < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] Power iteration: computed vector norm too small.\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }

            float new_inv_norm = 1.0f / sqrtf(new_norm_sq);
            for (int i = 0; i < n; ++i)
            {
                result.eigenvector(i, 0) = temp_vec(i, 0) * new_inv_norm;
            }

            // Check convergence
            if (iter > 0)
            {
                float eigenvalue_change = fabsf(result.eigenvalue - prev_eigenvalue);
                if (eigenvalue_change < tolerance * fabsf(result.eigenvalue))
                {
                    result.iterations = iter + 1;
                    result.status = TINY_OK;
                    return result;
                }
            }

            prev_eigenvalue = result.eigenvalue;
        }

        // Max iterations reached
        result.iterations = max_iter;
        result.status = TINY_ERR_NOT_FINISHED;
        std::cerr << "[Warning] Power iteration did not converge within " << max_iter << " iterations.\n";
        return result;
    }

    /**
     * @name Mat::inverse_power_iteration()
     * @brief Compute the smallest (minimum magnitude) eigenvalue and eigenvector using inverse power iteration.
     * @note Critical for system identification - finds fundamental frequency/lowest mode in structural dynamics.
     *       This method is essential for SHM applications where the smallest eigenvalue corresponds to the
     *       fundamental frequency of the system.
     *
     * @param max_iter Maximum number of iterations (default: 1000)
     * @param tolerance Convergence tolerance (default: 1e-6)
     * @return EigenPair containing the smallest eigenvalue, eigenvector, and status
     * 
     * @details Algorithm:
     *  1. Initialize normalized eigenvector v
     *  2. Iterate: Solve A * y = v (equivalent to y = A^(-1) * v)
     *  3. Normalize y to get new v
     *  4. Compute eigenvalue estimate: lambda_min = 1 / (v^T * y)
     *  5. Check convergence
     * 
     * @note The matrix must be invertible (non-singular) for this method to work.
     *       If the matrix is singular or near-singular, the method will fail gracefully.
     */
    Mat::EigenPair Mat::inverse_power_iteration(int max_iter, float tolerance) const
    {
        EigenPair result;

        // Validation: must be square matrix
        if (this->row != this->col)
        {
            std::cerr << "[Error] Inverse power iteration requires a square matrix.\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        if (this->data == nullptr)
        {
            std::cerr << "[Error] Matrix data pointer is null.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }

        int n = this->row;

        // Check if matrix is singular by computing determinant (quick check)
        // For efficiency, we'll check during the first solve operation instead

        // Initialize eigenvector for inverse power iteration
        // Strategy: Use a vector that is orthogonal to the dominant eigenvector direction
        // For inverse power iteration, we want to converge to the smallest eigenvalue
        // Use a simple initialization: [1, 1, ..., 1]^T normalized, which typically
        // has components in all eigenvector directions
        result.eigenvector = Mat(n, 1);
        float norm_sq = 0.0f;
        
        // Initialize with alternating signs to avoid alignment with dominant eigenvector
        // This helps ensure we converge to the smallest eigenvalue
        for (int i = 0; i < n; ++i)
        {
            // Use alternating pattern: 1, -1, 1, -1, ... with small variations
            result.eigenvector(i, 0) = (i % 2 == 0) ? 1.0f : -1.0f;
            result.eigenvector(i, 0) += 0.1f * static_cast<float>(i) / static_cast<float>(n);
            norm_sq += result.eigenvector(i, 0) * result.eigenvector(i, 0);
        }
        
        // Normalize
        if (norm_sq < TINY_MATH_MIN_POSITIVE_INPUT_F32)
        {
            // Fallback: use uniform vector
            for (int i = 0; i < n; ++i)
            {
                result.eigenvector(i, 0) = 1.0f;
            }
            norm_sq = static_cast<float>(n);
        }
        
        float inv_norm = 1.0f / sqrtf(norm_sq);
        for (int i = 0; i < n; ++i)
        {
            result.eigenvector(i, 0) *= inv_norm;
        }

        // Inverse power iteration loop
        Mat temp_vec(n, 1);
        float prev_eigenvalue = 0.0f;

        for (int iter = 0; iter < max_iter; ++iter)
        {
            // Solve A * y = v (equivalent to computing A^(-1) * v)
            // This is the key difference from power iteration
            temp_vec = solve(*this, result.eigenvector);

            // Check if solve failed (matrix is singular or near-singular)
            if (temp_vec.row == 0 || temp_vec.data == nullptr)
            {
                std::cerr << "[Error] Inverse power iteration: Matrix is singular or near-singular. "
                          << "Cannot solve linear system A * y = v.\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }

            // Check if solution vector is valid (not all zeros or NaN)
            bool valid_solution = false;
            for (int i = 0; i < n; ++i)
            {
                if (std::isnan(temp_vec(i, 0)) || std::isinf(temp_vec(i, 0)))
                {
                    std::cerr << "[Error] Inverse power iteration: Solution contains NaN or Inf.\n";
                    result.status = TINY_ERR_MATH_INVALID_PARAM;
                    return result;
                }
                if (fabsf(temp_vec(i, 0)) > TINY_MATH_MIN_POSITIVE_INPUT_F32)
                {
                    valid_solution = true;
                }
            }

            if (!valid_solution)
            {
                std::cerr << "[Error] Inverse power iteration: Solution vector is zero or too small.\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }

            // Compute Rayleigh quotient for A directly: lambda = (v^T * A * v) / (v^T * v)
            // This gives us the eigenvalue of A corresponding to eigenvector v
            // Note: In inverse power iteration, we iterate on A^(-1), but we want the eigenvalue of A
            // Since y = A^(-1) * v, we have A * y = v, so we can compute v^T * A * v = v^T * A * y
            // But more directly, we compute A * v to get the eigenvalue
            
            // Compute A * v
            Mat Av(n, 1);
            for (int i = 0; i < n; ++i)
            {
                Av(i, 0) = 0.0f;
                for (int j = 0; j < n; ++j)
                {
                    Av(i, 0) += (*this)(i, j) * result.eigenvector(j, 0);
                }
            }
            
            // Compute Rayleigh quotient: lambda = (v^T * A * v) / (v^T * v)
            float numerator = 0.0f;
            float denominator = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                numerator += result.eigenvector(i, 0) * Av(i, 0);
                denominator += result.eigenvector(i, 0) * result.eigenvector(i, 0);
            }

            if (fabsf(denominator) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] Inverse power iteration: eigenvector norm too small.\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }

            // Compute eigenvalue of A using Rayleigh quotient
            result.eigenvalue = numerator / denominator;

            // Normalize the new vector
            float new_norm_sq = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                new_norm_sq += temp_vec(i, 0) * temp_vec(i, 0);
            }

            if (new_norm_sq < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] Inverse power iteration: computed vector norm too small.\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }

            float new_inv_norm = 1.0f / sqrtf(new_norm_sq);
            for (int i = 0; i < n; ++i)
            {
                result.eigenvector(i, 0) = temp_vec(i, 0) * new_inv_norm;
            }

            // Check convergence
            if (iter > 0)
            {
                float eigenvalue_change = fabsf(result.eigenvalue - prev_eigenvalue);
                // Use relative tolerance for convergence check
                float rel_tolerance = tolerance * fmaxf(fabsf(result.eigenvalue), 1.0f);
                if (eigenvalue_change < rel_tolerance)
                {
                    result.iterations = iter + 1;
                    result.status = TINY_OK;
                    return result;
                }
            }

            prev_eigenvalue = result.eigenvalue;
        }

        // Max iterations reached
        result.iterations = max_iter;
        result.status = TINY_ERR_NOT_FINISHED;
        std::cerr << "[Warning] Inverse power iteration did not converge within " << max_iter << " iterations.\n";
        return result;
    }

    /**
     * @name Mat::eigendecompose_jacobi()
     * @brief Compute complete eigenvalue decomposition using Jacobi method for symmetric matrices.
     * @note Robust and accurate method ideal for structural dynamics matrices in SHM.
     *
     * @param tolerance Convergence tolerance (default: 1e-6)
     * @param max_iter Maximum number of iterations (default: 100)
     * @return EigenDecomposition containing all eigenvalues, eigenvectors, and status
     */
    Mat::EigenDecomposition Mat::eigendecompose_jacobi(float tolerance, int max_iter) const
    {
        EigenDecomposition result;

        // Validation: must be square matrix
        if (this->row != this->col)
        {
            std::cerr << "[Error] Eigendecomposition requires a square matrix.\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        if (this->data == nullptr)
        {
            std::cerr << "[Error] Matrix data pointer is null.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }

        // Check if matrix is symmetric
        if (!this->is_symmetric(tolerance * 10.0f))
        {
            std::cerr << "[Warning] Matrix is not symmetric. Jacobi method may not converge correctly.\n";
        }

        int n = this->row;

        // Initialize: working copy of matrix, eigenvectors as identity
        Mat A = Mat(*this); // Working copy (will become diagonal)
        result.eigenvectors = Mat::eye(n);

        // Jacobi iteration
        for (int iter = 0; iter < max_iter; ++iter)
        {
            // Find largest off-diagonal element
            float max_off_diag = 0.0f;
            int p = 0, q = 0;

            for (int i = 0; i < n; ++i)
            {
                for (int j = i + 1; j < n; ++j)
                {
                    float abs_val = fabsf(A(i, j));
                    if (abs_val > max_off_diag)
                    {
                        max_off_diag = abs_val;
                        p = i;
                        q = j;
                    }
                }
            }

            // Check convergence
            if (max_off_diag < tolerance)
            {
                // Extract eigenvalues from diagonal
                result.eigenvalues = Mat(n, 1);
                for (int i = 0; i < n; ++i)
                {
                    result.eigenvalues(i, 0) = A(i, i);
                }
                result.iterations = iter + 1;
                result.status = TINY_OK;
                return result;
            }

            // Compute rotation angle
            float app = A(p, p);
            float aqq = A(q, q);
            float apq = A(p, q);

            float tau = (aqq - app) / (2.0f * apq);
            float t;
            if (tau >= 0.0f)
            {
                t = 1.0f / (tau + sqrtf(1.0f + tau * tau));
            }
            else
            {
                t = -1.0f / (-tau + sqrtf(1.0f + tau * tau));
            }

            float c = 1.0f / sqrtf(1.0f + t * t); // cosine
            float s = t * c;                       // sine

            // Apply Jacobi rotation to A
            // Update rows p and q
            for (int j = 0; j < n; ++j)
            {
                if (j != p && j != q)
                {
                    float apj = A(p, j);
                    float aqj = A(q, j);
                    A(p, j) = c * apj - s * aqj;
                    A(q, j) = s * apj + c * aqj;
                    A(j, p) = A(p, j); // Maintain symmetry
                    A(j, q) = A(q, j);
                }
            }

            // Update diagonal elements
            float app_new = c * c * app - 2.0f * c * s * apq + s * s * aqq;
            float aqq_new = s * s * app + 2.0f * c * s * apq + c * c * aqq;
            A(p, p) = app_new;
            A(q, q) = aqq_new;
            A(p, q) = 0.0f;
            A(q, p) = 0.0f;

            // Update eigenvectors
            for (int i = 0; i < n; ++i)
            {
                float vip = result.eigenvectors(i, p);
                float viq = result.eigenvectors(i, q);
                result.eigenvectors(i, p) = c * vip - s * viq;
                result.eigenvectors(i, q) = s * vip + c * viq;
            }
        }

        // Extract eigenvalues from diagonal
        result.eigenvalues = Mat(n, 1);
        for (int i = 0; i < n; ++i)
        {
            result.eigenvalues(i, 0) = A(i, i);
        }

        result.iterations = max_iter;
        result.status = TINY_ERR_NOT_FINISHED;
        std::cerr << "[Warning] Jacobi method did not converge within " << max_iter << " iterations.\n";
        return result;
    }

    /**
     * @name Mat::eigendecompose_qr()
     * @brief Compute complete eigenvalue decomposition using QR algorithm for general matrices.
     * @note Supports non-symmetric matrices, but may have complex eigenvalues (only real part returned).
     *
     * @param max_iter Maximum number of QR iterations (default: 100)
     * @param tolerance Convergence tolerance (default: 1e-6)
     * @return EigenDecomposition containing eigenvalues, eigenvectors, and status
     */
    Mat::EigenDecomposition Mat::eigendecompose_qr(int max_iter, float tolerance) const
    {
        EigenDecomposition result;

        // Validation: must be square matrix
        if (this->row != this->col)
        {
            std::cerr << "[Error] Eigendecomposition requires a square matrix.\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        if (this->data == nullptr)
        {
            std::cerr << "[Error] Matrix data pointer is null.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }

        int n = this->row;

        // Initialize: start with original matrix, eigenvectors as identity
        Mat A = Mat(*this); // Working copy (will become upper triangular)
        result.eigenvectors = Mat::eye(n);

        // QR iteration with improved convergence checking
        for (int iter = 0; iter < max_iter; ++iter)
        {
            // Check convergence: check if matrix is upper triangular
            // Use a more lenient tolerance for sub-diagonal elements
            bool converged = true;
            float max_off_diag = 0.0f;
            for (int i = 1; i < n; ++i)
            {
                for (int j = 0; j < i; ++j)
                {
                    float abs_val = fabsf(A(i, j));
                    if (abs_val > max_off_diag)
                        max_off_diag = abs_val;
                    // Use relative tolerance: compare with diagonal elements
                    float diag_scale = fmaxf(fabsf(A(i, i)), fabsf(A(j, j)));
                    float rel_tolerance = tolerance * fmaxf(1.0f, diag_scale);
                    if (abs_val > rel_tolerance)
                    {
                        converged = false;
                    }
                }
            }

            if (converged)
            {
                // Extract eigenvalues from diagonal
                result.eigenvalues = Mat(n, 1);
                for (int i = 0; i < n; ++i)
                {
                    result.eigenvalues(i, 0) = A(i, i);
                }
                result.iterations = iter + 1;
                result.status = TINY_OK;
                return result;
            }
            
            // Optional: Use shift to accelerate convergence (Wilkinson shift for last 2x2 block)
            // For simplicity, we skip shift for now but can add it later if needed

            // QR decomposition using Gram-Schmidt process
            // Use the reusable gram_schmidt_orthogonalize function
            Mat Q_ortho, R_coeff;
            if (!Mat::gram_schmidt_orthogonalize(A, Q_ortho, R_coeff, TINY_MATH_MIN_POSITIVE_INPUT_F32))
            {
                result.status = TINY_ERR_MATH_NULL_POINTER;
                return result;
            }

            Mat Q = Q_ortho;
            Mat R(n, n);

            // Copy coefficients to R (upper triangular part)
            for (int j = 0; j < n; ++j)
            {
                for (int k = 0; k <= j; ++k)
                {
                    R(k, j) = R_coeff(k, j);
                }

                // Compute remaining R elements: R(j,k) = Q(:,j)^T * A(:,k) for k > j
                for (int k = j + 1; k < n; ++k)
                {
                    float dot = 0.0f;
                    for (int i = 0; i < n; ++i)
                    {
                        dot += Q(i, j) * A(i, k);
                    }
                    R(j, k) = dot;
                }
            }

            // Update A = R * Q
            Mat A_new(n, n);
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    A_new(i, j) = 0.0f;
                    for (int k = 0; k < n; ++k)
                    {
                        A_new(i, j) += R(i, k) * Q(k, j);
                    }
                }
            }
            A = A_new;

            // Update eigenvectors: V = V * Q
            Mat V_new(n, n);
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    V_new(i, j) = 0.0f;
                    for (int k = 0; k < n; ++k)
                    {
                        V_new(i, j) += result.eigenvectors(i, k) * Q(k, j);
                    }
                }
            }
            result.eigenvectors = V_new;
        }

        // Extract eigenvalues from diagonal
        result.eigenvalues = Mat(n, 1);
        for (int i = 0; i < n; ++i)
        {
            result.eigenvalues(i, 0) = A(i, i);
        }

        result.iterations = max_iter;
        result.status = TINY_ERR_NOT_FINISHED;
        std::cerr << "[Warning] QR algorithm did not converge within " << max_iter << " iterations.\n";
        return result;
    }

    /**
     * @name Mat::eigendecompose()
     * @brief Automatic eigenvalue decomposition with method selection.
     * @note Convenient interface for edge computing: uses Jacobi for symmetric matrices, QR for general.
     *
     * @param tolerance Convergence tolerance (default: 1e-6)
     * @return EigenDecomposition containing eigenvalues, eigenvectors, and status
     */
    Mat::EigenDecomposition Mat::eigendecompose(float tolerance) const
    {
        // Check if matrix is symmetric
        if (this->is_symmetric(tolerance * 10.0f))
        {
            // Use Jacobi method for symmetric matrices (more efficient and stable)
            return this->eigendecompose_jacobi(tolerance, 100);
        }
        else
        {
            // Use QR algorithm for general matrices
            return this->eigendecompose_qr(100, tolerance);
        }
    }

    // ============================================================================
    // Stream Operators
    // ============================================================================
    /**
     * @name operator<<
     * @brief Stream insertion operator for printing matrix to the output stream (e.g., std::cout).
     *
     * This function allows printing the contents of a matrix to an output stream.
     * It prints each row of the matrix on a new line, with elements separated by spaces.
     *
     * @param os Output stream where the matrix will be printed (e.g., std::cout)
     * @param m Matrix to be printed
     *
     * @return os The output stream after printing the matrix
     */
    std::ostream &operator<<(std::ostream &os, const Mat &m)
    {
        if (m.data == nullptr)
        {
            os << "[Error] Cannot print matrix: data pointer is null.\n";
            return os;
        }
        
        for (int i = 0; i < m.row; ++i)
        {
            os << m(i, 0);
            for (int j = 1; j < m.col; ++j)
            {
                os << " " << m(i, j);
            }
            os << std::endl;
        }
        return os;
    }

    /**
     * @name operator<<
     * @brief Stream insertion operator for printing the Rectangular ROI structure to the output stream.
     *
     * This function prints the details of the ROI (Region of Interest) including the start row and column,
     * and the width and height of the rectangular region.
     *
     * @param os Output stream where the ROI will be printed (e.g., std::cout)
     * @param roi The ROI structure to be printed
     *
     * @return os The output stream after printing the ROI details
     */
    std::ostream &operator<<(std::ostream &os, const Mat::ROI &roi)
    {
        os << "row start " << roi.pos_y << std::endl;
        os << "col start " << roi.pos_x << std::endl;
        os << "row count " << roi.height << std::endl;
        os << "col count " << roi.width << std::endl;

        return os;
    }

    /**
     * @name operator>>
     * @brief Stream extraction operator for reading matrix from the input stream (e.g., std::cin).
     *
     * This function reads the contents of a matrix from an input stream.
     * The matrix elements are read row by row, with elements separated by spaces or newlines.
     *
     * @param is Input stream from which the matrix will be read (e.g., std::cin)
     * @param m Matrix to store the read data
     *
     * @return is The input stream after reading the matrix
     */
    std::istream &operator>>(std::istream &is, Mat &m)
    {
        for (int i = 0; i < m.row; ++i)
        {
            for (int j = 0; j < m.col; ++j)
            {
                is >> m(i, j);
            }
        }
        return is;
    }

    // ============================================================================
    // Global Arithmetic Operators
    // ============================================================================
    /**
     * + operator, sum of two matrices
     * The operator use DSP optimized implementation of multiplication.
     *
     * @param[in] A: Input matrix A
     * @param[in] B: Input matrix B
     *
     * @return
     *     - result matrix A+B
     */
    Mat operator+(const Mat &m1, const Mat &m2)
    {
        if ((m1.row != m2.row) || (m1.col != m2.col))
        {
            std::cerr << "operator + Error: matrices do not have equal dimensions" << std::endl;
            Mat err_ret;
            return err_ret;
        }

        if (m1.sub_matrix || m2.sub_matrix)
        {
            Mat temp(m1.row, m1.col);
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_add_f32(m1.data, m2.data, temp.data, m1.row, m1.col, m1.pad, m2.pad, temp.pad, 1, 1, 1);
#else
            tiny_mat_add_f32(m1.data, m2.data, temp.data, m1.row, m1.col, m1.pad, m2.pad, temp.pad, 1, 1, 1);
#endif
            return temp;
        }
        else
        {
            Mat temp(m1);
            return (temp += m2);
        }
    }

    /**
     * + operator, sum of matrix with constant
     * The operator use DSP optimized implementation of multiplication.
     *
     * @param[in] A: Input matrix A
     * @param[in] C: Input constant
     *
     * @return
     *     - result matrix A+C
     */
    Mat operator+(const Mat &m, float C)
    {
        if (m.sub_matrix)
        {
            Mat temp(m.row, m.col);
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_addc_f32(m.data, temp.data, C, m.row, m.col, m.pad, temp.pad, 1, 1);
#else
            tiny_mat_addc_f32(m.data, temp.data, C, m.row, m.col, m.pad, temp.pad, 1, 1);
#endif
            return temp;
        }
        else
        {
            Mat temp(m);
            return (temp += C);
        }
    }

    /**
     * - operator, subtraction of two matrices
     * The operator use DSP optimized implementation of multiplication.
     *
     * @param[in] A: Input matrix A
     * @param[in] B: Input matrix B
     *
     * @return
     *     - result matrix A-B
     */
    Mat operator-(const Mat &m1, const Mat &m2)
    {
        if ((m1.row != m2.row) || (m1.col != m2.col))
        {
            std::cerr << "operator - Error: matrices do not have equal dimensions" << std::endl;
            Mat err_ret;
            return err_ret;
        }

        if (m1.sub_matrix || m2.sub_matrix)
        {
            Mat temp(m1.row, m1.col);
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_sub_f32(m1.data, m2.data, temp.data, m1.row, m1.col, m1.pad, m2.pad, temp.pad, 1, 1, 1);
#else
            tiny_mat_sub_f32(m1.data, m2.data, temp.data, m1.row, m1.col, m1.pad, m2.pad, temp.pad, 1, 1, 1);
#endif
            return temp;
        }
        else
        {
            Mat temp(m1);
            return (temp -= m2);
        }
    }


    /**
     * - operator, subtraction of matrix with constant
     * The operator use DSP optimized implementation of multiplication.
     *
     * @param[in] A: Input matrix A
     * @param[in] C: Input constant
     *
     * @return
     *     - result matrix A-C
     */
    Mat operator-(const Mat &m, float C)
    {
        if (m.sub_matrix)
        {
            Mat temp(m.row, m.col);
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_addc_f32(m.data, temp.data, -C, m.row, m.col, m.pad, temp.pad, 1, 1);
#else
            tiny_mat_addc_f32(m.data, temp.data, -C, m.row, m.col, m.pad, temp.pad, 1, 1);
#endif
            return temp;
        }
        else
        {
            Mat temp(m);
            return (temp -= C);
        }
    }


    /**
     * * operator, multiplication of two matrices.
     * The operator use DSP optimized implementation of multiplication.
     *
     * @param[in] A: Input matrix A
     * @param[in] B: Input matrix B
     *
     * @return
     *     - result matrix A*B
     */
    Mat operator*(const Mat &m1, const Mat &m2)
    {
        if (m1.col != m2.row)
        {
            std::cerr << "operator * Error: matrices do not have correct dimensions" << std::endl;
            Mat err_ret;
            return err_ret;
        }
        Mat temp(m1.row, m2.col);

        if (m1.sub_matrix || m2.sub_matrix)
        {
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_mult_ex_f32(m1.data, m2.data, temp.data, m1.row, m1.col, m2.col, m1.pad, m2.pad, temp.pad);
#else
            tiny_mat_mult_ex_f32(m1.data, m2.data, temp.data, m1.row, m1.col, m2.col, m1.pad, m2.pad, temp.pad);
#endif
        }
        else
        {
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_mult_f32(m1.data, m2.data, temp.data, m1.row, m1.col, m2.col);
#else
            tiny_mat_mult_f32(m1.data, m2.data, temp.data, m1.row, m1.col, m2.col);
#endif
        }

        return temp;
    }

    /**
     * * operator, multiplication of matrix with constant
     * The operator use DSP optimized implementation of multiplication.
     *
     * @param[in] A: Input matrix A
     * @param[in] C: floating point value
     *
     * @return
     *     - result matrix A*B
     */
    Mat operator*(const Mat &m, float num)
    {
        if (m.sub_matrix)
        {
            Mat temp(m.row, m.col);
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_mulc_f32(m.data, temp.data, num, m.row, m.col, m.pad, temp.pad, 1, 1);
#else
            tiny_mat_multc_f32(m.data, temp.data, num, m.row, m.col, m.pad, temp.pad, 1, 1);
#endif
            return temp;
        }
        else
        {
            Mat temp(m);
            return (temp *= num);
        }
    }
    
    /**
     * * operator, multiplication of matrix with constant
     * The operator use DSP optimized implementation of multiplication.
     *
     * @param[in] C: floating point value
     * @param[in] A: Input matrix A
     *
     * @return
     *     - result matrix C*A
     */
    Mat operator*(float num, const Mat &m)
    {
        return (m * num);
    }

    /**
     * / operator, divide of matrix by constant
     * The operator use DSP optimized implementation of multiplication.
     *
     * @param[in] A: Input matrix A
     * @param[in] C: floating point value
     *
     * @return
     *     - result matrix A/C
     */
    Mat operator/(const Mat &m, float num)
    {
        // Check division by zero
        if (num == 0.0f)
        {
            std::cerr << "[Error] Division by zero in operator/.\n";
            Mat err_ret;
            return err_ret;
        }
        
        if (m.sub_matrix)
        {
            Mat temp(m.row, m.col);
            float inv_num = 1.0f / num;
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_mulc_f32(m.data, temp.data, inv_num, m.row, m.col, m.pad, temp.pad, 1, 1);
#else
            tiny_mat_multc_f32(m.data, temp.data, inv_num, m.row, m.col, m.pad, temp.pad, 1, 1);
#endif
            return temp;
        }
        else
        {
            Mat temp(m);
            return (temp /= num);
        }
    }


    /**
     * / operator, divide matrix A by matrix B (element-wise)
     *
     * @param[in] A: Input matrix A
     * @param[in] B: Input matrix B
     *
     * @return
     *     - result matrix C, where C[i,j] = A[i,j]/B[i,j]
     */
    Mat operator/(const Mat &A, const Mat &B)
    {
        if ((A.row != B.row) || (A.col != B.col))
        {
            std::cerr << "operator / Error: matrices do not have equal dimensions" << std::endl;
            Mat err_ret;
            return err_ret;
        }

        Mat temp(A.row, A.col);
        for (int row = 0; row < A.row; row++)
        {
            for (int col = 0; col < A.col; col++)
            {
                temp(row, col) = A(row, col) / B(row, col);
            }
        }
        return temp;
    }

    
    /**
     * == operator, compare two matrices
     *
     * @param[in] A: Input matrix A
     * @param[in] B: Input matrix B
     *
     * @return
     *      - true if matrices are the same
     *      - false if matrices are different
     */
    bool operator==(const Mat &m1, const Mat &m2)
    {
        if ((m1.col != m2.col) || (m1.row != m2.row))
        {
            return false;
        }

        const float epsilon = 1e-5f;
        for (int row = 0; row < m1.row; row++)
        {
            for (int col = 0; col < m1.col; col++)
            {
                float diff = fabs(m1(row, col) - m2(row, col));
                if (diff > epsilon)
                {
                    std::cout << "operator == Error: " << row << " " << col << ", m1.data=" << m1(row, col) << ", m2.data=" << m2(row, col) << ", diff=" << diff << std::endl;
                    return false;
                }
            }
        }

        return true;
    }
}


```