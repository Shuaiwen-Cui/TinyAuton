# CODE 

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
         * @return EigenDecomposition containing eigenvalues, eigenvectors, and status
         * @note Automatically selects Jacobi method for symmetric matrices, QR algorithm for general matrices.
         */
        EigenDecomposition eigendecompose(float tolerance = 1e-6f) const;

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
     * @name Mat::print_info()
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
    void Mat::print_matrix(bool show_padding) const
    {
        if (this->data == nullptr)
        {
            std::cout << "[Error] Cannot print matrix: data pointer is null.\n";
            return;
        }
        
        if (this->row < 0 || this->col < 0 || this->stride < 0)
        {
            std::cout << "[Error] Invalid matrix dimensions\n";
            return;
        }
        
        if (this->stride < this->col)
        {
            std::cout << "[Warning] Stride < cols, potential data corruption\n";
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
     * @name Mat::alloc_mem()
     * @brief Allocate memory for the matrix according to the memory required.
     * @note For ESP32, it will automatically determine if using RAM or PSRAM based on the size of the matrix.
     * @note This function sets ext_buff to false and allocates memory based on row * stride.
     *       If allocation fails or parameters are invalid, data will be set to nullptr.
     */
    void Mat::alloc_mem()
    {
        // Parameter validation: check if row and stride are non-negative
        if (this->row < 0 || this->stride < 0)
        {
            std::cerr << "[Error] Invalid matrix dimensions in alloc_mem(): row=" << this->row 
                      << ", stride=" << this->stride << "\n";
            this->data = nullptr;
            this->ext_buff = false;
            this->memory = 0;
            return;
        }
        
        // Check for integer overflow: row * stride might overflow
        if (this->row > 0 && this->stride > INT_MAX / this->row)
        {
            std::cerr << "[Error] Matrix size too large, integer overflow: row=" << this->row 
                      << ", stride=" << this->stride << "\n";
            this->data = nullptr;
            this->ext_buff = false;
            this->memory = 0;
            return;
        }
        
        this->ext_buff = false;
        this->memory = this->row * this->stride;
        
        // Handle empty matrix case (memory = 0)
        if (this->memory == 0)
        {
            this->data = nullptr;
            return;
        }
        
        // Use nothrow new to return nullptr on failure instead of throwing exception
        // This allows callers to check for nullptr, which is consistent with existing code
        this->data = new(std::nothrow) float[this->memory];
        
        // If allocation failed, data will be nullptr (caller should check)
        if (this->data == nullptr)
        {
            this->memory = 0;
        }
    }

    /**
     * @name Mat::Mat()
     * @brief Constructor - default constructor: create a 1x1 matrix with only a zero element.
     * @note If memory allocation fails, the object will be in an invalid state (data = nullptr).
     *       Caller should check the data pointer before using the matrix.
     */
    Mat::Mat()
        : row(1), col(1), pad(0), stride(1), element(1), memory(1),
          data(nullptr), temp(nullptr),
          ext_buff(false), sub_matrix(false)
    {
        // memory will be recalculated by alloc_mem() based on row * stride
        alloc_mem();
        if (this->data == nullptr)
        {
            std::cerr << "[>>> Error ! <<<] Memory allocation failed in alloc_mem()\n";
            // Memory allocation failed, object is in invalid state (data = nullptr)
            // Caller should check data pointer before using the matrix
            return;
        }
        // Initialize all elements to zero
        std::memset(this->data, 0, this->memory * sizeof(float));
    }

    /**
     * @name Mat::Mat(int rows, int cols)
     * @brief Constructor - create a matrix with the specified number of rows and columns.
     * @param rows Number of rows (must be non-negative)
     * @param cols Number of columns (must be non-negative)
     * @note If rows or cols is negative, the object will be in an invalid state.
     * @note If memory allocation fails, the object will be in an invalid state (data = nullptr).
     *       Caller should check the data pointer before using the matrix.
     */
    Mat::Mat(int rows, int cols)
        : row(rows), col(cols), pad(0), stride(cols),
          element(rows * cols), memory(rows * cols),
          data(nullptr), temp(nullptr),
          ext_buff(false), sub_matrix(false)
    {
        // Parameter validation: check if rows and cols are non-negative
        if (rows < 0 || cols < 0)
        {
            std::cerr << "[Error] Invalid matrix dimensions: rows=" << rows 
                      << ", cols=" << cols << " (must be non-negative)\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        // Check for integer overflow: rows * cols might overflow
        if (rows > 0 && cols > INT_MAX / rows)
        {
            std::cerr << "[Error] Matrix size too large, integer overflow: rows=" << rows 
                      << ", cols=" << cols << "\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        // memory will be recalculated by alloc_mem() based on row * stride
        alloc_mem();
        if (this->data == nullptr)
        {
            std::cerr << "[>>> Error ! <<<] Memory allocation failed in alloc_mem()\n";
            // Memory allocation failed, object is in invalid state (data = nullptr)
            // Caller should check data pointer before using the matrix
            return;
        }
        // Initialize all elements to zero
        std::memset(this->data, 0, this->memory * sizeof(float));
    }
    /**
     * @name Mat::Mat(int rows, int cols, int stride)
     * @brief Constructor - create a matrix with the specified number of rows, columns and stride.
     * @param rows Number of rows (must be non-negative)
     * @param cols Number of columns (must be non-negative)
     * @param stride Stride (number of elements in a row, must be >= cols)
     * @note If rows, cols is negative, or stride < cols, the object will be in an invalid state.
     * @note If memory allocation fails, the object will be in an invalid state (data = nullptr).
     *       Caller should check the data pointer before using the matrix.
     */
    Mat::Mat(int rows, int cols, int stride)
        : row(rows), col(cols), pad(stride - cols), stride(stride),
          element(rows * cols), memory(rows * stride),
          data(nullptr), temp(nullptr),
          ext_buff(false), sub_matrix(false)
    {
        // Parameter validation: check if rows, cols are non-negative
        if (rows < 0 || cols < 0)
        {
            std::cerr << "[Error] Invalid matrix dimensions: rows=" << rows 
                      << ", cols=" << cols << " (must be non-negative)\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        // Validate stride: must be >= cols (padding cannot be negative)
        if (stride < cols)
        {
            std::cerr << "[Error] Invalid stride: stride=" << stride 
                      << " must be >= cols=" << cols << "\n";
            this->data = nullptr;
            this->memory = 0;
            this->pad = 0;  // Reset pad to avoid negative value
            return;
        }
        
        // Check for integer overflow: rows * cols and rows * stride might overflow
        if (rows > 0 && cols > INT_MAX / rows)
        {
            std::cerr << "[Error] Matrix size too large, integer overflow: rows=" << rows 
                      << ", cols=" << cols << "\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        if (rows > 0 && stride > INT_MAX / rows)
        {
            std::cerr << "[Error] Matrix size too large, integer overflow: rows=" << rows 
                      << ", stride=" << stride << "\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        // memory will be recalculated by alloc_mem() based on row * stride
        alloc_mem();
        if (this->data == nullptr)
        {
            std::cerr << "[>>> Error ! <<<] Memory allocation failed in alloc_mem()\n";
            // Memory allocation failed, object is in invalid state (data = nullptr)
            // Caller should check data pointer before using the matrix
            return;
        }
        // Initialize all elements to zero
        std::memset(this->data, 0, this->memory * sizeof(float));
    }

    /**
     * @name Mat::Mat(float *data, int rows, int cols)
     * @brief Constructor - create a matrix with the specified number of rows, columns and external data.
     * @param data Pointer to external data buffer (can be nullptr for empty matrix)
     * @param rows Number of rows (must be non-negative)
     * @param cols Number of columns (must be non-negative)
     * @note This constructor does not allocate memory. The matrix uses the external buffer.
     * @note If rows or cols is negative, the object will be in an invalid state.
     * @note The caller is responsible for ensuring the buffer is large enough and valid.
     */
    Mat::Mat(float *data, int rows, int cols)
        : row(rows), col(cols), pad(0), stride(cols),
          element(rows * cols), memory(rows * cols),
          data(data), temp(nullptr),
          ext_buff(true), sub_matrix(false)
    {
        // Parameter validation: check if rows and cols are non-negative
        if (rows < 0 || cols < 0)
        {
            std::cerr << "[Error] Invalid matrix dimensions: rows=" << rows 
                      << ", cols=" << cols << " (must be non-negative)\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        // Check for integer overflow: rows * cols might overflow
        if (rows > 0 && cols > INT_MAX / rows)
        {
            std::cerr << "[Error] Matrix size too large, integer overflow: rows=" << rows 
                      << ", cols=" << cols << "\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        // Note: data can be nullptr for empty matrix, but caller should ensure buffer validity
    }

    /**
     * @name Mat::Mat(float *data, int rows, int cols, int stride)
     * @brief Constructor - create a matrix with the specified number of rows, columns and external data.
     * @param data Pointer to external data buffer (can be nullptr for empty matrix)
     * @param rows Number of rows (must be non-negative)
     * @param cols Number of columns (must be non-negative)
     * @param stride Stride (number of elements in a row, must be >= cols)
     * @note This constructor does not allocate memory. The matrix uses the external buffer.
     * @note If rows, cols is negative, or stride < cols, the object will be in an invalid state.
     * @note The caller is responsible for ensuring the buffer is large enough and valid.
     */
    Mat::Mat(float *data, int rows, int cols, int stride)
        : row(rows), col(cols), pad(stride - cols), stride(stride),
          element(rows * cols), memory(rows * stride),
          data(data), temp(nullptr),
          ext_buff(true), sub_matrix(false)
    {
        // Parameter validation: check if rows, cols are non-negative
        if (rows < 0 || cols < 0)
        {
            std::cerr << "[Error] Invalid matrix dimensions: rows=" << rows 
                      << ", cols=" << cols << " (must be non-negative)\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        // Validate stride: must be >= cols (padding cannot be negative)
        if (stride < cols)
        {
            std::cerr << "[Error] Invalid stride: stride=" << stride 
                      << " must be >= cols=" << cols << "\n";
            this->data = nullptr;
            this->memory = 0;
            this->pad = 0;  // Reset pad to avoid negative value
            return;
        }
        
        // Check for integer overflow: rows * cols and rows * stride might overflow
        if (rows > 0 && cols > INT_MAX / rows)
        {
            std::cerr << "[Error] Matrix size too large, integer overflow: rows=" << rows 
                      << ", cols=" << cols << "\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        if (rows > 0 && stride > INT_MAX / rows)
        {
            std::cerr << "[Error] Matrix size too large, integer overflow: rows=" << rows 
                      << ", stride=" << stride << "\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        // Note: data can be nullptr for empty matrix, but caller should ensure buffer validity
    }

    /**
     * @name Mat::Mat(const Mat &src)
     * @brief Copy constructor - create a matrix with the same properties as the source matrix.
     * @param src Source matrix
     * @note If source is a submatrix view (sub_matrix && ext_buff), performs shallow copy (shares data pointer).
     *       Otherwise, performs deep copy (allocates new memory and copies data).
     * @note If memory allocation fails, the object will be in an invalid state (data = nullptr).
     *       Caller should check the data pointer before using the matrix.
     * @warning Shallow copy: If source is destroyed, the copied matrix's data pointer will be invalid.
     */
    Mat::Mat(const Mat &src)
        : row(src.row), col(src.col), pad(src.pad), stride(src.stride),
          element(src.element), memory(src.memory),
          data(nullptr), temp(nullptr),
          ext_buff(false), sub_matrix(false)
    {
        if (src.sub_matrix && src.ext_buff)
        {
            // if the source is a view (submatrix), do shallow copy
            // WARNING: This creates a shared reference. If source is destroyed, this pointer becomes invalid.
            this->data = src.data;
            this->ext_buff = true;
            this->sub_matrix = true;
        }
        else
        {
            // otherwise do deep copy
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
                
                // Copy data row by row to handle different strides correctly
                // This ensures correct copying even if source and destination have different strides
                for (int i = 0; i < this->row; ++i)
                {
                    std::memcpy(
                        &this->data[i * this->stride],
                        &src.data[i * src.stride],
                        this->col * sizeof(float)
                    );
                }
            }
            // If src.data == nullptr, this->data remains nullptr (empty matrix)
        }
    }

    /**
     * @name ~Mat()
     * @brief Destructor - free the memory allocated for the matrix.
     * @note Only deletes memory if it was allocated by this object (ext_buff == false).
     *       External buffers are not deleted.
     * @note temp buffer is always deleted if it exists (assumed to be allocated by this object).
     */
    Mat::~Mat()
    {
        // Only delete data if it was allocated by this object (not external buffer)
        if (!this->ext_buff && this->data != nullptr)
        {
            delete[] this->data;
            this->data = nullptr;  // Set to nullptr after deletion (good practice)
        }
        
        // Delete temporary buffer if it exists
        // Note: temp is assumed to be allocated by this object, not external
        if (this->temp != nullptr)
        {
            delete[] this->temp;
            this->temp = nullptr;  // Set to nullptr after deletion (good practice)
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
     * @brief Copy the elements of the source matrix into the destination matrix. 
     *        The dimension of the current matrix must be larger than the source matrix.
     * @brief This one does not share memory with the source matrix.
     * @param src Source matrix (must be valid and non-empty)
     * @param row_pos Start row position of the destination matrix (must be non-negative)
     * @param col_pos Start column position of the destination matrix (must be non-negative)
     * @return TINY_OK on success, TINY_ERR_INVALID_ARG on error
     */
    tiny_error_t Mat::copy_paste(const Mat &src, int row_pos, int col_pos)
    {
        // Check for null pointers
        if (this->data == nullptr)
        {
            std::cerr << "[Error] copy_paste: destination matrix data pointer is null\n";
            return TINY_ERR_INVALID_ARG;
        }
        if (src.data == nullptr)
        {
            std::cerr << "[Error] copy_paste: source matrix data pointer is null\n";
            return TINY_ERR_INVALID_ARG;
        }
        
        // Validate source matrix dimensions
        if (src.row <= 0 || src.col <= 0)
        {
            std::cerr << "[Error] copy_paste: invalid source matrix dimensions: rows=" 
                      << src.row << ", cols=" << src.col << "\n";
            return TINY_ERR_INVALID_ARG;
        }
        
        // Validate position parameters (must be non-negative)
        if (row_pos < 0 || col_pos < 0)
        {
            std::cerr << "[Error] copy_paste: invalid position: row_pos=" << row_pos 
                      << ", col_pos=" << col_pos << " (must be non-negative)\n";
            return TINY_ERR_INVALID_ARG;
        }
        
        // Validate destination matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] copy_paste: invalid destination matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return TINY_ERR_INVALID_ARG;
        }
        
        // Check if source matrix fits in destination at the specified position
        if ((row_pos + src.row) > this->row)
        {
            std::cerr << "[Error] copy_paste: source matrix exceeds destination row boundary: "
                      << "row_pos=" << row_pos << ", src.rows=" << src.row 
                      << ", dest.rows=" << this->row << "\n";
            return TINY_ERR_INVALID_ARG;
        }
        if ((col_pos + src.col) > this->col)
        {
            std::cerr << "[Error] copy_paste: source matrix exceeds destination column boundary: "
                      << "col_pos=" << col_pos << ", src.cols=" << src.col 
                      << ", dest.cols=" << this->col << "\n";
            return TINY_ERR_INVALID_ARG;
        }
        
        // Copy data row by row (handles different strides correctly)
        for (int r = 0; r < src.row; r++)
        {
            memcpy(&this->data[(r + row_pos) * this->stride + col_pos], 
                   &src.data[r * src.stride], 
                   src.col * sizeof(float));
        }

        return TINY_OK;
    }

    /**
     * @name Mat::copy_head(const Mat &src)
     * @brief Copy the header (metadata) of the source matrix into the destination matrix. 
     *        The data pointer is shared (shallow copy).
     * @param src Source matrix (must be valid)
     * @return TINY_OK on success, TINY_ERR_INVALID_ARG on error
     * @warning This function performs a SHALLOW COPY. The destination matrix shares the 
     *          data pointer with the source matrix. If the source matrix is destroyed, 
     *          the destination matrix's data pointer will become invalid.
     * @note The temp pointer is NOT shared (set to nullptr) to prevent double-free issues.
     *       Each object manages its own temp buffer independently.
     */
    tiny_error_t Mat::copy_head(const Mat &src)
    {
        // Delete current data if it was allocated by this object
        if (!this->ext_buff && this->data != nullptr)
        {
            delete[] this->data;
            this->data = nullptr;
        }
        
        // Delete current temp if it exists (assuming it was allocated by this object)
        if (this->temp != nullptr)
        {
            delete[] this->temp;
            this->temp = nullptr;
        }
        
        // Copy all metadata from source matrix
        this->row = src.row;
        this->col = src.col;
        this->element = src.element;
        this->pad = src.pad;
        this->stride = src.stride;
        this->memory = src.memory;
        
        // Shallow copy: share data pointer ONLY if source uses external buffer or is a submatrix view
        // If source owns its memory (ext_buff=false), we must NOT share the pointer to avoid double-free
        // In that case, copy_head should not be used - use copy assignment or copy constructor instead
        if (src.ext_buff || src.sub_matrix)
        {
            // Source uses external buffer or is a view - safe to share pointer
            // WARNING: If source is destroyed, this pointer becomes invalid
            this->data = src.data;
            this->ext_buff = src.ext_buff;
            this->sub_matrix = src.sub_matrix;
        }
        else
        {
            // Source owns its memory - cannot share pointer (would cause double-free)
            // This is an error condition - copy_head should only be used for external buffers or views
            std::cerr << "[Error] copy_head: source matrix owns its memory (ext_buff=false). "
                      << "Cannot share pointer - would cause double-free. "
                      << "Use copy assignment or copy constructor instead.\n";
            this->data = nullptr;
            this->ext_buff = false;
            this->sub_matrix = false;
            return TINY_ERR_INVALID_ARG;
        }
        
        // Do NOT share temp pointer - temp is a temporary buffer that should not be shared
        // Setting temp to nullptr prevents double-free issues when either object is destroyed
        // Each object should manage its own temp buffer if needed
        this->temp = nullptr;

        return TINY_OK;
    }

    /**
     * @name Mat::view_roi(int start_row, int start_col, int roi_rows, int roi_cols)
     * @brief Make a shallow copy of ROI matrix. Create a view of the ROI matrix. 
     *        Low level function. Unlike ESP-DSP, it is not allowed to setup stride here, 
     *        stride is automatically calculated inside the function.
     * @param start_row Start row position of source matrix (must be non-negative)
     * @param start_col Start column position of source matrix (must be non-negative)
     * @param roi_rows Size of row elements of source matrix to copy (must be positive)
     * @param roi_cols Size of column elements of source matrix to copy (must be positive)
     * @return result matrix size roi_rows x roi_cols, or empty matrix on error
     * @warning The returned matrix is a VIEW (shallow copy) that shares data with the source matrix.
     *          If the source matrix is destroyed, the view's data pointer will become invalid.
     * @note The stride of the result matrix is inherited from the source matrix.
     */
    Mat Mat::view_roi(int start_row, int start_col, int roi_rows, int roi_cols) const
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] view_roi: source matrix data pointer is null\n";
            return Mat();  // Return empty matrix as error indicator
        }
        
        // Validate source matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] view_roi: invalid source matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return Mat();
        }
        
        // Validate position parameters (must be non-negative)
        if (start_row < 0 || start_col < 0)
        {
            std::cerr << "[Error] view_roi: invalid position: start_row=" << start_row 
                      << ", start_col=" << start_col << " (must be non-negative)\n";
            return Mat();
        }
        
        // Validate ROI size parameters (must be positive)
        if (roi_rows <= 0 || roi_cols <= 0)
        {
            std::cerr << "[Error] view_roi: invalid ROI size: roi_rows=" << roi_rows 
                      << ", roi_cols=" << roi_cols << " (must be positive)\n";
            return Mat();
        }
        
        // Check if ROI fits within source matrix boundaries
        if ((start_row + roi_rows) > this->row)
        {
            std::cerr << "[Error] view_roi: ROI exceeds row boundary: start_row=" << start_row 
                      << ", roi_rows=" << roi_rows << ", source.rows=" << this->row << "\n";
            return Mat();
        }
        if ((start_col + roi_cols) > this->col)
        {
            std::cerr << "[Error] view_roi: ROI exceeds column boundary: start_col=" << start_col 
                      << ", roi_cols=" << roi_cols << ", source.cols=" << this->col << "\n";
            return Mat();
        }
        
        // Validate stride: must be >= roi_cols (padding cannot be negative)
        if (this->stride < roi_cols)
        {
            std::cerr << "[Error] view_roi: stride < roi_cols: stride=" << this->stride 
                      << ", roi_cols=" << roi_cols << "\n";
            return Mat();
        }
        
        // Check for integer overflow
        if (roi_rows > 0 && this->stride > INT_MAX / roi_rows)
        {
            std::cerr << "[Error] view_roi: integer overflow: roi_rows=" << roi_rows 
                      << ", stride=" << this->stride << "\n";
            return Mat();
        }
        if (roi_rows > 0 && roi_cols > INT_MAX / roi_rows)
        {
            std::cerr << "[Error] view_roi: integer overflow: roi_rows=" << roi_rows 
                      << ", roi_cols=" << roi_cols << "\n";
            return Mat();
        }
        
        // Create ROI view (shallow copy)
        Mat result;
        result.row = roi_rows;
        result.col = roi_cols;
        result.stride = this->stride;
        result.pad = this->stride - roi_cols;  // Now guaranteed to be non-negative
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
     * @brief Make a shallow copy of ROI matrix. Create a view of the ROI matrix using ROI structure.
     * @param roi Rectangular area of interest (roi.pos_x, roi.pos_y must be non-negative, 
     *            roi.width, roi.height must be positive)
     * @return result matrix size roi.height x roi.width, or empty matrix on error
     * @warning The returned matrix is a VIEW (shallow copy) that shares data with the source matrix.
     *          If the source matrix is destroyed, the view's data pointer will become invalid.
     * @note This is a convenience wrapper that calls view_roi(roi.pos_y, roi.pos_x, roi.height, roi.width).
     */
    Mat Mat::view_roi(const Mat::ROI &roi) const
    {
        return view_roi(roi.pos_y, roi.pos_x, roi.height, roi.width);
    }

    /**
     * @name Mat::copy_roi(int start_row, int start_col, int height, int width)
     * @brief Make a deep copy of matrix. Compared to view_roi(), this one is a deep copy, 
     *        not sharing memory with the source matrix.
     * @param start_row Start row position of source matrix to copy (must be non-negative)
     * @param start_col Start column position of source matrix to copy (must be non-negative)
     * @param height Size of row elements of source matrix to copy (must be positive)
     * @param width Size of column elements of source matrix to copy (must be positive)
     * @return result matrix size height x width, or empty matrix on error
     * @note The returned matrix is a DEEP COPY with its own memory. It is independent 
     *       of the source matrix and can be safely used after the source is destroyed.
     */
    Mat Mat::copy_roi(int start_row, int start_col, int height, int width)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] copy_roi: source matrix data pointer is null\n";
            return Mat();  // Return empty matrix as error indicator
        }
        
        // Validate source matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] copy_roi: invalid source matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return Mat();
        }
        
        // Validate position parameters (must be non-negative)
        if (start_row < 0 || start_col < 0)
        {
            std::cerr << "[Error] copy_roi: invalid position: start_row=" << start_row 
                      << ", start_col=" << start_col << " (must be non-negative)\n";
            return Mat();
        }
        
        // Validate size parameters (must be positive)
        if (height <= 0 || width <= 0)
        {
            std::cerr << "[Error] copy_roi: invalid size: height=" << height 
                      << ", width=" << width << " (must be positive)\n";
            return Mat();
        }
        
        // Check if ROI fits within source matrix boundaries
        if ((start_row + height) > this->row)
        {
            std::cerr << "[Error] copy_roi: ROI exceeds row boundary: start_row=" << start_row 
                      << ", height=" << height << ", source.rows=" << this->row << "\n";
            return Mat();
        }
        if ((start_col + width) > this->col)
        {
            std::cerr << "[Error] copy_roi: ROI exceeds column boundary: start_col=" << start_col 
                      << ", width=" << width << ", source.cols=" << this->col << "\n";
            return Mat();
        }

        // Create result matrix (deep copy)
        Mat result(height, width);
        
        // Check if result matrix was created successfully
        if (result.data == nullptr)
        {
            std::cerr << "[Error] copy_roi: failed to allocate memory for result matrix\n";
            return Mat();
        }

        // Deep copy the data from the source matrix row by row
        // This handles different strides correctly
        for (int r = 0; r < result.row; r++)
        {
            memcpy(&result.data[r * result.stride], 
                   &this->data[(r + start_row) * this->stride + start_col], 
                   result.col * sizeof(float));
        }

        return result;
    }

    /**
     * @name Mat::copy_roi(const Mat::ROI &roi)
     * @brief Make a deep copy of matrix using ROI structure. Compared to view_roi(), 
     *        this one is a deep copy, not sharing memory with the source matrix.
     * @param roi Rectangular area of interest (roi.pos_x, roi.pos_y must be non-negative, 
     *            roi.width, roi.height must be positive)
     * @return result matrix size roi.height x roi.width, or empty matrix on error
     * @note The returned matrix is a DEEP COPY with its own memory. It is independent 
     *       of the source matrix and can be safely used after the source is destroyed.
     * @note This is a convenience wrapper that calls copy_roi(roi.pos_y, roi.pos_x, roi.height, roi.width).
     */
    Mat Mat::copy_roi(const Mat::ROI &roi)
    {
        return copy_roi(roi.pos_y, roi.pos_x, roi.height, roi.width);
    }

    /**
     * @name Mat::block(int start_row, int start_col, int block_rows, int block_cols)
     * @brief Get a block (submatrix) of the matrix. This is a deep copy operation.
     * @param start_row Start row position of the block (must be non-negative)
     * @param start_col Start column position of the block (must be non-negative)
     * @param block_rows Number of rows in the block (must be positive)
     * @param block_cols Number of columns in the block (must be positive)
     * @return result matrix size block_rows x block_cols, or empty matrix on error
     * @note The returned matrix is a DEEP COPY with its own memory. It is independent 
     *       of the source matrix and can be safely used after the source is destroyed.
     * @note This function is similar to copy_roi(), but uses element-by-element access.
     *       For better performance with large blocks, consider using copy_roi() instead.
     */
    Mat Mat::block(int start_row, int start_col, int block_rows, int block_cols)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] block: source matrix data pointer is null\n";
            return Mat();  // Return empty matrix as error indicator
        }
        
        // Validate source matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] block: invalid source matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return Mat();
        }
        
        // Boundary check: validate position parameters (must be non-negative)
        if (start_row < 0 || start_col < 0)
        {
            std::cerr << "[Error] block: invalid position: start_row=" << start_row 
                      << ", start_col=" << start_col << " (must be non-negative)\n";
            return Mat();
        }
        
        // Boundary check: validate block size parameters (must be positive)
        if (block_rows <= 0 || block_cols <= 0)
        {
            std::cerr << "[Error] block: invalid block size: block_rows=" << block_rows 
                      << ", block_cols=" << block_cols << " (must be positive)\n";
            return Mat();
        }
        
        // Check if block fits within source matrix boundaries
        if ((start_row + block_rows) > this->row)
        {
            std::cerr << "[Error] block: block exceeds row boundary: start_row=" << start_row 
                      << ", block_rows=" << block_rows << ", source.rows=" << this->row << "\n";
            return Mat();
        }
        if ((start_col + block_cols) > this->col)
        {
            std::cerr << "[Error] block: block exceeds column boundary: start_col=" << start_col 
                      << ", block_cols=" << block_cols << ", source.cols=" << this->col << "\n";
            return Mat();
        }
        
        // Create result matrix
        Mat result(block_rows, block_cols);
        
        // Check if result matrix was created successfully
        if (result.data == nullptr)
        {
            std::cerr << "[Error] block: failed to allocate memory for result matrix\n";
            return Mat();
        }
        
        // Copy block data element by element
        // Note: This uses operator() which handles stride correctly, but is slower than memcpy
        // For better performance, consider using copy_roi() which uses memcpy
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
     * @param row1 The index of the first row to swap (must be in range [0, row-1])
     * @param row2 The index of the second row to swap (must be in range [0, row-1])
     * @note If row1 == row2, the function returns immediately without doing anything.
     * @note This function is commonly used in matrix operations like Gaussian elimination with row pivoting.
     */
    void Mat::swap_rows(int row1, int row2)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] swap_rows: matrix data pointer is null\n";
            return;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] swap_rows: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return;
        }
        
        // Validate row indices
        if (row1 < 0 || row1 >= this->row)
        {
            std::cerr << "[Error] swap_rows: row1 index out of range: row1=" << row1 
                      << ", matrix.rows=" << this->row << "\n";
            return;
        }
        if (row2 < 0 || row2 >= this->row)
        {
            std::cerr << "[Error] swap_rows: row2 index out of range: row2=" << row2 
                      << ", matrix.rows=" << this->row << "\n";
            return;
        }
        
        // Optimization: if same row, no need to swap
        if (row1 == row2)
        {
            return;
        }
        
        // Allocate temporary buffer for row swap
        // Note: Using new/delete here is acceptable for this operation,
        // but could be optimized by using the matrix's temp buffer if available
        float *temp_row = new(std::nothrow) float[this->col];
        if (temp_row == nullptr)
        {
            std::cerr << "[Error] swap_rows: failed to allocate temporary buffer\n";
            return;
        }
        
        // Swap rows using memcpy (handles stride correctly)
        memcpy(temp_row, &this->data[row1 * this->stride], this->col * sizeof(float));
        memcpy(&this->data[row1 * this->stride], &this->data[row2 * this->stride], this->col * sizeof(float));
        memcpy(&this->data[row2 * this->stride], temp_row, this->col * sizeof(float));
        
        delete[] temp_row;
    }

    /**
     * @name Mat::swap_cols(int col1, int col2)
     * @brief Swap two columns of the matrix.
     * @param col1 The index of the first column to swap (must be in range [0, col-1])
     * @param col2 The index of the second column to swap (must be in range [0, col-1])
     * @note If col1 == col2, the function returns immediately without doing anything.
     * @note Useful for column pivoting in algorithms like Gaussian elimination with column pivoting.
     * @note This function swaps columns element by element, which correctly handles stride.
     */
    void Mat::swap_cols(int col1, int col2)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] swap_cols: matrix data pointer is null\n";
            return;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] swap_cols: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return;
        }
        
        // Validate column indices
        if (col1 < 0 || col1 >= this->col)
        {
            std::cerr << "[Error] swap_cols: col1 index out of range: col1=" << col1 
                      << ", matrix.cols=" << this->col << "\n";
            return;
        }
        if (col2 < 0 || col2 >= this->col)
        {
            std::cerr << "[Error] swap_cols: col2 index out of range: col2=" << col2 
                      << ", matrix.cols=" << this->col << "\n";
            return;
        }
        
        // Optimization: if same column, no need to swap
        if (col1 == col2)
        {
            return;
        }
        
        // Swap columns element by element (considering stride)
        // Note: This approach correctly handles different stride values
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
     * @note Only clears the actual matrix elements (col elements per row), not the padding area.
     * @note If the matrix has padding (stride > col), the padding elements are not cleared.
     */
    void Mat::clear(void)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] clear: matrix data pointer is null\n";
            return;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            // Empty matrix, nothing to clear
            return;
        }
        
        // Clear matrix row by row (handles stride correctly)
        // Only clear the actual matrix elements (col elements), not the padding
        for (int row = 0; row < this->row; row++)
        {
            memset(this->data + (row * this->stride), 0, this->col * sizeof(float));
        }
    }

    // ============================================================================
    // Arithmetic Operators
    // ============================================================================
    /**
     * @name Mat::operator=(const Mat &src)
     * @brief Copy assignment operator - copy the elements of the source matrix into the destination matrix.
     * @param src Source matrix to copy from
     * @return Reference to this matrix
     * @note Compared to the copy constructor, this operator is used for existing matrices.
     *       If dimensions differ, memory will be reallocated.
     * @note Assignment to sub-matrix views is not allowed.
     * @warning If memory allocation fails, the matrix may be in an invalid state.
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
        
        // Check source matrix validity
        if (src.data == nullptr)
        {
            std::cerr << "[Error] operator=: source matrix data pointer is null\n";
            return *this;
        }

        // 3. If dimensions differ, reallocate memory
        if (this->row != src.row || this->col != src.col)
        {
            if (!this->ext_buff && this->data != nullptr)
            {
                delete[] this->data;
                this->data = nullptr;
            }

            // Update dimensions and memory info
            this->row = src.row;
            this->col = src.col;
            this->stride = src.col; // Follow source's logical stride (no padding)
            this->pad = 0;
            
            // Check for integer overflow
            if (this->row > 0 && this->col > INT_MAX / this->row)
            {
                std::cerr << "[Error] operator=: integer overflow in element calculation\n";
                this->data = nullptr;
                return *this;
            }
            this->element = this->row * this->col;
            
            if (this->row > 0 && this->stride > INT_MAX / this->row)
            {
                std::cerr << "[Error] operator=: integer overflow in memory calculation\n";
                this->data = nullptr;
                return *this;
            }
            this->memory = this->row * this->stride;

            this->ext_buff = false;
            this->sub_matrix = false;

            alloc_mem();
            
            // Check if memory allocation succeeded
            if (this->data == nullptr)
            {
                std::cerr << "[Error] operator=: memory allocation failed\n";
                return *this;
            }
        }
        
        // Check if this->data is valid before copying
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator=: destination matrix data pointer is null\n";
            return *this;
        }

        // 4. Data copy (row-wise, handles different strides correctly)
        for (int r = 0; r < this->row; ++r)
        {
            std::memcpy(this->data + r * this->stride, 
                       src.data + r * src.stride, 
                       this->col * sizeof(float));
        }

        return *this;
    }

    /**
     * @name Mat::operator+=(const Mat &A)
     * @brief Element-wise addition of another matrix to this matrix.
     * @param A The matrix to add (must have same dimensions as this matrix)
     * @return Mat& Reference to the current matrix
     * @note This function performs element-wise addition: this[i,j] += A[i,j]
     * @note The function automatically handles padding and uses optimized vectorized operations when possible.
     */
    Mat &Mat::operator+=(const Mat &A)
    {
        // Check for null pointers
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator+=: this matrix data pointer is null\n";
            return *this;
        }
        if (A.data == nullptr)
        {
            std::cerr << "[Error] operator+=: source matrix data pointer is null\n";
            return *this;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] operator+=: invalid this matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return *this;
        }
        if (A.row <= 0 || A.col <= 0)
        {
            std::cerr << "[Error] operator+=: invalid source matrix dimensions: rows=" 
                      << A.row << ", cols=" << A.col << "\n";
            return *this;
        }

        // 1. Dimension check - matrices must have same dimensions
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
     * @param C The constant to add to each element
     * @return Mat& Reference to the current matrix
     * @note This function performs element-wise addition: this[i,j] += C
     * @note The function automatically handles padding and uses optimized vectorized operations when possible.
     */
    Mat &Mat::operator+=(float C)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator+=(float): matrix data pointer is null\n";
            return *this;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] operator+=(float): invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return *this;
        }

        // Check whether padding is present
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
     * @param A The matrix to subtract (must have same dimensions as this matrix)
     * @return Mat& Reference to the current matrix
     * @note This function performs element-wise subtraction: this[i,j] -= A[i,j]
     * @note The function automatically handles padding and uses optimized vectorized operations when possible.
     */
    Mat &Mat::operator-=(const Mat &A)
    {
        // Check for null pointers
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator-=: this matrix data pointer is null\n";
            return *this;
        }
        if (A.data == nullptr)
        {
            std::cerr << "[Error] operator-=: source matrix data pointer is null\n";
            return *this;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] operator-=: invalid this matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return *this;
        }
        if (A.row <= 0 || A.col <= 0)
        {
            std::cerr << "[Error] operator-=: invalid source matrix dimensions: rows=" 
                      << A.row << ", cols=" << A.col << "\n";
            return *this;
        }

        // 1. Dimension check - matrices must have same dimensions
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
     * @param C The constant to subtract from each element
     * @return Mat& Reference to the current matrix
     * @note This function performs element-wise subtraction: this[i,j] -= C
     * @note The function automatically handles padding and uses optimized vectorized operations when possible.
     * @note On ESP32, this uses addc with -C since subc is not available in DSP library.
     */
    Mat &Mat::operator-=(float C)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator-=(float): matrix data pointer is null\n";
            return *this;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] operator-=(float): invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return *this;
        }

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
     * @param m The matrix to multiply with (must have compatible dimensions: this.col == m.row)
     * @return Mat& Reference to the current matrix
     * @note Matrix multiplication requires: this.col == m.row
     *       Result dimensions: this.row x m.col
     * @note This function creates a temporary copy to avoid overwriting data during computation.
     */
    Mat &Mat::operator*=(const Mat &m)
    {
        // Check for null pointers
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator*=: this matrix data pointer is null\n";
            return *this;
        }
        if (m.data == nullptr)
        {
            std::cerr << "[Error] operator*=: source matrix data pointer is null\n";
            return *this;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] operator*=: invalid this matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return *this;
        }
        if (m.row <= 0 || m.col <= 0)
        {
            std::cerr << "[Error] operator*=: invalid source matrix dimensions: rows=" 
                      << m.row << ", cols=" << m.col << "\n";
            return *this;
        }

        // 1. Dimension check - matrix multiplication requires: this.col == m.row
        if (this->col != m.row)
        {
            std::cerr << "[Error] Matrix multiplication failed: incompatible dimensions ("
                      << this->row << "x" << this->col << " * "
                      << m.row << "x" << m.col << ")\n";
            return *this;
        }

        // 2. Prepare temp matrix (in case overwriting the original data)
        // Create a copy of this matrix to avoid overwriting during computation
        Mat temp = this->copy_roi(0, 0, this->row, this->col);
        
        // Check if copy_roi succeeded
        if (temp.data == nullptr)
        {
            std::cerr << "[Error] operator*=: failed to create temporary matrix copy\n";
            return *this;
        }

        // 3. Check whether padding is present in either matrix
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
     * @brief Element-wise multiplication by a constant.
     * @param num The constant multiplier
     * @return Mat& Reference to the current matrix
     * @note This function performs element-wise multiplication: this[i,j] *= num
     * @note The function automatically handles padding and uses optimized vectorized operations when possible.
     */
    Mat &Mat::operator*=(float num)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator*=(float): matrix data pointer is null\n";
            return *this;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] operator*=(float): invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return *this;
        }

        // Check whether padding is present
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
     * @param B The matrix divisor (must have same dimensions as this matrix, and no zero elements)
     * @return Mat& Reference to the current matrix
     * @note This function performs element-wise division: this[i,j] /= B[i,j]
     * @warning Division by zero will cause an error. All elements of B must be non-zero.
     */
    Mat &Mat::operator/=(const Mat &B)
    {
        // Check for null pointers
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator/=: this matrix data pointer is null\n";
            return *this;
        }
        if (B.data == nullptr)
        {
            std::cerr << "[Error] operator/=: divisor matrix data pointer is null\n";
            return *this;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] operator/=: invalid this matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return *this;
        }
        if (B.row <= 0 || B.col <= 0)
        {
            std::cerr << "[Error] operator/=: invalid divisor matrix dimensions: rows=" 
                      << B.row << ", cols=" << B.col << "\n";
            return *this;
        }

        // 1. Dimension check - matrices must have same dimensions
        if ((this->row != B.row) || (this->col != B.col))
        {
            std::cerr << "[Error] Matrix division failed: Dimension mismatch ("
                      << this->row << "x" << this->col << " vs "
                      << B.row << "x" << B.col << ")\n";
            return *this;
        }

        // 2. Zero division check - scan for near-zero elements
        bool zero_found = false;
        const float epsilon = 1e-9f;
        for (int i = 0; i < B.row; ++i)
        {
            for (int j = 0; j < B.col; ++j)
            {
                if (fabs(B(i, j)) < epsilon)
                {
                    zero_found = true;
                    std::cerr << "[Error] Matrix division failed: Division by zero detected at position ("
                              << i << ", " << j << ")\n";
                    break;
                }
            }
            if (zero_found)
                break;
        }

        if (zero_found)
        {
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
     * @param num The constant divisor (must be non-zero)
     * @return Mat& Reference to the current matrix
     * @note This function performs element-wise division: this[i,j] /= num
     * @note The function uses multiplication by 1/num for efficiency (division is slower than multiplication).
     * @note The function automatically handles padding and uses optimized vectorized operations when possible.
     * @warning Division by zero will cause an error.
     */
    Mat &Mat::operator/=(float num)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator/=(float): matrix data pointer is null\n";
            return *this;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] operator/=(float): invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return *this;
        }

        // 1. Check division by zero
        const float epsilon = 1e-9f;
        if (fabs(num) < epsilon)
        {
            std::cerr << "[Error] Matrix division by zero is undefined (divisor=" << num << ")\n";
            return *this;
        }

        // 2. Determine if padding handling is needed
        bool need_padding_handling = (this->pad > 0);

        // Use multiplication by inverse for better performance (division is slower)
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
     * @param num The exponent (integer, must be non-negative)
     * @return Mat New matrix after exponentiation
     * @note This function performs element-wise exponentiation: result[i,j] = this[i,j]^num
     * @note For num=0, all elements become 1.0. For num=1, returns a copy of the matrix.
     * @warning Negative exponents are not supported.
     */
    Mat Mat::operator^(int num)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator^: matrix data pointer is null\n";
            return Mat();  // Return empty matrix as error indicator
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] operator^: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return Mat();
        }

        // Handle special cases
        if (num == 0)
        {
            // Any number to the power of 0 is 1
            Mat result(this->row, this->col, this->stride);
            if (result.data == nullptr)
            {
                std::cerr << "[Error] operator^: failed to allocate memory for result matrix\n";
                return Mat();
            }
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
            std::cerr << "[Error] Negative exponent not supported in operator^ (exponent=" << num << ")\n";
            return Mat(*this); // Return a copy without modification
        }

        // General case: positive exponent > 1
        Mat result(this->row, this->col, this->stride);
        if (result.data == nullptr)
        {
            std::cerr << "[Error] operator^: failed to allocate memory for result matrix\n";
            return Mat();
        }
        
        // Element-wise exponentiation using iterative multiplication
        // Note: For large exponents, this could be optimized using fast exponentiation,
        // but for typical use cases (small exponents), this is acceptable
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
     * @name Mat::transpose()
     * @brief Transpose the matrix. Returns a new matrix with rows and columns swapped.
     * @return Transposed matrix (col x row), or empty matrix on error
     * @note If this matrix is m x n, the result will be n x m.
     * @note The transpose operation: result[j][i] = this[i][j]
     */
    Mat Mat::transpose()
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] transpose: matrix data pointer is null\n";
            return Mat();  // Return empty matrix as error indicator
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] transpose: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return Mat();
        }

        // Create result matrix with swapped dimensions
        Mat result(this->col, this->row);
        
        // Check if result matrix was created successfully
        if (result.data == nullptr)
        {
            std::cerr << "[Error] transpose: failed to allocate memory for result matrix\n";
            return Mat();
        }
        
        // Transpose: swap rows and columns
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
     * @return float The determinant value, or 0.0f on error
     * @note For small matrices (n <= 4), uses Laplace expansion (more accurate).
     *       For larger matrices, uses LU decomposition (O(n³)) for better efficiency.
     * @note Determinant can be 0.0f for singular matrices, which is a valid result.
     *       Use error checking to distinguish between error and zero determinant.
     */
    float Mat::determinant()
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] determinant: matrix data pointer is null\n";
            return 0.0f;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] determinant: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return 0.0f;
        }

        // Check if matrix is square
        if (this->row != this->col)
        {
            std::cerr << "[Error] Determinant requires a square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
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
     * @return float The determinant value, or 0.0f on error
     * @note Time complexity: O(n!) - suitable only for small matrices (n <= 4).
     *       Uses recursive method with first row expansion.
     * @note For n=1 and n=2, uses direct formulas for efficiency.
     * @note Determinant can be 0.0f for singular matrices, which is a valid result.
     */
    float Mat::determinant_laplace()
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] determinant_laplace: matrix data pointer is null\n";
            return 0.0f;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] determinant_laplace: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return 0.0f;
        }

        // Check if matrix is square
        if (this->row != this->col)
        {
            std::cerr << "[Error] Determinant requires a square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            return 0.0f;
        }

        int n = this->row;
        
        // Base case: 1x1 matrix
        if (n == 1)
        {
            return this->data[0];
        }
        
        // Base case: 2x2 matrix (direct formula)
        if (n == 2)
        {
            return this->data[0] * this->data[this->stride + 1] - 
                   this->data[1] * this->data[this->stride];
        }

        // Recursive case: use Laplace expansion along first row
        float det = 0.0f;
        for (int j = 0; j < n; ++j)
        {
            Mat minor_mat = this->minor(0, j);
            
            // Check if minor matrix was created successfully
            if (minor_mat.data == nullptr)
            {
                std::cerr << "[Error] determinant_laplace: failed to create minor matrix at (0, " << j << ")\n";
                return 0.0f;
            }
            
            // Compute cofactor: (-1)^(i+j) * det(minor)
            float cofactor_val = ((j % 2 == 0) ? 1.0f : -1.0f) * minor_mat.determinant_laplace();
            det += this->data[j] * cofactor_val;
        }
        
        return det;
    }

    /**
     * @name Mat::determinant_lu()
     * @brief Compute the determinant using LU decomposition.
     * @return float The determinant value, or 0.0f on error
     * @note Time complexity: O(n³) - efficient for large matrices.
     *       Formula: det(A) = det(P) * det(L) * det(U) = det(P) * 1 * (product of U diagonal)
     *       where det(P) = (-1)^(number of row swaps)
     * @note Uses pivoting for numerical stability.
     * @note Determinant can be 0.0f for singular matrices, which is a valid result.
     */
    float Mat::determinant_lu()
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] determinant_lu: matrix data pointer is null\n";
            return 0.0f;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] determinant_lu: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return 0.0f;
        }

        // Check if matrix is square
        if (this->row != this->col)
        {
            std::cerr << "[Error] Determinant requires a square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            return 0.0f;
        }

        // Perform LU decomposition with pivoting for numerical stability
        LUDecomposition lu = this->lu_decompose(true);
        
        if (lu.status != TINY_OK)
        {
            // Matrix is singular or near-singular, or decomposition failed
            std::cerr << "[Warning] determinant_lu: LU decomposition failed (status=" 
                      << lu.status << "), matrix may be singular\n";
            return 0.0f;
        }
        
        // Check if decomposition matrices are valid
        if (lu.U.data == nullptr)
        {
            std::cerr << "[Error] determinant_lu: LU decomposition U matrix is null\n";
            return 0.0f;
        }

        int n = this->row;
        
        // Compute det(P): permutation matrix determinant = (-1)^(permutation signature)
        float det_P = 1.0f;
        if (lu.pivoted && lu.P.data != nullptr)
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
                bool found_mapping = false;
                while (!visited[current])
                {
                    visited[current] = true;
                    cycle_length++;
                    found_mapping = false;
                    
                    // Find where P maps current row
                    for (int j = 0; j < n; ++j)
                    {
                        if (fabsf(lu.P(current, j) - 1.0f) < 1e-6f)
                        {
                            current = j;
                            found_mapping = true;
                            break;
                        }
                    }
                    
                    // Safety check: if no mapping found, break to avoid infinite loop
                    // This should not happen for a valid permutation matrix
                    if (!found_mapping)
                    {
                        std::cerr << "[Warning] determinant_lu: Could not find mapping for row " 
                                  << current << " in permutation matrix P. Matrix may be invalid.\n";
                        break;
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
     * @return float The determinant value, or 0.0f on error
     * @note Time complexity: O(n³) - efficient for large matrices.
     *       Converts matrix to upper triangular form, then multiplies diagonal elements.
     *       Tracks row swaps to account for sign changes.
     * @note Uses partial pivoting for numerical stability.
     * @note Determinant can be 0.0f for singular matrices, which is a valid result.
     */
    float Mat::determinant_gaussian()
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] determinant_gaussian: matrix data pointer is null\n";
            return 0.0f;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] determinant_gaussian: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return 0.0f;
        }

        // Check if matrix is square
        if (this->row != this->col)
        {
            std::cerr << "[Error] Determinant requires a square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            return 0.0f;
        }

        int n = this->row;
        Mat A = Mat(*this);  // Working copy
        
        // Check if copy was successful
        if (A.data == nullptr)
        {
            std::cerr << "[Error] determinant_gaussian: failed to create working copy\n";
            return 0.0f;
        }
        
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

            // Check for singular matrix (near-zero pivot)
            if (fabsf(A(k, k)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                // Matrix is singular or near-singular
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
     * @note The adjoint matrix is the transpose of the cofactor matrix.
     *       adj(A)_ij = (-1)^(i+j) * det(minor_ji)
     *       Note: The result is stored at (j,i) to achieve transpose.
     * @return Mat The adjoint matrix, or empty Mat() on error
     * @note Time complexity: O(n² * O(det)) - expensive for large matrices.
     *       For n×n matrix, requires computing n² determinants of (n-1)×(n-1) matrices.
     */
    Mat Mat::adjoint()
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] adjoint: matrix data pointer is null\n";
            return Mat();
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] adjoint: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return Mat();
        }
        
        // Check if matrix is square
        if (this->row != this->col)
        {
            std::cerr << "[Error] Adjoint requires a square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            return Mat();
        }

        int n = this->row;
        Mat result(n, n);
        
        // Check if result matrix was created successfully
        if (result.data == nullptr)
        {
            std::cerr << "[Error] adjoint: failed to create result matrix\n";
            return Mat();
        }

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                Mat cofactor_mat = this->cofactor(i, j);
                
                // Check if cofactor matrix was created successfully
                if (cofactor_mat.data == nullptr)
                {
                    std::cerr << "[Error] adjoint: failed to create cofactor matrix at (" 
                              << i << ", " << j << ")\n";
                    return Mat();
                }
                
                // Compute cofactor value: (-1)^(i+j) * det(minor)
                float sign = ((i + j) % 2 == 0) ? 1.0f : -1.0f;
                float cofactor_val = sign * cofactor_mat.determinant();
                
                // Store at (j,i) to achieve transpose: adj(A) = C^T
                result(j, i) = cofactor_val;
            }
        }

        return result;
    }

    /**
     * @name Mat::normalize()
     * @brief Normalize the matrix by dividing each element by the matrix norm (Frobenius norm).
     * @note Normalization: M_normalized = M / ||M||_F
     *       where ||M||_F = sqrt(sum of squares of all elements)
     * @note If the matrix norm is zero or too small, normalization is skipped
     *       and a warning is printed. The matrix remains unchanged.
     * @note This function modifies the matrix in-place.
     */
    void Mat::normalize()
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] normalize: matrix data pointer is null\n";
            return;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] normalize: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return;
        }
        
        // Check for empty matrix
        if (this->row == 0 || this->col == 0)
        {
            // Empty matrix, nothing to normalize
            return;
        }
        
        float n = this->norm();
        
        // Check for invalid norm (NaN, Inf, or too small)
        if (!(n > TINY_MATH_MIN_POSITIVE_INPUT_F32))
        {
            if (n == 0.0f)
            {
                std::cerr << "[Warning] normalize: matrix norm is zero (matrix is all zeros), "
                          << "normalization skipped\n";
            }
            else if (std::isnan(n) || std::isinf(n))
            {
                std::cerr << "[Error] normalize: matrix norm is invalid (NaN or Inf), "
                          << "normalization skipped\n";
            }
            else
            {
                std::cerr << "[Warning] normalize: matrix norm is too small (" << n 
                          << "), normalization skipped\n";
            }
            return;
        }
        
        // Normalize each element
        for (int i = 0; i < this->row; ++i)
        {
            for (int j = 0; j < this->col; ++j)
            {
                (*this)(i, j) /= n;
            }
        }
    }

    /**
     * @name Mat::norm()
     * @brief Compute the Frobenius norm (Euclidean norm) of the matrix.
     * @note Frobenius norm: ||M||_F = sqrt(Σ M_ij²)
     *       This is the square root of the sum of squares of all matrix elements.
     * @note For empty matrices, returns 0.0f.
     * @note For zero matrices, returns 0.0f.
     * 
     * @return float The Frobenius norm of the matrix, or 0.0f on error
     */
    float Mat::norm() const
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] norm: matrix data pointer is null\n";
            return 0.0f;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] norm: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return 0.0f;
        }
        
        // Handle empty matrix
        if (this->row == 0 || this->col == 0)
        {
            return 0.0f;
        }
        
        float sum_sq = 0.0f;
        for (int i = 0; i < this->row; ++i)
        {
            for (int j = 0; j < this->col; ++j)
            {
                float val = (*this)(i, j);
                sum_sq += val * val;
            }
        }
        
        // Compute square root
        // Note: sum_sq should always be non-negative (sum of squares)
        float result = sqrtf(sum_sq);
        
        // Safety check: if result is invalid, return 0.0f
        if (std::isnan(result) || std::isinf(result))
        {
            std::cerr << "[Warning] norm: computed norm is invalid (NaN or Inf), returning 0.0f\n";
            return 0.0f;
        }
        
        return result;
    }

    /**
     * @name Mat::inverse_adjoint()
     * @brief Compute the inverse matrix using the adjoint method.
     * @note Formula: A^(-1) = (1 / det(A)) * adj(A)
     *       where adj(A) is the adjoint (transpose of cofactor matrix).
     * @note WARNING: This method is SLOW for large matrices!
     *       - Time complexity: O(n² × n!) - exponential growth with matrix size
     *       - For n×n matrix, requires computing n² determinants of (n-1)×(n-1) matrices
     *       - Each determinant calculation uses Laplace expansion (O(n!) complexity)
     *       - Example: 4×4 matrix needs 16 determinants of 3×3 matrices
     *       - Example: 5×5 matrix needs 25 determinants of 4×4 matrices (very slow!)
     * @note Performance comparison:
     *       - 2×2 matrix: Fast (direct formula)
     *       - 3×3 matrix: Acceptable
     *       - 4×4 matrix: Slow but usable
     *       - 5×5+ matrix: VERY SLOW - use other methods instead
     * @note For larger matrices (n >= 4), strongly recommend using:
     *       - inverse_gje() (Gauss-Jordan elimination, O(n³))
     *       - LU decomposition methods (O(n³), more stable)
     * @note This method is mainly useful for:
     *       - Small matrices (n <= 3) where simplicity is preferred
     *       - Educational purposes to understand the adjoint formula
     *       - Cases where you need the adjoint matrix anyway
     * 
     * @return Mat The inverse matrix, or empty Mat() on error
     */
    Mat Mat::inverse_adjoint()
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] inverse_adjoint: matrix data pointer is null\n";
            return Mat();
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] inverse_adjoint: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return Mat();
        }
        
        // Check if matrix is square
        if (this->row != this->col)
        {
            std::cerr << "[Error] inverse_adjoint: requires square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            return Mat();
        }
        
        // Compute determinant
        float det = this->determinant();
        
        // Check if determinant is valid
        if (std::isnan(det) || std::isinf(det))
        {
            std::cerr << "[Error] inverse_adjoint: determinant is invalid (NaN or Inf)\n";
            return Mat();
        }
        
        // Check if matrix is singular (determinant is zero or too small)
        if (fabsf(det) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
        {
            std::cerr << "[Error] inverse_adjoint: matrix is singular (det=" << det 
                      << "), cannot compute inverse\n";
            return Mat();
        }

        // Compute adjoint matrix
        Mat adj = this->adjoint();
        
        // Check if adjoint computation was successful
        if (adj.data == nullptr)
        {
            std::cerr << "[Error] inverse_adjoint: failed to compute adjoint matrix\n";
            return Mat();
        }
        
        // Check if adjoint matrix has correct dimensions
        if (adj.row != this->row || adj.col != this->col)
        {
            std::cerr << "[Error] inverse_adjoint: adjoint matrix has incorrect dimensions: " 
                      << adj.row << "x" << adj.col << " (expected " << this->row << "x" << this->col << ")\n";
            return Mat();
        }
        
        // Compute inverse: A^(-1) = (1 / det(A)) * adj(A)
        float inv_det = 1.0f / det;
        Mat result = adj * inv_det;
        
        // Check if matrix multiplication was successful
        if (result.data == nullptr)
        {
            std::cerr << "[Error] inverse_adjoint: failed to compute inverse matrix (multiplication failed)\n";
            return Mat();
        }
        
        return result;
    }

    /**
     * @name Mat::dotprod(const Mat &A, const Mat &B)
     * @brief Compute the dot product (Frobenius inner product) of two matrices.
     * @note Mathematical definition: A · B = Σ A_ij * B_ij (sum over all elements)
     *       This is the element-wise multiplication followed by summation.
     * @note Also known as:
     *       - Frobenius inner product
     *       - Hadamard product sum
     *       - Element-wise dot product
     * @note For vectors, this is equivalent to the standard dot product.
     * @note For empty matrices, returns 0.0f.
     *
     * @param A First matrix
     * @param B Second matrix
     * @return float Dot product value, or 0.0f on error
     */
    float Mat::dotprod(const Mat &A, const Mat &B)
    {
        // Check for null pointers
        if (A.data == nullptr)
        {
            std::cerr << "[Error] dotprod: matrix A data pointer is null\n";
            return 0.0f;
        }
        
        if (B.data == nullptr)
        {
            std::cerr << "[Error] dotprod: matrix B data pointer is null\n";
            return 0.0f;
        }
        
        // Validate matrix dimensions
        if (A.row <= 0 || A.col <= 0)
        {
            std::cerr << "[Error] dotprod: invalid matrix A dimensions: rows=" 
                      << A.row << ", cols=" << A.col << "\n";
            return 0.0f;
        }
        
        if (B.row <= 0 || B.col <= 0)
        {
            std::cerr << "[Error] dotprod: invalid matrix B dimensions: rows=" 
                      << B.row << ", cols=" << B.col << "\n";
            return 0.0f;
        }
        
        // Check if matrices have the same size
        if (A.row != B.row || A.col != B.col)
        {
            std::cerr << "[Error] dotprod: matrices must have the same size (A: " 
                      << A.row << "x" << A.col << ", B: " << B.row << "x" << B.col << ")\n";
            return 0.0f;
        }
        
        // Handle empty matrices
        if (A.row == 0 || A.col == 0)
        {
            return 0.0f;
        }

        // Compute dot product: sum of element-wise products
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
     * @brief Create an identity matrix (unit matrix) of specified size.
     * @note Identity matrix I has:
     *       - I_ij = 1 if i == j (diagonal elements)
     *       - I_ij = 0 if i != j (off-diagonal elements)
     * @note Properties:
     *       - I * A = A (left identity)
     *       - A * I = A (right identity)
     *       - I^(-1) = I (self-inverse)
     * @note For size = 0, returns empty matrix.
     *
     * @param size Size of the square identity matrix (must be >= 0)
     * @return Mat Identity matrix, or empty Mat() on error
     */
    Mat Mat::eye(int size)
    {
        // Validate parameter
        if (size < 0)
        {
            std::cerr << "[Error] eye: size must be non-negative (got " << size << ")\n";
            return Mat();
        }
        
        // Handle size = 0 (empty matrix)
        if (size == 0)
        {
            return Mat(0, 0);
        }
        
        Mat identity(size, size);
        
        // Check if matrix was created successfully
        if (identity.data == nullptr)
        {
            std::cerr << "[Error] eye: failed to create identity matrix of size " << size << "\n";
            return Mat();
        }
        
        // Initialize identity matrix: diagonal = 1, others = 0
        // Note: Mat constructor already initializes to 0, so we only need to set diagonal
        for (int i = 0; i < size; ++i)
        {
            identity(i, i) = 1.0f;  // Set diagonal elements to 1
            // Off-diagonal elements are already 0 from constructor
        }
        
        return identity;
    }

    /**
     * @name Mat::ones(int rows, int cols)
     * @brief Create a matrix filled with ones (all elements = 1.0f).
     * @note Creates a matrix where every element is 1.0f.
     * @note For rows = 0 or cols = 0, returns empty matrix.
     *
     * @param rows Number of rows (must be >= 0)
     * @param cols Number of columns (must be >= 0)
     * @return Mat Matrix filled with ones, or empty Mat() on error
     */
    Mat Mat::ones(int rows, int cols)
    {
        // Validate parameters
        if (rows < 0 || cols < 0)
        {
            std::cerr << "[Error] ones: dimensions must be non-negative (got rows=" 
                      << rows << ", cols=" << cols << ")\n";
            return Mat();
        }
        
        // Handle empty matrix
        if (rows == 0 || cols == 0)
        {
            return Mat(rows, cols);
        }
        
        Mat result(rows, cols);
        
        // Check if matrix was created successfully
        if (result.data == nullptr)
        {
            std::cerr << "[Error] ones: failed to create matrix of size " 
                      << rows << "x" << cols << "\n";
            return Mat();
        }
        
        // Fill all elements with 1.0f
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
     * @brief Create a square matrix filled with ones (all elements = 1.0f).
     * @note Convenience function that creates a size×size matrix filled with ones.
     *       Equivalent to ones(size, size).
     * @note For size = 0, returns empty matrix.
     *
     * @param size Size of the square matrix (rows = cols, must be >= 0)
     * @return Mat Square matrix [size x size] with all elements = 1, or empty Mat() on error
     */
    Mat Mat::ones(int size)
    {
        // All validation is handled by ones(size, size)
        return Mat::ones(size, size);
    }

    /**
     * @name Mat::augment(const Mat &A, const Mat &B)
     * @brief Augment two matrices horizontally [A | B] (concatenate columns).
     * @note Creates a new matrix by placing B to the right of A.
     *       Result: [A | B] with dimensions (rows, A.cols + B.cols)
     *       where rows must be the same for both matrices.
     * @note Common use case: Creating augmented matrix [A | b] for solving Ax = b
     *       using Gaussian elimination.
     * @note For empty matrices, returns empty matrix if dimensions are valid.
     *
     * @param A Left matrix
     * @param B Right matrix
     * @return Mat Augmented matrix [A B], or empty Mat() on error
     */
    Mat Mat::augment(const Mat &A, const Mat &B)
    {
        // Check for null pointers
        if (A.data == nullptr)
        {
            std::cerr << "[Error] augment: matrix A data pointer is null\n";
            return Mat();
        }
        
        if (B.data == nullptr)
        {
            std::cerr << "[Error] augment: matrix B data pointer is null\n";
            return Mat();
        }
        
        // Validate matrix dimensions
        if (A.row < 0 || A.col < 0)
        {
            std::cerr << "[Error] augment: invalid matrix A dimensions: rows=" 
                      << A.row << ", cols=" << A.col << "\n";
            return Mat();
        }
        
        if (B.row < 0 || B.col < 0)
        {
            std::cerr << "[Error] augment: invalid matrix B dimensions: rows=" 
                      << B.row << ", cols=" << B.col << "\n";
            return Mat();
        }
        
        // Check if row counts match
        if (A.row != B.row)
        {
            std::cerr << "[Error] augment: row counts must match (A: " 
                      << A.row << ", B: " << B.row << ")\n";
            return Mat();
        }
        
        // Check for integer overflow in column sum
        if (A.col > INT_MAX - B.col)
        {
            std::cerr << "[Error] augment: combined column count too large, integer overflow "
                      << "(A.col=" << A.col << ", B.col=" << B.col << ")\n";
            return Mat();
        }
        
        // Create new matrix with combined columns
        Mat AB(A.row, A.col + B.col);
        
        // Check if matrix was created successfully
        if (AB.data == nullptr)
        {
            std::cerr << "[Error] augment: failed to create augmented matrix of size " 
                      << A.row << "x" << (A.col + B.col) << "\n";
            return Mat();
        }
        
        // Handle empty matrices
        if (A.row == 0)
        {
            // Empty result matrix already created, return it
            return AB;
        }
        
        // Copy data from A and B
        for (int i = 0; i < A.row; ++i)
        {
            // Copy A (left part)
            for (int j = 0; j < A.col; ++j)
            {
                AB(i, j) = A(i, j);
            }
            // Copy B (right part)
            for (int j = 0; j < B.col; ++j)
            {
                AB(i, A.col + j) = B(i, j);
            }
        }

        return AB;
    }

    /**
     * @name Mat::vstack(const Mat &A, const Mat &B)
     * @brief Vertically stack two matrices [A; B] (concatenate rows).
     * @note Creates a new matrix by placing B below A.
     *       Result: [A; B] with dimensions (A.rows + B.rows, cols)
     *       where cols must be the same for both matrices.
     * @note Common use case: Combining data from multiple sources vertically,
     *       or building block matrices in linear algebra operations.
     * @note For empty matrices, returns empty matrix if dimensions are valid.
     *
     * @param A Top matrix
     * @param B Bottom matrix
     * @return Mat Vertically stacked matrix [A; B], or empty Mat() on error
     */
    Mat Mat::vstack(const Mat &A, const Mat &B)
    {
        // Check for null pointers
        if (A.data == nullptr)
        {
            std::cerr << "[Error] vstack: matrix A data pointer is null\n";
            return Mat();
        }
        
        if (B.data == nullptr)
        {
            std::cerr << "[Error] vstack: matrix B data pointer is null\n";
            return Mat();
        }
        
        // Validate matrix dimensions
        if (A.row < 0 || A.col < 0)
        {
            std::cerr << "[Error] vstack: invalid matrix A dimensions: rows=" 
                      << A.row << ", cols=" << A.col << "\n";
            return Mat();
        }
        
        if (B.row < 0 || B.col < 0)
        {
            std::cerr << "[Error] vstack: invalid matrix B dimensions: rows=" 
                      << B.row << ", cols=" << B.col << "\n";
            return Mat();
        }
        
        // Check if column counts match
        if (A.col != B.col)
        {
            std::cerr << "[Error] vstack: column counts must match (A: " 
                      << A.col << ", B: " << B.col << ")\n";
            return Mat();
        }
        
        // Check for integer overflow in row sum
        if (A.row > INT_MAX - B.row)
        {
            std::cerr << "[Error] vstack: combined row count too large, integer overflow "
                      << "(A.row=" << A.row << ", B.row=" << B.row << ")\n";
            return Mat();
        }
        
        // Create new matrix with combined rows
        Mat AB(A.row + B.row, A.col);
        
        // Check if matrix was created successfully
        if (AB.data == nullptr)
        {
            std::cerr << "[Error] vstack: failed to create stacked matrix of size " 
                      << (A.row + B.row) << "x" << A.col << "\n";
            return Mat();
        }
        
        // Handle empty matrices
        if (A.col == 0)
        {
            // Empty result matrix already created, return it
            return AB;
        }
        
        // Copy data from A and B
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
        // Validation: check for null pointer
        if (vectors.data == nullptr)
        {
            std::cerr << "[Error] gram_schmidt_orthogonalize: Input matrix is null.\n";
            return false;
        }

        int m = vectors.row;  // Dimension of vectors
        int n = vectors.col;  // Number of vectors

        // Validate dimensions
        if (m <= 0 || n <= 0)
        {
            std::cerr << "[Error] gram_schmidt_orthogonalize: Invalid dimensions (m=" 
                      << m << ", n=" << n << ")\n";
            return false;
        }
        
        // Validate tolerance
        if (tolerance < 0.0f)
        {
            std::cerr << "[Error] gram_schmidt_orthogonalize: tolerance must be non-negative (got " 
                      << tolerance << ")\n";
            return false;
        }

        // Initialize output matrices
        orthogonal_vectors = Mat(m, n);
        coefficients = Mat(n, n);  // Upper triangular matrix for coefficients
        
        // Check if output matrices were created successfully
        if (orthogonal_vectors.data == nullptr)
        {
            std::cerr << "[Error] gram_schmidt_orthogonalize: failed to create orthogonal_vectors matrix\n";
            return false;
        }
        
        if (coefficients.data == nullptr)
        {
            std::cerr << "[Error] gram_schmidt_orthogonalize: failed to create coefficients matrix\n";
            return false;
        }
        
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
     * 
     * @note This function returns a MATRIX (the minor matrix), NOT a scalar value.
     *       The minor matrix is (n-1)×(n-1) for an n×n input matrix.
     * 
     * @note Difference from cofactor():
     *       - minor() and cofactor() return the SAME matrix (submatrix)
     *       - minor() is the general term for the submatrix
     *       - cofactor() is semantically used when computing cofactor values (with sign)
     *       - Both functions can be used interchangeably for getting the submatrix
     * 
     * @note To compute minor value: minor_value = det(minor_matrix)
     * @note To compute cofactor value: cofactor_value = (-1)^(i+j) * det(minor_matrix)
     *
     * @param target_row Row index to remove (0-based)
     * @param target_col Column index to remove (0-based)
     * @return Mat The (n-1)×(n-1) minor matrix, or empty Mat() on error
     */
    Mat Mat::minor(int target_row, int target_col)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] minor: matrix data pointer is null\n";
            return Mat();
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] minor: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return Mat();
        }
        
        // Check if matrix is square
        if (this->row != this->col)
        {
            std::cerr << "[Error] Minor requires square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            return Mat();
        }

        int n = this->row;
        
        // Validate indices
        if (target_row < 0 || target_row >= n)
        {
            std::cerr << "[Error] minor: target_row=" << target_row 
                      << " is out of range [0, " << (n-1) << "]\n";
            return Mat();
        }
        
        if (target_col < 0 || target_col >= n)
        {
            std::cerr << "[Error] minor: target_col=" << target_col 
                      << " is out of range [0, " << (n-1) << "]\n";
            return Mat();
        }
        
        // For 1×1 matrix, removing one row and one column results in 0×0 matrix
        // This is a valid but empty result
        if (n == 1)
        {
            return Mat(0, 0);
        }
        
        Mat result(n - 1, n - 1);
        
        // Check if result matrix was created successfully
        if (result.data == nullptr)
        {
            std::cerr << "[Error] minor: failed to create result matrix\n";
            return Mat();
        }

        // Copy elements, skipping the specified row and column
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
     * 
     * @note IMPORTANT DISTINCTION:
     *       - This function returns a MATRIX (the cofactor matrix), NOT a scalar value.
     *       - The cofactor matrix is mathematically identical to the minor matrix.
     *       - To get the COFACTOR VALUE (scalar), you must:
     *         cofactor_value = (-1)^(i+j) * det(cofactor_matrix)
     * 
     * @note Mathematical definitions:
     *       - Minor matrix M_ij = submatrix after removing row i and column j
     *       - Cofactor matrix C_ij = M_ij (same as minor matrix)
     *       - Cofactor VALUE = (-1)^(i+j) * det(M_ij)
     * 
     * @note Difference from minor():
     *       - minor() and cofactor() return the SAME matrix (submatrix)
     *       - The difference is semantic: cofactor() is used when computing
     *         cofactor values (with sign), while minor() is more general.
     *       - Both functions can be used interchangeably for getting the submatrix.
     * 
     * @note Usage example:
     *       Mat cofactor_mat = A.cofactor(i, j);  // Returns matrix
     *       float cofactor_val = ((i+j) % 2 == 0 ? 1.0f : -1.0f) * cofactor_mat.determinant();  // Value
     *
     * @param target_row Row index to remove (0-based)
     * @param target_col Column index to remove (0-based)
     * @return Mat The (n-1)×(n-1) cofactor matrix (same as minor matrix), or empty Mat() on error
     */
    Mat Mat::cofactor(int target_row, int target_col)
    {
        // Cofactor matrix is the same as minor matrix
        // The sign is applied when computing cofactor values, not to matrix elements
        // All validation is handled by minor()
        return this->minor(target_row, target_col);
    }

    /**
     * @name Mat::gaussian_eliminate
     * @brief Perform Gaussian Elimination to convert matrix to Row Echelon Form (REF).
     * @note Gaussian elimination transforms a matrix to upper triangular form (REF) using
     *       elementary row operations: row swapping, row scaling, and row addition.
     * @note Algorithm:
     *       1. For each row r, find a pivot (non-zero element) in column lead
     *       2. Swap rows if necessary to bring pivot to current row
     *       3. Eliminate elements below pivot by subtracting multiples of pivot row
     *       4. Move to next column and repeat
     * @note Uses partial pivoting (finds first non-zero element) for numerical stability.
     * @note Near-zero values are set to zero for numerical precision.
     *
     * @return Mat The upper triangular matrix (REF form), or empty Mat() on error
     */
    Mat Mat::gaussian_eliminate() const
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] gaussian_eliminate: matrix data pointer is null\n";
            return Mat();
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] gaussian_eliminate: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return Mat();
        }
        
        // Handle empty matrix
        if (this->row == 0 || this->col == 0)
        {
            return Mat(*this);  // Return copy of empty matrix
        }
        
        Mat result(*this); // Create a copy of the original matrix
        
        // Check if copy was successful
        if (result.data == nullptr)
        {
            std::cerr << "[Error] gaussian_eliminate: failed to create working copy\n";
            return Mat();
        }
        
        int rows = result.row;
        int cols = result.col;

        int lead = 0; // Leading column tracker

        for (int r = 0; r < rows; ++r)
        {
            if (lead >= cols)
                break;

            int i = r;

            // Find pivot row (partial pivoting)
            // Look for first non-zero (or near-zero) element in column lead
            while (fabsf(result(i, lead)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
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
            {
                result.swap_rows(i, r);
            }

            // Check if pivot is still valid after swap
            if (fabsf(result(r, lead)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                // Pivot is still too small, move to next column
                lead++;
                continue;
            }

            // Eliminate rows below
            for (int j = r + 1; j < rows; ++j)
            {
                if (fabsf(result(j, lead)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                    continue;

                float factor = result(j, lead) / result(r, lead);
                for (int k = lead; k < cols; ++k)
                {
                    result(j, k) -= factor * result(r, k);

                    // Numerical precision handling (set near-zero values to zero)
                    if (fabsf(result(j, k)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
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
     * @note This function assumes the input matrix is already in Row Echelon Form (REF).
     *       It performs back-substitution to convert REF to RREF.
     * @note RREF properties:
     *       - Leading entry (pivot) in each row is 1
     *       - Pivot is the only non-zero entry in its column
     *       - Rows with all zeros are at the bottom
     * @note Algorithm:
     *       1. Start from bottom row and work upwards
     *       2. For each row with a pivot:
     *          a. Normalize pivot to 1 (divide row by pivot value)
     *          b. Eliminate entries above pivot (make them zero)
     *       3. Continue until all rows are processed
     *
     * @return Mat The matrix in RREF form, or empty Mat() on error
     */
    Mat Mat::row_reduce_from_gaussian()
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] row_reduce_from_gaussian: matrix data pointer is null\n";
            return Mat();
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] row_reduce_from_gaussian: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return Mat();
        }
        
        // Handle empty matrix
        if (this->row == 0 || this->col == 0)
        {
            return Mat(*this);  // Return copy of empty matrix
        }
        
        Mat R(*this); // Make a copy to preserve original matrix
        
        // Check if copy was successful
        if (R.data == nullptr)
        {
            std::cerr << "[Error] row_reduce_from_gaussian: failed to create working copy\n";
            return Mat();
        }
        
        int rows = R.row;
        int cols = R.col;

        int pivot_row = rows - 1;

        while (pivot_row >= 0)
        {
            // Locate pivot in current row (first non-zero element)
            int current_pivot_col = -1;
            for (int k = 0; k < cols; ++k)
            {
                if (fabsf(R(pivot_row, k)) >= TINY_MATH_MIN_POSITIVE_INPUT_F32)
                {
                    current_pivot_col = k;
                    break;
                }
            }

            if (current_pivot_col != -1)
            {
                // Normalize pivot row (make pivot = 1)
                float pivot_val = R(pivot_row, current_pivot_col);
                
                // Check if pivot value is valid (not zero or too small)
                if (fabsf(pivot_val) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                {
                    // Pivot is too small, skip this row
                    pivot_row--;
                    continue;
                }
                
                for (int s = current_pivot_col; s < cols; ++s)
                {
                    R(pivot_row, s) /= pivot_val;
                    // Numerical precision handling
                    if (fabsf(R(pivot_row, s)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                    {
                        R(pivot_row, s) = 0.0f;
                    }
                }

                // Eliminate above pivot (make entries above pivot zero)
                for (int t = pivot_row - 1; t >= 0; --t)
                {
                    float factor = R(t, current_pivot_col);
                    // Skip if factor is already zero (optimization)
                    if (fabsf(factor) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                        continue;
                    
                    for (int s = current_pivot_col; s < cols; ++s)
                    {
                        R(t, s) -= factor * R(pivot_row, s);
                        // Numerical precision handling
                        if (fabsf(R(t, s)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
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
     * @note Algorithm:
     *       1. Create augmented matrix [A | I] where I is identity matrix
     *       2. Apply Gauss-Jordan elimination to get [I | A^(-1)]
     *       3. Extract the right half as the inverse matrix
     * @note Time complexity: O(n³) - efficient for large matrices.
     *       More efficient than adjoint method for n >= 4.
     * @note If matrix is singular (not invertible), returns empty matrix.
     *
     * @return Mat The inverse matrix if invertible, or empty Mat() on error
     */
    Mat Mat::inverse_gje()
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] inverse_gje: matrix data pointer is null\n";
            return Mat();
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] inverse_gje: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return Mat();
        }
        
        // Check if matrix is square
        if (this->row != this->col)
        {
            std::cerr << "[Error] inverse_gje: requires square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            return Mat();
        }

        int n = this->row;

        // Step 1: Create augmented matrix [A | I]
        Mat I = Mat::eye(n);            // Identity matrix
        
        // Check if identity matrix was created successfully
        if (I.data == nullptr)
        {
            std::cerr << "[Error] inverse_gje: failed to create identity matrix\n";
            return Mat();
        }
        
        Mat augmented = Mat::augment(*this, I); // Augment matrix A with I
        
        // Check if augmented matrix was created successfully
        if (augmented.data == nullptr)
        {
            std::cerr << "[Error] inverse_gje: failed to create augmented matrix\n";
            return Mat();
        }
        
        // Check augmented matrix dimensions
        if (augmented.col != 2 * n)
        {
            std::cerr << "[Error] inverse_gje: augmented matrix has incorrect dimensions: " 
                      << augmented.row << "x" << augmented.col << " (expected " << n << "x" << (2*n) << ")\n";
            return Mat();
        }

        // Step 2: Apply Gauss-Jordan elimination to get [I | A_inv]
        Mat rref = augmented.gaussian_eliminate();
        
        // Check if gaussian_eliminate was successful
        if (rref.data == nullptr)
        {
            std::cerr << "[Error] inverse_gje: gaussian_eliminate failed\n";
            return Mat();
        }
        
        rref = rref.row_reduce_from_gaussian();
        
        // Check if row_reduce_from_gaussian was successful
        if (rref.data == nullptr)
        {
            std::cerr << "[Error] inverse_gje: row_reduce_from_gaussian failed\n";
            return Mat();
        }

        // Check if the left half is the identity matrix
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                float expected = (i == j) ? 1.0f : 0.0f;
                float actual = rref(i, j);
                if (fabsf(actual - expected) > TINY_MATH_MIN_POSITIVE_INPUT_F32)
                {
                    std::cerr << "[Error] inverse_gje: matrix is singular (not invertible), "
                              << "left half is not identity matrix at (" << i << ", " << j 
                              << "): expected=" << expected << ", actual=" << actual << "\n";
                    return Mat();
                }
            }
        }

        // Step 3: Extract the right half as the inverse matrix
        Mat result(n, n);
        
        // Check if result matrix was created successfully
        if (result.data == nullptr)
        {
            std::cerr << "[Error] inverse_gje: failed to create result matrix\n";
            return Mat();
        }
        
        // Extract the right half (columns n to 2n-1)
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                int col_idx = j + n;  // Right half starts at column n
                // Boundary check (should not be needed, but safe)
                if (col_idx >= rref.col)
                {
                    std::cerr << "[Error] inverse_gje: column index out of bounds: " 
                              << col_idx << " >= " << rref.col << "\n";
                    return Mat();
                }
                result(i, j) = rref(i, col_idx); // Extract the right part
            }
        }

        return result;
    }

    /**
     * @name Mat::solve
     * @brief Solve the linear system Ax = b using Gaussian elimination with back-substitution.
     * @note Solves the system of linear equations: A × x = b
     *       where A is an n×n coefficient matrix and b is an n×1 vector.
     * @note Algorithm:
     *       1. Create augmented matrix [A | b]
     *       2. Apply Gaussian elimination to convert to upper triangular form
     *       3. Use back-substitution to solve for x
     * @note Time complexity: O(n³) - efficient for solving linear systems.
     * @note If matrix A is singular (not invertible), returns empty matrix.
     *
     * @param A Coefficient matrix (N×N, must be square)
     * @param b Result vector (N×1)
     * @return Mat Solution vector (N×1) containing x such that Ax = b, or empty Mat() on error
     */
    Mat Mat::solve(const Mat &A, const Mat &b) const
    {
        // Check for null pointers
        if (A.data == nullptr)
        {
            std::cerr << "[Error] solve: matrix A data pointer is null\n";
            return Mat();
        }
        
        if (b.data == nullptr)
        {
            std::cerr << "[Error] solve: vector b data pointer is null\n";
            return Mat();
        }
        
        // Validate matrix dimensions
        if (A.row <= 0 || A.col <= 0)
        {
            std::cerr << "[Error] solve: invalid matrix A dimensions: rows=" 
                      << A.row << ", cols=" << A.col << "\n";
            return Mat();
        }
        
        if (b.row <= 0 || b.col <= 0)
        {
            std::cerr << "[Error] solve: invalid vector b dimensions: rows=" 
                      << b.row << ", cols=" << b.col << "\n";
            return Mat();
        }
        
        // Check if the matrix A is square
        if (A.row != A.col)
        {
            std::cerr << "[Error] solve: matrix A must be square (got " 
                      << A.row << "x" << A.col << ")\n";
            return Mat();
        }

        // Check if A and b dimensions are compatible for solving
        if (A.row != b.row || b.col != 1)
        {
            std::cerr << "[Error] solve: dimensions do not match (A: " 
                      << A.row << "x" << A.col << ", b: " << b.row << "x" << b.col 
                      << ", expected b: " << A.row << "x1)\n";
            return Mat();
        }
        
        int n = A.row;
        
        // Check for integer overflow in augmented matrix column count
        if (A.col > INT_MAX - 1)
        {
            std::cerr << "[Error] solve: matrix size too large, integer overflow\n";
            return Mat();
        }

        // Create augmented matrix [A | b]
        Mat augmentedMatrix(n, A.col + 1);
        
        // Check if augmented matrix was created successfully
        if (augmentedMatrix.data == nullptr)
        {
            std::cerr << "[Error] solve: failed to create augmented matrix\n";
            return Mat();
        }
        
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < A.col; ++j)
            {
                augmentedMatrix(i, j) = A(i, j); // Copy matrix A into augmented matrix
            }
            augmentedMatrix(i, A.col) = b(i, 0); // Copy vector b into augmented matrix
        }

        // Perform Gaussian elimination
        for (int i = 0; i < n; ++i)
        {
            // Find pivot and make sure it's non-zero (or not too small)
            // Note: This is a simplified version without partial pivoting
            // For better numerical stability, consider using partial pivoting
            if (fabsf(augmentedMatrix(i, i)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] solve: pivot at (" << i << ", " << i 
                          << ") is zero or too small (" << augmentedMatrix(i, i) 
                          << "), matrix is singular or near-singular\n";
                return Mat();
            }

            // Normalize the pivot row
            float pivot = augmentedMatrix(i, i);
            for (int j = i; j < augmentedMatrix.col; ++j)
            {
                augmentedMatrix(i, j) /= pivot; // Normalize the pivot row
            }

            // Eliminate the entries below the pivot
            for (int j = i + 1; j < n; ++j)
            {
                float factor = augmentedMatrix(j, i);
                // Skip if factor is already zero (optimization)
                if (fabsf(factor) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                    continue;
                    
                for (int k = i; k < augmentedMatrix.col; ++k)
                {
                    augmentedMatrix(j, k) -= factor * augmentedMatrix(i, k);
                    
                    // Numerical precision handling
                    if (fabsf(augmentedMatrix(j, k)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                    {
                        augmentedMatrix(j, k) = 0.0f;
                    }
                }
            }
        }

        // Back-substitution to find the solution
        Mat solution(n, 1);
        
        // Check if solution matrix was created successfully
        if (solution.data == nullptr)
        {
            std::cerr << "[Error] solve: failed to create solution vector\n";
            return Mat();
        }
        
        for (int i = n - 1; i >= 0; --i)
        {
            float sum = augmentedMatrix(i, A.col); // Right-hand side value
            for (int j = i + 1; j < n; ++j)
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
     * @note Banded matrices have non-zero elements only in a narrow band around the diagonal.
     *       This function optimizes Gaussian elimination by only processing elements within the band.
     * @note Algorithm:
     *       1. Forward elimination: only eliminate elements within the band
     *       2. Back-substitution: solve for x
     * @note Time complexity: O(n × k²) where n is matrix size and k is bandwidth.
     *       More efficient than general solve() for banded matrices (k << n).
     * @note Bandwidth k: total width of non-zero band (including diagonal).
     *       For tridiagonal matrix, k = 3.
     *
     * @param A Coefficient matrix (N×N) - banded matrix (passed by value, will be modified)
     * @param b Result vector (N×1) (passed by value, will be modified)
     * @param k Bandwidth of the matrix (must be >= 1 and odd, typically 3, 5, 7, ...)
     * @return Mat Solution vector (N×1) containing x such that Ax = b, or empty Mat() on error
     */
    Mat Mat::band_solve(Mat A, Mat b, int k)
    {
        // Check for null pointers
        if (A.data == nullptr)
        {
            std::cerr << "[Error] band_solve: matrix A data pointer is null\n";
            return Mat();
        }
        
        if (b.data == nullptr)
        {
            std::cerr << "[Error] band_solve: vector b data pointer is null\n";
            return Mat();
        }
        
        // Validate matrix dimensions
        if (A.row <= 0 || A.col <= 0)
        {
            std::cerr << "[Error] band_solve: invalid matrix A dimensions: rows=" 
                      << A.row << ", cols=" << A.col << "\n";
            return Mat();
        }
        
        if (b.row <= 0 || b.col <= 0)
        {
            std::cerr << "[Error] band_solve: invalid vector b dimensions: rows=" 
                      << b.row << ", cols=" << b.col << "\n";
            return Mat();
        }
        
        // Dimension compatibility check
        if (A.row != A.col) // Check if A is a square matrix
        {
            std::cerr << "[Error] band_solve: matrix A must be square (got " 
                      << A.row << "x" << A.col << ")\n";
            return Mat();
        }

        if (A.row != b.row || b.col != 1) // Check if dimensions of A and b are compatible
        {
            std::cerr << "[Error] band_solve: dimensions do not match (A: " 
                      << A.row << "x" << A.col << ", b: " << b.row << "x" << b.col 
                      << ", expected b: " << A.row << "x1)\n";
            return Mat();
        }
        
        // Validate bandwidth parameter
        if (k < 1)
        {
            std::cerr << "[Error] band_solve: bandwidth k must be >= 1 (got " << k << ")\n";
            return Mat();
        }
        
        if (k > A.row)
        {
            std::cerr << "[Warning] band_solve: bandwidth k=" << k 
                      << " is larger than matrix size " << A.row 
                      << ", using general solve may be more efficient\n";
        }

        int n = A.row;
        int bandsBelow = (k - 1) / 2; // Number of bands below the main diagonal

        // Perform forward elimination to reduce the matrix
        for (int i = 0; i < n; ++i)
        {
            // Check if pivot is valid (not zero or too small)
            if (fabsf(A(i, i)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] band_solve: zero or near-zero pivot detected at (" 
                          << i << ", " << i << ") = " << A(i, i) 
                          << ", matrix is singular or near-singular\n";
                return Mat();
            }

            float a_ii = 1.0f / A(i, i); // Inverse of the pivot element

            // Eliminate elements below the pivot in the current column
            // Only process elements within the band (j <= i + bandsBelow)
            for (int j = i + 1; j < n && j <= i + bandsBelow; ++j)
            {
                if (fabsf(A(j, i)) >= TINY_MATH_MIN_POSITIVE_INPUT_F32)
                {
                    float factor = A(j, i) * a_ii;
                    for (int col_idx = i; col_idx < A.col; ++col_idx)
                    {
                        A(j, col_idx) -= A(i, col_idx) * factor; // Eliminate the element
                        
                        // Numerical precision handling
                        if (fabsf(A(j, col_idx)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                        {
                            A(j, col_idx) = 0.0f;
                        }
                    }
                    b(j, 0) -= b(i, 0) * factor; // Update the result vector
                    
                    // Numerical precision handling
                    if (fabsf(b(j, 0)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                    {
                        b(j, 0) = 0.0f;
                    }
                    
                    A(j, i) = 0.0f; // Set the element to zero as it has been eliminated
                }
            }
        }

        // Back substitution to solve for x
        Mat x(n, 1);
        
        // Check if solution matrix was created successfully
        if (x.data == nullptr)
        {
            std::cerr << "[Error] band_solve: failed to create solution vector\n";
            return Mat();
        }
        
        // Solve the last variable
        int last_idx = n - 1;
        if (fabsf(A(last_idx, last_idx)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
        {
            std::cerr << "[Error] band_solve: zero pivot at (" << last_idx << ", " 
                      << last_idx << ") during back-substitution\n";
            return Mat();
        }
        x(last_idx, 0) = b(last_idx, 0) / A(last_idx, last_idx);

        // Solve remaining variables
        for (int i = n - 2; i >= 0; --i)
        {
            float sum = 0.0f;
            // Only sum elements within the band
            int max_j = std::min(i + bandsBelow + 1, n);
            for (int j = i + 1; j < max_j; ++j)
            {
                sum += A(i, j) * x(j, 0); // Sum of the known terms
            }
            
            if (fabsf(A(i, i)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] band_solve: zero pivot at (" << i << ", " << i 
                          << ") during back-substitution\n";
                return Mat();
            }
            
            x(i, 0) = (b(i, 0) - sum) / A(i, i); // Solve for the current variable
        }

        return x; // Return the solution vector
    }

    /**
     * @name Mat::roots(Mat A, Mat y)
     * @brief Solve the linear system A * x = y using Gaussian elimination.
     * @note This is an alternative implementation of solve() function.
     *       It uses a slightly different Gaussian elimination approach:
     *       - Normalizes pivot row first (makes pivot = 1)
     *       - Then eliminates below pivot
     * @note Algorithm:
     *       1. Create augmented matrix [A | y]
     *       2. For each row: normalize pivot to 1, then eliminate below
     *       3. Back-substitution to solve for x
     * @note Time complexity: O(n³) - same as solve().
     * @note If matrix A is singular (not invertible), returns empty matrix.
     *
     * @param A Coefficient matrix (N×N, must be square, passed by value, will be modified)
     * @param y Result vector (N×1, passed by value, will be modified)
     * @return Mat Solution vector (N×1) containing x such that Ax = y, or empty Mat() on error
     */
    Mat Mat::roots(Mat A, Mat y)
    {
        // Check for null pointers
        if (A.data == nullptr)
        {
            std::cerr << "[Error] roots: matrix A data pointer is null\n";
            return Mat();
        }
        
        if (y.data == nullptr)
        {
            std::cerr << "[Error] roots: vector y data pointer is null\n";
            return Mat();
        }
        
        // Validate matrix dimensions
        if (A.row <= 0 || A.col <= 0)
        {
            std::cerr << "[Error] roots: invalid matrix A dimensions: rows=" 
                      << A.row << ", cols=" << A.col << "\n";
            return Mat();
        }
        
        if (y.row <= 0 || y.col <= 0)
        {
            std::cerr << "[Error] roots: invalid vector y dimensions: rows=" 
                      << y.row << ", cols=" << y.col << "\n";
            return Mat();
        }
        
        // Check if A is square
        if (A.row != A.col)
        {
            std::cerr << "[Error] roots: matrix A must be square (got " 
                      << A.row << "x" << A.col << ")\n";
            return Mat();
        }
        
        // Check if A and y dimensions are compatible
        if (A.row != y.row || y.col != 1)
        {
            std::cerr << "[Error] roots: dimensions do not match (A: " 
                      << A.row << "x" << A.col << ", y: " << y.row << "x" << y.col 
                      << ", expected y: " << A.row << "x1)\n";
            return Mat();
        }
        
        int n = A.row; // Number of rows and columns in A (A is square)

        // Create augmented matrix [A | y]
        Mat augmentedMatrix = Mat::augment(A, y);
        
        // Check if augmented matrix was created successfully
        if (augmentedMatrix.data == nullptr)
        {
            std::cerr << "[Error] roots: failed to create augmented matrix\n";
            return Mat();
        }
        
        // Verify augmented matrix dimensions
        if (augmentedMatrix.col != n + 1)
        {
            std::cerr << "[Error] roots: augmented matrix has incorrect dimensions: " 
                      << augmentedMatrix.row << "x" << augmentedMatrix.col 
                      << " (expected " << n << "x" << (n+1) << ")\n";
            return Mat();
        }

        // Perform Gaussian elimination
        for (int j = 0; j < n; j++)
        {
            // Normalize the pivot row (make pivot element equal to 1)
            float pivot = augmentedMatrix(j, j);
            
            // Check if pivot is valid (not zero or too small)
            if (fabsf(pivot) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] roots: pivot is zero or too small at (" << j << ", " << j 
                          << ") = " << pivot << ", system may have no solution\n";
                return Mat();
            }

            // Normalize the pivot row
            for (int k = 0; k < augmentedMatrix.col; k++)
            {
                augmentedMatrix(j, k) /= pivot;
                
                // Numerical precision handling
                if (fabsf(augmentedMatrix(j, k)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                {
                    augmentedMatrix(j, k) = 0.0f;
                }
            }

            // Eliminate the column below the pivot (set other elements in the column to zero)
            for (int i = j + 1; i < n; i++)
            {
                float factor = augmentedMatrix(i, j);
                
                // Skip if factor is already zero (optimization)
                if (fabsf(factor) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                    continue;
                
                for (int k = 0; k < augmentedMatrix.col; k++)
                {
                    augmentedMatrix(i, k) -= factor * augmentedMatrix(j, k);
                    
                    // Numerical precision handling
                    if (fabsf(augmentedMatrix(i, k)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                    {
                        augmentedMatrix(i, k) = 0.0f;
                    }
                }
            }
        }

        // Perform back-substitution
        Mat result(n, 1);
        
        // Check if result matrix was created successfully
        if (result.data == nullptr)
        {
            std::cerr << "[Error] roots: failed to create result vector\n";
            return Mat();
        }
        
        for (int i = n - 1; i >= 0; i--)
        {
            // Right-hand side of the augmented matrix (last column)
            int rhs_col = n;  // Last column index
            if (rhs_col >= augmentedMatrix.col)
            {
                std::cerr << "[Error] roots: column index out of bounds: " 
                          << rhs_col << " >= " << augmentedMatrix.col << "\n";
                return Mat();
            }
            
            float sum = augmentedMatrix(i, rhs_col);
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
     * @brief Check if matrix is positive definite (for Cholesky decomposition).
     * @note A matrix A is positive definite if:
     *       1. A is symmetric: A^T = A
     *       2. All eigenvalues are positive: λ_i > 0
     *       3. For all non-zero vectors x: x^T A x > 0
     * @note Uses Sylvester's criterion: all leading principal minors must be positive.
     *       Checks leading principal minors according to max_minors_to_check parameter.
     * @note Positive definite matrices have:
     *       - All diagonal elements > 0
     *       - All leading principal minors > 0
     *       - Can be decomposed as A = L L^T (Cholesky decomposition)
     * 
     * @param tolerance Tolerance for numerical checks (must be >= 0)
     * @param max_minors_to_check Maximum number of leading principal minors to check.
     *                            - If -1: check all minors (complete Sylvester's criterion)
     *                            - If > 0: check first max_minors_to_check minors
     *                            - Default: -1 (check all)
     * @return true if matrix is positive definite, false otherwise
     */
    bool Mat::is_positive_definite(float tolerance, int max_minors_to_check) const
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] is_positive_definite: matrix data pointer is null\n";
            return false;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] is_positive_definite: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return false;
        }
        
        // Validate tolerance
        if (tolerance < 0.0f)
        {
            std::cerr << "[Error] is_positive_definite: tolerance must be non-negative (got " 
                      << tolerance << ")\n";
            return false;
        }
        
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
        
        // Handle empty matrix
        if (n == 0)
        {
            return true;  // Empty matrix is trivially positive definite
        }

        // Determine how many minors to check
        int num_minors_to_check;
        if (max_minors_to_check < 0)
        {
            // Check all minors (complete Sylvester's criterion)
            num_minors_to_check = n;
        }
        else if (max_minors_to_check == 0)
        {
            std::cerr << "[Error] is_positive_definite: max_minors_to_check must be > 0 or -1 (got 0)\n";
            return false;
        }
        else
        {
            // Check first max_minors_to_check minors (or all if n is smaller)
            num_minors_to_check = (max_minors_to_check > n) ? n : max_minors_to_check;
        }

        // Check Sylvester's criterion: all leading principal minors must be positive
        for (int k = 1; k <= num_minors_to_check; ++k)
        {
            Mat submatrix(k, k);
            
            // Check if submatrix was created successfully
            if (submatrix.data == nullptr)
            {
                std::cerr << "[Error] is_positive_definite: failed to create submatrix of size " 
                          << k << "x" << k << "\n";
                return false;
            }
            
            // Copy leading principal minor
            for (int i = 0; i < k; ++i)
            {
                for (int j = 0; j < k; ++j)
                {
                    submatrix(i, j) = (*this)(i, j);
                }
            }
            
            float det = submatrix.determinant();
            
            // Check if determinant is valid
            if (std::isnan(det) || std::isinf(det))
            {
                std::cerr << "[Error] is_positive_definite: determinant is invalid (NaN or Inf) "
                          << "for leading principal minor of size " << k << "x" << k << "\n";
                return false;
            }
            
            // Sylvester's criterion: determinant must be > tolerance
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
     * @brief Compute LU decomposition: A = L * U (with optional pivoting).
     * @note LU decomposition factors a matrix A into:
     *       - L: Lower triangular matrix (with unit diagonal)
     *       - U: Upper triangular matrix
     *       - P: Permutation matrix (if pivoting used)
     * @note With pivoting: P * A = L * U
     *       Without pivoting: A = L * U
     * @note Algorithm: Modified Gaussian elimination that stores multipliers in L.
     *       Uses partial pivoting for numerical stability.
     * @note Time complexity: O(n³) - efficient for large matrices.
     *       More efficient than Gaussian elimination when solving multiple systems
     *       with the same coefficient matrix.
     * 
     * @param use_pivoting Whether to use partial pivoting (default: true).
     *                     Pivoting improves numerical stability but requires P matrix.
     * @return LUDecomposition containing L, U, P matrices and status
     */
    Mat::LUDecomposition Mat::lu_decompose(bool use_pivoting) const
    {
        LUDecomposition result;

        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] lu_decompose: matrix data pointer is null\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] lu_decompose: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        // Validation: must be square matrix
        if (this->row != this->col)
        {
            std::cerr << "[Error] lu_decompose: requires square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        int n = this->row;
        
        // Handle empty matrix
        if (n == 0)
        {
            // Empty matrix: L, U, P are all empty
            result.L = Mat(0, 0);
            result.U = Mat(0, 0);
            if (use_pivoting)
            {
                result.P = Mat(0, 0);
            }
            result.pivoted = use_pivoting;
            result.status = TINY_OK;
            return result;
        }
        
        Mat A = Mat(*this);  // Working copy
        
        // Check if working copy was created successfully
        if (A.data == nullptr)
        {
            std::cerr << "[Error] lu_decompose: failed to create working copy\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        result.L = Mat::eye(n);  // Initialize L as identity
        
        // Check if L matrix was created successfully
        if (result.L.data == nullptr)
        {
            std::cerr << "[Error] lu_decompose: failed to create L matrix\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        result.U = Mat(n, n);   // Initialize U
        
        // Check if U matrix was created successfully
        if (result.U.data == nullptr)
        {
            std::cerr << "[Error] lu_decompose: failed to create U matrix\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        result.pivoted = use_pivoting;

        if (use_pivoting)
        {
            result.P = Mat::eye(n);  // Initialize P as identity
            
            // Check if P matrix was created successfully
            if (result.P.data == nullptr)
            {
                std::cerr << "[Error] lu_decompose: failed to create P matrix\n";
                result.status = TINY_ERR_MATH_NULL_POINTER;
                return result;
            }
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
     * @brief Compute Cholesky decomposition: A = L * L^T (for symmetric positive definite matrices).
     * @note Cholesky decomposition factors a symmetric positive definite matrix A into:
     *       A = L * L^T
     *       where L is a lower triangular matrix with positive diagonal elements.
     * @note Algorithm: For each row i and column j (j <= i):
     *       - Diagonal (j == i): L[i][i] = sqrt(A[i][i] - sum(L[i][k]^2))
     *       - Off-diagonal (j < i): L[i][j] = (A[i][j] - sum(L[i][k]*L[j][k])) / L[j][j]
     * @note Requirements:
     *       - Matrix must be square
     *       - Matrix must be symmetric: A^T = A
     *       - Matrix must be positive definite: all eigenvalues > 0
     * @note Advantages over LU decomposition:
     *       - Faster: O(n³/3) vs O(n³) for LU
     *       - More stable: no pivoting needed
     *       - Uses less memory: only stores L (not L and U)
     * @note Applications: Structural dynamics, optimization, Kalman filtering, Monte Carlo simulation
     * @note Time complexity: O(n³/3) - about 2x faster than LU decomposition
     * 
     * @return CholeskyDecomposition containing L matrix and status
     */
    Mat::CholeskyDecomposition Mat::cholesky_decompose() const
    {
        CholeskyDecomposition result;

        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] cholesky_decompose: matrix data pointer is null\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] cholesky_decompose: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        // Validation: must be square matrix
        if (this->row != this->col)
        {
            std::cerr << "[Error] cholesky_decompose: requires square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        int n = this->row;
        
        // Handle empty matrix
        if (n == 0)
        {
            // Empty matrix: L is also empty
            result.L = Mat(0, 0);
            result.status = TINY_OK;
            return result;
        }

        // Check if symmetric (use standard tolerance)
        if (!this->is_symmetric(TINY_MATH_MIN_POSITIVE_INPUT_F32))
        {
            std::cerr << "[Error] cholesky_decompose: requires symmetric matrix\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        result.L = Mat(n, n);
        
        // Check if L matrix was created successfully
        if (result.L.data == nullptr)
        {
            std::cerr << "[Error] cholesky_decompose: failed to create L matrix\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }

        // Cholesky decomposition: A = L * L^T
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                float sum = 0.0f;
                
                if (j == i)
                {
                    // Diagonal elements: L[i][i] = sqrt(A[i][i] - sum(L[i][k]^2))
                    for (int k = 0; k < j; ++k)
                    {
                        sum += result.L(j, k) * result.L(j, k);
                    }
                    float diag_val = (*this)(j, j) - sum;
                    
                    // Check if matrix is positive definite
                    // For positive definite matrices, diag_val must be > tolerance
                    if (diag_val <= TINY_MATH_MIN_POSITIVE_INPUT_F32)
                    {
                        std::cerr << "[Error] cholesky_decompose: matrix is not positive definite "
                                  << "(diagonal value " << diag_val << " at position [" 
                                  << j << "][" << j << "] is not positive)\n";
                        result.status = TINY_ERR_MATH_INVALID_PARAM;
                        return result;
                    }
                    
                    float sqrt_result = sqrtf(diag_val);
                    
                    // Check if sqrt result is valid
                    if (std::isnan(sqrt_result) || std::isinf(sqrt_result))
                    {
                        std::cerr << "[Error] cholesky_decompose: sqrt result is invalid (NaN or Inf) "
                                  << "at position [" << j << "][" << j << "]\n";
                        result.status = TINY_ERR_MATH_INVALID_PARAM;
                        return result;
                    }
                    
                    result.L(j, j) = sqrt_result;
                }
                else
                {
                    // Off-diagonal elements: L[i][j] = (A[i][j] - sum(L[i][k]*L[j][k])) / L[j][j]
                    for (int k = 0; k < j; ++k)
                    {
                        sum += result.L(i, k) * result.L(j, k);
                    }
                    
                    // Check if divisor is valid (should be > 0 from previous diagonal calculation)
                    float divisor = result.L(j, j);
                    if (fabsf(divisor) < TINY_MATH_MIN_POSITIVE_INPUT_F32 || 
                        std::isnan(divisor) || std::isinf(divisor))
                    {
                        std::cerr << "[Error] cholesky_decompose: invalid divisor at position [" 
                                  << j << "][" << j << "] (value: " << divisor << ")\n";
                        result.status = TINY_ERR_MATH_INVALID_PARAM;
                        return result;
                    }
                    
                    result.L(i, j) = ((*this)(i, j) - sum) / divisor;
                    
                    // Check if result is valid
                    if (std::isnan(result.L(i, j)) || std::isinf(result.L(i, j)))
                    {
                        std::cerr << "[Error] cholesky_decompose: computed value is invalid (NaN or Inf) "
                                  << "at position [" << i << "][" << j << "]\n";
                        result.status = TINY_ERR_MATH_INVALID_PARAM;
                        return result;
                    }
                }
            }
        }

        result.status = TINY_OK;
        return result;
    }

    /**
     * @name Mat::qr_decompose()
     * @brief Compute QR decomposition: A = Q * R (Q orthogonal, R upper triangular).
     * @note QR decomposition factors a matrix A into:
     *       A = Q * R
     *       where:
     *       - Q: Orthogonal matrix (Q^T * Q = I, columns are orthonormal)
     *       - R: Upper triangular matrix
     * @note Algorithm: Uses Modified Gram-Schmidt process for orthogonalization.
     *       The Gram-Schmidt process orthogonalizes the columns of A to form Q,
     *       and stores the projection coefficients in R.
     * @note Mathematical relationship:
     *       - Q is m×min(m,n) matrix with orthonormal columns
     *       - R is min(m,n)×n upper triangular matrix
     *       - For m >= n: A = Q * R (full QR)
     *       - For m < n: A = Q * R (reduced QR, Q is m×m, R is m×n)
     * @note Applications:
     *       - Least squares problems: min ||A*x - b|| → solve R*x = Q^T*b
     *       - Solving overdetermined systems
     *       - Eigenvalue computation (QR algorithm)
     *       - Matrix rank estimation
     * @note Time complexity: O(m*n²) - efficient for tall matrices (m >> n)
     * @note Numerical stability: Modified Gram-Schmidt is more stable than classical GS
     * 
     * @return QRDecomposition containing Q and R matrices and status
     */
    Mat::QRDecomposition Mat::qr_decompose() const
    {
        QRDecomposition result;

        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] qr_decompose: matrix data pointer is null\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] qr_decompose: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        int m = this->row;
        int n = this->col;
        int min_dim = (m < n) ? m : n;
        
        // Handle empty matrix
        if (m == 0 || n == 0)
        {
            // Empty matrix: Q and R are also empty
            result.Q = Mat(0, 0);
            result.R = Mat(0, 0);
            result.status = TINY_OK;
            return result;
        }

        // QR decomposition using Gram-Schmidt process
        // Use the reusable gram_schmidt_orthogonalize function
        Mat Q_ortho, R_coeff;
        if (!Mat::gram_schmidt_orthogonalize(*this, Q_ortho, R_coeff, TINY_MATH_MIN_POSITIVE_INPUT_F32))
        {
            std::cerr << "[Error] qr_decompose: gram_schmidt_orthogonalize failed\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        // Verify that Q_ortho and R_coeff were created successfully
        if (Q_ortho.data == nullptr || R_coeff.data == nullptr)
        {
            std::cerr << "[Error] qr_decompose: failed to create Q or R coefficient matrices\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        // Verify dimensions of Q_ortho and R_coeff
        if (Q_ortho.row != m || Q_ortho.col != n || 
            R_coeff.row != n || R_coeff.col != n)
        {
            std::cerr << "[Error] qr_decompose: invalid dimensions from gram_schmidt_orthogonalize\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        // Extract Q and R from the orthogonalization results
        result.Q = Q_ortho;
        result.R = Mat(m, n);
        
        // Check if R matrix was created successfully
        if (result.R.data == nullptr)
        {
            std::cerr << "[Error] qr_decompose: failed to create R matrix\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }

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
        
        // Fill remaining rows of R with zeros (if m > n)
        for (int i = min_dim; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                result.R(i, j) = 0.0f;
            }
        }

        result.status = TINY_OK;
        return result;
    }

    /**
     * @name Mat::svd_decompose()
     * @brief Compute Singular Value Decomposition: A = U * S * V^T.
     * @note SVD decomposes a matrix A (m×n) into:
     *       A = U * S * V^T
     *       where:
     *       - U: m×min(m,n) matrix with orthonormal columns (left singular vectors)
     *       - S: min(m,n)×1 vector of singular values (diagonal matrix stored as vector)
     *       - V: n×n matrix with orthonormal columns (right singular vectors)
     * @note Algorithm: Uses eigendecomposition of A^T * A to compute V and singular values.
     *       Then computes U from A * V = U * S.
     *       This is a simplified approach; full SVD uses bidiagonalization + QR iteration.
     * @note Mathematical properties:
     *       - Singular values are non-negative and sorted in descending order
     *       - U and V are orthogonal: U^T * U = I, V^T * V = I
     *       - Rank of A = number of non-zero singular values
     * @note Applications:
     *       - Rank estimation and matrix rank computation
     *       - Pseudo-inverse: A^+ = V * S^+ * U^T
     *       - Dimension reduction (PCA, data compression)
     *       - Least squares problems
     *       - Image processing and signal processing
     * @note Time complexity: O(m*n² + n³) - dominated by eigendecomposition
     * @note Note: This is a simplified implementation. For production use, consider
     *       more robust algorithms like bidiagonalization + divide-and-conquer.
     * 
     * @param max_iter Maximum number of iterations for eigendecomposition (must be > 0)
     * @param tolerance Convergence tolerance for eigendecomposition (must be >= 0)
     * @return SVDDecomposition containing U, S, V matrices, rank, iterations, and status
     */
    Mat::SVDDecomposition Mat::svd_decompose(int max_iter, float tolerance) const
    {
        SVDDecomposition result;

        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] svd_decompose: matrix data pointer is null\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] svd_decompose: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        
        // Validate parameters
        if (max_iter <= 0)
        {
            std::cerr << "[Error] svd_decompose: max_iter must be > 0 (got " << max_iter << ")\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        
        if (tolerance < 0.0f)
        {
            std::cerr << "[Error] svd_decompose: tolerance must be >= 0 (got " << tolerance << ")\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        int m = this->row;
        int n = this->col;
        int min_dim = (m < n) ? m : n;
        
        // Handle empty matrix
        if (m == 0 || n == 0)
        {
            // Empty matrix: U, S, V are also empty
            result.U = Mat(0, 0);
            result.S = Mat(0, 0);
            result.V = Mat(0, 0);
            result.rank = 0;
            result.iterations = 0;
            result.status = TINY_OK;
            return result;
        }

        // For simplicity, we use a simplified SVD algorithm
        // Full SVD implementation is complex, so we use an iterative approach
        // based on eigendecomposition of A^T * A and A * A^T

        // Compute A^T * A (n x n matrix)
        Mat AtA(n, n);
        
        // Check if AtA matrix was created successfully
        if (AtA.data == nullptr)
        {
            std::cerr << "[Error] svd_decompose: failed to create AtA matrix\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
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
            std::cerr << "[Error] svd_decompose: eigendecomposition failed with status " 
                      << eig_AtA.status << "\n";
            result.status = eig_AtA.status;
            return result;
        }
        
        // Verify eigendecomposition results
        if (eig_AtA.eigenvalues.data == nullptr || eig_AtA.eigenvectors.data == nullptr)
        {
            std::cerr << "[Error] svd_decompose: eigendecomposition returned null pointers\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        // Verify dimensions
        if (eig_AtA.eigenvalues.row != n || eig_AtA.eigenvalues.col != 1 ||
            eig_AtA.eigenvectors.row != n || eig_AtA.eigenvectors.col != n)
        {
            std::cerr << "[Error] svd_decompose: invalid dimensions from eigendecomposition\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        // Extract singular values (square root of eigenvalues of A^T * A)
        result.S = Mat(min_dim, 1);
        result.V = Mat(n, n);
        
        // Check if result matrices were created successfully
        if (result.S.data == nullptr || result.V.data == nullptr)
        {
            std::cerr << "[Error] svd_decompose: failed to create S or V matrices\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        // Extract singular values from eigenvalues
        // Note: Eigenvalues should be sorted in descending order by eigendecompose_jacobi
        // but we verify and extract only positive eigenvalues
        int sv_count = 0;
        for (int i = 0; i < n && sv_count < min_dim; ++i)
        {
            float eigenval = eig_AtA.eigenvalues(i, 0);
            
            // Check if eigenvalue is valid
            if (std::isnan(eigenval) || std::isinf(eigenval))
            {
                std::cerr << "[Warning] svd_decompose: invalid eigenvalue at index " << i 
                          << " (NaN or Inf), skipping\n";
                continue;
            }
            
            // Only consider positive eigenvalues (singular values are non-negative)
            if (eigenval > tolerance)
            {
                float sqrt_result = sqrtf(eigenval);
                
                // Check if sqrt result is valid
                if (std::isnan(sqrt_result) || std::isinf(sqrt_result))
                {
                    std::cerr << "[Warning] svd_decompose: invalid sqrt result for eigenvalue " 
                              << eigenval << " at index " << i << ", skipping\n";
                    continue;
                }
                
                result.S(sv_count, 0) = sqrt_result;
                
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
        
        // Check if U matrix was created successfully
        if (result.U.data == nullptr)
        {
            std::cerr << "[Error] svd_decompose: failed to create U matrix\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        for (int i = 0; i < sv_count; ++i)
        {
            float sigma = result.S(i, 0);
            
            // Check if sigma is valid
            if (std::isnan(sigma) || std::isinf(sigma) || sigma <= tolerance)
            {
                // Fill U column with zeros if sigma is invalid or too small
                for (int j = 0; j < m; ++j)
                {
                    result.U(j, i) = 0.0f;
                }
                continue;
            }
            
            // U(:,i) = (A * V(:,i)) / sigma
            for (int j = 0; j < m; ++j)
            {
                float sum = 0.0f;
                for (int k = 0; k < n; ++k)
                {
                    sum += (*this)(j, k) * result.V(k, i);
                }
                
                float u_val = sum / sigma;
                
                // Check if result is valid
                if (std::isnan(u_val) || std::isinf(u_val))
                {
                    std::cerr << "[Warning] svd_decompose: invalid U value at [" 
                              << j << "][" << i << "], setting to 0\n";
                    result.U(j, i) = 0.0f;
                }
                else
                {
                    result.U(j, i) = u_val;
                }
            }
        }
        
        // Fill remaining columns of U with zeros (if sv_count < min_dim)
        for (int i = sv_count; i < min_dim; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                result.U(j, i) = 0.0f;
            }
        }

        result.iterations = eig_AtA.iterations;
        result.status = TINY_OK;
        return result;
    }

    /**
     * @name Mat::solve_lu()
     * @brief Solve linear system using LU decomposition (more efficient for multiple RHS).
     * @note Solves A * x = b using precomputed LU decomposition.
     *       Algorithm: P * A = L * U, so A * x = b becomes:
     *       1. Apply permutation: P * A * x = P * b → (L * U) * x = P * b
     *       2. Forward substitution: L * y = P * b → solve for y
     *       3. Backward substitution: U * x = y → solve for x
     * @note Advantages over direct solve():
     *       - More efficient when solving multiple systems with same A
     *       - LU decomposition computed once, reused for different b vectors
     *       - Time complexity: O(n²) vs O(n³) for direct solve
     * @note Requirements:
     *       - LU decomposition must be valid (status == TINY_OK)
     *       - Matrix A must be square and non-singular
     *       - Vector b must have same dimension as A
     * 
     * @param lu LU decomposition of coefficient matrix A
     * @param b Right-hand side vector (must be column vector)
     * @return Solution vector x such that A * x = b, or empty matrix on error
     */
    Mat Mat::solve_lu(const LUDecomposition &lu, const Mat &b)
    {
        // Check LU decomposition status
        if (lu.status != TINY_OK)
        {
            std::cerr << "[Error] solve_lu: invalid LU decomposition (status: " 
                      << lu.status << ")\n";
            return Mat();
        }
        
        // Check for null pointers
        if (lu.L.data == nullptr || lu.U.data == nullptr)
        {
            std::cerr << "[Error] solve_lu: LU decomposition matrices have null pointers\n";
            return Mat();
        }
        
        if (b.data == nullptr)
        {
            std::cerr << "[Error] solve_lu: right-hand side vector has null pointer\n";
            return Mat();
        }
        
        // Validate LU decomposition dimensions
        if (lu.L.row <= 0 || lu.L.col <= 0 || lu.U.row <= 0 || lu.U.col <= 0)
        {
            std::cerr << "[Error] solve_lu: invalid LU decomposition dimensions\n";
            return Mat();
        }
        
        // Check if L and U are square and have same size
        if (lu.L.row != lu.L.col || lu.U.row != lu.U.col || lu.L.row != lu.U.row)
        {
            std::cerr << "[Error] solve_lu: L and U must be square matrices of same size\n";
            return Mat();
        }
        
        int n = lu.L.row;
        
        // Handle empty matrix
        if (n == 0)
        {
            return Mat(0, 1);  // Return empty solution vector
        }
        
        // Validate right-hand side vector dimensions
        if (b.row != n || b.col != 1)
        {
            std::cerr << "[Error] solve_lu: dimension mismatch - b must be " 
                      << n << "x1 vector (got " << b.row << "x" << b.col << ")\n";
            return Mat();
        }
        
        // Check permutation matrix if pivoting was used
        if (lu.pivoted)
        {
            if (lu.P.data == nullptr)
            {
                std::cerr << "[Error] solve_lu: pivoting enabled but P matrix is null\n";
                return Mat();
            }
            
            if (lu.P.row != n || lu.P.col != n)
            {
                std::cerr << "[Error] solve_lu: P matrix dimensions mismatch (got " 
                          << lu.P.row << "x" << lu.P.col << ", expected " << n << "x" << n << ")\n";
                return Mat();
            }
        }

        // Apply permutation if pivoting was used
        Mat b_perm = b;
        if (lu.pivoted)
        {
            // b_perm = P * b
            b_perm = Mat(n, 1);
            
            // Check if b_perm was created successfully
            if (b_perm.data == nullptr)
            {
                std::cerr << "[Error] solve_lu: failed to create permuted vector\n";
                return Mat();
            }
            
            for (int i = 0; i < n; ++i)
            {
                bool found = false;
                for (int j = 0; j < n; ++j)
                {
                    // P is permutation matrix: each row/column has exactly one 1.0
                    // Use tolerance for floating-point comparison
                    if (fabsf(lu.P(i, j) - 1.0f) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                    {
                        b_perm(i, 0) = b(j, 0);
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    std::cerr << "[Warning] solve_lu: no 1.0 found in row " << i 
                              << " of permutation matrix, using 0.0\n";
                    b_perm(i, 0) = 0.0f;
                }
            }
        }

        // Solve L * y = b_perm (forward substitution)
        // L is lower triangular with unit diagonal
        Mat y(n, 1);
        
        // Check if y was created successfully
        if (y.data == nullptr)
        {
            std::cerr << "[Error] solve_lu: failed to create intermediate vector y\n";
            return Mat();
        }
        
        for (int i = 0; i < n; ++i)
        {
            float sum = b_perm(i, 0);
            for (int j = 0; j < i; ++j)
            {
                sum -= lu.L(i, j) * y(j, 0);
            }
            y(i, 0) = sum;  // L has unit diagonal, so no division needed
            
            // Check if result is valid
            if (std::isnan(y(i, 0)) || std::isinf(y(i, 0)))
            {
                std::cerr << "[Error] solve_lu: invalid intermediate value at index " << i << "\n";
                return Mat();
            }
        }

        // Solve U * x = y (backward substitution)
        // U is upper triangular
        Mat x(n, 1);
        
        // Check if x was created successfully
        if (x.data == nullptr)
        {
            std::cerr << "[Error] solve_lu: failed to create solution vector x\n";
            return Mat();
        }
        
        for (int i = n - 1; i >= 0; --i)
        {
            float sum = y(i, 0);
            for (int j = i + 1; j < n; ++j)
            {
                sum -= lu.U(i, j) * x(j, 0);
            }
            
            // Check if diagonal element is valid
            float u_ii = lu.U(i, i);
            if (std::isnan(u_ii) || std::isinf(u_ii))
            {
                std::cerr << "[Error] solve_lu: invalid diagonal element U[" << i << "][" << i << "]\n";
                return Mat();
            }
            
            if (fabsf(u_ii) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] solve_lu: singular matrix (U[" << i << "][" << i 
                          << "] = " << u_ii << " is too small)\n";
                return Mat();
            }
            
            x(i, 0) = sum / u_ii;
            
            // Check if result is valid
            if (std::isnan(x(i, 0)) || std::isinf(x(i, 0)))
            {
                std::cerr << "[Error] solve_lu: invalid solution value at index " << i << "\n";
                return Mat();
            }
        }

        return x;
    }

    /**
     * @name Mat::solve_cholesky()
     * @brief Solve linear system using Cholesky decomposition (for SPD matrices).
     * @note Solves A * x = b using precomputed Cholesky decomposition.
     *       Algorithm: A = L * L^T, so A * x = b becomes:
     *       1. Forward substitution: L * y = b → solve for y
     *       2. Backward substitution: L^T * x = y → solve for x
     * @note Advantages over LU decomposition:
     *       - Faster: O(n²) vs O(n²) but with better numerical stability
     *       - More stable: no pivoting needed for SPD matrices
     *       - Uses less memory: only stores L (not L and U)
     *       - Guaranteed to work for positive definite matrices
     * @note Requirements:
     *       - Cholesky decomposition must be valid (status == TINY_OK)
     *       - Matrix A must be symmetric positive definite (SPD)
     *       - Vector b must have same dimension as A
     * @note Time complexity: O(n²) - efficient for SPD systems
     * 
     * @param chol Cholesky decomposition of coefficient matrix A (A = L * L^T)
     * @param b Right-hand side vector (must be column vector)
     * @return Solution vector x such that A * x = b, or empty matrix on error
     */
    Mat Mat::solve_cholesky(const CholeskyDecomposition &chol, const Mat &b)
    {
        // Check Cholesky decomposition status
        if (chol.status != TINY_OK)
        {
            std::cerr << "[Error] solve_cholesky: invalid Cholesky decomposition (status: " 
                      << chol.status << ")\n";
            return Mat();
        }
        
        // Check for null pointers
        if (chol.L.data == nullptr)
        {
            std::cerr << "[Error] solve_cholesky: Cholesky L matrix has null pointer\n";
            return Mat();
        }
        
        if (b.data == nullptr)
        {
            std::cerr << "[Error] solve_cholesky: right-hand side vector has null pointer\n";
            return Mat();
        }
        
        // Validate Cholesky decomposition dimensions
        if (chol.L.row <= 0 || chol.L.col <= 0)
        {
            std::cerr << "[Error] solve_cholesky: invalid Cholesky decomposition dimensions\n";
            return Mat();
        }
        
        // Check if L is square
        if (chol.L.row != chol.L.col)
        {
            std::cerr << "[Error] solve_cholesky: L must be a square matrix (got " 
                      << chol.L.row << "x" << chol.L.col << ")\n";
            return Mat();
        }
        
        int n = chol.L.row;
        
        // Handle empty matrix
        if (n == 0)
        {
            return Mat(0, 1);  // Return empty solution vector
        }
        
        // Validate right-hand side vector dimensions
        if (b.row != n || b.col != 1)
        {
            std::cerr << "[Error] solve_cholesky: dimension mismatch - b must be " 
                      << n << "x1 vector (got " << b.row << "x" << b.col << ")\n";
            return Mat();
        }

        // Solve L * y = b (forward substitution)
        // L is lower triangular with positive diagonal elements
        Mat y(n, 1);
        
        // Check if y was created successfully
        if (y.data == nullptr)
        {
            std::cerr << "[Error] solve_cholesky: failed to create intermediate vector y\n";
            return Mat();
        }
        
        for (int i = 0; i < n; ++i)
        {
            float sum = b(i, 0);
            for (int j = 0; j < i; ++j)
            {
                sum -= chol.L(i, j) * y(j, 0);
            }
            
            // Check if diagonal element is valid
            float l_ii = chol.L(i, i);
            if (std::isnan(l_ii) || std::isinf(l_ii))
            {
                std::cerr << "[Error] solve_cholesky: invalid diagonal element L[" << i << "][" << i << "]\n";
                return Mat();
            }
            
            if (fabsf(l_ii) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] solve_cholesky: singular matrix (L[" << i << "][" << i 
                          << "] = " << l_ii << " is too small)\n";
                return Mat();
            }
            
            y(i, 0) = sum / l_ii;
            
            // Check if result is valid
            if (std::isnan(y(i, 0)) || std::isinf(y(i, 0)))
            {
                std::cerr << "[Error] solve_cholesky: invalid intermediate value at index " << i << "\n";
                return Mat();
            }
        }

        // Solve L^T * x = y (backward substitution)
        // L^T is upper triangular (transpose of L)
        Mat x(n, 1);
        
        // Check if x was created successfully
        if (x.data == nullptr)
        {
            std::cerr << "[Error] solve_cholesky: failed to create solution vector x\n";
            return Mat();
        }
        
        for (int i = n - 1; i >= 0; --i)
        {
            float sum = y(i, 0);
            for (int j = i + 1; j < n; ++j)
            {
                // L^T(j,i) = L(i,j), so we access L(j,i) for L^T(j,i)
                sum -= chol.L(j, i) * x(j, 0);
            }
            
            // Check if diagonal element is valid (same as forward substitution)
            float l_ii = chol.L(i, i);
            if (std::isnan(l_ii) || std::isinf(l_ii))
            {
                std::cerr << "[Error] solve_cholesky: invalid diagonal element L[" << i << "][" << i << "]\n";
                return Mat();
            }
            
            if (fabsf(l_ii) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] solve_cholesky: singular matrix (L[" << i << "][" << i 
                          << "] = " << l_ii << " is too small)\n";
                return Mat();
            }
            
            x(i, 0) = sum / l_ii;
            
            // Check if result is valid
            if (std::isnan(x(i, 0)) || std::isinf(x(i, 0)))
            {
                std::cerr << "[Error] solve_cholesky: invalid solution value at index " << i << "\n";
                return Mat();
            }
        }

        return x;
    }

    /**
     * @name Mat::solve_qr()
     * @brief Solve linear system using QR decomposition (least squares solution).
     * @note Solves A * x = b using precomputed QR decomposition.
     *       Algorithm: A = Q * R, so A * x = b becomes:
     *       1. Compute Q^T * b (project b onto column space of Q)
     *       2. Solve R * x = Q^T * b using backward substitution
     * @note This method provides least squares solution:
     *       - For overdetermined systems (m > n): minimizes ||A * x - b||
     *       - For determined systems (m = n): exact solution
     *       - For underdetermined systems (m < n): minimum norm solution
     * @note Advantages:
     *       - Numerically stable (no pivoting needed)
     *       - Works for rectangular matrices
     *       - Handles rank-deficient matrices gracefully
     * @note Requirements:
     *       - QR decomposition must be valid (status == TINY_OK)
     *       - Vector b must have same number of rows as Q
     * @note Time complexity: O(m*n) - efficient for least squares problems
     * 
     * @param qr QR decomposition of coefficient matrix A (A = Q * R)
     * @param b Right-hand side vector (must be column vector)
     * @return Least squares solution vector x such that ||A * x - b|| is minimized,
     *         or empty matrix on error
     */
    Mat Mat::solve_qr(const QRDecomposition &qr, const Mat &b)
    {
        // Check QR decomposition status
        if (qr.status != TINY_OK)
        {
            std::cerr << "[Error] solve_qr: invalid QR decomposition (status: " 
                      << qr.status << ")\n";
            return Mat();
        }
        
        // Check for null pointers
        if (qr.Q.data == nullptr || qr.R.data == nullptr)
        {
            std::cerr << "[Error] solve_qr: QR decomposition matrices have null pointers\n";
            return Mat();
        }
        
        if (b.data == nullptr)
        {
            std::cerr << "[Error] solve_qr: right-hand side vector has null pointer\n";
            return Mat();
        }
        
        // Validate QR decomposition dimensions
        if (qr.Q.row <= 0 || qr.Q.col <= 0 || qr.R.row <= 0 || qr.R.col <= 0)
        {
            std::cerr << "[Error] solve_qr: invalid QR decomposition dimensions\n";
            return Mat();
        }
        
        int m = qr.Q.row;
        int n = qr.R.col;
        int min_dim = (m < n) ? m : n;
        
        // Verify Q and R dimensions are consistent
        // Q should be m×min(m,n) or m×n, R should be m×n or min(m,n)×n
        if (qr.Q.col < min_dim || qr.R.row < min_dim)
        {
            std::cerr << "[Error] solve_qr: inconsistent QR decomposition dimensions\n";
            return Mat();
        }
        
        // Handle empty matrix
        if (m == 0 || n == 0)
        {
            return Mat(n, 1);  // Return empty solution vector with correct dimension
        }
        
        // Validate right-hand side vector dimensions
        if (b.row != m || b.col != 1)
        {
            std::cerr << "[Error] solve_qr: dimension mismatch - b must be " 
                      << m << "x1 vector (got " << b.row << "x" << b.col << ")\n";
            return Mat();
        }

        // Compute Q^T * b
        // Note: Q^T has dimensions min(m,n)×m, so Q^T * b has dimension min(m,n)×1
        // But we need n×1 for backward substitution, so we compute first min(m,n) components
        Mat Qt_b(n, 1);
        
        // Check if Qt_b was created successfully
        if (Qt_b.data == nullptr)
        {
            std::cerr << "[Error] solve_qr: failed to create Qt_b vector\n";
            return Mat();
        }
        
        // Initialize Qt_b to zero
        for (int i = 0; i < n; ++i)
        {
            Qt_b(i, 0) = 0.0f;
        }
        
        // Compute Q^T * b (only first min(m,n) components, rest are zero)
        for (int i = 0; i < min_dim; ++i)
        {
            float sum = 0.0f;
            for (int j = 0; j < m; ++j)
            {
                // Q^T(i,j) = Q(j,i)
                sum += qr.Q(j, i) * b(j, 0);
            }
            Qt_b(i, 0) = sum;
            
            // Check if result is valid
            if (std::isnan(Qt_b(i, 0)) || std::isinf(Qt_b(i, 0)))
            {
                std::cerr << "[Error] solve_qr: invalid Qt_b value at index " << i << "\n";
                return Mat();
            }
        }

        // Solve R * x = Q^T * b (backward substitution)
        // R is upper triangular (or upper trapezoidal if m < n)
        Mat x(n, 1);
        
        // Check if x was created successfully
        if (x.data == nullptr)
        {
            std::cerr << "[Error] solve_qr: failed to create solution vector x\n";
            return Mat();
        }
        
        // Initialize x to zero
        for (int i = 0; i < n; ++i)
        {
            x(i, 0) = 0.0f;
        }
        
        // Backward substitution (only for first min(m,n) rows of R)
        for (int i = min_dim - 1; i >= 0; --i)
        {
            // Check if diagonal element is valid
            float r_ii = qr.R(i, i);
            if (std::isnan(r_ii) || std::isinf(r_ii))
            {
                std::cerr << "[Error] solve_qr: invalid diagonal element R[" << i << "][" << i << "]\n";
                return Mat();
            }
            
            if (fabsf(r_ii) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                // Skip zero diagonal (underdetermined system or rank-deficient)
                // Set x[i] = 0 (minimum norm solution)
                x(i, 0) = 0.0f;
                continue;
            }
            
            float sum = Qt_b(i, 0);
            
            // Sum over upper triangular part (only up to n columns)
            int max_j = (qr.R.col < n) ? qr.R.col : n;
            for (int j = i + 1; j < max_j; ++j)
            {
                sum -= qr.R(i, j) * x(j, 0);
            }
            
            x(i, 0) = sum / r_ii;
            
            // Check if result is valid
            if (std::isnan(x(i, 0)) || std::isinf(x(i, 0)))
            {
                std::cerr << "[Error] solve_qr: invalid solution value at index " << i << "\n";
                return Mat();
            }
        }

        // Set remaining components to zero if n > m (underdetermined system)
        // This is already done by initialization, but we keep it explicit
        for (int i = min_dim; i < n; ++i)
        {
            x(i, 0) = 0.0f;
        }

        return x;
    }

    /**
     * @name Mat::pseudo_inverse()
     * @brief Compute pseudo-inverse using SVD: A^+ = V * S^+ * U^T.
     * @note Pseudo-inverse (Moore-Penrose inverse) is the generalization of matrix inverse
     *       for rectangular or singular matrices.
     * @note Algorithm: A = U * S * V^T, so A^+ = V * S^+ * U^T
     *       where S^+ is the pseudo-inverse of S (diagonal matrix):
     *       - If σ_i > tolerance: S^+[i][i] = 1/σ_i
     *       - If σ_i <= tolerance: S^+[i][i] = 0 (treat as zero)
     * @note Mathematical properties:
     *       - A * A^+ * A = A
     *       - A^+ * A * A^+ = A^+
     *       - (A * A^+)^T = A * A^+
     *       - (A^+ * A)^T = A^+ * A
     * @note Applications:
     *       - Solving least squares problems: x = A^+ * b
     *       - Solving underdetermined systems (minimum norm solution)
     *       - Solving overdetermined systems (least squares solution)
     *       - Rank-deficient matrix inversion
     * @note Time complexity: O(m*n*rank) - dominated by matrix multiplications
     * 
     * @param svd SVD decomposition of matrix A (A = U * S * V^T)
     * @param tolerance Tolerance for singular values (must be >= 0).
     *                  Singular values <= tolerance are treated as zero.
     * @return Pseudo-inverse matrix A^+ (n×m matrix), or empty matrix on error
     */
    Mat Mat::pseudo_inverse(const SVDDecomposition &svd, float tolerance)
    {
        // Check SVD decomposition status
        if (svd.status != TINY_OK)
        {
            std::cerr << "[Error] pseudo_inverse: invalid SVD decomposition (status: " 
                      << svd.status << ")\n";
            return Mat();
        }
        
        // Check for null pointers
        if (svd.U.data == nullptr || svd.V.data == nullptr || svd.S.data == nullptr)
        {
            std::cerr << "[Error] pseudo_inverse: SVD decomposition matrices have null pointers\n";
            return Mat();
        }
        
        // Validate parameters
        if (tolerance < 0.0f)
        {
            std::cerr << "[Error] pseudo_inverse: tolerance must be >= 0 (got " << tolerance << ")\n";
            return Mat();
        }
        
        // Validate SVD decomposition dimensions
        if (svd.U.row <= 0 || svd.U.col <= 0 || 
            svd.V.row <= 0 || svd.V.col <= 0 ||
            svd.S.row <= 0 || svd.S.col <= 0)
        {
            std::cerr << "[Error] pseudo_inverse: invalid SVD decomposition dimensions\n";
            return Mat();
        }
        
        int m = svd.U.row;  // Original matrix A has m rows
        int n = svd.V.row;  // Original matrix A has n columns
        int min_dim = (m < n) ? m : n;
        int rank = svd.rank;
        
        // Validate rank
        if (rank < 0 || rank > min_dim)
        {
            std::cerr << "[Error] pseudo_inverse: invalid rank " << rank 
                      << " (expected 0 to " << min_dim << ")\n";
            return Mat();
        }
        
        // Verify SVD dimensions
        // U should be m×min(m,n), V should be n×n, S should be min(m,n)×1
        if (svd.U.col < min_dim || svd.V.col < n || svd.S.row < min_dim || svd.S.col != 1)
        {
            std::cerr << "[Error] pseudo_inverse: inconsistent SVD dimensions\n";
            return Mat();
        }
        
        // Handle empty matrix
        if (m == 0 || n == 0)
        {
            return Mat(n, m);  // Return empty pseudo-inverse with correct dimensions
        }

        // Compute S^+ (pseudo-inverse of S)
        // S^+ is a min(m,n)×min(m,n) diagonal matrix
        // For efficiency, we'll compute V * S^+ directly without storing S^+ explicitly
        // S^+[i][i] = 1/σ_i if σ_i > tolerance, else 0
        
        // Compute A^+ = V * S^+ * U^T
        // Dimensions: V is n×n, S^+ is min(m,n)×min(m,n), U^T is min(m,n)×m
        // Result: A^+ is n×m
        
        Mat A_plus(n, m);
        
        // Check if A_plus was created successfully
        if (A_plus.data == nullptr)
        {
            std::cerr << "[Error] pseudo_inverse: failed to create result matrix\n";
            return Mat();
        }
        
        // Initialize A_plus to zero
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                A_plus(i, j) = 0.0f;
            }
        }
        
        // Compute A^+ = V * S^+ * U^T
        // For each element A^+[i][j]:
        //   A^+[i][j] = sum(V[i][k] * (1/σ_k) * U[j][k], k=0..rank-1)
        //   where σ_k > tolerance
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                float sum = 0.0f;
                for (int k = 0; k < rank; ++k)
                {
                    // Check if k is within valid range
                    if (k >= svd.S.row)
                    {
                        break;
                    }
                    
                    float sigma = svd.S(k, 0);
                    
                    // Check if sigma is valid
                    if (std::isnan(sigma) || std::isinf(sigma))
                    {
                        std::cerr << "[Warning] pseudo_inverse: invalid singular value at index " 
                                  << k << ", skipping\n";
                        continue;
                    }
                    
                    // Only use singular values above tolerance
                    if (sigma > tolerance)
                    {
                        float inv_sigma = 1.0f / sigma;
                        
                        // Check if inverse is valid
                        if (std::isnan(inv_sigma) || std::isinf(inv_sigma))
                        {
                            std::cerr << "[Warning] pseudo_inverse: invalid inverse of singular value " 
                                      << sigma << " at index " << k << ", skipping\n";
                            continue;
                        }
                        
                        // A^+[i][j] += V[i][k] * (1/σ_k) * U[j][k]
                        // Note: U^T[k][j] = U[j][k]
                        if (k < svd.V.col && k < svd.U.col)
                        {
                            float term = svd.V(i, k) * inv_sigma * svd.U(j, k);
                            
                            // Check if term is valid
                            if (std::isnan(term) || std::isinf(term))
                            {
                                std::cerr << "[Warning] pseudo_inverse: invalid term at [" 
                                          << i << "][" << j << "], k=" << k << ", skipping\n";
                                continue;
                            }
                            
                            sum += term;
                        }
                    }
                }
                
                A_plus(i, j) = sum;
                
                // Check if result is valid
                if (std::isnan(A_plus(i, j)) || std::isinf(A_plus(i, j)))
                {
                    std::cerr << "[Warning] pseudo_inverse: invalid result at [" 
                              << i << "][" << j << "], setting to 0\n";
                    A_plus(i, j) = 0.0f;
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
     * @note A matrix A is symmetric if A^T = A, i.e., A[i][j] = A[j][i] for all i, j.
     * @note Essential for SHM applications where structural matrices are typically symmetric.
     * @note Time complexity: O(n²) - checks upper triangular part only
     * 
     * @param tolerance Maximum allowed difference between A(i,j) and A(j,i) (must be >= 0).
     *                  Used to handle floating-point numerical errors.
     * @return true if matrix is symmetric, false otherwise
     */
    bool Mat::is_symmetric(float tolerance) const
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] is_symmetric: matrix data pointer is null\n";
            return false;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] is_symmetric: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return false;
        }
        
        // Validate tolerance
        if (tolerance < 0.0f)
        {
            std::cerr << "[Error] is_symmetric: tolerance must be >= 0 (got " << tolerance << ")\n";
            return false;
        }
        
        // Only square matrices can be symmetric
        if (this->row != this->col)
        {
            return false;
        }
        
        int n = this->row;
        
        // Handle empty matrix (0x0 is trivially symmetric)
        if (n == 0)
        {
            return true;
        }

        // Check symmetry: A(i,j) should equal A(j,i) within tolerance
        // Only check upper triangular part (i < j) to avoid redundant checks
        for (int i = 0; i < n; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                float a_ij = (*this)(i, j);
                float a_ji = (*this)(j, i);
                
                // Check if values are valid
                if (std::isnan(a_ij) || std::isnan(a_ji) || 
                    std::isinf(a_ij) || std::isinf(a_ji))
                {
                    std::cerr << "[Warning] is_symmetric: invalid matrix elements at [" 
                              << i << "][" << j << "] or [" << j << "][" << i << "]\n";
                    return false;
                }
                
                float diff = fabsf(a_ij - a_ji);
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
     * @note Power iteration finds the eigenvalue with the largest absolute value and its corresponding eigenvector.
     *       Algorithm: v_{k+1} = A * v_k / ||A * v_k||, converges to dominant eigenvector.
     * @note Fast method suitable for real-time SHM applications to quickly identify primary frequency.
     * @note Time complexity: O(n² * iterations) - efficient for sparse or large matrices
     * @note Convergence: Requires that the dominant eigenvalue is unique and has larger magnitude than others.
     * 
     * @param max_iter Maximum number of iterations (must be > 0)
     * @param tolerance Convergence tolerance (must be >= 0). Convergence when |λ_k - λ_{k-1}| < tolerance * |λ_k|
     * @return EigenPair containing the dominant eigenvalue, eigenvector, and status
     */
    Mat::EigenPair Mat::power_iteration(int max_iter, float tolerance) const
    {
        EigenPair result;

        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] power_iteration: matrix data pointer is null\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] power_iteration: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        // Validation: must be square matrix
        if (this->row != this->col)
        {
            std::cerr << "[Error] power_iteration: requires square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        
        // Validate parameters
        if (max_iter <= 0)
        {
            std::cerr << "[Error] power_iteration: max_iter must be > 0 (got " << max_iter << ")\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        
        if (tolerance < 0.0f)
        {
            std::cerr << "[Error] power_iteration: tolerance must be >= 0 (got " << tolerance << ")\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        int n = this->row;
        
        // Handle empty matrix
        if (n == 0)
        {
            result.eigenvalue = 0.0f;
            result.eigenvector = Mat(0, 1);
            result.iterations = 0;
            result.status = TINY_OK;
            return result;
        }

        // Initialize eigenvector with better strategy to avoid convergence to smaller eigenvalues
        // Strategy: Use sum of columns (or rows) to get a vector with components in all directions
        result.eigenvector = Mat(n, 1);
        
        // Check if eigenvector was created successfully
        if (result.eigenvector.data == nullptr)
        {
            std::cerr << "[Error] power_iteration: failed to create eigenvector\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
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
            norm_sq = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                result.eigenvector(i, 0) = 1.0f + 0.1f * static_cast<float>(i);
                norm_sq += result.eigenvector(i, 0) * result.eigenvector(i, 0);
            }
        }
        
        // Normalize initial eigenvector
        float sqrt_norm = sqrtf(norm_sq);
        if (std::isnan(sqrt_norm) || std::isinf(sqrt_norm) || sqrt_norm < TINY_MATH_MIN_POSITIVE_INPUT_F32)
        {
            std::cerr << "[Error] power_iteration: invalid initial eigenvector norm\n";
            result.status = TINY_ERR_MATH_INVALID_PARAM;
            return result;
        }
        
        float inv_norm = 1.0f / sqrt_norm;
        for (int i = 0; i < n; ++i)
        {
            result.eigenvector(i, 0) *= inv_norm;
        }

        // Power iteration loop
        Mat temp_vec(n, 1);
        
        // Check if temp_vec was created successfully
        if (temp_vec.data == nullptr)
        {
            std::cerr << "[Error] power_iteration: failed to create temporary vector\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
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
                
                // Check if result is valid
                if (std::isnan(temp_vec(i, 0)) || std::isinf(temp_vec(i, 0)))
                {
                    std::cerr << "[Error] power_iteration: invalid matrix-vector product at index " << i << "\n";
                    result.status = TINY_ERR_MATH_INVALID_PARAM;
                    return result;
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
                std::cerr << "[Error] power_iteration: eigenvector norm too small\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }

            result.eigenvalue = numerator / denominator;
            
            // Check if eigenvalue is valid
            if (std::isnan(result.eigenvalue) || std::isinf(result.eigenvalue))
            {
                std::cerr << "[Error] power_iteration: invalid eigenvalue computed\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }

            // Normalize the new vector
            float new_norm_sq = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                new_norm_sq += temp_vec(i, 0) * temp_vec(i, 0);
            }

            if (new_norm_sq < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] power_iteration: computed vector norm too small\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }

            float new_sqrt_norm = sqrtf(new_norm_sq);
            if (std::isnan(new_sqrt_norm) || std::isinf(new_sqrt_norm))
            {
                std::cerr << "[Error] power_iteration: invalid sqrt of vector norm\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }
            
            float new_inv_norm = 1.0f / new_sqrt_norm;
            for (int i = 0; i < n; ++i)
            {
                result.eigenvector(i, 0) = temp_vec(i, 0) * new_inv_norm;
                
                // Check if eigenvector component is valid
                if (std::isnan(result.eigenvector(i, 0)) || std::isinf(result.eigenvector(i, 0)))
                {
                    std::cerr << "[Error] power_iteration: invalid eigenvector component at index " << i << "\n";
                    result.status = TINY_ERR_MATH_INVALID_PARAM;
                    return result;
                }
            }

            // Check convergence
            if (iter > 0)
            {
                float eigenvalue_change = fabsf(result.eigenvalue - prev_eigenvalue);
                float abs_eigenvalue = fabsf(result.eigenvalue);
                
                // Avoid division by zero
                if (abs_eigenvalue < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                {
                    // If eigenvalue is near zero, check absolute change
                    if (eigenvalue_change < tolerance)
                    {
                        result.iterations = iter + 1;
                        result.status = TINY_OK;
                        return result;
                    }
                }
                else
                {
                    // Relative change check
                    if (eigenvalue_change < tolerance * abs_eigenvalue)
                    {
                        result.iterations = iter + 1;
                        result.status = TINY_OK;
                        return result;
                    }
                }
            }

            prev_eigenvalue = result.eigenvalue;
        }

        // Max iterations reached
        result.iterations = max_iter;
        result.status = TINY_ERR_NOT_FINISHED;
        std::cerr << "[Warning] power_iteration: did not converge within " << max_iter << " iterations\n";
        return result;
    }

    /**
     * @name Mat::inverse_power_iteration()
     * @brief Compute the smallest (minimum magnitude) eigenvalue and eigenvector using inverse power iteration.
     * @note Inverse power iteration finds the eigenvalue with the smallest absolute value and its eigenvector.
     *       Algorithm: v_{k+1} = A^(-1) * v_k / ||A^(-1) * v_k||, converges to smallest eigenvector.
     * @note Critical for system identification - finds fundamental frequency/lowest mode in structural dynamics.
     *       This method is essential for SHM applications where the smallest eigenvalue corresponds to the
     *       fundamental frequency of the system.
     * @note Time complexity: O(n³ * iterations) - each iteration requires solving a linear system
     * @note The matrix must be invertible (non-singular) for this method to work.
     *       If the matrix is singular or near-singular, the method will fail gracefully.
     * 
     * @param max_iter Maximum number of iterations (must be > 0)
     * @param tolerance Convergence tolerance (must be >= 0). Convergence when |λ_k - λ_{k-1}| < tolerance * max(|λ_k|, 1)
     * @return EigenPair containing the smallest eigenvalue, eigenvector, and status
     */
    Mat::EigenPair Mat::inverse_power_iteration(int max_iter, float tolerance) const
    {
        EigenPair result;

        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] inverse_power_iteration: matrix data pointer is null\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] inverse_power_iteration: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        // Validation: must be square matrix
        if (this->row != this->col)
        {
            std::cerr << "[Error] inverse_power_iteration: requires square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        
        // Validate parameters
        if (max_iter <= 0)
        {
            std::cerr << "[Error] inverse_power_iteration: max_iter must be > 0 (got " << max_iter << ")\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        
        if (tolerance < 0.0f)
        {
            std::cerr << "[Error] inverse_power_iteration: tolerance must be >= 0 (got " << tolerance << ")\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        int n = this->row;
        
        // Handle empty matrix
        if (n == 0)
        {
            result.eigenvalue = 0.0f;
            result.eigenvector = Mat(0, 1);
            result.iterations = 0;
            result.status = TINY_OK;
            return result;
        }

        // Check if matrix is singular by computing determinant (quick check)
        // For efficiency, we'll check during the first solve operation instead

        // Initialize eigenvector for inverse power iteration
        // Strategy: Use a vector that is orthogonal to the dominant eigenvector direction
        // For inverse power iteration, we want to converge to the smallest eigenvalue
        // Use a simple initialization: [1, 1, ..., 1]^T normalized, which typically
        // has components in all eigenvector directions
        result.eigenvector = Mat(n, 1);
        
        // Check if eigenvector was created successfully
        if (result.eigenvector.data == nullptr)
        {
            std::cerr << "[Error] inverse_power_iteration: failed to create eigenvector\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
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
            norm_sq = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                result.eigenvector(i, 0) = 1.0f;
                norm_sq += 1.0f;
            }
        }
        
        float sqrt_norm = sqrtf(norm_sq);
        if (std::isnan(sqrt_norm) || std::isinf(sqrt_norm) || sqrt_norm < TINY_MATH_MIN_POSITIVE_INPUT_F32)
        {
            std::cerr << "[Error] inverse_power_iteration: invalid initial eigenvector norm\n";
            result.status = TINY_ERR_MATH_INVALID_PARAM;
            return result;
        }
        
        float inv_norm = 1.0f / sqrt_norm;
        for (int i = 0; i < n; ++i)
        {
            result.eigenvector(i, 0) *= inv_norm;
        }

        // Inverse power iteration loop
        Mat temp_vec(n, 1);
        
        // Check if temp_vec was created successfully
        if (temp_vec.data == nullptr)
        {
            std::cerr << "[Error] inverse_power_iteration: failed to create temporary vector\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
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
            
            // Check if eigenvalue is valid
            if (std::isnan(result.eigenvalue) || std::isinf(result.eigenvalue))
            {
                std::cerr << "[Error] inverse_power_iteration: invalid eigenvalue computed\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }

            // Normalize the new vector
            float new_norm_sq = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                new_norm_sq += temp_vec(i, 0) * temp_vec(i, 0);
            }

            if (new_norm_sq < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] inverse_power_iteration: computed vector norm too small\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }

            float new_sqrt_norm = sqrtf(new_norm_sq);
            if (std::isnan(new_sqrt_norm) || std::isinf(new_sqrt_norm))
            {
                std::cerr << "[Error] inverse_power_iteration: invalid sqrt of vector norm\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }
            
            float new_inv_norm = 1.0f / new_sqrt_norm;
            for (int i = 0; i < n; ++i)
            {
                result.eigenvector(i, 0) = temp_vec(i, 0) * new_inv_norm;
                
                // Check if eigenvector component is valid
                if (std::isnan(result.eigenvector(i, 0)) || std::isinf(result.eigenvector(i, 0)))
                {
                    std::cerr << "[Error] inverse_power_iteration: invalid eigenvector component at index " << i << "\n";
                    result.status = TINY_ERR_MATH_INVALID_PARAM;
                    return result;
                }
            }

            // Check convergence
            if (iter > 0)
            {
                float eigenvalue_change = fabsf(result.eigenvalue - prev_eigenvalue);
                // Use relative tolerance for convergence check
                float abs_eigenvalue = fabsf(result.eigenvalue);
                float rel_tolerance = tolerance * fmaxf(abs_eigenvalue, 1.0f);
                
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
        std::cerr << "[Warning] inverse_power_iteration: did not converge within " << max_iter << " iterations\n";
        return result;
    }

    /**
     * @name Mat::eigendecompose_jacobi()
     * @brief Compute complete eigenvalue decomposition using Jacobi method for symmetric matrices.
     * @note Jacobi method iteratively applies Givens rotations to diagonalize a symmetric matrix.
     *       Algorithm: Repeatedly find largest off-diagonal element and eliminate it with a rotation.
     * @note Robust and accurate method ideal for structural dynamics matrices in SHM.
     * @note Time complexity: O(n³ * iterations) - typically converges in O(n²) iterations
     * @note Best for: Symmetric matrices (required), small to medium sized matrices (n < 100)
     * 
     * @param tolerance Convergence tolerance (must be >= 0). Convergence when max off-diagonal < tolerance
     * @param max_iter Maximum number of iterations (must be > 0)
     * @return EigenDecomposition containing all eigenvalues, eigenvectors, and status
     */
    Mat::EigenDecomposition Mat::eigendecompose_jacobi(float tolerance, int max_iter) const
    {
        EigenDecomposition result;

        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] eigendecompose_jacobi: matrix data pointer is null\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] eigendecompose_jacobi: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        // Validation: must be square matrix
        if (this->row != this->col)
        {
            std::cerr << "[Error] eigendecompose_jacobi: requires square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        
        // Validate parameters
        if (tolerance < 0.0f)
        {
            std::cerr << "[Error] eigendecompose_jacobi: tolerance must be >= 0 (got " << tolerance << ")\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        
        if (max_iter <= 0)
        {
            std::cerr << "[Error] eigendecompose_jacobi: max_iter must be > 0 (got " << max_iter << ")\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        // Check if matrix is symmetric
        if (!this->is_symmetric(tolerance * 10.0f))
        {
            std::cerr << "[Warning] eigendecompose_jacobi: matrix is not symmetric. "
                      << "Jacobi method may not converge correctly.\n";
        }

        int n = this->row;
        
        // Handle empty matrix
        if (n == 0)
        {
            result.eigenvalues = Mat(0, 1);
            result.eigenvectors = Mat(0, 0);
            result.iterations = 0;
            result.status = TINY_OK;
            return result;
        }

        // Initialize: working copy of matrix, eigenvectors as identity
        Mat A = Mat(*this); // Working copy (will become diagonal)
        
        // Check if working copy was created successfully
        if (A.data == nullptr)
        {
            std::cerr << "[Error] eigendecompose_jacobi: failed to create working copy\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        result.eigenvectors = Mat::eye(n);
        
        // Check if eigenvectors matrix was created successfully
        if (result.eigenvectors.data == nullptr)
        {
            std::cerr << "[Error] eigendecompose_jacobi: failed to create eigenvectors matrix\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }

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
                
                // Check if eigenvalues matrix was created successfully
                if (result.eigenvalues.data == nullptr)
                {
                    std::cerr << "[Error] eigendecompose_jacobi: failed to create eigenvalues matrix\n";
                    result.status = TINY_ERR_MATH_NULL_POINTER;
                    return result;
                }
                
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
        
        // Check if eigenvalues matrix was created successfully
        if (result.eigenvalues.data == nullptr)
        {
            std::cerr << "[Error] eigendecompose_jacobi: failed to create eigenvalues matrix\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        for (int i = 0; i < n; ++i)
        {
            result.eigenvalues(i, 0) = A(i, i);
        }

        result.iterations = max_iter;
        result.status = TINY_ERR_NOT_FINISHED;
        std::cerr << "[Warning] eigendecompose_jacobi: did not converge within " << max_iter << " iterations\n";
        return result;
    }

    /**
     * @name Mat::eigendecompose_qr()
     * @brief Compute complete eigenvalue decomposition using QR algorithm for general matrices.
     * @note QR algorithm iteratively applies QR decomposition: A_k = Q_k * R_k, A_{k+1} = R_k * Q_k.
     *       Converges to Schur form (upper triangular) for real eigenvalues.
     * @note Supports non-symmetric matrices, but may have complex eigenvalues (only real part returned).
     * @note Time complexity: O(n³ * iterations) - typically requires O(n) iterations
     * @note Best for: General matrices, when all eigenvalues are real
     * 
     * @param max_iter Maximum number of QR iterations (must be > 0)
     * @param tolerance Convergence tolerance (must be >= 0). Convergence when subdiagonal < tolerance
     * @return EigenDecomposition containing eigenvalues, eigenvectors, and status
     */
    Mat::EigenDecomposition Mat::eigendecompose_qr(int max_iter, float tolerance) const
    {
        EigenDecomposition result;

        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] eigendecompose_qr: matrix data pointer is null\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] eigendecompose_qr: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        // Validation: must be square matrix
        if (this->row != this->col)
        {
            std::cerr << "[Error] eigendecompose_qr: requires square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        
        // Validate parameters
        if (max_iter <= 0)
        {
            std::cerr << "[Error] eigendecompose_qr: max_iter must be > 0 (got " << max_iter << ")\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        
        if (tolerance < 0.0f)
        {
            std::cerr << "[Error] eigendecompose_qr: tolerance must be >= 0 (got " << tolerance << ")\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        int n = this->row;
        
        // Handle empty matrix
        if (n == 0)
        {
            result.eigenvalues = Mat(0, 1);
            result.eigenvectors = Mat(0, 0);
            result.iterations = 0;
            result.status = TINY_OK;
            return result;
        }

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
     * @note Convenient interface for edge computing: automatically selects the best method.
     *       - Symmetric matrices: Uses Jacobi method (more efficient and stable)
     *       - General matrices: Uses QR algorithm
     * @note Time complexity: Depends on selected method
     *       - Jacobi: O(n³ * iterations) for symmetric matrices
     *       - QR: O(n³ * iterations) for general matrices
     * 
     * @param tolerance Convergence tolerance (must be >= 0)
     * @return EigenDecomposition containing eigenvalues, eigenvectors, and status
     */
    Mat::EigenDecomposition Mat::eigendecompose(float tolerance) const
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            EigenDecomposition result;
            std::cerr << "[Error] eigendecompose: matrix data pointer is null\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        // Validate tolerance
        if (tolerance < 0.0f)
        {
            EigenDecomposition result;
            std::cerr << "[Error] eigendecompose: tolerance must be >= 0 (got " << tolerance << ")\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        
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