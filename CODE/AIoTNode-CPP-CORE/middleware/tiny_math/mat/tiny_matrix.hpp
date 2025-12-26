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
