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
