# 代码

## tiny_ica.h

```cpp
/**
 * @file tiny_ica.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_ica | code | header
 * @version 1.0
 * @date 2025-04-30
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
// tiny_dsp configuration file
#include "tiny_dsp_config.h"

// tiny_math for matrix operations
#include "tiny_matrix.hpp"

#ifdef __cplusplus

#include <cstdint>
#include <cmath>

namespace tiny
{
    /**
     * @brief Nonlinearity function types for FastICA
     */
    enum class ICANonlinearity
    {
        TANH = 0,      ///< tanh function (default, good for super-Gaussian sources)
        CUBE,          ///< x^3 function (good for sub-Gaussian sources)
        GAUSS,         ///< Gaussian function (good for symmetric sources)
        SKEW            ///< Skew function (good for skewed sources)
    };

    /**
     * @brief Structure to hold ICA decomposition results
     * @note X = A * S, where X is mixed signals, A is mixing matrix, S is source signals
     *       After ICA: S = W * X, where W is unmixing matrix
     */
    struct ICADecomposition
    {
        Mat unmixing_matrix;    ///< Unmixing matrix W (sources x sensors)
        Mat mixing_matrix;      ///< Estimated mixing matrix A = W^(-1) (sensors x sources)
        Mat sources;            ///< Estimated source signals (sources x samples)
        Mat whitening_matrix;   ///< Whitening matrix used in preprocessing
        Mat mean;               ///< Mean vector of input signals (for centering)
        int iterations;         ///< Number of iterations performed
        tiny_error_t status;    ///< Computation status

        ICADecomposition();
    };

    /**
     * @brief ICA class for Independent Component Analysis
     * @note Implements FastICA algorithm for blind source separation
     */
    class ICA
    {
    public:
        /**
         * @brief Default constructor
         * @param nonlinearity Nonlinearity function for FastICA (default: TANH)
         * @param max_iter Maximum number of iterations (default: 1000)
         * @param tolerance Convergence tolerance (default: 1e-6)
         * @param alpha Step size for FastICA (default: 1.0)
         */
        ICA(ICANonlinearity nonlinearity = ICANonlinearity::TANH,
            int max_iter = 1000,
            float tolerance = 1e-6f,
            float alpha = 1.0f);

        /**
         * @brief Destructor
         */
        ~ICA();

        /**
         * @brief Perform ICA decomposition on mixed signals
         * @param mixed_signals Input matrix (sensors x samples)
         * @param num_sources Number of sources to extract (0 = auto-detect, default: 0)
         * @return ICADecomposition structure containing results
         * @note Input matrix: each row is a sensor signal, each column is a time sample
         *       If num_sources is 0, it will be set to the number of sensors (rows)
         */
        ICADecomposition decompose(const Mat &mixed_signals, int num_sources = 0);

        /**
         * @brief Reconstruct mixed signals from sources using mixing matrix
         * @param sources Source signals matrix (sources x samples)
         * @param mixing_matrix Mixing matrix (sensors x sources)
         * @return Reconstructed mixed signals (sensors x samples)
         */
        static Mat reconstruct(const Mat &sources, const Mat &mixing_matrix);

        /**
         * @brief Apply learned unmixing matrix to new data
         * @param decomposition ICA decomposition result from previous decompose() call
         * @param new_mixed_signals New mixed signals (sensors x samples)
         * @return Separated sources (sources x samples)
         */
        static Mat apply(const ICADecomposition &decomposition, const Mat &new_mixed_signals);

        /**
         * @brief Set algorithm parameters
         */
        void set_max_iterations(int max_iter);
        void set_tolerance(float tolerance);
        void set_alpha(float alpha);
        void set_nonlinearity(ICANonlinearity nonlinearity);

    private:
        ICANonlinearity nonlinearity_;
        int max_iter_;
        float tolerance_;
        float alpha_;

        /**
         * @brief Center the data (remove mean)
         */
        static Mat center_data(const Mat &data, Mat &mean);

        /**
         * @brief Whiten the data (decorrelate and normalize variance)
         */
        static Mat whiten_data(const Mat &data, Mat &whitening_matrix);

        /**
         * @brief FastICA algorithm implementation
         */
        ICADecomposition fastica(const Mat &mixed_signals, int num_sources);

        /**
         * @brief Apply nonlinearity function
         */
        float apply_nonlinearity(float x) const;
        Mat apply_nonlinearity(const Mat &x) const;

        /**
         * @brief Compute derivative of nonlinearity function
         */
        float apply_nonlinearity_derivative(float x) const;
        Mat apply_nonlinearity_derivative(const Mat &x) const;

        /**
         * @brief Orthogonalize vector against previous vectors (Gram-Schmidt)
         */
        static void orthogonalize(Mat &w, const Mat &W_prev);

        /**
         * @brief Normalize vector to unit length
         */
        static void normalize(Mat &w);
    };

} // namespace tiny

#endif // __cplusplus

```


## tiny_ica.c

```cpp
/**
 * @file tiny_ica.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_ica | code | header
 * @version 1.0
 * @date 2025-04-30
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
// tiny_dsp configuration file
#include "tiny_dsp_config.h"

// tiny_math for matrix operations
#include "tiny_matrix.hpp"

#ifdef __cplusplus

#include <cstdint>
#include <cmath>

namespace tiny
{
    /**
     * @brief Nonlinearity function types for FastICA
     */
    enum class ICANonlinearity
    {
        TANH = 0,      ///< tanh function (default, good for super-Gaussian sources)
        CUBE,          ///< x^3 function (good for sub-Gaussian sources)
        GAUSS,         ///< Gaussian function (good for symmetric sources)
        SKEW            ///< Skew function (good for skewed sources)
    };

    /**
     * @brief Structure to hold ICA decomposition results
     * @note X = A * S, where X is mixed signals, A is mixing matrix, S is source signals
     *       After ICA: S = W * X, where W is unmixing matrix
     */
    struct ICADecomposition
    {
        Mat unmixing_matrix;    ///< Unmixing matrix W (sources x sensors)
        Mat mixing_matrix;      ///< Estimated mixing matrix A = W^(-1) (sensors x sources)
        Mat sources;            ///< Estimated source signals (sources x samples)
        Mat whitening_matrix;   ///< Whitening matrix used in preprocessing
        Mat mean;               ///< Mean vector of input signals (for centering)
        int iterations;         ///< Number of iterations performed
        tiny_error_t status;    ///< Computation status

        ICADecomposition();
    };

    /**
     * @brief ICA class for Independent Component Analysis
     * @note Implements FastICA algorithm for blind source separation
     */
    class ICA
    {
    public:
        /**
         * @brief Default constructor
         * @param nonlinearity Nonlinearity function for FastICA (default: TANH)
         * @param max_iter Maximum number of iterations (default: 1000)
         * @param tolerance Convergence tolerance (default: 1e-6)
         * @param alpha Step size for FastICA (default: 1.0)
         */
        ICA(ICANonlinearity nonlinearity = ICANonlinearity::TANH,
            int max_iter = 1000,
            float tolerance = 1e-6f,
            float alpha = 1.0f);

        /**
         * @brief Destructor
         */
        ~ICA();

        /**
         * @brief Perform ICA decomposition on mixed signals
         * @param mixed_signals Input matrix (sensors x samples)
         * @param num_sources Number of sources to extract (0 = auto-detect, default: 0)
         * @return ICADecomposition structure containing results
         * @note Input matrix: each row is a sensor signal, each column is a time sample
         *       If num_sources is 0, it will be set to the number of sensors (rows)
         */
        ICADecomposition decompose(const Mat &mixed_signals, int num_sources = 0);

        /**
         * @brief Reconstruct mixed signals from sources using mixing matrix
         * @param sources Source signals matrix (sources x samples)
         * @param mixing_matrix Mixing matrix (sensors x sources)
         * @return Reconstructed mixed signals (sensors x samples)
         */
        static Mat reconstruct(const Mat &sources, const Mat &mixing_matrix);

        /**
         * @brief Apply learned unmixing matrix to new data
         * @param decomposition ICA decomposition result from previous decompose() call
         * @param new_mixed_signals New mixed signals (sensors x samples)
         * @return Separated sources (sources x samples)
         */
        static Mat apply(const ICADecomposition &decomposition, const Mat &new_mixed_signals);

        /**
         * @brief Set algorithm parameters
         */
        void set_max_iterations(int max_iter);
        void set_tolerance(float tolerance);
        void set_alpha(float alpha);
        void set_nonlinearity(ICANonlinearity nonlinearity);

    private:
        ICANonlinearity nonlinearity_;
        int max_iter_;
        float tolerance_;
        float alpha_;

        /**
         * @brief Center the data (remove mean)
         */
        static Mat center_data(const Mat &data, Mat &mean);

        /**
         * @brief Whiten the data (decorrelate and normalize variance)
         */
        static Mat whiten_data(const Mat &data, Mat &whitening_matrix);

        /**
         * @brief FastICA algorithm implementation
         */
        ICADecomposition fastica(const Mat &mixed_signals, int num_sources);

        /**
         * @brief Apply nonlinearity function
         */
        float apply_nonlinearity(float x) const;
        Mat apply_nonlinearity(const Mat &x) const;

        /**
         * @brief Compute derivative of nonlinearity function
         */
        float apply_nonlinearity_derivative(float x) const;
        Mat apply_nonlinearity_derivative(const Mat &x) const;

        /**
         * @brief Orthogonalize vector against previous vectors (Gram-Schmidt)
         */
        static void orthogonalize(Mat &w, const Mat &W_prev);

        /**
         * @brief Normalize vector to unit length
         */
        static void normalize(Mat &w);
    };

} // namespace tiny

#endif // __cplusplus


```