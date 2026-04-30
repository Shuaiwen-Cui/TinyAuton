/**
 * @file tiny_ica.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_ica | code | source
 * @version 1.0
 * @date 2025-04-30
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_ica.hpp"

// Standard Libraries
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdlib>

/* LIBRARY CONTENTS */
#ifdef __cplusplus

namespace tiny
{
    // ============================================================================
    // ICADecomposition Structure
    // ============================================================================
    ICADecomposition::ICADecomposition()
        : iterations(0), status(TINY_OK)
    {
    }

    // ============================================================================
    // ICA Class Implementation
    // ============================================================================
    ICA::ICA(ICANonlinearity nonlinearity, int max_iter, float tolerance, float alpha)
        : nonlinearity_(nonlinearity), max_iter_(max_iter), tolerance_(tolerance), alpha_(alpha)
    {
    }

    ICA::~ICA()
    {
    }

    void ICA::set_max_iterations(int max_iter)
    {
        max_iter_ = max_iter;
    }

    void ICA::set_tolerance(float tolerance)
    {
        tolerance_ = tolerance;
    }

    void ICA::set_alpha(float alpha)
    {
        alpha_ = alpha;
    }

    void ICA::set_nonlinearity(ICANonlinearity nonlinearity)
    {
        nonlinearity_ = nonlinearity;
    }

    ICADecomposition ICA::decompose(const Mat &mixed_signals, int num_sources)
    {
        ICADecomposition result;

        // Validate input
        if (mixed_signals.data == nullptr || mixed_signals.row <= 0 || mixed_signals.col <= 0)
        {
            result.status = TINY_ERR_DSP_NULL_POINTER;
            return result;
        }

        // Auto-detect number of sources if not specified
        int n_sensors = mixed_signals.row;
        int n_samples = mixed_signals.col;
        int n_sources = (num_sources > 0) ? num_sources : n_sensors;

        if (n_sources > n_sensors)
        {
            result.status = TINY_ERR_DSP_INVALID_PARAM;
            return result;
        }

        if (n_samples < n_sensors)
        {
            result.status = TINY_ERR_DSP_INVALID_LENGTH;
            return result;
        }

        // Perform FastICA
        result = fastica(mixed_signals, n_sources);

        return result;
    }

    Mat ICA::reconstruct(const Mat &sources, const Mat &mixing_matrix)
    {
        if (sources.data == nullptr || mixing_matrix.data == nullptr)
        {
            return Mat();
        }

        // Reconstruct: X = A * S
        return mixing_matrix * sources;
    }

    Mat ICA::apply(const ICADecomposition &decomposition, const Mat &new_mixed_signals)
    {
        if (decomposition.unmixing_matrix.data == nullptr || new_mixed_signals.data == nullptr)
        {
            return Mat();
        }

        // Center the new data using the same mean
        Mat centered = new_mixed_signals;
        for (int i = 0; i < centered.row; i++)
        {
            for (int j = 0; j < centered.col; j++)
            {
                centered(i, j) -= decomposition.mean(i, 0);
            }
        }

        // Apply unmixing matrix: S = W * X
        return decomposition.unmixing_matrix * centered;
    }

    // ============================================================================
    // Private Methods
    // ============================================================================
    Mat ICA::center_data(const Mat &data, Mat &mean)
    {
        int n_sensors = data.row;
        int n_samples = data.col;

        // Compute mean for each sensor using matrix operations
        // Mean = (1/N) * sum of each row
        mean = Mat(n_sensors, 1);
        Mat ones_col = Mat::ones(n_samples, 1);
        
        for (int i = 0; i < n_sensors; i++)
        {
            Mat row = data.view_roi(i, 0, 1, n_samples).copy_roi(0, 0, 1, n_samples);
            // Compute mean: sum of row elements / n_samples
            float sum = 0.0f;
            for (int j = 0; j < n_samples; j++)
            {
                sum += row(0, j);
            }
            mean(i, 0) = sum / n_samples;
        }

        // Subtract mean using matrix operations
        // Create a copy to avoid modifying original data
        Mat centered = data;
        Mat mean_expanded = mean * Mat::ones(1, n_samples); // Broadcast mean to all columns: (n_sensors x 1) * (1 x n_samples) = (n_sensors x n_samples)
        centered -= mean_expanded;

        return centered;
    }

    Mat ICA::whiten_data(const Mat &data, Mat &whitening_matrix)
    {
        int n_sensors = data.row;
        int n_samples = data.col;

        // Compute covariance matrix: C = (1/N) * X * X^T
        // Create a non-const copy for transpose() call
        Mat data_copy = data;
        Mat covariance = (data * data_copy.transpose()) * (1.0f / n_samples);

        // Eigenvalue decomposition of covariance matrix
        Mat::EigenDecomposition eig = covariance.eigendecompose(1e-6f, 100);

        if (eig.status != TINY_OK)
        {
            return Mat();
        }

        // Extract eigenvalues and eigenvectors
        Mat eigenvalues = eig.eigenvalues;
        Mat eigenvectors = eig.eigenvectors;

        // Compute whitening matrix: W = D^(-1/2) * E^T
        // where D is diagonal matrix of eigenvalues, E is eigenvector matrix
        Mat D_inv_sqrt = Mat::eye(n_sensors);
        for (int i = 0; i < n_sensors; i++)
        {
            float lambda = eigenvalues(i, i);
            if (lambda > 1e-10f) // Avoid division by zero
            {
                D_inv_sqrt(i, i) = 1.0f / sqrtf(lambda);
            }
            else
            {
                D_inv_sqrt(i, i) = 0.0f;
            }
        }

        whitening_matrix = D_inv_sqrt * eigenvectors.transpose();

        // Whiten the data: X_white = W * X
        return whitening_matrix * data;
    }

    ICADecomposition ICA::fastica(const Mat &mixed_signals, int num_sources)
    {
        ICADecomposition result;

        int n_sensors = mixed_signals.row;
        int n_samples = mixed_signals.col;

        // Step 1: Center the data
        Mat centered = center_data(mixed_signals, result.mean);
        if (centered.data == nullptr)
        {
            result.status = TINY_ERR_DSP_MEMORY_ALLOC;
            return result;
        }

        // Step 2: Whiten the data
        Mat whitened = whiten_data(centered, result.whitening_matrix);
        if (whitened.data == nullptr || result.whitening_matrix.data == nullptr)
        {
            result.status = TINY_ERR_DSP_MEMORY_ALLOC;
            return result;
        }

        // Step 3: FastICA algorithm
        // Initialize unmixing matrix
        result.unmixing_matrix = Mat(num_sources, n_sensors);
        Mat W = result.unmixing_matrix;

        // Initialize with random vectors (better initialization for FastICA)
        // Use multiple random initializations and pick the best one for first component
        for (int i = 0; i < num_sources; i++)
        {
            // Generate random unit vector
            float norm = 0.0f;
            for (int j = 0; j < n_sensors; j++)
            {
                W(i, j) = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                norm += W(i, j) * W(i, j);
            }
            norm = sqrtf(norm);
            if (norm > 1e-10f)
            {
                for (int j = 0; j < n_sensors; j++)
                {
                    W(i, j) /= norm;
                }
            }
            else
            {
                // Fallback: use unit vector in direction i
                for (int j = 0; j < n_sensors; j++)
                {
                    W(i, j) = (j == i % n_sensors) ? 1.0f : 0.0f;
                }
            }
        }
        
        // For first component, try a few random initializations and pick the one with highest variance
        if (num_sources > 0)
        {
            float best_var = 0.0f;
            Mat best_w(1, n_sensors);
            
            // Try 5 different random initializations
            for (int trial = 0; trial < 5; trial++)
            {
                Mat test_w(1, n_sensors);
                float norm = 0.0f;
                for (int j = 0; j < n_sensors; j++)
                {
                    test_w(0, j) = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                    norm += test_w(0, j) * test_w(0, j);
                }
                norm = sqrtf(norm);
                if (norm > 1e-10f)
                {
                    for (int j = 0; j < n_sensors; j++)
                    {
                        test_w(0, j) /= norm;
                    }
                }
                
                // Compute variance of w^T * X
                Mat test_wTx = test_w * whitened;
                float test_mean = 0.0f, test_var = 0.0f;
                for (int i = 0; i < n_samples; i++)
                {
                    test_mean += test_wTx(0, i);
                }
                test_mean /= n_samples;
                for (int i = 0; i < n_samples; i++)
                {
                    float diff = test_wTx(0, i) - test_mean;
                    test_var += diff * diff;
                }
                test_var /= n_samples;
                
                if (test_var > best_var)
                {
                    best_var = test_var;
                    for (int j = 0; j < n_sensors; j++)
                    {
                        best_w(0, j) = test_w(0, j);
                    }
                }
            }
            
            // Use the best initialization
            if (best_var > 1e-10f)
            {
                for (int j = 0; j < n_sensors; j++)
                {
                    W(0, j) = best_w(0, j);
                }
            }
        }

        // Iterate for each source
        for (int comp = 0; comp < num_sources; comp++)
        {
            Mat w = W.view_roi(comp, 0, 1, n_sensors).copy_roi(0, 0, 1, n_sensors);

            // Orthogonalize against previous components
            if (comp > 0)
            {
                Mat W_prev = W.view_roi(0, 0, comp, n_sensors);
                orthogonalize(w, W_prev);
            }

            normalize(w);
            
            // Track reinitialization count to prevent infinite loops
            int reinit_count = 0;
            const int max_reinit = 3; // Maximum reinitializations per component

            // FastICA iteration
            for (int iter = 0; iter < max_iter_; iter++)
            {
                Mat w_old = w;

                // Compute: w_new = E{X * g(w^T * X)} - E{g'(w^T * X)} * w
                // where g is the nonlinearity function

                // Compute w^T * X for all samples
                Mat wTx = w * whitened; // (1 x n_sensors) * (n_sensors x n_samples) = (1 x n_samples)
                
                // Check if wTx has sufficient variance (indicates w is pointing in a meaningful direction)
                float wTx_mean = 0.0f;
                float wTx_var = 0.0f;
                for (int i = 0; i < n_samples; i++)
                {
                    wTx_mean += wTx(0, i);
                }
                wTx_mean /= n_samples;
                for (int i = 0; i < n_samples; i++)
                {
                    float diff = wTx(0, i) - wTx_mean;
                    wTx_var += diff * diff;
                }
                wTx_var /= n_samples;
                
                // If variance is too small, w is not pointing in a good direction, reinitialize
                // Note: For whitened data, variance should be around 1.0, so use a more reasonable threshold
                // Only reinitialize if variance is extremely small (indicating numerical issues)
                if (wTx_var < 1e-10f && iter > 2)
                {
                    // Reinitialize w randomly only if we've tried a few iterations
                    for (int j = 0; j < n_sensors; j++)
                    {
                        w(0, j) = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                    }
                    if (comp > 0)
                    {
                        Mat W_prev = W.view_roi(0, 0, comp, n_sensors);
                        orthogonalize(w, W_prev);
                    }
                    normalize(w);
                    continue; // Skip this iteration and try again
                }

                // Apply nonlinearity
                Mat g_wTx = apply_nonlinearity(wTx);
                Mat g_prime_wTx = apply_nonlinearity_derivative(wTx);

                // Compute expectations
                Mat Xg = whitened * g_wTx.transpose(); // (n_sensors x n_samples) * (n_samples x 1) = (n_sensors x 1)
                Xg *= (1.0f / n_samples);

                float g_prime_mean = 0.0f;
                for (int i = 0; i < n_samples; i++)
                {
                    g_prime_mean += g_prime_wTx(0, i);
                }
                g_prime_mean /= n_samples;

                // Update: w_new = Xg - g_prime_mean * w
                w = Xg.transpose();
                w -= g_prime_mean * w_old;

                // Orthogonalize against previous components
                if (comp > 0)
                {
                    Mat W_prev = W.view_roi(0, 0, comp, n_sensors);
                    orthogonalize(w, W_prev);
                }
                
                // Check if vector became too small (before normalization)
                float w_norm = w.norm();
                if (w_norm < 1e-6f)
                {
                    // Vector became nearly zero, reinitialize randomly
                    for (int j = 0; j < n_sensors; j++)
                    {
                        w(0, j) = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                    }
                    // Re-orthogonalize if needed
                    if (comp > 0)
                    {
                        Mat W_prev = W.view_roi(0, 0, comp, n_sensors);
                        orthogonalize(w, W_prev);
                        w_norm = w.norm();
                        // If still too small after orthogonalization, use unit vector
                        if (w_norm < 1e-6f)
                        {
                            for (int j = 0; j < n_sensors; j++)
                            {
                                w(0, j) = (j == comp % n_sensors) ? 1.0f : 0.0f;
                            }
                            if (comp > 0)
                            {
                                Mat W_prev2 = W.view_roi(0, 0, comp, n_sensors);
                                orthogonalize(w, W_prev2);
                            }
                        }
                    }
                }

                normalize(w);

                // Check convergence using Mat's dotprod method
                float dot_product = w.dotprod(w, w_old);
                float diff = 1.0f - fabsf(dot_product);
                
                // Require at least a few iterations before accepting convergence
                // This prevents premature convergence to wrong solutions
                if (diff < tolerance_ && iter >= 2)
                {
                    result.iterations += iter + 1;
                    break;
                }

                if (iter == max_iter_ - 1)
                {
                    result.iterations += max_iter_;
                }
            }

            // Verify w is valid before storing
            // Check if w produces meaningful separation by testing wTx variance
            Mat test_wTx = w * whitened;
            bool w_is_valid = true;
            
            // Check for NaN or Inf in w
            for (int j = 0; j < n_sensors; j++)
            {
                float val = w(0, j);
                if (!(val == val) || (val != 0.0f && (val * 2.0f == val))) // NaN or Inf check
                {
                    w_is_valid = false;
                    break;
                }
            }
            
            // Check wTx variance - for whitened data, variance should be around 1.0
            // If variance is extremely small, w is pointing in a bad direction
            if (w_is_valid)
            {
                float test_mean = 0.0f, test_var = 0.0f;
                for (int i = 0; i < n_samples; i++)
                {
                    test_mean += test_wTx(0, i);
                }
                test_mean /= n_samples;
                for (int i = 0; i < n_samples; i++)
                {
                    float diff = test_wTx(0, i) - test_mean;
                    test_var += diff * diff;
                }
                test_var /= n_samples;
                
                // If variance is extremely small, w is not useful
                // For whitened data, variance should be around 1.0
                // Use a very lenient threshold to only catch truly bad directions (near zero)
                // Some signals may naturally have lower variance after whitening
                if (test_var < 1e-6f) // Only reject if variance is truly negligible
                {
                    w_is_valid = false;
                }
            }
            
            // If w is invalid, reinitialize and retry FastICA iteration
            if (!w_is_valid)
            {
                // Reinitialize randomly
                for (int j = 0; j < n_sensors; j++)
                {
                    w(0, j) = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                }
                if (comp > 0)
                {
                    Mat W_prev = W.view_roi(0, 0, comp, n_sensors);
                    orthogonalize(w, W_prev);
                }
                normalize(w);
                
                // Retry FastICA iteration with new initialization (limited iterations)
                for (int retry_iter = 0; retry_iter < 10 && retry_iter < max_iter_; retry_iter++)
                {
                    Mat w_old_retry = w;
                    Mat wTx_retry = w * whitened;
                    Mat g_wTx_retry = apply_nonlinearity(wTx_retry);
                    Mat Xg_retry = whitened * g_wTx_retry.transpose();
                    Xg_retry *= (1.0f / n_samples);
                    
                    float g_prime_mean_retry = 0.0f;
                    for (int i = 0; i < n_samples; i++)
                    {
                        g_prime_mean_retry += apply_nonlinearity_derivative(wTx_retry(0, i));
                    }
                    g_prime_mean_retry /= n_samples;
                    
                    w = Xg_retry.transpose();
                    w -= g_prime_mean_retry * w_old_retry;
                    
                    if (comp > 0)
                    {
                        Mat W_prev_retry = W.view_roi(0, 0, comp, n_sensors);
                        orthogonalize(w, W_prev_retry);
                    }
                    normalize(w);
                    
                    // Check if this is better
                    Mat test_wTx_retry = w * whitened;
                    float test_mean_retry = 0.0f, test_var_retry = 0.0f;
                    for (int i = 0; i < n_samples; i++)
                    {
                        test_mean_retry += test_wTx_retry(0, i);
                    }
                    test_mean_retry /= n_samples;
                    for (int i = 0; i < n_samples; i++)
                    {
                        float diff = test_wTx_retry(0, i) - test_mean_retry;
                        test_var_retry += diff * diff;
                    }
                    test_var_retry /= n_samples;
                    
                    if (test_var_retry >= 1e-6f) break; // Found a reasonable direction
                }
            }
            
            // Store the converged vector
            for (int j = 0; j < n_sensors; j++)
            {
                W(comp, j) = w(0, j);
            }
        }

        // Compute sources: S = W * X_white
        result.sources = W * whitened;

        // Compute mixing matrix: A
        // In ICA: X = A * S, where X is mixed signals, A is mixing matrix, S is sources
        // After ICA: S = W * X_white, where W is unmixing matrix, X_white is whitened data
        // X_white = V * X_centered, where V is whitening matrix
        // So: S = W * V * X_centered, therefore: X_centered = (W * V)^(-1) * S
        // But we need: X = A * S, and X = X_centered + mean
        // So: A = (W * V)^(-1) for the centered case
        
        Mat WV = W * result.whitening_matrix;  // (num_sources x n_sensors) * (n_sensors x n_sensors) = (num_sources x n_sensors)
        
        // Compute mixing matrix: A = (W * V)^(-1)
        // W is (num_sources x n_sensors), V is (n_sensors x n_sensors)
        // WV = W * V is (num_sources x n_sensors)
        // We need A = WV^(-1) which should be (n_sensors x num_sources)
        
        // Always use pseudo-inverse for robustness (handles both square and non-square cases)
        Mat::SVDDecomposition svd = WV.svd_decompose(100, 1e-6f);
        if (svd.status == TINY_OK)
        {
            // Pseudo-inverse of (num_sources x n_sensors) gives (n_sensors x num_sources)
            result.mixing_matrix = Mat::pseudo_inverse(svd, 1e-6f);
        }
        else
        {
            // If SVD fails, try direct inverse only for square matrices
            if (num_sources == n_sensors)
            {
                Mat WV_inv = WV.inverse_gje();
                if (WV_inv.data != nullptr && WV_inv.row == num_sources && WV_inv.col == num_sources)
                {
                    result.mixing_matrix = WV_inv;
                }
                else
                {
                    // Last resort: create identity-based approximation
                    result.mixing_matrix = Mat::eye(n_sensors);
                }
            }
            else
            {
                // Non-square and SVD failed, create zero matrix
                result.mixing_matrix = Mat(n_sensors, num_sources);
            }
        }

        result.unmixing_matrix = W;
        result.status = TINY_OK;

        return result;
    }

    float ICA::apply_nonlinearity(float x) const
    {
        switch (nonlinearity_)
        {
        case ICANonlinearity::TANH:
            return tanhf(alpha_ * x);
        case ICANonlinearity::CUBE:
            return x * x * x;
        case ICANonlinearity::GAUSS:
            return x * expf(-0.5f * x * x);
        case ICANonlinearity::SKEW:
            return x * x;
        default:
            return tanhf(alpha_ * x);
        }
    }

    Mat ICA::apply_nonlinearity(const Mat &x) const
    {
        Mat result = x;
        for (int i = 0; i < result.row; i++)
        {
            for (int j = 0; j < result.col; j++)
            {
                result(i, j) = apply_nonlinearity(result(i, j));
            }
        }
        return result;
    }

    float ICA::apply_nonlinearity_derivative(float x) const
    {
        switch (nonlinearity_)
        {
        case ICANonlinearity::TANH:
        {
            float tanh_val = tanhf(alpha_ * x);
            return alpha_ * (1.0f - tanh_val * tanh_val);
        }
        case ICANonlinearity::CUBE:
            return 3.0f * x * x;
        case ICANonlinearity::GAUSS:
            return (1.0f - x * x) * expf(-0.5f * x * x);
        case ICANonlinearity::SKEW:
            return 2.0f * x;
        default:
        {
            float tanh_val = tanhf(alpha_ * x);
            return alpha_ * (1.0f - tanh_val * tanh_val);
        }
        }
    }

    Mat ICA::apply_nonlinearity_derivative(const Mat &x) const
    {
        Mat result = x;
        for (int i = 0; i < result.row; i++)
        {
            for (int j = 0; j < result.col; j++)
            {
                result(i, j) = apply_nonlinearity_derivative(result(i, j));
            }
        }
        return result;
    }

    void ICA::orthogonalize(Mat &w, const Mat &W_prev)
    {
        // Gram-Schmidt orthogonalization using Mat's dotprod method
        // w = w - sum((w^T * w_i) * w_i) for all previous w_i

        int n_prev = W_prev.row;
        int n_sensors = w.col;

        for (int i = 0; i < n_prev; i++)
        {
            // Get previous vector as a row matrix
            Mat w_i = W_prev.view_roi(i, 0, 1, n_sensors).copy_roi(0, 0, 1, n_sensors);
            
            // Compute dot product using Mat's dotprod method
            float dot = w.dotprod(w, w_i);

            // Subtract projection: w = w - dot * w_i
            w -= dot * w_i;
        }
    }

    void ICA::normalize(Mat &w)
    {
        // Normalize to unit length using Mat's norm() method
        float norm = w.norm();

        if (norm > 1e-10f)
        {
            w *= (1.0f / norm);
        }
    }

} // namespace tiny

#endif // __cplusplus

