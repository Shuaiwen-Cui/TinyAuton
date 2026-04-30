/**
 * @file tiny_dataset.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Lightweight dataset class for tiny_ai training.
 *        Wraps const float feature arrays and int label arrays.
 *        Supports shuffle, train/test split, and mini-batch iteration.
 * @version 1.0
 * @date 2026-04-30
 * @copyright Copyright (c) 2026
 */

#pragma once

#include "tiny_tensor.hpp"

#ifdef __cplusplus

namespace tiny
{

class Dataset
{
public:
    /**
     * @brief Construct from raw arrays (no copy — arrays must outlive Dataset).
     *
     * @param X          Feature matrix [n_samples × n_features], row-major
     * @param y          Integer labels  [n_samples]
     * @param n_samples  Total number of samples
     * @param n_features Number of input features per sample
     * @param n_classes  Number of output classes
     */
    Dataset(const float *X, const int *y,
            int n_samples, int n_features, int n_classes);

    ~Dataset();
    Dataset(const Dataset &other);              ///< Deep copy (indices array is duplicated)
    Dataset &operator=(const Dataset &other);
    Dataset(Dataset &&other) noexcept;          ///< Move (takes ownership of indices array)
    Dataset &operator=(Dataset &&other) noexcept;

    // =========================================================================
    // Shuffle
    // =========================================================================

    /**
     * @brief Shuffle the sample order.
     * @param seed  Random seed (default 0 uses fixed sequence)
     */
    void shuffle(uint32_t seed = 0);

    // =========================================================================
    // Mini-batch iteration
    // =========================================================================

    /// Reset iteration cursor to start
    void reset();

    /**
     * @brief Fill X_batch and y_batch with the next mini-batch.
     *        Allocates new Tensors each call (small overhead; batch_size is small).
     *
     * @param X_batch    Output: [batch_size, n_features]
     * @param y_batch    Output: int array of length batch_size (caller-allocated)
     * @param batch_size Desired batch size; may be smaller at end of epoch
     * @return           Actual number of samples filled; 0 at epoch end
     */
    int next_batch(Tensor &X_batch, int *y_batch, int batch_size);

    // =========================================================================
    // Train/test split
    // =========================================================================

    /**
     * @brief Split into train and test datasets.
     *        Both returned datasets reference the same underlying arrays;
     *        indices are partitioned accordingly.
     *
     * @param test_ratio  Fraction of samples to reserve for testing (e.g. 0.2)
     * @param train_out   Output train Dataset
     * @param test_out    Output test  Dataset
     * @param seed        Shuffle seed before splitting
     */
    void split(float test_ratio, Dataset &train_out, Dataset &test_out,
               uint32_t seed = 42) const;

    // =========================================================================
    // Accessors
    // =========================================================================

    int size()       const { return n_samples_; }
    int features()   const { return n_features_; }
    int classes()    const { return n_classes_; }
    bool at_end()    const { return cursor_ >= n_samples_; }

    /// Build a full-dataset Tensor (copies data) — used for evaluation
    Tensor to_tensor() const;
    const int *labels() const { return y_; }

private:
    const float *X_;
    const int   *y_;
    int          n_samples_;
    int          n_features_;
    int          n_classes_;

    int *indices_;   ///< Shuffled index array (owned)
    int  cursor_;    ///< Current position in iteration

    // Used by split() to create a sub-view with a subset of indices
    Dataset(const float *X, const int *y,
            int n_samples, int n_features, int n_classes,
            const int *given_indices);
};

} // namespace tiny

#endif // __cplusplus
