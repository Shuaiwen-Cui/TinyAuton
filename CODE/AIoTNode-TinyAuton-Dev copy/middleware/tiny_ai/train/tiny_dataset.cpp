/**
 * @file tiny_dataset.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Dataset implementation.
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#include "tiny_dataset.hpp"
#include <cstring>
#include <cstdlib>

#ifdef __cplusplus

namespace tiny
{

// ============================================================================
// Constructors / Destructor
// ============================================================================

Dataset::Dataset(const float *X, const int *y,
                 int n_samples, int n_features, int n_classes)
    : X_(X), y_(y),
      n_samples_(n_samples), n_features_(n_features), n_classes_(n_classes),
      cursor_(0)
{
    indices_ = (int *)TINY_AI_MALLOC((size_t)n_samples * sizeof(int));
    for (int i = 0; i < n_samples; i++) indices_[i] = i;
}

// Private constructor used by split() — copies given index array
Dataset::Dataset(const float *X, const int *y,
                 int n_samples, int n_features, int n_classes,
                 const int *given_indices)
    : X_(X), y_(y),
      n_samples_(n_samples), n_features_(n_features), n_classes_(n_classes),
      cursor_(0)
{
    indices_ = (int *)TINY_AI_MALLOC((size_t)n_samples * sizeof(int));
    memcpy(indices_, given_indices, (size_t)n_samples * sizeof(int));
}

Dataset::~Dataset()
{
    if (indices_) TINY_AI_FREE(indices_);
}

// ============================================================================
// Copy / Move
// ============================================================================

Dataset::Dataset(const Dataset &other)
    : X_(other.X_), y_(other.y_),
      n_samples_(other.n_samples_), n_features_(other.n_features_),
      n_classes_(other.n_classes_), cursor_(other.cursor_)
{
    indices_ = (int *)TINY_AI_MALLOC((size_t)n_samples_ * sizeof(int));
    if (indices_ && other.indices_)
        memcpy(indices_, other.indices_, (size_t)n_samples_ * sizeof(int));
}

Dataset &Dataset::operator=(const Dataset &other)
{
    if (this == &other) return *this;
    if (indices_) TINY_AI_FREE(indices_);
    X_          = other.X_;
    y_          = other.y_;
    n_samples_  = other.n_samples_;
    n_features_ = other.n_features_;
    n_classes_  = other.n_classes_;
    cursor_     = other.cursor_;
    indices_    = (int *)TINY_AI_MALLOC((size_t)n_samples_ * sizeof(int));
    if (indices_ && other.indices_)
        memcpy(indices_, other.indices_, (size_t)n_samples_ * sizeof(int));
    return *this;
}

Dataset::Dataset(Dataset &&other) noexcept
    : X_(other.X_), y_(other.y_),
      n_samples_(other.n_samples_), n_features_(other.n_features_),
      n_classes_(other.n_classes_), indices_(other.indices_),
      cursor_(other.cursor_)
{
    other.indices_ = nullptr;
}

Dataset &Dataset::operator=(Dataset &&other) noexcept
{
    if (this != &other)
    {
        if (indices_) TINY_AI_FREE(indices_);
        X_          = other.X_;
        y_          = other.y_;
        n_samples_  = other.n_samples_;
        n_features_ = other.n_features_;
        n_classes_  = other.n_classes_;
        cursor_     = other.cursor_;
        indices_    = other.indices_;
        other.indices_ = nullptr;
    }
    return *this;
}

void Dataset::shuffle(uint32_t seed)
{
    uint32_t s = seed ? seed : 1234567891u;
    for (int i = n_samples_ - 1; i > 0; i--)
    {
        // LCG: next = (a * s + c) mod 2^32
        s = s * 1664525u + 1013904223u;
        int j = (int)(s % (uint32_t)(i + 1));
        int tmp = indices_[i]; indices_[i] = indices_[j]; indices_[j] = tmp;
    }
    cursor_ = 0;
}

// ============================================================================
// Mini-batch iteration
// ============================================================================

void Dataset::reset() { cursor_ = 0; }

int Dataset::next_batch(Tensor &X_batch, int *y_batch, int batch_size)
{
    if (cursor_ >= n_samples_) return 0;

    int actual = batch_size;
    if (cursor_ + actual > n_samples_) actual = n_samples_ - cursor_;

    // Reallocate X_batch if shape doesn't match
    if (X_batch.size != actual * n_features_)
        X_batch = Tensor(actual, n_features_);

    for (int i = 0; i < actual; i++)
    {
        int idx = indices_[cursor_ + i];
        const float *src = X_ + (size_t)idx * n_features_;
        float       *dst = X_batch.data + (size_t)i * n_features_;
        memcpy(dst, src, (size_t)n_features_ * sizeof(float));
        y_batch[i] = y_[idx];
    }
    cursor_ += actual;
    return actual;
}

// ============================================================================
// Train/test split
// ============================================================================

void Dataset::split(float test_ratio, Dataset &train_out, Dataset &test_out,
                    uint32_t seed) const
{
    int n_test  = (int)(n_samples_ * test_ratio + 0.5f);
    if (n_test < 1)        n_test = 1;
    if (n_test >= n_samples_) n_test = n_samples_ - 1;
    int n_train = n_samples_ - n_test;

    // Make a shuffled copy of indices
    int *shuffled = (int *)TINY_AI_MALLOC((size_t)n_samples_ * sizeof(int));
    memcpy(shuffled, indices_, (size_t)n_samples_ * sizeof(int));

    // Fisher-Yates shuffle
    uint32_t s = seed ? seed : 1234567891u;
    for (int i = n_samples_ - 1; i > 0; i--)
    {
        s = s * 1664525u + 1013904223u;
        int j = (int)(s % (uint32_t)(i + 1));
        int tmp = shuffled[i]; shuffled[i] = shuffled[j]; shuffled[j] = tmp;
    }

    train_out = Dataset(X_, y_, n_train, n_features_, n_classes_, shuffled);
    test_out  = Dataset(X_, y_, n_test,  n_features_, n_classes_, shuffled + n_train);

    TINY_AI_FREE(shuffled);
}

// ============================================================================
// to_tensor
// ============================================================================

Tensor Dataset::to_tensor() const
{
    Tensor t(n_samples_, n_features_);
    for (int i = 0; i < n_samples_; i++)
    {
        int idx = indices_[i];
        memcpy(t.data + (size_t)i * n_features_,
               X_ + (size_t)idx * n_features_,
               (size_t)n_features_ * sizeof(float));
    }
    return t;
}

} // namespace tiny

#endif // __cplusplus
