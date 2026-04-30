/**
 * @file tiny_tensor.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief N-dimensional float32 tensor for tiny_ai.
 *        Supports up to 4 dimensions.  For 2D tensors, provides a zero-copy
 *        view as tiny::Mat.  PSRAM-aware allocation on ESP32-S3.
 * @version 1.0
 * @date 2026-04-30
 * @copyright Copyright (c) 2026
 */

#pragma once

#include "tiny_ai_config.h"
#include "tiny_matrix.hpp"

#ifdef __cplusplus

#include <stdint.h>
#include <cstdio>

namespace tiny
{

/* ============================================================================
 * Tensor
 * ============================================================================ */
class Tensor
{
public:
    // -------------------------------------------------------------------------
    // Metadata
    // -------------------------------------------------------------------------
    int   ndim;        ///< Number of active dimensions (1–4)
    int   shape[4];    ///< Dimension sizes; unused dims are 1
    int   size;        ///< Total elements = product of all shape[] values
    float *data;       ///< Pointer to flat row-major float buffer
    bool  owns_data;   ///< true → destructor frees data; false → external buffer

    // =========================================================================
    // Constructors / Destructor
    // =========================================================================

    Tensor();                                              ///< Empty tensor (no allocation)
    explicit Tensor(int n0);                               ///< 1-D: [n0]
    Tensor(int n0, int n1);                                ///< 2-D: [n0, n1]
    Tensor(int n0, int n1, int n2);                        ///< 3-D: [n0, n1, n2]
    Tensor(int n0, int n1, int n2, int n3);                ///< 4-D: [n0, n1, n2, n3]

    Tensor(const Tensor &other);                           ///< Deep copy
    Tensor(Tensor &&other) noexcept;                       ///< Move (takes ownership)
    ~Tensor();

    Tensor &operator=(const Tensor &other);
    Tensor &operator=(Tensor &&other) noexcept;

    // =========================================================================
    // Named constructors
    // =========================================================================

    /// Zero-initialised 1-D tensor
    static Tensor zeros(int n0);
    /// Zero-initialised 2-D tensor
    static Tensor zeros(int n0, int n1);
    /// Zero-initialised 3-D tensor
    static Tensor zeros(int n0, int n1, int n2);
    /// Zero-initialised 4-D tensor
    static Tensor zeros(int n0, int n1, int n2, int n3);

    /// Zero-initialised tensor with the same shape as another
    static Tensor zeros_like(const Tensor &other);

    /// Wrap an external buffer — no copy, no ownership transfer.
    /// Caller must ensure the buffer outlives this Tensor.
    static Tensor from_data(float *data, int ndim, const int *shape);

    // =========================================================================
    // Element access (inline for MCU performance)
    // =========================================================================

    inline float &at(int i)
    {
        return data[i];
    }
    inline float &at(int i, int j)
    {
        return data[i * shape[1] + j];
    }
    inline float &at(int i, int j, int k)
    {
        return data[(i * shape[1] + j) * shape[2] + k];
    }
    inline float &at(int i, int j, int k, int l)
    {
        return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
    }
    inline const float &at(int i)                        const { return data[i]; }
    inline const float &at(int i, int j)                 const { return data[i * shape[1] + j]; }
    inline const float &at(int i, int j, int k)          const { return data[(i * shape[1] + j) * shape[2] + k]; }
    inline const float &at(int i, int j, int k, int l)   const { return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l]; }

    // =========================================================================
    // Convenience shape accessors
    // =========================================================================

    /// Batch dimension (first dim when ndim >= 3, else 1)
    inline int batch()    const { return ndim >= 3 ? shape[0] : 1; }
    /// Rows (second-to-last dim for ndim>=2, or shape[0] for 1-D)
    inline int rows()     const { return ndim >= 2 ? shape[ndim - 2] : shape[0]; }
    /// Cols (last dimension)
    inline int cols()     const { return shape[ndim - 1]; }
    /// Channels (shape[1] for ndim==4, else 1)
    inline int channels() const { return ndim == 4 ? shape[1] : 1; }

    // =========================================================================
    // In-place operations
    // =========================================================================

    void zero();                                   ///< Fill with 0
    void fill(float val);                          ///< Fill with constant
    void copy_from(const Tensor &src);             ///< Deep copy of data (shapes must match)
    Tensor clone() const;                          ///< Return a deep copy

    // =========================================================================
    // Shape operations
    // =========================================================================

    /// Reshape in-place (total size must be unchanged)
    tiny_error_t reshape(int ndim, const int *new_shape);

    /// Convenience reshapes
    tiny_error_t reshape_2d(int n0, int n1);
    tiny_error_t reshape_3d(int n0, int n1, int n2);

    // =========================================================================
    // Interop with tiny::Mat (2-D only, zero-copy view)
    // =========================================================================

    /// Returns a tiny::Mat wrapping this tensor's buffer (valid only for 2-D tensors).
    /// No memory is allocated; the Mat does NOT own the data.
    Mat to_mat() const;

    // =========================================================================
    // Shape comparison
    // =========================================================================

    bool same_shape(const Tensor &other) const;

    // =========================================================================
    // Printing
    // =========================================================================

    void print(const char *name = "") const;

private:
    /// Allocate and zero-fill size_ floats, set owns_data = true
    void alloc(int total);
    /// Release buffer if owns_data
    void free_data();
    /// Set shape[] from given dimensions and recompute size
    void set_shape(int d, int n0, int n1 = 1, int n2 = 1, int n3 = 1);
};

} // namespace tiny

#endif // __cplusplus
