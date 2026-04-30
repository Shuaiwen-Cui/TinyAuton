# Code

## tiny_tensor.hpp

```cpp
/**
 * @file tiny_tensor.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief N-dimensional float32 tensor for tiny_ai.
 *        Supports up to 4 dimensions.  For 2D tensors, provides a zero-copy
 *        view as tiny::Mat.  PSRAM-aware allocation on ESP32-S3.
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#pragma once

#include "tiny_ai_config.h"
#include "tiny_matrix.hpp"

#ifdef __cplusplus

#include <stdint.h>
#include <cstdio>

namespace tiny
{

class Tensor
{
public:
    int   ndim;
    int   shape[4];
    int   size;
    float *data;
    bool  owns_data;

    Tensor();
    explicit Tensor(int n0);
    Tensor(int n0, int n1);
    Tensor(int n0, int n1, int n2);
    Tensor(int n0, int n1, int n2, int n3);

    Tensor(const Tensor &other);
    Tensor(Tensor &&other) noexcept;
    ~Tensor();

    Tensor &operator=(const Tensor &other);
    Tensor &operator=(Tensor &&other) noexcept;

    static Tensor zeros(int n0);
    static Tensor zeros(int n0, int n1);
    static Tensor zeros(int n0, int n1, int n2);
    static Tensor zeros(int n0, int n1, int n2, int n3);
    static Tensor zeros_like(const Tensor &other);
    static Tensor from_data(float *data, int ndim, const int *shape);

    inline float &at(int i)                              { return data[i]; }
    inline float &at(int i, int j)                       { return data[i * shape[1] + j]; }
    inline float &at(int i, int j, int k)                { return data[(i * shape[1] + j) * shape[2] + k]; }
    inline float &at(int i, int j, int k, int l)         { return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l]; }
    inline const float &at(int i)                        const { return data[i]; }
    inline const float &at(int i, int j)                 const { return data[i * shape[1] + j]; }
    inline const float &at(int i, int j, int k)          const { return data[(i * shape[1] + j) * shape[2] + k]; }
    inline const float &at(int i, int j, int k, int l)   const { return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l]; }

    inline int batch()    const { return ndim >= 3 ? shape[0] : 1; }
    inline int rows()     const { return ndim >= 2 ? shape[ndim - 2] : shape[0]; }
    inline int cols()     const { return shape[ndim - 1]; }
    inline int channels() const { return ndim == 4 ? shape[1] : 1; }

    void zero();
    void fill(float val);
    void copy_from(const Tensor &src);
    Tensor clone() const;

    tiny_error_t reshape(int ndim, const int *new_shape);
    tiny_error_t reshape_2d(int n0, int n1);
    tiny_error_t reshape_3d(int n0, int n1, int n2);

    Mat to_mat() const;

    bool same_shape(const Tensor &other) const;

    void print(const char *name = "") const;

private:
    void alloc(int total);
    void free_data();
    void set_shape(int d, int n0, int n1 = 1, int n2 = 1, int n3 = 1);
};

} // namespace tiny

#endif // __cplusplus
```

## tiny_tensor.cpp

```cpp
/**
 * @file tiny_tensor.cpp
 * @brief Tensor implementation for tiny_ai.
 */

#include "tiny_tensor.hpp"

#ifdef __cplusplus

#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace tiny
{

void Tensor::set_shape(int d, int n0, int n1, int n2, int n3)
{
    ndim     = d;
    shape[0] = n0; shape[1] = n1; shape[2] = n2; shape[3] = n3;
    size     = n0 * n1 * n2 * n3;
}

void Tensor::alloc(int total)
{
    if (total <= 0) { data = nullptr; owns_data = false; return; }
    data      = (float *)TINY_AI_MALLOC((size_t)total * sizeof(float));
    owns_data = true;
    if (data) memset(data, 0, (size_t)total * sizeof(float));
}

void Tensor::free_data()
{
    if (owns_data && data) { TINY_AI_FREE(data); }
    data      = nullptr;
    owns_data = false;
}

Tensor::Tensor() : ndim(0), size(0), data(nullptr), owns_data(false)
{
    shape[0] = shape[1] = shape[2] = shape[3] = 0;
}

Tensor::Tensor(int n0)               { set_shape(1, n0); alloc(size); }
Tensor::Tensor(int n0, int n1)       { set_shape(2, n0, n1); alloc(size); }
Tensor::Tensor(int n0, int n1, int n2) { set_shape(3, n0, n1, n2); alloc(size); }
Tensor::Tensor(int n0, int n1, int n2, int n3) { set_shape(4, n0, n1, n2, n3); alloc(size); }

Tensor::Tensor(const Tensor &other)
    : ndim(other.ndim), size(other.size), data(nullptr), owns_data(false)
{
    memcpy(shape, other.shape, sizeof(shape));
    alloc(size);
    if (data && other.data) memcpy(data, other.data, (size_t)size * sizeof(float));
}

Tensor::Tensor(Tensor &&other) noexcept
    : ndim(other.ndim), size(other.size), data(other.data), owns_data(other.owns_data)
{
    memcpy(shape, other.shape, sizeof(shape));
    other.data      = nullptr;
    other.owns_data = false;
    other.size      = 0;
}

Tensor::~Tensor() { free_data(); }

Tensor &Tensor::operator=(const Tensor &other)
{
    if (this == &other) return *this;
    if (size != other.size)
    {
        free_data();
        set_shape(other.ndim, other.shape[0], other.shape[1], other.shape[2], other.shape[3]);
        alloc(size);
    }
    else
    {
        ndim = other.ndim;
        memcpy(shape, other.shape, sizeof(shape));
        size = other.size;
    }
    if (data && other.data) memcpy(data, other.data, (size_t)size * sizeof(float));
    return *this;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept
{
    if (this == &other) return *this;
    free_data();
    ndim      = other.ndim;
    size      = other.size;
    data      = other.data;
    owns_data = other.owns_data;
    memcpy(shape, other.shape, sizeof(shape));
    other.data      = nullptr;
    other.owns_data = false;
    other.size      = 0;
    return *this;
}

Tensor Tensor::zeros(int n0)             { return Tensor(n0); }
Tensor Tensor::zeros(int n0, int n1)     { return Tensor(n0, n1); }
Tensor Tensor::zeros(int n0, int n1, int n2) { return Tensor(n0, n1, n2); }
Tensor Tensor::zeros(int n0, int n1, int n2, int n3) { return Tensor(n0, n1, n2, n3); }

Tensor Tensor::zeros_like(const Tensor &other)
{
    Tensor t;
    t.ndim = other.ndim;
    memcpy(t.shape, other.shape, sizeof(t.shape));
    t.size = other.size;
    t.alloc(t.size);
    return t;
}

Tensor Tensor::from_data(float *buf, int d, const int *sh)
{
    Tensor t;
    t.ndim = d;
    t.size = 1;
    for (int i = 0; i < 4; i++)
    {
        t.shape[i] = (i < d) ? sh[i] : 1;
        if (i < d) t.size *= sh[i];
    }
    t.data      = buf;
    t.owns_data = false;
    return t;
}

void Tensor::zero()
{
    if (data) memset(data, 0, (size_t)size * sizeof(float));
}

void Tensor::fill(float val)
{
    if (!data) return;
    for (int i = 0; i < size; i++) data[i] = val;
}

void Tensor::copy_from(const Tensor &src)
{
    if (data && src.data && size == src.size)
        memcpy(data, src.data, (size_t)size * sizeof(float));
}

Tensor Tensor::clone() const { Tensor t(*this); return t; }

tiny_error_t Tensor::reshape(int d, const int *sh)
{
    int new_size = 1;
    for (int i = 0; i < d; i++) new_size *= sh[i];
    if (new_size != size) return TINY_ERR_AI_INVALID_SHAPE;
    ndim = d;
    for (int i = 0; i < 4; i++) shape[i] = (i < d) ? sh[i] : 1;
    return TINY_OK;
}

tiny_error_t Tensor::reshape_2d(int n0, int n1)        { int sh[2] = {n0, n1};      return reshape(2, sh); }
tiny_error_t Tensor::reshape_3d(int n0, int n1, int n2){ int sh[3] = {n0, n1, n2};  return reshape(3, sh); }

Mat Tensor::to_mat() const
{
    int r = rows(), c = cols();
    Mat m(data, r, c, 0);
    return m;
}

bool Tensor::same_shape(const Tensor &other) const
{
    if (ndim != other.ndim) return false;
    for (int i = 0; i < ndim; i++)
        if (shape[i] != other.shape[i]) return false;
    return true;
}

void Tensor::print(const char *name) const
{
    printf("Tensor '%s'  ndim=%d  shape=[", name ? name : "", ndim);
    for (int i = 0; i < ndim; i++) printf("%d%s", shape[i], i < ndim - 1 ? "," : "");
    printf("]  size=%d\n", size);

    if (!data) { printf("  <null data>\n"); return; }

    int show = size < 32 ? size : 32;
    for (int i = 0; i < show; i++) printf("  [%d] %.6f\n", i, data[i]);
    if (size > 32) printf("  ... (%d more)\n", size - 32);
}

} // namespace tiny

#endif // __cplusplus
```
