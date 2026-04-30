# Notes

!!! note "Notes"
    `tiny::Tensor` is the universal data carrier of `tiny_ai`. It provides up to 4-D float32 tensors, PSRAM-aware allocation, and a zero-copy bridge to `tiny::Mat`. All other modules (activations, layers, losses) consume and produce `Tensor`.

## CONCEPT

### Shape and size

```cpp
class Tensor
{
    int   ndim;        // 1 ~ 4
    int   shape[4];    // unused dims are 1
    int   size;        // shape[0]*shape[1]*shape[2]*shape[3]
    float *data;       // row-major flat buffer
    bool  owns_data;   // free on destruction?
};
```

- **Row-major**: a 4-D element `(i, j, k, l)` lives at `((i * shape[1] + j) * shape[2] + k) * shape[3] + l`.
- **`size = shape[0] * shape[1] * shape[2] * shape[3]`** — unused dims are 1, so `size` always equals the element count.
- **Ownership**: tensors built via constructors own their buffer (`owns_data = true`); views built via `from_data()` do not, and the caller must keep the buffer alive.

### Common shape conventions

| Module | Shape | Notes |
| --- | --- | --- |
| Dense | `[batch, features]` | 2-D |
| Conv1D | `[batch, channels, length]` | 3-D |
| Conv2D | `[batch, channels, height, width]` | 4-D |
| Attention | `[batch, seq_len, embed_dim]` | 3-D |
| GlobalAvgPool input | `[batch, seq_len, feat]` → output `[batch, feat]` | seq → vector |

## CONSTRUCTORS & NAMED FACTORIES

### Direct construction

```cpp
Tensor t1(N);              // 1-D
Tensor t2(N, F);           // 2-D
Tensor t3(N, C, L);        // 3-D
Tensor t4(N, C, H, W);     // 4-D
Tensor t0;                 // empty (no allocation)
```

Construction calls the private `alloc()` which `TINY_AI_MALLOC`s `size * sizeof(float)` bytes and zero-fills.

### Named factories

```cpp
Tensor::zeros(n0);                          // 1-D zero
Tensor::zeros(n0, n1);                      // 2-D zero
Tensor::zeros(n0, n1, n2);                  // 3-D zero
Tensor::zeros(n0, n1, n2, n3);              // 4-D zero

Tensor::zeros_like(other);                  // shape-cloned zero tensor

Tensor::from_data(buf, ndim, shape);        // wrap external buffer; no copy, no ownership
```

!!! warning "from_data ownership"
    The Tensor returned by `from_data()` does *not* own the buffer. The caller must guarantee `buf` outlives the tensor (e.g. weights placed in PSRAM whose lifetime ≥ the model).

## ELEMENT ACCESS

`Tensor` exposes inline `at()` overloads keyed on the number of indices:

```cpp
inline float &at(int i);
inline float &at(int i, int j);
inline float &at(int i, int j, int k);
inline float &at(int i, int j, int k, int l);
```

Const overloads are provided too. `at()` is inline and does *not* bounds-check — this gives bare-array speed on MCUs like ESP32-S3.

```cpp
Tensor x(B, C, H, W);
for (int b = 0; b < B; b++)
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                x.at(b, c, h, w) = some_value;
```

Direct `x.data[idx]` access is also allowed and matches the row-major layout.

## SHAPE SEMANTIC ACCESSORS

```cpp
int batch();     // ndim>=3 ? shape[0] : 1
int rows();      // ndim>=2 ? shape[ndim-2] : shape[0]
int cols();      // shape[ndim-1]
int channels();  // ndim==4 ? shape[1] : 1
```

These let activations / losses treat 2-D and 3-D tensors uniformly with "last dim = class/feature axis". For example, `softmax_inplace` uses `rows = size / cols, cls = cols` to normalise along the last dim.

## IN-PLACE OPS

```cpp
void zero();                          // memset 0
void fill(float val);                 // fill constant
void copy_from(const Tensor &src);    // shapes must match
Tensor clone() const;                 // deep copy (owns_data=true)
```

## RESHAPE

```cpp
tiny_error_t reshape(int ndim, const int *new_shape);   // element count must match
tiny_error_t reshape_2d(int n0, int n1);
tiny_error_t reshape_3d(int n0, int n1, int n2);
```

`reshape` only changes `ndim` / `shape`; the buffer is *not* reallocated. If `new_shape` total ≠ `size`, returns `TINY_ERR_AI_INVALID_SHAPE`.

The Attention example uses `reshape_3d / reshape_2d` to flip between `[B, F]` and `[B, T, E]`.

## INTEROP WITH `tiny::Mat`

```cpp
Mat to_mat() const;
```

Defined for 2-D tensors only: returns a `tiny::Mat` that shares the same buffer (no copy, no ownership). Useful when an upstream stage wants matrix multiplication / row-column ops powered by `tiny_math`.

```cpp
Tensor x(B, F);
Mat xm = x.to_mat();          // [B, F] view, zero-copy
// dispatch tiny_math matrix APIs through xm
```

## SHAPE COMPARISON & PRINTING

```cpp
bool same_shape(const Tensor &other) const;
void print(const char *name = "") const;
```

`print()` shows the shape metadata then dumps up to 32 elements — a quick way to diagnose problems over the serial console.

## COPY / MOVE

`Tensor` follows the rule of five:

- Copy ctor / copy assignment → deep copy (re-`alloc` + `memcpy`).
- Move ctor / move assignment → take ownership and null out the source.
- Destructor → `TINY_AI_FREE(data)` only when `owns_data == true`.

Therefore Tensors can be returned by value or stored in `std::vector<Tensor>` safely.
