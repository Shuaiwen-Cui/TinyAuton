# C++ matrix API overview

`tiny::Mat` uses `float`, row-major storage and optional row padding. Owned buffers are released at destruction; callers must keep external buffers and ROI parent storage alive.

| Reading goal | Page |
|---|---|
| Construction, copying, external buffers and layout | [Objects and memory](../CORE/api.md) |
| ROI, data manipulation, arithmetic and streams | [Operations](../ARITH/api.md) |
| Determinants, inverses and linear systems | [Linear systems](../LINALG/api.md) |
| LU, Cholesky, QR, SVD and solving | [Decompositions](../DECOMP/api.md) |
| Power iteration, Jacobi and QR | [Eigenvalues](../EIGEN/api.md) |
| Full header and original links | [Complete reference](../tiny-matrix-api.md) |
| Implementation and verification | [Source](../tiny-matrix-code.md) · [Test overview](../TESTS/overview.md) |

## Minimal usage

```cpp
#include "tiny_matrix.hpp"

void matrix_demo()
{
    tiny::Mat A(2, 2);
    if (!A.data) return;
    A(0, 0) = 4.0f; A(0, 1) = 1.0f;
    A(1, 0) = 1.0f; A(1, 1) = 3.0f;
    auto lu = A.lu_decompose();
    if (lu.status != TINY_OK) return;
    lu.L.print_matrix(false);
    lu.U.print_matrix(false);
}
```

## Usage boundaries

- Fields are `row/col/step`; `stride` is a legacy alias of `step` in this class. Elements live at `data[i * step + j]`.
- C `tiny_mat` padding/stride arguments are not interchangeable with C++ metadata.
- Preconditions and failure states depend on the method. The QR eigenvalue API returns only real parts of complex results.
- Timing, code size and stability require measurements for the actual input and build; unspecified historical speedup factors are not guarantees.
