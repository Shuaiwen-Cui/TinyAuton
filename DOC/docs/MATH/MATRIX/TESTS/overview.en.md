# C++ matrix tests and results

**Duplicate historical record:** the original Basic Operations output is identical to Object Foundation and contains A-group tests. It does not verify the B-group operations. The original block remains intact below.

Historical records cover eight A–G stages. Counts below describe labels in the original output, not total coverage or current firmware success. Date, device and commit are unrecorded.

**Known historical failure:** C4.4 in [matrix properties](properties.md) contains one `[FAIL]` on the non-square adjoint error path. Its original output remains intact.

| Stage | Historical `[PASS]` labels | `[FAIL]` labels |
|---|---:|---:|
| [Object foundation](foundation.md) | 12 | 0 |
| [Basic operations](basic.md) | 12 | 0 |
| [Matrix properties](properties.md) | 39 | 1 |
| [Linear systems](systems.md) | 9 | 0 |
| [Decompositions](decompositions.md) | 53 | 0 |
| [Eigenvalue applications](eigen.md) | 58 | 0 |
| [Auxiliary functions](auxiliary.md) | 0 | 0 |
| [Quality assurance](quality.md) | 1 | 0 |

## Current reproduction entry

The MATH project's `main/AIoTNode.cpp` calls `tiny_matrix_test()`. The runner currently enables G1 `test_boundary_conditions()`, G2 `test_performance_benchmarks()` and G3 `test_memory_layout()`; most A–F calls are commented out. Enable the relevant calls explicitly and record new output to reproduce a stage.

[Full source and historical record](../tiny-matrix-test.md) · [API overview](../OVERVIEW/api.md)
