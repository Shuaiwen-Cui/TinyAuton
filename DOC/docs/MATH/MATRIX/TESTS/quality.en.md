# Quality assurance

Execution date, device and firmware commit are unrecorded. The complete stage output is preserved below. Label counts are not counts of independent test cases.

[Overview](overview.md) · [Full source and record](../tiny-matrix-test.md)

```text
============ [tiny_matrix_test start] ============

[Test Organization: Application-Oriented Logic]
  Foundation → Basic Ops → Properties → Linear Systems → Decompositions → Applications → Quality


[G1: Quality Assurance - Boundary Conditions and Error Handling Tests]

[G1.1] Null Pointer Handling in print_matrix
[Error] Cannot print matrix: data pointer is null.

[G1.2] Null Pointer Handling in operator<<
[Error] Cannot print matrix: data pointer is null.


[G1.3] Invalid Block Parameters
[Error] block: invalid position: start_row=-1, start_col=0 (must be non-negative)
block(-1, 0, 2, 2): Error
[Error] block: block exceeds row boundary: start_row=2, block_rows=2, source.rows=3
block(2, 2, 2, 2) on 3x3 matrix: Error
[Error] block: invalid block size: block_rows=0, block_cols=2 (must be positive)
block(0, 0, 0, 2): Error

[G1.4] Invalid swap_rows Parameters
Before invalid swap_rows:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

[Error] swap_rows: row1 index out of range: row1=-1, matrix.rows=3
After swap_rows(-1, 1):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

[Error] swap_rows: row2 index out of range: row2=5, matrix.rows=3
After swap_rows(0, 5):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements


[G1.5] Invalid swap_cols Parameters
Before invalid swap_cols:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

[Error] swap_cols: col1 index out of range: col1=-1, matrix.cols=3
After swap_cols(-1, 1):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

[Error] swap_cols: col2 index out of range: col2=5, matrix.cols=3
After swap_cols(0, 5):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements


[G1.6] Division by Zero
[Error] operator/: division by zero (num=0).
mat3 / 0.0f: Empty (correct)

[G1.7] Matrix Division with Zero Elements
[Error] Matrix division failed: Division by zero detected at position (0, 1)
mat4 /= divisor (with zero):
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements


[G1.8] Empty Matrix Operations
[Error] operator+: null matrix data pointer.
Empty matrix addition: Empty matrix or error state (Expected) [PASS]

[G2: Quality Assurance - Performance Benchmarks Tests]

[G2.1] Matrix Addition Performance
[Performance] 50x50 Matrix Addition (100 iterations): 18139.00 us total, 181.39 us avg

[G2.2] Matrix Multiplication Performance
[Performance] 30x30 Matrix Multiplication (100 iterations): 66757.00 us total, 667.57 us avg

[G2.3] Matrix Transpose Performance
[Performance] 50x30 Matrix Transpose (100 iterations): 21993.00 us total, 219.93 us avg

[G2.4] Determinant Calculation Performance Comparison

[G2.4.1] Small Matrix (4x4) - Laplace Expansion
[Performance] 4x4 Determinant (Laplace, 10 iterations): 3242.00 us total, 324.20 us avg

[G2.4.2] Large Matrix (8x8) - LU Decomposition
[Performance] 8x8 Determinant (LU, 10 iterations): 1666.00 us total, 166.60 us avg

[G2.4.3] Large Matrix (8x8) - Gaussian Elimination
[Performance] 8x8 Determinant (Gaussian, 10 iterations): 463.00 us total, 46.30 us avg

[G2.4.4] Large Matrix (8x8) - Auto-select Method
[Performance] 8x8 Determinant (auto-select, 10 iterations): 1625.00 us total, 162.50 us avg

[Note] Performance Summary:
  - Laplace expansion (O(n!)): Suitable only for small matrices (n <= 4)
  - LU decomposition (O(n³)): Efficient for large matrices, auto-selected for n > 4
  - Gaussian elimination (O(n³)): Alternative efficient method for large matrices
  - Auto-select: Automatically chooses the best method based on matrix size

[G2.5] Matrix Copy with Padding Performance
[Performance] 8x8 Copy ROI (with padding) (100 iterations): 2675.00 us total, 26.75 us avg

[G2.6] Element Access Performance
[Performance] Computing element access (warmup)...
[Performance] 50x50 Element Access (all elements) (100 iterations): 9709.00 us total, 97.09 us avg

[G3: Quality Assurance - Memory Layout Tests (Padding and Step)]

[G3.1] Contiguous Memory (no padding)
Matrix 3x4 (step=4, pad=0):
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        0
step            4
memory          12
data pointer    0x3fce9af4
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
        0.00         1.00         2.00         3.00       |
        4.00         5.00         6.00         7.00       |
        8.00         9.00        10.00        11.00       |
<<< Matrix Elements


[G3.2] Padded Memory (step > col)
Matrix 3x4 (step=5, pad=1):
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
step            5
memory          15
data pointer    0x3fc9a404
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
        0.00         1.00         2.00         3.00       |   0.00 
        4.00         5.00         6.00         7.00       |   0.00 
        8.00         9.00        10.00        11.00       |   0.00 
<<< Matrix Elements


[G3.3] Addition with Padded Matrices
Result of padded matrix addition:
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
step            5
memory          15
data pointer    0x3fce9b68
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
       11.00        22.00        33.00        44.00       |  33.00 
       55.00        66.00        77.00        88.00       |  38.00 
       99.00       110.00       121.00       132.00       |   1.61 
<<< Matrix Elements


[G3.4] ROI Operations with Padded Matrices
ROI (1,1,2,2) from padded matrix:
Matrix Info >>>
rows            2
cols            2
elements        4
paddings        3
step            5
memory          10
data pointer    0x3fc9a41c
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      1   (This is a Sub-Matrix View)
<<< Matrix Info
Matrix Elements >>>
        5.00         6.00       |   7.00         0.00         8.00 
        9.00        10.00       |  11.00         0.00         0.00 
<<< Matrix Elements


[G3.5] Copy Operations Preserve Step
Copied matrix (should have step=4, no padding):
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        0
step            4
memory          12
data pointer    0x3fce9ba8
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
        0.00         1.00         2.00         3.00       |
        4.00         5.00         6.00         7.00       |
        8.00         9.00        10.00        11.00       |
<<< Matrix Elements

============ [tiny_matrix_test end] ============
```
