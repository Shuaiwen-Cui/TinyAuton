# Matrix properties

Execution date, device and firmware commit are unrecorded. The complete stage output is preserved below. Label counts are not counts of independent test cases.

**This stage contains one `[FAIL]`.** C4.4 expected an empty result for the non-square adjoint error path, but printed a 1×1 zero matrix. This records a historical failure without inferring whether current firmware fixes it.

[Overview](overview.md) · [Full source and record](../tiny-matrix-test.md)

```text
============ [tiny_matrix_test start] ============

[Test Organization: Application-Oriented Logic]
  Foundation → Basic Ops → Properties → Linear Systems → Decompositions → Applications → Quality


[C1: Matrix Transpose Tests]
Goal: Verify transpose swaps rows/cols and preserves values.

[C1.1] Transpose of 2x3 Matrix
Test setup: input shape 2x3, values 1..6 row-major.
Expected: output shape 3x2, with out(j,i)=in(i,j).
Original 2x3 Matrix:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
<<< Matrix Elements

Transposed 3x2 Matrix:
Matrix Elements >>>
           1            4       |
           2            5       |
           3            6       |
<<< Matrix Elements

Result check: [PASS]

[C1.2] Transpose of 3x3 Square Matrix
Test setup: square matrix 3x3 values 1..9.
Expected: diagonal unchanged, off-diagonal mirrored.
Original 3x3 Matrix:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

Transposed 3x3 Matrix:
Matrix Elements >>>
           1            4            7       |
           2            5            8       |
           3            6            9       |
<<< Matrix Elements

Result check: [PASS]

[C1.3] Transpose of Matrix with Padding
Test setup: external-buffer matrix 4x2 with step=3 (padding present).
Expected: transpose reads only logical cols (ignores padding values).
Original 4x2 Matrix (with padding):
Matrix Elements >>>
           1            2       |      0 
           3            4       |      0 
           5            6       |      0 
           7            8       |      0 
<<< Matrix Elements

Transposed 2x4 Matrix:
Matrix Elements >>>
           1            3            5            7       |
           2            4            6            8       |
<<< Matrix Elements

Result check: [PASS]

[C1.4] Transpose of Empty Matrix
Test setup: default matrix (implementation-defined, typically 1x1 or error state).
Expected: function should not crash; inspect printed error/state.
Matrix Elements >>>
           0       |
<<< Matrix Elements

Matrix Elements >>>
           0       |
<<< Matrix Elements


[C2: Matrix Minor and Cofactor Tests]
Goal: Verify minor/cofactor matrix extraction semantics and boundary behavior.
Reminder: in this implementation, cofactor() returns the same submatrix as minor().

[C2.1] Minor of 3x3 Matrix (Remove Row 1, Col 1)
Original 3x3 Matrix:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

Minor Matrix (remove row 1, col 1, no sign):
Matrix Elements >>>
           1            3       |
           7            9       |
<<< Matrix Elements


[C2.2] Cofactor of 3x3 Matrix (Remove Row 1, Col 1)
Note: Cofactor matrix is the same as minor matrix.
      The sign (-1)^(i+j) is applied when computing cofactor value, not to matrix elements.
Cofactor Matrix (same as minor):
Matrix Elements >>>
           1            3       |
           7            9       |
<<< Matrix Elements


[C2.3] Minor (Remove Row 0, Col 0)
Matrix Elements >>>
           5            6       |
           8            9       |
<<< Matrix Elements


[C2.4] Cofactor (Remove Row 0, Col 0)
Note: Cofactor matrix is the same as minor matrix.
Matrix Elements >>>
           5            6       |
           8            9       |
<<< Matrix Elements


[C2.5] Cofactor (Remove Row 0, Col 1)
Note: Cofactor matrix is the same as minor matrix.
      When computing cofactor value, sign (-1)^(0+1) = -1 would be applied.
Cofactor Matrix (same as minor):
Matrix Elements >>>
           4            6       |
           7            9       |
<<< Matrix Elements


[C2.6] Minor (Remove Row 2, Col 2)
Matrix Elements >>>
           1            2       |
           4            5       |
<<< Matrix Elements


[C2.7] Cofactor (Remove Row 2, Col 2)
Note: Cofactor matrix is the same as minor matrix.
Matrix Elements >>>
           1            2       |
           4            5       |
<<< Matrix Elements


[C2.8] Minor of 4x4 Matrix (Remove Row 2, Col 1)
Matrix Elements >>>
           1            2            3            4       |
           5            6            7            8       |
           9           10           11           12       |
          13           14           15           16       |
<<< Matrix Elements

Minor Matrix:
Matrix Elements >>>
           1            3            4       |
           5            7            8       |
          13           15           16       |
<<< Matrix Elements


[C2.9] Cofactor of 4x4 Matrix (Remove Row 2, Col 1)
Note: Cofactor matrix is the same as minor matrix.
      When computing cofactor value, sign (-1)^(2+1) = -1 would be applied.
Cofactor Matrix (same as minor):
Matrix Elements >>>
           1            3            4       |
           5            7            8       |
          13           15           16       |
<<< Matrix Elements


[C2.10] Non-square Matrix
Testing minor() on a 3x4 matrix (expect 2x3 result):
minor() result shape: 2x3 [PASS]
minor() content:
Matrix Elements >>>
           1            3            4       |
           9           11           12       |
<<< Matrix Elements

Testing cofactor() on a 3x4 matrix (expect empty - square required):
[Error] cofactor: requires a square matrix (got 3x4)
cofactor() result: Empty matrix (Expected) [PASS]

[C2.11] minor() - Boundary Case - Out of Bounds Indices
[Error] minor: target_row=-1 is out of range [0, 2]
minor(-1, 0): Empty matrix (Expected) [PASS]
[Error] minor: target_col=-1 is out of range [0, 2]
minor(0, -1): Empty matrix (Expected) [PASS]
[Error] minor: target_row=3 is out of range [0, 2]
minor(3, 0) (out of bounds): Empty matrix (Expected) [PASS]

[C2.12] minor() - Boundary Case - 1x1 Matrix
1x1 matrix minor(0,0): Empty matrix (Expected) [PASS]

[C3: Matrix Determinant Tests]
Goal: Verify determinant correctness across sizes and methods.
Auto strategy: n<=4 uses Laplace, n>4 uses LU.

[C3.1] 1x1 Matrix Determinant
Matrix:
Matrix Elements >>>
           7       |
<<< Matrix Elements

Determinant: 7  (Expected: 7) [PASS]

[C3.2] 2x2 Matrix Determinant
Matrix:
Matrix Elements >>>
           3            8       |
           4            6       |
<<< Matrix Elements

Determinant: -14  (Expected: -14) [PASS]

[C3.3] 3x3 Matrix Determinant
Matrix:
Matrix Elements >>>
           1            2            3       |
           0            4            5       |
           1            0            6       |
<<< Matrix Elements

Determinant: 22  (Expected: 22) [PASS]

[C3.4] 4x4 Matrix Determinant
Matrix:
Matrix Elements >>>
           1            2            3            4       |
           5            6            7            8       |
           9           10           11           12       |
          13           14           15           16       |
<<< Matrix Elements

Note: This matrix has linearly dependent rows (each row differs by constant 4),
      so the determinant should be 0.
Determinant: 0  (Expected: 0) [PASS]

[C3.5] 5x5 Matrix Determinant (Tests Auto-select to LU Method)
Matrix (5x5, tridiagonal):
Matrix Elements >>>
           2            1            0            0            0       |
           1            2            1            0            0       |
           0            1            2            1            0       |
           0            0            1            2            1       |
           0            0            0            1            2       |
<<< Matrix Elements

Determinant (auto-select, should use LU for n > 4): 6
Note: For n = 5 > 4, auto-select should use LU decomposition (O(n³)).

[C3.6] Non-square Matrix (Expect Error)
Matrix (3x4, non-square):
Matrix Elements >>>
           0            0            0            0       |
           0            0            0            0       |
           0            0            0            0       |
<<< Matrix Elements

[Error] Determinant requires a square matrix (got 3x4)
Determinant: 0  (Expected: 0 with error message)

[C3.7] Comparison of Different Methods (5x5 Matrix)
Matrix (5x5):
Matrix Elements >>>
           2            2            3            4            5       |
           2            5            6            8           10       |
           3            6           10           12           15       |
           4            8           12           17           20       |
           5           10           15           20           26       |
<<< Matrix Elements

Determinant (auto-select): 56  (should use LU for n > 4)
Determinant (Laplace):     56  (O(n!), slow for n=5)
Determinant (LU):          56  (O(n³), efficient)
Determinant (Gaussian):    56  (O(n³), efficient)
Note: All methods should give the same result (within numerical precision).
      Auto-select should use LU for n > 4, avoiding slow Laplace expansion.

[C3.8] Large Matrix (6x6) - Tests Efficient Methods
Matrix (6x6, showing first 4x4 block):
       1.5          2          3          4 ...
         2        4.5          6          8 ...
         3          6        9.5         12 ...
         4          8         12       16.5 ...
...
Determinant (auto-select, uses LU): 2.85937
Determinant (LU):                   2.85937
Determinant (Gaussian):             2.85938
Note: For n > 4, auto-select uses LU decomposition (O(n³) instead of O(n!)).

[C3.9] Large Matrix (8x8) - Performance Comparison
Matrix (8x8, showing first 4x4 block):
         1          2          3          4 ...
         2          4          6          8 ...
         3          6          9         12 ...
         4          8         12         16 ...
...
[Error] lu_decompose: matrix is singular or near-singular at column 1 (pivot = 0).
[Warning] determinant_lu: LU decomposition failed (status=458754), matrix may be singular
Determinant (LU):       0
Determinant (Gaussian): 0
Note: Both methods are O(n³) and should be much faster than Laplace expansion.

[C3.10] determinant_laplace() - Boundary Case - Empty Matrix
Empty matrix determinant (Laplace): 1 (Expected: 1.0) [PASS]

[C3.11] determinant_lu() - Boundary Case - Empty Matrix
Empty matrix determinant (LU): 1 (Expected: 1.0) [PASS]

[C3.12] determinant_gaussian() - Boundary Case - Empty Matrix
Empty matrix determinant (Gaussian): 1 (Expected: 1.0) [PASS]

[C3.13] Determinant Methods - Non-Square Matrix
[Error] Determinant requires a square matrix (got 2x3)
[Error] Determinant requires a square matrix (got 2x3)
[Error] Determinant requires a square matrix (got 2x3)
Non-square matrix (2x3) determinant (Laplace): 0 (Expected: 0.0) [PASS]
Non-square matrix (2x3) determinant (LU): 0 (Expected: 0.0) [PASS]
Non-square matrix (2x3) determinant (Gaussian): 0 (Expected: 0.0) [PASS]

[C4: Matrix Adjoint Tests]
Goal: Verify adj(A)=cofactor(A)^T behavior and error handling for non-square matrices.

[C4.1] Adjoint of 1x1 Matrix
Original Matrix:
Matrix Elements >>>
           5       |
<<< Matrix Elements

Adjoint Matrix:
Matrix Elements >>>
           1       |
<<< Matrix Elements

Result check: [PASS]

[C4.2] Adjoint of 2x2 Matrix
Original Matrix:
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Adjoint Matrix:
Matrix Elements >>>
           4           -2       |
          -3            1       |
<<< Matrix Elements

Result check: [PASS]

[C4.3] Adjoint of 3x3 Matrix
Original Matrix:
Matrix Elements >>>
           1            2            3       |
           0            4            5       |
           1            0            6       |
<<< Matrix Elements

Adjoint Matrix:
Matrix Elements >>>
          24          -12           -2       |
           5            3           -5       |
          -4            2            4       |
<<< Matrix Elements


[C4.4] Adjoint of Non-Square Matrix (Expect Error)
Original Matrix (2x3, non-square):
Matrix Elements >>>
           0            0            0       |
           0            0            0       |
<<< Matrix Elements

[Error] Adjoint requires a square matrix (got 2x3)
Adjoint Matrix (should be empty due to error):
Matrix Elements >>>
           0       |
<<< Matrix Elements

Result check: [FAIL]

[C5: Matrix Normalization Tests]
Goal: Verify in-place normalization by Frobenius norm and edge-case handling.

[C5.1] Normalize a Standard 2x2 Matrix
Before normalization:
Matrix Elements >>>
           3            4       |
           3            4       |
<<< Matrix Elements

After normalization (Expected L2 norm = 1):
Matrix Elements >>>
    0.424264     0.565685       |
    0.424264     0.565685       |
<<< Matrix Elements

Norm after normalize: 1 (Expected: 1.0) [PASS]

[C5.2] Normalize a 2x2 Matrix with step=4 (Padding Test)
Before normalization:
Matrix Elements >>>
           3            4       |      0            0 
           3            4       |      0            0 
<<< Matrix Elements

After normalization:
Matrix Elements >>>
    0.424264     0.565685       |      0            0 
    0.424264     0.565685       |      0            0 
<<< Matrix Elements

Norm after normalize (padding case): 1 (Expected: 1.0) [PASS]

[C5.3] Normalize a Zero Matrix (Expect Warning)
Matrix Elements >>>
           0            0       |
           0            0       |
<<< Matrix Elements

[Warning] normalize: matrix norm is zero (matrix is all zeros), normalization skipped
Norm after normalize attempt on zero matrix: 0 (Expected: 0.0, unchanged) [PASS]

[C6: Matrix Norm Calculation Tests]
Goal: Verify Frobenius norm values including padding and empty matrix behavior.

[C6.1] 2x2 Matrix Norm (Expect 5.0)
Matrix:
Matrix Elements >>>
           3            4       |
           0            0       |
<<< Matrix Elements

Calculated Norm: 5 (Expected: 5.0) [PASS]

[C6.2] Zero Matrix Norm (Expect 0.0)
Matrix:
Matrix Elements >>>
           0            0            0       |
           0            0            0       |
           0            0            0       |
<<< Matrix Elements

Calculated Norm: 0 (Expected: 0.0) [PASS]

[C6.3] Matrix with Negative Values
Matrix:
Matrix Elements >>>
          -1           -2       |
          -3           -4       |
<<< Matrix Elements

Calculated Norm: 5.47723  (Expect sqrt(30) ≈ 5.477) [PASS]

[C6.4] 2x2 Matrix with step=4 (Padding Test)
Matrix:
Matrix Elements >>>
           1            2       |      0            0 
           3            4       |      0            0 
<<< Matrix Elements

Calculated Norm: 5.47723  (Expect sqrt(30) ≈ 5.477) [PASS]

[C6.5] Empty Matrix Norm (Expect 0.0, No Error)
Calculated Norm (0x0): 0  (Expected: 0.0) [PASS]

[C7: Matrix Inversion Tests]
Goal: Verify inverse_adjoint() on invertible/singular/non-square matrices.

[C7.1] Inverse of 2x2 Matrix
Original Matrix:
Matrix Elements >>>
           4            7       |
           2            6       |
<<< Matrix Elements

Inverse Matrix:
Matrix Elements >>>
         0.6         -0.7       |
        -0.2          0.4       |
<<< Matrix Elements

Expected Approx:
[ 0.6  -0.7 ]
[ -0.2  0.4 ]
Result check: [PASS]

[C7.2] Singular Matrix (Expect Error)
Original Matrix:
Matrix Elements >>>
           1            2       |
           2            4       |
<<< Matrix Elements

Note: This matrix is singular (determinant = 0), so inverse should fail.
[Error] inverse_adjoint: matrix is singular (det=0), cannot compute inverse
Inverse Matrix (Should be zero matrix):
Matrix Elements >>>
           0       |
<<< Matrix Elements

Result check: [PASS]

[C7.3] Inverse of 3x3 Matrix
Original Matrix:
Matrix Elements >>>
           3            0            2       |
           2            0           -2       |
           0            1            1       |
<<< Matrix Elements

Inverse Matrix:
Matrix Elements >>>
         0.2          0.2           -0       |
        -0.2          0.3            1       |
         0.2         -0.3            0       |
<<< Matrix Elements


[C7.4] Non-Square Matrix (Expect Error)
Original Matrix (2x3, non-square):
Matrix Elements >>>
           0            0            0       |
           0            0            0       |
<<< Matrix Elements

[Error] inverse_adjoint: requires square matrix (got 2x3)
Inverse Matrix (should be empty due to error):
Matrix Elements >>>
           0       |
<<< Matrix Elements

Result check: [PASS]

[C8: Matrix Utilities Tests]
Goal: Verify eye/ones/augment/vstack shapes and representative values.

[C8.1] Generate Identity Matrix (eye)
3x3 Identity Matrix:
Matrix Elements >>>
           1            0            0       |
           0            1            0       |
           0            0            1       |
<<< Matrix Elements

Result check (I3): [PASS]
5x5 Identity Matrix:
Matrix Elements >>>
           1            0            0            0            0       |
           0            1            0            0            0       |
           0            0            1            0            0       |
           0            0            0            1            0       |
           0            0            0            0            1       |
<<< Matrix Elements


[C8.2] Generate Ones Matrix
3x4 Ones Matrix:
Matrix Elements >>>
           1            1            1            1       |
           1            1            1            1       |
           1            1            1            1       |
<<< Matrix Elements

Result check (ones 3x4): [PASS]
4x4 Ones Matrix (Square):
Matrix Elements >>>
           1            1            1            1       |
           1            1            1            1       |
           1            1            1            1       |
           1            1            1            1       |
<<< Matrix Elements


[C8.3] Augment Two Matrices Horizontally [A | B]
Matrix A:
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Matrix B:
Matrix Elements >>>
           5            6            7       |
           8            9           10       |
<<< Matrix Elements

Augmented Matrix [A | B]:
Matrix Elements >>>
           1            2            5            6            7       |
           3            4            8            9           10       |
<<< Matrix Elements

Result check: [PASS]

[C8.4] Augment with Row Mismatch (Expect Error)
[Error] augment: row counts must match (A: 2, B: 3)
Matrix Info >>>
rows            1
cols            1
elements        1
paddings        0
step            1
memory          1
data pointer    0x3fce9a90
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Result check: [PASS]

[C8.5] Vertically Stack Two Matrices [A; B]
Matrix A (top):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
<<< Matrix Elements

Matrix B (bottom):
Matrix Elements >>>
           7            8            9       |
          10           11           12       |
<<< Matrix Elements

Vertically Stacked Matrix [A; B]:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
          10           11           12       |
<<< Matrix Elements

Expected: 4x3 matrix with A on top, B on bottom
Result check: [PASS]

[C8.6] Vertical Stack with Different Row Counts (Same Columns)
Matrix A (1x3):
Matrix Elements >>>
           1            2            3       |
<<< Matrix Elements

Matrix B (3x3):
Matrix Elements >>>
           4            5            6       |
           7            8            9       |
          10           11           12       |
<<< Matrix Elements

Vertically Stacked Matrix [A; B] (1x3 + 3x3 = 4x3):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
          10           11           12       |
<<< Matrix Elements

Result check: [PASS]

[C8.7] VStack with Column Mismatch (Expect Error)
Matrix A (2x2):
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Matrix B (2x3, different columns):
Matrix Elements >>>
           5            6            7       |
           8            9           10       |
<<< Matrix Elements

[Error] vstack: column counts must match (A: 2, B: 3)
Result (should be empty due to error):
Matrix Info >>>
rows            1
cols            1
elements        1
paddings        0
step            1
memory          1
data pointer    0x3fce9d40
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Result check: [PASS]
============ [tiny_matrix_test end] ============
```
