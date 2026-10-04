# Linear systems

Execution date, device and firmware commit are unrecorded. The complete stage output is preserved below. Label counts are not counts of independent test cases.

[Overview](overview.md) · [Full source and record](../tiny-matrix-test.md)

```text
============ [tiny_matrix_test start] ============

[Test Organization: Application-Oriented Logic]
  Foundation → Basic Ops → Properties → Linear Systems → Decompositions → Applications → Quality


[D1: Gaussian Elimination Tests]

[D1.1] 3x3 Matrix (Simple Upper Triangular)
Original Matrix:
Matrix Elements >>>
           2            1           -1       |
          -3           -1            2       |
          -2            1            2       |
<<< Matrix Elements

After Gaussian Elimination (Should be upper triangular):
Matrix Elements >>>
          -3           -1            2       |
           0      1.66667     0.666667       |
           0            0          0.2       |
<<< Matrix Elements


[D1.2] 3x4 Augmented Matrix (Linear System Ax = b)
Original Augmented Matrix [A | b]:
Matrix Elements >>>
           1            2           -1            8       |
          -3           -1            2          -11       |
          -2            1            2           -3       |
<<< Matrix Elements

After Gaussian Elimination (Row Echelon Form):
Matrix Elements >>>
          -3           -1            2          -11       |
           0      1.66667     0.666667      4.33333       |
           0            0           -1            0       |
<<< Matrix Elements


[D1.3] Singular Matrix (No Unique Solution)
Original Singular Matrix:
Matrix Elements >>>
           1            2       |
           2            4       |
<<< Matrix Elements

After Gaussian Elimination (Should show rows of zeros):
Matrix Elements >>>
           2            4       |
           0            0       |
<<< Matrix Elements


[D1.4] Zero Matrix
Matrix Elements >>>
           0            0            0       |
           0            0            0       |
           0            0            0       |
<<< Matrix Elements

After Gaussian Elimination (Should be a zero matrix):
Matrix Elements >>>
           0            0            0       |
           0            0            0       |
           0            0            0       |
<<< Matrix Elements


[D1.5] gaussian_eliminate() - Boundary Case - Empty Matrix
[Error] gaussian_eliminate: matrix data pointer is null
Empty matrix gaussian_eliminate: Empty matrix or error state (Expected) [PASS]

[D1.6] gaussian_eliminate() - Boundary Case - 1x1 Matrix
1x1 matrix after gaussian_eliminate:
Matrix Elements >>>
           5       |
<<< Matrix Elements

1x1 matrix gaussian_eliminate: [PASS]

[D2: Row Reduce from Gaussian (RREF) Tests]

[D2.1] 3x4 Augmented Matrix
Original Matrix:
Matrix Elements >>>
           1            2           -1           -4       |
           2            3           -1          -11       |
          -2            0           -3           22       |
<<< Matrix Elements

RREF Result:
Matrix Elements >>>
           1            0            0           -8       |
           0            1            0            1       |
           0            0            1           -2       |
<<< Matrix Elements


[D2.2] 2x3 Matrix
Original Matrix:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
<<< Matrix Elements

RREF Result:
Matrix Elements >>>
           1            0           -1       |
           0            1            2       |
<<< Matrix Elements


[D2.3] Already Reduced Matrix
Original Matrix:
Matrix Elements >>>
           1            0            2       |
           0            1            3       |
<<< Matrix Elements

RREF Result:
Matrix Elements >>>
           1            0            2       |
           0            1            3       |
<<< Matrix Elements


[D2.4] row_reduce_from_gaussian() - Boundary Case - Empty Matrix
[Error] row_reduce_from_gaussian: matrix data pointer is null
Empty matrix row_reduce_from_gaussian: Empty matrix or error state (Expected) [PASS]

[D3: Gaussian Inverse Tests]

[D3.1] 2x2 Matrix Inverse
Original matrix (mat1):
Matrix Elements >>>
           4            7       |
           2            6       |
<<< Matrix Elements

Inverse matrix (mat1):
Matrix Elements >>>
         0.6         -0.7       |
        -0.2          0.4       |
<<< Matrix Elements


[D3.2] Identity Matrix Inverse
Original matrix (Identity):
Matrix Elements >>>
           1            0            0       |
           0            1            0       |
           0            0            1       |
<<< Matrix Elements

Inverse matrix (Identity):
Matrix Elements >>>
           1            0            0       |
           0            1            0       |
           0            0            1       |
<<< Matrix Elements


[D3.3] Singular Matrix (Expected: No Inverse)
Original matrix (singular):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

[Error] inverse_gje: matrix is singular (left block not identity at (0, 2): expected 0, got -1).
Inverse matrix (singular):
[Error] Cannot print matrix: data pointer is null.

[D3.4] 3x3 Matrix Inverse
Original matrix (mat4):
Matrix Elements >>>
           4            7            2       |
           3            5            1       |
           8            6            9       |
<<< Matrix Elements

Inverse matrix (mat4):
Matrix Elements >>>
    -1.85714      2.42857     0.142857       |
    0.904762    -0.952381   -0.0952381       |
     1.04762     -1.52381     0.047619       |
<<< Matrix Elements


[D3.5] Non-square Matrix Inverse (Expected Error)
Original matrix (non-square):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
<<< Matrix Elements

[Error] inverse_gje: requires a square matrix (got 2x3).
Inverse matrix (non-square):
[Error] Cannot print matrix: data pointer is null.

[D4: Dot Product Tests]

[D4.1] Valid Dot Product (Same Length Vectors)
Vector A:
Matrix Elements >>>
           1       |
           2       |
           3       |
<<< Matrix Elements

Vector B:
Matrix Elements >>>
           4       |
           5       |
           6       |
<<< Matrix Elements

Dot product of vectorA and vectorB: 32

[D4.2] Invalid Dot Product (Dimension Mismatch)
Vector A (3x1):
Matrix Elements >>>
           1       |
           2       |
           3       |
<<< Matrix Elements

Vector C (2x1, different size):
Matrix Elements >>>
           1       |
           2       |
<<< Matrix Elements

[Error] dotprod: matrices must have the same size (A: 3x1, B: 2x1)
Dot product (dimension mismatch): 0

[D4.3] Dot Product of Zero Vectors
Zero Vector A:
Matrix Elements >>>
           0       |
           0       |
           0       |
<<< Matrix Elements

Zero Vector B:
Matrix Elements >>>
           0       |
           0       |
           0       |
<<< Matrix Elements

Dot product of zero vectors: 0

[D5: Solve Linear System Tests]

[D5.1] Solving a Simple 2x2 System Ax = b
Matrix A:
Matrix Elements >>>
           2            1       |
           1            3       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           5       |
           6       |
<<< Matrix Elements

Solution x:
Matrix Elements >>>
         1.8       |
         1.4       |
<<< Matrix Elements


[D5.2] Solving a 3x3 System Ax = b
Matrix A:
Matrix Elements >>>
           1            2            1       |
           2            0            3       |
           3            2            1       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           9       |
           8       |
           7       |
<<< Matrix Elements

Solution x:
Matrix Elements >>>
          -1       |
     3.33333       |
     3.33333       |
<<< Matrix Elements


[D5.3] Solving a System Where One Row is All Zeros (Expect Failure or Infinite Solutions)
Matrix A (has zero row):
Matrix Elements >>>
           1            2            3       |
           0            0            0       |
           4            5            6       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           9       |
           0       |
          15       |
<<< Matrix Elements

[Error] solve: zero or near-zero pivot at (2, 2), system is singular or rank-deficient.
Solution x:
[Error] Cannot print matrix: data pointer is null.

[D5.4] Solving a System with Zero Determinant (Singular Matrix)
Matrix A (singular, determinant = 0):
Matrix Elements >>>
           2            4            1       |
           1            2            3       |
           3            6            2       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           5       |
           6       |
           7       |
<<< Matrix Elements

[Error] solve: zero or near-zero pivot at (2, 2), system is singular or rank-deficient.
Solution x:
[Error] Cannot print matrix: data pointer is null.

[D5.5] Solving a System with Linearly Dependent Rows (Expect Failure or Infinite Solutions)
Matrix A (all rows linearly dependent):
Matrix Elements >>>
           1            1            1       |
           2            2            2       |
           3            3            3       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           6       |
          12       |
          18       |
<<< Matrix Elements

[Error] solve: zero or near-zero pivot at (2, 2), system is singular or rank-deficient.
Solution x:
[Error] Cannot print matrix: data pointer is null.

[D5.6] Solving a Larger 4x4 System Ax = b
Matrix A:
Matrix Elements >>>
           4            2            3            1       |
           2            5            1            2       |
           3            1            6            3       |
           1            2            3            4       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
          10       |
          12       |
          14       |
          16       |
<<< Matrix Elements

Solution x:
Matrix Elements >>>
     1.80645       |
    0.258065       |
   -0.516129       |
     3.80645       |
<<< Matrix Elements


[D5.7] solve() - Boundary Case - Empty Matrix
[Error] solve: matrix A data pointer is null.
Empty system solve: Empty matrix or error state (Expected) [PASS]

[D5.8] solve() - Error Handling - Dimension Mismatch
[Error] solve: dimensions do not match (A: 2x2, b: 3x1, expected b: 2x1).
Dimension mismatch solve: Empty matrix or error state (Expected) [PASS]

[D6: Band Solve Tests]

[D6.1] Simple 3x3 Band Matrix
Matrix A:
Matrix Elements >>>
           2            1            0       |
           1            3            2       |
           0            1            4       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           5       |
           6       |
           7       |
<<< Matrix Elements

Solution x:
Matrix Elements >>>
         2.5       |
           0       |
        1.75       |
<<< Matrix Elements


[D6.2] 4x4 Band Matrix
Matrix A:
Matrix Elements >>>
           2            1            0            0       |
           1            3            2            0       |
           0            1            4            2       |
           0            0            1            5       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           8       |
           9       |
          10       |
          11       |
<<< Matrix Elements

Solution x:
Matrix Elements >>>
     3.51429       |
    0.971429       |
     1.28571       |
     1.94286       |
<<< Matrix Elements


[D6.3] Incompatible Dimensions (Expect Error)
Matrix A (3x3):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

Vector b (2x1, incompatible):
Matrix Elements >>>
          10       |
          11       |
<<< Matrix Elements

[Error] band_solve: dimensions do not match (A: 3x3, b: 2x1, expected b: 3x1).
Solution x:
[Error] Cannot print matrix: data pointer is null.

[D6.4] Singular Matrix (No Unique Solution)
Matrix A (singular, linearly dependent rows):
Matrix Elements >>>
           1            2            3       |
           2            4            6       |
           3            6            9       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
          10       |
          20       |
          30       |
<<< Matrix Elements

[Error] band_solve: zero or near-zero pivot at (1, 1) = 0; matrix is singular or requires pivoting.
Solution x:
[Error] Cannot print matrix: data pointer is null.

[D6.5] band_solve() - Boundary Case - Empty Matrix
[Error] band_solve: matrix A data pointer is null.
Empty system band_solve: Empty matrix or error state (Expected) [PASS]

[D6.6] band_solve() - Error Handling - Invalid Bandwidth
[Error] band_solve: bandwidth k must be >= 1 (got -1).
band_solve with k=-1: Empty matrix or error state (Expected) [PASS]

[D7: Roots Tests]

[D7.1] Solving a Simple 2x2 System Ax = b
Matrix A:
Matrix Elements >>>
           2            1       |
           1            3       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           5       |
           6       |
<<< Matrix Elements

Solution x:
Matrix Elements >>>
         1.8       |
         1.4       |
<<< Matrix Elements


[D7.2] Solving a 3x3 System Ax = b
Matrix A:
Matrix Elements >>>
           1            2            1       |
           2            0            3       |
           3            2            1       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           9       |
           8       |
           7       |
<<< Matrix Elements

Solution x:
Matrix Elements >>>
          -1       |
     3.33333       |
     3.33333       |
<<< Matrix Elements


[D7.3] Singular Matrix (No Unique Solution)
Matrix A (singular, linearly dependent rows):
Matrix Elements >>>
           1            2       |
           2            4       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           5       |
           6       |
<<< Matrix Elements

[Error] solve: zero or near-zero pivot at (1, 1), system is singular or rank-deficient.
Solution x:
[Error] Cannot print matrix: data pointer is null.

[D7.4] Incompatible Dimensions (Expect Error)
Matrix A (3x3):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

Vector b (2x1, incompatible):
Matrix Elements >>>
          10       |
          11       |
<<< Matrix Elements

[Error] solve: dimensions do not match (A: 3x3, b: 2x1, expected b: 3x1).
Dimension mismatch roots: Empty matrix or error state (Expected) [PASS]

[D7.5] roots() - Boundary Case - Empty Matrix
[Error] solve: matrix A data pointer is null.
Empty system roots: Empty matrix or error state (Expected) [PASS]
============ [tiny_matrix_test end] ============
```
