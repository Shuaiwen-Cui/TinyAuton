# 矩阵分解

原始输出的运行日期、硬件和固件提交未记录；下方完整保留该阶段输出。标签数量不是独立测试用例数量。

[测试概览](overview.md) · [完整源码与记录](../tiny-matrix-test.md)

```text
============ [tiny_matrix_test start] ============

[Test Organization: Application-Oriented Logic]
  Foundation → Basic Ops → Properties → Linear Systems → Decompositions → Applications → Quality


[E1: Matrix Decomposition Tests]

[E1.1] is_positive_definite() - Basic Functionality

[E1.11] Positive Definite 3x3 Matrix
Matrix:
Matrix Elements >>>
           4            1            0       |
           1            3            0       |
           0            0            2       |
<<< Matrix Elements

Is positive definite: True (Expected: True) [PASS]

[E1.12] Non-Positive Definite Matrix
Matrix:
Matrix Elements >>>
           1            2       |
           2            1       |
<<< Matrix Elements

Is positive definite: False (Expected: False) [PASS]

[E1.13] max_minors_to_check Parameter Testing
max_minors_to_check = -1 (check all): True (Expected: True) [PASS]
max_minors_to_check = 3 (check first 3): True (Expected: True) [PASS]
[Error] is_positive_definite: max_minors_to_check must be > 0 or -1 (got 0).
max_minors_to_check = 0 (invalid): False (Expected: False) [PASS]

[E1.14] Parameter Validation - Negative Tolerance
[Error] is_positive_definite: tolerance must be >= 0 (got -1e-06).
tolerance = -1e-6 (invalid): False (Expected: False) [PASS]

[E1.15] Boundary Case - Empty Matrix (0x0)
[Error] is_positive_definite: matrix data pointer is null.
Empty matrix (0x0): False (Expected: False, empty matrix is invalid) [PASS]

[E1.16] Boundary Case - Invalid Dimensions
Non-square matrix (2x3): False (Expected: False) [PASS]

[E1.2] LU Decomposition

[E1.21] 3x3 Matrix - LU Decomposition with Pivoting
Matrix A:
Matrix Elements >>>
           2            1            1       |
           4            3            3       |
           2            1            2       |
<<< Matrix Elements


[Results]
Status: OK
L matrix (lower triangular):
Matrix Elements >>>
           1            0            0       |
         0.5            1            0       |
         0.5            1            1       |
<<< Matrix Elements

U matrix (upper triangular):
Matrix Elements >>>
           4            3            3       |
           0         -0.5         -0.5       |
           0            0            1       |
<<< Matrix Elements

P matrix (permutation):
Matrix Elements >>>
           0            1            0       |
           1            0            0       |
           0            0            1       |
<<< Matrix Elements


[Verification] P * A should equal L * U
Total difference: 0 [PASS]

[E1.22] Solve Linear System using LU Decomposition
System: A * x = b
A:
Matrix Elements >>>
           2            1            1       |
           4            3            3       |
           2            1            2       |
<<< Matrix Elements

b:
Matrix Elements >>>
           1       |
           2       |
           3       |
<<< Matrix Elements


[Results]
Solution x:
Matrix Elements >>>
         0.5       |
          -2       |
           2       |
<<< Matrix Elements

Verification error: 0 [PASS]

[E1.26] solve_lu() - Boundary Case - Empty Matrix
[Error] lu_decompose: matrix data pointer is null.
[Error] solve_lu: invalid LU decomposition (status=458759).
Empty system: x rows = 0 (Expected: 0 or error state) [PASS]

[E1.27] solve_lu() - Invalid LU Decomposition
[Error] solve_lu: invalid LU decomposition (status=258).
Invalid LU decomposition: x rows = 0 (Expected: 0 or error state) [PASS]

[E1.23] Boundary Case - Empty Matrix (0x0)
[Error] lu_decompose: matrix data pointer is null.
Empty matrix: Status = Error, L rows = 0 (Expected: Error status, L is 0x0 or error state) [PASS]

[E1.24] LU Decomposition without Pivoting
Status: OK
Pivoted: No (Expected) [PASS]
Verification (A = L * U): difference = 0 [PASS]

[E1.25] lu_decompose() - Error Handling - Non-Square Matrix
[Error] lu_decompose: requires a square matrix (got 2x3).
Non-square matrix (2x3): Status = Error (Expected) [PASS]

[E1.3] Cholesky Decomposition

[E1.31] SPD Matrix - Cholesky Decomposition
Matrix A (SPD):
Matrix Elements >>>
           4            2            0       |
           2            5            1       |
           0            1            3       |
<<< Matrix Elements


[Results]
Status: OK
L matrix (lower triangular):
Matrix Elements >>>
           2            0            0       |
           1            2            0       |
           0          0.5      1.65831       |
<<< Matrix Elements


[Verification] L * L^T should equal A
Total difference: 2.38419e-07 [PASS]

[E1.32] Solve Linear System using Cholesky Decomposition
Solution x:
Matrix Elements >>>
    0.272727       |
    0.454545       |
    0.181818       |
<<< Matrix Elements

Verification error: 0 [PASS]

[E1.35] solve_cholesky() - Boundary Case - Empty Matrix
[Error] cholesky_decompose: matrix data pointer is null
[Error] solve_cholesky: invalid decomposition (status=458759).
Empty system: x rows = 0 (Expected: 0 or error state) [PASS]

[E1.36] solve_cholesky() - Invalid Cholesky Decomposition
[Error] solve_cholesky: invalid decomposition (status=258).
Invalid Cholesky decomposition: x rows = 0 (Expected: 0 or error state) [PASS]

[E1.33] Boundary Case - Empty Matrix (0x0)
[Error] cholesky_decompose: matrix data pointer is null
Empty matrix: Status = Error, L rows = 0 (Expected: Error status, L is 0x0 or error state) [PASS]

[E1.34] Non-Symmetric Matrix (Should Fail)
[Error] cholesky_decompose: requires symmetric matrix
Non-symmetric matrix: Status = Error (Expected) [PASS]

[E1.37] solve_cholesky() - Error Handling - Dimension Mismatch
[Error] solve_cholesky: b must be 3x1 (got 4x1).
Dimension mismatch solve_cholesky: Empty matrix or error state (Expected) [PASS]

[E1.4] QR Decomposition

[E1.41] General 3x3 Matrix - QR Decomposition
Matrix A:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements


[Results]
Status: OK
Q matrix (orthogonal):
Matrix Elements >>>
    0.123091     0.904534     0.408248       |
    0.492366     0.301511    -0.816497       |
     0.86164    -0.301511     0.408248       |
<<< Matrix Elements

R matrix (upper triangular):
Matrix Elements >>>
     8.12404      9.60114      11.0782       |
           0     0.904534      1.80907       |
           0            0            0       |
<<< Matrix Elements


[Verification] Q * R should equal A
Total difference: 9.53674e-07 [PASS]
Q orthogonality error: 2.35673e-07 [PASS]

[E1.42] Least Squares Solution using QR Decomposition
Overdetermined system: A * x ≈ b
A:
Matrix Elements >>>
           1            1       |
           1            2       |
           1            3       |
<<< Matrix Elements

b:
Matrix Elements >>>
           2       |
           3       |
           4       |
<<< Matrix Elements


[Results]
Least squares solution x:
Matrix Elements >>>
           1       |
           1       |
<<< Matrix Elements

Residual norm ||A*x - b||: 4.12953e-07

[E1.46] solve_qr() - Boundary Case - Empty Matrix
[Error] qr_decompose: matrix data pointer is null.
[Error] solve_qr: invalid QR decomposition (status=458759).
Empty system: x rows = 0 (Expected: 0 or error state) [PASS]

[E1.47] solve_qr() - Invalid QR Decomposition
[Error] solve_qr: invalid QR decomposition (status=258).
Invalid QR decomposition: x rows = 0 (Expected: 0 or error state) [PASS]

[E1.43] Boundary Case - Empty Matrix (0x0)
[Error] qr_decompose: matrix data pointer is null.
Empty matrix: Status = Error, Q rows = 0 (Expected: Error status, Q is 0x0 or error state) [PASS]

[E1.44] Boundary Case - Zero Rows or Columns
[Error] qr_decompose: matrix data pointer is null.
Matrix with 0 rows (0x3): Status = Error [PASS]
[Error] qr_decompose: matrix data pointer is null.
Matrix with 0 cols (3x0): Status = Error [PASS]

[E1.45] solve_qr() - Error Handling - Dimension Mismatch
[Error] solve_qr: b must be 3x1 (got 4x1).
Dimension mismatch solve_qr: Empty matrix or error state (Expected) [PASS]

[E1.5] Singular Value Decomposition (SVD)

[E1.51] General 3x3 Matrix - SVD Decomposition
Matrix A:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements


[Results]
Status: OK
Singular values:
Matrix Elements >>>
     16.8481       |
     1.06837       |
           0       |
<<< Matrix Elements

Numerical rank: 2
Iterations: 7
Reconstruction error: 1.74046e-05 [PASS]

[E1.52] Pseudo-inverse using SVD
Matrix A (3x2):
Matrix Elements >>>
           1            2       |
           3            4       |
           5            6       |
<<< Matrix Elements


[Results]
Pseudo-inverse A^+ (2x3):
Matrix Elements >>>
    -1.33333    -0.333333     0.666666       |
     1.08333     0.333333    -0.416666       |
<<< Matrix Elements

Verification error (A * A^+ * A ≈ A): 5.48363e-06 [PASS]

[E1.57] pseudo_inverse() - Parameter Validation - tolerance < 0
[Error] pseudo_inverse: tolerance must be >= 0 (got -1e-06).
tolerance = -1e-6: A_plus rows = 0 (Expected: 0 or error state) [PASS]

[E1.58] pseudo_inverse() - Invalid SVD Decomposition
[Error] pseudo_inverse: invalid SVD decomposition (status: 258)
Invalid SVD decomposition: A_plus rows = 0 (Expected: 0 or error state) [PASS]

[E1.53] Parameter Validation - max_iter <= 0
[Error] svd_decompose: max_iter must be > 0 (got 0).
max_iter = 0: Status = Error (Expected) [PASS]
[Error] svd_decompose: max_iter must be > 0 (got -1).
max_iter = -1: Status = Error (Expected) [PASS]

[E1.54] Parameter Validation - tolerance < 0
[Error] svd_decompose: tolerance must be >= 0 (got -1e-06).
tolerance = -1e-6: Status = Error (Expected) [PASS]

[E1.55] Boundary Case - Empty Matrix (m=0 or n=0)
[Error] svd_decompose: matrix data pointer is null.
Matrix with 0 rows (0x3): Status = Error [PASS]
[Error] svd_decompose: matrix data pointer is null.
Matrix with 0 cols (3x0): Status = Error [PASS]

[E1.56] pseudo_inverse() - Error Handling - Invalid SVD Decomposition
[Error] pseudo_inverse: invalid SVD decomposition (status: 258)
Invalid SVD decomposition: A_plus rows = 0 (Expected: 0 or error state) [PASS]

[E1.59] SVD - Singular Values Descending Order
Status: OK
Singular values: 5 3 2 1 
Descending order check: [PASS]

[E1.510] SVD - V Orthogonality (V^T * V approx I)
Max |V^T*V - I| = 2.38419e-07 [PASS]

[E1.511] SVD - U Orthogonality (first rank columns)
Detected rank = 3
Max |U^T*U - I|_rank = 4.76837e-07 [PASS]

[E1.512] SVD - Wide Matrix (2x3)
Dimensions (U=2x2, S=2x1, V=3x3): [PASS]
Reconstruction error: 4.29153e-06 [PASS]

[E1.513] SVD - Rank-Deficient Detection
Status: OK
S = [16.8481, 1.06837, 0]  (last expected ~0)
Detected rank = 2 (expected 2) [PASS]

[E1.6] Matrix Decomposition Performance Tests

[E1.61] LU Decomposition Performance
[Performance] LU Decomposition (4x4 matrix): 117.00 us

[E1.62] Cholesky Decomposition Performance
[Performance] Cholesky Decomposition (4x4 SPD matrix): 70.00 us

[E1.63] QR Decomposition Performance
[Performance] QR Decomposition (4x4 matrix): 163.00 us

[E1.64] SVD Decomposition Performance
[Performance] SVD Decomposition (4x4 matrix): 323.00 us

[Matrix Decomposition Tests Complete]

[E2: Gram-Schmidt Orthogonalization Tests]

[E2.1] Basic Orthogonalization - Linearly Independent Vectors
Input vectors (each column is a vector):
Matrix Elements >>>
        1.00         1.00         0.00       |
        0.00         1.00         1.00       |
        1.00         0.00         1.00       |
<<< Matrix Elements


[Results]
Status: OK
Orthogonalized vectors Q (each column is orthogonal):
Matrix Elements >>>
        0.71         0.41        -0.58       |
        0.00         0.82         0.58       |
        0.71        -0.41         0.58       |
<<< Matrix Elements

Coefficients R (upper triangular):
Matrix Elements >>>
        1.41         0.71         0.71       |
        0.00         1.22         0.41       |
        0.00         0.00         1.15       |
<<< Matrix Elements


[Verification] Q^T * Q should be identity
Orthogonality error: 0.00 [PASS]

[Verification] Each column of Q should be normalized
  Column 0 norm: 1.00 (error: 0.00) [PASS]
  Column 1 norm: 1.00 (error: 0.00) [PASS]
  Column 2 norm: 1.00 (error: 0.00) [PASS]

[Verification] Q * R should reconstruct original vectors
Reconstruction error: 0.00 [PASS]

[E2.2] Orthogonalization - Near-Linear-Dependent Vectors
Input vectors (third vector is nearly linear dependent):
Matrix Elements >>>
        1.00         0.00         1.00       |
        0.00         1.00         1.00       |
        0.00         0.00         0.00       |
<<< Matrix Elements


[Results]
Status: OK
Orthogonalized vectors Q:
Matrix Elements >>>
        1.00         0.00         0.00       |
        0.00         1.00         0.00       |
        0.00         0.00         1.00       |
<<< Matrix Elements

Coefficients R:
Matrix Elements >>>
        1.00         0.00         1.00       |
        0.00         1.00         1.00       |
        0.00         0.00         0.00       |
<<< Matrix Elements


[Note] Third column norm: 1.00 (should be 0 if linearly dependent, or 1 if orthogonalized)

[E2.3] Orthogonalization - 2D Vectors (2x2)
Input vectors:
Matrix Elements >>>
        3.00         1.00       |
        1.00         2.00       |
<<< Matrix Elements


[Results]
Status: OK
Orthogonalized vectors Q:
Matrix Elements >>>
        0.95        -0.32       |
        0.32         0.95       |
<<< Matrix Elements

Coefficients R:
Matrix Elements >>>
        3.16         1.58       |
        0.00         1.58       |
<<< Matrix Elements


[Verification] Dot product of Q columns: 0.00 (should be ~0 for orthogonal) [PASS]

[E2.4] Error Handling - Invalid Input
[Error] gram_schmidt_orthogonalize: input matrix is null.
Empty matrix test: PASS (correctly rejected)

[E2.5] gram_schmidt_orthogonalize() - Parameter Validation - Negative Tolerance
[Error] gram_schmidt_orthogonalize: tolerance must be non-negative (got -1e-06)
tolerance = -1e-6: PASS (correctly rejected)

[E2.6] gram_schmidt_orthogonalize() - Boundary Case - Zero Rows
[Error] gram_schmidt_orthogonalize: input matrix is null.
Zero rows (0x2): PASS (correctly rejected)

[E2.7] gram_schmidt_orthogonalize() - Boundary Case - Zero Columns
[Error] gram_schmidt_orthogonalize: input matrix is null.
Zero columns (2x0): PASS (correctly rejected)
============ [tiny_matrix_test end] ============
```
