# Eigenvalue applications

Execution date, device and firmware commit are unrecorded. The complete stage output is preserved below. Label counts are not counts of independent test cases.

[Overview](overview.md) · [Full source and record](../tiny-matrix-test.md)

```text
============ [tiny_matrix_test start] ============

[Test Organization: Application-Oriented Logic]
  Foundation → Basic Ops → Properties → Linear Systems → Decompositions → Applications → Quality


[E3: Eigenvalue Decomposition Tests]

[E3.1] is_symmetric() - Basic Functionality
[E3.11] Symmetric 3x3 Matrix
Matrix:
Matrix Elements >>>
           4            1            2       |
           1            3            0       |
           2            0            5       |
<<< Matrix Elements

Is symmetric: True (Expected: True)

[E3.12] Non-Symmetric 3x3 Matrix
Matrix:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

Is symmetric: False (Expected: False)

[E3.13] Non-Square Matrix (2x3)
Is symmetric: False (Expected: False)

[E3.14] Symmetric Matrix with Small Numerical Errors
Matrix with error 1e-05:
Matrix Elements >>>
           1      2.00001       |
           2            3       |
<<< Matrix Elements

Difference: |A(0,1) - A(1,0)| = 1.001358e-05 (Expected: 0.000010)
A(0,1) stored value: 2.00001001 (Expected: 2.00001001)
Is symmetric (tolerance=1e-4): True (Expected: True, tolerance > error) [PASS]
Is symmetric (tolerance=1e-6): False (Expected: False, tolerance < error) [PASS]
Difference accuracy: |actual_diff - expected_diff| = 1.36e-08 [PASS - difference stored correctly]

[E3.15] Parameter Validation - Negative Tolerance
[Error] is_symmetric: tolerance must be >= 0 (got -1e-06).
tolerance = -1e-6 (invalid): False (Expected: False) [PASS]

[E3.16] Boundary Case - Empty Matrix (0x0)
Empty matrix (0x0): True (Expected: True, 0x0 is vacuously symmetric) [PASS]

[E3.2] power_iteration() - Dominant Eigenvalue

[E3.21] Simple 2x2 Matrix
Matrix:
Matrix Elements >>>
        2.00         1.00       |
        1.00         2.00       |
<<< Matrix Elements


[Expected Results]
  Expected eigenvalues: 3.0 (largest), 1.0 (smallest)
  Expected dominant eigenvector (for λ=3): approximately [0.707, 0.707] or [-0.707, -0.707] (normalized)
  Expected dominant eigenvector (for λ=1): approximately [0.707, -0.707] or [-0.707, 0.707] (normalized)

[Actual Results]
  Dominant eigenvalue: 3.00 (Expected: 3.0, largest eigenvalue)
  Iterations: 2
  Status: OK
  Dominant eigenvector:
Matrix Elements >>>
        0.71       |
        0.71       |
<<< Matrix Elements

  Error from expected (3.0): 0.00 [PASS]

[E3.22] 3x3 Stiffness Matrix (SHM Application)
Stiffness Matrix:
Matrix Elements >>>
        2.00        -1.00         0.00       |
       -1.00         2.00        -1.00       |
        0.00        -1.00         2.00       |
<<< Matrix Elements


[Expected Results]
  Expected eigenvalues (approximate): 3.414 (largest), 2.000, 0.586 (smallest)
  Expected primary frequency: sqrt(3.414) ≈ 1.848 rad/s

[Actual Results]
  Dominant eigenvalue (primary frequency squared): 3.41
  Primary frequency: 1.85 rad/s (Expected: ~1.848 rad/s)
  Iterations: 8
  Status: OK
  Error from expected (3.41): 0.00 [PASS]

[E3.23] Non-Square Matrix (Expect Error)
[Error] power_iteration: requires square matrix (got 2x3).
Status: Error (Expected)

[E3.25] Parameter Validation - max_iter <= 0
[Error] power_iteration: max_iter must be > 0 (got 0).
max_iter = 0: Status = Error (Expected) [PASS]
[Error] power_iteration: max_iter must be > 0 (got -1).
max_iter = -1: Status = Error (Expected) [PASS]

[E3.26] Parameter Validation - tolerance < 0
[Error] power_iteration: tolerance must be >= 0 (got -1e-06).
tolerance = -1e-6: Status = Error (Expected) [PASS]

[E3.27] Boundary Case - Empty Matrix (0x0)
[Error] power_iteration: matrix data pointer is null.
Empty matrix: Status = Error, eigenvalue = 0.00 (Expected: Error status) [PASS]

[E3.24] inverse_power_iteration() - Smallest Eigenvalue (System Identification)

[E3.28] Simple 2x2 Matrix - Smallest Eigenvalue
Matrix (same as E3.21):
Matrix Elements >>>
        2.00         1.00       |
        1.00         2.00       |
<<< Matrix Elements


[Expected Results]
  Expected eigenvalues: 3.0 (largest), 1.0 (smallest)
  Expected smallest eigenvalue: 1.0
  Expected smallest eigenvector (for λ=1): approximately [0.707, -0.707] or [-0.707, 0.707] (normalized)
  Note: This is critical for system identification - smallest eigenvalue = fundamental frequency

[Actual Results]
  Smallest eigenvalue: 1.00 (Expected: 1.0, smallest eigenvalue)
  Iterations: 5
  Status: OK
  Smallest eigenvector:
Matrix Elements >>>
        0.71       |
       -0.71       |
<<< Matrix Elements

  Error from expected (1.0): 0.00 [PASS]

[Comparison] Power vs Inverse Power Iteration:
  Power iteration (λ_max): 3.00
  Inverse power iteration (λ_min): 1.00
  Ratio (λ_max/λ_min): 3.00 (Expected: ~3.0) [PASS]

[E3.29] 3x3 Stiffness Matrix - Smallest Eigenvalue (SHM Application)
Stiffness Matrix (same as E3.22):
Matrix Elements >>>
        2.00        -1.00         0.00       |
       -1.00         2.00        -1.00       |
        0.00        -1.00         2.00       |
<<< Matrix Elements


[Expected Results]
  Expected eigenvalues (approximate): 3.414 (largest), 2.000, 0.586 (smallest)
  Expected smallest eigenvalue: ~0.586 (fundamental frequency squared)
  Expected fundamental frequency: sqrt(0.586) ≈ 0.765 rad/s
  Note: Smallest eigenvalue is critical for system identification - represents fundamental mode

[Actual Results]
  Smallest eigenvalue (fundamental frequency squared): 0.59
  Fundamental frequency: 0.77 rad/s (Expected: ~0.765 rad/s)
  Iterations: 7
  Status: OK
  Smallest eigenvector (fundamental mode shape):
Matrix Elements >>>
        0.50       |
        0.71       |
        0.50       |
<<< Matrix Elements

  Error from expected (0.59): 0.00 [PASS]

[Comparison] Power vs Inverse Power Iteration for SHM:
  Power iteration (primary frequency²): 3.41 → frequency: 1.85 rad/s
  Inverse power iteration (fundamental frequency²): 0.59 → frequency: 0.77 rad/s
  Frequency ratio: 2.41 (Expected: ~2.4, ratio of highest to lowest mode)

[E3.210] Non-Square Matrix (Expect Error)
[Error] inverse_power_iteration: requires square matrix (got 2x3).
Status: Error (Expected)
Error handling: [PASS]

[E3.211] Near-Singular Matrix (Edge Case)
Matrix (near-singular but invertible):
Matrix Elements >>>
        1.00         0.00         0.00       |
        0.00         1.00         0.00       |
        0.00         0.00         1.00       |
<<< Matrix Elements


[Results]
  Status: OK
  Smallest eigenvalue: 1.00
  Iterations: 2
  Note: Successfully handled near-singular matrix [PASS]

[E3.212] Parameter Validation - max_iter <= 0
[Error] inverse_power_iteration: max_iter must be > 0 (got 0).
max_iter = 0: Status = Error (Expected) [PASS]
[Error] inverse_power_iteration: max_iter must be > 0 (got -1).
max_iter = -1: Status = Error (Expected) [PASS]

[E3.213] Parameter Validation - tolerance < 0
[Error] inverse_power_iteration: tolerance must be >= 0 (got -1e-06).
tolerance = -1e-6: Status = Error (Expected) [PASS]

[E3.214] Boundary Case - Empty Matrix (0x0)
[Error] inverse_power_iteration: matrix data pointer is null.
Empty matrix: Status = Error, eigenvalue = 0.00 (Expected: Error status) [PASS]

[E3.215] Singular Matrix (Should Fail)
[Error] lu_decompose: matrix is singular or near-singular at column 1 (pivot = 0).
[Error] inverse_power_iteration: matrix is singular or near-singular (LU status=458754).
Singular matrix: Status = Error (Expected: Error or OK with eigenvalue ≈ 0) [PASS]

[E3.3] eigendecompose_jacobi() - Symmetric Matrix Decomposition

[E3.31] 2x2 Symmetric Matrix - Complete Decomposition
[Expected Results]
  Expected eigenvalues: 3.0, 1.0 (in any order)
  Expected eigenvectors (for λ=3): [0.707, 0.707] or [-0.707, -0.707] (normalized)
  Expected eigenvectors (for λ=1): [0.707, -0.707] or [-0.707, 0.707] (normalized)

[Actual Results]
Eigenvalues:
Matrix Elements >>>
        1.00       |
        3.00       |
<<< Matrix Elements

Eigenvectors (each column is an eigenvector):
Matrix Elements >>>
        0.71         0.71       |
       -0.71         0.71       |
<<< Matrix Elements

Iterations: 2
Status: OK
Eigenvalue check (should be 3.0 and 1.0): [PASS]

[Verification] Check A * v = lambda * v for first eigenvector:
A * v:
Matrix Elements >>>
        0.71       |
       -0.71       |
<<< Matrix Elements

lambda * v:
Matrix Elements >>>
        0.71       |
       -0.71       |
<<< Matrix Elements

Verification (A*v = λ*v): [PASS]

[E3.32] 3x3 Stiffness Matrix (SHM Application)
[Expected Results]
  Expected eigenvalues (approximate): 3.414, 2.000, 0.586
  Expected natural frequencies: 1.848, 1.414, 0.765 rad/s
  Note: Eigenvalues may appear in any order

[Actual Results]
Eigenvalues (natural frequencies squared):
Matrix Elements >>>
        3.41       |
        0.59       |
        2.00       |
<<< Matrix Elements

Natural frequencies (rad/s):
  Mode 0: 1.85 rad/s (Expected: ~1.85 rad/s) [PASS]
  Mode 1: 0.77 rad/s (Expected: ~0.76 rad/s) [PASS]
  Mode 2: 1.41 rad/s (Expected: ~1.41 rad/s) [PASS]
Eigenvectors (mode shapes):
Matrix Elements >>>
        0.50         0.50        -0.71       |
       -0.71         0.71         0.00       |
        0.50         0.50         0.71       |
<<< Matrix Elements

Iterations: 9
Status: OK

[E3.33] Diagonal Matrix (Eigenvalues on diagonal)
Matrix:
Matrix Elements >>>
        5.00         0.00         0.00       |
        0.00         3.00         0.00       |
        0.00         0.00         1.00       |
<<< Matrix Elements


[Expected Results]
  Expected eigenvalues: 5.0, 3.0, 1.0 (diagonal elements, may be in any order)
  Expected eigenvectors: standard basis vectors [1,0,0], [0,1,0], [0,0,1] (or their negatives)
  Expected iterations: 1 (diagonal matrix should converge immediately)

[Actual Results]
Eigenvalues:
Matrix Elements >>>
        5.00       |
        3.00       |
        1.00       |
<<< Matrix Elements

Eigenvectors:
Matrix Elements >>>
        1.00         0.00         0.00       |
        0.00         1.00         0.00       |
        0.00         0.00         1.00       |
<<< Matrix Elements

Iterations: 1 (Expected: 1)
Eigenvalue check (should be 5.0, 3.0, 1.0): [PASS]

[E3.34] Parameter Validation - tolerance < 0
[Error] eigendecompose_jacobi: tolerance must be >= 0 (got -1e-06).
tolerance = -1e-6: Status = Error (Expected) [PASS]

[E3.35] Parameter Validation - max_iter <= 0
[Error] eigendecompose_jacobi: max_iter must be > 0 (got 0).
max_iter = 0: Status = Error (Expected) [PASS]
[Error] eigendecompose_jacobi: max_iter must be > 0 (got -1).
max_iter = -1: Status = Error (Expected) [PASS]

[E3.36] Boundary Case - Empty Matrix (0x0)
[Error] eigendecompose_jacobi: matrix data pointer is null.
Empty matrix: Status = Error, eigenvalues rows = 1 (Expected: Error status, eigenvalues is 0x0 or error state) [PASS]

[E3.4] eigendecompose_qr() - General Matrix Decomposition

[E3.41] General 2x2 Matrix
Matrix:
Matrix Elements >>>
        1.00         2.00       |
        3.00         4.00       |
<<< Matrix Elements


[Expected Results]
  Expected eigenvalues: (5+√33)/2 ≈ 5.372, (5-√33)/2 ≈ -0.372
  Note: This is a non-symmetric matrix, eigenvalues are real but may have complex eigenvectors

[Actual Results]
Eigenvalues:
Matrix Elements >>>
        5.37       |
       -0.37       |
<<< Matrix Elements

Eigenvectors:
Matrix Elements >>>
        0.42         0.91       |
        0.91        -0.42       |
<<< Matrix Elements

Iterations: 5
Status: OK
Eigenvalue 1: 5.37 (Expected: 5.37, Error: 0.00, Rel Error: 0.01%) [PASS]
Eigenvalue 2: -0.37 (Expected: -0.37, Error: 0.00, Rel Error: 0.07%) [PASS]
Overall eigenvalue check: [PASS]

[E3.42] Non-Symmetric 3x3 Matrix
Matrix [1,2,3; 4,5,6; 7,8,9]:
Matrix Elements >>>
        1.00         2.00         3.00       |
        4.00         5.00         6.00       |
        7.00         8.00         9.00       |
<<< Matrix Elements


[Expected Results]
  Expected eigenvalues (theoretical): 16.12, -1.12, 0.00
  Note: This matrix is rank-deficient (determinant = 0), so one eigenvalue is 0
  Note: QR algorithm may have numerical errors, especially for non-symmetric matrices
  Acceptable range: largest eigenvalue ~15-18, smallest eigenvalue near 0

[Actual Results]
Eigenvalues:
Matrix Elements >>>
       16.12       |
       -1.12       |
        0.00       |
<<< Matrix Elements

Eigenvectors:
Matrix Elements >>>
        0.23         0.88         0.41       |
        0.53         0.24        -0.82       |
        0.82        -0.40         0.41       |
<<< Matrix Elements

Iterations: 5
Status: OK
Eigenvalue 0: 16.12 (Expected: 16.12, Error: 0.00, Rel Error: 0.02%) [PASS]
Eigenvalue 1: -1.12 (Expected: -1.12, Error: 0.00, Rel Error: 0.28%) [PASS]
Eigenvalue 2: 0.00 (Expected: 0.00, Error: 0.00, Rel Error: 0.00%) [PASS]
Overall eigenvalue check: [PASS]

[E3.43] Parameter Validation - max_iter <= 0
[Error] eigendecompose_qr: max_iter must be > 0 (got 0).
max_iter = 0: Status = Error (Expected) [PASS]
[Error] eigendecompose_qr: max_iter must be > 0 (got -1).
max_iter = -1: Status = Error (Expected) [PASS]

[E3.44] Parameter Validation - tolerance < 0
[Error] eigendecompose_qr: tolerance must be >= 0 (got -1e-06).
tolerance = -1e-6: Status = Error (Expected) [PASS]

[E3.45] Boundary Case - Empty Matrix (0x0)
[Error] eigendecompose_qr: matrix data pointer is null.
Empty matrix: Status = Error, eigenvalues rows = 1 (Expected: Error status, eigenvalues is 0x0 or error state) [PASS]

[E3.5] eigendecompose() - Automatic Method Selection

[E3.51] Symmetric Matrix (Auto-select: Jacobi)
Matrix:
Matrix Elements >>>
        4.00         1.00         2.00       |
        1.00         3.00         0.00       |
        2.00         0.00         5.00       |
<<< Matrix Elements


[Expected Results]
  Method: Should automatically use Jacobi (symmetric matrix detected)
  Expected eigenvalues (approximate): 6.67, 3.48, 1.85
  Note: Eigenvalues may appear in any order

[Actual Results]
Eigenvalues:
Matrix Elements >>>
        1.85       |
        3.48       |
        6.67       |
<<< Matrix Elements

Iterations: 8
Status: OK
Method used: Jacobi (auto-selected for symmetric matrix)

[E3.52] Non-Symmetric Matrix (Auto-select: QR)
[Expected Results]
  Method: Should automatically use QR (non-symmetric matrix detected)
  Expected eigenvalues (theoretical): 16.12, -1.12, 0.00
  Note: One eigenvalue should be near 0 (rank-deficient matrix)
  Note: QR algorithm may have numerical errors for non-symmetric matrices
  Acceptable: largest ~15-18, smallest near 0, one near -1 to -3

[Actual Results]
Eigenvalues:
Matrix Elements >>>
       16.12       |
       -1.12       |
        0.00       |
<<< Matrix Elements

Iterations: 5
Status: OK
Method used: QR (auto-selected for non-symmetric matrix)
Eigenvalue 0: 16.12 (Expected: 16.12, Error: 0.00, Rel Error: 0.02%) [PASS]
Eigenvalue 1: -1.12 (Expected: -1.12, Error: 0.00, Rel Error: 0.28%) [PASS]
Eigenvalue 2: 0.00 (Expected: 0.00, Error: 0.00, Rel Error: 0.00%) [PASS]
Overall eigenvalue check: [PASS]

[E3.53] Parameter Validation - tolerance < 0
[Error] eigendecompose: tolerance must be >= 0 (got -1e-06).
tolerance = -1e-6: Status = Error (Expected) [PASS]

[E3.531] Parameter Validation - max_iter <= 0
[Error] eigendecompose: max_iter must be > 0 (got 0).
max_iter = 0: Status = Error (Expected) [PASS]
[Error] eigendecompose: max_iter must be > 0 (got -10).
max_iter = -10: Status = Error (Expected) [PASS]

[E3.532] Custom max_iter Parameter Test
[Test 1] max_iter = 10 (may not converge)
  Status: OK (Converged)
  Iterations: 8
[Test 2] max_iter = 200 (should converge)
  Status: OK (Converged)
  Iterations: 8
  Eigenvalues:
Matrix Elements >>>
        1.85       |
        3.48       |
        6.67       |
<<< Matrix Elements

Custom max_iter test: [PASS]

[E3.54] eigendecompose() - Boundary Case - Empty Matrix (0x0)
[Error] eigendecompose: matrix data pointer is null.
Empty matrix: Status = Error, eigenvalues rows = 1 (Expected: Error status, eigenvalues is 0x0 or error state) [PASS]

[E3.55] eigendecompose() - Error Handling - Non-Square Matrix
[Error] eigendecompose_qr: requires square matrix (got 2x3).
Non-square matrix (2x3): Status = Error (Expected) [PASS]

[E3.6] SHM Application - Structural Dynamics Analysis

[E3.61] 4-DOF Mass-Spring System
Stiffness Matrix K:
Matrix Elements >>>
        2.00        -1.00         0.00         0.00       |
       -1.00         2.00        -1.00         0.00       |
        0.00        -1.00         2.00        -1.00       |
        0.00         0.00        -1.00         1.00       |
<<< Matrix Elements

Is symmetric: Yes

[Quick Analysis] Primary frequency using power_iteration():
[Expected Results]
  Expected primary eigenvalue: ~3.53 (largest eigenvalue)
  Expected primary frequency: sqrt(3.53) ≈ 1.88 rad/s

[Actual Results]
  Primary eigenvalue: 3.53 (Expected: ~3.53)
  Primary frequency: 1.88 rad/s (Expected: ~1.88 rad/s)
  Iterations: 13
  Error from expected: 0.00 [PASS]

[Complete Analysis] Full modal analysis using eigendecompose_jacobi():
[Expected Results]
  Expected eigenvalues (approximate): 3.53, 2.35, 1.00, 0.12
  Expected natural frequencies: 1.88, 1.53, 1.00, 0.35 rad/s
  Note: These are approximate values for the 4-DOF system

[Actual Results]
All eigenvalues (natural frequencies squared):
Matrix Elements >>>
        3.53       |
        1.00       |
        2.35       |
        0.12       |
<<< Matrix Elements

Natural frequencies (rad/s):
  Mode 0: 1.88 rad/s (Expected: ~1.88 rad/s) [PASS]
  Mode 1: 1.00 rad/s (Expected: ~1.00 rad/s) [PASS]
  Mode 2: 1.53 rad/s (Expected: ~1.53 rad/s) [PASS]
  Mode 3: 0.35 rad/s (Expected: ~0.35 rad/s) [PASS]
Mode shapes (eigenvectors):
Matrix Elements >>>
        0.43         0.58        -0.66         0.23       |
       -0.66         0.58         0.23         0.43       |
        0.58        -0.00         0.58         0.58       |
       -0.23        -0.58        -0.43         0.66       |
<<< Matrix Elements

Total iterations: 17

[E3.7] Edge Cases and Error Handling

[E3.71] 1x1 Matrix
Matrix: [5.0]
[Expected Results]
  Expected eigenvalue: 5.0 (the matrix element itself)
  Expected eigenvector: [1.0] (normalized)

[Actual Results]
Eigenvalue: 5.00 (Expected: 5.0)
Eigenvector:
Matrix Elements >>>
        1.00       |
<<< Matrix Elements

Error from expected: 0.00 [PASS]

[E3.72] Zero Matrix
[Error] power_iteration: matrix-vector product collapsed to zero.
Status: Error (Expected)

[E3.73] Identity Matrix
Matrix (3x3 Identity):
Matrix Elements >>>
        1.00         0.00         0.00       |
        0.00         1.00         0.00       |
        0.00         0.00         1.00       |
<<< Matrix Elements


[Expected Results]
  Expected eigenvalues: 1.0, 1.0, 1.0 (all eigenvalues are 1)
  Expected eigenvectors: Any orthonormal basis (e.g., standard basis vectors)
  Expected iterations: 1 (should converge immediately)

[Actual Results]
Eigenvalues (should all be 1.0):
Matrix Elements >>>
        1.00       |
        1.00       |
        1.00       |
<<< Matrix Elements

Eigenvectors:
Matrix Elements >>>
        1.00         0.00         0.00       |
        0.00         1.00         0.00       |
        0.00         0.00         1.00       |
<<< Matrix Elements

Iterations: 1 (Expected: 1)
All eigenvalues = 1.0: [PASS]

[E3.8] Performance Test for SHM Applications

[E3.81] Power Iteration Performance (Real-time SHM - Dominant Eigenvalue)
[Performance] Power Iteration (3x3 matrix): 91.00 us

[E3.82] Inverse Power Iteration Performance (System Identification - Smallest Eigenvalue)
[Performance] Inverse Power Iteration (3x3 matrix): 455.00 us

[E3.83] Jacobi Method Performance (Complete Eigendecomposition - Symmetric Matrices)
[Performance] Jacobi Decomposition (3x3 symmetric matrix): 150.00 us

[E3.84] QR Method Performance (Complete Eigendecomposition - General Matrices)
[Performance] QR Decomposition (3x3 general matrix): 492.00 us

[Eigenvalue Decomposition Tests Complete]
============ [tiny_matrix_test end] ============
```
