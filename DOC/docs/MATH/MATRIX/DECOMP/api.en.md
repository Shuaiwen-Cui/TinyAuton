# Decomposition and solving

APIs follow the MATH project’s `middleware/tiny_math/mat/tiny_matrix.hpp`. Elements are `float`; check dimensions, layout and result status before use.

[Overview](../OVERVIEW/api.md) · [Historical tests](../TESTS/overview.md)

## MATRIX PROPERTIES & DECOMPOSITIONS {#matrix-properties-decompositions}

!!! INFO "Matrix Decompositions Overview"
    Matrix decompositions are fundamental tools in numerical linear algebra. They break down a matrix into simpler components that reveal its structure and enable efficient computations. Different decompositions are suited for different types of matrices and applications.

### Matrix Property Checks {#matrix-property-checks}

#### Check Symmetry {#check-symmetry}

```cpp
bool Mat::is_symmetric(float tolerance = 1e-6f) const;
```

**Description**:

Check whether a matrix is symmetric within the given tolerance. A matrix A is symmetric if A = A^T, i.e., A(i,j) = A(j,i) for all i, j.

**Mathematical Principle**:

For a symmetric matrix, all eigenvalues are real, and eigenvectors can be chosen to be orthogonal. Symmetric matrices are fundamental in many applications, especially in structural dynamics and optimization.

**Parameters**:

- `float tolerance` : Maximum allowed difference |A(i,j) - A(j,i)| (default: 1e-6).

**Returns**:

`bool` - `true` if approximately symmetric, `false` otherwise.

**Usage Insights**:

- **Eigendecomposition**: Symmetric matrices can use more efficient and stable eigendecomposition methods (e.g., Jacobi method).

- **Cholesky Decomposition**: Only symmetric positive definite matrices can be decomposed using Cholesky decomposition.

- **Structural Dynamics**: Stiffness and mass matrices in structural analysis are typically symmetric.

#### Check Positive Definiteness {#check-positive-definiteness}

```cpp
bool Mat::is_positive_definite(float tolerance = 1e-6f, int max_minors_to_check = -1) const;
```

**Description**:

Check if a matrix is positive definite using Sylvester's criterion. A symmetric matrix A is positive definite if x^T A x > 0 for all non-zero vectors x, or equivalently, all eigenvalues are positive.

**Mathematical Principle**:

Sylvester's criterion states that a symmetric matrix is positive definite if and only if all leading principal minors are positive. The function checks the first few leading minors and diagonal elements for efficiency.

**Parameters**:

- `float tolerance` : Tolerance for numerical checks (default: 1e-6).
- `int max_minors_to_check` : Number of leading principal minors to check. `-1` checks all; `>0` checks only the first N.

**Returns**:

`bool` - `true` if matrix is positive definite, `false` otherwise.

**Usage Insights**:

- **Cholesky Decomposition**: Positive definite matrices can be decomposed using Cholesky decomposition, which is faster and more stable than LU decomposition.

- **Optimization**: Positive definite Hessian matrices indicate local minima in optimization problems.

- **Stability Analysis**: In control systems, positive definiteness of certain matrices ensures system stability.

### Matrix Decomposition Structures {#matrix-decomposition-structures}

#### LU Decomposition Structure {#lu-decomposition-structure}

```cpp
struct Mat::LUDecomposition
{
    Mat L;                 // Lower triangular matrix (with unit diagonal)
    Mat U;                 // Upper triangular matrix
    Mat P;                 // Permutation matrix (if pivoting used)
    bool pivoted;          // Whether pivoting was used
    tiny_error_t status;   // Computation status
    
    LUDecomposition();
};
```

**Description**:

Container for LU decomposition results. The decomposition A = P * L * U (with pivoting) or A = L * U (without pivoting), where L is lower triangular with unit diagonal, U is upper triangular, and P is a permutation matrix.

**Mathematical Principle**:

LU decomposition factors a matrix into lower and upper triangular matrices, enabling efficient solution of linear systems. With pivoting, it handles near-singular matrices better.

#### Cholesky Decomposition Structure {#cholesky-decomposition-structure}

```cpp
struct Mat::CholeskyDecomposition
{
    Mat L;                 // Lower triangular matrix
    tiny_error_t status;   // Computation status
    
    CholeskyDecomposition();
};
```

**Description**:

Container for Cholesky decomposition results. For symmetric positive definite matrices, A = L * L^T, where L is lower triangular.

**Mathematical Principle**:

Cholesky decomposition is a specialized LU decomposition for symmetric positive definite matrices. It requires only half the storage and computation of LU decomposition.

#### QR Decomposition Structure {#qr-decomposition-structure}

```cpp
struct Mat::QRDecomposition
{
    Mat Q;                 // Orthogonal matrix (Q^T * Q = I)
    Mat R;                 // Upper triangular matrix
    tiny_error_t status;   // Computation status
    
    QRDecomposition();
};
```

**Description**:

Container for QR decomposition results. A = Q * R, where Q is orthogonal (Q^T * Q = I) and R is upper triangular.

**Mathematical Principle**:

QR decomposition expresses a matrix as the product of an orthogonal matrix and an upper triangular matrix. It's numerically stable and fundamental for least squares problems.

#### SVD Decomposition Structure {#svd-decomposition-structure}

```cpp
struct Mat::SVDDecomposition
{
    Mat U;                 // Left singular vectors (orthogonal matrix)
    Mat S;                 // Singular values (diagonal matrix or vector)
    Mat V;                 // Right singular vectors (orthogonal matrix, V^T)
    int rank;              // Numerical rank of the matrix
    int iterations;        // Number of iterations performed
    tiny_error_t status;   // Computation status
    
    SVDDecomposition();
};
```

**Description**:

Container for SVD decomposition results. A = U * S * V^T, where U and V are orthogonal matrices, and S contains singular values on the diagonal.

**Mathematical Principle**:

SVD is the most general matrix decomposition. The singular values reveal the matrix's rank, condition number, and enable computation of pseudo-inverse for rank-deficient matrices.

### Matrix Decomposition Methods {#matrix-decomposition-methods}

#### LU Decomposition {#lu-decomposition}

```cpp
Mat::LUDecomposition Mat::lu_decompose(bool use_pivoting = true) const;
```

**Description**:

Compute LU decomposition: A = P * L * U (with pivoting) or A = L * U (without pivoting). Efficient for solving multiple systems with the same coefficient matrix.

**Mathematical Principle**: 

- **Without pivoting**: A = L * U, where L has unit diagonal

- **With pivoting**: P * A = L * U, where P is a permutation matrix

The decomposition enables solving Ax = b by solving Ly = Pb (forward substitution) then Ux = y (back substitution).

**Parameters**:

- `bool use_pivoting` : Whether to use partial pivoting for numerical stability (default: true).

**Returns**:

`LUDecomposition` containing L, U, P matrices and status.

**Usage Insights**:

- **Multiple RHS**: Once decomposed, solve multiple systems with different right-hand sides efficiently using `solve_lu()`.

- **Determinant**: det(A) = det(P) * det(L) * det(U) = det(P) * det(U) (since det(L) = 1).

- **Inverse**: Can compute A^(-1) by solving LUx = eᵢ for each unit vector eᵢ.

- **Performance**: O(n³) for decomposition, O(n²) for each solve after decomposition.

**Pitfalls**:

- **Pivoting is not optional in practice**: Without pivoting, a perfectly valid matrix can still produce catastrophic round-off error.

- **L is unit-diagonal**: Do not expect diagonal values in `L` to carry scale information; the scale lives in `U`.

- **Sign matters for determinants**: Every row permutation changes the determinant sign through `P`.

#### Cholesky Decomposition {#cholesky-decomposition}

```cpp
Mat::CholeskyDecomposition Mat::cholesky_decompose() const;
```

**Description**:

Compute Cholesky decomposition: A = L * L^T for symmetric positive definite matrices. Faster than LU for SPD matrices, commonly used in structural dynamics.

**Mathematical Principle**:

For a symmetric positive definite matrix A, there exists a unique lower triangular matrix L with positive diagonal elements such that A = L * L^T. This is essentially a specialized LU decomposition that takes advantage of symmetry.

**Parameters**:

None (matrix must be symmetric positive definite).

**Returns**:

`CholeskyDecomposition` containing L matrix and status.

**Usage Insights**:

- **Efficiency**: Requires approximately half the computation and storage of LU decomposition.

- **Stability**: More stable than LU for symmetric positive definite matrices.

- **Applications**: 

  - Structural dynamics: Mass and stiffness matrices are often SPD

  - Optimization: Hessian matrices in Newton's method

  - Statistics: Covariance matrices

- **Error Handling**: Returns error if matrix is not symmetric or not positive definite.

**Pitfalls**:

- **SPD only**: A matrix that is merely symmetric is not enough; positive definiteness is required.

- **Symmetry tolerance**: In floating-point code, tiny asymmetries can cause a failure unless the implementation tolerates them.

- **Best when structure is known**: If you already know the matrix is SPD, this is the right solver family; otherwise check first with `is_symmetric()` and `is_positive_definite()`.

#### QR Decomposition {#qr-decomposition}

```cpp
Mat::QRDecomposition Mat::qr_decompose() const;
```

**Description**:

Compute QR decomposition: A = Q * R, where Q is orthogonal and R is upper triangular. Numerically stable, used for least squares and orthogonalization.

**Mathematical Principle**:

QR decomposition expresses any matrix as the product of an orthogonal matrix Q (Q^T * Q = I) and an upper triangular matrix R. The decomposition is computed using the modified Gram-Schmidt process with re-orthogonalization.

**Parameters**:

None.

**Returns**:

`QRDecomposition` containing Q and R matrices and status.

**Usage Insights**:

- **Least Squares**: For overdetermined system Ax ≈ b, the solution minimizes ||Ax - b||₂ is x = R^(-1) * Q^T * b.

- **Numerical Stability**: QR decomposition is more stable than normal equations for least squares problems.

- **Eigendecomposition**: QR algorithm uses QR decomposition iteratively to find eigenvalues.

- **Rank Revealing**: The rank of A equals the number of non-zero diagonal elements of R.

**Pitfalls**:

- **Orthogonality can drift**: Gram-Schmidt based QR is convenient, but for badly conditioned matrices it is less robust than Householder QR.

- **Not a general inverse**: QR is excellent for least squares, but not the first choice for square exact solves unless you need its stability properties.

- **Rank deficiency**: Small diagonal values in `R` usually mean near-dependence. Treat them with tolerance, not exact comparisons.

#### SVD Decomposition {#svd-decomposition}

```cpp
Mat::SVDDecomposition Mat::svd_decompose(int max_iter = 100, float tolerance = 1e-6f) const;
```

**Description**:

Compute Singular Value Decomposition: A = U * S * V^T. Most general decomposition, used for rank estimation, pseudo-inverse, dimension reduction. Uses iterative method based on eigendecomposition.

**Mathematical Principle**:

SVD decomposes any m × n matrix A into:
- U: m × m orthogonal matrix (left singular vectors)

- S: m × n diagonal matrix (singular values σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0)

- V: n × n orthogonal matrix (right singular vectors)

The singular values reveal the matrix's fundamental properties: rank, condition number, and numerical behavior.

**Parameters**:

- `int max_iter` : Maximum number of iterations (default: 100).

- `float tolerance` : Convergence tolerance (default: 1e-6).

**Returns**:

`SVDDecomposition` containing U, S, V matrices, rank, and status.

**Usage Insights**:

- **Rank Estimation**: The numerical rank is the number of singular values above the tolerance threshold.

- **Pseudo-Inverse**: A⁺ = V * S⁺ * U^T, where S⁺ has 1/σᵢ for non-zero σᵢ.

- **Dimension Reduction**: Truncated SVD (keeping only largest singular values) provides low-rank approximation.

- **Condition Number**: κ(A) = σ₁ / σᵣ, where σᵣ is the smallest non-zero singular value.

- **Applications**: 

  - Least squares for rank-deficient systems

  - Principal Component Analysis (PCA)

  - Image compression

  - Noise reduction

**Pitfalls**:

- **Cost**: SVD is the most informative decomposition, but also one of the most expensive.

- **Tolerance defines rank**: Numerical rank is not absolute; it changes with the threshold you choose.

- **Memory footprint**: Full U and V storage can be heavy on embedded targets.

- **Interpretation**: Singular values tell you conditioning and energy concentration, but they do not by themselves solve a model-selection problem.

### Solving Linear Systems Using Decompositions {#solving-linear-systems-using-decompositions}

#### Solve using LU Decomposition {#solve-using-lu-decomposition}

```cpp
static Mat Mat::solve_lu(const LUDecomposition &lu, const Mat &b);
```

**Description**:

Solve linear system Ax = b using precomputed LU decomposition. More efficient than `solve()` when solving multiple systems with the same coefficient matrix.

**Mathematical Principle**:

Given A = P * L * U, solve Ax = b by:
1. Solve Ly = Pb (forward substitution)
2. Solve Ux = y (back substitution)

**Parameters**:

- `const LUDecomposition &lu` : Precomputed LU decomposition.

- `const Mat &b` : Right-hand side vector (N×1).

**Returns**:

Mat - Solution vector (N×1).

**Usage Insights**:

- **Multiple RHS**: After computing LU decomposition once, solve multiple systems efficiently.

- **Performance**: O(n²) per solve vs O(n³) for full solve, significant savings for multiple RHS.

- **Memory**: Reuses the decomposition, avoiding repeated computation.

**Pitfalls**:

- **Same matrix, many right-hand sides**: This is where LU shines. If A changes every time, the gain shrinks quickly.

- **Pivot information matters**: The permutation matrix is part of the solve; ignoring it gives the wrong answer.

#### Solve using Cholesky Decomposition {#solve-using-cholesky-decomposition}

```cpp
static Mat Mat::solve_cholesky(const CholeskyDecomposition &chol, const Mat &b);
```

**Description**:

Solve linear system Ax = b using precomputed Cholesky decomposition. More efficient than LU for symmetric positive definite matrices.

**Mathematical Principle**:

Given A = L * L^T, solve Ax = b by:
1. Solve Ly = b (forward substitution)
2. Solve L^T x = y (back substitution)

**Parameters**:

- `const CholeskyDecomposition &chol` : Precomputed Cholesky decomposition.

- `const Mat &b` : Right-hand side vector (N×1).

**Returns**:

Mat - Solution vector (N×1).

**Usage Insights**:

- **Efficiency**: Faster than LU for SPD matrices, both in decomposition and solving.

- **Stability**: More numerically stable for SPD matrices.

- **Applications**: Structural dynamics, optimization, statistics.

**Pitfalls**:

- **Only SPD**: If the matrix is not SPD, this method is not just inefficient; it is invalid.

- **Check before decomposing**: A quick symmetry/positive-definite check is cheaper than debugging a failed factorization later.

#### Solve using QR Decomposition (Least Squares) {#solve-using-qr-decomposition-least-squares}

```cpp
static Mat Mat::solve_qr(const QRDecomposition &qr, const Mat &b);
```

**Description**:

Solve linear system using QR decomposition. Provides least squares solution for overdetermined systems (more equations than unknowns).

**Mathematical Principle**:

For Ax ≈ b (overdetermined), the least squares solution minimizes ||Ax - b||₂. Using A = Q * R:
- x = R^(-1) * Q^T * b

This avoids the numerically unstable normal equations A^T * A * x = A^T * b.

**Parameters**:

- `const QRDecomposition &qr` : Precomputed QR decomposition.

- `const Mat &b` : Right-hand side vector (M×1, where M ≥ N).

**Returns**:

Mat - Least squares solution vector (N×1).

**Usage Insights**:

- **Overdetermined Systems**: Handles cases where there are more equations than unknowns.

- **Numerical Stability**: More stable than solving normal equations directly.

- **Applications**: 

  - Curve fitting

  - Data regression

  - Signal processing

**Pitfalls**:

- **Least squares, not exact solve**: If the system is inconsistent, QR gives the best fit, not a contradiction-free exact answer.

- **Still needs rank awareness**: Very small diagonal values in `R` mean the fit may be ill-conditioned or underdetermined.

### Pseudo-Inverse {#pseudo-inverse}

```cpp
static Mat Mat::pseudo_inverse(const SVDDecomposition &svd, float tolerance = 1e-6f);
```

**Description**:

Compute the Moore-Penrose pseudo-inverse A⁺ using SVD decomposition. Works for rank-deficient or non-square matrices where the regular inverse doesn't exist.

**Mathematical Principle**:

For A = U * S * V^T, the pseudo-inverse is A⁺ = V * S⁺ * U^T, where S⁺ has 1/σᵢ for singular values σᵢ > tolerance, and 0 otherwise.

**Properties of Pseudo-Inverse**:

- A * A⁺ * A = A

- A⁺ * A * A⁺ = A⁺

- (A * A⁺)^T = A * A⁺

- (A⁺ * A)^T = A⁺ * A

**Parameters**:

- `const SVDDecomposition &svd` : Precomputed SVD decomposition.

- `float tolerance` : Threshold for singular values (default: 1e-6). Singular values below this are treated as zero.

**Returns**:

Mat - Pseudo-inverse matrix.

**Usage Insights**:

- **Rank-Deficient Systems**: Provides solution for systems where A is not full rank.

- **Minimum Norm Solution**: For underdetermined systems, gives the solution with minimum ||x||₂.

- **Least Squares**: For overdetermined systems, gives the least squares solution.

- **Applications**:

  - Control systems

  - Signal processing

  - Machine learning (regularization)

**Pitfalls**:

- **Tolerance is a modeling choice**: The pseudo-inverse depends on the singular-value cutoff. Changing tolerance can change the effective solution.

- **Not always sparse-friendly**: A dense pseudo-inverse can destroy structure and inflate memory use.

- **Prefer solve when possible**: If the system is square and well-conditioned, direct solving is usually cheaper.
