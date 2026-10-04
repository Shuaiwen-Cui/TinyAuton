# Matrix operations and linear systems

APIs follow the MATH project’s `middleware/tiny_math/mat/tiny_matrix.hpp`. Elements are `float`; check dimensions, layout and result status before use.

[Overview](../OVERVIEW/api.md) · [Historical tests](../TESTS/overview.md)

## LINEAR ALGEBRA {#linear-algebra}

### Transpose {#transpose}

```cpp
Mat Mat::transpose();
```

**Description**:

Calculates the transpose of the matrix, returning a new matrix. The transpose A^T of a matrix A is obtained by interchanging rows and columns: (A^T)ᵢⱼ = Aⱼᵢ.

**Mathematical Principle**: 
- For any matrix A, (A^T)^T = A

- (A + B)^T = A^T + B^T

- (AB)^T = B^T * A^T

- For square matrices, det(A) = det(A^T)

**Parameters**:

None.

**Returns**:

Mat - Transposed matrix (col × row).

**Usage Insights**:

- **Memory Layout**: Creates a new matrix, so memory usage doubles temporarily. For large matrices, consider memory constraints.

- **Symmetric Matrices**: If A = A^T, the matrix is symmetric. Use `is_symmetric()` to check.

- **Applications**: 

  - Inner products: u^T * v

  - Quadratic forms: x^T * A * x

  - Matrix equations: A^T * A (normal equations)

### Minor matrix {#minor-matrix}

```cpp
Mat Mat::minor(int target_row, int target_col);
```

**Description**:

Calculates the minor matrix by removing the specified row and column. The minor is the submatrix obtained by removing one row and one column.

**Parameters**: 

- `int target_row`: Row index to remove.

- `int target_col`: Column index to remove.

**Returns**:

Mat - The (n-1)x(n-1) minor matrix.

### Cofactor matrix {#cofactor-matrix}

```cpp
Mat Mat::cofactor(int target_row, int target_col);
```

**Description**:

Calculates the cofactor matrix (same as minor matrix). The cofactor matrix is the same as the minor matrix. The sign (-1)^(i+j) is applied when computing the cofactor value, not to the matrix elements themselves.

**Parameters**: 

- `int target_row`: Row index to remove.

- `int target_col`: Column index to remove.

**Returns**:

Mat - The (n-1)x(n-1) cofactor matrix (same as minor matrix).

### Determinant (Auto-select Method) {#determinant-auto-select-method}

```cpp
float Mat::determinant();
```

**Description**: 

Computes the determinant of a square matrix, automatically selecting the optimal method based on matrix size. For small matrices (n ≤ 4), uses Laplace expansion; for larger matrices (n > 4), uses LU decomposition for better efficiency.

**Mathematical Principle**: 

The determinant is an important numerical characteristic of a square matrix with the following properties:

- det(AB) = det(A) * det(B)

- det(A^T) = det(A)

- det(A^(-1)) = 1 / det(A)

- If A is singular, det(A) = 0

**Method Selection**:

- **Small matrices (n ≤ 4)**: Uses `determinant_laplace()` - Laplace expansion method, time complexity O(n!), more accurate for small matrices

- **Large matrices (n > 4)**: Uses `determinant_lu()` - LU decomposition method, time complexity O(n³), more efficient

**Parameters**:

None.

**Returns**: 

float - The determinant value.

**Usage Insights**:

- **Automatic Selection**: For most applications, simply use `determinant()` and the function will automatically select the optimal method

- **Performance Optimization**: If you need to compute determinants of matrices of the same size multiple times, consider directly calling `determinant_lu()` or `determinant_gaussian()`

- **Precision Requirements**: For small matrices, `determinant_laplace()` may provide better numerical precision

### Determinant - Laplace Expansion {#determinant-laplace-expansion}

```cpp
float Mat::determinant_laplace();
```

**Description**: 

Computes the determinant of a square matrix using Laplace expansion (cofactor expansion). Time complexity is O(n!), suitable only for small matrices (n ≤ 4).

**Mathematical Principle**: 

Laplace expansion is the recursive definition of the determinant:

- For 1×1 matrix: det([a]) = a

- For 2×2 matrix: det([[a,b],[c,d]]) = ad - bc

- For n×n matrix: det(A) = Σⱼ₌₁ⁿ (-1)ⁱ⁺ʲ aᵢⱼ * det(Mᵢⱼ), where Mᵢⱼ is the minor matrix

This implementation uses first-row expansion, recursively computing the determinant of minors.

**Parameters**:

None.

**Returns**: 

float - The determinant value.

!!! warning "Performance Warning"
    Time complexity is O(n!), suitable only for small matrices (n ≤ 4). For large matrices, use `determinant_lu()` or `determinant_gaussian()`.

### Determinant - LU Decomposition {#determinant-lu-decomposition}

```cpp
float Mat::determinant_lu();
```

**Description**: 

Computes the determinant of a square matrix using LU decomposition. Time complexity is O(n³), suitable for large matrices.

**Mathematical Principle**: 

LU decomposition factorizes the matrix as A = P * L * U, where:

- P is a permutation matrix (if pivoting is used)

- L is a lower triangular matrix with unit diagonal

- U is an upper triangular matrix

Determinant formula: det(A) = det(P) * det(L) * det(U)

Where:

- det(P) = (-1)^(permutation signature), determined by the number of row swaps

- det(L) = 1 (since L has unit diagonal)

- det(U) = ∏ᵢ Uᵢᵢ (product of diagonal elements of U)

**Algorithm Steps**:

1. Perform LU decomposition (with pivoting for numerical stability)
2. Compute the determinant of the permutation matrix det(P)
3. Compute the product of diagonal elements of U: det(U)
4. Return det(P) * det(U)

**Parameters**:

None.

**Returns**: 

float - The determinant value. Returns 0.0 if the matrix is singular or near-singular.

**Usage Insights**:

- **Efficiency**: Much faster than Laplace expansion for matrices with n > 4

- **Numerical Stability**: Uses pivoting to improve numerical stability

- **Singular Matrices**: If the matrix is singular, LU decomposition fails and the function returns 0.0

### Determinant - Gaussian Elimination {#determinant-gaussian-elimination}

```cpp
float Mat::determinant_gaussian();
```

**Description**: 

Computes the determinant of a square matrix using Gaussian elimination. Time complexity is O(n³), suitable for large matrices.

**Mathematical Principle**: 

Gaussian elimination converts the matrix to upper triangular form, then computes the product of diagonal elements. The determinant value equals the product of diagonal elements of the upper triangular matrix, adjusted for the sign based on the number of row swaps.

**Algorithm Steps**:

1. Use partial pivoting Gaussian elimination to convert matrix to upper triangular form
2. Track the number of row swaps
3. Compute the product of diagonal elements of the upper triangular matrix
4. Adjust the sign based on row swaps: each row swap multiplies the determinant by -1

**Parameters**:

None.

**Returns**: 

float - The determinant value. Returns 0.0 if the matrix is singular.

**Usage Insights**:

- **Efficiency**: Time complexity O(n³) for large matrices, comparable to LU decomposition

- **Numerical Stability**: Uses partial pivoting to improve numerical stability

- **Implementation Simplicity**: More intuitive than LU decomposition, but less versatile (cannot be used for solving linear systems)

- **Applications**:

  - Check invertibility: det(A) ≠ 0 means A is invertible

  - Volume scaling: |det(A)| is the scaling factor of the linear transformation

  - System solvability: det(A) = 0 indicates singular system

### Adjoint {#adjoint}

```cpp
Mat Mat::adjoint();
```

**Description**:

Calculates the adjoint (adjugate) matrix of a square matrix.

**Parameters**:

None.

**Returns**:

Mat - Adjoint matrix.

### Normalize {#normalize}

```cpp
void Mat::normalize();
```

**Description**:

Normalizes the matrix using L2 norm (Frobenius norm). After normalization, ||Matrix|| = 1.

**Parameters**:

None.

**Returns**:

void

### Norm {#norm}

```cpp
float Mat::norm() const;
```

**Description**:

Calculates the Frobenius norm (also called Euclidean norm or L2 norm) of the matrix. The Frobenius norm is the square root of the sum of squares of all matrix elements.

**Mathematical Principle**: 
- Frobenius norm: ||A||_F = √(Σᵢ Σⱼ |aᵢⱼ|²) = √(trace(A^T * A))
- For vectors, this reduces to the standard L2 norm
- Properties:
  - ||A + B||_F ≤ ||A||_F + ||B||_F (triangle inequality)
  - ||AB||_F ≤ ||A||_F * ||B||_F
  - ||A||_F = ||A^T||_F

**Parameters**:

None.

**Returns**:

float - The computed matrix norm.

**Usage Insights**:

- **Error Measurement**: Useful for measuring the "size" of a matrix or error in numerical computations.

- **Normalization**: Used in `normalize()` to scale matrices to unit norm.

- **Convergence**: Often used as a convergence criterion in iterative algorithms.

- **Comparison**: For vectors, this is equivalent to the standard Euclidean norm ||v||₂.

### Inverse using Adjoint {#inverse-using-adjoint}

```cpp
Mat Mat::inverse_adjoint();
```

**Description**:

Computes the inverse of a square matrix using adjoint method. If the matrix is singular, returns a zero matrix.

**Parameters**:

None.

**Returns**:

Mat - The inverse matrix. If singular, returns a zero matrix.

### Identity Matrix {#identity-matrix}

```cpp
static Mat Mat::eye(int size);
```

**Description**:

Generates an identity matrix of given size.

**Parameters**: 

- `int size` : Dimension of the square identity matrix.

**Returns**:

Mat - Identity matrix (size x size).


### Augmentation Matrix (Horizontal Concatenation) {#augmentation-matrix-horizontal-concatenation}

```cpp
static Mat Mat::augment(const Mat &A, const Mat &B);
```

**Description**:

Creates an augmented matrix by horizontally concatenating two matrices [A | B]. The row counts of A and B must match.

**Parameters**:

- `const Mat &A` : Left matrix.

- `const Mat &B` : Right matrix.

**Returns**:

Mat - Augmented matrix [A B].

### Vertical Stack {#vertical-stack}

```cpp
static Mat Mat::vstack(const Mat &A, const Mat &B);
```

**Description**:

Vertically stacks two matrices [A; B]. The column counts of A and B must match.

**Parameters**:

- `const Mat &A` : Top matrix.

- `const Mat &B` : Bottom matrix.

**Returns**:

Mat - Vertically stacked matrix [A; B].

### Gram-Schmidt Orthogonalization {#gram-schmidt-orthogonalization}

```cpp
static bool Mat::gram_schmidt_orthogonalize(const Mat &vectors, Mat &orthogonal_vectors, 
                                            Mat &coefficients, float tolerance = 1e-6f);
```

**Description**:

Orthogonalizes a set of vectors using the Gram-Schmidt process. This is a general-purpose orthogonalization function that can be reused for QR decomposition and other applications requiring orthogonal bases. Uses the modified Gram-Schmidt algorithm with re-orthogonalization for improved numerical stability.

**Mathematical Principle**:

Given a set of vectors {v₁, v₂, ..., vₙ}, the Gram-Schmidt process produces an orthogonal set {q₁, q₂, ..., qₙ} where:

- q₁ = v₁ / ||v₁||

- qⱼ = (vⱼ - Σᵢ₌₁ʲ⁻¹⟨vⱼ, qᵢ⟩qᵢ) / ||vⱼ - Σᵢ₌₁ʲ⁻¹⟨vⱼ, qᵢ⟩qᵢ||

The modified version subtracts projections immediately, which improves numerical stability.

**Parameters**:

- `const Mat &vectors` : Input matrix where each column is a vector to be orthogonalized (m × n).

- `Mat &orthogonal_vectors` : Output matrix for orthogonalized vectors (m × n), each column is orthogonal and normalized.

- `Mat &coefficients` : Output matrix for projection coefficients (n × n, upper triangular), similar to R in QR decomposition.

- `float tolerance` : Minimum norm threshold for linear independence check (default: 1e-6).

**Returns**:

`bool` - `true` if successful, `false` if input is invalid.

**Usage Insights**:

- **Numerical Stability**: The implementation uses modified Gram-Schmidt with re-orthogonalization, which significantly improves stability for near-linearly-dependent vectors.

- **QR Decomposition**: This function is internally used by `qr_decompose()`. For QR decomposition, the coefficients matrix corresponds to the R matrix.

- **Basis Construction**: Useful for constructing orthogonal bases from a set of vectors, which is fundamental in many linear algebra applications.

- **Performance**: For large matrices, consider the computational cost. The complexity is O(mn²) for m-dimensional vectors and n vectors.

### All-Ones Matrix (Rectangular) {#all-ones-matrix-rectangular}

```cpp
static Mat Mat::ones(int rows, int cols);
```

**Description**:

Creates a matrix of specified size filled with ones.

**Parameters**:

- `int rows` : Number of rows.

- `int cols` : Number of columns.

**Returns**:

Mat - Matrix [rows x cols] with all elements = 1.

### All-Ones Matrix (Square) {#all-ones-matrix-square}

```cpp
static Mat Mat::ones(int size);
```

**Description**:

Creates a square matrix filled with ones of the specified size.

**Parameters**:

- `int size` : Size of the square matrix (rows = cols).

**Returns**:

Mat - Square matrix [size x size] with all elements = 1.


### Gaussian Elimination {#gaussian-elimination}

```cpp
Mat Mat::gaussian_eliminate() const;
```

**Description**:

Performs Gaussian Elimination to convert matrix to Row Echelon Form (REF). This is the first step in solving linear systems and computing matrix rank.

**Mathematical Principle**:

Gaussian elimination transforms a matrix into row echelon form through elementary row operations:

1. **Row swapping**: Exchange two rows

2. **Row scaling**: Multiply a row by a non-zero scalar

3. **Row addition**: Add a multiple of one row to another

**Row Echelon Form (REF) properties**:

- All zero rows are at the bottom

- The leading coefficient (pivot) of each non-zero row is to the right of the pivot in the row above

- All entries below a pivot are zero

**Parameters**:

None.

**Returns**:

Mat - The upper triangular matrix (REF form).

**Usage Insights**:

- **Linear System Solving**: First step in solving Ax = b. After REF, use back substitution.

- **Rank Computation**: The rank equals the number of non-zero rows in REF.

- **Determinant**: Can compute determinant from REF (product of diagonal elements, adjusted for row swaps).

- **Numerical Stability**: The implementation uses partial pivoting to improve numerical stability.

- **Performance**: O(n³) for n×n matrices. For multiple systems, prefer LU decomposition.

**Pitfalls**:

- **Near-zero pivots**: If a pivot is tiny, the elimination can amplify round-off error. Partial pivoting reduces, but does not eliminate, this risk.

- **Singular or rank-deficient input**: You may still obtain a visually valid REF with zero rows; that does not imply the system has a unique solution.

- **Determinant sign**: Each row swap flips the determinant sign. If you use REF to estimate det(A), track swaps carefully.

- **In-place mental model**: The algorithm mutates an equivalent working copy, not the mathematical object you started with. Do not assume the original row order survives.

### Row Reduce from Gaussian {#row-reduce-from-gaussian}

```cpp
Mat Mat::row_reduce_from_gaussian() const;
```

**Description**:

Converts a matrix (assumed in row echelon form) to Reduced Row Echelon Form (RREF).

**Mathematical Principle**:

This is the back-substitution phase of Gaussian elimination turned into a normalization pass. Each pivot row is scaled to make the pivot 1, then the entries above each pivot are eliminated so the pivot becomes the only non-zero element in its column.

**Usage Insights**:

- **Assumes REF first**: This function is best used after `gaussian_eliminate()`. Calling it on an arbitrary matrix can still work sometimes, but the result is not guaranteed to be meaningful.

- **Not a solver by itself**: RREF exposes solution structure, free variables, and inconsistency, but you still need to interpret the reduced form.

- **Numerical cleanup**: Very small values may survive as floating-point noise. Compare against a tolerance instead of expecting exact zeros.

**Pitfalls**:

- **Pivot normalization**: If a pivot is close to zero, dividing by it can blow up noise. This is one reason to start from a pivoted REF.

- **Exact vs approximate algebra**: Symbolic expectations like perfect zeros are too strict in floating-point arithmetic.

**Parameters**:

None.

**Returns**:

Mat - The matrix in RREF form.

### Inverse using Gaussian-Jordan Elimination {#inverse-using-gaussian-jordan-elimination}

```cpp
Mat Mat::inverse_gje() const;
```

**Description**:

Computes the inverse of a square matrix using Gauss-Jordan elimination.

**Mathematical Principle**:

The method forms the augmented matrix [A | I] and applies row operations until the left block becomes I. The right block then becomes A⁻¹. This is conceptually simple, but it is also one of the most numerically sensitive ways to invert a matrix.

**Usage Insights**:

- **Good for pedagogy and small matrices**: It makes the inverse construction explicit and easy to inspect.

- **Bad for repeated solves**: If you only need x from Ax = b, solve the system directly instead of computing A⁻¹.

- **Use conditioning awareness**: A matrix can be invertible but still numerically dangerous if it is ill-conditioned.

**Pitfalls**:

- **Square matrix required**: Non-square input should be treated as invalid.

- **Singular / near-singular matrices**: Exact singular matrices fail, but near-singular matrices may also produce unstable results.

- **Prefer factorization**: For repeated right-hand sides, LU or Cholesky is usually a better engineering choice.

**Parameters**:

None.

**Returns**:

Mat - The inverse matrix if invertible, otherwise returns empty matrix.

### Dot Product {#dot-product}

```cpp
float Mat::dotprod(const Mat &A, const Mat &B);
```

**Description**:

Calculates the dot product of two vectors (Nx1).

**Parameters**:

- `const Mat &A` : Input vector A (Nx1).

- `const Mat &B` : Input vector B (Nx1).

**Returns**:

float - The computed dot product value.

### Solve Linear System {#solve-linear-system}

```cpp
Mat Mat::solve(const Mat &A, const Mat &b) const;
```

**Description**:

Solves the linear system Ax = b using Gaussian elimination with back substitution. This is a direct method suitable for well-conditioned systems.

**Mathematical Principle**:

The method consists of two phases:

1. **Forward elimination**: Transform augmented matrix [A|b] to upper triangular form

2. **Back substitution**: Solve Ux = y from bottom to top

**Algorithm**:

- Create augmented matrix [A | b]

- Apply Gaussian elimination to get [U | y] where U is upper triangular

- Solve Ux = y using back substitution: xᵢ = (yᵢ - Σⱼ₌ᵢ₊₁ⁿ Uᵢⱼxⱼ) / Uᵢᵢ

**Parameters**:

- `const Mat &A` : Coefficient matrix (N×N), must be square and non-singular.

- `const Mat &b` : Right-hand side vector (N×1).

**Returns**:

Mat - Solution vector (N×1) containing the roots of the equation Ax = b. Returns empty matrix if system is singular or incompatible.

**Usage Insights**:

- **Single System**: Efficient for solving one system. For multiple systems with same A, use LU decomposition + `solve_lu()`.

- **Condition Number**: Performance degrades for ill-conditioned matrices. Check condition number if results are inaccurate.

- **Singular Systems**: Returns empty matrix if A is singular (det(A) = 0). Use SVD + pseudo-inverse for rank-deficient systems.

- **Performance**: O(n³) for elimination, O(n²) for back substitution. Total O(n³).

- **Alternative Methods**:

  - For SPD matrices: Use Cholesky decomposition + `solve_cholesky()` (faster)

  - For multiple RHS: Use LU decomposition + `solve_lu()` (more efficient)

  - For overdetermined: Use QR decomposition + `solve_qr()` (least squares)

**Pitfalls**:

- **Dimension compatibility**: `A` must be square and `b` must have matching rows. A mismatch usually means the model, not just the inputs, is wrong.

- **Hidden inconsistency**: If the augmented system contains contradictory rows, elimination will expose it only after the fact. Do not assume every linear system has a solution.

- **No inverse shortcut**: Forming A⁻¹ and multiplying by b is less stable and usually slower than direct solve.

### Band Solve {#band-solve}

```cpp
Mat Mat::band_solve(Mat A, Mat b, int k) const;
```

**Description**:

Solves the system of equations Ax = b using optimized Gaussian elimination for banded matrices.

**Mathematical Principle**:

This is still Gaussian elimination, but the elimination is restricted to the non-zero band around the diagonal. When the matrix really is banded, this avoids touching far-off-zero entries and reduces wasted work. The gain comes from exploiting structure, not from using a different algebraic method.

**Usage Insights**:

- **Best when the band is real**: If the matrix is only “roughly” banded and the off-band entries matter, the optimization is less useful.

- **Bandwidth matters**: Smaller `k` means less work. If `k` is large enough to cover most of the matrix, the benefit collapses toward ordinary elimination.

- **Structured problems**: Common in discretized differential equations, spline systems, and local-coupling models.

**Pitfalls**:

- **Wrong bandwidth**: Underestimating `k` can silently ignore influential couplings and give a wrong answer.

- **Singularity still applies**: A banded matrix can still be singular or ill-conditioned.

- **No magic stability**: The method is an optimization of elimination, not a numerical cure-all.

**Parameters**:

- `Mat A` : Coefficient matrix (NxN) - banded matrix.

- `Mat b` : Result vector (Nx1).

- `int k` : Bandwidth of the matrix (the width of the non-zero bands).

**Returns**:

Mat - Solution vector (Nx1) containing the roots of the equation Ax = b.



### Roots {#roots}

```cpp
Mat Mat::roots(Mat A, Mat y) const;
```

**Description**:

Solves the matrix using a different method. Another implementation of the 'solve' function, no difference in principle. This method solves the linear system A * x = y using Gaussian elimination.

**Insight**:

`roots()` is effectively a semantic alias for `solve()` with the same algebraic core. It is kept for compatibility and readability when the problem is framed as “finding roots” rather than “solving a linear system.”

**Pitfalls**:

- **Do not expect different numerics**: If `solve()` is unstable for a case, `roots()` will not rescue it.

- **Same preconditions**: `A` must still be square and nonsingular, and `y` must have matching rows.

**Parameters**:

- `Mat A` : Matrix [N]x[N] with input coefficients.

- `Mat y` : Vector [N]x[1] with result values.

**Returns**:

Mat - Matrix [N]x[1] with roots.
