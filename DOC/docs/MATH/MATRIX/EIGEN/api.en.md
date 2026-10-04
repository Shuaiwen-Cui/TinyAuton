# Eigenvalues and eigenvectors

!!! info "Result scope"
    The header states that `eigendecompose_qr()` returns only real parts for complex eigenvalues. Check `status` and convergence criteria; it is not a complete complex-eigenvalue solver.

APIs follow the MATH project’s `middleware/tiny_math/mat/tiny_matrix.hpp`. Elements are `float`; check dimensions, layout and result status before use.

[Overview](../OVERVIEW/api.md) · [Historical tests](../TESTS/overview.md)

## LINEAR ALGEBRA - Eigenvalues & Eigenvectors {#linear-algebra-eigenvalues-eigenvectors}

### Struct: `Mat::EigenPair` {#struct-mateigenpair}

```cpp
Mat::EigenPair::EigenPair();
// fields:
// float eigenvalue;      // eigenvalue (largest-magnitude for power_iteration, smallest for inverse_power_iteration)
// Mat eigenvector;       // corresponding eigenvector (n x 1)
// int iterations;        // number of iterations (for iterative methods)
// tiny_error_t status;   // computation status (TINY_OK / error code)
```

**Description**:

Container for a single eigenvalue/eigenvector result and related metadata. Typically returned by `power_iteration` or `inverse_power_iteration`.

**Insight**:

The pair is intentionally minimal: one eigenvalue, one eigenvector, and convergence metadata. That makes it practical for iterative methods where you care about one mode, not the full spectrum.

### Struct: `Mat::EigenDecomposition` {#struct-mateigendecomposition}

```cpp
Mat::EigenDecomposition::EigenDecomposition();
// fields:
// Mat eigenvalues;    // n x 1 matrix storing eigenvalues
// Mat eigenvectors;   // n x n matrix, columns are eigenvectors
// int iterations;     // iterations used by the algorithm
// tiny_error_t status; // computation status
```

**Description**:

Container for a full eigendecomposition result (all eigenvalues and eigenvectors).

**Insight**:

This is a full spectral snapshot. It is valuable when the matrix is small enough that the cost and memory are acceptable, and when you need more than one mode or a basis change.

### Power Iteration (dominant eigenpair) {#power-iteration-dominant-eigenpair}

```cpp
Mat::EigenPair Mat::power_iteration(int max_iter, float tolerance) const;
```

**Description**:

Compute the dominant (largest-magnitude) eigenvalue and its eigenvector using the power iteration method. Fast method suitable for real-time SHM applications to quickly identify primary frequency.

**Mathematical Principle**:

Power iteration finds the eigenvalue with the largest absolute value by iteratively applying the matrix to a vector:

1. Start with random vector v₀

2. Iterate: vₖ₊₁ = A * vₖ / ||A * vₖ||

3. Eigenvalue estimate: λₖ = (vₖ^T * A * vₖ) / (vₖ^T * vₖ) (Rayleigh quotient)

**Convergence**:

The method converges to the dominant eigenvalue if:

- The dominant eigenvalue is unique (|λ₁| > |λ₂| ≥ ... ≥ |λₙ|)

- The initial vector has a non-zero component in the direction of the dominant eigenvector

**Parameters**:

- `int max_iter` : Maximum number of iterations (typical default: 1000).

- `float tolerance` : Convergence tolerance (e.g. 1e-6). Convergence is checked by |λₖ - λₖ₋₁| < tolerance * |λₖ|.

**Returns**:

`EigenPair` containing `eigenvalue`, `eigenvector`, `iterations`, and `status`.

**Usage Insights**:

- **Real-Time Applications**: Fast convergence for well-separated eigenvalues, suitable for real-time structural health monitoring.

- **Initialization**: The implementation uses a smart initialization strategy (sum of column absolute values) to avoid convergence to smaller eigenvalues.

- **Convergence Rate**: Convergence is linear with rate |λ₂|/|λ₁|. Slower when eigenvalues are close.

- **Limitations**: 

  - Only finds one eigenvalue-eigenvector pair

  - Requires |λ₁| > |λ₂| (dominant eigenvalue must be unique)

  - May converge slowly if eigenvalues are close

- **Applications**:

  - Principal component analysis (first principal component)

  - PageRank algorithm

  - Structural dynamics (fundamental frequency)

**Pitfalls**:

- **Only one mode**: It will not tell you the rest of the spectrum.

- **Dominance requirement**: If the largest-magnitude eigenvalue is not clearly separated, convergence slows or becomes ambiguous.

- **Scaling matters**: Poorly scaled matrices can slow convergence or worsen the eigenvector estimate.

### Inverse Power Iteration (smallest eigenpair) {#inverse-power-iteration-smallest-eigenpair}

```cpp
Mat::EigenPair Mat::inverse_power_iteration(int max_iter, float tolerance) const;
```

**Description**:

Compute the smallest (minimum magnitude) eigenvalue and its eigenvector using the inverse power iteration method. Critical for system identification - finds fundamental frequency/lowest mode in structural dynamics. This method is essential for SHM applications where the smallest eigenvalue corresponds to the fundamental frequency of the system.

**Mathematical Principle**:

Inverse power iteration applies power iteration to A^(-1), which has eigenvalues 1/λᵢ. Since 1/λₙ is the largest eigenvalue of A^(-1), the method converges to the smallest eigenvalue of A:

1. Start with vector v₀

2. Iterate: Solve A * yₖ = vₖ, then vₖ₊₁ = yₖ / ||yₖ||

3. Eigenvalue estimate: λₖ = (vₖ^T * A * vₖ) / (vₖ^T * vₖ) (Rayleigh quotient)

**Convergence**:

Converges to the smallest eigenvalue if:

- The smallest eigenvalue is unique (|λₙ| < |λₙ₋₁| ≤ ... ≤ |λ₁|)

- Matrix A is invertible (non-singular)

- Initial vector has component in direction of smallest eigenvector

**Parameters**:

- `int max_iter` : Maximum number of iterations (default: 1000).

- `float tolerance` : Convergence tolerance (default: 1e-6). Uses relative tolerance: |λₖ - λₖ₋₁| < tolerance * max(|λₖ|, 1.0).

**Returns**:

`EigenPair` containing the smallest eigenvalue, eigenvector, iterations, and status.

**Algorithm Steps**:

1. Initialize normalized eigenvector v (with alternating signs to avoid alignment with dominant eigenvector)
2. Iterate: Solve A * y = v (equivalent to y = A^(-1) * v) using `solve()`
3. Normalize y to get new v
4. Compute eigenvalue estimate using Rayleigh quotient: λ = (v^T * A * v) / (v^T * v)
5. Check convergence using relative tolerance

**Usage Insights**:

- **System Identification**: Essential for finding fundamental frequencies in structural dynamics, where the smallest eigenvalue corresponds to the lowest natural frequency.

- **Numerical Stability**: The implementation includes checks for singular matrices and handles near-singular cases gracefully.

- **Initialization Strategy**: Uses alternating sign pattern to avoid convergence to larger eigenvalues, ensuring convergence to the smallest eigenvalue.

- **Performance**: Each iteration requires solving a linear system (O(n³) for dense matrices), but typically converges in fewer iterations than power iteration.

- **Complementary to Power Iteration**: 

  - Power iteration: finds λ_max (highest frequency)

  - Inverse power iteration: finds λ_min (fundamental frequency)

  - Together they provide the frequency range of the system

- **Applications**:

  - Structural health monitoring (fundamental frequency detection)

  - Modal analysis (lowest mode shape)

  - System identification

  - Stability analysis (smallest eigenvalue indicates stability margin)

**Notes**:

- Requires a square matrix and non-null data pointer; returns an error status otherwise.

- The matrix must be invertible (non-singular) for this method to work. If the matrix is singular or near-singular, the method will fail gracefully.

- Inverse power iteration only returns the smallest eigenpair. For full spectrum, use eigendecomposition functions below.

- This method is complementary to power iteration: power iteration finds the largest eigenvalue, while inverse power iteration finds the smallest eigenvalue.

**Pitfalls**:

- **Requires invertibility**: Singular or near-singular matrices can make each iteration unstable or impossible.

- **Solve step dominates cost**: Each iteration calls a linear solver, so the method is only attractive when you need a single extremal eigenpair.

- **Smallest magnitude is not always smallest numeric value**: For signed spectra, “minimum magnitude” and “most negative” are different concepts.

### Jacobi Eigendecomposition (symmetric matrices) {#jacobi-eigendecomposition-symmetric-matrices}

```cpp
Mat::EigenDecomposition Mat::eigendecompose_jacobi(float tolerance, int max_iter) const;
```

**Description**:

Compute full eigendecomposition using the Jacobi method. Recommended for symmetric matrices (good accuracy and stability for structural dynamics applications). Robust and accurate, ideal for structural dynamics matrices in SHM.

**Mathematical Principle**:

The Jacobi method diagonalizes a symmetric matrix through a series of orthogonal similarity transformations (Givens rotations):

1. Find largest off-diagonal element aₚq

2. Compute rotation angle θ to zero this element

3. Apply rotation: A' = J^T * A * J, where J is the rotation matrix

4. Repeat until all off-diagonal elements are below tolerance

**Convergence**:

The method converges when the maximum off-diagonal element is below tolerance. Each rotation zeros one off-diagonal element, and the process continues until the matrix is diagonal.

**Parameters**:

- `float tolerance` : Convergence threshold (e.g. 1e-6). Maximum allowed magnitude of off-diagonal elements.

- `int max_iter` : Maximum iterations (e.g. 100). Typically converges in O(n²) iterations for n×n matrices.

**Returns**:

`EigenDecomposition` with `eigenvalues`, `eigenvectors`, `iterations`, and `status`.

**Usage Insights**:

- **Symmetric Matrices**: Designed for symmetric matrices. For non-symmetric matrices, use QR method.

- **Numerical Stability**: Very stable for symmetric matrices, with good preservation of orthogonality.

- **Accuracy**: High accuracy, suitable for applications requiring precise eigenvalue/eigenvector pairs.

- **Performance**: O(n³) per iteration, but typically requires fewer iterations than QR for symmetric matrices.

- **Applications**:

  - Structural dynamics: Stiffness and mass matrices are symmetric

  - Principal Component Analysis (PCA)

  - Spectral clustering

  - Quadratic forms optimization

**Notes**:

If the matrix is not approximately symmetric the function will warn, though it may still run. For non-symmetric matrices prefer the QR method.

**Pitfalls**:

- **Symmetry is the contract**: The method is designed around orthogonal similarity transforms. Non-symmetric input breaks the assumption behind the algorithm.

- **Iteration budget**: Very tight tolerances may require many rotations; keep an eye on `max_iter`.

- **Off-diagonal noise**: Floating-point cleanup can leave tiny residuals. Treat the result as approximate diagonalization.

### QR Eigendecomposition (general matrices) {#qr-eigendecomposition-general-matrices}

```cpp
Mat::EigenDecomposition Mat::eigendecompose_qr(int max_iter, float tolerance) const;
```

**Description**:

Compute eigendecomposition using the QR algorithm. Works for general (possibly non-symmetric) matrices. Supports non-symmetric matrices, but may have complex eigenvalues (only real part returned).

**Mathematical Principle**:

The QR algorithm iteratively applies QR decomposition:

1. Start with A₀ = A

2. For k = 0, 1, 2, ...: Compute QR decomposition: Aₖ = Qₖ * Rₖ, then update: Aₖ₊₁ = Rₖ * Qₖ

3. Aₖ converges to upper triangular form (Schur form), with eigenvalues on the diagonal

**Convergence**:

The algorithm converges when Aₖ is approximately upper triangular (sub-diagonal elements < tolerance). The eigenvalues appear on the diagonal, and eigenvectors are accumulated from Q matrices.

**Parameters**:

- `int max_iter` : Maximum number of QR iterations (default: 100).

- `float tolerance` : Convergence tolerance (e.g. 1e-6). Uses relative tolerance comparing sub-diagonal elements to diagonal elements.

**Returns**:

`EigenDecomposition` containing eigenvalues, eigenvectors, iterations and status.

**Usage Insights**:

- **General Matrices**: Can handle non-symmetric matrices, unlike Jacobi method.

- **Complex Eigenvalues**: Non-symmetric matrices may have complex eigenvalues; current implementation returns real parts only.

- **Numerical Stability**: Uses modified Gram-Schmidt with re-orthogonalization for improved stability.

- **Performance**: O(n³) per iteration. May require many iterations for convergence, especially for ill-conditioned matrices.

- **Convergence Acceleration**: The implementation could benefit from shifts (Wilkinson shift) for faster convergence, but current version uses basic QR iteration.

- **Applications**:

  - General matrix eigenvalue problems

  - Dynamical systems analysis

  - Control theory (system poles)

**Notes**:

QR uses Gram–Schmidt for Q/R in this implementation; it can be less stable for ill-conditioned matrices. For symmetric matrices, Jacobi is preferred due to better stability and accuracy.

**Pitfalls**:

- **Complex spectrum caveat**: The current interface keeps only the real part of eigenvalues, so it is not a full complex eigensolver.

- **Convergence can be slow**: Without shifts, QR can need many iterations on difficult matrices.

- **Orthogonalization quality matters**: Because Q is built through Gram-Schmidt, ill-conditioned inputs can degrade the iteration.

### Automatic Eigendecomposition {#automatic-eigendecomposition}

```cpp
Mat::EigenDecomposition Mat::eigendecompose(float tolerance = 1e-6f, int max_iter = 100) const;
```

**Description**:

Convenience interface that automatically selects the optimal algorithm based on matrix properties. It tests symmetry with `is_symmetric(tolerance * 10.0f)`. If approximately symmetric, it uses Jacobi; otherwise it runs QR. Convenient interface for edge computing applications.

**Algorithm Selection**:

1. Test if matrix is symmetric: `is_symmetric(tolerance * 10.0f)`
2. If symmetric → use `eigendecompose_jacobi(tolerance, max_iter)` (more stable and accurate)
3. If not symmetric → use `eigendecompose_qr(max_iter, tolerance)` (handles general matrices)

**Parameters**:

- `float tolerance` : Used for symmetry test and decomposition convergence (default 1e-6f).
- `int max_iter` : Maximum number of iterations (must be > 0, default = 100).

**Returns**:

`EigenDecomposition` containing all eigenvalues and eigenvectors.

**Usage Insights**:

- **Automatic Optimization**: Saves the user from manually choosing the algorithm, while still providing optimal performance.

- **Edge Computing**: Ideal for embedded systems where you want good performance without manual tuning.

- **Robustness**: The symmetry test uses a relaxed tolerance (10×) to handle numerical errors, ensuring symmetric matrices are correctly identified.

**Usage Tips**:

- **Known Symmetry**: If the matrix is known to be symmetric (e.g. stiffness or mass matrices), call `eigendecompose_jacobi` directly for best stability and slightly better performance.

- **Unknown Properties**: For general matrices or unknown symmetry, use `eigendecompose` for automatic selection.

- **Performance Considerations**: 
  - Eigendecomposition is computationally expensive for large matrices on embedded platforms
  - For n > 20, consider reduced-order methods or iterative methods (power iteration) when only a few eigenvalues are needed
  - For real-time applications, use `power_iteration()` or `inverse_power_iteration()` for single eigenvalues

- **Memory Usage**: Full eigendecomposition requires storing all eigenvectors (n×n matrix), which can be memory-intensive for large matrices.

**Pitfalls**:

- **Automatic does not mean free**: The symmetry test and the decomposition both cost time. If you already know the matrix class, call the specialized method directly.

- **Tolerance is dual-purpose**: It influences both symmetry detection and convergence behavior, so changing it affects algorithm selection as well as numerical stopping.

- **Large matrices**: For embedded use, full eigendecomposition can dominate both compute time and RAM.
