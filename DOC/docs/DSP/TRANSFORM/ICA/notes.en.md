# NOTES

!!! note "Note"
    Independent Component Analysis (ICA) is a blind source separation technique that separates mixed signals into their independent source components. It assumes that the observed signals are linear mixtures of statistically independent sources. ICA is widely used in signal processing, neuroscience, image processing, and audio source separation applications.

## ICA OVERVIEW

### Mathematical Principle

ICA addresses the blind source separation problem:

\[
\mathbf{X} = \mathbf{A} \cdot \mathbf{S}
\]

Where:

- \( \mathbf{X} \) is the matrix of observed (mixed) signals (num_obs × num_samples)

- \( \mathbf{A} \) is the unknown mixing matrix (num_obs × num_sources)

- \( \mathbf{S} \) is the matrix of independent source signals (num_sources × num_samples)

**Goal**: Find the unmixing matrix \( \mathbf{W} \) such that:

\[
\mathbf{S} = \mathbf{W} \cdot \mathbf{X}
\]

**Key Assumptions**:

1. **Statistical Independence**: Source signals are statistically independent

2. **Non-Gaussianity**: At most one source can be Gaussian (for identifiability)

3. **Linear Mixing**: Observations are linear combinations of sources

4. **Square or Overdetermined**: Number of observations ≥ number of sources

### ICA vs PCA

- **PCA**: Finds orthogonal directions of maximum variance (second-order statistics)
- **ICA**: Finds statistically independent directions (higher-order statistics)
- **PCA**: Decorrelates data (removes linear dependencies)
- **ICA**: Separates independent sources (removes all dependencies)

## ALGORITHMS

### FastICA

The library implements the FastICA algorithm, which is based on maximizing non-Gaussianity:

**Objective Function**: Maximize non-Gaussianity of \( \mathbf{w}^T \mathbf{x} \)

**Nonlinearity Functions**:

- **tanh**: \( g(u) = \tanh(u) \) - Good for super-Gaussian sources

- **exp**: \( g(u) = u \cdot e^{-u^2/2} \) - Good for sub-Gaussian sources

- **cube**: \( g(u) = u^3 \) - Good for super-Gaussian sources

**Algorithm Steps**:

1. Center data: \( \mathbf{x}_c = \mathbf{x} - \text{mean}(\mathbf{x}) \)

2. Whiten data: \( \mathbf{z} = \mathbf{D}^{-1/2} \mathbf{E}^T \mathbf{x}_c \)

3. Extract components using fixed-point iteration

4. Orthogonalize components (Gram-Schmidt)

## PREPROCESSING

### Centering

Subtract the mean from each observation:

\[
\mathbf{x}_c = \mathbf{x} - \bar{\mathbf{x}}
\]

Where \( \bar{\mathbf{x}} \) is the mean vector.

### Whitening

Transform data to have unit variance and zero correlation:

\[
\mathbf{z} = \mathbf{D}^{-1/2} \mathbf{E}^T \mathbf{x}_c
\]

Where:

- \( \mathbf{E} \) are eigenvectors of covariance matrix

- \( \mathbf{D} \) are eigenvalues of covariance matrix

**Whitening Matrix**:

\[
\mathbf{W}_{whiten} = \mathbf{D}^{-1/2} \mathbf{E}^T
\]

## FUNCTIONS

### tiny_ica_separate_f32

```c
/**
 * @name tiny_ica_separate_f32
 * @brief Perform ICA separation on mixed signals
 * @param mixed_signals Input mixed signals (num_obs x num_samples, row-major)
 * @param num_obs Number of observations (mixed signals)
 * @param num_samples Number of samples per signal
 * @param num_sources Number of independent sources to extract
 * @param separated_sources Output separated sources (num_sources x num_samples, row-major)
 * @param algorithm ICA algorithm to use (default: TINY_ICA_FASTICA)
 * @param nonlinearity Nonlinearity function for FastICA (default: TINY_ICA_NONLINEARITY_TANH)
 * @param max_iter Maximum number of iterations (default: 100)
 * @param tolerance Convergence tolerance (default: 1e-4)
 * @return tiny_error_t
 */
tiny_error_t tiny_ica_separate_f32(const float *mixed_signals,
                                   int num_obs,
                                   int num_samples,
                                   int num_sources,
                                   float *separated_sources,
                                   tiny_ica_algorithm_t algorithm,
                                   tiny_ica_nonlinearity_t nonlinearity,
                                   int max_iter,
                                   float tolerance);
```

**Description**: 

Performs complete ICA separation in one function call. This is the simplest interface for ICA.

**Parameters**:

- `mixed_signals`: Pointer to input mixed signals array. Data layout is row-major: `mixed_signals[i * num_samples + j]` is sample `j` of observation `i`.

- `num_obs`: Number of observations (mixed signals). Must be ≥ `num_sources`.

- `num_samples`: Number of samples per signal.

- `num_sources`: Number of independent sources to extract. Must be ≤ `num_obs`.

- `separated_sources`: Pointer to output array for separated sources. Data layout is row-major: `separated_sources[i * num_samples + j]` is sample `j` of source `i`. Size must be at least `num_sources * num_samples`.

- `algorithm`: ICA algorithm type. Currently only `TINY_ICA_FASTICA` is supported.

- `nonlinearity`: Nonlinearity function for FastICA:
  - `TINY_ICA_NONLINEARITY_TANH`: tanh (default, good for super-Gaussian)
  - `TINY_ICA_NONLINEARITY_EXP`: exp(-u²/2) (good for sub-Gaussian)
  - `TINY_ICA_NONLINEARITY_CUBE`: u³ (good for super-Gaussian)

- `max_iter`: Maximum number of iterations for FastICA. Default: 100 if ≤ 0.

- `tolerance`: Convergence tolerance. Algorithm stops when change < tolerance. Default: 1e-4 if ≤ 0.

**Return Value**: 

Returns `TINY_OK` on success, or error code on failure.

**Processing Steps**:

1. **Center data**: Subtract mean from each observation
2. **Whiten data**: Decorrelate and normalize variance using eigenvalue decomposition
3. **Extract components**: Use FastICA to find independent components
4. **Reconstruct sources**: Apply unmixing matrix to whitened data

**Note**: 

This function performs all steps internally. For repeated separations, use the structure-based API (`tiny_ica_init`, `tiny_ica_fit`, `tiny_ica_transform`) to avoid recomputing whitening matrix.

### tiny_ica_init

```c
/**
 * @name tiny_ica_init
 * @brief Initialize ICA structure for repeated use
 * @param ica Pointer to ICA structure
 * @param num_obs Number of observations (mixed signals)
 * @param num_sources Number of sources to extract
 * @return tiny_error_t
 */
tiny_error_t tiny_ica_init(tiny_ica_t *ica, int num_obs, int num_sources);
```

**Description**: 

Initializes an ICA structure for repeated use. Allocates memory for mixing matrix, unmixing matrix, whitening matrix, and mean vector.

**Parameters**:

- `ica`: Pointer to `tiny_ica_t` structure.

- `num_obs`: Number of observations (mixed signals). Must be ≥ `num_sources`.

- `num_sources`: Number of sources to extract. Must be ≤ `num_obs`.

**Return Value**: 

Returns `TINY_OK` on success, or error code on failure.

**Memory Management**: 

Function allocates memory internally. Call `tiny_ica_deinit()` to free it.

### tiny_ica_fit

```c
/**
 * @name tiny_ica_fit
 * @brief Fit ICA model to mixed signals (learn unmixing matrix)
 * @param ica Pointer to initialized ICA structure
 * @param mixed_signals Input mixed signals (num_obs x num_samples, row-major)
 * @param num_samples Number of samples per signal
 * @param algorithm ICA algorithm to use
 * @param nonlinearity Nonlinearity function for FastICA
 * @param max_iter Maximum number of iterations
 * @param tolerance Convergence tolerance
 * @return tiny_error_t
 */
tiny_error_t tiny_ica_fit(tiny_ica_t *ica,
                          const float *mixed_signals,
                          int num_samples,
                          tiny_ica_algorithm_t algorithm,
                          tiny_ica_nonlinearity_t nonlinearity,
                          int max_iter,
                          float tolerance);
```

**Description**: 

Fits an ICA model to the training data. Learns the unmixing matrix and whitening matrix. After fitting, use `tiny_ica_transform()` to separate new signals.

**Parameters**:

- `ica`: Pointer to initialized `tiny_ica_t` structure.

- `mixed_signals`: Pointer to input mixed signals array (row-major layout).

- `num_samples`: Number of samples per signal.

- `algorithm`: ICA algorithm type. Currently only `TINY_ICA_FASTICA` is supported.

- `nonlinearity`: Nonlinearity function for FastICA.

- `max_iter`: Maximum number of iterations. Default: 100 if ≤ 0.

- `tolerance`: Convergence tolerance. Default: 1e-4 if ≤ 0.

**Return Value**: 

Returns `TINY_OK` on success, or error code on failure.

**Note**: 

After fitting, the ICA structure contains the learned unmixing matrix and whitening matrix. These can be reused for transforming new data.

### tiny_ica_transform

```c
/**
 * @name tiny_ica_transform
 * @brief Apply learned ICA model to separate signals
 * @param ica Pointer to fitted ICA structure
 * @param mixed_signals Input mixed signals (num_obs x num_samples, row-major)
 * @param num_samples Number of samples per signal
 * @param separated_sources Output separated sources (num_sources x num_samples, row-major)
 * @return tiny_error_t
 */
tiny_error_t tiny_ica_transform(const tiny_ica_t *ica,
                                const float *mixed_signals,
                                int num_samples,
                                float *separated_sources);
```

**Description**: 

Applies a previously fitted ICA model to separate new signals. Much faster than `tiny_ica_separate_f32()` since it reuses the learned unmixing matrix.

**Parameters**:

- `ica`: Pointer to fitted `tiny_ica_t` structure.

- `mixed_signals`: Pointer to input mixed signals array (row-major layout).

- `num_samples`: Number of samples per signal.

- `separated_sources`: Pointer to output array for separated sources (row-major layout). Size must be at least `num_sources * num_samples`.

**Return Value**: 

Returns `TINY_OK` on success, or error code on failure.

**Note**: 

Requires `ica` to be fitted first using `tiny_ica_fit()`. The input signals are centered using the mean from the training data.

### tiny_ica_deinit

```c
/**
 * @name tiny_ica_deinit
 * @brief Deinitialize ICA structure and free memory
 * @param ica Pointer to ICA structure
 * @return tiny_error_t
 */
tiny_error_t tiny_ica_deinit(tiny_ica_t *ica);
```

**Description**: 

Deinitializes an ICA structure and frees all allocated memory.

**Parameters**:

- `ica`: Pointer to `tiny_ica_t` structure.

**Return Value**: 

Returns `TINY_OK` on success, or error code on failure.

## USAGE WORKFLOW

### Simple One-Shot Separation

```c
float mixed_signals[2 * 512];  // 2 observations, 512 samples each
float separated_sources[2 * 512];  // 2 sources, 512 samples each

// Perform ICA separation
tiny_error_t ret = tiny_ica_separate_f32(
    mixed_signals, 2, 512, 2, separated_sources,
    TINY_ICA_FASTICA, TINY_ICA_NONLINEARITY_TANH, 100, 1e-4f);
```

### Repeated Separations (Structure API)

```c
tiny_ica_t ica;

// Initialize
tiny_ica_init(&ica, 2, 2);  // 2 observations, 2 sources

// Fit model to training data
tiny_ica_fit(&ica, training_mixed, 512,
             TINY_ICA_FASTICA, TINY_ICA_NONLINEARITY_TANH, 100, 1e-4f);

// Transform new data (can be called multiple times)
tiny_ica_transform(&ica, new_mixed, 512, separated);

// Cleanup
tiny_ica_deinit(&ica);
```

## APPLICATIONS

ICA is widely used in:

- **Audio Source Separation**: Separating individual instruments or voices from mixed audio
- **Biomedical Signal Processing**: Separating EEG/ECG signals from artifacts
- **Image Processing**: Feature extraction, denoising
- **Communications**: Blind channel equalization
- **Neuroscience**: Analyzing brain signals, fMRI data
- **Sensor Array Processing**: Separating signals from multiple sensors

## ADVANTAGES

- **Blind Separation**: No prior knowledge of mixing matrix needed
- **Statistical Independence**: Finds truly independent sources
- **Non-Gaussian Sources**: Works well with non-Gaussian signals
- **Flexible**: Can handle different numbers of sources and observations

## DISADVANTAGES

- **Ambiguity**: Scale and sign of separated sources are ambiguous
- **Order Ambiguity**: Order of separated sources is arbitrary
- **Non-Gaussian Requirement**: At most one source can be Gaussian
- **Computational Cost**: Whitening and eigenvalue decomposition can be expensive
- **Convergence**: May not converge for some signal types

## DESIGN CONSIDERATIONS

### Number of Sources vs Observations

- **Square Case** (num_obs = num_sources): Standard ICA problem
- **Overdetermined** (num_obs > num_sources): Can use PCA to reduce dimensionality first
- **Underdetermined** (num_obs < num_sources): Not supported (cannot extract more sources than observations)

### Nonlinearity Selection

- **tanh**: Default choice, works well for most super-Gaussian sources (speech, music)
- **exp**: Use for sub-Gaussian sources (uniform noise, some image signals)
- **cube**: Alternative for super-Gaussian sources, simpler but less robust

### Convergence Parameters

- **max_iter**: Typically 50-200 iterations. More iterations for difficult cases.
- **tolerance**: Typically 1e-4 to 1e-6. Smaller tolerance = more accurate but slower.

### Data Requirements

- **Sample Size**: More samples = better separation quality
- **Independence**: Sources must be statistically independent
- **Non-Gaussianity**: At most one source can be Gaussian

## NOTES

- ICA can only separate sources up to a scaling factor and permutation
- The order of separated sources may not match the original order
- ICA works best when sources have different statistical properties
- Whitening is a critical preprocessing step for ICA
- FastICA is a popular algorithm due to its speed and simplicity

