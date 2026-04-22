/**
 * @file tiny_tensor.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Tensor data structure for multi-dimensional arrays
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 *
 * @details
 * This module provides tensor (multi-dimensional array) data structure for AI operations.
 * 
 * Features:
 * - Multi-dimensional array support (1D, 2D, 3D, etc.)
 * - Gradient storage for backpropagation
 * - Flexible memory management (static/dynamic allocation)
 * - Shape and stride information
 * - Extensible design for future data types
 * 
 * Design considerations:
 * - Memory-efficient: Support for views and in-place operations
 * - Training-ready: Built-in gradient storage
 * - MCU-friendly: Static allocation option, minimal overhead
 */

#pragma once

#include "tiny_ai_config.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* ============================================================================
 * TYPE DEFINITIONS
 * ============================================================================ */

/**
 * @brief Data type enumeration (extensible for future types)
 */
typedef enum
{
    TINY_AI_DTYPE_FLOAT32 = 0,  // 32-bit float (default)
    TINY_AI_DTYPE_FLOAT16,      // 16-bit float (future)
    TINY_AI_DTYPE_INT8,          // 8-bit integer (future)
    TINY_AI_DTYPE_INT16,         // 16-bit integer (future)
    TINY_AI_DTYPE_INT32,         // 32-bit integer (future)
} tiny_ai_dtype_t;

/**
 * @brief Memory ownership flag
 */
typedef enum
{
    TINY_AI_MEM_OWNED = 0,       // Tensor owns the memory (will free on destroy)
    TINY_AI_MEM_EXTERNAL,        // External memory (will not free on destroy)
} tiny_ai_mem_ownership_t;

/**
 * @brief Tensor structure
 * 
 * @note This structure is designed to be extensible:
 *       - Can add more fields without breaking existing code
 *       - Supports future features like views, quantization, etc.
 */
typedef struct tiny_tensor_t
{
    // Data pointer
    void* data;                  // Pointer to data buffer
    
    // Gradient pointer (for training)
#if TINY_AI_ENABLE_GRADIENTS
    void* grad;                   // Pointer to gradient buffer (same shape as data)
    bool requires_grad;           // Whether gradients are needed
#endif

    // Shape information
    int ndim;                     // Number of dimensions (0-8)
    int shape[TINY_AI_TENSOR_MAX_DIMS];  // Shape of each dimension
    int stride[TINY_AI_TENSOR_MAX_DIMS]; // Stride for each dimension (for future views)
    
    // Metadata
    tiny_ai_dtype_t dtype;        // Data type
    int numel;                    // Total number of elements (product of shape)
    size_t data_size;             // Size of data buffer in bytes
    
    // Memory management
    tiny_ai_mem_ownership_t mem_ownership;  // Memory ownership flag
    bool is_view;                 // Whether this is a view (shares memory)
    
    // Internal flags
    bool initialized;             // Whether tensor is initialized
} tiny_tensor_t;

/* ============================================================================
 * FUNCTION PROTOTYPES - Creation and Destruction
 * ============================================================================ */

/**
 * @brief Create a new tensor with specified shape and allocate memory internally
 * 
 * @param shape Array of dimension sizes (length = ndim). Each element must be > 0.
 *              Example: For a 3x4 matrix, use shape = {3, 4}
 * @param ndim Number of dimensions (1-8). Must match the length of shape array.
 * @param dtype Data type. Currently only TINY_AI_DTYPE_FLOAT32 is supported.
 * @return tiny_tensor_t* Pointer to created tensor on success, NULL on failure.
 *         Failure cases: NULL shape, invalid ndim, invalid shape values, memory allocation failure.
 * 
 * @note Memory is allocated internally and owned by tensor (will be freed on destroy)
 * @note Data buffer is aligned to TINY_AI_TENSOR_DEFAULT_ALIGN bytes for SIMD optimization
 * @note Gradient buffer is allocated if TINY_AI_ENABLE_GRADIENTS is enabled
 * @note All data is zero-initialized after allocation
 * 
 * @example
 *   // Create a 3x4 2D tensor (matrix)
 *   int shape[] = {3, 4};
 *   tiny_tensor_t* tensor = tiny_tensor_create(shape, 2, TINY_AI_DTYPE_FLOAT32);
 *   if (tensor != NULL) {
 *       // Use tensor...
 *       tiny_tensor_destroy(tensor);
 *   }
 */
tiny_tensor_t* tiny_tensor_create(const int* shape, int ndim, tiny_ai_dtype_t dtype);

/**
 * @brief Create a tensor from existing data buffer (wrap external memory)
 * 
 * @param data Pointer to existing data buffer. Must be valid and match the shape/dtype.
 *             The buffer size must be at least: numel * sizeof(dtype) bytes.
 * @param shape Array of dimension sizes (length = ndim). Each element must be > 0.
 * @param ndim Number of dimensions (1-8)
 * @param dtype Data type. Currently only TINY_AI_DTYPE_FLOAT32 is supported.
 * @param own_memory If true, tensor takes ownership and will free memory on destroy.
 *                   If false, caller must manage the data buffer lifetime.
 * @return tiny_tensor_t* Pointer to created tensor on success, NULL on failure.
 * 
 * @note This function does NOT copy data, it wraps the existing buffer
 * @note If own_memory is false, the data buffer must remain valid for the tensor's lifetime
 * @note Gradient buffer is only allocated if own_memory is true
 * @note Useful for integrating with existing data sources or avoiding data copies
 * 
 * @example
 *   // Wrap a static array (don't own memory)
 *   float static_data[12] = {1.0, 2.0, ...};
 *   int shape[] = {3, 4};
 *   tiny_tensor_t* tensor = tiny_tensor_from_buffer(
 *       static_data, shape, 2, TINY_AI_DTYPE_FLOAT32, false
 *   );
 * 
 * @example
 *   // Wrap dynamically allocated memory (own it)
 *   float* dynamic_data = malloc(12 * sizeof(float));
 *   tiny_tensor_t* tensor = tiny_tensor_from_buffer(
 *       dynamic_data, shape, 2, TINY_AI_DTYPE_FLOAT32, true
 *   );
 *   // tensor will free dynamic_data when destroyed
 */
tiny_tensor_t* tiny_tensor_from_buffer(void* data, const int* shape, int ndim, 
                                       tiny_ai_dtype_t dtype, bool own_memory);

/**
 * @brief Destroy a tensor and free all associated memory
 * 
 * @param tensor Pointer to tensor to destroy. Can be NULL (safe to call with NULL).
 * 
 * @note This function is safe to call with NULL pointer (does nothing)
 * @note Only frees data/gradient buffers if tensor owns them (mem_ownership == TINY_AI_MEM_OWNED)
 * @note Always frees the tensor structure itself (allocated with malloc)
 * @note After calling this function, the tensor pointer becomes invalid and must not be used
 * 
 * @example
 *   tiny_tensor_t* tensor = tiny_tensor_create(shape, 2, TINY_AI_DTYPE_FLOAT32);
 *   // ... use tensor ...
 *   tiny_tensor_destroy(tensor);  // Safe, even if tensor is NULL
 */
void tiny_tensor_destroy(tiny_tensor_t* tensor);

/* ============================================================================
 * FUNCTION PROTOTYPES - Basic Operations
 * ============================================================================ */

/**
 * @brief Copy data and gradients from source tensor to destination tensor
 * 
 * @param src Source tensor (read-only). Must be initialized and valid.
 * @param dst Destination tensor. Must be initialized, valid, and have matching shape/type.
 * @return tiny_error_t TINY_OK on success, error code on failure.
 *         Possible errors: NULL pointer, uninitialized tensor, shape mismatch, type mismatch.
 * 
 * @note Both tensors must have the same shape and data type
 * @note Data is copied using memcpy (efficient for large tensors)
 * @note Gradients are also copied if both tensors have gradient buffers
 * @note The requires_grad flag is copied from source to destination
 * 
 * @example
 *   tiny_tensor_t* src = tiny_tensor_create(shape, 2, TINY_AI_DTYPE_FLOAT32);
 *   tiny_tensor_t* dst = tiny_tensor_create(shape, 2, TINY_AI_DTYPE_FLOAT32);
 *   tiny_tensor_fill(src, 3.14f);
 *   tiny_tensor_copy(src, dst);  // dst now contains the same data as src
 */
tiny_error_t tiny_tensor_copy(const tiny_tensor_t* src, tiny_tensor_t* dst);

/**
 * @brief Fill all elements of a tensor with a constant value
 * 
 * @param tensor Tensor to fill. Must be initialized and valid.
 * @param value Value to fill all elements with. Currently only float32 is supported.
 * @return tiny_error_t TINY_OK on success, error code on failure.
 *         Possible errors: NULL pointer, uninitialized tensor, unsupported dtype.
 * 
 * @note Currently only supports TINY_AI_DTYPE_FLOAT32
 * @note Uses a loop to set each element (necessary for arbitrary float values)
 * @note For zero-filling, use tiny_tensor_zero() which is more efficient
 * 
 * @example
 *   tiny_tensor_t* tensor = tiny_tensor_create(shape, 2, TINY_AI_DTYPE_FLOAT32);
 *   tiny_tensor_fill(tensor, 2.5f);  // All elements become 2.5
 */
tiny_error_t tiny_tensor_fill(tiny_tensor_t* tensor, float value);

/**
 * @brief Set all elements of a tensor to zero
 * 
 * @param tensor Tensor to zero. Must be initialized and valid.
 * @return tiny_error_t TINY_OK on success, error code on failure.
 *         Possible errors: NULL pointer, uninitialized tensor.
 * 
 * @note Uses memset for efficient zero-filling (faster than fill with 0.0f)
 * @note Works for all data types (zero is valid for all types)
 * @note This only zeros the data buffer, not the gradient buffer
 * 
 * @example
 *   tiny_tensor_t* tensor = tiny_tensor_create(shape, 2, TINY_AI_DTYPE_FLOAT32);
 *   // ... use tensor ...
 *   tiny_tensor_zero(tensor);  // Reset all data to zero
 */
tiny_error_t tiny_tensor_zero(tiny_tensor_t* tensor);

/**
 * @brief Set all gradient values to zero (used in training)
 * 
 * @param tensor Tensor whose gradients to zero. Must be initialized and valid.
 * @return tiny_error_t TINY_OK on success, error code on failure.
 *         Possible errors: NULL pointer, uninitialized tensor.
 * 
 * @note Only available if TINY_AI_ENABLE_GRADIENTS is enabled
 * @note Uses memset for efficient zero-filling
 * @note Does nothing if gradient buffer doesn't exist
 * @note Typically called at the start of each training iteration before backpropagation
 * 
 * @example
 *   tiny_tensor_t* weights = tiny_tensor_create(shape, 2, TINY_AI_DTYPE_FLOAT32);
 *   tiny_tensor_requires_grad(weights, true);
 *   for (int epoch = 0; epoch < 100; epoch++) {
 *       tiny_tensor_zero_grad(weights);  // Clear gradients before backprop
 *       // ... forward pass ...
 *       // ... backward pass (accumulates gradients) ...
 *   }
 */
tiny_error_t tiny_tensor_zero_grad(tiny_tensor_t* tensor);

/* ============================================================================
 * FUNCTION PROTOTYPES - Shape and Indexing
 * ============================================================================ */

/**
 * @brief Get total number of elements in the tensor
 * 
 * @param tensor Tensor to query. Can be NULL or uninitialized (returns 0).
 * @return int Total number of elements (product of all dimensions).
 *         Returns 0 if tensor is NULL or uninitialized.
 * 
 * @note This is the product of all shape dimensions
 * @example For shape [3, 4, 5], returns 3 * 4 * 5 = 60
 */
int tiny_tensor_numel(const tiny_tensor_t* tensor);

/**
 * @brief Get the size of a specific dimension
 * 
 * @param tensor Tensor to query. Must be initialized and valid.
 * @param dim Dimension index (0-based). 0 is the first dimension, 1 is the second, etc.
 * @return int Size of the specified dimension.
 *         Returns -1 if tensor is NULL, uninitialized, or dim is out of range.
 * 
 * @example
 *   int shape[] = {3, 4, 5};
 *   tiny_tensor_t* t = tiny_tensor_create(shape, 3, TINY_AI_DTYPE_FLOAT32);
 *   int dim0_size = tiny_tensor_shape(t, 0);  // Returns 3
 *   int dim1_size = tiny_tensor_shape(t, 1);  // Returns 4
 */
int tiny_tensor_shape(const tiny_tensor_t* tensor, int dim);

/**
 * @brief Get the number of dimensions (rank) of the tensor
 * 
 * @param tensor Tensor to query. Must be initialized and valid.
 * @return int Number of dimensions (rank).
 *         Returns -1 if tensor is NULL or uninitialized.
 * 
 * @example
 *   int shape[] = {3, 4, 5};
 *   tiny_tensor_t* t = tiny_tensor_create(shape, 3, TINY_AI_DTYPE_FLOAT32);
 *   int rank = tiny_tensor_ndim(t);  // Returns 3
 */
int tiny_tensor_ndim(const tiny_tensor_t* tensor);

/**
 * @brief Check if two tensors have exactly the same shape
 * 
 * @param a First tensor. Can be NULL (returns false).
 * @param b Second tensor. Can be NULL (returns false).
 * @return bool True if both tensors are valid, initialized, and have matching shapes.
 *         Returns false if either is NULL, uninitialized, or shapes don't match.
 * 
 * @note This checks both the number of dimensions and the size of each dimension
 * @note Does NOT check data type or data values
 * 
 * @example
 *   int shape1[] = {3, 4};
 *   int shape2[] = {3, 4};
 *   tiny_tensor_t* t1 = tiny_tensor_create(shape1, 2, TINY_AI_DTYPE_FLOAT32);
 *   tiny_tensor_t* t2 = tiny_tensor_create(shape2, 2, TINY_AI_DTYPE_FLOAT32);
 *   bool match = tiny_tensor_shape_equal(t1, t2);  // Returns true
 */
bool tiny_tensor_shape_equal(const tiny_tensor_t* a, const tiny_tensor_t* b);

/* ============================================================================
 * FUNCTION PROTOTYPES - Data Access
 * ============================================================================ */

/**
 * @brief Get direct pointer to the data buffer
 * 
 * @param tensor Tensor to query. Must be initialized and valid.
 * @return void* Pointer to the data buffer, or NULL if tensor is NULL or uninitialized.
 * 
 * @note This provides direct access to the underlying memory
 * @note Use with caution - modifying data directly bypasses bounds checking
 * @note The pointer remains valid until the tensor is destroyed
 * 
 * @example
 *   tiny_tensor_t* tensor = tiny_tensor_create(shape, 2, TINY_AI_DTYPE_FLOAT32);
 *   float* data = (float*)tiny_tensor_data(tensor);
 *   data[0] = 1.0f;  // Direct access
 */
void* tiny_tensor_data(const tiny_tensor_t* tensor);

/**
 * @brief Get direct pointer to the gradient buffer
 * 
 * @param tensor Tensor to query. Must be initialized and valid.
 * @return void* Pointer to the gradient buffer, or NULL if:
 *         - tensor is NULL or uninitialized
 *         - TINY_AI_ENABLE_GRADIENTS is disabled
 *         - gradient buffer doesn't exist (requires_grad is false)
 * 
 * @note Only available if TINY_AI_ENABLE_GRADIENTS is enabled
 * @note Gradient buffer has the same shape and size as the data buffer
 * @note Use with caution - modifying gradients directly bypasses bounds checking
 * 
 * @example
 *   tiny_tensor_t* tensor = tiny_tensor_create(shape, 2, TINY_AI_DTYPE_FLOAT32);
 *   tiny_tensor_requires_grad(tensor, true);
 *   float* grad = (float*)tiny_tensor_grad(tensor);
 *   grad[0] = 0.5f;  // Direct gradient access
 */
void* tiny_tensor_grad(const tiny_tensor_t* tensor);

/**
 * @brief Get a single element value by flat index (FLOAT32 only)
 * 
 * @param tensor Tensor to read from. Must be initialized and valid.
 * @param index Flat (linear) index into the tensor. 0-based, row-major order.
 *              Must be in range [0, numel-1].
 * @return float Element value at the given index.
 *         Returns 0.0f if tensor is NULL, uninitialized, wrong dtype, or index out of range.
 * 
 * @note Currently only supports TINY_AI_DTYPE_FLOAT32
 * @note Flat index means linear indexing: for shape [3,4], index 5 refers to [1,1]
 * @note For 2D tensors: index = row * cols + col
 * 
 * @example
 *   int shape[] = {3, 4};
 *   tiny_tensor_t* tensor = tiny_tensor_create(shape, 2, TINY_AI_DTYPE_FLOAT32);
 *   tiny_tensor_set_f32(tensor, 5, 2.5f);
 *   float val = tiny_tensor_get_f32(tensor, 5);  // Returns 2.5f
 */
float tiny_tensor_get_f32(const tiny_tensor_t* tensor, int index);

/**
 * @brief Set a single element value by flat index (FLOAT32 only)
 * 
 * @param tensor Tensor to write to. Must be initialized and valid.
 * @param index Flat (linear) index into the tensor. 0-based, row-major order.
 *              Must be in range [0, numel-1].
 * @param value Value to set at the given index.
 * @return tiny_error_t TINY_OK on success, error code on failure.
 *         Possible errors: NULL pointer, uninitialized tensor, wrong dtype, index out of range.
 * 
 * @note Currently only supports TINY_AI_DTYPE_FLOAT32
 * @note Flat index means linear indexing: for shape [3,4], index 5 refers to [1,1]
 * 
 * @example
 *   int shape[] = {3, 4};
 *   tiny_tensor_t* tensor = tiny_tensor_create(shape, 2, TINY_AI_DTYPE_FLOAT32);
 *   tiny_tensor_set_f32(tensor, 0, 1.0f);  // Set first element
 *   tiny_tensor_set_f32(tensor, 11, 2.5f); // Set last element (3*4-1=11)
 */
tiny_error_t tiny_tensor_set_f32(tiny_tensor_t* tensor, int index, float value);

/* ============================================================================
 * FUNCTION PROTOTYPES - Gradient Management
 * ============================================================================ */

#if TINY_AI_ENABLE_GRADIENTS

/**
 * @brief Enable or disable gradient computation for a tensor (for training)
 * 
 * @param tensor Tensor to configure. Must be initialized and valid.
 * @param enable If true, enable gradient computation and allocate gradient buffer if needed.
 *               If false, disable gradient computation (but don't free existing buffer).
 * @return tiny_error_t TINY_OK on success, error code on failure.
 *         Possible errors: NULL pointer, uninitialized tensor, memory allocation failure.
 * 
 * @note Only available if TINY_AI_ENABLE_GRADIENTS is enabled
 * @note Gradient buffer is allocated on-demand when enable=true and buffer doesn't exist
 * @note Setting enable=false doesn't free the gradient buffer (may be reused later)
 * @note Typically called on model parameters (weights, bias) that need to be trained
 * 
 * @example
 *   tiny_tensor_t* weights = tiny_tensor_create(shape, 2, TINY_AI_DTYPE_FLOAT32);
 *   tiny_tensor_requires_grad(weights, true);  // Enable gradients for training
 *   // ... training loop ...
 *   tiny_tensor_requires_grad(weights, false); // Disable (freeze weights)
 */
tiny_error_t tiny_tensor_requires_grad(tiny_tensor_t* tensor, bool enable);

/**
 * @brief Check if a tensor requires gradient computation
 * 
 * @param tensor Tensor to query. Must be initialized and valid.
 * @return bool True if gradients are enabled, false otherwise.
 *         Returns false if tensor is NULL, uninitialized, or gradients are disabled.
 * 
 * @note Only available if TINY_AI_ENABLE_GRADIENTS is enabled
 * @note Returns false if TINY_AI_ENABLE_GRADIENTS is disabled (compile-time check)
 * 
 * @example
 *   tiny_tensor_t* tensor = tiny_tensor_create(shape, 2, TINY_AI_DTYPE_FLOAT32);
 *   bool needs_grad = tiny_tensor_get_requires_grad(tensor);  // Returns false
 *   tiny_tensor_requires_grad(tensor, true);
 *   needs_grad = tiny_tensor_get_requires_grad(tensor);  // Returns true
 */
bool tiny_tensor_get_requires_grad(const tiny_tensor_t* tensor);

#endif /* TINY_AI_ENABLE_GRADIENTS */

/* ============================================================================
 * FUNCTION PROTOTYPES - Utility
 * ============================================================================ */

/**
 * @brief Print detailed tensor information for debugging
 * 
 * @param tensor Tensor to print. Can be NULL (prints "NULL").
 * @param name Optional name/label for the tensor. Can be NULL (uses "unknown").
 * 
 * @note Prints: shape, dtype, numel, data_size, and requires_grad (if enabled)
 * @note Useful for debugging and verifying tensor properties
 * @note Output format is human-readable text
 * 
 * @example
 *   tiny_tensor_t* tensor = tiny_tensor_create(shape, 2, TINY_AI_DTYPE_FLOAT32);
 *   tiny_tensor_print_info(tensor, "weights");
 *   // Output:
 *   // Tensor [weights]:
 *   //   Shape: [3, 4]
 *   //   Dtype: 0
 *   //   Numel: 12
 *   //   Data size: 48 bytes
 *   //   Requires grad: false
 */
void tiny_tensor_print_info(const tiny_tensor_t* tensor, const char* name);

#ifdef __cplusplus
}
#endif

