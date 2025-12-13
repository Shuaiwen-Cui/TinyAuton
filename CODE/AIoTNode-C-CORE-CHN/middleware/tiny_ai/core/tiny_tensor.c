/**
 * @file tiny_tensor.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Tensor data structure implementation
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 */

#include "tiny_tensor.h"
#include <string.h>

/* ============================================================================
 * INTERNAL HELPER FUNCTIONS
 * ============================================================================ */

/**
 * @brief Calculate total number of elements from shape
 */
static int calculate_numel(const int* shape, int ndim)
{
    if (ndim <= 0 || shape == NULL) {
        return 0;
    }
    
    int numel = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            return 0;  // Invalid shape
        }
        numel *= shape[i];
    }
    return numel;
}

/**
 * @brief Calculate stride from shape (row-major order)
 */
static void calculate_stride(const int* shape, int ndim, int* stride)
{
    if (ndim <= 0 || shape == NULL || stride == NULL) {
        return;
    }
    
    // Row-major: stride[i] = product of all dimensions after i
    stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
}

/**
 * @brief Get size of data type in bytes
 */
static size_t dtype_size(tiny_ai_dtype_t dtype)
{
    switch (dtype) {
        case TINY_AI_DTYPE_FLOAT32:
            return sizeof(float);
        case TINY_AI_DTYPE_FLOAT16:
            return sizeof(uint16_t);
        case TINY_AI_DTYPE_INT8:
            return sizeof(int8_t);
        case TINY_AI_DTYPE_INT16:
            return sizeof(int16_t);
        case TINY_AI_DTYPE_INT32:
            return sizeof(int32_t);
        default:
            return sizeof(float);  // Default to float32
    }
}

/**
 * @brief Allocate memory aligned to a specific byte boundary
 * 
 * @details
 * This function allocates memory that is aligned to a specific byte boundary (e.g., 16 bytes).
 * Alignment is important for:
 * - SIMD instructions (SSE, NEON) that require aligned memory
 * - Cache line optimization
 * - Hardware DMA requirements
 * 
 * How it works (concise):
 * 1) Allocate: sizeof(void*) (store base) + align (worst-case padding) + size (user data)
 * 2) Find aligned_addr inside the block (round up to align)
 * 3) Store original base at aligned_addr - sizeof(void*)
 * 4) Return aligned_addr to the caller
 *
 * Memory layout (low → high):
 *   base (malloc)
 *   [ pointer slot | padding ... | aligned_addr | user data ... ]
 *                  ^             ^
 *                  |             └─ aligned_addr (returned, aligned)
 *                  └─ where we store original base (aligned_addr - sizeof(void*))
 * 
 * @param size Number of bytes to allocate (user's actual need)
 * @param align Alignment requirement (must be power of 2, e.g., 4, 8, 16, 32)
 * @return void* Aligned memory pointer, or NULL on failure
 * 
 * @note The returned pointer must be freed using tiny_aligned_free(), NOT free()
 * @note This is a safe implementation that stores the original pointer for correct deallocation
 * 
 * @example
 *   // Allocate 100 bytes aligned to 16-byte boundary
 *   void* mem = tiny_aligned_alloc(100, 16);
 *   // mem is now 16-byte aligned (address like 0x1010, 0x1020, etc.)
 *   // Use mem...
 *   tiny_aligned_free(mem);  // Must use this, not free(mem)!
 */
static void* tiny_aligned_alloc(size_t size, size_t align)
{
    // Step 1: Ensure alignment is at least pointer size
    // This guarantees we have enough space to store the original pointer
    // (We need to store the base pointer somewhere, and it needs sizeof(void*) bytes)
    if (align < sizeof(void*)) {
        align = sizeof(void*);
    }
    
    // Step 2: Allocate extra memory (total = pointer storage + padding + user data)
    // We need: sizeof(void*) (store base pointer) + align (worst-case padding) + size (user's data)
    // Example: If user needs 100 bytes with 16-byte alignment:
    //   - Allocate: 4 + 16 + 100 = 120 bytes (total)
    //   - This ensures we can always find an aligned address within this block
    void* base = malloc(size + align + sizeof(void*));
    if (base == NULL) {
        return NULL;  // Out of memory
    }

    // Step 3: Calculate where we'll store the original pointer
    // We'll put it right before the aligned address, so we need to skip sizeof(void*) bytes
    // from the base address to leave room for storing the pointer
    uintptr_t addr = (uintptr_t)base + sizeof(void*);
    
    // Step 4: Calculate the aligned address
    // Formula: (addr + align - 1) & ~(align - 1)
    // This rounds UP to the nearest multiple of 'align'
    // 
    // How it works:
    //   - (align - 1) creates a mask: for align=16, this is 15 (0x0F = 0b00001111)
    //   - ~(align - 1) creates inverse mask: 0xFFFFFFF0 (clears lower bits)
    //   - (addr + align - 1) adds enough to round up
    //   - & ~(align - 1) clears lower bits, making it a multiple of align
    //
    // Example with align=16:
    //   If addr = 0x1003:
    //     addr + 15 = 0x1012
    //     0x1012 & 0xFFFFFFF0 = 0x1010 (aligned to 16 bytes)
    uintptr_t aligned_addr = (addr + (align - 1)) & ~(align - 1);

    // Step 5: Store the original base pointer right before the aligned address
    // This is crucial! When we free, we need the original malloc pointer, not the aligned one.
    // We store it at: aligned_addr - sizeof(void*)
    // 
    // Memory layout:
    //   base → [4 bytes: base pointer stored here] [padding] aligned_addr → [user data]
    ((void**)(aligned_addr - sizeof(void*)))[0] = base;

    // Step 6: Return the aligned address to the user
    return (void*)aligned_addr;
}

/**
 * @brief Free memory allocated by tiny_aligned_alloc()
 * 
 * @details
 * This function must be used to free memory allocated by tiny_aligned_alloc().
 * DO NOT use free() directly on the aligned pointer!
 * 
 * How it works:
 * 1. The aligned pointer we received is NOT the original malloc pointer
 * 2. We stored the original pointer right before the aligned address
 * 3. We retrieve it by going back sizeof(void*) bytes
 * 4. We free the original pointer (which is what malloc expects)
 * 
 * Why we can't use free() directly:
 * - malloc() and free() must use the SAME pointer
 * - The aligned address is offset from the original malloc pointer
 * - free() on the wrong address will crash or corrupt memory
 * 
 * @param ptr Pointer returned by tiny_aligned_alloc() (the aligned address)
 * 
 * @note This function is safe to call with NULL (does nothing)
 * @note Always use this function, never free() the aligned pointer directly
 * 
 * @example
 *   void* mem = tiny_aligned_alloc(100, 16);
 *   // ... use mem ...
 *   tiny_aligned_free(mem);  // Correct!
 *   // free(mem);            // WRONG! Will crash!
 */
static void tiny_aligned_free(void* ptr)
{
    // Safety check: NULL pointer is safe to free (standard C behavior)
    if (ptr == NULL) {
        return;
    }

    // Step 1: Retrieve the original base pointer
    // When we allocated, we stored the original malloc pointer at:
    //   aligned_addr - sizeof(void*)
    // Now we need to go back and get it:
    //   - ptr is the aligned address we returned to the user
    //   - (uintptr_t)ptr - sizeof(void*) gets us to where we stored the base pointer
    //   - ((void**)(...))[0] reads the stored pointer value
    void* base = ((void**)((uintptr_t)ptr - sizeof(void*)))[0];
    
    // Step 2: Free the original pointer
    // This is the pointer that malloc() actually returned
    // free() must receive the exact pointer that malloc() returned
    free(base);
}

/* ============================================================================
 * CREATION AND DESTRUCTION
 * ============================================================================ */

tiny_tensor_t* tiny_tensor_create(const int* shape, int ndim, tiny_ai_dtype_t dtype)
{
    // Validate inputs
    if (shape == NULL || ndim <= 0 || ndim > TINY_AI_TENSOR_MAX_DIMS) {
        return NULL;
    }
    
    // Validate shape
    int numel = calculate_numel(shape, ndim);
    if (numel <= 0) {
        return NULL;
    }
    
    // Allocate tensor structure
    tiny_tensor_t* tensor = (tiny_tensor_t*)malloc(sizeof(tiny_tensor_t));
    if (tensor == NULL) {
        return NULL;
    }
    
    // Initialize structure
    memset(tensor, 0, sizeof(tiny_tensor_t));
    
    // Set shape and metadata
    tensor->ndim = ndim;
    memcpy(tensor->shape, shape, ndim * sizeof(int));
    calculate_stride(shape, ndim, tensor->stride);
    tensor->numel = numel;
    tensor->dtype = dtype;
    tensor->data_size = numel * dtype_size(dtype);
    
    // Allocate data buffer (aligned for future SIMD optimization)
    size_t alloc_size = tensor->data_size;
    if (alloc_size % TINY_AI_TENSOR_DEFAULT_ALIGN != 0) {
        alloc_size = ((alloc_size / TINY_AI_TENSOR_DEFAULT_ALIGN) + 1) * TINY_AI_TENSOR_DEFAULT_ALIGN;
    }
    
    tensor->data = tiny_aligned_alloc(alloc_size, TINY_AI_TENSOR_DEFAULT_ALIGN);
    if (tensor->data == NULL) {
        free(tensor);
        return NULL;
    }
    
    // Zero initialize
    memset(tensor->data, 0, tensor->data_size);
    
#if TINY_AI_ENABLE_GRADIENTS
    // Allocate gradient buffer
    tensor->grad = tiny_aligned_alloc(alloc_size, TINY_AI_TENSOR_DEFAULT_ALIGN);
    if (tensor->grad == NULL) {
        tiny_aligned_free(tensor->data);
        free(tensor);
        return NULL;
    }
    memset(tensor->grad, 0, tensor->data_size);
    tensor->requires_grad = false;
#endif
    
    // Set flags
    tensor->mem_ownership = TINY_AI_MEM_OWNED;
    tensor->is_view = false;
    tensor->initialized = true;
    
    return tensor;
}

tiny_tensor_t* tiny_tensor_from_buffer(void* data, const int* shape, int ndim, 
                                       tiny_ai_dtype_t dtype, bool own_memory)
{
    // Validate inputs
    if (data == NULL || shape == NULL || ndim <= 0 || ndim > TINY_AI_TENSOR_MAX_DIMS) {
        return NULL;
    }
    
    // Validate shape
    int numel = calculate_numel(shape, ndim);
    if (numel <= 0) {
        return NULL;
    }
    
    // Allocate tensor structure
    tiny_tensor_t* tensor = (tiny_tensor_t*)malloc(sizeof(tiny_tensor_t));
    if (tensor == NULL) {
        return NULL;
    }
    
    // Initialize structure
    memset(tensor, 0, sizeof(tiny_tensor_t));
    
    // Set shape and metadata
    tensor->ndim = ndim;
    memcpy(tensor->shape, shape, ndim * sizeof(int));
    calculate_stride(shape, ndim, tensor->stride);
    tensor->numel = numel;
    tensor->dtype = dtype;
    tensor->data_size = numel * dtype_size(dtype);
    
    // Use external buffer
    tensor->data = data;
    tensor->mem_ownership = own_memory ? TINY_AI_MEM_OWNED : TINY_AI_MEM_EXTERNAL;
    
#if TINY_AI_ENABLE_GRADIENTS
    // Allocate gradient buffer if needed
    if (own_memory) {
        size_t alloc_size = tensor->data_size;
        if (alloc_size % TINY_AI_TENSOR_DEFAULT_ALIGN != 0) {
            alloc_size = ((alloc_size / TINY_AI_TENSOR_DEFAULT_ALIGN) + 1) * TINY_AI_TENSOR_DEFAULT_ALIGN;
        }
        tensor->grad = tiny_aligned_alloc(alloc_size, TINY_AI_TENSOR_DEFAULT_ALIGN);
        if (tensor->grad == NULL) {
            if (own_memory) {
                free(data);
            }
            free(tensor);
            return NULL;
        }
        memset(tensor->grad, 0, tensor->data_size);
    } else {
        tensor->grad = NULL;
    }
    tensor->requires_grad = false;
#endif
    
    // Set flags
    tensor->is_view = false;
    tensor->initialized = true;
    
    return tensor;
}

void tiny_tensor_destroy(tiny_tensor_t* tensor)
{
    if (tensor == NULL) {
        return;
    }
    
    // Free data buffer if owned
    if (tensor->mem_ownership == TINY_AI_MEM_OWNED) {
        if (tensor->data != NULL) {
            tiny_aligned_free(tensor->data);
        }
#if TINY_AI_ENABLE_GRADIENTS
        if (tensor->grad != NULL) {
            tiny_aligned_free(tensor->grad);
        }
#endif
    }
    
    // Free structure
    free(tensor);
}

/* ============================================================================
 * BASIC OPERATIONS
 * ============================================================================ */

tiny_error_t tiny_tensor_copy(const tiny_tensor_t* src, tiny_tensor_t* dst)
{
    if (src == NULL || dst == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    if (!src->initialized || !dst->initialized) {
        return TINY_ERR_AI_UNINITIALIZED;
    }
    
    if (!tiny_tensor_shape_equal(src, dst)) {
        return TINY_ERR_AI_SHAPE_MISMATCH;
    }
    
    if (src->dtype != dst->dtype) {
        return TINY_ERR_AI_NOT_SUPPORTED;  // Type conversion not supported yet
    }
    
    // Copy data
    memcpy(dst->data, src->data, src->data_size);
    
#if TINY_AI_ENABLE_GRADIENTS
    // Copy gradients if both have them
    if (src->grad != NULL && dst->grad != NULL) {
        memcpy(dst->grad, src->grad, src->data_size);
    }
    dst->requires_grad = src->requires_grad;
#endif
    
    return TINY_OK;
}

tiny_error_t tiny_tensor_fill(tiny_tensor_t* tensor, float value)
{
    if (tensor == NULL || !tensor->initialized) {
        return tensor == NULL ? TINY_ERR_AI_NULL_POINTER : TINY_ERR_AI_UNINITIALIZED;
    }
    
    if (tensor->dtype != TINY_AI_DTYPE_FLOAT32) {
        return TINY_ERR_AI_NOT_SUPPORTED;  // Only float32 supported for now
    }
    
    float* data = (float*)tensor->data;
    for (int i = 0; i < tensor->numel; i++) {
        data[i] = value;
    }
    
    return TINY_OK;
}

tiny_error_t tiny_tensor_zero(tiny_tensor_t* tensor)
{
    if (tensor == NULL || !tensor->initialized) {
        return tensor == NULL ? TINY_ERR_AI_NULL_POINTER : TINY_ERR_AI_UNINITIALIZED;
    }
    
    memset(tensor->data, 0, tensor->data_size);
    
    return TINY_OK;
}

tiny_error_t tiny_tensor_zero_grad(tiny_tensor_t* tensor)
{
    if (tensor == NULL || !tensor->initialized) {
        return tensor == NULL ? TINY_ERR_AI_NULL_POINTER : TINY_ERR_AI_UNINITIALIZED;
    }
    
#if TINY_AI_ENABLE_GRADIENTS
    if (tensor->grad != NULL) {
        memset(tensor->grad, 0, tensor->data_size);
    }
#endif
    
    return TINY_OK;
}

/* ============================================================================
 * SHAPE AND INDEXING
 * ============================================================================ */

int tiny_tensor_numel(const tiny_tensor_t* tensor)
{
    if (tensor == NULL || !tensor->initialized) {
        return 0;
    }
    return tensor->numel;
}

int tiny_tensor_shape(const tiny_tensor_t* tensor, int dim)
{
    if (tensor == NULL || !tensor->initialized) {
        return -1;
    }
    
    if (dim < 0 || dim >= tensor->ndim) {
        return -1;
    }
    
    return tensor->shape[dim];
}

int tiny_tensor_ndim(const tiny_tensor_t* tensor)
{
    if (tensor == NULL || !tensor->initialized) {
        return -1;
    }
    return tensor->ndim;
}

bool tiny_tensor_shape_equal(const tiny_tensor_t* a, const tiny_tensor_t* b)
{
    if (a == NULL || b == NULL) {
        return false;
    }
    
    if (!a->initialized || !b->initialized) {
        return false;
    }
    
    if (a->ndim != b->ndim) {
        return false;
    }
    
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }
    
    return true;
}

/* ============================================================================
 * DATA ACCESS
 * ============================================================================ */

void* tiny_tensor_data(const tiny_tensor_t* tensor)
{
    if (tensor == NULL || !tensor->initialized) {
        return NULL;
    }
    return tensor->data;
}

void* tiny_tensor_grad(const tiny_tensor_t* tensor)
{
    if (tensor == NULL || !tensor->initialized) {
        return NULL;
    }
#if TINY_AI_ENABLE_GRADIENTS
    return tensor->grad;
#else
    return NULL;
#endif
}

float tiny_tensor_get_f32(const tiny_tensor_t* tensor, int index)
{
    if (tensor == NULL || !tensor->initialized) {
        return 0.0f;
    }
    
    if (tensor->dtype != TINY_AI_DTYPE_FLOAT32) {
        return 0.0f;
    }
    
    if (index < 0 || index >= tensor->numel) {
        return 0.0f;
    }
    
    float* data = (float*)tensor->data;
    return data[index];
}

tiny_error_t tiny_tensor_set_f32(tiny_tensor_t* tensor, int index, float value)
{
    if (tensor == NULL || !tensor->initialized) {
        return tensor == NULL ? TINY_ERR_AI_NULL_POINTER : TINY_ERR_AI_UNINITIALIZED;
    }
    
    if (tensor->dtype != TINY_AI_DTYPE_FLOAT32) {
        return TINY_ERR_AI_NOT_SUPPORTED;
    }
    
    if (index < 0 || index >= tensor->numel) {
        return TINY_ERR_AI_INVALID_ARG;
    }
    
    float* data = (float*)tensor->data;
    data[index] = value;
    
    return TINY_OK;
}

/* ============================================================================
 * GRADIENT MANAGEMENT
 * ============================================================================ */

#if TINY_AI_ENABLE_GRADIENTS

tiny_error_t tiny_tensor_requires_grad(tiny_tensor_t* tensor, bool enable)
{
    if (tensor == NULL || !tensor->initialized) {
        return tensor == NULL ? TINY_ERR_AI_NULL_POINTER : TINY_ERR_AI_UNINITIALIZED;
    }
    
    tensor->requires_grad = enable;
    
    // Ensure gradient buffer exists if needed
    if (enable && tensor->grad == NULL) {
        size_t alloc_size = tensor->data_size;
        if (alloc_size % TINY_AI_TENSOR_DEFAULT_ALIGN != 0) {
            alloc_size = ((alloc_size / TINY_AI_TENSOR_DEFAULT_ALIGN) + 1) * TINY_AI_TENSOR_DEFAULT_ALIGN;
        }
        tensor->grad = tiny_aligned_alloc(alloc_size, TINY_AI_TENSOR_DEFAULT_ALIGN);
        if (tensor->grad == NULL) {
            return TINY_ERR_AI_NO_MEM;
        }
        memset(tensor->grad, 0, tensor->data_size);
    }
    
    return TINY_OK;
}

bool tiny_tensor_get_requires_grad(const tiny_tensor_t* tensor)
{
    if (tensor == NULL || !tensor->initialized) {
        return false;
    }
#if TINY_AI_ENABLE_GRADIENTS
    return tensor->requires_grad;
#else
    return false;
#endif
}

#endif /* TINY_AI_ENABLE_GRADIENTS */

/* ============================================================================
 * UTILITY
 * ============================================================================ */

void tiny_tensor_print_info(const tiny_tensor_t* tensor, const char* name)
{
    if (tensor == NULL) {
        printf("Tensor [%s]: NULL\n", name ? name : "unknown");
        return;
    }
    
    if (!tensor->initialized) {
        printf("Tensor [%s]: Uninitialized\n", name ? name : "unknown");
        return;
    }
    
    printf("Tensor [%s]:\n", name ? name : "unknown");
    printf("  Shape: [");
    for (int i = 0; i < tensor->ndim; i++) {
        printf("%d", tensor->shape[i]);
        if (i < tensor->ndim - 1) printf(", ");
    }
    printf("]\n");
    printf("  Dtype: %d\n", tensor->dtype);
    printf("  Numel: %d\n", tensor->numel);
    printf("  Data size: %zu bytes\n", tensor->data_size);
#if TINY_AI_ENABLE_GRADIENTS
    printf("  Requires grad: %s\n", tensor->requires_grad ? "true" : "false");
#endif
}

