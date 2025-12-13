/**
 * @file tiny_dataloader.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief DataLoader for batch processing in neural network training
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 *
 * @details
 * This module provides data loading functionality for neural network training.
 * It handles batch processing, data shuffling, and iteration over datasets.
 * 
 * Features:
 * - Batch processing with configurable batch size
 * - Optional data shuffling for each epoch
 * - Iterator interface for easy integration with training loops
 * - Support for different data formats
 * - Memory-efficient data access
 * 
 * Design:
 * - Lightweight implementation for MCU
 * - Minimal memory overhead
 * - Flexible data organization
 */

#pragma once

#include "tiny_ai_config.h"
#include "tiny_tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* ============================================================================
 * TYPE DEFINITIONS
 * ============================================================================ */

/**
 * @brief Dataset structure containing input and target tensors
 */
typedef struct tiny_dataset_t
{
    tiny_tensor_t** inputs;        // Array of input tensor pointers
    tiny_tensor_t** targets;       // Array of target tensor pointers
    int num_samples;                // Number of samples in dataset
    int capacity;                   // Current capacity (for dynamic growth)
} tiny_dataset_t;

/**
 * @brief DataLoader structure
 */
typedef struct tiny_dataloader_t
{
    tiny_dataset_t* dataset;       // Pointer to dataset
    int batch_size;                 // Batch size
    bool shuffle;                    // Whether to shuffle data
    int current_idx;                 // Current index in dataset
    int* indices;                    // Shuffled indices (if shuffle enabled)
    int num_batches;                 // Total number of batches
    bool initialized;                // Whether dataloader is initialized
} tiny_dataloader_t;

/* ============================================================================
 * FUNCTION PROTOTYPES - Dataset Management
 * ============================================================================ */

/**
 * @brief Create a new dataset
 * 
 * @param initial_capacity Initial capacity for the dataset (can grow dynamically)
 * @return tiny_dataset_t* Pointer to created dataset, NULL on failure
 */
tiny_dataset_t* tiny_dataset_create(int initial_capacity);

/**
 * @brief Destroy a dataset and free all resources
 * 
 * @param dataset Dataset to destroy (can be NULL)
 * 
 * @note This does NOT destroy the tensors, only the dataset structure
 * @note Caller is responsible for destroying the tensors separately
 */
void tiny_dataset_destroy(tiny_dataset_t* dataset);

/**
 * @brief Add a sample to the dataset
 * 
 * @param dataset Dataset to add sample to
 * @param input Input tensor (will be stored as pointer, not copied)
 * @param target Target tensor (will be stored as pointer, not copied)
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @note The dataset stores pointers to tensors, not copies
 * @note The caller is responsible for managing tensor lifetimes
 */
tiny_error_t tiny_dataset_add_sample(tiny_dataset_t* dataset, 
                                     tiny_tensor_t* input, 
                                     tiny_tensor_t* target);

/**
 * @brief Get the number of samples in the dataset
 * 
 * @param dataset Dataset to query
 * @return int Number of samples, 0 if dataset is NULL
 */
int tiny_dataset_size(const tiny_dataset_t* dataset);

/* ============================================================================
 * FUNCTION PROTOTYPES - DataLoader Management
 * ============================================================================ */

/**
 * @brief Create a new DataLoader
 * 
 * @param dataset Dataset to load from (must not be NULL)
 * @param batch_size Batch size (must be > 0)
 * @param shuffle Whether to shuffle data at the start of each epoch
 * @return tiny_dataloader_t* Pointer to created DataLoader, NULL on failure
 * 
 * @note The DataLoader does not take ownership of the dataset
 * @note The dataset must remain valid for the lifetime of the DataLoader
 */
tiny_dataloader_t* tiny_dataloader_create(tiny_dataset_t* dataset, 
                                          int batch_size, 
                                          bool shuffle);

/**
 * @brief Destroy a DataLoader and free all resources
 * 
 * @param loader DataLoader to destroy (can be NULL)
 * 
 * @note This does NOT destroy the dataset
 */
void tiny_dataloader_destroy(tiny_dataloader_t* loader);

/**
 * @brief Reset the DataLoader for a new epoch
 * 
 * @param loader DataLoader to reset
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @note If shuffle is enabled, this will reshuffle the data
 * @note This should be called at the start of each epoch
 */
tiny_error_t tiny_dataloader_reset(tiny_dataloader_t* loader);

/**
 * @brief Check if there are more batches available
 * 
 * @param loader DataLoader to check
 * @return bool True if there are more batches, false otherwise
 */
bool tiny_dataloader_has_next(const tiny_dataloader_t* loader);

/**
 * @brief Get the next batch of data
 * 
 * @param loader DataLoader to get batch from
 * @param batch_inputs Output array of input tensor pointers (must be pre-allocated)
 * @param batch_targets Output array of target tensor pointers (must be pre-allocated)
 * @param actual_batch_size Output actual batch size (may be less than batch_size for last batch)
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @note batch_inputs and batch_targets must be arrays of size at least batch_size
 * @note The function fills these arrays with pointers to tensors from the dataset
 * @note The actual_batch_size may be less than batch_size for the last batch
 */
tiny_error_t tiny_dataloader_get_batch(tiny_dataloader_t* loader,
                                       tiny_tensor_t** batch_inputs,
                                       tiny_tensor_t** batch_targets,
                                       int* actual_batch_size);

/**
 * @brief Get the total number of batches
 * 
 * @param loader DataLoader to query
 * @return int Number of batches, 0 if loader is NULL
 */
int tiny_dataloader_num_batches(const tiny_dataloader_t* loader);

#ifdef __cplusplus
}
#endif

