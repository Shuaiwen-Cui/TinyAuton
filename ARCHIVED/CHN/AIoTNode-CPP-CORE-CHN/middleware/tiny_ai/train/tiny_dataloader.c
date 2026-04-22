/**
 * @file tiny_dataloader.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief DataLoader implementation
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 */

#include "tiny_dataloader.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ============================================================================
 * DATASET MANAGEMENT
 * ============================================================================ */

tiny_dataset_t* tiny_dataset_create(int initial_capacity)
{
    if (initial_capacity <= 0) {
        initial_capacity = 16;  // Default capacity
    }
    
    tiny_dataset_t* dataset = (tiny_dataset_t*)malloc(sizeof(tiny_dataset_t));
    if (dataset == NULL) {
        return NULL;
    }
    
    memset(dataset, 0, sizeof(tiny_dataset_t));
    dataset->capacity = initial_capacity;
    dataset->num_samples = 0;
    
    // Allocate arrays for input and target pointers
    dataset->inputs = (tiny_tensor_t**)malloc(initial_capacity * sizeof(tiny_tensor_t*));
    if (dataset->inputs == NULL) {
        free(dataset);
        return NULL;
    }
    memset(dataset->inputs, 0, initial_capacity * sizeof(tiny_tensor_t*));
    
    dataset->targets = (tiny_tensor_t**)malloc(initial_capacity * sizeof(tiny_tensor_t*));
    if (dataset->targets == NULL) {
        free(dataset->inputs);
        free(dataset);
        return NULL;
    }
    memset(dataset->targets, 0, initial_capacity * sizeof(tiny_tensor_t*));
    
    return dataset;
}

void tiny_dataset_destroy(tiny_dataset_t* dataset)
{
    if (dataset == NULL) {
        return;
    }
    
    if (dataset->inputs != NULL) {
        free(dataset->inputs);
    }
    
    if (dataset->targets != NULL) {
        free(dataset->targets);
    }
    
    free(dataset);
}

tiny_error_t tiny_dataset_add_sample(tiny_dataset_t* dataset, 
                                     tiny_tensor_t* input, 
                                     tiny_tensor_t* target)
{
    if (dataset == NULL || input == NULL || target == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    // Expand capacity if needed
    if (dataset->num_samples >= dataset->capacity) {
        int new_capacity = dataset->capacity * 2;
        
        tiny_tensor_t** new_inputs = (tiny_tensor_t**)realloc(
            dataset->inputs, new_capacity * sizeof(tiny_tensor_t*)
        );
        if (new_inputs == NULL) {
            return TINY_ERR_AI_NO_MEM;
        }
        dataset->inputs = new_inputs;
        
        tiny_tensor_t** new_targets = (tiny_tensor_t**)realloc(
            dataset->targets, new_capacity * sizeof(tiny_tensor_t*)
        );
        if (new_targets == NULL) {
            return TINY_ERR_AI_NO_MEM;
        }
        dataset->targets = new_targets;
        
        // Zero out new space
        memset(dataset->inputs + dataset->capacity, 0, 
               (new_capacity - dataset->capacity) * sizeof(tiny_tensor_t*));
        memset(dataset->targets + dataset->capacity, 0, 
               (new_capacity - dataset->capacity) * sizeof(tiny_tensor_t*));
        
        dataset->capacity = new_capacity;
    }
    
    // Add sample
    dataset->inputs[dataset->num_samples] = input;
    dataset->targets[dataset->num_samples] = target;
    dataset->num_samples++;
    
    return TINY_OK;
}

int tiny_dataset_size(const tiny_dataset_t* dataset)
{
    if (dataset == NULL) {
        return 0;
    }
    return dataset->num_samples;
}

/* ============================================================================
 * DATALOADER MANAGEMENT
 * ============================================================================ */

tiny_dataloader_t* tiny_dataloader_create(tiny_dataset_t* dataset, 
                                          int batch_size, 
                                          bool shuffle)
{
    if (dataset == NULL || batch_size <= 0) {
        return NULL;
    }
    
    tiny_dataloader_t* loader = (tiny_dataloader_t*)malloc(sizeof(tiny_dataloader_t));
    if (loader == NULL) {
        return NULL;
    }
    
    memset(loader, 0, sizeof(tiny_dataloader_t));
    loader->dataset = dataset;
    loader->batch_size = batch_size;
    loader->shuffle = shuffle;
    loader->current_idx = 0;
    loader->num_batches = (dataset->num_samples + batch_size - 1) / batch_size;
    loader->initialized = true;
    
    // Allocate indices array if shuffle is enabled
    if (shuffle && dataset->num_samples > 0) {
        loader->indices = (int*)malloc(dataset->num_samples * sizeof(int));
        if (loader->indices == NULL) {
            free(loader);
            return NULL;
        }
        
        // Initialize indices
        for (int i = 0; i < dataset->num_samples; i++) {
            loader->indices[i] = i;
        }
    } else {
        loader->indices = NULL;
    }
    
    return loader;
}

void tiny_dataloader_destroy(tiny_dataloader_t* loader)
{
    if (loader == NULL) {
        return;
    }
    
    if (loader->indices != NULL) {
        free(loader->indices);
    }
    
    free(loader);
}

/**
 * @brief Fisher-Yates shuffle algorithm
 */
static void shuffle_array(int* array, int n, unsigned int seed)
{
    if (array == NULL || n <= 1) {
        return;
    }
    
    // Simple LCG for random number generation
    unsigned int state = seed;
    if (seed == 0) {
        state = 12345;  // Default seed
    }
    
    for (int i = n - 1; i > 0; i--) {
        // Generate random index between 0 and i
        state = (1103515245u * state + 12345u) & 0x7FFFFFFFu;
        int j = state % (i + 1);
        
        // Swap
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

tiny_error_t tiny_dataloader_reset(tiny_dataloader_t* loader)
{
    if (loader == NULL || !loader->initialized) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    loader->current_idx = 0;
    
    // Reshuffle if enabled
    if (loader->shuffle && loader->indices != NULL && loader->dataset->num_samples > 0) {
        // Reinitialize indices
        for (int i = 0; i < loader->dataset->num_samples; i++) {
            loader->indices[i] = i;
        }
        
        // Shuffle using current time as seed (or a fixed seed for reproducibility)
        // For MCU, we can use a simple counter or fixed seed
        static unsigned int shuffle_seed = 12345;
        shuffle_seed = (shuffle_seed * 1103515245u + 12345u) & 0x7FFFFFFFu;
        shuffle_array(loader->indices, loader->dataset->num_samples, shuffle_seed);
    }
    
    return TINY_OK;
}

bool tiny_dataloader_has_next(const tiny_dataloader_t* loader)
{
    if (loader == NULL || !loader->initialized || loader->dataset == NULL) {
        return false;
    }
    
    return loader->current_idx < loader->dataset->num_samples;
}

tiny_error_t tiny_dataloader_get_batch(tiny_dataloader_t* loader,
                                       tiny_tensor_t** batch_inputs,
                                       tiny_tensor_t** batch_targets,
                                       int* actual_batch_size)
{
    if (loader == NULL || !loader->initialized || 
        batch_inputs == NULL || batch_targets == NULL || actual_batch_size == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    if (!tiny_dataloader_has_next(loader)) {
        *actual_batch_size = 0;
        return TINY_ERR_AI_INVALID_STATE;  // No more batches
    }
    
    // Calculate actual batch size (may be less for last batch)
    int remaining = loader->dataset->num_samples - loader->current_idx;
    int batch_size = (remaining < loader->batch_size) ? remaining : loader->batch_size;
    
    // Get batch
    for (int i = 0; i < batch_size; i++) {
        int sample_idx;
        
        if (loader->shuffle && loader->indices != NULL) {
            // Use shuffled index
            sample_idx = loader->indices[loader->current_idx + i];
        } else {
            // Use sequential index
            sample_idx = loader->current_idx + i;
        }
        
        batch_inputs[i] = loader->dataset->inputs[sample_idx];
        batch_targets[i] = loader->dataset->targets[sample_idx];
    }
    
    loader->current_idx += batch_size;
    *actual_batch_size = batch_size;
    
    return TINY_OK;
}

int tiny_dataloader_num_batches(const tiny_dataloader_t* loader)
{
    if (loader == NULL || !loader->initialized) {
        return 0;
    }
    return loader->num_batches;
}

