/**
 * @file tiny_graph.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Computation graph implementation
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 */

#include "tiny_graph.h"
#include <string.h>
#include <stdlib.h>

/* ============================================================================
 * INTERNAL HELPER FUNCTIONS
 * ============================================================================ */

/**
 * @brief Expand graph node capacity if needed
 */
static tiny_error_t graph_expand_capacity(tiny_graph_t* graph)
{
    if (graph == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    if (graph->num_nodes < graph->capacity) {
        return TINY_OK;  // No need to expand
    }
    
    // Double the capacity
    int new_capacity = graph->capacity * 2;
    if (new_capacity < 4) {
        new_capacity = 4;  // Minimum capacity
    }
    
    tiny_graph_node_t** new_nodes = (tiny_graph_node_t**)realloc(
        graph->nodes, new_capacity * sizeof(tiny_graph_node_t*)
    );
    
    if (new_nodes == NULL) {
        return TINY_ERR_AI_NO_MEM;
    }
    
    graph->nodes = new_nodes;
    graph->capacity = new_capacity;
    
    return TINY_OK;
}

/* ============================================================================
 * GRAPH CREATION AND DESTRUCTION
 * ============================================================================ */

tiny_graph_t* tiny_graph_create(int initial_capacity)
{
    if (initial_capacity <= 0) {
        initial_capacity = 8;  // Default capacity
    }
    
    // Allocate graph structure
    tiny_graph_t* graph = (tiny_graph_t*)malloc(sizeof(tiny_graph_t));
    if (graph == NULL) {
        return NULL;
    }
    
    // Initialize structure
    memset(graph, 0, sizeof(tiny_graph_t));
    
    // Allocate node array
    graph->nodes = (tiny_graph_node_t**)malloc(initial_capacity * sizeof(tiny_graph_node_t*));
    if (graph->nodes == NULL) {
        free(graph);
        return NULL;
    }
    
    graph->capacity = initial_capacity;
    graph->num_nodes = 0;
    graph->training_mode = false;
    graph->initialized = true;
    
    return graph;
}

void tiny_graph_destroy(tiny_graph_t* graph)
{
    if (graph == NULL) {
        return;
    }
    
    // Free all nodes
    if (graph->nodes != NULL) {
        for (int i = 0; i < graph->num_nodes; i++) {
            if (graph->nodes[i] != NULL) {
                // Free node's internal arrays
                if (graph->nodes[i]->inputs != NULL) {
                    free(graph->nodes[i]->inputs);
                }
                if (graph->nodes[i]->outputs != NULL) {
                    free(graph->nodes[i]->outputs);
                }
                if (graph->nodes[i]->children != NULL) {
                    free(graph->nodes[i]->children);
                }
                if (graph->nodes[i]->parents != NULL) {
                    free(graph->nodes[i]->parents);
                }
                // Note: params and tensors are not freed here (managed externally)
                free(graph->nodes[i]);
            }
        }
        free(graph->nodes);
    }
    
    // Free execution order arrays
    if (graph->forward_order != NULL) {
        free(graph->forward_order);
    }
    if (graph->backward_order != NULL) {
        free(graph->backward_order);
    }
    
    // Free graph structure
    free(graph);
}

/* ============================================================================
 * NODE MANAGEMENT
 * ============================================================================ */

tiny_graph_node_t* tiny_graph_add_node(tiny_graph_t* graph, tiny_ai_op_type_t op_type,
                                       int num_inputs, int num_outputs)
{
    if (graph == NULL || !graph->initialized) {
        return NULL;
    }
    
    if (num_inputs < 0 || num_outputs < 0) {
        return NULL;
    }
    
    // Expand capacity if needed
    tiny_error_t err = graph_expand_capacity(graph);
    if (err != TINY_OK) {
        return NULL;
    }
    
    // Allocate node
    tiny_graph_node_t* node = (tiny_graph_node_t*)malloc(sizeof(tiny_graph_node_t));
    if (node == NULL) {
        return NULL;
    }
    
    // Initialize node
    memset(node, 0, sizeof(tiny_graph_node_t));
    node->op_type = op_type;
    node->num_inputs = num_inputs;
    node->num_outputs = num_outputs;
    
    // Allocate input/output arrays
    if (num_inputs > 0) {
        node->inputs = (tiny_tensor_t**)malloc(num_inputs * sizeof(tiny_tensor_t*));
        if (node->inputs == NULL) {
            free(node);
            return NULL;
        }
        memset(node->inputs, 0, num_inputs * sizeof(tiny_tensor_t*));
    }
    
    if (num_outputs > 0) {
        node->outputs = (tiny_tensor_t**)malloc(num_outputs * sizeof(tiny_tensor_t*));
        if (node->outputs == NULL) {
            if (node->inputs != NULL) free(node->inputs);
            free(node);
            return NULL;
        }
        memset(node->outputs, 0, num_outputs * sizeof(tiny_tensor_t*));
    }
    
    // Add to graph
    graph->nodes[graph->num_nodes] = node;
    graph->num_nodes++;
    
    return node;
}

tiny_error_t tiny_graph_connect(tiny_graph_t* graph, tiny_graph_node_t* from_node, int from_output_idx,
                                tiny_graph_node_t* to_node, int to_input_idx)
{
    if (graph == NULL || from_node == NULL || to_node == NULL) {
        return TINY_ERR_AI_NULL_POINTER;
    }
    
    if (from_output_idx < 0 || from_output_idx >= from_node->num_outputs) {
        return TINY_ERR_AI_INVALID_ARG;
    }
    
    if (to_input_idx < 0 || to_input_idx >= to_node->num_inputs) {
        return TINY_ERR_AI_INVALID_ARG;
    }
    
    // Check if already connected (avoid duplicate connections)
    if (to_node->inputs[to_input_idx] == from_node->outputs[from_output_idx]) {
        // Already connected, check if parent/child relationship exists
        bool already_connected = false;
        for (int i = 0; i < from_node->num_children; i++) {
            if (from_node->children[i] == to_node) {
                already_connected = true;
                break;
            }
        }
        if (already_connected) {
            return TINY_OK;  // Already connected, no error
        }
    }
    
    // Connect tensors (data flow)
    to_node->inputs[to_input_idx] = from_node->outputs[from_output_idx];
    
    // Update graph structure: add parent/child relationship
    // from_node is parent of to_node
    // to_node is child of from_node
    
    // Add to_node to from_node's children list
    if (from_node->children == NULL) {
        from_node->children = (tiny_graph_node_t**)malloc(sizeof(tiny_graph_node_t*));
        if (from_node->children == NULL) {
            return TINY_ERR_AI_NO_MEM;
        }
        from_node->num_children = 0;
    } else {
        tiny_graph_node_t** new_children = (tiny_graph_node_t**)realloc(
            from_node->children, (from_node->num_children + 1) * sizeof(tiny_graph_node_t*)
        );
        if (new_children == NULL) {
            return TINY_ERR_AI_NO_MEM;
        }
        from_node->children = new_children;
    }
    from_node->children[from_node->num_children] = to_node;
    from_node->num_children++;
    
    // Add from_node to to_node's parents list
    if (to_node->parents == NULL) {
        to_node->parents = (tiny_graph_node_t**)malloc(sizeof(tiny_graph_node_t*));
        if (to_node->parents == NULL) {
            return TINY_ERR_AI_NO_MEM;
        }
        to_node->num_parents = 0;
    } else {
        tiny_graph_node_t** new_parents = (tiny_graph_node_t**)realloc(
            to_node->parents, (to_node->num_parents + 1) * sizeof(tiny_graph_node_t*)
        );
        if (new_parents == NULL) {
            return TINY_ERR_AI_NO_MEM;
        }
        to_node->parents = new_parents;
    }
    to_node->parents[to_node->num_parents] = from_node;
    to_node->num_parents++;
    
    // Update in_degree (number of incoming edges)
    to_node->in_degree++;
    
    return TINY_OK;
}

/* ============================================================================
 * GRAPH EXECUTION
 * ============================================================================ */

tiny_error_t tiny_graph_build_order(tiny_graph_t* graph)
{
    if (graph == NULL || !graph->initialized) {
        return graph == NULL ? TINY_ERR_AI_NULL_POINTER : TINY_ERR_AI_UNINITIALIZED;
    }
    
    if (graph->num_nodes == 0) {
        return TINY_OK;  // Empty graph
    }
    
    // Free existing order arrays if any
    if (graph->forward_order != NULL) {
        free(graph->forward_order);
        graph->forward_order = NULL;
    }
    if (graph->backward_order != NULL) {
        free(graph->backward_order);
        graph->backward_order = NULL;
    }
    
    // Allocate order arrays
    graph->forward_order = (tiny_graph_node_t**)malloc(graph->num_nodes * sizeof(tiny_graph_node_t*));
    if (graph->forward_order == NULL) {
        return TINY_ERR_AI_NO_MEM;
    }
    
    graph->backward_order = (tiny_graph_node_t**)malloc(graph->num_nodes * sizeof(tiny_graph_node_t*));
    if (graph->backward_order == NULL) {
        free(graph->forward_order);
        graph->forward_order = NULL;
        return TINY_ERR_AI_NO_MEM;
    }
    
    // Initialize: reset visited flags and calculate in_degree for all nodes
    int* in_degree = (int*)malloc(graph->num_nodes * sizeof(int));
    if (in_degree == NULL) {
        free(graph->forward_order);
        free(graph->backward_order);
        graph->forward_order = NULL;
        graph->backward_order = NULL;
        return TINY_ERR_AI_NO_MEM;
    }
    
    for (int i = 0; i < graph->num_nodes; i++) {
        graph->nodes[i]->visited = false;
        in_degree[i] = graph->nodes[i]->in_degree;
    }
    
    // Topological sort using Kahn's algorithm
    // Step 1: Find all nodes with in_degree = 0 (no dependencies)
    int queue_start = 0;
    int queue_end = 0;
    tiny_graph_node_t** queue = (tiny_graph_node_t**)malloc(graph->num_nodes * sizeof(tiny_graph_node_t*));
    if (queue == NULL) {
        free(in_degree);
        free(graph->forward_order);
        free(graph->backward_order);
        graph->forward_order = NULL;
        graph->backward_order = NULL;
        return TINY_ERR_AI_NO_MEM;
    }
    
    // Add all nodes with in_degree = 0 to queue
    for (int i = 0; i < graph->num_nodes; i++) {
        if (in_degree[i] == 0) {
            queue[queue_end++] = graph->nodes[i];
        }
    }
    
    // Step 2: Process nodes in topological order
    int forward_idx = 0;
    while (queue_start < queue_end) {
        // Dequeue a node
        tiny_graph_node_t* node = queue[queue_start++];
        
        // Add to forward order
        graph->forward_order[forward_idx++] = node;
        node->visited = true;
        
        // Reduce in_degree of all children
        for (int i = 0; i < node->num_children; i++) {
            tiny_graph_node_t* child = node->children[i];
            // Find child's index in graph
            int child_idx = -1;
            for (int j = 0; j < graph->num_nodes; j++) {
                if (graph->nodes[j] == child) {
                    child_idx = j;
                    break;
                }
            }
            if (child_idx >= 0) {
                in_degree[child_idx]--;
                // If in_degree becomes 0, add to queue
                if (in_degree[child_idx] == 0) {
                    queue[queue_end++] = child;
                }
            }
        }
    }
    
    // Step 3: Check for cycles
    if (forward_idx < graph->num_nodes) {
        // Not all nodes were processed, cycle detected
        free(in_degree);
        free(queue);
        free(graph->forward_order);
        free(graph->backward_order);
        graph->forward_order = NULL;
        graph->backward_order = NULL;
        return TINY_ERR_AI_INVALID_STATE;  // Cycle detected
    }
    
    // Step 4: Build backward order (reverse of forward order)
    for (int i = 0; i < graph->num_nodes; i++) {
        graph->backward_order[i] = graph->forward_order[graph->num_nodes - 1 - i];
    }
    
    free(in_degree);
    free(queue);
    
    return TINY_OK;
}

tiny_error_t tiny_graph_forward(tiny_graph_t* graph)
{
    if (graph == NULL || !graph->initialized) {
        return graph == NULL ? TINY_ERR_AI_NULL_POINTER : TINY_ERR_AI_UNINITIALIZED;
    }
    
    if (graph->forward_order == NULL) {
        return TINY_ERR_AI_INVALID_STATE;  // Must call build_order first
    }
    
    // Execute nodes in topological order (forward order)
    for (int i = 0; i < graph->num_nodes; i++) {
        tiny_graph_node_t* node = graph->forward_order[i];
        
        // Check if node has forward function
        if (node->forward_func == NULL) {
            // Node doesn't have forward function yet (operator not implemented)
            // Skip for now, but this should be an error in production
            continue;
        }
        
        // Execute forward function
        node->forward_func(node);
    }
    
    return TINY_OK;
}

tiny_error_t tiny_graph_backward(tiny_graph_t* graph)
{
    if (graph == NULL || !graph->initialized) {
        return graph == NULL ? TINY_ERR_AI_NULL_POINTER : TINY_ERR_AI_UNINITIALIZED;
    }
    
    if (!graph->training_mode) {
        // Not in training mode, skip backward propagation
        return TINY_OK;
    }
    
    if (graph->backward_order == NULL) {
        return TINY_ERR_AI_INVALID_STATE;  // Must call build_order first
    }
    
    // Execute nodes in reverse topological order (backward order)
    for (int i = 0; i < graph->num_nodes; i++) {
        tiny_graph_node_t* node = graph->backward_order[i];
        
        // Check if node needs gradients
        bool needs_grad = false;
        for (int j = 0; j < node->num_outputs; j++) {
            if (node->outputs[j] != NULL) {
#if TINY_AI_ENABLE_GRADIENTS
                if (tiny_tensor_get_requires_grad(node->outputs[j])) {
                    needs_grad = true;
                    break;
                }
#endif
            }
        }
        
        if (!needs_grad) {
            continue;  // Skip nodes that don't need gradients
        }
        
        // Check if node has backward function
        if (node->backward_func == NULL) {
            // Node doesn't have backward function yet (operator not implemented)
            // Skip for now, but this should be an error in production
            continue;
        }
        
        // Execute backward function
        node->backward_func(node);
    }
    
    return TINY_OK;
}

/* ============================================================================
 * GRAPH CONFIGURATION
 * ============================================================================ */

void tiny_graph_set_training_mode(tiny_graph_t* graph, bool training)
{
    if (graph == NULL || !graph->initialized) {
        return;
    }
    graph->training_mode = training;
}

bool tiny_graph_get_training_mode(const tiny_graph_t* graph)
{
    if (graph == NULL || !graph->initialized) {
        return false;
    }
    return graph->training_mode;
}

/* ============================================================================
 * UTILITY
 * ============================================================================ */

void tiny_graph_print_info(const tiny_graph_t* graph)
{
    if (graph == NULL) {
        printf("Graph: NULL\n");
        return;
    }
    
    if (!graph->initialized) {
        printf("Graph: Uninitialized\n");
        return;
    }
    
    printf("Computation Graph:\n");
    printf("  Nodes: %d\n", graph->num_nodes);
    printf("  Capacity: %d\n", graph->capacity);
    printf("  Training mode: %s\n", graph->training_mode ? "true" : "false");
    
    for (int i = 0; i < graph->num_nodes; i++) {
        printf("  Node %d: op_type=%d, inputs=%d, outputs=%d\n",
               i, graph->nodes[i]->op_type,
               graph->nodes[i]->num_inputs,
               graph->nodes[i]->num_outputs);
    }
}

