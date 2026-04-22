/**
 * @file tiny_graph.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Computation graph for forward and backward propagation
 * @version 1.0
 * @date 2025-12-12
 * @copyright Copyright (c) 2025
 *
 * @details
 * This module provides computation graph functionality for AI operations.
 * The graph manages nodes (operators) and edges (data flow) to enable
 * automatic forward and backward propagation.
 * 
 * Features:
 * - Graph-based model representation
 * - Forward and backward propagation
 * - Automatic gradient computation
 * - Topological sorting for execution order
 * 
 * Design:
 * - Lightweight graph structure for MCU
 * - Static allocation option
 * - Support for training and inference modes
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
 * @brief Operator type enumeration
 */
typedef enum
{
    TINY_AI_OP_FULLY_CONNECTED = 0,  // Fully connected (dense) layer
    TINY_AI_OP_CONV2D,                // 2D Convolution layer
    TINY_AI_OP_POOL2D,                // 2D Pooling layer (Max/Avg)
    TINY_AI_OP_BATCHNORM,             // Batch Normalization layer
    TINY_AI_OP_RELU,                  // ReLU activation
    TINY_AI_OP_SIGMOID,               // Sigmoid activation
    TINY_AI_OP_TANH,                  // Tanh activation
    TINY_AI_OP_MSE_LOSS,              // Mean Squared Error loss
    TINY_AI_OP_CROSS_ENTROPY_LOSS,    // Cross Entropy loss
    TINY_AI_OP_ADD,                   // Element-wise addition
    TINY_AI_OP_MUL,                   // Element-wise multiplication
} tiny_ai_op_type_t;

/**
 * @brief Graph node structure
 * 
 * @note Each node represents an operation in the computation graph
 */
typedef struct tiny_graph_node_t
{
    tiny_ai_op_type_t op_type;        // Operator type
    
    // Input and output tensors
    tiny_tensor_t** inputs;            // Array of input tensor pointers
    int num_inputs;                    // Number of inputs
    tiny_tensor_t** outputs;           // Array of output tensor pointers
    int num_outputs;                   // Number of outputs
    
    // Forward and backward functions
    void (*forward_func)(struct tiny_graph_node_t* node);   // Forward propagation function
    void (*backward_func)(struct tiny_graph_node_t* node);  // Backward propagation function
    
    // Operator-specific parameters (e.g., weights, bias for FC layer)
    void* params;                      // Pointer to operator parameters
    
    // Graph structure
    struct tiny_graph_node_t** children;  // Nodes that depend on this node's output
    int num_children;                     // Number of children
    struct tiny_graph_node_t** parents;   // Nodes that this node depends on
    int num_parents;                      // Number of parents
    
    // Internal flags
    bool visited;                      // For topological sort
    int in_degree;                     // Number of incoming edges (for topological sort)
} tiny_graph_node_t;

/**
 * @brief Computation graph structure
 */
typedef struct tiny_graph_t
{
    tiny_graph_node_t** nodes;         // Array of node pointers
    int num_nodes;                     // Number of nodes
    int capacity;                      // Current capacity (for dynamic growth)
    
    // Execution order (topologically sorted)
    tiny_graph_node_t** forward_order;   // Forward execution order
    tiny_graph_node_t** backward_order;  // Backward execution order
    
    // Mode
    bool training_mode;                // true for training, false for inference
    
    // Internal flags
    bool initialized;                  // Whether graph is initialized
} tiny_graph_t;

/* ============================================================================
 * FUNCTION PROTOTYPES - Graph Creation and Destruction
 * ============================================================================ */

/**
 * @brief Create a new computation graph
 * 
 * @param initial_capacity Initial capacity for nodes (can grow dynamically)
 * @return tiny_graph_t* Pointer to created graph, NULL on failure
 * 
 * @note Graph starts with training_mode = false (inference mode)
 */
tiny_graph_t* tiny_graph_create(int initial_capacity);

/**
 * @brief Destroy a computation graph and free all resources
 * 
 * @param graph Graph to destroy (can be NULL)
 * 
 * @note This does NOT destroy tensors, only the graph structure
 */
void tiny_graph_destroy(tiny_graph_t* graph);

/* ============================================================================
 * FUNCTION PROTOTYPES - Node Management
 * ============================================================================ */

/**
 * @brief Add a node to the graph
 * 
 * @param graph Graph to add node to
 * @param op_type Operator type
 * @param num_inputs Number of input tensors
 * @param num_outputs Number of output tensors
 * @return tiny_graph_node_t* Pointer to created node, NULL on failure
 */
tiny_graph_node_t* tiny_graph_add_node(tiny_graph_t* graph, tiny_ai_op_type_t op_type,
                                       int num_inputs, int num_outputs);

/**
 * @brief Connect two nodes (create an edge)
 * 
 * @param graph Graph containing the nodes
 * @param from_node Source node
 * @param from_output_idx Output index of source node
 * @param to_node Destination node
 * @param to_input_idx Input index of destination node
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @note This connects from_node's output[from_output_idx] to to_node's input[to_input_idx]
 */
tiny_error_t tiny_graph_connect(tiny_graph_t* graph, tiny_graph_node_t* from_node, int from_output_idx,
                                tiny_graph_node_t* to_node, int to_input_idx);

/* ============================================================================
 * FUNCTION PROTOTYPES - Graph Execution
 * ============================================================================ */

/**
 * @brief Build execution order (topological sort)
 * 
 * @param graph Graph to build execution order for
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @note Must be called before forward/backward propagation
 * @note Detects cycles and returns error if graph has cycles
 */
tiny_error_t tiny_graph_build_order(tiny_graph_t* graph);

/**
 * @brief Execute forward propagation
 * 
 * @param graph Graph to execute
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @note Executes nodes in topological order
 * @note Must call tiny_graph_build_order() first
 */
tiny_error_t tiny_graph_forward(tiny_graph_t* graph);

/**
 * @brief Execute backward propagation
 * 
 * @param graph Graph to execute
 * @return tiny_error_t TINY_OK on success, error code on failure
 * 
 * @note Executes nodes in reverse topological order
 * @note Must call tiny_graph_build_order() first
 * @note Only computes gradients for tensors with requires_grad = true
 */
tiny_error_t tiny_graph_backward(tiny_graph_t* graph);

/* ============================================================================
 * FUNCTION PROTOTYPES - Graph Configuration
 * ============================================================================ */

/**
 * @brief Set training mode
 * 
 * @param graph Graph to configure
 * @param training If true, enable training mode (gradients computed)
 *                 If false, enable inference mode (no gradients)
 */
void tiny_graph_set_training_mode(tiny_graph_t* graph, bool training);

/**
 * @brief Get training mode
 * 
 * @param graph Graph to query
 * @return bool True if in training mode, false if in inference mode
 */
bool tiny_graph_get_training_mode(const tiny_graph_t* graph);

/* ============================================================================
 * FUNCTION PROTOTYPES - Utility
 * ============================================================================ */

/**
 * @brief Print graph structure (for debugging)
 * 
 * @param graph Graph to print
 */
void tiny_graph_print_info(const tiny_graph_t* graph);

#ifdef __cplusplus
}
#endif

