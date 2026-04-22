#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    NODE_ESPNOW_QOS0 = 0,
    NODE_ESPNOW_QOS1 = 1,
    NODE_ESPNOW_QOS2 = 2,
} node_espnow_qos_t;

typedef struct
{
    uint8_t channel;
    uint16_t chunk_payload_bytes;
    uint8_t tx_window_size;
    uint32_t ack_timeout_ms;
    uint8_t max_retries;
    uint32_t session_timeout_ms;
    uint32_t max_batch_bytes;
    node_espnow_qos_t qos_default;
    uint16_t worker_stack_size;
    uint8_t worker_priority;
    uint8_t event_queue_len;
} node_espnow_config_t;

typedef struct
{
    uint32_t transfer_id;
    uint32_t total_bytes;
    uint16_t total_chunks;
    uint16_t acked_chunks;
    uint16_t tx_attempts;
    uint16_t tx_success_frames;
    uint16_t tx_failed_frames;
    uint32_t duration_ms;
    esp_err_t fail_reason;
} node_espnow_tx_result_t;

typedef struct
{
    uint32_t transfer_id;
    uint32_t total_bytes;
    uint16_t total_chunks;
    uint16_t received_chunks;
    uint32_t duration_ms;
    const uint8_t *payload;
    size_t payload_len;
} node_espnow_rx_batch_t;

typedef void (*node_espnow_tx_result_cb_t)(const uint8_t peer_mac[6],
                                           const node_espnow_tx_result_t *result,
                                           void *user_ctx);

typedef void (*node_espnow_rx_batch_cb_t)(const uint8_t peer_mac[6],
                                          const node_espnow_rx_batch_t *batch,
                                          void *user_ctx);

typedef struct
{
    node_espnow_tx_result_cb_t tx_result_cb;
    void *user_ctx;
} node_espnow_handlers_t;

/**
 * @brief Fill a config with safe defaults.
 */
void node_espnow_default_config(node_espnow_config_t *config);

/**
 * @brief Initialize ESPNOW TX engine with config and callbacks.
 */
esp_err_t node_espnow_init(const node_espnow_config_t *config,
                           const node_espnow_handlers_t *handlers);

/**
 * @brief Send one batch byte-stream to a target peer.
 *
 * @param payload Pointer to any memory block (array/string/struct/serialized bytes).
 *                The pointer value itself is NOT transmitted, only payload bytes are sent.
 * @note The library copies payload internally, caller can free its buffer after return.
 */
esp_err_t node_espnow_send_to(const uint8_t peer_mac[6], const void *payload, size_t payload_len);

/**
 * @brief Register RX complete-batch callback.
 *
 * @note Callback runs in node_espnow worker task context.
 */
esp_err_t node_espnow_set_rx_batch_cb(node_espnow_rx_batch_cb_t cb, void *user_ctx);

/**
 * @brief Get current default qos.
 */
node_espnow_qos_t node_espnow_get_qos(void);

/**
 * @brief Deinitialize driver and release resources.
 */
void node_espnow_deinit(void);

#ifdef __cplusplus
}
#endif
