#include "node_espnow.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#define NODE_ESPNOW_MAGIC 0x4E45
#define NODE_ESPNOW_VERSION 1
#define NODE_ESPNOW_WORKER_NAME "node_espnow"
#define NODE_ESPNOW_TASK_WAIT_MS 50
#define NODE_ESPNOW_BROADCAST_MAC "\xFF\xFF\xFF\xFF\xFF\xFF"
#define NODE_ESPNOW_MAX_RX_SESSIONS 4
#define NODE_ACK_FLAG_ALL_ACKED 0x01U
#define NODE_TX_INTER_CHUNK_DELAY_MS 2
#define NODE_QOS2_COMPLETED_CACHE_SIZE 8
#define NODE_QOS2_COMPLETED_TTL_MS 60000

typedef enum
{
    NODE_FRAME_START = 1,
    NODE_FRAME_DATA = 2,
    NODE_FRAME_END = 3,
    NODE_FRAME_ACK = 4,
} node_espnow_frame_type_t;

typedef enum
{
    NODE_EVENT_SEND_REQUEST = 1,
    NODE_EVENT_SEND_DONE = 2,
    NODE_EVENT_RX_FRAME = 3,
} node_espnow_event_type_t;

typedef struct __attribute__((packed))
{
    uint16_t magic;
    uint8_t version;
    uint8_t type;
    uint32_t transfer_id;
    uint32_t total_len;
    uint16_t chunk_idx;
    uint16_t chunk_total;
    uint16_t payload_len;
    uint8_t flags;
    uint8_t reserved;
} node_espnow_frame_header_t;

typedef struct
{
    node_espnow_event_type_t type;
    union
    {
        struct
        {
            uint8_t peer_mac[6];
            uint8_t *payload;
            size_t payload_len;
        } send_req;
        struct
        {
            uint8_t peer_mac[6];
            esp_now_send_status_t status;
        } send_done;
        struct
        {
            uint8_t peer_mac[6];
            uint16_t len;
            uint8_t data[ESP_NOW_MAX_DATA_LEN];
        } rx;
    } data;
} node_espnow_event_t;

/* [STATE] TX lifecycle for one logical batch transfer. */
typedef struct
{
    bool active;
    bool sent_start;
    bool sent_end;
    uint8_t peer_mac[6];
    uint8_t *payload;
    size_t payload_len;
    uint32_t transfer_id;
    uint16_t chunk_total;
    uint16_t next_chunk_to_send;
    uint16_t acked_chunks;
    uint16_t tx_attempts;
    uint16_t tx_success_frames;
    uint16_t tx_failed_frames;
    uint8_t retry_count;
    int64_t started_us;
    int64_t last_tx_us;
    int64_t last_progress_us;
    uint8_t *acked_bitmap;
} node_espnow_tx_session_t;

/* [STATE] RX reassembly state for one peer + transfer_id. */
typedef struct
{
    bool active;
    bool got_end;
    uint8_t peer_mac[6];
    uint32_t transfer_id;
    uint32_t total_len;
    uint16_t chunk_total;
    uint16_t received_chunks;
    uint8_t *payload;
    uint8_t *received_bitmap;
    int64_t started_us;
    int64_t last_activity_us;
} node_espnow_rx_session_t;

typedef struct
{
    bool valid;
    uint8_t peer_mac[6];
    uint32_t transfer_id;
    int64_t completed_us;
} node_espnow_completed_transfer_t;

typedef struct
{
    bool inited;
    node_espnow_config_t cfg;
    node_espnow_handlers_t handlers;
    node_espnow_rx_batch_cb_t rx_batch_cb;
    void *rx_user_ctx;
    QueueHandle_t event_queue;
    TaskHandle_t worker_task;
    node_espnow_tx_session_t tx_session;
    node_espnow_rx_session_t rx_sessions[NODE_ESPNOW_MAX_RX_SESSIONS];
    node_espnow_completed_transfer_t completed_transfers[NODE_QOS2_COMPLETED_CACHE_SIZE];
    uint32_t transfer_seed;
} node_espnow_ctx_t;

static const char *TAG = "NODE_ESPNOW";
static node_espnow_ctx_t g_ctx = {0};

static void node_espnow_worker_task(void *arg);
static void node_espnow_handle_event(const node_espnow_event_t *evt);
static void node_espnow_handle_rx_frame(const uint8_t peer_mac[6], const uint8_t *data, uint16_t len);
static void node_espnow_handle_ack_frame(const uint8_t peer_mac[6], const node_espnow_frame_header_t *hdr, uint16_t payload_len);
static void node_espnow_handle_data_frame(const uint8_t peer_mac[6], const node_espnow_frame_header_t *hdr,
                                          const uint8_t *payload, uint16_t payload_len);
static esp_err_t node_espnow_send_tx_frame(node_espnow_frame_type_t type, uint16_t chunk_idx,
                                           const uint8_t *payload, uint16_t payload_len, uint8_t flags);
static esp_err_t node_espnow_send_ack_frame(const uint8_t peer_mac[6], const node_espnow_frame_header_t *rx_hdr,
                                            uint16_t chunk_idx, uint8_t flags);
static void node_espnow_try_send_more(void);
static void node_espnow_finish_tx_session(esp_err_t reason);
static void node_espnow_cleanup_rx_sessions(int64_t now_us, bool force);
static void node_espnow_release_rx_session(node_espnow_rx_session_t *session);
static void node_espnow_maybe_complete_rx_session(node_espnow_rx_session_t *session);
static node_espnow_rx_session_t *node_espnow_find_rx_session(const uint8_t peer_mac[6], uint32_t transfer_id);
static node_espnow_rx_session_t *node_espnow_alloc_rx_session(const uint8_t peer_mac[6], uint32_t transfer_id,
                                                              uint32_t total_len, uint16_t chunk_total);
static bool node_espnow_is_bit_set(const uint8_t *bitmap, uint16_t idx);
static void node_espnow_set_bit(uint8_t *bitmap, uint16_t idx);
static void node_espnow_mark_ack(uint16_t chunk_idx);
static bool node_espnow_chunk_is_acked(uint16_t chunk_idx);
static bool node_espnow_all_chunks_acked(void);
static esp_err_t node_espnow_ensure_peer(const uint8_t peer_mac[6]);
static uint16_t node_espnow_max_chunk_payload(void);
static bool node_espnow_validate_data_bounds(const node_espnow_frame_header_t *hdr, uint16_t payload_len);
static void node_espnow_cleanup_completed_cache(int64_t now_us);
static bool node_espnow_qos2_is_completed(const uint8_t peer_mac[6], uint32_t transfer_id, int64_t now_us);
static void node_espnow_qos2_mark_completed(const uint8_t peer_mac[6], uint32_t transfer_id, int64_t now_us);

void node_espnow_default_config(node_espnow_config_t *config)
{
    if (config == NULL)
    {
        return;
    }

    memset(config, 0, sizeof(*config));
    config->channel = 1;
    config->chunk_payload_bytes = 180;
    config->tx_window_size = 4;
    config->ack_timeout_ms = 400;
    config->max_retries = 4;
    config->session_timeout_ms = 5000;
    config->max_batch_bytes = 64 * 1024;
    config->qos_default = NODE_ESPNOW_QOS1;
    config->worker_stack_size = 4096;
    config->worker_priority = 5;
    config->event_queue_len = 24;
}

static void node_espnow_send_cb(const wifi_tx_info_t *info, esp_now_send_status_t status)
{
    if (!g_ctx.inited || info == NULL || g_ctx.event_queue == NULL)
    {
        return;
    }

    node_espnow_event_t evt = {0};
    evt.type = NODE_EVENT_SEND_DONE;
    memcpy(evt.data.send_done.peer_mac, info->des_addr, 6);
    evt.data.send_done.status = status;
    (void)xQueueSend(g_ctx.event_queue, &evt, 0);
}

static void node_espnow_recv_cb(const esp_now_recv_info_t *recv_info, const uint8_t *data, int len)
{
    if (!g_ctx.inited || recv_info == NULL || data == NULL || len <= 0 || g_ctx.event_queue == NULL)
    {
        return;
    }
    if (len > ESP_NOW_MAX_DATA_LEN)
    {
        return;
    }

    node_espnow_event_t evt = {0};
    evt.type = NODE_EVENT_RX_FRAME;
    memcpy(evt.data.rx.peer_mac, recv_info->src_addr, 6);
    evt.data.rx.len = (uint16_t)len;
    memcpy(evt.data.rx.data, data, (size_t)len);
    (void)xQueueSend(g_ctx.event_queue, &evt, 0);
}

esp_err_t node_espnow_init(const node_espnow_config_t *config,
                           const node_espnow_handlers_t *handlers)
{
    if (g_ctx.inited)
    {
        return ESP_ERR_INVALID_STATE;
    }
    if (config == NULL)
    {
        return ESP_ERR_INVALID_ARG;
    }

    memset(&g_ctx, 0, sizeof(g_ctx));
    g_ctx.cfg = *config;
    if (handlers != NULL)
    {
        g_ctx.handlers.tx_result_cb = handlers->tx_result_cb;
        g_ctx.handlers.user_ctx = handlers->user_ctx;
    }

    if (g_ctx.cfg.event_queue_len == 0 || g_ctx.cfg.worker_stack_size < 2048)
    {
        return ESP_ERR_INVALID_ARG;
    }

    uint16_t max_chunk = node_espnow_max_chunk_payload();
    if (g_ctx.cfg.chunk_payload_bytes == 0 || g_ctx.cfg.chunk_payload_bytes > max_chunk)
    {
        g_ctx.cfg.chunk_payload_bytes = max_chunk;
    }
    if (g_ctx.cfg.tx_window_size == 0)
    {
        g_ctx.cfg.tx_window_size = 1;
    }
    if (g_ctx.cfg.ack_timeout_ms == 0)
    {
        g_ctx.cfg.ack_timeout_ms = 300;
    }
    if (g_ctx.cfg.session_timeout_ms < g_ctx.cfg.ack_timeout_ms)
    {
        g_ctx.cfg.session_timeout_ms = g_ctx.cfg.ack_timeout_ms * 4;
    }

    ESP_ERROR_CHECK(esp_netif_init());
    esp_err_t ret = esp_event_loop_create_default();
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE)
    {
        return ret;
    }

    wifi_init_config_t wifi_cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wifi_cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_channel(g_ctx.cfg.channel, WIFI_SECOND_CHAN_NONE));
    ESP_ERROR_CHECK(esp_now_init());

    ESP_ERROR_CHECK(esp_now_register_send_cb(node_espnow_send_cb));
    ESP_ERROR_CHECK(esp_now_register_recv_cb(node_espnow_recv_cb));

    g_ctx.event_queue = xQueueCreate(g_ctx.cfg.event_queue_len, sizeof(node_espnow_event_t));
    if (g_ctx.event_queue == NULL)
    {
        return ESP_ERR_NO_MEM;
    }

    BaseType_t created = xTaskCreate(node_espnow_worker_task,
                                     NODE_ESPNOW_WORKER_NAME,
                                     g_ctx.cfg.worker_stack_size,
                                     NULL,
                                     g_ctx.cfg.worker_priority,
                                     &g_ctx.worker_task);
    if (created != pdPASS)
    {
        vQueueDelete(g_ctx.event_queue);
        g_ctx.event_queue = NULL;
        return ESP_ERR_NO_MEM;
    }

    g_ctx.inited = true;
    ESP_LOGI(TAG, "Unified library initialized (qos=%d, chunk=%u, window=%u)",
             g_ctx.cfg.qos_default, g_ctx.cfg.chunk_payload_bytes, g_ctx.cfg.tx_window_size);
    return ESP_OK;
}

esp_err_t node_espnow_set_rx_batch_cb(node_espnow_rx_batch_cb_t cb, void *user_ctx)
{
    if (!g_ctx.inited)
    {
        return ESP_ERR_INVALID_STATE;
    }
    g_ctx.rx_batch_cb = cb;
    g_ctx.rx_user_ctx = user_ctx;
    return ESP_OK;
}

node_espnow_qos_t node_espnow_get_qos(void)
{
    return g_ctx.cfg.qos_default;
}

esp_err_t node_espnow_send_to(const uint8_t peer_mac[6], const void *payload, size_t payload_len)
{
    if (!g_ctx.inited || peer_mac == NULL || payload == NULL || payload_len == 0)
    {
        return ESP_ERR_INVALID_ARG;
    }
    if (g_ctx.tx_session.active)
    {
        return ESP_ERR_INVALID_STATE;
    }
    if (payload_len > g_ctx.cfg.max_batch_bytes)
    {
        return ESP_ERR_INVALID_SIZE;
    }

    uint8_t *copied = (uint8_t *)malloc(payload_len);
    if (copied == NULL)
    {
        return ESP_ERR_NO_MEM;
    }
    memcpy(copied, payload, payload_len);

    node_espnow_event_t evt = {0};
    evt.type = NODE_EVENT_SEND_REQUEST;
    memcpy(evt.data.send_req.peer_mac, peer_mac, 6);
    evt.data.send_req.payload = copied;
    evt.data.send_req.payload_len = payload_len;

    if (xQueueSend(g_ctx.event_queue, &evt, pdMS_TO_TICKS(100)) != pdTRUE)
    {
        free(copied);
        return ESP_ERR_TIMEOUT;
    }

    return ESP_OK;
}

void node_espnow_deinit(void)
{
    if (!g_ctx.inited)
    {
        return;
    }

    if (g_ctx.worker_task != NULL)
    {
        vTaskDelete(g_ctx.worker_task);
    }
    if (g_ctx.event_queue != NULL)
    {
        vQueueDelete(g_ctx.event_queue);
    }

    node_espnow_finish_tx_session(ESP_ERR_INVALID_STATE);
    node_espnow_cleanup_rx_sessions(0, true);

    esp_now_unregister_send_cb();
    esp_now_unregister_recv_cb();
    esp_now_deinit();
    esp_wifi_stop();
    esp_wifi_deinit();
    memset(&g_ctx, 0, sizeof(g_ctx));
}

static void node_espnow_worker_task(void *arg)
{
    (void)arg;
    node_espnow_event_t evt;

    while (1)
    {
        /* [FLOW] ISR/driver callbacks only enqueue events; state changes happen here. */
        if (xQueueReceive(g_ctx.event_queue, &evt, pdMS_TO_TICKS(NODE_ESPNOW_TASK_WAIT_MS)) == pdTRUE)
        {
            node_espnow_handle_event(&evt);
        }

        int64_t now_us = esp_timer_get_time();
        node_espnow_cleanup_rx_sessions(now_us, false);
        node_espnow_cleanup_completed_cache(now_us);

        if (g_ctx.tx_session.active)
        {
            int64_t ack_deadline_us = (int64_t)g_ctx.cfg.ack_timeout_ms * 1000;
            int64_t session_deadline_us = (int64_t)g_ctx.cfg.session_timeout_ms * 1000;

            if (g_ctx.tx_session.last_tx_us > 0 && (now_us - g_ctx.tx_session.last_tx_us) >= ack_deadline_us)
            {
                if (node_espnow_all_chunks_acked())
                {
                    node_espnow_finish_tx_session(ESP_OK);
                }
                else if (g_ctx.tx_session.retry_count >= g_ctx.cfg.max_retries)
                {
                    node_espnow_finish_tx_session(ESP_ERR_TIMEOUT);
                }
                else
                {
                    /* [RELIABILITY] Timeout path: restart from START and resend pending chunks. */
                    g_ctx.tx_session.retry_count++;
                    g_ctx.tx_session.sent_start = false;
                    g_ctx.tx_session.sent_end = false;
                    g_ctx.tx_session.next_chunk_to_send = 0;
                    ESP_LOGW(TAG, "transfer=%lu retry=%u",
                             (unsigned long)g_ctx.tx_session.transfer_id,
                             (unsigned)g_ctx.tx_session.retry_count);
                    node_espnow_try_send_more();
                }
            }

            if ((now_us - g_ctx.tx_session.started_us) >= session_deadline_us)
            {
                node_espnow_finish_tx_session(ESP_ERR_TIMEOUT);
            }
        }
    }
}

static void node_espnow_handle_event(const node_espnow_event_t *evt)
{
    if (evt == NULL)
    {
        return;
    }

    switch (evt->type)
    {
    case NODE_EVENT_SEND_REQUEST:
    {
        if (g_ctx.tx_session.active)
        {
            free(evt->data.send_req.payload);
            break;
        }

        if (node_espnow_ensure_peer(evt->data.send_req.peer_mac) != ESP_OK)
        {
            free(evt->data.send_req.payload);
            break;
        }

        node_espnow_tx_session_t *s = &g_ctx.tx_session;
        memset(s, 0, sizeof(*s));
        s->active = true;
        memcpy(s->peer_mac, evt->data.send_req.peer_mac, 6);
        s->payload = evt->data.send_req.payload;
        s->payload_len = evt->data.send_req.payload_len;
        s->transfer_id = ++g_ctx.transfer_seed;

        /* [FLOW] Split one logical payload into DATA chunks. */
        size_t chunk_total = (s->payload_len + g_ctx.cfg.chunk_payload_bytes - 1U) / g_ctx.cfg.chunk_payload_bytes;
        if (chunk_total == 0 || chunk_total > UINT16_MAX)
        {
            node_espnow_finish_tx_session(ESP_ERR_INVALID_SIZE);
            break;
        }
        s->chunk_total = (uint16_t)chunk_total;
        s->started_us = esp_timer_get_time();
        s->last_progress_us = s->started_us;
        s->last_tx_us = 0;

        size_t bitmap_bytes = (s->chunk_total + 7U) / 8U;
        s->acked_bitmap = (uint8_t *)calloc(bitmap_bytes, 1);
        if (s->acked_bitmap == NULL)
        {
            node_espnow_finish_tx_session(ESP_ERR_NO_MEM);
            break;
        }

        ESP_LOGI(TAG, "TX transfer=%lu start bytes=%u chunks=%u",
                 (unsigned long)s->transfer_id,
                 (unsigned)s->payload_len,
                 (unsigned)s->chunk_total);
        node_espnow_try_send_more();
        break;
    }

    case NODE_EVENT_SEND_DONE:
    {
        if (g_ctx.tx_session.active &&
            memcmp(g_ctx.tx_session.peer_mac, evt->data.send_done.peer_mac, 6) == 0)
        {
            if (evt->data.send_done.status == ESP_NOW_SEND_SUCCESS)
            {
                g_ctx.tx_session.tx_success_frames++;
            }
            else
            {
                g_ctx.tx_session.tx_failed_frames++;
            }
        }
        break;
    }

    case NODE_EVENT_RX_FRAME:
        node_espnow_handle_rx_frame(evt->data.rx.peer_mac, evt->data.rx.data, evt->data.rx.len);
        break;

    default:
        break;
    }
}

static void node_espnow_handle_rx_frame(const uint8_t peer_mac[6], const uint8_t *data, uint16_t len)
{
    if (len < (uint16_t)sizeof(node_espnow_frame_header_t))
    {
        ESP_LOGW(TAG, "drop short frame len=%u", (unsigned)len);
        return;
    }

    node_espnow_frame_header_t hdr = {0};
    memcpy(&hdr, data, sizeof(hdr));
    uint16_t payload_len = (uint16_t)(len - sizeof(hdr));
    const uint8_t *payload = data + sizeof(hdr);

    if (hdr.magic != NODE_ESPNOW_MAGIC || hdr.version != NODE_ESPNOW_VERSION)
    {
        ESP_LOGW(TAG, "drop invalid header magic=0x%04X version=%u", hdr.magic, (unsigned)hdr.version);
        return;
    }
    if (hdr.payload_len != payload_len)
    {
        ESP_LOGW(TAG, "drop payload mismatch hdr=%u actual=%u", (unsigned)hdr.payload_len, (unsigned)payload_len);
        return;
    }

    switch ((node_espnow_frame_type_t)hdr.type)
    {
    case NODE_FRAME_ACK:
        node_espnow_handle_ack_frame(peer_mac, &hdr, payload_len);
        break;
    case NODE_FRAME_START:
    case NODE_FRAME_DATA:
    case NODE_FRAME_END:
        node_espnow_handle_data_frame(peer_mac, &hdr, payload, payload_len);
        break;
    default:
        ESP_LOGW(TAG, "drop unknown frame type=%u", (unsigned)hdr.type);
        break;
    }
}

static void node_espnow_handle_ack_frame(const uint8_t peer_mac[6], const node_espnow_frame_header_t *hdr, uint16_t payload_len)
{
    if (payload_len != 0U)
    {
        ESP_LOGW(TAG, "drop ACK with payload len=%u", (unsigned)payload_len);
        return;
    }

    node_espnow_tx_session_t *s = &g_ctx.tx_session;
    if (!s->active)
    {
        return;
    }
    if (memcmp(s->peer_mac, peer_mac, 6) != 0 || hdr->transfer_id != s->transfer_id)
    {
        return;
    }

    /* [RELIABILITY] ACK marks one chunk, or all chunks via ALL_ACKED flag. */
    if ((hdr->flags & NODE_ACK_FLAG_ALL_ACKED) != 0U)
    {
        for (uint16_t i = 0; i < s->chunk_total; i++)
        {
            node_espnow_mark_ack(i);
        }
        s->acked_chunks = s->chunk_total;
    }
    else if (hdr->chunk_idx < s->chunk_total && !node_espnow_chunk_is_acked(hdr->chunk_idx))
    {
        node_espnow_mark_ack(hdr->chunk_idx);
        s->acked_chunks++;
    }

    s->last_progress_us = esp_timer_get_time();
    if (node_espnow_all_chunks_acked())
    {
        node_espnow_finish_tx_session(ESP_OK);
    }
    else
    {
        node_espnow_try_send_more();
    }
}

static void node_espnow_handle_data_frame(const uint8_t peer_mac[6], const node_espnow_frame_header_t *hdr,
                                          const uint8_t *payload, uint16_t payload_len)
{
    if (hdr->total_len == 0 || hdr->total_len > g_ctx.cfg.max_batch_bytes || hdr->chunk_total == 0)
    {
        ESP_LOGW(TAG, "drop invalid data header transfer=%lu total=%lu chunks=%u",
                 (unsigned long)hdr->transfer_id, (unsigned long)hdr->total_len, (unsigned)hdr->chunk_total);
        return;
    }

    node_espnow_rx_session_t *s = node_espnow_find_rx_session(peer_mac, hdr->transfer_id);

    if ((node_espnow_frame_type_t)hdr->type == NODE_FRAME_START)
    {
        if (payload_len != 0U)
        {
            ESP_LOGW(TAG, "drop START with payload transfer=%lu", (unsigned long)hdr->transfer_id);
            return;
        }
        if (s == NULL)
        {
            if (g_ctx.cfg.qos_default == NODE_ESPNOW_QOS2 &&
                node_espnow_qos2_is_completed(peer_mac, hdr->transfer_id, esp_timer_get_time()))
            {
                /* [RELIABILITY] QoS2 duplicate START: ack all and skip app re-delivery. */
                (void)node_espnow_send_ack_frame(peer_mac, hdr, hdr->chunk_total, NODE_ACK_FLAG_ALL_ACKED);
                ESP_LOGI(TAG, "QoS2 duplicate transfer=%lu ignored (already delivered)",
                         (unsigned long)hdr->transfer_id);
                return;
            }

            s = node_espnow_alloc_rx_session(peer_mac, hdr->transfer_id, hdr->total_len, hdr->chunk_total);
            if (s == NULL)
            {
                ESP_LOGW(TAG, "drop START no resources transfer=%lu", (unsigned long)hdr->transfer_id);
                return;
            }
            ESP_LOGI(TAG, "RX transfer=%lu start from %02X:%02X:%02X:%02X:%02X:%02X bytes=%lu chunks=%u",
                     (unsigned long)s->transfer_id,
                     s->peer_mac[0], s->peer_mac[1], s->peer_mac[2],
                     s->peer_mac[3], s->peer_mac[4], s->peer_mac[5],
                     (unsigned long)s->total_len, (unsigned)s->chunk_total);
        }
        s->last_activity_us = esp_timer_get_time();
        return;
    }

    if (s == NULL)
    {
        ESP_LOGW(TAG, "drop frame no session type=%u transfer=%lu", (unsigned)hdr->type, (unsigned long)hdr->transfer_id);
        return;
    }

    if (s->total_len != hdr->total_len || s->chunk_total != hdr->chunk_total)
    {
        ESP_LOGW(TAG, "drop transfer mismatch transfer=%lu", (unsigned long)hdr->transfer_id);
        return;
    }

    if ((node_espnow_frame_type_t)hdr->type == NODE_FRAME_END)
    {
        if (payload_len != 0U)
        {
            ESP_LOGW(TAG, "drop END with payload transfer=%lu", (unsigned long)hdr->transfer_id);
            return;
        }
        s->got_end = true;
        s->last_activity_us = esp_timer_get_time();
        /* [FLOW] Completion requires END frame and all DATA chunks present. */
        node_espnow_maybe_complete_rx_session(s);
        return;
    }

    if (!node_espnow_validate_data_bounds(hdr, payload_len))
    {
        ESP_LOGW(TAG, "drop DATA out-of-range transfer=%lu idx=%u payload=%u",
                 (unsigned long)hdr->transfer_id, (unsigned)hdr->chunk_idx, (unsigned)payload_len);
        return;
    }

    if (!node_espnow_is_bit_set(s->received_bitmap, hdr->chunk_idx))
    {
        uint32_t offset = (uint32_t)hdr->chunk_idx * g_ctx.cfg.chunk_payload_bytes;
        memcpy(s->payload + offset, payload, payload_len);
        node_espnow_set_bit(s->received_bitmap, hdr->chunk_idx);
        s->received_chunks++;
    }

    s->last_activity_us = esp_timer_get_time();
    /* [RELIABILITY] ACK each DATA chunk for QoS1 progress tracking. */
    (void)node_espnow_send_ack_frame(peer_mac, hdr, hdr->chunk_idx, 0U);
    node_espnow_maybe_complete_rx_session(s);
}

static void node_espnow_try_send_more(void)
{
    node_espnow_tx_session_t *s = &g_ctx.tx_session;
    if (!s->active)
    {
        return;
    }

    if (!s->sent_start)
    {
        if (node_espnow_send_tx_frame(NODE_FRAME_START, 0, NULL, 0, 0) == ESP_OK)
        {
            s->sent_start = true;
            s->tx_attempts++;
            s->last_tx_us = esp_timer_get_time();
        }
        else
        {
            node_espnow_finish_tx_session(ESP_FAIL);
            return;
        }
    }

    uint8_t sent_in_burst = 0;
    /* [FLOW] Sliding-window send: emit next unacked chunks up to tx_window_size. */
    while (s->next_chunk_to_send < s->chunk_total && sent_in_burst < g_ctx.cfg.tx_window_size)
    {
        uint16_t idx = s->next_chunk_to_send++;
        if (node_espnow_chunk_is_acked(idx))
        {
            continue;
        }

        uint32_t offset = (uint32_t)idx * g_ctx.cfg.chunk_payload_bytes;
        uint16_t remaining = (uint16_t)(s->payload_len - offset);
        uint16_t chunk_len = remaining > g_ctx.cfg.chunk_payload_bytes ? g_ctx.cfg.chunk_payload_bytes : remaining;

        if (node_espnow_send_tx_frame(NODE_FRAME_DATA, idx, s->payload + offset, chunk_len, 0) != ESP_OK)
        {
            node_espnow_finish_tx_session(ESP_FAIL);
            return;
        }

        s->tx_attempts++;
        sent_in_burst++;
        s->last_tx_us = esp_timer_get_time();
        /* [RELIABILITY] Small pacing improves large-transfer stability on noisy links. */
        if (NODE_TX_INTER_CHUNK_DELAY_MS > 0)
        {
            vTaskDelay(pdMS_TO_TICKS(NODE_TX_INTER_CHUNK_DELAY_MS));
        }
    }

    if (s->next_chunk_to_send >= s->chunk_total && !s->sent_end)
    {
        if (node_espnow_send_tx_frame(NODE_FRAME_END, s->chunk_total, NULL, 0, 0) == ESP_OK)
        {
            s->tx_attempts++;
            s->sent_end = true;
            s->last_tx_us = esp_timer_get_time();
        }
        else
        {
            node_espnow_finish_tx_session(ESP_FAIL);
            return;
        }
    }

    if (g_ctx.cfg.qos_default == NODE_ESPNOW_QOS0 && s->sent_end)
    {
        node_espnow_finish_tx_session(ESP_OK);
    }
}

static esp_err_t node_espnow_send_tx_frame(node_espnow_frame_type_t type, uint16_t chunk_idx,
                                           const uint8_t *payload, uint16_t payload_len, uint8_t flags)
{
    node_espnow_tx_session_t *s = &g_ctx.tx_session;
    if (!s->active)
    {
        return ESP_ERR_INVALID_STATE;
    }

    uint8_t frame[ESP_NOW_MAX_DATA_LEN] = {0};
    node_espnow_frame_header_t hdr = {0};
    hdr.magic = NODE_ESPNOW_MAGIC;
    hdr.version = NODE_ESPNOW_VERSION;
    hdr.type = (uint8_t)type;
    hdr.transfer_id = s->transfer_id;
    hdr.total_len = (uint32_t)s->payload_len;
    hdr.chunk_idx = chunk_idx;
    hdr.chunk_total = s->chunk_total;
    hdr.payload_len = payload_len;
    hdr.flags = flags;

    size_t hdr_len = sizeof(hdr);
    if ((hdr_len + payload_len) > sizeof(frame))
    {
        return ESP_ERR_INVALID_SIZE;
    }

    memcpy(frame, &hdr, hdr_len);
    if (payload_len > 0U)
    {
        memcpy(frame + hdr_len, payload, payload_len);
    }

    return esp_now_send(s->peer_mac, frame, hdr_len + payload_len);
}

static esp_err_t node_espnow_send_ack_frame(const uint8_t peer_mac[6], const node_espnow_frame_header_t *rx_hdr,
                                            uint16_t chunk_idx, uint8_t flags)
{
    if (node_espnow_ensure_peer(peer_mac) != ESP_OK)
    {
        return ESP_FAIL;
    }

    node_espnow_frame_header_t ack = {0};
    ack.magic = NODE_ESPNOW_MAGIC;
    ack.version = NODE_ESPNOW_VERSION;
    ack.type = NODE_FRAME_ACK;
    ack.transfer_id = rx_hdr->transfer_id;
    ack.total_len = rx_hdr->total_len;
    ack.chunk_idx = chunk_idx;
    ack.chunk_total = rx_hdr->chunk_total;
    ack.payload_len = 0;
    ack.flags = flags;
    return esp_now_send(peer_mac, (const uint8_t *)&ack, sizeof(ack));
}

static void node_espnow_finish_tx_session(esp_err_t reason)
{
    node_espnow_tx_session_t *s = &g_ctx.tx_session;
    if (!s->active)
    {
        if (s->payload != NULL)
        {
            free(s->payload);
            s->payload = NULL;
        }
        if (s->acked_bitmap != NULL)
        {
            free(s->acked_bitmap);
            s->acked_bitmap = NULL;
        }
        return;
    }

    node_espnow_tx_result_t result = {0};
    result.transfer_id = s->transfer_id;
    result.total_bytes = s->payload_len;
    result.total_chunks = s->chunk_total;
    result.acked_chunks = s->acked_chunks;
    result.tx_attempts = s->tx_attempts;
    result.tx_success_frames = s->tx_success_frames;
    result.tx_failed_frames = s->tx_failed_frames;
    result.duration_ms = (uint32_t)((esp_timer_get_time() - s->started_us) / 1000);
    result.fail_reason = reason;

    if (reason == ESP_OK)
    {
        ESP_LOGI(TAG, "TX transfer=%lu done in %lu ms (attempts=%u, retries=%u)",
                 (unsigned long)s->transfer_id,
                 (unsigned long)result.duration_ms,
                 (unsigned)result.tx_attempts,
                 (unsigned)s->retry_count);
    }
    else
    {
        ESP_LOGE(TAG, "TX transfer=%lu failed (%s), acked=%u/%u, retries=%u",
                 (unsigned long)s->transfer_id,
                 esp_err_to_name(reason),
                 (unsigned)result.acked_chunks,
                 (unsigned)result.total_chunks,
                 (unsigned)s->retry_count);
    }

    if (g_ctx.handlers.tx_result_cb != NULL)
    {
        g_ctx.handlers.tx_result_cb(s->peer_mac, &result, g_ctx.handlers.user_ctx);
    }

    if (s->payload != NULL)
    {
        free(s->payload);
    }
    if (s->acked_bitmap != NULL)
    {
        free(s->acked_bitmap);
    }
    memset(s, 0, sizeof(*s));
}

static void node_espnow_cleanup_rx_sessions(int64_t now_us, bool force)
{
    for (size_t i = 0; i < NODE_ESPNOW_MAX_RX_SESSIONS; i++)
    {
        node_espnow_rx_session_t *s = &g_ctx.rx_sessions[i];
        if (!s->active)
        {
            continue;
        }

        if (!force && now_us > 0)
        {
            int64_t timeout_us = (int64_t)g_ctx.cfg.session_timeout_ms * 1000;
            if ((now_us - s->last_activity_us) < timeout_us)
            {
                continue;
            }

            ESP_LOGW(TAG, "RX transfer=%lu timeout, received=%u/%u end=%d",
                     (unsigned long)s->transfer_id,
                     (unsigned)s->received_chunks,
                     (unsigned)s->chunk_total,
                     (int)s->got_end);
        }

        node_espnow_release_rx_session(s);
    }
}

static void node_espnow_release_rx_session(node_espnow_rx_session_t *session)
{
    if (session == NULL)
    {
        return;
    }
    if (session->payload != NULL)
    {
        free(session->payload);
    }
    if (session->received_bitmap != NULL)
    {
        free(session->received_bitmap);
    }
    memset(session, 0, sizeof(*session));
}

static void node_espnow_maybe_complete_rx_session(node_espnow_rx_session_t *session)
{
    if (session == NULL || !session->active || !session->got_end || session->received_chunks < session->chunk_total)
    {
        return;
    }

    /* [FLOW] Notify TX full assembly via ALL_ACKED, then deliver payload to app callback. */
    node_espnow_frame_header_t rx_hdr = {0};
    rx_hdr.transfer_id = session->transfer_id;
    rx_hdr.total_len = session->total_len;
    rx_hdr.chunk_total = session->chunk_total;
    (void)node_espnow_send_ack_frame(session->peer_mac, &rx_hdr, session->chunk_total, NODE_ACK_FLAG_ALL_ACKED);

    node_espnow_rx_batch_t batch = {0};
    batch.transfer_id = session->transfer_id;
    batch.total_bytes = session->total_len;
    batch.total_chunks = session->chunk_total;
    batch.received_chunks = session->received_chunks;
    batch.duration_ms = (uint32_t)((esp_timer_get_time() - session->started_us) / 1000);
    batch.payload = session->payload;
    batch.payload_len = session->total_len;

    ESP_LOGI(TAG, "RX transfer=%lu completed in %lu ms (%u chunks)",
             (unsigned long)batch.transfer_id,
             (unsigned long)batch.duration_ms,
             (unsigned)batch.total_chunks);

    if (g_ctx.rx_batch_cb != NULL)
    {
        g_ctx.rx_batch_cb(session->peer_mac, &batch, g_ctx.rx_user_ctx);
    }

    if (g_ctx.cfg.qos_default == NODE_ESPNOW_QOS2)
    {
        node_espnow_qos2_mark_completed(session->peer_mac, session->transfer_id, esp_timer_get_time());
    }

    node_espnow_release_rx_session(session);
}

static void node_espnow_cleanup_completed_cache(int64_t now_us)
{
    if (now_us <= 0 || g_ctx.cfg.qos_default != NODE_ESPNOW_QOS2)
    {
        return;
    }

    int64_t ttl_us = (int64_t)NODE_QOS2_COMPLETED_TTL_MS * 1000;
    for (size_t i = 0; i < NODE_QOS2_COMPLETED_CACHE_SIZE; i++)
    {
        node_espnow_completed_transfer_t *c = &g_ctx.completed_transfers[i];
        if (!c->valid)
        {
            continue;
        }
        if ((now_us - c->completed_us) >= ttl_us)
        {
            memset(c, 0, sizeof(*c));
        }
    }
}

static bool node_espnow_qos2_is_completed(const uint8_t peer_mac[6], uint32_t transfer_id, int64_t now_us)
{
    if (peer_mac == NULL || g_ctx.cfg.qos_default != NODE_ESPNOW_QOS2)
    {
        return false;
    }

    for (size_t i = 0; i < NODE_QOS2_COMPLETED_CACHE_SIZE; i++)
    {
        node_espnow_completed_transfer_t *c = &g_ctx.completed_transfers[i];
        if (c->valid && c->transfer_id == transfer_id && memcmp(c->peer_mac, peer_mac, 6) == 0)
        {
            if (now_us > 0)
            {
                c->completed_us = now_us;
            }
            return true;
        }
    }
    return false;
}

static void node_espnow_qos2_mark_completed(const uint8_t peer_mac[6], uint32_t transfer_id, int64_t now_us)
{
    if (peer_mac == NULL || g_ctx.cfg.qos_default != NODE_ESPNOW_QOS2)
    {
        return;
    }

    size_t replace_idx = 0;
    int64_t oldest_us = INT64_MAX;

    for (size_t i = 0; i < NODE_QOS2_COMPLETED_CACHE_SIZE; i++)
    {
        node_espnow_completed_transfer_t *c = &g_ctx.completed_transfers[i];
        if (c->valid && c->transfer_id == transfer_id && memcmp(c->peer_mac, peer_mac, 6) == 0)
        {
            c->completed_us = now_us;
            return;
        }
        if (!c->valid)
        {
            replace_idx = i;
            oldest_us = -1;
            break;
        }
        if (c->completed_us < oldest_us)
        {
            oldest_us = c->completed_us;
            replace_idx = i;
        }
    }

    node_espnow_completed_transfer_t *dst = &g_ctx.completed_transfers[replace_idx];
    memset(dst, 0, sizeof(*dst));
    dst->valid = true;
    memcpy(dst->peer_mac, peer_mac, 6);
    dst->transfer_id = transfer_id;
    dst->completed_us = now_us;
}

static node_espnow_rx_session_t *node_espnow_find_rx_session(const uint8_t peer_mac[6], uint32_t transfer_id)
{
    for (size_t i = 0; i < NODE_ESPNOW_MAX_RX_SESSIONS; i++)
    {
        node_espnow_rx_session_t *s = &g_ctx.rx_sessions[i];
        if (s->active && s->transfer_id == transfer_id && memcmp(s->peer_mac, peer_mac, 6) == 0)
        {
            return s;
        }
    }
    return NULL;
}

static node_espnow_rx_session_t *node_espnow_alloc_rx_session(const uint8_t peer_mac[6], uint32_t transfer_id,
                                                              uint32_t total_len, uint16_t chunk_total)
{
    for (size_t i = 0; i < NODE_ESPNOW_MAX_RX_SESSIONS; i++)
    {
        if (!g_ctx.rx_sessions[i].active)
        {
            node_espnow_rx_session_t *s = &g_ctx.rx_sessions[i];
            size_t bitmap_bytes = (chunk_total + 7U) / 8U;
            uint8_t *payload = (uint8_t *)malloc(total_len);
            uint8_t *bitmap = (uint8_t *)calloc(bitmap_bytes, 1);
            if (payload == NULL || bitmap == NULL)
            {
                free(payload);
                free(bitmap);
                return NULL;
            }

            memset(s, 0, sizeof(*s));
            s->active = true;
            memcpy(s->peer_mac, peer_mac, 6);
            s->transfer_id = transfer_id;
            s->total_len = total_len;
            s->chunk_total = chunk_total;
            s->payload = payload;
            s->received_bitmap = bitmap;
            s->started_us = esp_timer_get_time();
            s->last_activity_us = s->started_us;
            return s;
        }
    }
    return NULL;
}

static bool node_espnow_is_bit_set(const uint8_t *bitmap, uint16_t idx)
{
    if (bitmap == NULL)
    {
        return false;
    }
    return (bitmap[idx / 8U] & (1U << (idx % 8U))) != 0U;
}

static void node_espnow_set_bit(uint8_t *bitmap, uint16_t idx)
{
    if (bitmap == NULL)
    {
        return;
    }
    bitmap[idx / 8U] |= (1U << (idx % 8U));
}

static bool node_espnow_validate_data_bounds(const node_espnow_frame_header_t *hdr, uint16_t payload_len)
{
    if (hdr->type != NODE_FRAME_DATA || payload_len == 0U || hdr->chunk_idx >= hdr->chunk_total)
    {
        return false;
    }

    uint32_t offset = (uint32_t)hdr->chunk_idx * g_ctx.cfg.chunk_payload_bytes;
    if (offset >= hdr->total_len)
    {
        return false;
    }
    return (offset + payload_len) <= hdr->total_len;
}

static bool node_espnow_chunk_is_acked(uint16_t chunk_idx)
{
    node_espnow_tx_session_t *s = &g_ctx.tx_session;
    if (s->acked_bitmap == NULL || chunk_idx >= s->chunk_total)
    {
        return false;
    }
    return (s->acked_bitmap[chunk_idx / 8U] & (1U << (chunk_idx % 8U))) != 0U;
}

static void node_espnow_mark_ack(uint16_t chunk_idx)
{
    node_espnow_tx_session_t *s = &g_ctx.tx_session;
    if (s->acked_bitmap == NULL || chunk_idx >= s->chunk_total)
    {
        return;
    }
    s->acked_bitmap[chunk_idx / 8U] |= (1U << (chunk_idx % 8U));
}

static bool node_espnow_all_chunks_acked(void)
{
    node_espnow_tx_session_t *s = &g_ctx.tx_session;
    if (!s->active)
    {
        return false;
    }
    return s->acked_chunks >= s->chunk_total;
}

static esp_err_t node_espnow_ensure_peer(const uint8_t peer_mac[6])
{
    if (esp_now_is_peer_exist(peer_mac))
    {
        return ESP_OK;
    }

    esp_now_peer_info_t peer = {0};
    memcpy(peer.peer_addr, peer_mac, 6);
    peer.channel = g_ctx.cfg.channel;
    peer.ifidx = WIFI_IF_STA;
    peer.encrypt = false;

    if (memcmp(peer_mac, NODE_ESPNOW_BROADCAST_MAC, 6) == 0)
    {
        peer.channel = 0;
    }

    return esp_now_add_peer(&peer);
}

static uint16_t node_espnow_max_chunk_payload(void)
{
    size_t hdr_len = sizeof(node_espnow_frame_header_t);
    if (ESP_NOW_MAX_DATA_LEN <= hdr_len)
    {
        return 1;
    }
    return (uint16_t)(ESP_NOW_MAX_DATA_LEN - hdr_len);
}
