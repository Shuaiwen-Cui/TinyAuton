/**
 * @file AIoTNode.cpp
 * @brief RX application based on unified node_espnow library.
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_log.h"
#include "esp_psram.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "nvs_flash.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "node_espnow.h"
#include "node_exit.h"
#include "node_led.h"
#include "node_rtc.h"
#include "node_sdcard.h"
#include "node_spi.h"
#include "node_timer.h"

#ifdef __cplusplus
extern "C" {
#endif

static const char *TAG = "ESP_NOW_RX_APP";
static const uint16_t kFrameMagic = 0x4E44U;
static const uint8_t kFrameVersion = 1U;
static const size_t kPrintableArrayCount = 128U;
static const size_t kPrintablePayloadBytes = kPrintableArrayCount * sizeof(uint16_t);
static const size_t kMediumPayloadBytes = 600U;
static const size_t kRawPayloadBytes = 300U;
static const size_t kLargeArrayCount = 4096U;
static const size_t kLargePayloadBytes = kLargeArrayCount * sizeof(uint16_t);
static const uint64_t kLargeExpectedSum = 8386560ULL;
static const size_t kMaxBatchBytes = 16384U;
static const uint8_t kTotalCases = 6U;
static const uint8_t kTotalQosRounds = 3U;
static const uint8_t kCaseDiscoveryReq = 0xF0U;
static const uint8_t kCaseDiscoveryResp = 0xF1U;

static uint32_t s_rx_total = 0;
static uint32_t s_rx_ok = 0;
static uint32_t s_rx_fail = 0;
static uint32_t s_rx_printable_ok = 0;
static uint32_t s_rx_medium_ok = 0;
static uint32_t s_rx_string_ok = 0;
static uint32_t s_rx_struct_ok = 0;
static uint32_t s_rx_raw_ok = 0;
static uint32_t s_rx_large_ok = 0;
static bool s_case_seen[kTotalCases + 1] = {0};
static bool s_case_pass[kTotalCases + 1] = {0};
static bool s_summary_printed = false;
static bool s_matrix_done[kTotalQosRounds][kTotalCases + 1] = {0};
static bool s_matrix_pass[kTotalQosRounds][kTotalCases + 1] = {0};
static bool s_matrix_summary_printed = false;
static uint8_t s_qos_round_idx = 0U;
static bool s_round_complete_pending = false;
static bool s_all_rounds_finished = false;

typedef struct __attribute__((packed))
{
    uint16_t magic;
    uint8_t version;
    uint8_t case_id;
    uint32_t seq;
    uint32_t payload_len;
    uint32_t checksum;
} test_frame_header_t;

enum
{
    TEST_CASE_PRINTABLE_ARRAY = 1,
    TEST_CASE_MEDIUM = 2,
    TEST_CASE_STRING = 3,
    TEST_CASE_STRUCT = 4,
    TEST_CASE_RAW_BYTES = 5,
    TEST_CASE_LARGE = 6,
};

typedef struct __attribute__((packed))
{
    uint32_t tick_s;
    float temperature_c;
    int16_t accel_x;
    int16_t accel_y;
    int16_t accel_z;
    uint8_t status_flags;
    char tag[8];
} demo_sensor_packet_t;
static const node_espnow_qos_t kQosPlan[kTotalQosRounds] = {
    NODE_ESPNOW_QOS0,
    NODE_ESPNOW_QOS1,
    NODE_ESPNOW_QOS2,
};

static const char *qos_name(node_espnow_qos_t qos)
{
    switch (qos)
    {
    case NODE_ESPNOW_QOS0:
        return "QOS0";
    case NODE_ESPNOW_QOS1:
        return "QOS1";
    case NODE_ESPNOW_QOS2:
        return "QOS2";
    default:
        return "UNKNOWN_QOS";
    }
}

static node_espnow_qos_t current_qos(void)
{
    return kQosPlan[s_qos_round_idx];
}

static const char *result_name(bool done, bool pass)
{
    if (!done)
    {
        return "NOT_RUN";
    }
    return pass ? "PASS" : "FAIL";
}

static const char *test_case_name(uint8_t case_id)
{
    switch (case_id)
    {
    case TEST_CASE_PRINTABLE_ARRAY:
        return "PRINTABLE_LONG_ARRAY";
    case TEST_CASE_MEDIUM:
        return "MEDIUM_MULTI_CHUNK";
    case TEST_CASE_STRING:
        return "STRING_TEXT";
    case TEST_CASE_STRUCT:
        return "STRUCT_PACKET";
    case TEST_CASE_RAW_BYTES:
        return "RAW_BINARY";
    case TEST_CASE_LARGE:
        return "LARGE_VECTOR_0_TO_4095";
    default:
        return "UNKNOWN";
    }
}

static const char *test_case_goal(uint8_t case_id)
{
    switch (case_id)
    {
    case TEST_CASE_PRINTABLE_ARRAY:
        return "Array payload can be reconstructed and printed";
    case TEST_CASE_MEDIUM:
        return "Medium payload split/reassemble path works";
    case TEST_CASE_STRING:
        return "Null-terminated string transmitted correctly";
    case TEST_CASE_STRUCT:
        return "Packed struct fields are preserved";
    case TEST_CASE_RAW_BYTES:
        return "Arbitrary binary payload is intact";
    case TEST_CASE_LARGE:
        return "Large payload multi-chunk transfer succeeds";
    default:
        return "Unknown goal";
    }
}

static bool all_cases_seen(void)
{
    for (uint8_t i = 1U; i <= kTotalCases; i++)
    {
        if (!s_case_seen[i])
        {
            return false;
        }
    }
    return true;
}

static void log_rx_summary_once(void)
{
    if (s_summary_printed)
    {
        return;
    }

    ESP_LOGI(TAG, "========== RX TEST SUMMARY (%s) ==========", qos_name(current_qos()));
    for (uint8_t i = 1U; i <= kTotalCases; i++)
    {
        s_matrix_done[s_qos_round_idx][i] = s_case_seen[i];
        s_matrix_pass[s_qos_round_idx][i] = s_case_pass[i];
        ESP_LOGI(TAG, "Stage-%u %-22s : %s",
                 (unsigned)i,
                 test_case_name(i),
                 s_case_pass[i] ? "PASS" : "FAIL");
    }
    ESP_LOGI(TAG, "RX one-shot verification finished for %s.", qos_name(current_qos()));
    s_summary_printed = true;
}

static void log_rx_matrix_summary_once(void)
{
    if (s_matrix_summary_printed)
    {
        return;
    }

    ESP_LOGI(TAG, "========== RX FINAL MATRIX SUMMARY ==========");
    for (uint8_t q = 0U; q < kTotalQosRounds; q++)
    {
        ESP_LOGI(TAG, "-- %s --", qos_name(kQosPlan[q]));
        for (uint8_t c = 1U; c <= kTotalCases; c++)
        {
            ESP_LOGI(TAG, "Stage-%u %-22s : %s",
                     (unsigned)c,
                     test_case_name(c),
                     result_name(s_matrix_done[q][c], s_matrix_pass[q][c]));
        }
    }
    s_matrix_summary_printed = true;
}

static void reset_round_state(void)
{
    s_rx_total = 0;
    s_rx_ok = 0;
    s_rx_fail = 0;
    s_rx_printable_ok = 0;
    s_rx_medium_ok = 0;
    s_rx_string_ok = 0;
    s_rx_struct_ok = 0;
    s_rx_raw_ok = 0;
    s_rx_large_ok = 0;
    memset(s_case_seen, 0, sizeof(s_case_seen));
    memset(s_case_pass, 0, sizeof(s_case_pass));
    s_summary_printed = false;
    s_round_complete_pending = false;
}

static uint32_t calc_checksum32(const uint8_t *data, size_t len)
{
    uint32_t sum = 0U;
    for (size_t i = 0; i < len; i++)
    {
        sum += data[i];
    }
    return sum;
}

static bool build_control_frame(uint8_t case_id, uint32_t seq, const char *text, uint8_t *frame, size_t frame_cap, size_t *out_len)
{
    if (text == NULL || frame == NULL || out_len == NULL || frame_cap < sizeof(test_frame_header_t))
    {
        return false;
    }

    size_t payload_len = strlen(text) + 1U;
    if ((sizeof(test_frame_header_t) + payload_len) > frame_cap)
    {
        return false;
    }

    uint8_t *payload = frame + sizeof(test_frame_header_t);
    memcpy(payload, text, payload_len);

    test_frame_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = kFrameMagic;
    hdr.version = kFrameVersion;
    hdr.case_id = case_id;
    hdr.seq = seq;
    hdr.payload_len = (uint32_t)payload_len;
    hdr.checksum = calc_checksum32(payload, payload_len);
    memcpy(frame, &hdr, sizeof(hdr));

    *out_len = sizeof(test_frame_header_t) + payload_len;
    return true;
}

static bool validate_pattern(const uint8_t *payload, size_t payload_len, uint8_t case_id, uint32_t seq)
{
    for (size_t i = 0; i < payload_len; i++)
    {
        uint8_t expected = (uint8_t)((seq + case_id + (uint32_t)i) & 0xFFU);
        if (payload[i] != expected)
        {
            return false;
        }
    }
    return true;
}

static const char *expected_string_for_seq(uint32_t seq)
{
    static const char *messages[] = {
        "hello from TX: ESPNOW string payload",
        "NexNode test: pointers send pointed bytes",
        "CASE_STRING: printable text over node_espnow",
    };
    return messages[seq % (sizeof(messages) / sizeof(messages[0]))];
}

static uint16_t make_printable_value(uint32_t seq, uint32_t idx)
{
    return (uint16_t)((seq * 10U + idx) & 0xFFFFU);
}

static bool validate_printable_array(const uint16_t *arr, size_t count, uint32_t seq)
{
    if (arr == NULL || count != kPrintableArrayCount)
    {
        return false;
    }
    for (size_t i = 0; i < count; i++)
    {
        if (arr[i] != make_printable_value(seq, i))
        {
            return false;
        }
    }
    return true;
}

static void log_printable_array(const uint16_t *arr, size_t count, uint32_t seq)
{
    ESP_LOGI(TAG, "Printable array content (seq=%lu, count=%u):",
             (unsigned long)seq,
             (unsigned)count);
    for (size_t i = 0; i < count; i += 16U)
    {
        size_t end = (i + 16U < count) ? (i + 16U) : count;
        char line[160];
        int pos = snprintf(line, sizeof(line), "[%03u..%03u] ",
                           (unsigned)i,
                           (unsigned)(end - 1U));
        for (size_t j = i; j < end && pos > 0 && (size_t)pos < sizeof(line); j++)
        {
            int wrote = snprintf(line + pos, sizeof(line) - (size_t)pos, "%u ", (unsigned)arr[j]);
            if (wrote <= 0)
            {
                break;
            }
            pos += wrote;
        }
        ESP_LOGI(TAG, "%s", line);
    }
}

static void log_printable_array_compare(const uint16_t *arr, size_t count, uint32_t seq)
{
    if (arr == NULL || count == 0U)
    {
        return;
    }

    size_t show = count < 8U ? count : 8U;
    char expected_line[192];
    char actual_line[192];
    int ep = snprintf(expected_line, sizeof(expected_line), "Expected first[%u]:", (unsigned)show);
    int ap = snprintf(actual_line, sizeof(actual_line), "Received first[%u]:", (unsigned)show);
    for (size_t i = 0; i < show; i++)
    {
        if (ep > 0 && (size_t)ep < sizeof(expected_line))
        {
            ep += snprintf(expected_line + ep, sizeof(expected_line) - (size_t)ep, " %u",
                           (unsigned)make_printable_value(seq, (uint32_t)i));
        }
        if (ap > 0 && (size_t)ap < sizeof(actual_line))
        {
            ap += snprintf(actual_line + ap, sizeof(actual_line) - (size_t)ap, " %u", (unsigned)arr[i]);
        }
    }
    ESP_LOGI(TAG, "%s", expected_line);
    ESP_LOGI(TAG, "%s", actual_line);
}

static bool validate_large_array(const uint8_t *payload, size_t payload_len)
{
    if (payload == NULL || payload_len != kLargePayloadBytes)
    {
        return false;
    }

    const uint16_t *vec = (const uint16_t *)payload;
    if (vec[0] != 0U || vec[kLargeArrayCount - 1U] != (uint16_t)(kLargeArrayCount - 1U))
    {
        return false;
    }

    uint64_t sum = 0U;
    for (size_t i = 0; i < kLargeArrayCount; i++)
    {
        sum += vec[i];
    }
    return sum == kLargeExpectedSum;
}

static bool validate_struct_case(const uint8_t *payload, size_t payload_len, uint32_t seq)
{
    if (payload == NULL || payload_len != sizeof(demo_sensor_packet_t))
    {
        return false;
    }

    demo_sensor_packet_t pkt;
    memcpy(&pkt, payload, sizeof(pkt));

    if (pkt.tick_s != (seq * 3U))
    {
        return false;
    }

    if (pkt.accel_x != (int16_t)(100 + (int16_t)seq) ||
        pkt.accel_y != (int16_t)(-50 - (int16_t)seq) ||
        pkt.accel_z != (int16_t)(1024 + (int16_t)(seq % 32U)) ||
        pkt.status_flags != (uint8_t)(seq & 0x0FU))
    {
        return false;
    }

    return memcmp(pkt.tag, "NEXNODE", 8) == 0;
}

static void log_struct_case(const uint8_t *payload)
{
    demo_sensor_packet_t pkt;
    memcpy(&pkt, payload, sizeof(pkt));
    ESP_LOGI(TAG,
             "Struct fields: tick_s=%lu temp=%.2f accel=[%d,%d,%d] status=0x%02X tag=%.8s",
             (unsigned long)pkt.tick_s,
             (double)pkt.temperature_c,
             (int)pkt.accel_x,
             (int)pkt.accel_y,
             (int)pkt.accel_z,
             (unsigned)pkt.status_flags,
             pkt.tag);
}

static void log_struct_case_compare(const uint8_t *payload, uint32_t seq)
{
    demo_sensor_packet_t actual;
    demo_sensor_packet_t expected;
    memcpy(&actual, payload, sizeof(actual));
    memset(&expected, 0, sizeof(expected));
    expected.tick_s = seq * 3U;
    expected.temperature_c = 23.5f + (float)(seq % 7U) * 0.25f;
    expected.accel_x = (int16_t)(100 + (int16_t)seq);
    expected.accel_y = (int16_t)(-50 - (int16_t)seq);
    expected.accel_z = (int16_t)(1024 + (int16_t)(seq % 32U));
    expected.status_flags = (uint8_t)(seq & 0x0FU);
    memcpy(expected.tag, "NEXNODE", 8);

    ESP_LOGI(TAG,
             "Expected struct: tick_s=%lu temp=%.2f accel=[%d,%d,%d] status=0x%02X tag=%.8s",
             (unsigned long)expected.tick_s,
             (double)expected.temperature_c,
             (int)expected.accel_x,
             (int)expected.accel_y,
             (int)expected.accel_z,
             (unsigned)expected.status_flags,
             expected.tag);
    ESP_LOGI(TAG,
             "Received struct: tick_s=%lu temp=%.2f accel=[%d,%d,%d] status=0x%02X tag=%.8s",
             (unsigned long)actual.tick_s,
             (double)actual.temperature_c,
             (int)actual.accel_x,
             (int)actual.accel_y,
             (int)actual.accel_z,
             (unsigned)actual.status_flags,
             actual.tag);
}

static bool validate_string_case(const uint8_t *payload, size_t payload_len)
{
    if (payload == NULL || payload_len == 0U)
    {
        return false;
    }
    return payload[payload_len - 1U] == '\0';
}

static void log_raw_preview(const uint8_t *payload, size_t payload_len)
{
    size_t show = payload_len < 16U ? payload_len : 16U;
    char line[128];
    int pos = snprintf(line, sizeof(line), "RAW preview:");
    for (size_t i = 0; i < show && pos > 0 && (size_t)pos < sizeof(line); i++)
    {
        int wrote = snprintf(line + pos, sizeof(line) - (size_t)pos, " %02X", payload[i]);
        if (wrote <= 0)
        {
            break;
        }
        pos += wrote;
    }
    ESP_LOGI(TAG, "%s", line);
}

static void log_pattern_compare_preview(const uint8_t *payload, size_t payload_len, uint8_t case_id, uint32_t seq, const char *label)
{
    size_t show = payload_len < 16U ? payload_len : 16U;
    char exp_line[192];
    char act_line[192];
    int ep = snprintf(exp_line, sizeof(exp_line), "%s expected[%u]:", label, (unsigned)show);
    int ap = snprintf(act_line, sizeof(act_line), "%s received[%u]:", label, (unsigned)show);
    for (size_t i = 0; i < show; i++)
    {
        uint8_t expected = (uint8_t)((seq + case_id + (uint32_t)i) & 0xFFU);
        if (ep > 0 && (size_t)ep < sizeof(exp_line))
        {
            ep += snprintf(exp_line + ep, sizeof(exp_line) - (size_t)ep, " %02X", expected);
        }
        if (ap > 0 && (size_t)ap < sizeof(act_line))
        {
            ap += snprintf(act_line + ap, sizeof(act_line) - (size_t)ap, " %02X", payload[i]);
        }
    }
    ESP_LOGI(TAG, "%s", exp_line);
    ESP_LOGI(TAG, "%s", act_line);
}

static void log_large_array_compare(const uint8_t *payload, size_t payload_len)
{
    if (payload == NULL || payload_len < sizeof(uint16_t))
    {
        return;
    }
    const uint16_t *vec = (const uint16_t *)payload;
    size_t count = payload_len / sizeof(uint16_t);
    uint64_t sum = 0U;
    for (size_t i = 0; i < count; i++)
    {
        sum += vec[i];
    }
    ESP_LOGI(TAG,
             "Large expected: first=0 last=%u sum=%llu bytes=%u",
             (unsigned)(kLargeArrayCount - 1U),
             (unsigned long long)kLargeExpectedSum,
             (unsigned)kLargePayloadBytes);
    ESP_LOGI(TAG,
             "Large received: first=%u last=%u sum=%llu bytes=%u",
             (unsigned)vec[0],
             (unsigned)vec[count - 1U],
             (unsigned long long)sum,
             (unsigned)payload_len);
}

static void log_local_mac(void)
{
    uint8_t local_mac[6] = {0};
    if (esp_wifi_get_mac(WIFI_IF_STA, local_mac) == ESP_OK)
    {
        ESP_LOGI(TAG, "Local MAC: %02X:%02X:%02X:%02X:%02X:%02X",
                 local_mac[0], local_mac[1], local_mac[2],
                 local_mac[3], local_mac[4], local_mac[5]);
    }
}

static void on_rx_batch(const uint8_t peer_mac[6], const node_espnow_rx_batch_t *batch, void *user_ctx)
{
    (void)user_ctx;
    s_rx_total++;

    bool valid = false;
    uint8_t case_id = 0U;
    uint32_t case_seq = 0U;
    size_t case_payload_len = 0U;

    if (batch->payload != NULL && batch->payload_len >= sizeof(test_frame_header_t))
    {
        test_frame_header_t hdr;
        memset(&hdr, 0, sizeof(hdr));
        memcpy(&hdr, batch->payload, sizeof(hdr));
        const uint8_t *case_payload = batch->payload + sizeof(hdr);
        size_t case_payload_bytes = batch->payload_len - sizeof(hdr);
        uint32_t checksum = calc_checksum32(case_payload, case_payload_bytes);

        case_id = hdr.case_id;
        case_seq = hdr.seq;
        case_payload_len = case_payload_bytes;
        valid = (hdr.magic == kFrameMagic) &&
                (hdr.version == kFrameVersion) &&
                (hdr.payload_len == case_payload_bytes) &&
                (hdr.checksum == checksum);

        if (valid && hdr.case_id == kCaseDiscoveryReq)
        {
            uint8_t reply[96];
            size_t reply_len = 0U;
            if (build_control_frame(kCaseDiscoveryResp, hdr.seq, "DISCOVERY_RESP", reply, sizeof(reply), &reply_len))
            {
                esp_err_t ret = node_espnow_send_to(peer_mac, reply, reply_len);
                ESP_LOGI(TAG,
                         "Discovery request from %02X:%02X:%02X:%02X:%02X:%02X seq=%lu -> reply %s",
                         peer_mac[0], peer_mac[1], peer_mac[2], peer_mac[3], peer_mac[4], peer_mac[5],
                         (unsigned long)hdr.seq,
                         ret == ESP_OK ? "queued" : esp_err_to_name(ret));
            }
            return;
        }

        if (valid && hdr.case_id == kCaseDiscoveryResp)
        {
            return;
        }

        if (valid && hdr.case_id == TEST_CASE_PRINTABLE_ARRAY)
        {
            log_printable_array_compare((const uint16_t *)case_payload,
                                        case_payload_bytes / sizeof(uint16_t),
                                        hdr.seq);
            valid = (case_payload_bytes == kPrintablePayloadBytes) &&
                    ((case_payload_bytes % sizeof(uint16_t)) == 0U) &&
                    validate_printable_array((const uint16_t *)case_payload,
                                             case_payload_bytes / sizeof(uint16_t),
                                             hdr.seq);
            if (valid)
            {
                s_rx_printable_ok++;
                log_printable_array((const uint16_t *)case_payload,
                                    case_payload_bytes / sizeof(uint16_t),
                                    hdr.seq);
            }
        }
        else if (valid && hdr.case_id == TEST_CASE_MEDIUM)
        {
            log_pattern_compare_preview(case_payload, case_payload_bytes, hdr.case_id, hdr.seq, "MEDIUM");
            valid = (case_payload_bytes == kMediumPayloadBytes) &&
                    validate_pattern(case_payload, case_payload_bytes, hdr.case_id, hdr.seq);
            if (valid)
            {
                s_rx_medium_ok++;
            }
        }
        else if (valid && hdr.case_id == TEST_CASE_STRING)
        {
            const char *expected = expected_string_for_seq(hdr.seq);
            ESP_LOGI(TAG, "Expected string: %s", expected);
            ESP_LOGI(TAG, "Received string: %s", (const char *)case_payload);
            valid = validate_string_case(case_payload, case_payload_bytes);
            valid = valid && (strcmp((const char *)case_payload, expected) == 0);
            if (valid)
            {
                s_rx_string_ok++;
            }
        }
        else if (valid && hdr.case_id == TEST_CASE_STRUCT)
        {
            log_struct_case_compare(case_payload, hdr.seq);
            valid = validate_struct_case(case_payload, case_payload_bytes, hdr.seq);
            if (valid)
            {
                s_rx_struct_ok++;
                log_struct_case(case_payload);
            }
        }
        else if (valid && hdr.case_id == TEST_CASE_RAW_BYTES)
        {
            log_pattern_compare_preview(case_payload, case_payload_bytes, hdr.case_id, hdr.seq, "RAW");
            valid = (case_payload_bytes == kRawPayloadBytes) &&
                    validate_pattern(case_payload, case_payload_bytes, hdr.case_id, hdr.seq);
            if (valid)
            {
                s_rx_raw_ok++;
                log_raw_preview(case_payload, case_payload_bytes);
            }
        }
        else if (valid && hdr.case_id == TEST_CASE_LARGE)
        {
            log_large_array_compare(case_payload, case_payload_bytes);
            valid = validate_large_array(case_payload, case_payload_bytes);
            if (valid)
            {
                s_rx_large_ok++;
            }
        }
        else
        {
            valid = false;
        }
    }

    if (valid)
    {
        s_rx_ok++;
        led_toggle();
    }
    else
    {
        s_rx_fail++;
    }

    ESP_LOGI(TAG,
             "RX %s | qos=%s case=%s seq=%lu payload_bytes=%lu | from=%02X:%02X:%02X:%02X:%02X:%02X transfer=%lu chunks=%u/%u duration=%lums",
             valid ? "PASS" : "FAIL",
             qos_name(current_qos()),
             test_case_name(case_id),
             (unsigned long)case_seq,
             (unsigned long)case_payload_len,
             peer_mac[0], peer_mac[1], peer_mac[2], peer_mac[3], peer_mac[4], peer_mac[5],
             (unsigned long)batch->transfer_id,
             (unsigned)batch->received_chunks,
             (unsigned)batch->total_chunks,
             (unsigned long)batch->duration_ms);

    if (case_id >= 1U && case_id <= kTotalCases && !s_case_seen[case_id])
    {
        s_case_seen[case_id] = true;
        s_case_pass[case_id] = valid;
        ESP_LOGI(TAG, "========== RX STAGE %u/%u ==========", (unsigned)case_id, (unsigned)kTotalCases);
        ESP_LOGI(TAG, "QoS round: %s", qos_name(current_qos()));
        ESP_LOGI(TAG, "Case: %s", test_case_name(case_id));
        ESP_LOGI(TAG, "Goal: %s", test_case_goal(case_id));
        ESP_LOGI(TAG, "Result: %s", valid ? "PASS" : "FAIL");
    }
    else if (case_id >= 1U && case_id <= kTotalCases)
    {
        ESP_LOGI(TAG, "Duplicate case data ignored for stage summary: case=%s", test_case_name(case_id));
    }

    if (all_cases_seen())
    {
        log_rx_summary_once();
        s_round_complete_pending = true;
    }

    if (!valid && case_id == TEST_CASE_LARGE && batch->payload_len >= sizeof(test_frame_header_t))
    {
        const uint8_t *raw = batch->payload + sizeof(test_frame_header_t);
        const uint16_t *vec = (const uint16_t *)raw;
        size_t count = (batch->payload_len - sizeof(test_frame_header_t)) / sizeof(uint16_t);
        ESP_LOGW(TAG, "Vector check fail: len=%u, first=%u, last=%u, expected_last=%u, expected_sum=%llu",
                 (unsigned)(batch->payload_len - sizeof(test_frame_header_t)),
                 (unsigned)vec[0],
                 (unsigned)vec[count - 1U],
                 (unsigned)(kLargeArrayCount - 1U),
                 (unsigned long long)kLargeExpectedSum);
    }
}

static bool start_node_espnow_with_qos(node_espnow_qos_t qos)
{
    node_espnow_config_t cfg;
    node_espnow_default_config(&cfg);
    cfg.channel = 1;
    cfg.qos_default = qos;
    cfg.chunk_payload_bytes = 160;
    cfg.tx_window_size = 1;
    cfg.ack_timeout_ms = 1200;
    cfg.max_retries = 8;
    cfg.session_timeout_ms = 30000;
    cfg.max_batch_bytes = kMaxBatchBytes;

    node_espnow_handlers_t handlers = {
        .tx_result_cb = NULL,
        .user_ctx = NULL,
    };

    esp_err_t ret = node_espnow_init(&cfg, &handlers);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "node_espnow_init failed for %s: %s", qos_name(qos), esp_err_to_name(ret));
        return false;
    }

    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    ret = node_espnow_set_rx_batch_cb(on_rx_batch, NULL);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "node_espnow_set_rx_batch_cb failed for %s: %s", qos_name(qos), esp_err_to_name(ret));
        node_espnow_deinit();
        return false;
    }

    ESP_LOGI(TAG, "node_espnow started for %s", qos_name(qos));
    return true;
}

static bool switch_to_next_qos_round(void)
{
    if (s_qos_round_idx + 1U >= kTotalQosRounds)
    {
        s_all_rounds_finished = true;
        return false;
    }

    s_qos_round_idx++;
    ESP_LOGI(TAG, "Switching to next round: %s", qos_name(current_qos()));
    node_espnow_deinit();
    reset_round_state();
    if (!start_node_espnow_with_qos(current_qos()))
    {
        s_all_rounds_finished = true;
        return false;
    }
    return true;
}

void app_main(void)
{
    esp_err_t ret;
    uint32_t flash_size;
    esp_chip_info_t chip_info;

    ESP_LOGI(TAG, "========== System Initialization ==========");
    ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    esp_flash_get_size(NULL, &flash_size);
    esp_chip_info(&chip_info);
    ESP_LOGI(TAG, "CPU cores: %d", chip_info.cores);
    ESP_LOGI(TAG, "Flash size: %ld MB", flash_size / (1024 * 1024));
    ESP_LOGI(TAG, "PSRAM size: %d bytes", esp_psram_get_size());

    ESP_LOGI(TAG, "========== Hardware Initialization ==========");
    led_init();
    exit_init();
    spi2_init();

    ESP_LOGI(TAG, "========== node_espnow Initialization ==========");
    memset(s_matrix_done, 0, sizeof(s_matrix_done));
    memset(s_matrix_pass, 0, sizeof(s_matrix_pass));
    s_matrix_summary_printed = false;
    reset_round_state();
    s_qos_round_idx = 0U;
    if (!start_node_espnow_with_qos(current_qos()))
    {
        while (1)
        {
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }

    log_local_mac();
    ESP_LOGI(TAG, "RX app started. Waiting application test frames...");
    ESP_LOGI(TAG, "QoS plan: QOS0 -> QOS1 -> QOS2");
    ESP_LOGI(TAG, "RX discovery mode: reply DISCOVERY_REQ with DISCOVERY_RESP (unicast).");
    ESP_LOGI(TAG, "Validation mode: one-shot structured verification (each case once)");
    for (uint8_t i = 1U; i <= kTotalCases; i++)
    {
        ESP_LOGI(TAG, "  Stage-%u %-22s | Goal: %s",
                 (unsigned)i,
                 test_case_name(i),
                 test_case_goal(i));
    }
    ESP_LOGI(TAG, "Large-vector spec: bytes=%u, first=0, last=%u, sum=%llu",
             (unsigned)kLargePayloadBytes,
             (unsigned)(kLargeArrayCount - 1U),
             (unsigned long long)kLargeExpectedSum);
    while (1)
    {
        if (s_round_complete_pending && !s_all_rounds_finished)
        {
            s_round_complete_pending = false;
            if (!switch_to_next_qos_round())
            {
                ESP_LOGI(TAG, "No next QoS round, RX matrix test completed.");
            }
        }
        else if (s_all_rounds_finished)
        {
            log_rx_matrix_summary_once();
        }

        vTaskDelay(pdMS_TO_TICKS(5000));
        ESP_LOGI(TAG, "Heartbeat... qos=%s rx_total=%lu rx_ok=%lu rx_fail=%lu [print=%lu medium=%lu string=%lu struct=%lu raw=%lu large=%lu]",
                 qos_name(current_qos()),
                 (unsigned long)s_rx_total,
                 (unsigned long)s_rx_ok,
                 (unsigned long)s_rx_fail,
                 (unsigned long)s_rx_printable_ok,
                 (unsigned long)s_rx_medium_ok,
                 (unsigned long)s_rx_string_ok,
                 (unsigned long)s_rx_struct_ok,
                 (unsigned long)s_rx_raw_ok,
                 (unsigned long)s_rx_large_ok);
    }
}

#ifdef __cplusplus
}
#endif