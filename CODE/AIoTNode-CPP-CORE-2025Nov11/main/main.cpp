/**
 * @file main.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Main program for testing tiny_time module
 * @version 1.0
 * @date 2025-10-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/* DEPENDENCIES */
// ESP
#include "nvs_flash.h"
#include "esp_log.h"

#ifdef __cplusplus
extern "C" {
#endif

// FreeRTOS (must be inside extern "C" for C++ files)
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"

// ESP Timer (high-precision timer)
#include "esp_timer.h"

// WiFi (required for time sync)
#include "node_wifi.h"

// TinyToolbox
#include "tiny_time.h"

/* Variables */
const char *TAG = "tiny_time_test";

/* Timer precision test variables */
#define TIMESTAMP_COUNT 15
static TinyTimeMark_t s_timestamps[TIMESTAMP_COUNT] = {0};
static int s_timestamp_index = 0;
static bool s_timer_test_complete = false;
static esp_timer_handle_t s_timer_handle = NULL;
static SemaphoreHandle_t s_timer_mutex = NULL;

/**
 * @brief Timer callback function - records timestamp at the very beginning
 * @param arg Timer argument (not used)
 * @return None
 */
static void timer_precision_callback(void *arg)
{
    // CRITICAL: Get timestamp IMMEDIATELY at the start of callback
    // to avoid any execution overhead affecting the measurement
    TinyTimeMark_t timestamp = tiny_get_running_time();
    
    // Store timestamp in array (thread-safe access)
    if (xSemaphoreTake(s_timer_mutex, portMAX_DELAY) == pdTRUE)
    {
        if (s_timestamp_index < TIMESTAMP_COUNT)
        {
            s_timestamps[s_timestamp_index++] = timestamp;
            
            // Stop timer when we have collected all timestamps
            if (s_timestamp_index >= TIMESTAMP_COUNT)
            {
                s_timer_test_complete = true;
                esp_timer_stop(s_timer_handle);
            }
        }
        xSemaphoreGive(s_timer_mutex);
    }
}

/**
 * @brief Entry point of the program - Testing tiny_time module
 * @param None
 * @retval None
 */
void app_main(void)
{
    esp_err_t ret;

    // Initialize NVS (required for WiFi)
    ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  tiny_time Module Test Program");
    ESP_LOGI(TAG, "========================================");

    // Initialize WiFi (required for time synchronization)
    ESP_LOGI(TAG, "Initializing WiFi...");
    ret = wifi_sta_wpa2_init();
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "WiFi initialization failed!");
        return;
    }
    ESP_LOGI(TAG, "WiFi initialized successfully");

    // Wait for WiFi connection
    ESP_LOGI(TAG, "Waiting for WiFi connection...");
    EventBits_t ev = xEventGroupWaitBits(wifi_event_group, CONNECTED_BIT, 
                                         pdTRUE, pdFALSE, portMAX_DELAY);
    if (ev & CONNECTED_BIT)
    {
        ESP_LOGI(TAG, "WiFi connected!");
        
        // ============================================================
        // Test 1: Get running time (microseconds since boot)
        // ============================================================
        ESP_LOGI(TAG, "\n--- Test 1: Get Running Time ---");
        TinyTimeMark_t start_time = tiny_get_running_time();
        ESP_LOGI(TAG, "Running time: %lld microseconds", start_time);
        ESP_LOGI(TAG, "Running time: %.3f seconds", start_time / 1000000.0);

        // ============================================================
        // Test 2: Sync time with timezone
        // ============================================================
        ESP_LOGI(TAG, "\n--- Test 2: Sync Time with Timezone ---");
        ESP_LOGI(TAG, "Syncing time with timezone CST-8...");
        sync_time_with_timezone("CST-8");
        
        // Wait for time synchronization (SNTP may take a few seconds)
        ESP_LOGI(TAG, "Waiting for time synchronization...");
        vTaskDelay(5000 / portTICK_PERIOD_MS);

        // ============================================================
        // Test 3: Get current datetime
        // ============================================================
        ESP_LOGI(TAG, "\n--- Test 3: Get Current DateTime ---");
        (void)tiny_get_current_datetime(true);  // Function prints internally

        // ============================================================
        // Test 4: Measure time elapsed
        // ============================================================
        ESP_LOGI(TAG, "\n--- Test 4: Measure Time Elapsed ---");
        TinyTimeMark_t end_time = tiny_get_running_time();
        TinyTimeMark_t elapsed = end_time - start_time;
        ESP_LOGI(TAG, "Time elapsed: %lld microseconds", elapsed);
        ESP_LOGI(TAG, "Time elapsed: %.3f seconds", elapsed / 1000000.0);

        ESP_LOGI(TAG, "\n========================================");
        ESP_LOGI(TAG, "  Initial Tests Completed");
        ESP_LOGI(TAG, "========================================\n");
    }
    else
    {
        ESP_LOGE(TAG, "WiFi connection failed!");
        return;
    }

    // ============================================================
    // Timer precision test: Record 15 timestamps at 2-second intervals
    // ============================================================
    ESP_LOGI(TAG, "\n========================================");
    ESP_LOGI(TAG, "  Timer Precision Test");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Recording 15 timestamps at 2-second intervals...");
    ESP_LOGI(TAG, "No printing during recording to avoid timing overhead.\n");
    
    // Create mutex for thread-safe access to timestamp array
    s_timer_mutex = xSemaphoreCreateMutex();
    if (s_timer_mutex == NULL)
    {
        ESP_LOGE(TAG, "Failed to create mutex!");
        return;
    }
    
    // Initialize timer
    const uint64_t TIMER_PERIOD_US = 2000000;  // 2 seconds in microseconds
    esp_timer_create_args_t timer_args;
    timer_args.callback = &timer_precision_callback;
    timer_args.arg = NULL;
    timer_args.dispatch_method = ESP_TIMER_TASK;  // Execute callback in timer task
    timer_args.name = "precision_timer";
    timer_args.skip_unhandled_events = false;  // Don't skip events
    
    ret = esp_timer_create(&timer_args, &s_timer_handle);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to create timer: %s", esp_err_to_name(ret));
        vSemaphoreDelete(s_timer_mutex);
        return;
    }
    
    // Start periodic timer
    ret = esp_timer_start_periodic(s_timer_handle, TIMER_PERIOD_US);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to start timer: %s", esp_err_to_name(ret));
        esp_timer_delete(s_timer_handle);
        vSemaphoreDelete(s_timer_mutex);
        return;
    }
    
    // Wait for all timestamps to be collected
    ESP_LOGI(TAG, "Timer started. Waiting for %d timestamps...", TIMESTAMP_COUNT);
    while (!s_timer_test_complete)
    {
        vTaskDelay(100 / portTICK_PERIOD_MS);  // Check every 100ms
    }
    
    // Wait a bit more to ensure timer has stopped
    vTaskDelay(100 / portTICK_PERIOD_MS);
    
    // Print all results
    ESP_LOGI(TAG, "\n========================================");
    ESP_LOGI(TAG, "  Timer Precision Test Results");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Expected interval: 2000000 microseconds (2.000000 seconds)\n");
    
    for (int i = 0; i < TIMESTAMP_COUNT; i++)
    {
        if (i == 0)
        {
            // First timestamp - show absolute time
            ESP_LOGI(TAG, "Timestamp #%2d: %lld microseconds (%.6f seconds) [baseline]",
                     i + 1, s_timestamps[i], s_timestamps[i] / 1000000.0);
        }
        else
        {
            // Calculate interval from previous timestamp
            TinyTimeMark_t interval = s_timestamps[i] - s_timestamps[i - 1];
            int64_t error = interval - 2000000;  // Expected 2 seconds = 2000000 microseconds
            double error_ms = error / 1000.0;
            
            ESP_LOGI(TAG, "Timestamp #%2d: %lld microseconds (%.6f seconds) | "
                     "Interval: %lld us (%.6f s) | Error: %lld us (%.3f ms)",
                     i + 1, 
                     s_timestamps[i], 
                     s_timestamps[i] / 1000000.0,
                     interval,
                     interval / 1000000.0,
                     error,
                     error_ms);
        }
    }
    
    // Calculate statistics
    ESP_LOGI(TAG, "\n--- Statistics ---");
    int64_t total_interval = s_timestamps[TIMESTAMP_COUNT - 1] - s_timestamps[0];
    int64_t expected_total = 2000000 * (TIMESTAMP_COUNT - 1);
    int64_t total_error = total_interval - expected_total;
    
    ESP_LOGI(TAG, "Total time: %lld microseconds (%.6f seconds)", 
             total_interval, total_interval / 1000000.0);
    ESP_LOGI(TAG, "Expected total: %lld microseconds (%.6f seconds)", 
             expected_total, expected_total / 1000000.0);
    ESP_LOGI(TAG, "Total error: %lld microseconds (%.3f milliseconds)", 
             total_error, total_error / 1000.0);
    
    // Calculate average interval
    double avg_interval = (double)total_interval / (TIMESTAMP_COUNT - 1);
    ESP_LOGI(TAG, "Average interval: %.6f seconds (%.3f microseconds)", 
             avg_interval / 1000000.0, avg_interval);
    
    // Cleanup
    esp_timer_delete(s_timer_handle);
    vSemaphoreDelete(s_timer_mutex);
    
    ESP_LOGI(TAG, "\n========================================");
    ESP_LOGI(TAG, "  Test Complete");
    ESP_LOGI(TAG, "========================================\n");
    
    // Main loop: Keep running
    while (1)
    {
        vTaskDelay(10000 / portTICK_PERIOD_MS);
    }
}

#ifdef __cplusplus
}
#endif
