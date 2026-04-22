#include "cnn_gesture_demo.h"

#ifdef ESP_PLATFORM
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_task_wdt.h"
#endif

extern "C" void app_main(void)
{
    printf("\n");
    printf("========================================\n");
    printf("AIoTNode ESP32S3 AI Test Program\n");
    printf("========================================\n");
    printf("\n");

    // 1. Run Gesture CNN classification example
    cnn_gesture_demo_run();
    
    // 2. Run CNN performance benchmark test
    cnn_gesture_demo_benchmark(1000);  // 1000 iterations
    
    printf("\n");
    printf("========================================\n");
    printf("All tests completed\n");
    printf("========================================\n");
    printf("\n");
    
#ifdef ESP_PLATFORM
    // Keep main task alive to prevent watchdog timeout
    // Try to add current task to watchdog (may fail if not initialized)
    esp_err_t ret = esp_task_wdt_add(NULL);
    if (ret == ESP_ERR_INVALID_STATE) {
        // Watchdog not initialized, initialize it first
        esp_task_wdt_config_t wdt_config = {
            .timeout_ms = 10 * 1000,  // 10 seconds timeout
            .idle_core_mask = 0,
            .trigger_panic = false
        };
        esp_task_wdt_init(&wdt_config);
        esp_task_wdt_add(NULL);  // Add current task to watchdog
    }
    // If ret == ESP_OK, watchdog was already initialized and task was added
    
    // Feed watchdog periodically
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));  // Delay 1 second
        esp_task_wdt_reset();  // Feed watchdog
    }
#endif
}
