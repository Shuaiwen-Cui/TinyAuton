#include "tiny_ai_sine_regression_test.h"

#ifdef ESP_PLATFORM
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_task_wdt.h"
#endif

extern "C" void app_main(void)
{
    // Run TinyAI sine regression tests
    tiny_ai_sine_regression_test_all();
    
    printf("\n\n");
    
#ifdef ESP_PLATFORM
    // Keep main task alive to prevent watchdog timeout
    // Feed watchdog periodically
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));  // Delay 1 second
        esp_task_wdt_reset();  // Feed watchdog
    }
#endif
}
