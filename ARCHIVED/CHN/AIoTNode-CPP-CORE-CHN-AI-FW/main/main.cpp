#include "tiny_dsp.h"

#ifdef ESP_PLATFORM
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_task_wdt.h"
#endif

extern "C" void app_main(void)
{
#ifdef ESP_PLATFORM
    // Add current task to watchdog (watchdog is already initialized by ESP-IDF)
    esp_task_wdt_add(NULL);
#endif

    tiny::tiny_ica_test_all();

#ifdef ESP_PLATFORM
    while (1)
    {
        vTaskDelay(pdMS_TO_TICKS(1000));
        esp_task_wdt_reset();
    }
#endif
}
