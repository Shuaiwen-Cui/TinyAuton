#include "tiny_math.h"

#ifdef ESP_PLATFORM
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_task_wdt.h"
#endif

extern "C" void app_main(void)
{

    // Run TinyMath unit tests (vector, matrix, C++ matrix)
    // tiny_vec_test();
    // tiny_mat_test();
    tiny_matrix_test();

    printf("\n\n");

}
