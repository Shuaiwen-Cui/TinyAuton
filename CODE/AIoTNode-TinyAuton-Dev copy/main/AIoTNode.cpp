/**
 * @file AIoTNode.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief AIoTNode application entry point — tiny_ai example runner
 */

/* ============================================================
 * tiny_ai example selector
 * ============================================================ */

#define TEST_TINY_AI_MLP         1
#define TEST_TINY_AI_CNN         1
#define TEST_TINY_AI_ATTENTION   1

/* ============================================================
 * Includes
 * ============================================================ */

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tiny_ai.h"

/* ============================================================
 * Application entry
 * ============================================================ */

extern "C" void app_main(void)
{
    vTaskDelay(pdMS_TO_TICKS(500));

    printf("\n");
    printf("##############################################\n");
    printf("#          tiny_ai  example runner           #\n");
    printf("##############################################\n");

#if TEST_TINY_AI_MLP
    printf("\n>>> [tiny_ai] example_mlp  (Iris, MLP)\n");
    example_mlp();
#endif

#if TEST_TINY_AI_CNN
    printf("\n>>> [tiny_ai] example_cnn  (Signal, CNN1D)\n");
    example_cnn();
#endif

#if TEST_TINY_AI_ATTENTION
    printf("\n>>> [tiny_ai] example_attention  (Iris, Attention)\n");
    example_attention();
#endif

    printf("\n##############################################\n");
    printf("#         tiny_ai  example runner finished   #\n");
    printf("##############################################\n");

    while (true)
    {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
