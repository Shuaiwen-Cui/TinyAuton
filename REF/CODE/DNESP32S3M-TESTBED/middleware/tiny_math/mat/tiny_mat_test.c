/**
 * @file tiny_mat_test.c
 * @brief This file implements test functions for the submodule mat of the tiny_math middleware.
 */

#include "tiny_mat_test.h"

#define ROWS 2
#define COLS 3
#define PAD 1
#define FULL_WIDTH (COLS + PAD)

void tiny_mat_test(void)
{
    float A[ROWS * FULL_WIDTH] = {
        1, 2, 3, 0,
        4, 5, 6, 0};

    float B[COLS * FULL_WIDTH] = {
        7, 8, 9, 0,
        10, 11, 12, 0,
        13, 14, 15, 0};

    float A_compact[ROWS * COLS] = {0};
    float B_compact[COLS * COLS] = {0};
    float C_compact[ROWS * COLS] = {0};

    float output_general[ROWS * FULL_WIDTH] = {0};
    float constant = 2.0f;

    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++)
            A_compact[i * COLS + j] = A[i * FULL_WIDTH + j];

    for (int i = 0; i < COLS; i++)
        for (int j = 0; j < COLS; j++)
            B_compact[i * COLS + j] = B[i * FULL_WIDTH + j];

    printf("============ [tiny_mat_test - Full Coverage] ============\n\r");
    printf("[TEST SETTINGS] ROWS=%d, COLS=%d, PAD=%d, FULL_WIDTH=%d\n\r", ROWS, COLS, PAD, FULL_WIDTH);

    print_matrix_padded("A (padded) | memory layout", A, ROWS, FULL_WIDTH, FULL_WIDTH);
    print_matrix_padded("B (padded) | memory layout", B, COLS, FULL_WIDTH, FULL_WIDTH);
    print_matrix("A_compact", A_compact, ROWS, COLS);
    print_matrix("B_compact", B_compact, COLS, COLS);
    printf("constant = %f\n\r\n\r", constant);

    TinyTimeMark_t t0, t1;
    tiny_error_t err;

    t0 = tiny_get_running_time();
    err = tiny_mat_add_f32(A, B, output_general, ROWS, COLS, PAD, PAD, PAD, 1, 1, 1);
    t1 = tiny_get_running_time();
    printf("--- tiny_mat_add_f32(rows=%d, cols=%d, paddings=1) | Time: %6.2f us | Error: %d ---\n\r", ROWS, COLS, (double)(t1 - t0), err);
    print_matrix_padded("Output (padded)", output_general, ROWS, COLS, FULL_WIDTH);

    t0 = tiny_get_running_time();
    err = tiny_mat_addc_f32(A, output_general, constant, ROWS, COLS, PAD, PAD, 1, 1);
    t1 = tiny_get_running_time();
    printf("--- tiny_mat_addc_f32(C=%f) | Time: %6.2f us | Error: %d ---\n\r", constant, (double)(t1 - t0), err);
    print_matrix_padded("Output (padded)", output_general, ROWS, COLS, FULL_WIDTH);

    t0 = tiny_get_running_time();
    err = tiny_mat_sub_f32(A, B, output_general, ROWS, COLS, PAD, PAD, PAD, 1, 1, 1);
    t1 = tiny_get_running_time();
    printf("--- tiny_mat_sub_f32(rows=%d, cols=%d) | Time: %6.2f us | Error: %d ---\n\r", ROWS, COLS, (double)(t1 - t0), err);
    print_matrix_padded("Output (padded)", output_general, ROWS, COLS, FULL_WIDTH);

    t0 = tiny_get_running_time();
    err = tiny_mat_subc_f32(A, output_general, constant, ROWS, COLS, PAD, PAD, 1, 1);
    t1 = tiny_get_running_time();
    printf("--- tiny_mat_subc_f32(C=%f) | Time: %6.2f us | Error: %d ---\n\r", constant, (double)(t1 - t0), err);
    print_matrix_padded("Output (padded)", output_general, ROWS, COLS, FULL_WIDTH);

    t0 = tiny_get_running_time();
    err = tiny_mat_multc_f32(A, output_general, constant, ROWS, COLS, PAD, PAD, 1, 1);
    t1 = tiny_get_running_time();
    printf("--- tiny_mat_multc_f32(C=%f) | Time: %6.2f us | Error: %d ---\n\r", constant, (double)(t1 - t0), err);
    print_matrix_padded("Output (padded)", output_general, ROWS, COLS, FULL_WIDTH);

    t0 = tiny_get_running_time();
    err = tiny_mat_mult_f32(A_compact, B_compact, C_compact, ROWS, COLS, COLS);
    t1 = tiny_get_running_time();
    printf("--- tiny_mat_mult_f32 (compact) | Time: %6.2f us | Error: %d ---\n\r", (double)(t1 - t0), err);
    print_matrix("Output (compact)", C_compact, ROWS, COLS);

    t0 = tiny_get_running_time();
    err = tiny_mat_mult_ex_f32(A, B, output_general, ROWS, COLS, COLS, PAD, PAD, PAD);
    t1 = tiny_get_running_time();
    printf("--- tiny_mat_mult_ex_f32(padded) | Time: %6.2f us | Error: %d ---\n\r", (double)(t1 - t0), err);
    print_matrix_padded("Output (padded)", output_general, ROWS, COLS, FULL_WIDTH);

    printf("============ [test complete] ============\n\r");
}
