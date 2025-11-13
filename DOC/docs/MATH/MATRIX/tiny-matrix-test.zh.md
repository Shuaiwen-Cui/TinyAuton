# 测试

!!! tip
    以下的测试用代码和案例也作为使用教学案例。

## tiny_matrix_test.hpp

```cpp
/**
 * @file tiny_matrix_test.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the header file for the test of the submodule matrix (advanced matrix operations) of the tiny_math middleware.
 * @version 1.0
 * @date 2025-04-17
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
#include "tiny_matrix.hpp" // TinyMatrix Header

/* STATEMENTS */
void tiny_matrix_test();  // C-compatible test entry

```

## tiny_matrix_test.cpp

```cpp
/**
 * @file tiny_matrix_test.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the source file for the test of the submodule matrix (advanced matrix operations) of the tiny_math middleware.
 * @version 1.0
 * @date 2025-04-17
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_matrix_test.hpp" // TinyMatrix Test Header
#include "tiny_time.h"           // For performance testing

#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>  // For std::istringstream (used in stream operator tests)

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
#include "esp_task_wdt.h"       // For FreeRTOS task watchdog
#endif

/* PERFORMANCE TESTING MACROS */
// Reduced matrix sizes and iterations to prevent watchdog timeout and memory issues
#define PERFORMANCE_TEST_ITERATIONS 100        // Reduced from 1000 to prevent timeout
#define PERFORMANCE_TEST_ITERATIONS_HEAVY 10   // Reduced from 100 for compute-intensive operations
#define PERFORMANCE_TEST_WARMUP 3              // Reduced from 10

// Macro for timing a single operation
#define TIME_OPERATION(operation, description) \
    do { \
        TinyTimeMark_t t0 = tiny_get_running_time(); \
        operation; \
        TinyTimeMark_t t1 = tiny_get_running_time(); \
        double dt_us = (double)(t1 - t0); \
        std::cout << "[Performance] " << description << ": " << std::fixed << std::setprecision(2) << dt_us << " us\n"; \
    } while(0)

// Helper function to feed watchdog (inline to avoid function call overhead in tight loops)
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
// Static flag to track if task has been added to watchdog
static bool task_wdt_added = false;

inline void ensure_task_wdt_added()
{
    if (!task_wdt_added)
    {
        esp_task_wdt_add(NULL);  // Add current task to watchdog (NULL = current task)
        task_wdt_added = true;
    }
}

inline void feed_watchdog()
{
    ensure_task_wdt_added();
    esp_task_wdt_reset();
}

inline void feed_watchdog_if_needed(int iteration, int interval)
{
    if ((iteration + 1) % interval == 0)
    {
        feed_watchdog();
    }
}
#else
inline void feed_watchdog()
{
    // No-op for non-ESP32 platforms
}

inline void feed_watchdog_if_needed(int iteration, int interval)
{
    // No-op for non-ESP32 platforms
    (void)iteration;
    (void)interval;
}
#endif

// Macro for timing repeated operations
#define TIME_REPEATED_OPERATION(operation, iterations, description) \
    do { \
        /* Feed watchdog before starting */ \
        feed_watchdog(); \
        /* Warmup */ \
        for (int w = 0; w < PERFORMANCE_TEST_WARMUP; ++w) { \
            feed_watchdog();  /* Feed watchdog before each operation */ \
            operation; \
            feed_watchdog();  /* Feed watchdog after each operation */ \
        } \
        /* Actual test */ \
        TinyTimeMark_t perf_t0 = tiny_get_running_time(); \
        for (int i = 0; i < iterations; ++i) { \
            /* Feed watchdog every 10 iterations (increased interval to reduce overhead) */ \
            if (i % 10 == 0) feed_watchdog(); \
            operation; \
        } \
        feed_watchdog();  /* Final feed after loop */ \
        TinyTimeMark_t perf_t1 = tiny_get_running_time(); \
        double perf_dt_total_us = (double)(perf_t1 - perf_t0); \
        double perf_dt_avg_us = perf_dt_total_us / iterations; \
        std::cout << "[Performance] " << description << " (" << iterations << " iterations): " \
                  << std::fixed << std::setprecision(2) << perf_dt_total_us << " us total, " \
                  << perf_dt_avg_us << " us avg\n"; \
    } while(0)

// Helper function to check if two matrices are approximately equal
bool matrices_approximately_equal(const tiny::Mat &m1, const tiny::Mat &m2, float epsilon = 1e-5f)
{
    if (m1.row != m2.row || m1.col != m2.col)
        return false;
    
    for (int i = 0; i < m1.row; ++i)
    {
        for (int j = 0; j < m1.col; ++j)
        {
            if (std::fabs(m1(i, j) - m2(i, j)) > epsilon)
                return false;
        }
    }
    return true;
}

// Group 1: constructor & destructor
void test_constructor_destructor()
{
    std::cout << "\n--- Test: Constructor & Destructor ---\n";

    // Test 1.1: default constructor
    std::cout << "[Test 1.1] Default Constructor\n";
    tiny::Mat mat1;
    mat1.print_info();
    mat1.print_matrix(true);

    // Test 1.2: constructor with rows and cols, using internal allocation
    std::cout << "[Test 1.2] Constructor with Rows and Cols\n";
    tiny::Mat mat2(3, 4);
    mat2.print_info();
    mat2.print_matrix(true);

    // Test 1.3: constructor with rows and cols, specifying stride, using internal allocation
    std::cout << "[Test 1.3] Constructor with Rows, Cols and Stride\n";
    tiny::Mat mat3(3, 4, 5);
    mat3.print_info();
    mat3.print_matrix(true);

    // Test 1.4: constructor with external data
    std::cout << "[Test 1.4] Constructor with External Data\n";
    float data[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    tiny::Mat mat4(data, 3, 4);
    mat4.print_info();
    mat4.print_matrix(true);

    // Test 1.5: constructor with external data and stride
    std::cout << "[Test 1.5] Constructor with External Data and Stride\n";
    float data_stride[15] = {0, 1, 2, 3, 0, 4, 5, 6, 7, 0, 8, 9, 10, 11, 0};
    tiny::Mat mat5(data_stride, 3, 4, 5);
    mat5.print_info();
    mat5.print_matrix(true);

    // Test 1.6: copy constructor
    std::cout << "[Test 1.6] Copy Constructor\n";
    tiny::Mat mat6(mat5);
    mat6.print_info();
    mat6.print_matrix(true);
}

// Group 2: element access
void test_element_access()
{
    std::cout << "\n--- Test: Element Access ---\n";
    tiny::Mat mat(2, 3);

    // Test 2.1: non-const access
    std::cout << "[Test 2.1] Non-const Access\n";
    mat(0, 0) = 1.1f;
    mat(0, 1) = 2.2f;
    mat(0, 2) = 3.3f;
    mat(1, 0) = 4.4f;
    mat(1, 1) = 5.5f;
    mat(1, 2) = 6.6f;
    mat.print_info();
    mat.print_matrix(true);

    // Test 2.2: const access
    std::cout << "[Test 2.2] Const Access\n";
    const tiny::Mat const_mat = mat;
    std::cout << "const_mat(0, 0): " << const_mat(0, 0) << "\n";
}

// Group 3: data manipulation
void test_roi_operations()
{
    std::cout << "\n--- Test: Data Manipulation ---\n";

    // Material Matrices
    tiny::Mat matA(2, 3);
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            matA(i, j) = i * 3 + j + 1;
            matA(i, j) = matA(i, j) / 10;
        }
    }

    float data[15] = {0, 1, 2, 3, 0, 4, 5, 6, 7, 0, 8, 9, 10, 11, 0};
    tiny::Mat matB(data, 3, 4, 5);

    tiny::Mat matC;

    std::cout << "[Material Matrices]\n";
    std::cout << "matA:\n";
    matA.print_info();
    matA.print_matrix(true);
    std::cout << "matB:\n";
    matB.print_info();
    matB.print_matrix(true);
    std::cout << "matC:\n";
    matC.print_info();
    matC.print_matrix(true);

    // Test 3.1: Copy ROI
    std::cout << "[Test 3.1] Copy ROI - Over Range Case\n";
    matB.copy_paste(matA, 1, 2);
    std::cout << "matB after copy_paste matA at (1, 2):\n";
    matB.print_matrix(true);
    std::cout << "nothing changed.\n";

    std::cout << "[Test 3.1] Copy ROI - Suitable Range Case\n";
    matB.copy_paste(matA, 1, 1);
    std::cout << "matB after copy_paste matA at (1, 1):\n";
    matB.print_info();
    matB.print_matrix(true);
    std::cout << "successfully copied.\n";

    // Test 3.2: Copy Head
    std::cout << "[Test 3.2] Copy Head\n";
    matC.copy_head(matB);
    std::cout << "matC after copy_head matB:\n";
    matC.print_info();
    matC.print_matrix(true);

    std::cout << "[Test 3.2] Copy Head - Memory Sharing Check\n"; // matB and matC share the same data pointer
    matB(0, 0) = 99.99f;
    std::cout << "matB(0, 0) = 99.99f\n";
    std::cout << "matC:\n";
    matC.print_info();
    matC.print_matrix(true);

    // Test 3.3: Get a View of ROI - low level function
    std::cout << "[Test 3.3] Get a View of ROI - Low Level Function\n";
    std::cout << "get a view of ROI with overrange dimensions - rows:\n";
    tiny::Mat roi1 = matB.view_roi(1, 1, 3, 2); // note here, C++ will use the copy constructor, which will copy according to the case (submatrix - shallow copy | normal - deep copy)
    std::cout << "get a view of ROI with overrange dimensions - cols:\n";
    tiny::Mat roi2 = matB.view_roi(1, 1, 2, 4); // note here, C++ will use the copy constructor, which will copy according to the case (submatrix - shallow copy | normal - deep copy)
    std::cout << "get a view of ROI with suitable dimensions:\n";
    tiny::Mat roi3 = matB.view_roi(1, 1, 2, 2); // note here, C++ will use the copy constructor, which will copy according to the case (submatrix - shallow copy | normal - deep copy)
    std::cout << "roi3:\n";
    roi3.print_info();
    roi3.print_matrix(true);

    // Test 3.4: Get a View of ROI - using ROI structure
    std::cout << "[Test 3.4] Get a View of ROI - Using ROI Structure\n";
    tiny::Mat::ROI roi_struct(1, 1, 2, 2);
    tiny::Mat roi4 = matB.view_roi(roi_struct);
    roi4.print_info();
    roi4.print_matrix(true);

    // Test 3.5: Copy ROI - low level function
    std::cout << "[Test 3.5] Copy ROI - Low Level Function\n";
    tiny::Mat mat_deep_copy = matB.copy_roi(1, 1, 2, 2);
    mat_deep_copy.print_info();
    mat_deep_copy.print_matrix(true);

    // Test 3.6: Copy ROI - using ROI structure
    std::cout << "[Test 3.6] Copy ROI - Using ROI Structure\n";
    TinyTimeMark_t tic1 = tiny_get_running_time();
    tiny::Mat::ROI roi_struct2(1, 1, 2, 2);
    tiny::Mat mat_deep_copy2 = matB.copy_roi(roi_struct2);
    TinyTimeMark_t toc1 = tiny_get_running_time();
    TinyTimeMark_t copy_roi_time = toc1 - tic1;
    std::cout << "time for copy_roi using ROI structure: " << copy_roi_time << " ms\n";
    mat_deep_copy2.print_info();
    mat_deep_copy2.print_matrix(true);

    // Test 3.7: Block
    std::cout << "[Test 3.7] Block\n";
    TinyTimeMark_t tic2 = tiny_get_running_time();
    tiny::Mat mat_block = matB.block(1, 1, 2, 2);
    TinyTimeMark_t toc2 = tiny_get_running_time();
    TinyTimeMark_t block_roi_time = toc2 - tic2;
    std::cout << "time for block: " << block_roi_time << " ms\n";
    mat_block.print_info();
    mat_block.print_matrix(true);

    // Test 3.8: Swap Rows
    std::cout << "[Test 3.8] Swap Rows\n";
    std::cout << "matB before swap rows:\n";
    matB.print_info();
    matB.print_matrix(true);
    std::cout << "matB after swap_rows(0, 2):\n";
    matB.swap_rows(0, 2);
    matB.print_info();
    matB.print_matrix(true);

    // Test 3.9: Swap Columns
    std::cout << "[Test 3.9] Swap Columns\n";
    std::cout << "matB before swap columns:\n";
    matB.print_info();
    matB.print_matrix(true);
    std::cout << "matB after swap_cols(0, 2):\n";
    matB.swap_cols(0, 2);
    matB.print_info();
    matB.print_matrix(true);

    // Test 3.10: Clear
    std::cout << "[Test 3.10] Clear\n";
    std::cout << "matB before clear:\n";
    matB.print_info();
    matB.print_matrix(true);
    std::cout << "matB after clear:\n";
    matB.clear();
    matB.print_info();
    matB.print_matrix(true);
}

// Group 4.1: Assignment Operator
void test_assignment_operator()
{
    std::cout << "\n[Group 4.1: Assignment Operator Tests]\n";

    std::cout << "\n[Test 4.1.1] Assignment (Same Dimensions)\n";
    tiny::Mat dst(2, 3), src(2, 3);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            src(i, j) = static_cast<float>(i * 3 + j + 1);
    dst = src;
    dst.print_matrix(true);

    std::cout << "\n[Test 4.1.2] Assignment (Different Dimensions)\n";
    tiny::Mat dst2(4, 2);
    dst2 = src;
    dst2.print_matrix(true);

    std::cout << "\n[Test 4.1.3] Assignment to Sub-Matrix (Expect Error)\n";
    float data[15] = {0, 1, 2, 3, 0, 4, 5, 6, 7, 0, 8, 9, 10, 11, 0};
    tiny::Mat base(data, 3, 4, 5);
    tiny::Mat subView = base.view_roi(1, 1, 2, 2);
    subView = src;
    subView.print_matrix(true);

    std::cout << "\n[Test 4.1.4] Self-Assignment\n";
    src = src;
    src.print_matrix(true);
}

// Group 4.2: Matrix Addition
void test_matrix_addition()
{
    std::cout << "\n[Group 4.2: Matrix Addition Tests]\n";

    std::cout << "\n[Test 4.2.1] Matrix Addition (Same Dimensions)\n";
    tiny::Mat A(2, 3), B(2, 3);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
        {
            A(i, j) = static_cast<float>(i * 3 + j + 1);
            B(i, j) = 1.0f;
        }
    A += B;
    A.print_matrix(true);

    std::cout << "\n[Test 4.2.2] Sub-Matrix Addition\n";
    float data[20] = {0,1,2,3,0,4,5,6,7,0,8,9,10,11,0,12,13,14,15,0};
    tiny::Mat base(data, 4, 4, 5);
    tiny::Mat subA = base.view_roi(1,1,2,2);
    tiny::Mat subB = base.view_roi(1,1,2,2);
    subA += subB;
    subA.print_matrix(true);

    std::cout << "\n[Test 4.2.3] Full Matrix + Sub-Matrix Addition\n";
    tiny::Mat full(2,2);
    for(int i=0;i<2;++i) for(int j=0;j<2;++j) full(i,j)=2.0f;
    full += subB;
    full.print_matrix(true);

    std::cout << "\n[Test 4.2.4] Addition Dimension Mismatch (Expect Error)\n";
    tiny::Mat wrongDim(3,3);
    full += wrongDim;
}

// Group 4.3: Constant Addition
void test_constant_addition()
{
    std::cout << "\n[Group 4.3: Constant Addition Tests]\n";

    std::cout << "\n[Test 4.3.1] Full Matrix + Constant\n";
    tiny::Mat mat1(2,3);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            mat1(i,j) = static_cast<float>(i*3 + j);
    mat1 += 5.0f;
    mat1.print_matrix(true);

    std::cout << "\n[Test 4.3.2] Sub-Matrix + Constant\n";
    float data[20] = {0,1,2,3,0,4,5,6,7,0,8,9,10,11,0,12,13,14,15,0};
    tiny::Mat base(data,4,4,5);
    tiny::Mat sub = base.view_roi(1,1,2,2);
    sub += 3.0f;
    sub.print_matrix(true);

    std::cout << "\n[Test 4.3.3] Add Zero\n";
    tiny::Mat mat2(2,2);
    mat2(0,0)=1; mat2(0,1)=2; mat2(1,0)=3; mat2(1,1)=4;
    mat2 += 0.0f;
    mat2.print_matrix(true);

    std::cout << "\n[Test 4.3.4] Add Negative Constant\n";
    tiny::Mat mat3(2,2);
    mat3(0,0)=10; mat3(0,1)=20; mat3(1,0)=30; mat3(1,1)=40;
    mat3 += -15.0f;
    mat3.print_matrix(true);
}

// Group 4.4: Matrix Subtraction
void test_matrix_subtraction()
{
    std::cout << "\n[Group 4.4: Matrix Subtraction Tests]\n";

    std::cout << "\n[Test 4.4.1] Matrix Subtraction\n";
    tiny::Mat A(2,2), B(2,2);
    A(0,0)=5; A(0,1)=7; A(1,0)=9; A(1,1)=11;
    B(0,0)=1; B(0,1)=2; B(1,0)=3; B(1,1)=4;
    A -= B;
    A.print_matrix(true);

    std::cout << "\n[Test 4.4.2] Subtraction Dimension Mismatch (Expect Error)\n";
    tiny::Mat wrong(3,3);
    A -= wrong;
}

// Group 4.5: Constant Subtraction
void test_constant_subtraction()
{
    std::cout << "\n[Group 4.5: Constant Subtraction Tests]\n";

    std::cout << "\n[Test 4.5.1] Full Matrix - Constant\n";
    tiny::Mat mat(2,3);
    for (int i=0;i<2;++i) for(int j=0;j<3;++j) mat(i,j) = i*3+j+1;
    mat -= 2.0f;
    mat.print_matrix(true);

    std::cout << "\n[Test 4.5.2] Sub-Matrix - Constant\n";
    float data[15] = {0,1,2,3,0,4,5,6,7,0,8,9,10,11,0};
    tiny::Mat base(data,3,4,5);
    tiny::Mat sub = base.view_roi(1,1,2,2);
    sub -= 1.5f;
    sub.print_matrix(true);
}

// Group 4.6: Matrix Element-wise Division
void test_matrix_division()
{
    std::cout << "\n[Group 4.6: Matrix Element-wise Division Tests]\n";

    std::cout << "\n[Test 4.6.1] Element-wise Division (Same Dimensions, No Zero)\n";
    tiny::Mat A(2, 2), B(2, 2);
    A(0,0) = 10; A(0,1) = 20; A(1,0) = 30; A(1,1) = 40;
    B(0,0) = 2;  B(0,1) = 4;  B(1,0) = 5;  B(1,1) = 8;
    A /= B;
    A.print_matrix(true);

    std::cout << "\n[Test 4.6.2] Dimension Mismatch (Expect Error)\n";
    tiny::Mat wrongDim(3, 3);
    A /= wrongDim;

    std::cout << "\n[Test 4.6.3] Division by Matrix Containing Zero (Expect Error)\n";
    tiny::Mat C(2, 2), D(2, 2);
    C(0,0)=5; C(0,1)=10; C(1,0)=15; C(1,1)=20;
    D(0,0)=1; D(0,1)=0;  D(1,0)=3;  D(1,1)=4;  // Contains zero
    C /= D;
    C.print_matrix(true);  // Should remain unchanged
}

// Group 4.7: Constant Division
void test_constant_division()
{
    std::cout << "\n[Group 4.7: Matrix Division by Constant Tests]\n";

    std::cout << "\n[Test 4.7.1] Divide Full Matrix by Positive Constant\n";
    tiny::Mat mat1(2, 3);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            mat1(i, j) = static_cast<float>(i * 3 + j + 2);  // Avoid zero
    mat1 /= 2.0f;
    mat1.print_matrix(true);

    std::cout << "\n[Test 4.7.2] Divide Matrix by Negative Constant\n";
    tiny::Mat mat2(2, 2);
    mat2(0,0)=6; mat2(0,1)=12; mat2(1,0)=18; mat2(1,1)=24;
    mat2 /= -3.0f;
    mat2.print_matrix(true);

    std::cout << "\n[Test 4.7.3] Division by Zero Constant (Expect Error)\n";
    tiny::Mat mat3(2, 2);
    mat3(0,0)=1; mat3(0,1)=2; mat3(1,0)=3; mat3(1,1)=4;
    mat3 /= 0.0f;
    mat3.print_matrix(true);  // Should remain unchanged
}

// Group 4.8: Matrix Exponentiation
void test_matrix_exponentiation()
{
    std::cout << "\n[Group 4.8: Matrix Exponentiation Tests]\n";

    std::cout << "\n[Test 4.8.1] Raise Each Element to Power of 2\n";
    tiny::Mat mat1(2, 2);
    mat1(0,0)=2; mat1(0,1)=3; mat1(1,0)=4; mat1(1,1)=5;
    tiny::Mat result1 = mat1 ^ 2;
    result1.print_matrix(true);

    std::cout << "\n[Test 4.8.2] Raise Each Element to Power of 0\n";
    tiny::Mat mat2(2, 2);
    mat2(0,0)=7; mat2(0,1)=-3; mat2(1,0)=0.5f; mat2(1,1)=10;
    tiny::Mat result2 = mat2 ^ 0;
    result2.print_matrix(true);  // Expect all 1

    std::cout << "\n[Test 4.8.3] Raise Each Element to Power of 1\n";
    tiny::Mat mat3(2, 2);
    mat3(0,0)=9; mat3(0,1)=8; mat3(1,0)=7; mat3(1,1)=6;
    tiny::Mat result3 = mat3 ^ 1;
    result3.print_matrix(true);  // Expect same as original

    std::cout << "\n[Test 4.8.4] Raise Each Element to Power of -1 (Expect Error or Warning)\n";
    tiny::Mat mat4(2, 2);
    mat4(0,0)=1; mat4(0,1)=2; mat4(1,0)=4; mat4(1,1)=5;
    tiny::Mat result4 = mat4 ^ -1;
    result4.print_matrix(true);

    std::cout << "\n[Test 4.8.5] Raise Matrix Containing Zero to Power of 3\n";
    tiny::Mat mat5(2, 2);
    mat5(0,0)=0; mat5(0,1)=2; mat5(1,0)=-1; mat5(1,1)=3;
    tiny::Mat result5 = mat5 ^ 3;
    result5.print_matrix(true);
}

// Group 5: Linear Algebra
// Group 5.1: Matrix Transpose
void test_matrix_transpose()
{
    std::cout << "\n[Group 5.1: Matrix Transpose Tests]\n";

    // Test 5.1.1: Basic 2x3 matrix transpose
    std::cout << "\n[Test 5.1.1] Transpose of 2x3 Matrix\n";
    tiny::Mat mat1(2, 3);
    int val = 1;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            mat1(i, j) = val++;

    std::cout << "Original 2x3 Matrix:\n";
    mat1.print_matrix(true);

    tiny::Mat transposed1 = mat1.transpose();
    std::cout << "Transposed 3x2 Matrix:\n";
    transposed1.print_matrix(true);

    // Test 5.1.2: Square matrix transpose (3x3)
    std::cout << "\n[Test 5.1.2] Transpose of 3x3 Square Matrix\n";
    tiny::Mat mat2(3, 3);
    val = 1;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            mat2(i, j) = val++;

    std::cout << "Original 3x3 Matrix:\n";
    mat2.print_matrix(true);

    tiny::Mat transposed2 = mat2.transpose();
    std::cout << "Transposed 3x3 Matrix:\n";
    transposed2.print_matrix(true);

    // Test 5.1.3: Matrix with padding (4x2, stride=3)
    std::cout << "\n[Test 5.1.3] Transpose of Matrix with Padding\n";
    float data[12] = {1, 2, 0, 3, 4, 0, 5, 6, 0, 7, 8, 0};  // stride=3, 4 rows
    tiny::Mat mat3(data, 4, 2, 3);
    std::cout << "Original 4x2 Matrix (with padding):\n";
    mat3.print_matrix(true);

    tiny::Mat transposed3 = mat3.transpose();
    std::cout << "Transposed 2x4 Matrix:\n";
    transposed3.print_matrix(true);

    // Test 5.1.4: Transpose of empty matrix
    std::cout << "\n[Test 5.1.4] Transpose of Empty Matrix\n";
    tiny::Mat mat4;
    mat4.print_matrix(true);

    tiny::Mat transposed4 = mat4.transpose();
    transposed4.print_matrix(true);
}

// Group 5.2: Matrix Minor and Cofactor
void test_matrix_cofactor()
{
    std::cout << "\n[Group 5.2: Matrix Minor and Cofactor Tests]\n";

    // Test 5.2.1: Minor of 3x3 Matrix - Standard Case
    std::cout << "\n[Test 5.2.1] Minor of 3x3 Matrix (Remove Row 1, Col 1)\n";
    tiny::Mat mat1(3, 3);
    int val = 1;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            mat1(i, j) = val++;

    std::cout << "Original 3x3 Matrix:\n";
    mat1.print_matrix(true);

    tiny::Mat minor1 = mat1.minor(1, 1);
    std::cout << "Minor Matrix (remove row 1, col 1, no sign):\n";
    minor1.print_matrix(true);  // Expected: [[1,3],[7,9]]

    // Test 5.2.2: Cofactor of 3x3 Matrix - Same position
    std::cout << "\n[Test 5.2.2] Cofactor of 3x3 Matrix (Remove Row 1, Col 1)\n";
    std::cout << "Note: Cofactor matrix is the same as minor matrix.\n";
    std::cout << "      The sign (-1)^(i+j) is applied when computing cofactor value, not to matrix elements.\n";
    tiny::Mat cof1 = mat1.cofactor(1, 1);
    std::cout << "Cofactor Matrix (same as minor):\n";
    cof1.print_matrix(true);  // Expected: [[1,3],[7,9]] (same as minor)

    // Test 5.2.3: Minor - Remove first row and first column
    std::cout << "\n[Test 5.2.3] Minor (Remove Row 0, Col 0)\n";
    tiny::Mat minor2 = mat1.minor(0, 0);
    minor2.print_matrix(true);  // Expected: [[5,6],[8,9]]

    // Test 5.2.4: Cofactor - Remove first row and first column
    std::cout << "\n[Test 5.2.4] Cofactor (Remove Row 0, Col 0)\n";
    std::cout << "Note: Cofactor matrix is the same as minor matrix.\n";
    tiny::Mat cof2 = mat1.cofactor(0, 0);
    cof2.print_matrix(true);  // Expected: [[5,6],[8,9]] (same as minor)

    // Test 5.2.5: Cofactor - Remove row 0, col 1
    std::cout << "\n[Test 5.2.5] Cofactor (Remove Row 0, Col 1)\n";
    std::cout << "Note: Cofactor matrix is the same as minor matrix.\n";
    std::cout << "      When computing cofactor value, sign (-1)^(0+1) = -1 would be applied.\n";
    tiny::Mat cof2_neg = mat1.cofactor(0, 1);
    std::cout << "Cofactor Matrix (same as minor):\n";
    cof2_neg.print_matrix(true);  // Expected: [[4,6],[7,9]] (same as minor, no sign in matrix)

    // Test 5.2.6: Minor - Remove last row and last column
    std::cout << "\n[Test 5.2.6] Minor (Remove Row 2, Col 2)\n";
    tiny::Mat minor3 = mat1.minor(2, 2);
    minor3.print_matrix(true);  // Expected: [[1,2],[4,5]]

    // Test 5.2.7: Cofactor - Remove last row and last column
    std::cout << "\n[Test 5.2.7] Cofactor (Remove Row 2, Col 2)\n";
    std::cout << "Note: Cofactor matrix is the same as minor matrix.\n";
    tiny::Mat cof3 = mat1.cofactor(2, 2);
    cof3.print_matrix(true);  // Expected: [[1,2],[4,5]] (same as minor)

    // Test 5.2.8: 4x4 Matrix Example - Minor
    std::cout << "\n[Test 5.2.8] Minor of 4x4 Matrix (Remove Row 2, Col 1)\n";
    tiny::Mat mat4(4, 4);
    val = 1;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            mat4(i, j) = val++;

    mat4.print_matrix(true);
    tiny::Mat minor4 = mat4.minor(2, 1);
    std::cout << "Minor Matrix:\n";
    minor4.print_matrix(true);

    // Test 5.2.9: 4x4 Matrix Example - Cofactor
    std::cout << "\n[Test 5.2.9] Cofactor of 4x4 Matrix (Remove Row 2, Col 1)\n";
    std::cout << "Note: Cofactor matrix is the same as minor matrix.\n";
    std::cout << "      When computing cofactor value, sign (-1)^(2+1) = -1 would be applied.\n";
    tiny::Mat cof4 = mat4.cofactor(2, 1);
    std::cout << "Cofactor Matrix (same as minor):\n";
    cof4.print_matrix(true);

    // Test 5.2.10: Non-square Matrix (Expect Error)
    std::cout << "\n[Test 5.2.10] Non-square Matrix (Expect Error)\n";
    tiny::Mat rectMat(3, 4);
    std::cout << "Testing minor():\n";
    rectMat.minor(1, 1).print_matrix(true);  // Should trigger error and return empty matrix
    std::cout << "Testing cofactor():\n";
    rectMat.cofactor(1, 1).print_matrix(true);  // Should trigger error and return empty matrix
}

// Group 5.3: Matrix Determinant
void test_matrix_determinant()
{
    std::cout << "\n[Group 5.3: Matrix Determinant Tests]\n";

    // Test 5.3.1: 1x1 Matrix
    std::cout << "\n[Test 5.3.1] 1x1 Matrix Determinant\n";
    tiny::Mat mat1(1, 1);
    mat1(0, 0) = 7;
    std::cout << "Matrix:\n";
    mat1.print_matrix(true);
    std::cout << "Determinant: " << mat1.determinant() << "  (Expected: 7)\n";

    // Test 5.3.2: 2x2 Matrix
    std::cout << "\n[Test 5.3.2] 2x2 Matrix Determinant\n";
    tiny::Mat mat2(2, 2);
    mat2(0, 0) = 3; mat2(0, 1) = 8;
    mat2(1, 0) = 4; mat2(1, 1) = 6;
    std::cout << "Matrix:\n";
    mat2.print_matrix(true);
    std::cout << "Determinant: " << mat2.determinant() << "  (Expected: -14)\n";

    // Test 5.3.3: 3x3 Matrix
    std::cout << "\n[Test 5.3.3] 3x3 Matrix Determinant\n";
    tiny::Mat mat3(3, 3);
    mat3(0,0) = 1; mat3(0,1) = 2; mat3(0,2) = 3;
    mat3(1,0) = 0; mat3(1,1) = 4; mat3(1,2) = 5;
    mat3(2,0) = 1; mat3(2,1) = 0; mat3(2,2) = 6;
    std::cout << "Matrix:\n";
    mat3.print_matrix(true);
    std::cout << "Determinant: " << mat3.determinant() << "  (Expected: 22)\n";

    // Test 5.3.4: 4x4 Matrix
    std::cout << "\n[Test 5.3.4] 4x4 Matrix Determinant\n";
    tiny::Mat mat4(4, 4);
    int val = 1;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            mat4(i, j) = val++;
    std::cout << "Matrix:\n";
    mat4.print_matrix(true);
    std::cout << "Note: This matrix has linearly dependent rows (each row differs by constant 4),\n";
    std::cout << "      so the determinant should be 0.\n";
    std::cout << "Determinant: " << mat4.determinant() << "  (Expected: 0)\n";  

    // Test 5.3.5: Non-square Matrix (Expect Error)
    std::cout << "\n[Test 5.3.5] Non-square Matrix (Expect Error)\n";
    tiny::Mat rectMat(3, 4);
    std::cout << "Matrix (3x4, non-square):\n";
    rectMat.print_matrix(true);
    float det_rect = rectMat.determinant();  // should trigger error
    std::cout << "Determinant: " << det_rect << "  (Expected: 0 with error message)\n";

}

// Group 5.4: Matrix Adjoint
void test_matrix_adjoint()
{
    std::cout << "\n[Group 5.4: Matrix Adjoint Tests]\n";

    // Test 5.4.1: 1x1 Matrix
    std::cout << "\n[Test 5.4.1] Adjoint of 1x1 Matrix\n";
    tiny::Mat mat1(1, 1);
    mat1(0, 0) = 5;
    std::cout << "Original Matrix:\n";
    mat1.print_matrix(true);
    tiny::Mat adj1 = mat1.adjoint();
    std::cout << "Adjoint Matrix:\n";
    adj1.print_matrix(true);  // Expected: [1]

    // Test 5.4.2: 2x2 Matrix
    std::cout << "\n[Test 5.4.2] Adjoint of 2x2 Matrix\n";
    tiny::Mat mat2(2, 2);
    mat2(0, 0) = 1; mat2(0, 1) = 2;
    mat2(1, 0) = 3; mat2(1, 1) = 4;
    std::cout << "Original Matrix:\n";
    mat2.print_matrix(true);
    tiny::Mat adj2 = mat2.adjoint();
    std::cout << "Adjoint Matrix:\n";
    adj2.print_matrix(true);  // Expected: [4, -2; -3, 1]

    // Test 5.4.3: 3x3 Matrix
    std::cout << "\n[Test 5.4.3] Adjoint of 3x3 Matrix\n";
    tiny::Mat mat3(3, 3);
    mat3(0,0) = 1; mat3(0,1) = 2; mat3(0,2) = 3;
    mat3(1,0) = 0; mat3(1,1) = 4; mat3(1,2) = 5;
    mat3(2,0) = 1; mat3(2,1) = 0; mat3(2,2) = 6;
    std::cout << "Original Matrix:\n";
    mat3.print_matrix(true);
    tiny::Mat adj3 = mat3.adjoint();
    std::cout << "Adjoint Matrix:\n";
    adj3.print_matrix(true);
    // No simple expected value, but should compute correctly

    // Test 5.4.4: Non-Square Matrix (Expect Error)
    std::cout << "\n[Test 5.4.4] Adjoint of Non-Square Matrix (Expect Error)\n";
    tiny::Mat rectMat(2, 3);
    std::cout << "Original Matrix (2x3, non-square):\n";
    rectMat.print_matrix(true);
    tiny::Mat adjRect = rectMat.adjoint();
    std::cout << "Adjoint Matrix (should be empty due to error):\n";
    adjRect.print_matrix(true);  // Should be empty or default matrix

}

// Group 5.5: Matrix Normalization
void test_matrix_normalize()
{
    std::cout << "\n[Group 5.5: Matrix Normalization Tests]\n";

    // Test 5.5.1: Standard normalization
    std::cout << "\n[Test 5.5.1] Normalize a Standard 2x2 Matrix\n";
    tiny::Mat mat1(2, 2);
    mat1(0, 0) = 3.0f; mat1(0, 1) = 4.0f;
    mat1(1, 0) = 3.0f; mat1(1, 1) = 4.0f;

    std::cout << "Before normalization:\n";
    mat1.print_matrix(true);

    mat1.normalize();

    std::cout << "After normalization (Expected L2 norm = 1):\n";
    mat1.print_matrix(true);

    // Test 5.5.2: Matrix with padding
    std::cout << "\n[Test 5.5.2] Normalize a 2x2 Matrix with Stride=4 (Padding Test)\n";
    float data_with_padding[8] = {3.0f, 4.0f, 0.0f, 0.0f, 3.0f, 4.0f, 0.0f, 0.0f};
    tiny::Mat mat2(data_with_padding, 2, 2, 4);  // 2x2 matrix, stride 4

    std::cout << "Before normalization:\n";
    mat2.print_matrix(true);

    mat2.normalize();

    std::cout << "After normalization:\n";
    mat2.print_matrix(true);

    // Test 5.5.3: Zero matrix normalization
    std::cout << "\n[Test 5.5.3] Normalize a Zero Matrix (Expect Warning)\n";
    tiny::Mat mat3(2, 2);
    mat3.clear();  // Assuming clear() sets all elements to zero

    mat3.print_matrix(true);
    mat3.normalize();  // Should trigger warning
}

// Group 5.6: Matrix Norm Calculation
void test_matrix_norm()
{
    std::cout << "\n[Group 5.6: Matrix Norm Calculation Tests]\n";

    // Test 5.6.1: Simple 2x2 Matrix
    std::cout << "\n[Test 5.6.1] 2x2 Matrix Norm (Expect 5.0)\n";
    tiny::Mat mat1(2, 2);
    mat1(0, 0) = 3.0f; mat1(0, 1) = 4.0f;
    mat1(1, 0) = 0.0f; mat1(1, 1) = 0.0f;
    std::cout << "Matrix:\n";
    mat1.print_matrix(true);
    float norm1 = mat1.norm();
    std::cout << "Calculated Norm: " << norm1 << "\n";

    // Test 5.6.2: Zero Matrix
    std::cout << "\n[Test 5.6.2] Zero Matrix Norm (Expect 0.0)\n";
    tiny::Mat mat2(3, 3);
    mat2.clear();  // Assuming clear() sets all elements to zero
    std::cout << "Matrix:\n";
    mat2.print_matrix(true);
    float norm2 = mat2.norm();
    std::cout << "Calculated Norm: " << norm2 << "\n";

    // Test 5.6.3: Matrix with Negative Values
    std::cout << "\n[Test 5.6.3] Matrix with Negative Values\n";
    tiny::Mat mat3(2, 2);
    mat3(0, 0) = -1.0f; mat3(0, 1) = -2.0f;
    mat3(1, 0) = -3.0f; mat3(1, 1) = -4.0f;
    std::cout << "Matrix:\n";
    mat3.print_matrix(true);
    float norm3 = mat3.norm();
    std::cout << "Calculated Norm: " << norm3 << "  (Expect sqrt(30) ≈ 5.477)\n";

    // Test 5.6.4: Matrix with Padding
    std::cout << "\n[Test 5.6.4] 2x2 Matrix with Stride=4 (Padding Test)\n";
    float data4[8] = {1.0f, 2.0f, 0.0f, 0.0f, 3.0f, 4.0f, 0.0f, 0.0f};
    tiny::Mat mat4(data4, 2, 2, 4);  // 2x2 matrix, stride 4
    std::cout << "Matrix:\n";
    mat4.print_matrix(true);
    float norm4 = mat4.norm();
    std::cout << "Calculated Norm: " << norm4 << "  (Expect sqrt(30) ≈ 5.477)\n";
}

// Group 5.7: Matrix Inversion
void test_inverse_adjoint_adjoint()
{
    std::cout << "\n[Group 5.7: Matrix Inversion Tests]\n";

    // Test 5.7.1: 2x2 Regular Matrix
    std::cout << "\n[Test 5.7.1] Inverse of 2x2 Matrix\n";
    tiny::Mat mat1(2, 2);
    mat1(0, 0) = 4;  mat1(0, 1) = 7;
    mat1(1, 0) = 2;  mat1(1, 1) = 6;
    std::cout << "Original Matrix:\n";
    mat1.print_matrix(true);
    tiny::Mat inv1 = mat1.inverse_adjoint();
    std::cout << "Inverse Matrix:\n";
    inv1.print_matrix(true);
    std::cout << "Expected Approx:\n[ 0.6  -0.7 ]\n[ -0.2  0.4 ]\n";

    // Test 5.7.2: Singular Matrix (Determinant = 0)
    std::cout << "\n[Test 5.7.2] Singular Matrix (Expect Error)\n";
    tiny::Mat mat2(2, 2);
    mat2(0, 0) = 1;  mat2(0, 1) = 2;
    mat2(1, 0) = 2;  mat2(1, 1) = 4;   // Rank-deficient, det = 0
    std::cout << "Original Matrix:\n";
    mat2.print_matrix(true);
    std::cout << "Note: This matrix is singular (determinant = 0), so inverse should fail.\n";
    tiny::Mat inv2 = mat2.inverse_adjoint();
    std::cout << "Inverse Matrix (Should be zero matrix):\n";
    inv2.print_matrix(true);

    // Test 5.7.3: 3x3 Regular Matrix
    std::cout << "\n[Test 5.7.3] Inverse of 3x3 Matrix\n";
    tiny::Mat mat3(3, 3);
    mat3(0,0) = 3; mat3(0,1) = 0; mat3(0,2) = 2;
    mat3(1,0) = 2; mat3(1,1) = 0; mat3(1,2) = -2;
    mat3(2,0) = 0; mat3(2,1) = 1; mat3(2,2) = 1;
    std::cout << "Original Matrix:\n";
    mat3.print_matrix(true);
    tiny::Mat inv3 = mat3.inverse_adjoint();
    std::cout << "Inverse Matrix:\n";
    inv3.print_matrix(true);

    // Test 5.7.4: Non-Square Matrix (Expect Error)
    std::cout << "\n[Test 5.7.4] Non-Square Matrix (Expect Error)\n";
    tiny::Mat mat4(2, 3);
    std::cout << "Original Matrix (2x3, non-square):\n";
    mat4.print_matrix(true);
    tiny::Mat inv4 = mat4.inverse_adjoint();
    std::cout << "Inverse Matrix (should be empty due to error):\n";
    inv4.print_matrix(true);
}

// Group 5.8: Matrix Utilities
void test_matrix_utilities()
{
    std::cout << "\n[Group 5.8: Matrix Utilities Tests]\n";

    // Test 5.8.1: Identity Matrix (eye)
    std::cout << "\n[Test 5.8.1] Generate Identity Matrix (eye)\n";
    tiny::Mat I3 = tiny::Mat::eye(3);
    std::cout << "3x3 Identity Matrix:\n";
    I3.print_matrix(true);

    tiny::Mat I5 = tiny::Mat::eye(5);
    std::cout << "5x5 Identity Matrix:\n";
    I5.print_matrix(true);

    // Test 5.8.2: Ones Matrix
    std::cout << "\n[Test 5.8.2] Generate Ones Matrix\n";
    tiny::Mat ones_3x4 = tiny::Mat::ones(3, 4);
    std::cout << "3x4 Ones Matrix:\n";
    ones_3x4.print_matrix(true);

    tiny::Mat ones_4x4 = tiny::Mat::ones(4);
    std::cout << "4x4 Ones Matrix (Square):\n";
    ones_4x4.print_matrix(true);

    // Test 5.8.3: Matrix Augmentation
    std::cout << "\n[Test 5.8.3] Augment Two Matrices Horizontally [A | B]\n";

    // Prepare matrices A (2x2) and B (2x3)
    tiny::Mat A(2, 2);
    A(0,0) = 1;  A(0,1) = 2;
    A(1,0) = 3;  A(1,1) = 4;

    tiny::Mat B(2, 3);
    B(0,0) = 5;  B(0,1) = 6;  B(0,2) = 7;
    B(1,0) = 8;  B(1,1) = 9;  B(1,2) = 10;

    std::cout << "Matrix A:\n";
    A.print_matrix(true);
    std::cout << "Matrix B:\n";
    B.print_matrix(true);

    tiny::Mat AB = tiny::Mat::augment(A, B);
    std::cout << "Augmented Matrix [A | B]:\n";
    AB.print_matrix(true);

    // Test 5.8.4: Row mismatch case
    std::cout << "\n[Test 5.8.4] Augment with Row Mismatch (Expect Error)\n";
    tiny::Mat C(3, 2);  // 3x2 matrix
    tiny::Mat invalidAug = tiny::Mat::augment(A, C);
    invalidAug.print_info();  // Should show empty matrix due to error

    // Test 5.8.5: Vertical Stack (vstack)
    std::cout << "\n[Test 5.8.5] Vertically Stack Two Matrices [A; B]\n";

    // Prepare matrices A (2x3) and B (2x3)
    tiny::Mat A_vstack(2, 3);
    A_vstack(0,0) = 1;  A_vstack(0,1) = 2;  A_vstack(0,2) = 3;
    A_vstack(1,0) = 4;  A_vstack(1,1) = 5;  A_vstack(1,2) = 6;

    tiny::Mat B_vstack(2, 3);
    B_vstack(0,0) = 7;  B_vstack(0,1) = 8;  B_vstack(0,2) = 9;
    B_vstack(1,0) = 10; B_vstack(1,1) = 11; B_vstack(1,2) = 12;

    std::cout << "Matrix A (top):\n";
    A_vstack.print_matrix(true);
    std::cout << "Matrix B (bottom):\n";
    B_vstack.print_matrix(true);

    tiny::Mat AB_vstack = tiny::Mat::vstack(A_vstack, B_vstack);
    std::cout << "Vertically Stacked Matrix [A; B]:\n";
    AB_vstack.print_matrix(true);
    std::cout << "Expected: 4x3 matrix with A on top, B on bottom\n";

    // Test 5.8.6: Vertical Stack with different row counts
    std::cout << "\n[Test 5.8.6] Vertical Stack with Different Row Counts (Same Columns)\n";
    tiny::Mat A_small(1, 3);
    A_small(0,0) = 1; A_small(0,1) = 2; A_small(0,2) = 3;

    tiny::Mat B_large(3, 3);
    B_large(0,0) = 4;  B_large(0,1) = 5;  B_large(0,2) = 6;
    B_large(1,0) = 7;  B_large(1,1) = 8;  B_large(1,2) = 9;
    B_large(2,0) = 10; B_large(2,1) = 11; B_large(2,2) = 12;

    std::cout << "Matrix A (1x3):\n";
    A_small.print_matrix(true);
    std::cout << "Matrix B (3x3):\n";
    B_large.print_matrix(true);

    tiny::Mat AB_mixed = tiny::Mat::vstack(A_small, B_large);
    std::cout << "Vertically Stacked Matrix [A; B] (1x3 + 3x3 = 4x3):\n";
    AB_mixed.print_matrix(true);

    // Test 5.8.7: Column mismatch case (Expect Error)
    std::cout << "\n[Test 5.8.7] VStack with Column Mismatch (Expect Error)\n";
    tiny::Mat A_col(2, 2);
    A_col(0,0) = 1; A_col(0,1) = 2;
    A_col(1,0) = 3; A_col(1,1) = 4;

    tiny::Mat B_col(2, 3);  // Different column count
    B_col(0,0) = 5; B_col(0,1) = 6; B_col(0,2) = 7;
    B_col(1,0) = 8; B_col(1,1) = 9; B_col(1,2) = 10;

    std::cout << "Matrix A (2x2):\n";
    A_col.print_matrix(true);
    std::cout << "Matrix B (2x3, different columns):\n";
    B_col.print_matrix(true);

    tiny::Mat invalidVStack = tiny::Mat::vstack(A_col, B_col);
    std::cout << "Result (should be empty due to error):\n";
    invalidVStack.print_info();  // Should show empty matrix due to error

}

// Group 5.9: Gaussian Elimination
void test_gaussian_eliminate()
{
    std::cout << "\n[Group 5.9: Gaussian Elimination Tests]\n";

    // Test 5.9.1: Simple 3x3 System
    std::cout << "\n[Test 5.9.1] 3x3 Matrix (Simple Upper Triangular)\n";
    tiny::Mat mat1(3, 3);
    mat1(0,0) = 2; mat1(0,1) = 1; mat1(0,2) = -1;
    mat1(1,0) = -3; mat1(1,1) = -1; mat1(1,2) = 2;
    mat1(2,0) = -2; mat1(2,1) = 1; mat1(2,2) = 2;

    std::cout << "Original Matrix:\n";
    mat1.print_matrix(true);

    tiny::Mat result1 = mat1.gaussian_eliminate();

    std::cout << "After Gaussian Elimination (Should be upper triangular):\n";
    result1.print_matrix(true);

    // Test 5.9.2: 3x4 Augmented Matrix
    std::cout << "\n[Test 5.9.2] 3x4 Augmented Matrix (Linear System Ax = b)\n";
    tiny::Mat mat2(3, 4);
    mat2(0,0) = 1; mat2(0,1) = 2; mat2(0,2) = -1; mat2(0,3) =  8;
    mat2(1,0) = -3; mat2(1,1) = -1; mat2(1,2) = 2; mat2(1,3) = -11;
    mat2(2,0) = -2; mat2(2,1) = 1; mat2(2,2) = 2; mat2(2,3) = -3;

    std::cout << "Original Augmented Matrix [A | b]:\n";
    mat2.print_matrix(true);

    tiny::Mat result2 = mat2.gaussian_eliminate();

    std::cout << "After Gaussian Elimination (Row Echelon Form):\n";
    result2.print_matrix(true);

    // Test 5.9.3: Singular Matrix
    std::cout << "\n[Test 5.9.3] Singular Matrix (No Unique Solution)\n";
    tiny::Mat mat3(2, 2);
    mat3(0,0) = 1; mat3(0,1) = 2;
    mat3(1,0) = 2; mat3(1,1) = 4;  // Linearly dependent rows

    std::cout << "Original Singular Matrix:\n";
    mat3.print_matrix(true);

    tiny::Mat result3 = mat3.gaussian_eliminate();
    std::cout << "After Gaussian Elimination (Should show rows of zeros):\n";
    result3.print_matrix(true);

    // Test 5.9.4: Zero Matrix
    std::cout << "\n[Test 5.9.4] Zero Matrix\n";
    tiny::Mat mat4(3, 3);
    mat4.clear();  // Assuming clear() sets all elements to zero
    mat4.print_matrix(true);

    tiny::Mat result4 = mat4.gaussian_eliminate();
    std::cout << "After Gaussian Elimination (Should be a zero matrix):\n";
    result4.print_matrix(true);
}


// Group 5.10: Row Reduce from Gaussian (RREF Calculation)
void test_row_reduce_from_gaussian()
{
    std::cout << "\n[Group 5.10: Row Reduce from Gaussian (RREF) Tests]\n";

    // Test 5.10.1: Simple 3x4 augmented matrix (representing a system of equations)
    std::cout << "\n[Test 5.10.1] 3x4 Augmented Matrix\n";
    tiny::Mat mat1(3, 4);

    // Matrix:
    // [ 1  2 -1  -4 ]
    // [ 2  3 -1 -11 ]
    // [-2  0 -3  22 ]
    mat1(0,0) = 1;  mat1(0,1) = 2;  mat1(0,2) = -1; mat1(0,3) = -4;
    mat1(1,0) = 2;  mat1(1,1) = 3;  mat1(1,2) = -1; mat1(1,3) = -11;
    mat1(2,0) = -2; mat1(2,1) = 0;  mat1(2,2) = -3; mat1(2,3) = 22;

    std::cout << "Original Matrix:\n";
    mat1.print_matrix(true);

    tiny::Mat rref1 = mat1.gaussian_eliminate().row_reduce_from_gaussian();
    std::cout << "RREF Result:\n";
    rref1.print_matrix(true);

    // Test 5.10.2: 2x3 Matrix
    std::cout << "\n[Test 5.10.2] 2x3 Matrix\n";
    tiny::Mat mat2(2, 3);
    mat2(0,0) = 1; mat2(0,1) = 2;  mat2(0,2) = 3;
    mat2(1,0) = 4; mat2(1,1) = 5;  mat2(1,2) = 6;

    std::cout << "Original Matrix:\n";
    mat2.print_matrix(true);

    tiny::Mat rref2 = mat2.gaussian_eliminate().row_reduce_from_gaussian();
    std::cout << "RREF Result:\n";
    rref2.print_matrix(true);

    // Test 5.10.3: Already reduced matrix (should remain the same)
    std::cout << "\n[Test 5.10.3] Already Reduced Matrix\n";
    tiny::Mat mat3(2, 3);
    mat3(0,0) = 1; mat3(0,1) = 0; mat3(0,2) = 2;
    mat3(1,0) = 0; mat3(1,1) = 1; mat3(1,2) = 3;

    std::cout << "Original Matrix:\n";
    mat3.print_matrix(true);

    tiny::Mat rref3 = mat3.row_reduce_from_gaussian();
    std::cout << "RREF Result:\n";
    rref3.print_matrix(true);
}

// Group 5.11: Gaussian Inverse
void test_inverse_gje()
{
    std::cout << "\n[Group 5.11: Gaussian Inverse Tests]\n";

    // Test 5.11.1: Regular 2x2 Matrix
    std::cout << "\n[Test 5.11.1] 2x2 Matrix Inverse\n";
    tiny::Mat mat1(2, 2);
    mat1(0, 0) = 4; mat1(0, 1) = 7;
    mat1(1, 0) = 2; mat1(1, 1) = 6;
    std::cout << "Original matrix (mat1):\n";
    mat1.print_matrix(true);
    
    tiny::Mat invMat1 = mat1.inverse_gje();
    std::cout << "Inverse matrix (mat1):\n";
    invMat1.print_matrix(true);

    // Test 5.11.2: Identity Matrix (should return identity matrix)
    std::cout << "\n[Test 5.11.2] Identity Matrix Inverse\n";
    tiny::Mat mat2 = tiny::Mat::eye(3);
    std::cout << "Original matrix (Identity):\n";
    mat2.print_matrix(true);
    
    tiny::Mat invMat2 = mat2.inverse_gje();
    std::cout << "Inverse matrix (Identity):\n";
    invMat2.print_matrix(true); // Expected: Identity matrix

    // Test 5.11.3: Singular Matrix (should return empty matrix or indicate error)
    std::cout << "\n[Test 5.11.3] Singular Matrix (Expected: No Inverse)\n";
    tiny::Mat mat3(3, 3);
    mat3(0, 0) = 1; mat3(0, 1) = 2; mat3(0, 2) = 3;
    mat3(1, 0) = 4; mat3(1, 1) = 5; mat3(1, 2) = 6;
    mat3(2, 0) = 7; mat3(2, 1) = 8; mat3(2, 2) = 9;  // Determinant is 0
    std::cout << "Original matrix (singular):\n";
    mat3.print_matrix(true);
    
    tiny::Mat invMat3 = mat3.inverse_gje();
    std::cout << "Inverse matrix (singular):\n";
    invMat3.print_matrix(true); // Expected: empty matrix or error message

    // Test 5.11.4: 3x3 Matrix with a valid inverse
    std::cout << "\n[Test 5.11.4] 3x3 Matrix Inverse\n";
    tiny::Mat mat4(3, 3);
    mat4(0, 0) = 4; mat4(0, 1) = 7; mat4(0, 2) = 2;
    mat4(1, 0) = 3; mat4(1, 1) = 5; mat4(1, 2) = 1;
    mat4(2, 0) = 8; mat4(2, 1) = 6; mat4(2, 2) = 9;
    std::cout << "Original matrix (mat4):\n";
    mat4.print_matrix(true);
    
    tiny::Mat invMat4 = mat4.inverse_gje();
    std::cout << "Inverse matrix (mat4):\n";
    invMat4.print_matrix(true); // Check that the inverse is calculated correctly

    // Test 5.11.5: Non-square Matrix (should return error or empty matrix)
    std::cout << "\n[Test 5.11.5] Non-square Matrix Inverse (Expected Error)\n";
    tiny::Mat mat5(2, 3);
    mat5(0, 0) = 1; mat5(0, 1) = 2; mat5(0, 2) = 3;
    mat5(1, 0) = 4; mat5(1, 1) = 5; mat5(1, 2) = 6;
    std::cout << "Original matrix (non-square):\n";
    mat5.print_matrix(true);
    
    tiny::Mat invMat5 = mat5.inverse_gje();
    std::cout << "Inverse matrix (non-square):\n";
    invMat5.print_matrix(true); // Expected: Error message or empty matrix
}

// Group 5.12: Dot Product
void test_dotprod()
{
    std::cout << "\n[Group 5.12: Dot Product Tests]\n";

    // Test 5.12.1: Valid Dot Product Calculation (Same Length Vectors)
    std::cout << "\n[Test 5.12.1] Valid Dot Product (Same Length Vectors)\n";
    tiny::Mat vectorA(3, 1);  // Create a 3x1 vector
    tiny::Mat vectorB(3, 1);  // Create a 3x1 vector

    // Initialize vectors
    vectorA(0, 0) = 1.0f;
    vectorA(1, 0) = 2.0f;
    vectorA(2, 0) = 3.0f;

    vectorB(0, 0) = 4.0f;
    vectorB(1, 0) = 5.0f;
    vectorB(2, 0) = 6.0f;

    std::cout << "Vector A:\n";
    vectorA.print_matrix(true);
    std::cout << "Vector B:\n";
    vectorB.print_matrix(true);

    // Compute the dot product
    float result = vectorA.dotprod(vectorA, vectorB);
    std::cout << "Dot product of vectorA and vectorB: " << result << std::endl;  // Expected result: 1*4 + 2*5 + 3*6 = 32

    // Test 5.12.2: Dot Product with Dimension Mismatch (Different Length Vectors)
    std::cout << "\n[Test 5.12.2] Invalid Dot Product (Dimension Mismatch)\n";
    tiny::Mat vectorC(2, 1);  // Create a 2x1 vector (different size)
    vectorC(0, 0) = 1.0f;
    vectorC(1, 0) = 2.0f;

    std::cout << "Vector A (3x1):\n";
    vectorA.print_matrix(true);
    std::cout << "Vector C (2x1, different size):\n";
    vectorC.print_matrix(true);

    float invalidResult = vectorA.dotprod(vectorA, vectorC);  // Should print an error and return 0
    std::cout << "Dot product (dimension mismatch): " << invalidResult << std::endl;  // Expected: 0 and error message

    // Test 5.12.3: Dot Product of Zero Vectors
    std::cout << "\n[Test 5.12.3] Dot Product of Zero Vectors\n";
    tiny::Mat zeroVectorA(3, 1);  // Create a 3x1 zero vector
    tiny::Mat zeroVectorB(3, 1);  // Create a 3x1 zero vector

    // Initialize vectors
    zeroVectorA(0, 0) = 0.0f;
    zeroVectorA(1, 0) = 0.0f;
    zeroVectorA(2, 0) = 0.0f;

    zeroVectorB(0, 0) = 0.0f;
    zeroVectorB(1, 0) = 0.0f;
    zeroVectorB(2, 0) = 0.0f;

    std::cout << "Zero Vector A:\n";
    zeroVectorA.print_matrix(true);
    std::cout << "Zero Vector B:\n";
    zeroVectorB.print_matrix(true);

    float zeroResult = zeroVectorA.dotprod(zeroVectorA, zeroVectorB);
    std::cout << "Dot product of zero vectors: " << zeroResult << std::endl;  // Expected: 0

}

// Group 5.13: Solve Linear System
void test_solve()
{
    std::cout << "\n[Group 5.13: Solve Linear System Tests]\n";

    // Test 5.13.1: Solving a simple 2x2 system
    std::cout << "\n[Test 5.13.1] Solving a Simple 2x2 System Ax = b\n";
    tiny::Mat A(2, 2);
    tiny::Mat b(2, 1);

    A(0, 0) = 2; A(0, 1) = 1;
    A(1, 0) = 1; A(1, 1) = 3;

    b(0, 0) = 5;
    b(1, 0) = 6;

    std::cout << "Matrix A:\n";
    A.print_matrix(true);
    std::cout << "Vector b:\n";
    b.print_matrix(true);

    tiny::Mat solution = A.solve(A, b);
    std::cout << "Solution x:\n";
    solution.print_matrix(true);

    // Test 5.13.2: Solving a 3x3 system
    std::cout << "\n[Test 5.13.2] Solving a 3x3 System Ax = b\n";
    tiny::Mat A2(3, 3);
    tiny::Mat b2(3, 1);

    A2(0, 0) = 1; A2(0, 1) = 2; A2(0, 2) = 1;
    A2(1, 0) = 2; A2(1, 1) = 0; A2(1, 2) = 3;
    A2(2, 0) = 3; A2(2, 1) = 2; A2(2, 2) = 1;

    b2(0, 0) = 9;
    b2(1, 0) = 8;
    b2(2, 0) = 7;

    std::cout << "Matrix A:\n";
    A2.print_matrix(true);
    std::cout << "Vector b:\n";
    b2.print_matrix(true);

    tiny::Mat solution2 = A2.solve(A2, b2);
    std::cout << "Solution x:\n";
    solution2.print_matrix(true);

    // Test 5.13.3: Solving a system where one row is all zeros
    std::cout << "\n[Test 5.13.3] Solving a System Where One Row is All Zeros (Expect Failure or Infinite Solutions)\n";
    tiny::Mat A3(3, 3);
    tiny::Mat b3(3, 1);

    A3(0, 0) = 1; A3(0, 1) = 2; A3(0, 2) = 3;
    A3(1, 0) = 0; A3(1, 1) = 0; A3(1, 2) = 0; // Zero row
    A3(2, 0) = 4; A3(2, 1) = 5; A3(2, 2) = 6;

    b3(0, 0) = 9;
    b3(1, 0) = 0; // Inconsistent, no solution should be possible
    b3(2, 0) = 15;

    std::cout << "Matrix A (has zero row):\n";
    A3.print_matrix(true);
    std::cout << "Vector b:\n";
    b3.print_matrix(true);

    tiny::Mat solution3 = A3.solve(A3, b3);
    std::cout << "Solution x:\n";
    solution3.print_matrix(true);

    // Test 5.13.4: Solving a system with zero determinant (singular matrix)
    std::cout << "\n[Test 5.13.4] Solving a System with Zero Determinant (Singular Matrix)\n";
    tiny::Mat A4(3, 3);
    tiny::Mat b4(3, 1);

    A4(0, 0) = 2; A4(0, 1) = 4; A4(0, 2) = 1;
    A4(1, 0) = 1; A4(1, 1) = 2; A4(1, 2) = 3;
    A4(2, 0) = 3; A4(2, 1) = 6; A4(2, 2) = 2; // The matrix is singular (row 2 = 2 * row 1)

    b4(0, 0) = 5;
    b4(1, 0) = 6;
    b4(2, 0) = 7;

    std::cout << "Matrix A (singular, determinant = 0):\n";
    A4.print_matrix(true);
    std::cout << "Vector b:\n";
    b4.print_matrix(true);

    tiny::Mat solution4 = A4.solve(A4, b4);
    std::cout << "Solution x:\n";
    solution4.print_matrix(true); // Expect no solution or an error message

    // Test 5.13.5: Solving a system with linearly dependent rows
    std::cout << "\n[Test 5.13.5] Solving a System with Linearly Dependent Rows (Expect Failure or Infinite Solutions)\n";
    tiny::Mat A5(3, 3);
    tiny::Mat b5(3, 1);

    A5(0, 0) = 1; A5(0, 1) = 1; A5(0, 2) = 1;
    A5(1, 0) = 2; A5(1, 1) = 2; A5(1, 2) = 2;
    A5(2, 0) = 3; A5(2, 1) = 3; A5(2, 2) = 3; // All rows are linearly dependent

    b5(0, 0) = 6;
    b5(1, 0) = 12;
    b5(2, 0) = 18;

    std::cout << "Matrix A (all rows linearly dependent):\n";
    A5.print_matrix(true);
    std::cout << "Vector b:\n";
    b5.print_matrix(true);

    tiny::Mat solution5 = A5.solve(A5, b5);
    std::cout << "Solution x:\n";
    solution5.print_matrix(true); // Expect an error message or infinite solutions

    // Test 5.13.6: Solving a larger 4x4 system
    std::cout << "\n[Test 5.13.6] Solving a Larger 4x4 System Ax = b\n";
    tiny::Mat A6(4, 4);
    tiny::Mat b6(4, 1);

    A6(0, 0) = 4; A6(0, 1) = 2; A6(0, 2) = 3; A6(0, 3) = 1;
    A6(1, 0) = 2; A6(1, 1) = 5; A6(1, 2) = 1; A6(1, 3) = 2;
    A6(2, 0) = 3; A6(2, 1) = 1; A6(2, 2) = 6; A6(2, 3) = 3;
    A6(3, 0) = 1; A6(3, 1) = 2; A6(3, 2) = 3; A6(3, 3) = 4;

    b6(0, 0) = 10;
    b6(1, 0) = 12;
    b6(2, 0) = 14;
    b6(3, 0) = 16;

    std::cout << "Matrix A:\n";
    A6.print_matrix(true);
    std::cout << "Vector b:\n";
    b6.print_matrix(true);

    tiny::Mat solution6 = A6.solve(A6, b6);
    std::cout << "Solution x:\n";
    solution6.print_matrix(true); // Should print the solution vector

}

// Group 5.14: Band Solve
void test_band_solve()
{
    std::cout << "\n[Group 5.14: Band Solve Tests]\n";

    // Test 5.14.1: Simple 3x3 Band Matrix
    std::cout << "\n[Test 5.14.1] Simple 3x3 Band Matrix\n";
    tiny::Mat A1(3, 3);
    tiny::Mat b1(3, 1);

    // Define the matrix A and vector b for the system Ax = b
    A1(0, 0) = 2; A1(0, 1) = 1; A1(0, 2) = 0;
    A1(1, 0) = 1; A1(1, 1) = 3; A1(1, 2) = 2;
    A1(2, 0) = 0; A1(2, 1) = 1; A1(2, 2) = 4;

    b1(0, 0) = 5;
    b1(1, 0) = 6;
    b1(2, 0) = 7;

    std::cout << "Matrix A:\n";
    A1.print_matrix(true);
    std::cout << "Vector b:\n";
    b1.print_matrix(true);

    // Solve Ax = b using band_solve
    tiny::Mat solution1 = A1.band_solve(A1, b1, 3);
    std::cout << "Solution x:\n";
    solution1.print_matrix(true);

    // Test 5.14.2: 4x4 Band Matrix with different right-hand side vector
    std::cout << "\n[Test 5.14.2] 4x4 Band Matrix\n";
    tiny::Mat A2(4, 4);
    tiny::Mat b2(4, 1);

    // Define the matrix A and vector b
    A2(0, 0) = 2; A2(0, 1) = 1; A2(0, 2) = 0; A2(0, 3) = 0;
    A2(1, 0) = 1; A2(1, 1) = 3; A2(1, 2) = 2; A2(1, 3) = 0;
    A2(2, 0) = 0; A2(2, 1) = 1; A2(2, 2) = 4; A2(2, 3) = 2;
    A2(3, 0) = 0; A2(3, 1) = 0; A2(3, 2) = 1; A2(3, 3) = 5;

    b2(0, 0) = 8;
    b2(1, 0) = 9;
    b2(2, 0) = 10;
    b2(3, 0) = 11;

    std::cout << "Matrix A:\n";
    A2.print_matrix(true);
    std::cout << "Vector b:\n";
    b2.print_matrix(true);

    // Solve Ax = b using band_solve
    tiny::Mat solution2 = A2.band_solve(A2, b2, 3);
    std::cout << "Solution x:\n";
    solution2.print_matrix(true);

    // Test 5.14.3: Incompatible dimensions (expect error)
    std::cout << "\n[Test 5.14.3] Incompatible Dimensions (Expect Error)\n";
    tiny::Mat A3(3, 3);
    tiny::Mat b3(2, 1);  // Incompatible dimension

    A3(0, 0) = 1; A3(0, 1) = 2; A3(0, 2) = 3;
    A3(1, 0) = 4; A3(1, 1) = 5; A3(1, 2) = 6;
    A3(2, 0) = 7; A3(2, 1) = 8; A3(2, 2) = 9;

    b3(0, 0) = 10;
    b3(1, 0) = 11;

    std::cout << "Matrix A (3x3):\n";
    A3.print_matrix(true);
    std::cout << "Vector b (2x1, incompatible):\n";
    b3.print_matrix(true);

    // This should print an error because of incompatible dimensions
    tiny::Mat solution3 = A3.band_solve(A3, b3, 3);
    std::cout << "Solution x:\n";
    solution3.print_matrix(true);

    // Test 5.14.4: Singular Matrix (Should fail)
    std::cout << "\n[Test 5.14.4] Singular Matrix (No Unique Solution)\n";
    tiny::Mat A4(3, 3);
    tiny::Mat b4(3, 1);

    // Define a singular matrix (linearly dependent rows)
    A4(0, 0) = 1; A4(0, 1) = 2; A4(0, 2) = 3;
    A4(1, 0) = 2; A4(1, 1) = 4; A4(1, 2) = 6;
    A4(2, 0) = 3; A4(2, 1) = 6; A4(2, 2) = 9;

    b4(0, 0) = 10;
    b4(1, 0) = 20;
    b4(2, 0) = 30;

    std::cout << "Matrix A (singular, linearly dependent rows):\n";
    A4.print_matrix(true);
    std::cout << "Vector b:\n";
    b4.print_matrix(true);

    // This should print an error as the matrix is singular and does not have a unique solution
    tiny::Mat solution4 = A4.band_solve(A4, b4, 3);
    std::cout << "Solution x:\n";
    solution4.print_matrix(true);
}

// Group 5.15: Roots
void test_roots()
{
    std::cout << "\n[Group 5.15: Roots Tests]\n";

    // Test 5.15.1: Simple 2x2 System
    std::cout << "\n[Test 5.15.1] Solving a Simple 2x2 System Ax = b\n";
    tiny::Mat A1(2, 2);
    tiny::Mat b1(2, 1);

    // Define the matrix A and vector b for the system Ax = b
    A1(0, 0) = 2; A1(0, 1) = 1;
    A1(1, 0) = 1; A1(1, 1) = 3;

    b1(0, 0) = 5;
    b1(1, 0) = 6;

    std::cout << "Matrix A:\n";
    A1.print_matrix(true);
    std::cout << "Vector b:\n";
    b1.print_matrix(true);

    // Solve Ax = b using roots
    tiny::Mat solution1 = A1.roots(A1, b1);
    std::cout << "Solution x:\n";
    solution1.print_matrix(true);

    // Test 5.15.2: 3x3 System
    std::cout << "\n[Test 5.15.2] Solving a 3x3 System Ax = b\n";
    tiny::Mat A2(3, 3);
    tiny::Mat b2(3, 1);

    A2(0, 0) = 1; A2(0, 1) = 2; A2(0, 2) = 1;
    A2(1, 0) = 2; A2(1, 1) = 0; A2(1, 2) = 3;
    A2(2, 0) = 3; A2(2, 1) = 2; A2(2, 2) = 1;

    b2(0, 0) = 9;
    b2(1, 0) = 8;
    b2(2, 0) = 7;

    std::cout << "Matrix A:\n";
    A2.print_matrix(true);
    std::cout << "Vector b:\n";
    b2.print_matrix(true);

    // Solve Ax = b using roots
    tiny::Mat solution2 = A2.roots(A2, b2);
    std::cout << "Solution x:\n";
    solution2.print_matrix(true);

    // Test 5.15.3: Singular Matrix
    std::cout << "\n[Test 5.15.3] Singular Matrix (No Unique Solution)\n";
    tiny::Mat A3(2, 2);
    tiny::Mat b3(2, 1);

    // Define a singular matrix (linearly dependent rows)
    A3(0, 0) = 1; A3(0, 1) = 2;
    A3(1, 0) = 2; A3(1, 1) = 4;

    b3(0, 0) = 5;
    b3(1, 0) = 6;

    std::cout << "Matrix A (singular, linearly dependent rows):\n";
    A3.print_matrix(true);
    std::cout << "Vector b:\n";
    b3.print_matrix(true);

    // This should print an error as the matrix is singular and does not have a unique solution
    tiny::Mat solution3 = A3.roots(A3, b3);
    std::cout << "Solution x:\n";
    solution3.print_matrix(true);

    // Test 5.15.4: Incompatible Dimensions (Expect Error)
    std::cout << "\n[Test 5.15.4] Incompatible Dimensions (Expect Error)\n";
    tiny::Mat A4(3, 3);
    tiny::Mat b4(2, 1);  // Incompatible dimension

    A4(0, 0) = 1; A4(0, 1) = 2; A4(0, 2) = 3;
    A4(1, 0) = 4; A4(1, 1) = 5; A4(1, 2) = 6;
    A4(2, 0) = 7; A4(2, 1) = 8; A4(2, 2) = 9;

    b4(0, 0) = 10;
    b4(1, 0) = 11;

    std::cout << "Matrix A (3x3):\n";
    A4.print_matrix(true);
    std::cout << "Vector b (2x1, incompatible):\n";
    b4.print_matrix(true);

    // This should print an error because of incompatible dimensions
    tiny::Mat solution4 = A4.roots(A4, b4);
    std::cout << "Solution x:\n";
    solution4.print_matrix(true);
}

// Group 6: Stream Operators
void test_stream_operators()
{
    std::cout << "\n[Group 6: Stream Operators Tests]\n";

    // Test 6.1: Test stream insertion operator (<<) for Mat
    std::cout << "\n[Test 6.1] Stream Insertion Operator (<<) for Mat\n";
    tiny::Mat mat1(3, 3);
    mat1(0, 0) = 1; mat1(0, 1) = 2; mat1(0, 2) = 3;
    mat1(1, 0) = 4; mat1(1, 1) = 5; mat1(1, 2) = 6;
    mat1(2, 0) = 7; mat1(2, 1) = 8; mat1(2, 2) = 9;

    std::cout << "Matrix mat1:\n";
    std::cout << mat1 << std::endl; // Use the << operator to print mat1

    // Test 6.2: Test stream insertion operator (<<) for Mat::ROI
    std::cout << "\n[Test 6.2] Stream Insertion Operator (<<) for Mat::ROI\n";
    tiny::Mat::ROI roi(1, 2, 3, 4);
    // ROI constructor: ROI(pos_x, pos_y, width, height)
    // roi(1, 2, 3, 4) means: start at column 1, row 2, with width 3, height 4
    std::cout << "ROI created: ROI(pos_x=1, pos_y=2, width=3, height=4)\n";
    std::cout << "Expected output:\n";
    std::cout << "  row start: 2 (pos_y)\n";
    std::cout << "  col start: 1 (pos_x)\n";
    std::cout << "  row count: 4 (height)\n";
    std::cout << "  col count: 3 (width)\n";
    std::cout << "\nActual output:\n";
    std::cout << roi << std::endl; // Use the << operator to print roi

    // Test 6.3: Test stream extraction operator (>>) for Mat
    std::cout << "\n[Test 6.3] Stream Extraction Operator (>>) for Mat\n";
    tiny::Mat mat2(2, 2);
    // Use istringstream to simulate input (for automated testing)
    std::istringstream input1("10 20 30 40");
    std::cout << "Simulated input: \"10 20 30 40\"\n";
    input1 >> mat2; // Use the >> operator to read from string stream
    std::cout << "Matrix mat2 after input:\n";
    std::cout << mat2 << std::endl; // Use the << operator to print mat2
    std::cout << "Expected: [10, 20; 30, 40]\n";

    // Test 6.4: Test stream extraction operator (>>) for Mat (with different values)
    std::cout << "\n[Test 6.4] Stream Extraction Operator (>>) for Mat (2x3 matrix)\n";
    tiny::Mat mat3(2, 3);
    // Use istringstream to simulate input (for automated testing)
    std::istringstream input2("1.5 2.5 3.5 4.5 5.5 6.5");
    std::cout << "Simulated input: \"1.5 2.5 3.5 4.5 5.5 6.5\"\n";
    input2 >> mat3; // Use the >> operator to read from string stream
    std::cout << "Matrix mat3 after input:\n";
    std::cout << mat3 << std::endl; // Use the << operator to print mat3
    std::cout << "Expected: [1.5, 2.5, 3.5; 4.5, 5.5, 6.5]\n";
}

// Group 7: Global Arithmetic Operators
void test_matrix_operations()
{
    std::cout << "\n[Group 7: Global Arithmetic Operators Tests]\n";

    // Test 7.1: Matrix Addition (operator+)
    std::cout << "\n[Test 7.1] Matrix Addition (operator+)\n";
    tiny::Mat matA(2, 2);
    tiny::Mat matB(2, 2);
    
    matA(0, 0) = 1; matA(0, 1) = 2;
    matA(1, 0) = 3; matA(1, 1) = 4;

    matB(0, 0) = 5; matB(0, 1) = 6;
    matB(1, 0) = 7; matB(1, 1) = 8;

    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Matrix B:\n";
    matB.print_matrix(true);

    tiny::Mat resultAdd = matA + matB;
    std::cout << "matA + matB:\n";
    std::cout << resultAdd << std::endl;  // Expected: [6, 8], [10, 12]

    // Test 7.2: Matrix Addition with Constant (operator+)
    std::cout << "\n[Test 7.2] Matrix Addition with Constant (operator+)\n";
    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Constant: 5.0\n";
    tiny::Mat resultAddConst = matA + 5.0f;
    std::cout << "matA + 5.0f:\n";
    std::cout << resultAddConst << std::endl;  // Expected: [6, 7], [8, 9]

    // Test 7.3: Matrix Subtraction (operator-)
    std::cout << "\n[Test 7.3] Matrix Subtraction (operator-)\n";
    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Matrix B:\n";
    matB.print_matrix(true);
    tiny::Mat resultSub = matA - matB;
    std::cout << "matA - matB:\n";
    std::cout << resultSub << std::endl;  // Expected: [-4, -4], [-4, -4]

    // Test 7.4: Matrix Subtraction with Constant (operator-)
    std::cout << "\n[Test 7.4] Matrix Subtraction with Constant (operator-)\n";
    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Constant: 2.0\n";
    tiny::Mat resultSubConst = matA - 2.0f;
    std::cout << "matA - 2.0f:\n";
    std::cout << resultSubConst << std::endl;  // Expected: [-1, 0], [1, 2]

    // Test 7.5: Matrix Multiplication (operator*)
    std::cout << "\n[Test 7.5] Matrix Multiplication (operator*)\n";
    tiny::Mat matC(2, 3);
    tiny::Mat matD(3, 2);

    matC(0, 0) = 1; matC(0, 1) = 2; matC(0, 2) = 3;
    matC(1, 0) = 4; matC(1, 1) = 5; matC(1, 2) = 6;

    matD(0, 0) = 7; matD(0, 1) = 8;
    matD(1, 0) = 9; matD(1, 1) = 10;
    matD(2, 0) = 11; matD(2, 1) = 12;

    std::cout << "Matrix C (2x3):\n";
    matC.print_matrix(true);
    std::cout << "Matrix D (3x2):\n";
    matD.print_matrix(true);

    tiny::Mat resultMul = matC * matD;
    std::cout << "matC * matD:\n";
    std::cout << resultMul << std::endl;  // Expected: [58, 64], [139, 154]

    // Test 7.6: Matrix Multiplication with Constant (operator*)
    std::cout << "\n[Test 7.6] Matrix Multiplication with Constant (operator*)\n";
    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Constant: 2.0\n";
    tiny::Mat resultMulConst = matA * 2.0f;
    std::cout << "matA * 2.0f:\n";
    std::cout << resultMulConst << std::endl;  // Expected: [2, 4], [6, 8]

    // Test 7.7: Matrix Division (operator/)
    std::cout << "\n[Test 7.7] Matrix Division (operator/)\n";
    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Constant: 2.0\n";
    tiny::Mat resultDiv = matA / 2.0f;
    std::cout << "matA / 2.0f:\n";
    std::cout << resultDiv << std::endl;  // Expected: [0.5, 1], [1.5, 2]

    // Test 7.8: Matrix Division Element-wise (operator/)
    std::cout << "\n[Test 7.8] Matrix Division Element-wise (operator/)\n";
    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Matrix B:\n";
    matB.print_matrix(true);
    tiny::Mat resultDivElem = matA / matB;
    std::cout << "matA / matB:\n";
    std::cout << resultDivElem << std::endl;  // Expected: [0.2, 0.333], [0.428, 0.5]

    // Test 7.9: Matrix Comparison (operator==)
    std::cout << "\n[Test 7.9] Matrix Comparison (operator==)\n";
    tiny::Mat matE(2, 2);
    matE(0, 0) = 1; matE(0, 1) = 2;
    matE(1, 0) = 3; matE(1, 1) = 4;

    tiny::Mat matF(2, 2);
    matF(0, 0) = 1; matF(0, 1) = 2;
    matF(1, 0) = 3; matF(1, 1) = 4;

    std::cout << "Matrix E:\n";
    matE.print_matrix(true);
    std::cout << "Matrix F:\n";
    matF.print_matrix(true);

    bool isEqual = (matE == matF);
    std::cout << "matE == matF: " << (isEqual ? "True" : "False") << std::endl;  // Expected: True

    matF(0, 0) = 5;  // Modify matF
    std::cout << "\nAfter modifying matF(0,0) = 5:\n";
    std::cout << "Matrix E:\n";
    matE.print_matrix(true);
    std::cout << "Matrix F:\n";
    matF.print_matrix(true);
    isEqual = (matE == matF);
    std::cout << "matE == matF after modification: " << (isEqual ? "True" : "False") << std::endl;  // Expected: False
}

// Group 8: Boundary Conditions and Error Handling
void test_boundary_conditions()
{
    std::cout << "\n[Group 8: Boundary Conditions and Error Handling Tests]\n";

    // Test 8.1: Null pointer handling in print functions
    std::cout << "\n[Test 8.1] Null Pointer Handling in print_matrix\n";
    tiny::Mat null_mat;
    null_mat.data = nullptr;  // Simulate null pointer
    null_mat.print_matrix(true);  // Should handle gracefully

    // Test 8.2: Null pointer handling in operator<<
    std::cout << "\n[Test 8.2] Null Pointer Handling in operator<<\n";
    tiny::Mat null_mat2;
    null_mat2.data = nullptr;
    std::cout << null_mat2 << std::endl;  // Should handle gracefully

    // Test 8.3: Invalid block parameters
    std::cout << "\n[Test 8.3] Invalid Block Parameters\n";
    tiny::Mat mat(3, 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            mat(i, j) = i * 3 + j + 1;
    
    // Negative start position
    tiny::Mat block1 = mat.block(-1, 0, 2, 2);
    std::cout << "block(-1, 0, 2, 2): " << (block1.data == nullptr ? "Empty (correct)" : "Error") << "\n";
    
    // Block exceeds boundaries
    tiny::Mat block2 = mat.block(2, 2, 2, 2);
    std::cout << "block(2, 2, 2, 2) on 3x3 matrix: " << (block2.data == nullptr ? "Empty (correct)" : "Error") << "\n";
    
    // Zero or negative block size
    tiny::Mat block3 = mat.block(0, 0, 0, 2);
    std::cout << "block(0, 0, 0, 2): " << (block3.data == nullptr ? "Empty (correct)" : "Error") << "\n";

    // Test 8.4: Invalid swap_rows parameters
    std::cout << "\n[Test 8.4] Invalid swap_rows Parameters\n";
    tiny::Mat mat2(3, 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            mat2(i, j) = i * 3 + j + 1;
    
    std::cout << "Before invalid swap_rows:\n";
    mat2.print_matrix(true);
    
    // Negative index
    mat2.swap_rows(-1, 1);
    std::cout << "After swap_rows(-1, 1):\n";
    mat2.print_matrix(true);
    
    // Index out of range
    mat2.swap_rows(0, 5);
    std::cout << "After swap_rows(0, 5):\n";
    mat2.print_matrix(true);

    // Test 8.5: Invalid swap_cols parameters
    std::cout << "\n[Test 8.5] Invalid swap_cols Parameters\n";
    tiny::Mat mat2_cols(3, 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            mat2_cols(i, j) = i * 3 + j + 1;
    
    std::cout << "Before invalid swap_cols:\n";
    mat2_cols.print_matrix(true);
    
    // Negative index
    mat2_cols.swap_cols(-1, 1);
    std::cout << "After swap_cols(-1, 1):\n";
    mat2_cols.print_matrix(true);
    
    // Index out of range
    mat2_cols.swap_cols(0, 5);
    std::cout << "After swap_cols(0, 5):\n";
    mat2_cols.print_matrix(true);

    // Test 8.6: Division by zero
    std::cout << "\n[Test 8.6] Division by Zero\n";
    tiny::Mat mat3(2, 2);
    mat3(0, 0) = 1; mat3(0, 1) = 2;
    mat3(1, 0) = 3; mat3(1, 1) = 4;
    
    tiny::Mat result = mat3 / 0.0f;
    std::cout << "mat3 / 0.0f: " << (result.data == nullptr ? "Empty (correct)" : "Error") << "\n";

    // Test 8.7: Matrix division with zero elements
    std::cout << "\n[Test 8.7] Matrix Division with Zero Elements\n";
    tiny::Mat mat4(2, 2);
    mat4(0, 0) = 1; mat4(0, 1) = 2;
    mat4(1, 0) = 3; mat4(1, 1) = 4;
    
    tiny::Mat divisor(2, 2);
    divisor(0, 0) = 1; divisor(0, 1) = 0;  // Contains zero
    divisor(1, 0) = 3; divisor(1, 1) = 4;
    
    mat4 /= divisor;
    std::cout << "mat4 /= divisor (with zero):\n";
    mat4.print_matrix(true);

    // Test 8.8: Empty matrix operations
    std::cout << "\n[Test 8.8] Empty Matrix Operations\n";
    tiny::Mat empty1, empty2;
    tiny::Mat empty_sum = empty1 + empty2;
    std::cout << "Empty matrix addition: " << (empty_sum.data == nullptr ? "Empty (correct)" : "Error") << "\n";
}

// Group 9: Performance Benchmarks
void test_performance_benchmarks()
{
    std::cout << "\n[Group 9: Performance Benchmarks Tests]\n";
    
    // Ensure current task is added to watchdog before starting performance tests
    #if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    ensure_task_wdt_added();
    #endif

    // Test 9.1: Matrix Addition Performance (reduced size to prevent timeout)
    std::cout << "\n[Test 9.1] Matrix Addition Performance\n";
    tiny::Mat A(50, 50);  // Reduced from 100x100 to 50x50
    tiny::Mat B(50, 50);
    for (int i = 0; i < 50; ++i)
    {
        for (int j = 0; j < 50; ++j)
        {
            A(i, j) = static_cast<float>(i * 50 + j);
            B(i, j) = static_cast<float>(i * 50 + j + 1);
        }
    }
    TIME_REPEATED_OPERATION(tiny::Mat C = A + B;, PERFORMANCE_TEST_ITERATIONS, "50x50 Matrix Addition");

    // Test 9.2: Matrix Multiplication Performance (reduced size)
    std::cout << "\n[Test 9.2] Matrix Multiplication Performance\n";
    tiny::Mat D(30, 30);  // Reduced from 50x50 to 30x30
    tiny::Mat E(30, 30);
    for (int i = 0; i < 30; ++i)
    {
        for (int j = 0; j < 30; ++j)
        {
            D(i, j) = static_cast<float>(i * 30 + j);
            E(i, j) = static_cast<float>(i * 30 + j + 1);
        }
    }
    TIME_REPEATED_OPERATION(tiny::Mat F = D * E;, PERFORMANCE_TEST_ITERATIONS, "30x30 Matrix Multiplication");

    // Test 9.3: Matrix Transpose Performance (reduced size)
    std::cout << "\n[Test 9.3] Matrix Transpose Performance\n";
    tiny::Mat G(50, 30);  // Reduced from 100x50 to 50x30
    for (int i = 0; i < 50; ++i)
        for (int j = 0; j < 30; ++j)
            G(i, j) = static_cast<float>(i * 30 + j);
    TIME_REPEATED_OPERATION(tiny::Mat H = G.transpose();, PERFORMANCE_TEST_ITERATIONS, "50x30 Matrix Transpose");

    // Test 9.4: Determinant Performance (reduced size significantly due to recursive nature)
    // Note: Determinant calculation uses recursive Laplace expansion which is O(n!) complexity
    // For performance testing, we use smaller matrices (5x5) to avoid timeout
    std::cout << "\n[Test 9.4] Determinant Calculation Performance\n";
    tiny::Mat I(5, 5);  // Reduced to 5x5 to prevent timeout (8x8 was too slow)
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            I(i, j) = static_cast<float>(i * 5 + j + 1);
    
    // Skip warmup for determinant (too slow), test directly
    std::cout << "[Performance] Computing determinant (no warmup due to recursive nature)...\n";
    feed_watchdog();  // Feed watchdog before starting
    
    TinyTimeMark_t det_t0 = tiny_get_running_time();
    for (int i = 0; i < PERFORMANCE_TEST_ITERATIONS_HEAVY; ++i)
    {
        feed_watchdog();  // Feed watchdog before each operation
        float det = I.determinant();
        (void)det;  // Suppress unused variable warning
        feed_watchdog();  // Feed watchdog after each operation
    }
    TinyTimeMark_t det_t1 = tiny_get_running_time();
    double det_dt_total_us = (double)(det_t1 - det_t0);
    double det_dt_avg_us = det_dt_total_us / PERFORMANCE_TEST_ITERATIONS_HEAVY;
    std::cout << "[Performance] 5x5 Determinant (" << PERFORMANCE_TEST_ITERATIONS_HEAVY << " iterations): "
              << std::fixed << std::setprecision(2) << det_dt_total_us << " us total, "
              << det_dt_avg_us << " us avg\n";
    std::cout << "[Note] Determinant uses recursive Laplace expansion (O(n!)), suitable only for small matrices.\n";

    // Test 9.5: Matrix Copy Performance (with padding, reduced size)
    std::cout << "\n[Test 9.5] Matrix Copy with Padding Performance\n";
    float data[80] = {0};  // Reduced from 150 to 80
    for (int i = 0; i < 80; ++i) data[i] = static_cast<float>(i);
    tiny::Mat J(data, 8, 8, 10);  // Reduced from 10x10 stride 15 to 8x8 stride 10
    TIME_REPEATED_OPERATION(tiny::Mat K = J.copy_roi(0, 0, 8, 8);, PERFORMANCE_TEST_ITERATIONS, "8x8 Copy ROI (with padding)");

    // Test 9.6: Element Access Performance (reduced size)
    std::cout << "\n[Test 9.6] Element Access Performance\n";
    tiny::Mat L(50, 50);  // Reduced from 100x100 to 50x50
    for (int i = 0; i < 50; ++i)
        for (int j = 0; j < 50; ++j)
            L(i, j) = static_cast<float>(i * 50 + j);
    
    // Test 6: Element Access Performance (custom implementation for multi-line operation)
    float sum = 0.0f;
    std::cout << "[Performance] Computing element access (warmup)...\n";
    feed_watchdog();  // Feed watchdog before starting
    for (int w = 0; w < PERFORMANCE_TEST_WARMUP; ++w)
    {
        feed_watchdog();  // Feed watchdog before each warmup
        sum = 0.0f;
        for (int i = 0; i < 50; ++i)
            for (int j = 0; j < 50; ++j)
                sum += L(i, j);
        feed_watchdog();  // Feed watchdog after each warmup
    }
    
    TinyTimeMark_t elem_t0 = tiny_get_running_time();
    for (int i = 0; i < PERFORMANCE_TEST_ITERATIONS; ++i)
    {
        if (i % 20 == 0) feed_watchdog();  // Feed watchdog every 20 iterations (element access is fast)
        sum = 0.0f;
        for (int row = 0; row < 50; ++row)
            for (int col = 0; col < 50; ++col)
                sum += L(row, col);
    }
    feed_watchdog();  // Final feed after loop
    TinyTimeMark_t elem_t1 = tiny_get_running_time();
    double elem_dt_total_us = (double)(elem_t1 - elem_t0);
    double dt_avg_us = elem_dt_total_us / PERFORMANCE_TEST_ITERATIONS;
    std::cout << "[Performance] 50x50 Element Access (all elements) (" << PERFORMANCE_TEST_ITERATIONS << " iterations): "
              << std::fixed << std::setprecision(2) << elem_dt_total_us << " us total, "
              << dt_avg_us << " us avg\n";
}

// Group 10: Memory Layout Tests (Padding and Stride)
void test_memory_layout()
{
    std::cout << "\n[Group 10: Memory Layout Tests (Padding and Stride)]\n";

    // Test 10.1: Contiguous memory (pad=0, step=1)
    std::cout << "\n[Test 10.1] Contiguous Memory (no padding)\n";
    tiny::Mat mat1(3, 4);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            mat1(i, j) = static_cast<float>(i * 4 + j);
    std::cout << "Matrix 3x4 (stride=4, pad=0):\n";
    mat1.print_info();
    mat1.print_matrix(true);

    // Test 10.2: Padded memory (stride > col)
    std::cout << "\n[Test 10.2] Padded Memory (stride > col)\n";
    float data[15] = {0, 1, 2, 3, 0, 4, 5, 6, 7, 0, 8, 9, 10, 11, 0};
    tiny::Mat mat2(data, 3, 4, 5);
    std::cout << "Matrix 3x4 (stride=5, pad=1):\n";
    mat2.print_info();
    mat2.print_matrix(true);

    // Test 10.3: Operations with padded matrices
    std::cout << "\n[Test 10.3] Addition with Padded Matrices\n";
    float data1[15] = {1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 9, 10, 11, 12, 0};
    float data2[15] = {10, 20, 30, 40, 0, 50, 60, 70, 80, 0, 90, 100, 110, 120, 0};
    tiny::Mat mat3(data1, 3, 4, 5);
    tiny::Mat mat4(data2, 3, 4, 5);
    tiny::Mat mat5 = mat3 + mat4;
    std::cout << "Result of padded matrix addition:\n";
    mat5.print_info();
    mat5.print_matrix(true);

    // Test 10.4: ROI operations with padded matrices
    std::cout << "\n[Test 10.4] ROI Operations with Padded Matrices\n";
    tiny::Mat roi = mat2.view_roi(1, 1, 2, 2);
    std::cout << "ROI (1,1,2,2) from padded matrix:\n";
    roi.print_info();
    roi.print_matrix(true);

    // Test 10.5: Copy operations preserve stride
    std::cout << "\n[Test 10.5] Copy Operations Preserve Stride\n";
    tiny::Mat copied = mat2.copy_roi(0, 0, 3, 4);
    std::cout << "Copied matrix (should have stride=4, no padding):\n";
    copied.print_info();
    copied.print_matrix(true);
}

// Group 11: Eigenvalue and Eigenvector Decomposition
void test_eigenvalue_decomposition()
{
    std::cout << "\n[Group 11: Eigenvalue and Eigenvector Decomposition Tests]\n";

    // Test 11.1: is_symmetric() - Basic functionality
    std::cout << "\n[Test 11.1] is_symmetric() - Basic Functionality\n";
    
    // Test 11.1.1: Symmetric matrix
    {
        std::cout << "[Test 11.1.1] Symmetric 3x3 Matrix\n";
        tiny::Mat sym_mat1(3, 3);
        sym_mat1(0, 0) = 4.0f; sym_mat1(0, 1) = 1.0f; sym_mat1(0, 2) = 2.0f;
        sym_mat1(1, 0) = 1.0f; sym_mat1(1, 1) = 3.0f; sym_mat1(1, 2) = 0.0f;
        sym_mat1(2, 0) = 2.0f; sym_mat1(2, 1) = 0.0f; sym_mat1(2, 2) = 5.0f;
        bool is_sym1 = sym_mat1.is_symmetric(1e-5f);
        std::cout << "Matrix:\n";
        sym_mat1.print_matrix(true);
        std::cout << "Is symmetric: " << (is_sym1 ? "True" : "False") << " (Expected: True)\n";
    }

    // Test 11.1.2: Non-symmetric matrix (keep for later tests)
    tiny::Mat non_sym_mat(3, 3);
    {
        std::cout << "\n[Test 11.1.2] Non-Symmetric 3x3 Matrix\n";
        non_sym_mat(0, 0) = 1.0f; non_sym_mat(0, 1) = 2.0f; non_sym_mat(0, 2) = 3.0f;
        non_sym_mat(1, 0) = 4.0f; non_sym_mat(1, 1) = 5.0f; non_sym_mat(1, 2) = 6.0f;
        non_sym_mat(2, 0) = 7.0f; non_sym_mat(2, 1) = 8.0f; non_sym_mat(2, 2) = 9.0f;
        bool is_sym2 = non_sym_mat.is_symmetric(1e-5f);
        std::cout << "Matrix:\n";
        non_sym_mat.print_matrix(true);
        std::cout << "Is symmetric: " << (is_sym2 ? "True" : "False") << " (Expected: False)\n";
    }

    // Test 11.1.3: Non-square matrix
    {
        std::cout << "\n[Test 11.1.3] Non-Square Matrix (2x3)\n";
        tiny::Mat rect_mat(2, 3);
        bool is_sym3 = rect_mat.is_symmetric(1e-5f);
        std::cout << "Is symmetric: " << (is_sym3 ? "True" : "False") << " (Expected: False)\n";
    }

    // Test 11.1.4: Symmetric matrix with small numerical errors
    {
        std::cout << "\n[Test 11.1.4] Symmetric Matrix with Small Numerical Errors\n";
        tiny::Mat sym_mat2(2, 2);
        // Use 1e-5 error which is within float precision (float has ~7 significant digits)
        // For 2.0, we can represent 2.00001 accurately
        float error_value = 1e-5f;
        sym_mat2(0, 0) = 1.0f; 
        sym_mat2(0, 1) = 2.0f + error_value;
        sym_mat2(1, 0) = 2.0f; 
        sym_mat2(1, 1) = 3.0f;
        std::cout << "Matrix with error " << error_value << ":\n";
        sym_mat2.print_matrix(true);
        float diff = fabsf(sym_mat2(0, 1) - sym_mat2(1, 0));
        std::cout << "Difference: |A(0,1) - A(1,0)| = ";
        // Use scientific notation for small values
        if (diff < 1e-3f)
        {
            std::cout << std::scientific << std::setprecision(6) << diff << std::fixed;
        }
        else
        {
            std::cout << std::setprecision(6) << diff;
        }
        std::cout << " (Expected: " << error_value << ")\n";
        
        // Verify the difference is actually stored
        float stored_value = sym_mat2(0, 1);
        float expected_stored = 2.0f + error_value;
        std::cout << "A(0,1) stored value: " << std::setprecision(8) << stored_value 
                  << " (Expected: " << expected_stored << ")\n";
        
        bool is_sym4 = sym_mat2.is_symmetric(1e-4f); // tolerance > error, should pass
        std::cout << "Is symmetric (tolerance=1e-4): " << (is_sym4 ? "True" : "False") 
                  << " (Expected: True, tolerance > error) ";
        std::cout << (is_sym4 ? "[PASS]" : "[FAIL]") << "\n";
        
        bool is_sym5 = sym_mat2.is_symmetric(1e-6f); // tolerance < error, should fail
        std::cout << "Is symmetric (tolerance=1e-6): " << (is_sym5 ? "True" : "False") 
                  << " (Expected: False, tolerance < error) ";
        bool correct_result = !is_sym5; // Should be False (not symmetric)
        std::cout << (correct_result ? "[PASS]" : "[FAIL]") << "\n";
        
        // Additional check: verify the difference is close to expected
        float diff_error = fabsf(diff - error_value);
        std::cout << "Difference accuracy: |actual_diff - expected_diff| = " 
                  << std::scientific << std::setprecision(2) << diff_error << std::fixed;
        bool diff_accurate = (diff_error < error_value * 0.1f); // Within 10% of error value
        std::cout << " " << (diff_accurate ? "[PASS - difference stored correctly]" : "[FAIL - float precision issue]") << "\n";
    }

    // Test 11.2: power_iteration() - Dominant eigenvalue
    std::cout << "\n[Test 11.2] power_iteration() - Dominant Eigenvalue\n";
    
    // Test 11.2.1: Simple 2x2 symmetric matrix (known eigenvalues)
    tiny::Mat mat2x2(2, 2);
    {
        std::cout << "\n[Test 11.2.1] Simple 2x2 Matrix\n";
        mat2x2(0, 0) = 2.0f; mat2x2(0, 1) = 1.0f;
        mat2x2(1, 0) = 1.0f; mat2x2(1, 1) = 2.0f;
        std::cout << "Matrix:\n";
        mat2x2.print_matrix(true);
        
        // Expected values: eigenvalues are 3 and 1 (for matrix [2,1; 1,2])
        // Characteristic equation: det([2-λ, 1; 1, 2-λ]) = (2-λ)² - 1 = λ² - 4λ + 3 = 0
        // Solutions: λ = (4 ± √(16-12))/2 = (4 ± 2)/2 = 3 or 1
        std::cout << "\n[Expected Results]\n";
        std::cout << "  Expected eigenvalues: 3.0 (largest), 1.0 (smallest)\n";
        std::cout << "  Expected dominant eigenvector (for λ=3): approximately [0.707, 0.707] or [-0.707, -0.707] (normalized)\n";
        std::cout << "  Expected dominant eigenvector (for λ=1): approximately [0.707, -0.707] or [-0.707, 0.707] (normalized)\n";
        
        tiny::Mat::EigenPair result_power = mat2x2.power_iteration(1000, 1e-6f);
        std::cout << "\n[Actual Results]\n";
        std::cout << "  Dominant eigenvalue: " << result_power.eigenvalue 
                  << " (Expected: 3.0, largest eigenvalue)\n";
        std::cout << "  Iterations: " << result_power.iterations << "\n";
        std::cout << "  Status: " << (result_power.status == TINY_OK ? "OK" : "Error") << "\n";
        std::cout << "  Dominant eigenvector:\n";
        result_power.eigenvector.print_matrix(true);
        
        // Check if result matches expected
        float error = fabsf(result_power.eigenvalue - 3.0f);
        std::cout << "  Error from expected (3.0): " << error << (error < 0.01f ? " [PASS]" : " [FAIL]") << "\n";
    }

    // Test 11.2.2: 3x3 matrix (SHM-like stiffness matrix) - keep for later tests
    tiny::Mat stiffness(3, 3);
    {
        std::cout << "\n[Test 11.2.2] 3x3 Stiffness Matrix (SHM Application)\n";
        stiffness(0, 0) = 2.0f; stiffness(0, 1) = -1.0f; stiffness(0, 2) = 0.0f;
        stiffness(1, 0) = -1.0f; stiffness(1, 1) = 2.0f; stiffness(1, 2) = -1.0f;
        stiffness(2, 0) = 0.0f; stiffness(2, 1) = -1.0f; stiffness(2, 2) = 2.0f;
        std::cout << "Stiffness Matrix:\n";
        stiffness.print_matrix(true);
        
        // Expected values for 3x3 tridiagonal symmetric matrix [2,-1,0; -1,2,-1; 0,-1,2]
        // This is a standard tridiagonal matrix with known eigenvalues
        // Approximate eigenvalues: λ₁ ≈ 3.414, λ₂ ≈ 2.000, λ₃ ≈ 0.586
        std::cout << "\n[Expected Results]\n";
        std::cout << "  Expected eigenvalues (approximate): 3.414 (largest), 2.000, 0.586 (smallest)\n";
        std::cout << "  Expected primary frequency: sqrt(3.414) ≈ 1.848 rad/s\n";
        
        tiny::Mat::EigenPair result_stiff = stiffness.power_iteration(500, 1e-6f);
        std::cout << "\n[Actual Results]\n";
        std::cout << "  Dominant eigenvalue (primary frequency squared): " << result_stiff.eigenvalue << "\n";
        std::cout << "  Primary frequency: " << sqrtf(result_stiff.eigenvalue) << " rad/s (Expected: ~1.848 rad/s)\n";
        std::cout << "  Iterations: " << result_stiff.iterations << "\n";
        std::cout << "  Status: " << (result_stiff.status == TINY_OK ? "OK" : "Error") << "\n";
        
        float expected_eigen = 3.414f;
        float error = fabsf(result_stiff.eigenvalue - expected_eigen);
        std::cout << "  Error from expected (" << expected_eigen << "): " << error << (error < 0.1f ? " [PASS]" : " [FAIL]") << "\n";
    }

    // Test 11.2.3: Non-square matrix (should fail)
    {
        std::cout << "\n[Test 11.2.3] Non-Square Matrix (Expect Error)\n";
        tiny::Mat non_square(2, 3);
        tiny::Mat::EigenPair result_error = non_square.power_iteration(100, 1e-6f);
        std::cout << "Status: " << (result_error.status == TINY_OK ? "OK" : "Error (Expected)") << "\n";
    }

    // Test 11.3: eigendecompose_jacobi() - Symmetric matrix decomposition
    std::cout << "\n[Test 11.3] eigendecompose_jacobi() - Symmetric Matrix Decomposition\n";
    
    // Test 11.3.1: Simple 2x2 symmetric matrix
    {
        std::cout << "\n[Test 11.3.1] 2x2 Symmetric Matrix - Complete Decomposition\n";
        std::cout << "[Expected Results]\n";
        std::cout << "  Expected eigenvalues: 3.0, 1.0 (in any order)\n";
        std::cout << "  Expected eigenvectors (for λ=3): [0.707, 0.707] or [-0.707, -0.707] (normalized)\n";
        std::cout << "  Expected eigenvectors (for λ=1): [0.707, -0.707] or [-0.707, 0.707] (normalized)\n";
        
        tiny::Mat::EigenDecomposition result_jacobi1 = mat2x2.eigendecompose_jacobi(1e-6f, 100);
        std::cout << "\n[Actual Results]\n";
        std::cout << "Eigenvalues:\n";
        result_jacobi1.eigenvalues.print_matrix(true);
        std::cout << "Eigenvectors (each column is an eigenvector):\n";
        result_jacobi1.eigenvectors.print_matrix(true);
        std::cout << "Iterations: " << result_jacobi1.iterations << "\n";
        std::cout << "Status: " << (result_jacobi1.status == TINY_OK ? "OK" : "Error") << "\n";
        
        // Check eigenvalues
        float ev1 = result_jacobi1.eigenvalues(0, 0);
        float ev2 = result_jacobi1.eigenvalues(1, 0);
        bool ev_check = ((fabsf(ev1 - 3.0f) < 0.01f && fabsf(ev2 - 1.0f) < 0.01f) ||
                         (fabsf(ev1 - 1.0f) < 0.01f && fabsf(ev2 - 3.0f) < 0.01f));
        std::cout << "Eigenvalue check (should be 3.0 and 1.0): " << (ev_check ? "[PASS]" : "[FAIL]") << "\n";
        
        // Verify: A * v = lambda * v
        std::cout << "\n[Verification] Check A * v = lambda * v for first eigenvector:\n";
        tiny::Mat Av = mat2x2 * result_jacobi1.eigenvectors.block(0, 0, 2, 1);
        tiny::Mat lambda_v = result_jacobi1.eigenvalues(0, 0) * result_jacobi1.eigenvectors.block(0, 0, 2, 1);
        std::cout << "A * v:\n";
        Av.print_matrix(true);
        std::cout << "lambda * v:\n";
        lambda_v.print_matrix(true);
        bool verify1 = matrices_approximately_equal(Av, lambda_v, 1e-4f);
        std::cout << "Verification (A*v = λ*v): " << (verify1 ? "[PASS]" : "[FAIL]") << "\n";
    }

    // Test 11.3.2: 3x3 symmetric matrix (SHM stiffness matrix)
    {
        std::cout << "\n[Test 11.3.2] 3x3 Stiffness Matrix (SHM Application)\n";
        std::cout << "[Expected Results]\n";
        std::cout << "  Expected eigenvalues (approximate): 3.414, 2.000, 0.586\n";
        std::cout << "  Expected natural frequencies: 1.848, 1.414, 0.765 rad/s\n";
        std::cout << "  Note: Eigenvalues may appear in any order\n";
        
        tiny::Mat::EigenDecomposition result_jacobi2 = stiffness.eigendecompose_jacobi(1e-5f, 100);
        std::cout << "\n[Actual Results]\n";
        std::cout << "Eigenvalues (natural frequencies squared):\n";
        result_jacobi2.eigenvalues.print_matrix(true);
        std::cout << "Natural frequencies (rad/s):\n";
        float expected_freqs[3] = {1.848f, 1.414f, 0.765f};
        for (int i = 0; i < result_jacobi2.eigenvalues.row; ++i)
        {
            float freq = sqrtf(result_jacobi2.eigenvalues(i, 0));
            std::cout << "  Mode " << i << ": " << freq << " rad/s";
            // Check if frequency matches any expected value
            bool matched = false;
            for (int j = 0; j < 3; ++j)
            {
                if (fabsf(freq - expected_freqs[j]) < 0.1f)
                {
                    std::cout << " (Expected: ~" << expected_freqs[j] << " rad/s) [PASS]";
                    matched = true;
                    break;
                }
            }
            if (!matched) std::cout << " [CHECK]";
            std::cout << "\n";
        }
        std::cout << "Eigenvectors (mode shapes):\n";
        result_jacobi2.eigenvectors.print_matrix(true);
        std::cout << "Iterations: " << result_jacobi2.iterations << "\n";
        std::cout << "Status: " << (result_jacobi2.status == TINY_OK ? "OK" : "Error") << "\n";
    }

    // Test 11.3.3: Diagonal matrix (trivial case)
    {
        std::cout << "\n[Test 11.3.3] Diagonal Matrix (Eigenvalues on diagonal)\n";
        tiny::Mat diag_mat(3, 3);
        diag_mat(0, 0) = 5.0f; diag_mat(0, 1) = 0.0f; diag_mat(0, 2) = 0.0f;
        diag_mat(1, 0) = 0.0f; diag_mat(1, 1) = 3.0f; diag_mat(1, 2) = 0.0f;
        diag_mat(2, 0) = 0.0f; diag_mat(2, 1) = 0.0f; diag_mat(2, 2) = 1.0f;
        std::cout << "Matrix:\n";
        diag_mat.print_matrix(true);
        std::cout << "\n[Expected Results]\n";
        std::cout << "  Expected eigenvalues: 5.0, 3.0, 1.0 (diagonal elements, may be in any order)\n";
        std::cout << "  Expected eigenvectors: standard basis vectors [1,0,0], [0,1,0], [0,0,1] (or their negatives)\n";
        std::cout << "  Expected iterations: 1 (diagonal matrix should converge immediately)\n";
        
        tiny::Mat::EigenDecomposition result_diag = diag_mat.eigendecompose_jacobi(1e-6f, 10);
        std::cout << "\n[Actual Results]\n";
        std::cout << "Eigenvalues:\n";
        result_diag.eigenvalues.print_matrix(true);
        std::cout << "Eigenvectors:\n";
        result_diag.eigenvectors.print_matrix(true);
        std::cout << "Iterations: " << result_diag.iterations << " (Expected: 1)\n";
        
        // Check eigenvalues
        float ev1 = result_diag.eigenvalues(0, 0);
        float ev2 = result_diag.eigenvalues(1, 0);
        float ev3 = result_diag.eigenvalues(2, 0);
        bool ev_check = ((fabsf(ev1 - 5.0f) < 0.01f || fabsf(ev1 - 3.0f) < 0.01f || fabsf(ev1 - 1.0f) < 0.01f) &&
                         (fabsf(ev2 - 5.0f) < 0.01f || fabsf(ev2 - 3.0f) < 0.01f || fabsf(ev2 - 1.0f) < 0.01f) &&
                         (fabsf(ev3 - 5.0f) < 0.01f || fabsf(ev3 - 3.0f) < 0.01f || fabsf(ev3 - 1.0f) < 0.01f));
        std::cout << "Eigenvalue check (should be 5.0, 3.0, 1.0): " << (ev_check ? "[PASS]" : "[FAIL]") << "\n";
    }

    // Test 11.4: eigendecompose_qr() - General matrix decomposition
    std::cout << "\n[Test 11.4] eigendecompose_qr() - General Matrix Decomposition\n";
    
    // Test 11.4.1: General 2x2 matrix
    {
        std::cout << "\n[Test 11.4.1] General 2x2 Matrix\n";
        tiny::Mat gen_mat(2, 2);
        gen_mat(0, 0) = 1.0f; gen_mat(0, 1) = 2.0f;
        gen_mat(1, 0) = 3.0f; gen_mat(1, 1) = 4.0f;
        std::cout << "Matrix:\n";
        gen_mat.print_matrix(true);
        
        // Expected values for matrix [1,2; 3,4]
        // Characteristic equation: det([1-λ, 2; 3, 4-λ]) = (1-λ)(4-λ) - 6 = λ² - 5λ - 2 = 0
        // Solutions: λ = (5 ± √(25+8))/2 = (5 ± √33)/2 ≈ 5.372, -0.372
        std::cout << "\n[Expected Results]\n";
        std::cout << "  Expected eigenvalues: (5+√33)/2 ≈ 5.372, (5-√33)/2 ≈ -0.372\n";
        std::cout << "  Note: This is a non-symmetric matrix, eigenvalues are real but may have complex eigenvectors\n";
        
        tiny::Mat::EigenDecomposition result_qr1 = gen_mat.eigendecompose_qr(100, 1e-5f);
        std::cout << "\n[Actual Results]\n";
        std::cout << "Eigenvalues:\n";
        result_qr1.eigenvalues.print_matrix(true);
        std::cout << "Eigenvectors:\n";
        result_qr1.eigenvectors.print_matrix(true);
        std::cout << "Iterations: " << result_qr1.iterations << "\n";
        std::cout << "Status: " << (result_qr1.status == TINY_OK ? "OK" : "Error") << "\n";
        
        // Check eigenvalues with detailed error reporting
        float ev1 = result_qr1.eigenvalues(0, 0);
        float ev2 = result_qr1.eigenvalues(1, 0);
        float expected_ev1 = 5.372f;
        float expected_ev2 = -0.372f;
        
        // Match eigenvalues to expected values
        float error1a = fabsf(ev1 - expected_ev1);
        float error1b = fabsf(ev1 - expected_ev2);
        float error2a = fabsf(ev2 - expected_ev1);
        float error2b = fabsf(ev2 - expected_ev2);
        
        bool match1 = (error1a < error1b); // ev1 matches expected_ev1 better
        float matched_ev1 = match1 ? expected_ev1 : expected_ev2;
        float matched_ev2 = match1 ? expected_ev2 : expected_ev1;
        float actual_error1 = match1 ? error1a : error1b;
        float actual_error2 = match1 ? error2b : error2a;
        float rel_error1 = actual_error1 / fabsf(matched_ev1);
        float rel_error2 = actual_error2 / fabsf(matched_ev2);
        
        std::cout << "Eigenvalue 1: " << ev1 << " (Expected: " << matched_ev1 << ", Error: " << actual_error1 
                  << ", Rel Error: " << (rel_error1 * 100.0f) << "%) ";
        bool pass1 = (rel_error1 < 0.05f); // 5% relative error tolerance
        std::cout << (pass1 ? "[PASS]" : "[FAIL - error too large]") << "\n";
        
        std::cout << "Eigenvalue 2: " << ev2 << " (Expected: " << matched_ev2 << ", Error: " << actual_error2 
                  << ", Rel Error: " << (rel_error2 * 100.0f) << "%) ";
        bool pass2 = (rel_error2 < 0.05f); // 5% relative error tolerance
        std::cout << (pass2 ? "[PASS]" : "[FAIL - error too large]") << "\n";
        
        bool ev_check = pass1 && pass2;
        std::cout << "Overall eigenvalue check: " << (ev_check ? "[PASS]" : "[FAIL]") << "\n";
    }

    // Test 11.4.2: Non-symmetric 3x3 matrix
    {
        std::cout << "\n[Test 11.4.2] Non-Symmetric 3x3 Matrix\n";
        std::cout << "Matrix [1,2,3; 4,5,6; 7,8,9]:\n";
        non_sym_mat.print_matrix(true);
        
        // Expected values for matrix [1,2,3; 4,5,6; 7,8,9]
        // Characteristic equation: λ³ - 15λ² - 18λ = 0
        // Solutions: λ(λ² - 15λ - 18) = 0
        // λ₁ = 0, λ₂,₃ = (15 ± √(225+72))/2 = (15 ± √297)/2 ≈ 16.12, -1.12
        // However, QR algorithm may have numerical errors, especially for non-symmetric matrices
        std::cout << "\n[Expected Results]\n";
        std::cout << "  Expected eigenvalues (theoretical): 16.12, -1.12, 0.00\n";
        std::cout << "  Note: This matrix is rank-deficient (determinant = 0), so one eigenvalue is 0\n";
        std::cout << "  Note: QR algorithm may have numerical errors, especially for non-symmetric matrices\n";
        std::cout << "  Acceptable range: largest eigenvalue ~15-18, smallest eigenvalue near 0\n";
        
        tiny::Mat::EigenDecomposition result_qr2 = non_sym_mat.eigendecompose_qr(100, 1e-5f);
        std::cout << "\n[Actual Results]\n";
        std::cout << "Eigenvalues:\n";
        result_qr2.eigenvalues.print_matrix(true);
        std::cout << "Eigenvectors:\n";
        result_qr2.eigenvectors.print_matrix(true);
        std::cout << "Iterations: " << result_qr2.iterations << "\n";
        std::cout << "Status: " << (result_qr2.status == TINY_OK ? "OK" : "Error") << "\n";
        
        // Check results with detailed error reporting
        float expected_evs[3] = {16.12f, -1.12f, 0.0f};
        bool matched[3] = {false, false, false};
        float errors[3] = {0.0f, 0.0f, 0.0f};
        float rel_errors[3] = {0.0f, 0.0f, 0.0f};
        
        // Match each computed eigenvalue to the closest expected value
        for (int i = 0; i < result_qr2.eigenvalues.row; ++i)
        {
            float ev = result_qr2.eigenvalues(i, 0);
            float min_error = 1e10f;
            int best_match = -1;
            
            // Find closest expected eigenvalue
            for (int j = 0; j < 3; ++j)
            {
                if (!matched[j])
                {
                    float error = fabsf(ev - expected_evs[j]);
                    if (error < min_error)
                    {
                        min_error = error;
                        best_match = j;
                    }
                }
            }
            
            if (best_match >= 0)
            {
                matched[best_match] = true;
                errors[best_match] = min_error;
                float expected = expected_evs[best_match];
                rel_errors[best_match] = (fabsf(expected) > 1e-6f) ? (min_error / fabsf(expected)) : min_error;
                
                std::cout << "Eigenvalue " << i << ": " << ev << " (Expected: " << expected 
                          << ", Error: " << min_error << ", Rel Error: " << (rel_errors[best_match] * 100.0f) << "%) ";
                
                // For zero eigenvalue, use absolute tolerance; for others, use relative tolerance
                bool pass = (fabsf(expected) < 0.1f) ? (min_error < 0.1f) : (rel_errors[best_match] < 0.15f); // 15% tolerance for QR
                std::cout << (pass ? "[PASS]" : "[FAIL - error too large]") << "\n";
            }
        }
        
        // Overall check
        bool overall_pass = true;
        for (int i = 0; i < 3; ++i)
        {
            bool pass = (fabsf(expected_evs[i]) < 0.1f) ? (errors[i] < 0.1f) : (rel_errors[i] < 0.15f);
            if (!pass) overall_pass = false;
        }
        std::cout << "Overall eigenvalue check: " << (overall_pass ? "[PASS]" : "[FAIL - some eigenvalues have large errors]") << "\n";
    }

    // Test 11.5: eigendecompose() - Automatic method selection
    std::cout << "\n[Test 11.5] eigendecompose() - Automatic Method Selection\n";
    
    // Test 11.5.1: Symmetric matrix (should use Jacobi)
    {
        std::cout << "\n[Test 11.5.1] Symmetric Matrix (Auto-select: Jacobi)\n";
        tiny::Mat sym_mat1(3, 3);
        sym_mat1(0, 0) = 4.0f; sym_mat1(0, 1) = 1.0f; sym_mat1(0, 2) = 2.0f;
        sym_mat1(1, 0) = 1.0f; sym_mat1(1, 1) = 3.0f; sym_mat1(1, 2) = 0.0f;
        sym_mat1(2, 0) = 2.0f; sym_mat1(2, 1) = 0.0f; sym_mat1(2, 2) = 5.0f;
        std::cout << "Matrix:\n";
        sym_mat1.print_matrix(true);
        std::cout << "\n[Expected Results]\n";
        std::cout << "  Method: Should automatically use Jacobi (symmetric matrix detected)\n";
        std::cout << "  Expected eigenvalues (approximate): 6.67, 3.48, 1.85\n";
        std::cout << "  Note: Eigenvalues may appear in any order\n";
        
        tiny::Mat::EigenDecomposition result_auto1 = sym_mat1.eigendecompose(1e-5f);
        std::cout << "\n[Actual Results]\n";
        std::cout << "Eigenvalues:\n";
        result_auto1.eigenvalues.print_matrix(true);
        std::cout << "Iterations: " << result_auto1.iterations << "\n";
        std::cout << "Status: " << (result_auto1.status == TINY_OK ? "OK" : "Error") << "\n";
        std::cout << "Method used: Jacobi (auto-selected for symmetric matrix)\n";
    }

    // Test 11.5.2: Non-symmetric matrix (should use QR)
    {
        std::cout << "\n[Test 11.5.2] Non-Symmetric Matrix (Auto-select: QR)\n";
        std::cout << "[Expected Results]\n";
        std::cout << "  Method: Should automatically use QR (non-symmetric matrix detected)\n";
        std::cout << "  Expected eigenvalues (theoretical): 16.12, -1.12, 0.00\n";
        std::cout << "  Note: One eigenvalue should be near 0 (rank-deficient matrix)\n";
        std::cout << "  Note: QR algorithm may have numerical errors for non-symmetric matrices\n";
        std::cout << "  Acceptable: largest ~15-18, smallest near 0, one near -1 to -3\n";
        
        tiny::Mat::EigenDecomposition result_auto2 = non_sym_mat.eigendecompose(1e-5f);
        std::cout << "\n[Actual Results]\n";
        std::cout << "Eigenvalues:\n";
        result_auto2.eigenvalues.print_matrix(true);
        std::cout << "Iterations: " << result_auto2.iterations << "\n";
        std::cout << "Status: " << (result_auto2.status == TINY_OK ? "OK" : "Error") << "\n";
        std::cout << "Method used: QR (auto-selected for non-symmetric matrix)\n";
        
        // Check results with detailed error reporting (same as Test 4.2)
        float expected_evs[3] = {16.12f, -1.12f, 0.0f};
        bool matched[3] = {false, false, false};
        float errors[3] = {0.0f, 0.0f, 0.0f};
        float rel_errors[3] = {0.0f, 0.0f, 0.0f};
        
        for (int i = 0; i < result_auto2.eigenvalues.row; ++i)
        {
            float ev = result_auto2.eigenvalues(i, 0);
            float min_error = 1e10f;
            int best_match = -1;
            
            for (int j = 0; j < 3; ++j)
            {
                if (!matched[j])
                {
                    float error = fabsf(ev - expected_evs[j]);
                    if (error < min_error)
                    {
                        min_error = error;
                        best_match = j;
                    }
                }
            }
            
            if (best_match >= 0)
            {
                matched[best_match] = true;
                errors[best_match] = min_error;
                float expected = expected_evs[best_match];
                rel_errors[best_match] = (fabsf(expected) > 1e-6f) ? (min_error / fabsf(expected)) : min_error;
                
                std::cout << "Eigenvalue " << i << ": " << ev << " (Expected: " << expected 
                          << ", Error: " << min_error << ", Rel Error: " << (rel_errors[best_match] * 100.0f) << "%) ";
                
                bool pass = (fabsf(expected) < 0.1f) ? (min_error < 0.1f) : (rel_errors[best_match] < 0.15f);
                std::cout << (pass ? "[PASS]" : "[FAIL - error too large]") << "\n";
            }
        }
        
        bool overall_pass = true;
        for (int i = 0; i < 3; ++i)
        {
            bool pass = (fabsf(expected_evs[i]) < 0.1f) ? (errors[i] < 0.1f) : (rel_errors[i] < 0.15f);
            if (!pass) overall_pass = false;
        }
        std::cout << "Overall eigenvalue check: " << (overall_pass ? "[PASS]" : "[FAIL - some eigenvalues have large errors]") << "\n";
    }

    // Test 11.6: SHM Application Scenario - Structural Dynamics
    std::cout << "\n[Test 11.6] SHM Application - Structural Dynamics Analysis\n";
    
    // Create a simple 4-DOF structural system (mass-spring system)
    {
        std::cout << "\n[Test 11.6.1] 4-DOF Mass-Spring System\n";
        tiny::Mat K(4, 4);  // Stiffness matrix
        K(0, 0) = 2.0f; K(0, 1) = -1.0f; K(0, 2) = 0.0f; K(0, 3) = 0.0f;
        K(1, 0) = -1.0f; K(1, 1) = 2.0f; K(1, 2) = -1.0f; K(1, 3) = 0.0f;
        K(2, 0) = 0.0f; K(2, 1) = -1.0f; K(2, 2) = 2.0f; K(2, 3) = -1.0f;
        K(3, 0) = 0.0f; K(3, 1) = 0.0f; K(3, 2) = -1.0f; K(3, 3) = 1.0f;
        
        std::cout << "Stiffness Matrix K:\n";
        K.print_matrix(true);
        std::cout << "Is symmetric: " << (K.is_symmetric(1e-6f) ? "Yes" : "No") << "\n";
        
        // Quick frequency identification using power iteration
        std::cout << "\n[Quick Analysis] Primary frequency using power_iteration():\n";
        std::cout << "[Expected Results]\n";
        std::cout << "  Expected primary eigenvalue: ~3.53 (largest eigenvalue)\n";
        std::cout << "  Expected primary frequency: sqrt(3.53) ≈ 1.88 rad/s\n";
        
        tiny::Mat::EigenPair primary = K.power_iteration(500, 1e-6f);
        std::cout << "\n[Actual Results]\n";
        std::cout << "  Primary eigenvalue: " << primary.eigenvalue << " (Expected: ~3.53)\n";
        std::cout << "  Primary frequency: " << sqrtf(primary.eigenvalue) << " rad/s (Expected: ~1.88 rad/s)\n";
        std::cout << "  Iterations: " << primary.iterations << "\n";
        float error_primary = fabsf(primary.eigenvalue - 3.53f);
        std::cout << "  Error from expected: " << error_primary << (error_primary < 0.2f ? " [PASS]" : " [FAIL]") << "\n";
        
        // Complete modal analysis using Jacobi
        std::cout << "\n[Complete Analysis] Full modal analysis using eigendecompose_jacobi():\n";
        std::cout << "[Expected Results]\n";
        std::cout << "  Expected eigenvalues (approximate): 3.53, 2.35, 1.00, 0.12\n";
        std::cout << "  Expected natural frequencies: 1.88, 1.53, 1.00, 0.35 rad/s\n";
        std::cout << "  Note: These are approximate values for the 4-DOF system\n";
        
        tiny::Mat::EigenDecomposition modal = K.eigendecompose_jacobi(1e-5f, 100);
        std::cout << "\n[Actual Results]\n";
        std::cout << "All eigenvalues (natural frequencies squared):\n";
        modal.eigenvalues.print_matrix(true);
        std::cout << "Natural frequencies (rad/s):\n";
        float expected_freqs_4dof[4] = {1.88f, 1.53f, 1.00f, 0.35f};
        for (int i = 0; i < modal.eigenvalues.row; ++i)
        {
            float freq = sqrtf(modal.eigenvalues(i, 0));
            std::cout << "  Mode " << i << ": " << freq << " rad/s";
            // Check if frequency matches any expected value
            bool matched = false;
            for (int j = 0; j < 4; ++j)
            {
                if (fabsf(freq - expected_freqs_4dof[j]) < 0.15f)
                {
                    std::cout << " (Expected: ~" << expected_freqs_4dof[j] << " rad/s) [PASS]";
                    matched = true;
                    break;
                }
            }
            if (!matched) std::cout << " [CHECK]";
            std::cout << "\n";
        }
        std::cout << "Mode shapes (eigenvectors):\n";
        modal.eigenvectors.print_matrix(true);
        std::cout << "Total iterations: " << modal.iterations << "\n";
    }

    // Test 11.7: Edge Cases and Error Handling
    std::cout << "\n[Test 11.7] Edge Cases and Error Handling\n";
    
    // Test 11.7.1: 1x1 matrix
    {
        std::cout << "\n[Test 11.7.1] 1x1 Matrix\n";
        tiny::Mat mat1x1(1, 1);
        mat1x1(0, 0) = 5.0f;
        std::cout << "Matrix: [5.0]\n";
        std::cout << "[Expected Results]\n";
        std::cout << "  Expected eigenvalue: 5.0 (the matrix element itself)\n";
        std::cout << "  Expected eigenvector: [1.0] (normalized)\n";
        
        tiny::Mat::EigenDecomposition result_1x1 = mat1x1.eigendecompose(1e-6f);
        std::cout << "\n[Actual Results]\n";
        std::cout << "Eigenvalue: " << result_1x1.eigenvalues(0, 0) << " (Expected: 5.0)\n";
        std::cout << "Eigenvector:\n";
        result_1x1.eigenvectors.print_matrix(true);
        float error = fabsf(result_1x1.eigenvalues(0, 0) - 5.0f);
        std::cout << "Error from expected: " << error << (error < 0.01f ? " [PASS]" : " [FAIL]") << "\n";
    }
    
    // Test 11.7.2: Zero matrix
    {
        std::cout << "\n[Test 11.7.2] Zero Matrix\n";
        tiny::Mat zero_mat(3, 3);
        zero_mat.clear();
        tiny::Mat::EigenPair result_zero = zero_mat.power_iteration(100, 1e-6f);
        std::cout << "Status: " << (result_zero.status == TINY_OK ? "OK" : "Error (Expected)") << "\n";
    }
    
    // Test 11.7.3: Identity matrix
    {
        std::cout << "\n[Test 11.7.3] Identity Matrix\n";
        tiny::Mat I = tiny::Mat::eye(3);
        std::cout << "Matrix (3x3 Identity):\n";
        I.print_matrix(true);
        std::cout << "\n[Expected Results]\n";
        std::cout << "  Expected eigenvalues: 1.0, 1.0, 1.0 (all eigenvalues are 1)\n";
        std::cout << "  Expected eigenvectors: Any orthonormal basis (e.g., standard basis vectors)\n";
        std::cout << "  Expected iterations: 1 (should converge immediately)\n";
        
        tiny::Mat::EigenDecomposition result_I = I.eigendecompose_jacobi(1e-6f, 10);
        std::cout << "\n[Actual Results]\n";
        std::cout << "Eigenvalues (should all be 1.0):\n";
        result_I.eigenvalues.print_matrix(true);
        std::cout << "Eigenvectors:\n";
        result_I.eigenvectors.print_matrix(true);
        std::cout << "Iterations: " << result_I.iterations << " (Expected: 1)\n";
        
        // Check all eigenvalues are 1.0
        bool all_one = true;
        for (int i = 0; i < result_I.eigenvalues.row; ++i)
        {
            if (fabsf(result_I.eigenvalues(i, 0) - 1.0f) > 0.01f)
            {
                all_one = false;
                break;
            }
        }
        std::cout << "All eigenvalues = 1.0: " << (all_one ? "[PASS]" : "[FAIL]") << "\n";
    }

    // Test 11.8: Performance Test for SHM Applications
    std::cout << "\n[Test 11.8] Performance Test for SHM Applications\n";
    
    // Test 11.8.1: Power iteration performance (fast method)
    std::cout << "\n[Test 11.8.1] Power Iteration Performance (Real-time SHM)\n";
    TIME_OPERATION(
        tiny::Mat::EigenPair perf_result = stiffness.power_iteration(500, 1e-6f);
        (void)perf_result;
    , "Power Iteration (3x3 matrix)");
    
    // Test 11.8.2: Jacobi method performance
    std::cout << "\n[Test 11.8.2] Jacobi Method Performance\n";
    TIME_OPERATION(
        tiny::Mat::EigenDecomposition perf_jacobi = stiffness.eigendecompose_jacobi(1e-5f, 100);
        (void)perf_jacobi;
    , "Jacobi Decomposition (3x3 symmetric matrix)");
    
    // Test 11.8.3: QR method performance
    std::cout << "\n[Test 11.8.3] QR Method Performance\n";
    TIME_OPERATION(
        tiny::Mat::EigenDecomposition perf_qr = non_sym_mat.eigendecompose_qr(100, 1e-5f);
        (void)perf_qr;
    , "QR Decomposition (3x3 general matrix)");

    std::cout << "\n[Eigenvalue Decomposition Tests Complete]\n";
}

void tiny_matrix_test()
{
    std::cout << "============ [tiny_matrix_test start] ============\n";

    // Group 1: constructor & destructor
    test_constructor_destructor();

    // Group 2: element access
    test_element_access();

    // Group 3: ROI operations
    test_roi_operations();

    // Group 4: arithmetic operators
    test_assignment_operator();
    test_matrix_addition();
    test_constant_addition();
    test_matrix_subtraction();
    test_constant_subtraction();
    test_matrix_division();
    test_constant_division();
    test_matrix_exponentiation();

    // Group 5: Linear algebra tests
    test_matrix_transpose();
    test_matrix_cofactor();
    test_matrix_determinant();
    test_matrix_adjoint();
    test_matrix_normalize();
    test_matrix_norm();
    test_inverse_adjoint_adjoint();
    test_matrix_utilities();
    test_gaussian_eliminate();
    test_row_reduce_from_gaussian();
    test_inverse_gje();
    test_dotprod();
    test_solve();
    test_band_solve();
    test_roots();

    // Group 6: Stream operators
    test_stream_operators();

    // Group 7: Matrix operations
    test_matrix_operations();

    // Group 8: Boundary conditions and error handling
    test_boundary_conditions();

    // Group 9: Performance benchmarks
    test_performance_benchmarks();

    // Group 10: Memory layout tests
    test_memory_layout();

    // Group 11: Eigenvalue and Eigenvector Decomposition
    test_eigenvalue_decomposition();

    std::cout << "============ [tiny_matrix_test end] ============\n";
    
    // Remove current task from watchdog after all tests complete
    // This prevents watchdog timeout after app_main() returns
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    if (task_wdt_added)
    {
        esp_task_wdt_delete(NULL);  // Remove current task from watchdog
        task_wdt_added = false;
    }
#endif
}
```

## 结果输出

### Group 1: 构造与析构

```txt
============ [tiny_matrix_test start] ============

--- Test: Constructor & Destructor ---
[Test 1.1] Default Constructor
Matrix Info >>>
rows            1
cols            1
elements        1
paddings        0
stride          1
memory          1
data pointer    0x3fce9a78
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0       |
<<< Matrix Elements

[Test 1.2] Constructor with Rows and Cols
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        0
stride          4
memory          12
data pointer    0x3fce9a9c
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            0            0            0       |
           0            0            0            0       |
           0            0            0            0       |
<<< Matrix Elements

[Test 1.3] Constructor with Rows, Cols and Stride
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fce9ad0
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            0            0            0       |      0 
           0            0            0            0       |      0 
           0            0            0            0       |      0 
<<< Matrix Elements

[Test 1.4] Constructor with External Data
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        0
stride          4
memory          12
data pointer    0x3fc9928c
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            1            2            3       |
           4            5            6            7       |
           8            9           10           11       |
<<< Matrix Elements

[Test 1.5] Constructor with External Data and Stride
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc992e0
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            1            2            3       |      0 
           4            5            6            7       |      0 
           8            9           10           11       |      0 
<<< Matrix Elements

[Test 1.6] Copy Constructor
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fce9bd8
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            1            2            3       |      0 
           4            5            6            7       |      0 
           8            9           10           11       |      0 
<<< Matrix Elements

============ [tiny_matrix_test end] ============
```

### Group 2: 元素访问

```txt
============ [tiny_matrix_test start] ============

--- Test: Constructor & Destructor ---
[Test 1.1] Default Constructor
Matrix Info >>>
rows            1
cols            1
elements        1
paddings        0
stride          1
memory          1
data pointer    0x3fce9a78
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0       |
<<< Matrix Elements

[Test 1.2] Constructor with Rows and Cols
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        0
stride          4
memory          12
data pointer    0x3fce9a9c
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            0            0            0       |
           0            0            0            0       |
           0            0            0            0       |
<<< Matrix Elements

[Test 1.3] Constructor with Rows, Cols and Stride
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fce9ad0
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            0            0            0       |      0 
           0            0            0            0       |      0 
           0            0            0            0       |      0 
<<< Matrix Elements

[Test 1.4] Constructor with External Data
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        0
stride          4
memory          12
data pointer    0x3fc9928c
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            1            2            3       |
           4            5            6            7       |
           8            9           10           11       |
<<< Matrix Elements

[Test 1.5] Constructor with External Data and Stride
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc992e0
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            1            2            3       |      0 
           4            5            6            7       |      0 
           8            9           10           11       |      0 
<<< Matrix Elements

[Test 1.6] Copy Constructor
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fce9bd8
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            1            2            3       |      0 
           4            5            6            7       |      0 
           8            9           10           11       |      0 
<<< Matrix Elements

============ [tiny_matrix_test end] ============
```

### Group 3: ROI (Region of Interest) 区域操作

```txt
============ [tiny_matrix_test start] ============

--- Test: Data Manipulation ---
[Material Matrices]
matA:
Matrix Info >>>
rows            2
cols            3
elements        6
paddings        0
stride          3
memory          6
data pointer    0x3fce9a78
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
         0.1          0.2          0.3       |
         0.4          0.5          0.6       |
<<< Matrix Elements

matB:
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc991c4
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            1            2            3       |      0 
           4            5            6            7       |      0 
           8            9           10           11       |      0 
<<< Matrix Elements

matC:
Matrix Info >>>
rows            1
cols            1
elements        1
paddings        0
stride          1
memory          1
data pointer    0x3fce9a94
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0       |
<<< Matrix Elements

[Test 3.1] Copy ROI - Over Range Case
[>>> Error ! <<<] Invalid column position 
matB after copy_paste matA at (1, 2):
Matrix Elements >>>
           0            1            2            3       |      0 
           4            5            6            7       |      0 
           8            9           10           11       |      0 
<<< Matrix Elements

nothing changed.
[Test 3.1] Copy ROI - Suitable Range Case
matB after copy_paste matA at (1, 1):
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc991c4
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            1            2            3       |      0 
           4          0.1          0.2          0.3       |      0 
           8          0.4          0.5          0.6       |      0 
<<< Matrix Elements

successfully copied.
[Test 3.2] Copy Head
matC after copy_head matB:
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc991c4
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            1            2            3       |      0 
           4          0.1          0.2          0.3       |      0 
           8          0.4          0.5          0.6       |      0 
<<< Matrix Elements

[Test 3.2] Copy Head - Memory Sharing Check
matB(0, 0) = 99.99f
matC:
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc991c4
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
       99.99            1            2            3       |      0 
           4          0.1          0.2          0.3       |      0 
           8          0.4          0.5          0.6       |      0 
<<< Matrix Elements

[Test 3.3] Get a View of ROI - Low Level Function
get a view of ROI with overrange dimensions - rows:
[Error] Invalid ROI request.
get a view of ROI with overrange dimensions - cols:
[Error] Invalid ROI request.
get a view of ROI with suitable dimensions:
roi3:
Matrix Info >>>
rows            2
cols            2
elements        4
paddings        3
stride          5
memory          10
data pointer    0x3fc991dc
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      1   (This is a Sub-Matrix View)
<<< Matrix Info
Matrix Elements >>>
         0.1          0.2       |    0.3            0            8 
         0.4          0.5       |    0.6            0   4.2039e-45 
<<< Matrix Elements

[Test 3.4] Get a View of ROI - Using ROI Structure
Matrix Info >>>
rows            2
cols            2
elements        4
paddings        3
stride          5
memory          10
data pointer    0x3fc991dc
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      1   (This is a Sub-Matrix View)
<<< Matrix Info
Matrix Elements >>>
         0.1          0.2       |    0.3            0            8 
         0.4          0.5       |    0.6            0   4.2039e-45 
<<< Matrix Elements

[Test 3.5] Copy ROI - Low Level Function
Matrix Info >>>
rows            2
cols            2
elements        4
paddings        0
stride          2
memory          4
data pointer    0x3fce9bf4
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
         0.1          0.2       |
         0.4          0.5       |
<<< Matrix Elements

[Test 3.6] Copy ROI - Using ROI Structure
time for copy_roi using ROI structure: 31 ms
Matrix Info >>>
rows            2
cols            2
elements        4
paddings        0
stride          2
memory          4
data pointer    0x3fce9c08
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
         0.1          0.2       |
         0.4          0.5       |
<<< Matrix Elements

[Test 3.7] Block
time for block: 35 ms
Matrix Info >>>
rows            2
cols            2
elements        4
paddings        0
stride          2
memory          4
data pointer    0x3fce9c1c
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
         0.1          0.2       |
         0.4          0.5       |
<<< Matrix Elements

[Test 3.8] Swap Rows
matB before swap rows:
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc991c4
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
       99.99            1            2            3       |      0 
           4          0.1          0.2          0.3       |      0 
           8          0.4          0.5          0.6       |      0 
<<< Matrix Elements

matB after swap_rows(0, 2):
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc991c4
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           8          0.4          0.5          0.6       |      0 
           4          0.1          0.2          0.3       |      0 
       99.99            1            2            3       |      0 
<<< Matrix Elements

[Test 3.9] Swap Columns
matB before swap columns:
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc991c4
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           8          0.4          0.5          0.6       |      0 
           4          0.1          0.2          0.3       |      0 
       99.99            1            2            3       |      0 
<<< Matrix Elements

matB after swap_cols(0, 2):
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc991c4
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
         0.5          0.4            8          0.6       |      0 
         0.2          0.1            4          0.3       |      0 
           2            1        99.99            3       |      0 
<<< Matrix Elements

[Test 3.10] Clear
matB before clear:
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc991c4
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
         0.5          0.4            8          0.6       |      0 
         0.2          0.1            4          0.3       |      0 
           2            1        99.99            3       |      0 
<<< Matrix Elements

matB after clear:
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc991c4
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            0            0            0       |      0 
           0            0            0            0       |      0 
           0            0            0            0       |      0 
<<< Matrix Elements

============ [tiny_matrix_test end] ============
```


### Group 4: 算术运算符

```txt
============ [tiny_matrix_test start] ============

[Group 4.1: Assignment Operator Tests]

[Test 4.1.1] Assignment (Same Dimensions)
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
<<< Matrix Elements


[Test 4.1.2] Assignment (Different Dimensions)
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
<<< Matrix Elements


[Test 4.1.3] Assignment to Sub-Matrix (Expect Error)
[Error] Assignment to a sub-matrix is not allowed.
Matrix Elements >>>
           5            6       |      7            0            8 
           9           10       |     11            0   4.2039e-45 
<<< Matrix Elements


[Test 4.1.4] Self-Assignment
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
<<< Matrix Elements


[Group 4.2: Matrix Addition Tests]

[Test 4.2.1] Matrix Addition (Same Dimensions)
Matrix Elements >>>
           2            3            4       |
           5            6            7       |
<<< Matrix Elements


[Test 4.2.2] Sub-Matrix Addition
Matrix Elements >>>
          10           12       |      7            0            8 
          18           20       |     11            0           12 
<<< Matrix Elements


[Test 4.2.3] Full Matrix + Sub-Matrix Addition
Matrix Elements >>>
          12           14       |
          20           22       |
<<< Matrix Elements


[Test 4.2.4] Addition Dimension Mismatch (Expect Error)
[Error] Matrix addition failed: Dimension mismatch (2x2 vs 3x3)

[Group 4.3: Constant Addition Tests]

[Test 4.3.1] Full Matrix + Constant
Matrix Elements >>>
           5            6            7       |
           8            9           10       |
<<< Matrix Elements


[Test 4.3.2] Sub-Matrix + Constant
Matrix Elements >>>
           8            9       |      7            0            8 
          12           13       |     11            0           12 
<<< Matrix Elements


[Test 4.3.3] Add Zero
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements


[Test 4.3.4] Add Negative Constant
Matrix Elements >>>
          -5            5       |
          15           25       |
<<< Matrix Elements


[Group 4.4: Matrix Subtraction Tests]

[Test 4.4.1] Matrix Subtraction
Matrix Elements >>>
           4            5       |
           6            7       |
<<< Matrix Elements


[Test 4.4.2] Subtraction Dimension Mismatch (Expect Error)
[Error] Matrix subtraction failed: Dimension mismatch (2x2 vs 3x3)

[Group 4.5: Constant Subtraction Tests]

[Test 4.5.1] Full Matrix - Constant
Matrix Elements >>>
          -1            0            1       |
           2            3            4       |
<<< Matrix Elements


[Test 4.5.2] Sub-Matrix - Constant
Matrix Elements >>>
         3.5          4.5       |      7            0            8 
         7.5          8.5       |     11            0   4.2039e-45 
<<< Matrix Elements


[Group 4.6: Matrix Element-wise Division Tests]

[Test 4.6.1] Element-wise Division (Same Dimensions, No Zero)
Matrix Elements >>>
           5            5       |
           6            5       |
<<< Matrix Elements


[Test 4.6.2] Dimension Mismatch (Expect Error)
[Error] Matrix division failed: Dimension mismatch (2x2 vs 3x3)

[Test 4.6.3] Division by Matrix Containing Zero (Expect Error)
[Error] Matrix division failed: Division by zero detected.
Matrix Elements >>>
           5           10       |
          15           20       |
<<< Matrix Elements


[Group 4.7: Matrix Division by Constant Tests]

[Test 4.7.1] Divide Full Matrix by Positive Constant
Matrix Elements >>>
           1          1.5            2       |
         2.5            3          3.5       |
<<< Matrix Elements


[Test 4.7.2] Divide Matrix by Negative Constant
Matrix Elements >>>
          -2           -4       |
          -6           -8       |
<<< Matrix Elements


[Test 4.7.3] Division by Zero Constant (Expect Error)
[Error] Matrix division by zero is undefined.
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements


[Group 4.8: Matrix Exponentiation Tests]

[Test 4.8.1] Raise Each Element to Power of 2
Matrix Elements >>>
           4            9       |
          16           25       |
<<< Matrix Elements


[Test 4.8.2] Raise Each Element to Power of 0
Matrix Elements >>>
           1            1       |
           1            1       |
<<< Matrix Elements


[Test 4.8.3] Raise Each Element to Power of 1
Matrix Elements >>>
           9            8       |
           7            6       |
<<< Matrix Elements


[Test 4.8.4] Raise Each Element to Power of -1 (Expect Error or Warning)
[Error] Negative exponent not supported in operator^.
Matrix Elements >>>
           1            2       |
           4            5       |
<<< Matrix Elements


[Test 4.8.5] Raise Matrix Containing Zero to Power of 3
Matrix Elements >>>
           0            8       |
          -1           27       |
<<< Matrix Elements

============ [tiny_matrix_test end] ============
```

### Group 5: 线性代数测试

```txt
============ [tiny_matrix_test start] ============

[Group 5.1: Matrix Transpose Tests]

[Test 5.1.1] Transpose of 2x3 Matrix
Original 2x3 Matrix:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
<<< Matrix Elements

Transposed 3x2 Matrix:
Matrix Elements >>>
           1            4       |
           2            5       |
           3            6       |
<<< Matrix Elements


[Test 5.1.2] Transpose of 3x3 Square Matrix
Original 3x3 Matrix:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

Transposed 3x3 Matrix:
Matrix Elements >>>
           1            4            7       |
           2            5            8       |
           3            6            9       |
<<< Matrix Elements


[Test 5.1.3] Transpose of Matrix with Padding
Original 4x2 Matrix (with padding):
Matrix Elements >>>
           1            2       |      0 
           3            4       |      0 
           5            6       |      0 
           7            8       |      0 
<<< Matrix Elements

Transposed 2x4 Matrix:
Matrix Elements >>>
           1            3            5            7       |
           2            4            6            8       |
<<< Matrix Elements


[Test 5.1.4] Transpose of Empty Matrix
Matrix Elements >>>
           0       |
<<< Matrix Elements

Matrix Elements >>>
           0       |
<<< Matrix Elements


[Group 5.2: Matrix Minor and Cofactor Tests]

[Test 5.2.1] Minor of 3x3 Matrix (Remove Row 1, Col 1)
Original 3x3 Matrix:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

Minor Matrix (remove row 1, col 1, no sign):
Matrix Elements >>>
           1            3       |
           7            9       |
<<< Matrix Elements


[Test 5.2.2] Cofactor of 3x3 Matrix (Remove Row 1, Col 1)
Note: Cofactor matrix is the same as minor matrix.
      The sign (-1)^(i+j) is applied when computing cofactor value, not to matrix elements.
Cofactor Matrix (same as minor):
Matrix Elements >>>
           1            3       |
           7            9       |
<<< Matrix Elements


[Test 5.2.3] Minor (Remove Row 0, Col 0)
Matrix Elements >>>
           5            6       |
           8            9       |
<<< Matrix Elements


[Test 5.2.4] Cofactor (Remove Row 0, Col 0)
Note: Cofactor matrix is the same as minor matrix.
Matrix Elements >>>
           5            6       |
           8            9       |
<<< Matrix Elements


[Test 5.2.5] Cofactor (Remove Row 0, Col 1)
Note: Cofactor matrix is the same as minor matrix.
      When computing cofactor value, sign (-1)^(0+1) = -1 would be applied.
Cofactor Matrix (same as minor):
Matrix Elements >>>
           4            6       |
           7            9       |
<<< Matrix Elements


[Test 5.2.6] Minor (Remove Row 2, Col 2)
Matrix Elements >>>
           1            2       |
           4            5       |
<<< Matrix Elements


[Test 5.2.7] Cofactor (Remove Row 2, Col 2)
Note: Cofactor matrix is the same as minor matrix.
Matrix Elements >>>
           1            2       |
           4            5       |
<<< Matrix Elements


[Test 5.2.8] Minor of 4x4 Matrix (Remove Row 2, Col 1)
Matrix Elements >>>
           1            2            3            4       |
           5            6            7            8       |
           9           10           11           12       |
          13           14           15           16       |
<<< Matrix Elements

Minor Matrix:
Matrix Elements >>>
           1            3            4       |
           5            7            8       |
          13           15           16       |
<<< Matrix Elements


[Test 5.2.9] Cofactor of 4x4 Matrix (Remove Row 2, Col 1)
Note: Cofactor matrix is the same as minor matrix.
      When computing cofactor value, sign (-1)^(2+1) = -1 would be applied.
Cofactor Matrix (same as minor):
Matrix Elements >>>
           1            3            4       |
           5            7            8       |
          13           15           16       |
<<< Matrix Elements


[Test 5.2.10] Non-square Matrix (Expect Error)
Testing minor():
[Error] Minor requires square matrix.
Matrix Elements >>>
           0       |
<<< Matrix Elements

Testing cofactor():
[Error] Minor requires square matrix.
Matrix Elements >>>
           0       |
<<< Matrix Elements


[Group 5.3: Matrix Determinant Tests]

[Test 5.3.1] 1x1 Matrix Determinant
Matrix:
Matrix Elements >>>
           7       |
<<< Matrix Elements

Determinant: 7  (Expected: 7)

[Test 5.3.2] 2x2 Matrix Determinant
Matrix:
Matrix Elements >>>
           3            8       |
           4            6       |
<<< Matrix Elements

Determinant: -14  (Expected: -14)

[Test 5.3.3] 3x3 Matrix Determinant
Matrix:
Matrix Elements >>>
           1            2            3       |
           0            4            5       |
           1            0            6       |
<<< Matrix Elements

Determinant: 22  (Expected: 22)

[Test 5.3.4] 4x4 Matrix Determinant
Matrix:
Matrix Elements >>>
           1            2            3            4       |
           5            6            7            8       |
           9           10           11           12       |
          13           14           15           16       |
<<< Matrix Elements

Note: This matrix has linearly dependent rows (each row differs by constant 4),
      so the determinant should be 0.
Determinant: 0  (Expected: 0)

[Test 5.3.5] Non-square Matrix (Expect Error)
Matrix (3x4, non-square):
Matrix Elements >>>
           0            0            0            0       |
           0            0            0            0       |
           0            0            0            0       |
<<< Matrix Elements

[Error] Determinant can only be calculated for square matrices.
Determinant: 0  (Expected: 0 with error message)

[Group 5.4: Matrix Adjoint Tests]

[Test 5.4.1] Adjoint of 1x1 Matrix
Original Matrix:
Matrix Elements >>>
           5       |
<<< Matrix Elements

Adjoint Matrix:
Matrix Elements >>>
           1       |
<<< Matrix Elements


[Test 5.4.2] Adjoint of 2x2 Matrix
Original Matrix:
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Adjoint Matrix:
Matrix Elements >>>
           4           -2       |
          -3            1       |
<<< Matrix Elements


[Test 5.4.3] Adjoint of 3x3 Matrix
Original Matrix:
Matrix Elements >>>
           1            2            3       |
           0            4            5       |
           1            0            6       |
<<< Matrix Elements

Adjoint Matrix:
Matrix Elements >>>
          24          -12           -2       |
           5            3           -5       |
          -4            2            4       |
<<< Matrix Elements


[Test 5.4.4] Adjoint of Non-Square Matrix (Expect Error)
Original Matrix (2x3, non-square):
Matrix Elements >>>
           0            0            0       |
           0            0            0       |
<<< Matrix Elements

[Error] Adjoint can only be computed for square matrices.
Adjoint Matrix (should be empty due to error):
Matrix Elements >>>
           0       |
<<< Matrix Elements


[Group 5.5: Matrix Normalization Tests]

[Test 5.5.1] Normalize a Standard 2x2 Matrix
Before normalization:
Matrix Elements >>>
           3            4       |
           3            4       |
<<< Matrix Elements

After normalization (Expected L2 norm = 1):
Matrix Elements >>>
    0.424264     0.565685       |
    0.424264     0.565685       |
<<< Matrix Elements


[Test 5.5.2] Normalize a 2x2 Matrix with Stride=4 (Padding Test)
Before normalization:
Matrix Elements >>>
           3            4       |      0            0 
           3            4       |      0            0 
<<< Matrix Elements

After normalization:
Matrix Elements >>>
    0.424264     0.565685       |      0            0 
    0.424264     0.565685       |      0            0 
<<< Matrix Elements


[Test 5.5.3] Normalize a Zero Matrix (Expect Warning)
Matrix Elements >>>
           0            0       |
           0            0       |
<<< Matrix Elements

[Warning] Cannot normalize a zero matrix.

[Group 5.6: Matrix Norm Calculation Tests]

[Test 5.6.1] 2x2 Matrix Norm (Expect 5.0)
Matrix:
Matrix Elements >>>
           3            4       |
           0            0       |
<<< Matrix Elements

Calculated Norm: 5

[Test 5.6.2] Zero Matrix Norm (Expect 0.0)
Matrix:
Matrix Elements >>>
           0            0            0       |
           0            0            0       |
           0            0            0       |
<<< Matrix Elements

Calculated Norm: 0

[Test 5.6.3] Matrix with Negative Values
Matrix:
Matrix Elements >>>
          -1           -2       |
          -3           -4       |
<<< Matrix Elements

Calculated Norm: 5.47723  (Expect sqrt(30) ≈ 5.477)

[Test 5.6.4] 2x2 Matrix with Stride=4 (Padding Test)
Matrix:
Matrix Elements >>>
           1            2       |      0            0 
           3            4       |      0            0 
<<< Matrix Elements

Calculated Norm: 5.47723  (Expect sqrt(30) ≈ 5.477)

[Group 5.7: Matrix Inversion Tests]

[Test 5.7.1] Inverse of 2x2 Matrix
Original Matrix:
Matrix Elements >>>
           4            7       |
           2            6       |
<<< Matrix Elements

Inverse Matrix:
Matrix Elements >>>
         0.6         -0.7       |
        -0.2          0.4       |
<<< Matrix Elements

Expected Approx:
[ 0.6  -0.7 ]
[ -0.2  0.4 ]

[Test 5.7.2] Singular Matrix (Expect Error)
Original Matrix:
Matrix Elements >>>
           1            2       |
           2            4       |
<<< Matrix Elements

Note: This matrix is singular (determinant = 0), so inverse should fail.
[Error] Singular matrix, inverse does not exist.
Inverse Matrix (Should be zero matrix):
Matrix Elements >>>
           0            0       |
           0            0       |
<<< Matrix Elements


[Test 5.7.3] Inverse of 3x3 Matrix
Original Matrix:
Matrix Elements >>>
           3            0            2       |
           2            0           -2       |
           0            1            1       |
<<< Matrix Elements

Inverse Matrix:
Matrix Elements >>>
         0.2          0.2           -0       |
        -0.2          0.3            1       |
         0.2         -0.3            0       |
<<< Matrix Elements


[Test 5.7.4] Non-Square Matrix (Expect Error)
Original Matrix (2x3, non-square):
Matrix Elements >>>
           0            0            0       |
           0            0            0       |
<<< Matrix Elements

[Error] Inverse can only be computed for square matrices.
Inverse Matrix (should be empty due to error):
Matrix Elements >>>
           0       |
<<< Matrix Elements


[Group 5.8: Matrix Utilities Tests]

[Test 5.8.1] Generate Identity Matrix (eye)
3x3 Identity Matrix:
Matrix Elements >>>
           1            0            0       |
           0            1            0       |
           0            0            1       |
<<< Matrix Elements

5x5 Identity Matrix:
Matrix Elements >>>
           1            0            0            0            0       |
           0            1            0            0            0       |
           0            0            1            0            0       |
           0            0            0            1            0       |
           0            0            0            0            1       |
<<< Matrix Elements


[Test 5.8.2] Generate Ones Matrix
3x4 Ones Matrix:
Matrix Elements >>>
           1            1            1            1       |
           1            1            1            1       |
           1            1            1            1       |
<<< Matrix Elements

4x4 Ones Matrix (Square):
Matrix Elements >>>
           1            1            1            1       |
           1            1            1            1       |
           1            1            1            1       |
           1            1            1            1       |
<<< Matrix Elements


[Test 5.8.3] Augment Two Matrices Horizontally [A | B]
Matrix A:
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Matrix B:
Matrix Elements >>>
           5            6            7       |
           8            9           10       |
<<< Matrix Elements

Augmented Matrix [A | B]:
Matrix Elements >>>
           1            2            5            6            7       |
           3            4            8            9           10       |
<<< Matrix Elements


[Test 5.8.4] Augment with Row Mismatch (Expect Error)
[Error] Cannot augment matrices: Row counts do not match (2 vs 3)
Matrix Info >>>
rows            1
cols            1
elements        1
paddings        0
stride          1
memory          1
data pointer    0x3fce9d04
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info

[Test 5.8.5] Vertically Stack Two Matrices [A; B]
Matrix A (top):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
<<< Matrix Elements

Matrix B (bottom):
Matrix Elements >>>
           7            8            9       |
          10           11           12       |
<<< Matrix Elements

Vertically Stacked Matrix [A; B]:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
          10           11           12       |
<<< Matrix Elements

Expected: 4x3 matrix with A on top, B on bottom

[Test 5.8.6] Vertical Stack with Different Row Counts (Same Columns)
Matrix A (1x3):
Matrix Elements >>>
           1            2            3       |
<<< Matrix Elements

Matrix B (3x3):
Matrix Elements >>>
           4            5            6       |
           7            8            9       |
          10           11           12       |
<<< Matrix Elements

Vertically Stacked Matrix [A; B] (1x3 + 3x3 = 4x3):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
          10           11           12       |
<<< Matrix Elements


[Test 5.8.7] VStack with Column Mismatch (Expect Error)
Matrix A (2x2):
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Matrix B (2x3, different columns):
Matrix Elements >>>
           5            6            7       |
           8            9           10       |
<<< Matrix Elements

[Error] Cannot vstack matrices: Column counts do not match (2 vs 3)
Result (should be empty due to error):
Matrix Info >>>
rows            1
cols            1
elements        1
paddings        0
stride          1
memory          1
data pointer    0x3fce9dfc
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info

[Group 5.9: Gaussian Elimination Tests]

[Test 5.9.1] 3x3 Matrix (Simple Upper Triangular)
Original Matrix:
Matrix Elements >>>
           2            1           -1       |
          -3           -1            2       |
          -2            1            2       |
<<< Matrix Elements

After Gaussian Elimination (Should be upper triangular):
Matrix Elements >>>
           2            1           -1       |
           0          0.5          0.5       |
           0            0           -1       |
<<< Matrix Elements


[Test 5.9.2] 3x4 Augmented Matrix (Linear System Ax = b)
Original Augmented Matrix [A | b]:
Matrix Elements >>>
           1            2           -1            8       |
          -3           -1            2          -11       |
          -2            1            2           -3       |
<<< Matrix Elements

After Gaussian Elimination (Row Echelon Form):
Matrix Elements >>>
           1            2           -1            8       |
           0            5           -1           13       |
           0            0            1            0       |
<<< Matrix Elements


[Test 5.9.3] Singular Matrix (No Unique Solution)
Original Singular Matrix:
Matrix Elements >>>
           1            2       |
           2            4       |
<<< Matrix Elements

After Gaussian Elimination (Should show rows of zeros):
Matrix Elements >>>
           1            2       |
           0            0       |
<<< Matrix Elements


[Test 5.9.4] Zero Matrix
Matrix Elements >>>
           0            0            0       |
           0            0            0       |
           0            0            0       |
<<< Matrix Elements

After Gaussian Elimination (Should be a zero matrix):
Matrix Elements >>>
           0            0            0       |
           0            0            0       |
           0            0            0       |
<<< Matrix Elements


[Group 5.10: Row Reduce from Gaussian (RREF) Tests]

[Test 5.10.1] 3x4 Augmented Matrix
Original Matrix:
Matrix Elements >>>
           1            2           -1           -4       |
           2            3           -1          -11       |
          -2            0           -3           22       |
<<< Matrix Elements

RREF Result:
Matrix Elements >>>
           1            0            0           -8       |
           0            1            0            1       |
           0            0            1           -2       |
<<< Matrix Elements


[Test 5.10.2] 2x3 Matrix
Original Matrix:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
<<< Matrix Elements

RREF Result:
Matrix Elements >>>
           1            0           -1       |
           0            1            2       |
<<< Matrix Elements


[Test 5.10.3] Already Reduced Matrix
Original Matrix:
Matrix Elements >>>
           1            0            2       |
           0            1            3       |
<<< Matrix Elements

RREF Result:
Matrix Elements >>>
           1            0            2       |
           0            1            3       |
<<< Matrix Elements


[Group 5.11: Gaussian Inverse Tests]

[Test 5.11.1] 2x2 Matrix Inverse
Original matrix (mat1):
Matrix Elements >>>
           4            7       |
           2            6       |
<<< Matrix Elements

Inverse matrix (mat1):
Matrix Elements >>>
         0.6         -0.7       |
        -0.2          0.4       |
<<< Matrix Elements


[Test 5.11.2] Identity Matrix Inverse
Original matrix (Identity):
Matrix Elements >>>
           1            0            0       |
           0            1            0       |
           0            0            1       |
<<< Matrix Elements

Inverse matrix (Identity):
Matrix Elements >>>
           1            0            0       |
           0            1            0       |
           0            0            1       |
<<< Matrix Elements


[Test 5.11.3] Singular Matrix (Expected: No Inverse)
Original matrix (singular):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

[Error] Matrix is singular, cannot compute inverse.
Inverse matrix (singular):
Matrix Elements >>>
           0       |
<<< Matrix Elements


[Test 5.11.4] 3x3 Matrix Inverse
Original matrix (mat4):
Matrix Elements >>>
           4            7            2       |
           3            5            1       |
           8            6            9       |
<<< Matrix Elements

Inverse matrix (mat4):
Matrix Elements >>>
    -1.85714      2.42857     0.142857       |
    0.904762    -0.952381   -0.0952381       |
     1.04762     -1.52381     0.047619       |
<<< Matrix Elements


[Test 5.11.5] Non-square Matrix Inverse (Expected Error)
Original matrix (non-square):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
<<< Matrix Elements

[Error] Inversion requires a square matrix.
Inverse matrix (non-square):
Matrix Elements >>>
           0       |
<<< Matrix Elements


[Group 5.12: Dot Product Tests]

[Test 5.12.1] Valid Dot Product (Same Length Vectors)
Vector A:
Matrix Elements >>>
           1       |
           2       |
           3       |
<<< Matrix Elements

Vector B:
Matrix Elements >>>
           4       |
           5       |
           6       |
<<< Matrix Elements

Dot product of vectorA and vectorB: 32

[Test 5.12.2] Invalid Dot Product (Dimension Mismatch)
Vector A (3x1):
Matrix Elements >>>
           1       |
           2       |
           3       |
<<< Matrix Elements

Vector C (2x1, different size):
Matrix Elements >>>
           1       |
           2       |
<<< Matrix Elements

[Error] Dot product can only be computed for two vectors of the same length.
Dot product (dimension mismatch): 0

[Test 5.12.3] Dot Product of Zero Vectors
Zero Vector A:
Matrix Elements >>>
           0       |
           0       |
           0       |
<<< Matrix Elements

Zero Vector B:
Matrix Elements >>>
           0       |
           0       |
           0       |
<<< Matrix Elements

Dot product of zero vectors: 0

[Group 5.13: Solve Linear System Tests]

[Test 5.13.1] Solving a Simple 2x2 System Ax = b
Matrix A:
Matrix Elements >>>
           2            1       |
           1            3       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           5       |
           6       |
<<< Matrix Elements

Solution x:
Matrix Elements >>>
         1.8       |
         1.4       |
<<< Matrix Elements


[Test 5.13.2] Solving a 3x3 System Ax = b
Matrix A:
Matrix Elements >>>
           1            2            1       |
           2            0            3       |
           3            2            1       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           9       |
           8       |
           7       |
<<< Matrix Elements

Solution x:
Matrix Elements >>>
          -1       |
     3.33333       |
     3.33333       |
<<< Matrix Elements


[Test 5.13.3] Solving a System Where One Row is All Zeros (Expect Failure or Infinite Solutions)
Matrix A (has zero row):
Matrix Elements >>>
           1            2            3       |
           0            0            0       |
           4            5            6       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           9       |
           0       |
          15       |
<<< Matrix Elements

[Error] Pivot is zero, matrix is singular.
Solution x:
Matrix Elements >>>
           0       |
<<< Matrix Elements


[Test 5.13.4] Solving a System with Zero Determinant (Singular Matrix)
Matrix A (singular, determinant = 0):
Matrix Elements >>>
           2            4            1       |
           1            2            3       |
           3            6            2       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           5       |
           6       |
           7       |
<<< Matrix Elements

[Error] Pivot is zero, matrix is singular.
Solution x:
Matrix Elements >>>
           0       |
<<< Matrix Elements


[Test 5.13.5] Solving a System with Linearly Dependent Rows (Expect Failure or Infinite Solutions)
Matrix A (all rows linearly dependent):
Matrix Elements >>>
           1            1            1       |
           2            2            2       |
           3            3            3       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           6       |
          12       |
          18       |
<<< Matrix Elements

[Error] Pivot is zero, matrix is singular.
Solution x:
Matrix Elements >>>
           0       |
<<< Matrix Elements


[Test 5.13.6] Solving a Larger 4x4 System Ax = b
Matrix A:
Matrix Elements >>>
           4            2            3            1       |
           2            5            1            2       |
           3            1            6            3       |
           1            2            3            4       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
          10       |
          12       |
          14       |
          16       |
<<< Matrix Elements

Solution x:
Matrix Elements >>>
     1.80645       |
    0.258065       |
   -0.516129       |
     3.80645       |
<<< Matrix Elements


[Group 5.14: Band Solve Tests]

[Test 5.14.1] Simple 3x3 Band Matrix
Matrix A:
Matrix Elements >>>
           2            1            0       |
           1            3            2       |
           0            1            4       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           5       |
           6       |
           7       |
<<< Matrix Elements

Solution x:
Matrix Elements >>>
         2.5       |
           0       |
        1.75       |
<<< Matrix Elements


[Test 5.14.2] 4x4 Band Matrix
Matrix A:
Matrix Elements >>>
           2            1            0            0       |
           1            3            2            0       |
           0            1            4            2       |
           0            0            1            5       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           8       |
           9       |
          10       |
          11       |
<<< Matrix Elements

Solution x:
Matrix Elements >>>
     3.51429       |
    0.971429       |
     1.28571       |
     1.94286       |
<<< Matrix Elements


[Test 5.14.3] Incompatible Dimensions (Expect Error)
Matrix A (3x3):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

Vector b (2x1, incompatible):
Matrix Elements >>>
          10       |
          11       |
<<< Matrix Elements

[Error] Matrix dimensions are not compatible for solving.
Solution x:
Matrix Elements >>>
           0       |
<<< Matrix Elements


[Test 5.14.4] Singular Matrix (No Unique Solution)
Matrix A (singular, linearly dependent rows):
Matrix Elements >>>
           1            2            3       |
           2            4            6       |
           3            6            9       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
          10       |
          20       |
          30       |
<<< Matrix Elements

[Error] Zero pivot detected in bandSolve. Cannot proceed.
Solution x:
Matrix Elements >>>
           0       |
           0       |
           0       |
<<< Matrix Elements


[Group 5.15: Roots Tests]

[Test 5.15.1] Solving a Simple 2x2 System Ax = b
Matrix A:
Matrix Elements >>>
           2            1       |
           1            3       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           5       |
           6       |
<<< Matrix Elements

Solution x:
Matrix Elements >>>
         1.8       |
         1.4       |
<<< Matrix Elements


[Test 5.15.2] Solving a 3x3 System Ax = b
Matrix A:
Matrix Elements >>>
           1            2            1       |
           2            0            3       |
           3            2            1       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           9       |
           8       |
           7       |
<<< Matrix Elements

Solution x:
Matrix Elements >>>
          -1       |
     3.33333       |
     3.33333       |
<<< Matrix Elements


[Test 5.15.3] Singular Matrix (No Unique Solution)
Matrix A (singular, linearly dependent rows):
Matrix Elements >>>
           1            2       |
           2            4       |
<<< Matrix Elements

Vector b:
Matrix Elements >>>
           5       |
           6       |
<<< Matrix Elements

[Error] Pivot is zero, system may have no solution.
Solution x:
Matrix Elements >>>
           0       |
<<< Matrix Elements


[Test 5.15.4] Incompatible Dimensions (Expect Error)
Matrix A (3x3):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

Vector b (2x1, incompatible):
Matrix Elements >>>
          10       |
          11       |
<<< Matrix Elements

[Error] Cannot augment matrices: Row counts do not match (3 vs 2)
[Error] Pivot is zero, system may have no solution.
Solution x:
Matrix Elements >>>
           0       |
<<< Matrix Elements

============ [tiny_matrix_test end] ============
```

### Group 6: 流操作符

```txt
============ [tiny_matrix_test start] ============

[Group 6: Stream Operators Tests]

[Test 6.1] Stream Insertion Operator (<<) for Mat
Matrix mat1:
1 2 3
4 5 6
7 8 9


[Test 6.2] Stream Insertion Operator (<<) for Mat::ROI
ROI created: ROI(pos_x=1, pos_y=2, width=3, height=4)
Expected output:
  row start: 2 (pos_y)
  col start: 1 (pos_x)
  row count: 4 (height)
  col count: 3 (width)

Actual output:
row start 2
col start 1
row count 4
col count 3


[Test 6.3] Stream Extraction Operator (>>) for Mat
Simulated input: "10 20 30 40"
Matrix mat2 after input:
10 20
30 40

Expected: [10, 20; 30, 40]

[Test 6.4] Stream Extraction Operator (>>) for Mat (2x3 matrix)
Simulated input: "1.5 2.5 3.5 4.5 5.5 6.5"
Matrix mat3 after input:
1.5 2.5 3.5
4.5 5.5 6.5

Expected: [1.5, 2.5, 3.5; 4.5, 5.5, 6.5]
============ [tiny_matrix_test end] ============
```

### Group 7: 矩阵操作

```txt
============ [tiny_matrix_test start] ============

[Group 7: Global Arithmetic Operators Tests]

[Test 7.1] Matrix Addition (operator+)
Matrix A:
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Matrix B:
Matrix Elements >>>
           5            6       |
           7            8       |
<<< Matrix Elements

matA + matB:
6 8
10 12


[Test 7.2] Matrix Addition with Constant (operator+)
Matrix A:
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Constant: 5.0
matA + 5.0f:
6 7
8 9


[Test 7.3] Matrix Subtraction (operator-)
Matrix A:
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Matrix B:
Matrix Elements >>>
           5            6       |
           7            8       |
<<< Matrix Elements

matA - matB:
-4 -4
-4 -4


[Test 7.4] Matrix Subtraction with Constant (operator-)
Matrix A:
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Constant: 2.0
matA - 2.0f:
-1 0
1 2


[Test 7.5] Matrix Multiplication (operator*)
Matrix C (2x3):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
<<< Matrix Elements

Matrix D (3x2):
Matrix Elements >>>
           7            8       |
           9           10       |
          11           12       |
<<< Matrix Elements

matC * matD:
58 64
139 154


[Test 7.6] Matrix Multiplication with Constant (operator*)
Matrix A:
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Constant: 2.0
matA * 2.0f:
2 4
6 8


[Test 7.7] Matrix Division (operator/)
Matrix A:
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Constant: 2.0
matA / 2.0f:
0.5 1
1.5 2


[Test 7.8] Matrix Division Element-wise (operator/)
Matrix A:
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Matrix B:
Matrix Elements >>>
           5            6       |
           7            8       |
<<< Matrix Elements

matA / matB:
0.2 0.333333
0.428571 0.5


[Test 7.9] Matrix Comparison (operator==)
Matrix E:
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Matrix F:
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

matE == matF: True

After modifying matF(0,0) = 5:
Matrix E:
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Matrix F:
Matrix Elements >>>
           5            2       |
           3            4       |
<<< Matrix Elements

operator == Error: 0 0, m1.data=1, m2.data=5, diff=4
matE == matF after modification: False
============ [tiny_matrix_test end] ============
```

### Group 8: 边界条件与错误处理

```txt
============ [tiny_matrix_test start] ============

[Group 8: Boundary Conditions and Error Handling Tests]

[Test 8.1] Null Pointer Handling in print_matrix
[Error] Cannot print matrix: data pointer is null.

[Test 8.2] Null Pointer Handling in operator<<
[Error] Cannot print matrix: data pointer is null.


[Test 8.3] Invalid Block Parameters
[Error] Invalid block parameters: negative start position or non-positive block size.
block(-1, 0, 2, 2): Error
[Error] Block exceeds matrix boundaries.
block(2, 2, 2, 2) on 3x3 matrix: Error
[Error] Invalid block parameters: negative start position or non-positive block size.
block(0, 0, 0, 2): Error

[Test 8.4] Invalid swap_rows Parameters
Before invalid swap_rows:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

Error: row index out of range
After swap_rows(-1, 1):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

Error: row index out of range
After swap_rows(0, 5):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements


[Test 8.5] Invalid swap_cols Parameters
Before invalid swap_cols:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

Error: column index out of range
After swap_cols(-1, 1):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

Error: column index out of range
After swap_cols(0, 5):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements


[Test 8.6] Division by Zero
[Error] Division by zero in operator/.
mat3 / 0.0f: Error

[Test 8.7] Matrix Division with Zero Elements
[Error] Matrix division failed: Division by zero detected.
mat4 /= divisor (with zero):
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements


[Test 8.8] Empty Matrix Operations
Empty matrix addition: Error
============ [tiny_matrix_test end] ============
```

### Group 9: 性能基准测试

```txt
============ [tiny_matrix_test start] ============

[Group 9: Performance Benchmarks Tests]

[Test 9.1] Matrix Addition Performance
[Performance] 50x50 Matrix Addition (100 iterations): 15963.00 us total, 159.63 us avg

[Test 9.2] Matrix Multiplication Performance
[Performance] 30x30 Matrix Multiplication (100 iterations): 66050.00 us total, 660.50 us avg

[Test 9.3] Matrix Transpose Performance
[Performance] 50x30 Matrix Transpose (100 iterations): 17895.00 us total, 178.95 us avg

[Test 9.4] Determinant Calculation Performance
[Performance] Computing determinant (no warmup due to recursive nature)...
[Performance] 5x5 Determinant (10 iterations): 16679.00 us total, 1667.90 us avg
[Note] Determinant uses recursive Laplace expansion (O(n!)), suitable only for small matrices.

[Test 9.5] Matrix Copy with Padding Performance
[Performance] 8x8 Copy ROI (with padding) (100 iterations): 2221.00 us total, 22.21 us avg

[Test 9.6] Element Access Performance
[Performance] Computing element access (warmup)...
[Performance] 50x50 Element Access (all elements) (100 iterations): 9691.00 us total, 96.91 us avg
============ [tiny_matrix_test end] ============
```

### Group 10: 内存布局测试

```txt
============ [tiny_matrix_test start] ============

[Group 10: Memory Layout Tests (Padding and Stride)]

[Test 10.1] Contiguous Memory (no padding)
Matrix 3x4 (stride=4, pad=0):
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        0
stride          4
memory          12
data pointer    0x3fce9a78
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            1            2            3       |
           4            5            6            7       |
           8            9           10           11       |
<<< Matrix Elements


[Test 10.2] Padded Memory (stride > col)
Matrix 3x4 (stride=5, pad=1):
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc991e4
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            1            2            3       |      0 
           4            5            6            7       |      0 
           8            9           10           11       |      0 
<<< Matrix Elements


[Test 10.3] Addition with Padded Matrices
Result of padded matrix addition:
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fce9bc8
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
          11           22           33           44       |      0 
          55           66           77           88       |      0 
          99          110          121          132       |      0 
<<< Matrix Elements


[Test 10.4] ROI Operations with Padded Matrices
ROI (1,1,2,2) from padded matrix:
Matrix Info >>>
rows            2
cols            2
elements        4
paddings        3
stride          5
memory          10
data pointer    0x3fc991fc
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      1   (This is a Sub-Matrix View)
<<< Matrix Info
Matrix Elements >>>
           5            6       |      7            0            8 
           9           10       |     11            0   4.2039e-45 
<<< Matrix Elements


[Test 10.5] Copy Operations Preserve Stride
Copied matrix (should have stride=4, no padding):
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        0
stride          4
memory          12
data pointer    0x3fce9c3c
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            1            2            3       |
           4            5            6            7       |
           8            9           10           11       |
<<< Matrix Elements

============ [tiny_matrix_test end] ============
```

### Group 11: 特征值与特征向量分解

```txt
============ [tiny_matrix_test start] ============

[Group 11: Eigenvalue and Eigenvector Decomposition Tests]

[Test 11.1] is_symmetric() - Basic Functionality
[Test 11.1.1] Symmetric 3x3 Matrix
Matrix:
Matrix Elements >>>
           4            1            2       |
           1            3            0       |
           2            0            5       |
<<< Matrix Elements

Is symmetric: True (Expected: True)

[Test 11.1.2] Non-Symmetric 3x3 Matrix
Matrix:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

Is symmetric: False (Expected: False)

[Test 11.1.3] Non-Square Matrix (2x3)
Is symmetric: False (Expected: False)

[Test 11.1.4] Symmetric Matrix with Small Numerical Errors
Matrix with error 1e-05:
Matrix Elements >>>
           1      2.00001       |
           2            3       |
<<< Matrix Elements

Difference: |A(0,1) - A(1,0)| = 1.001358e-05 (Expected: 0.000010)
A(0,1) stored value: 2.00001001 (Expected: 2.00001001)
Is symmetric (tolerance=1e-4): True (Expected: True, tolerance > error) [PASS]
Is symmetric (tolerance=1e-6): False (Expected: False, tolerance < error) [PASS]
Difference accuracy: |actual_diff - expected_diff| = 1.36e-08 [PASS - difference stored correctly]

[Test 11.2] power_iteration() - Dominant Eigenvalue

[Test 11.2.1] Simple 2x2 Matrix
Matrix:
Matrix Elements >>>
        2.00         1.00       |
        1.00         2.00       |
<<< Matrix Elements


[Expected Results]
  Expected eigenvalues: 3.0 (largest), 1.0 (smallest)
  Expected dominant eigenvector (for λ=3): approximately [0.707, 0.707] or [-0.707, -0.707] (normalized)
  Expected dominant eigenvector (for λ=1): approximately [0.707, -0.707] or [-0.707, 0.707] (normalized)

[Actual Results]
  Dominant eigenvalue: 3.00 (Expected: 3.0, largest eigenvalue)
  Iterations: 2
  Status: OK
  Dominant eigenvector:
Matrix Elements >>>
        0.71       |
        0.71       |
<<< Matrix Elements

  Error from expected (3.0): 0.00 [PASS]

[Test 11.2.2] 3x3 Stiffness Matrix (SHM Application)
Stiffness Matrix:
Matrix Elements >>>
        2.00        -1.00         0.00       |
       -1.00         2.00        -1.00       |
        0.00        -1.00         2.00       |
<<< Matrix Elements


[Expected Results]
  Expected eigenvalues (approximate): 3.414 (largest), 2.000, 0.586 (smallest)
  Expected primary frequency: sqrt(3.414) ≈ 1.848 rad/s

[Actual Results]
  Dominant eigenvalue (primary frequency squared): 3.41
  Primary frequency: 1.85 rad/s (Expected: ~1.848 rad/s)
  Iterations: 8
  Status: OK
  Error from expected (3.41): 0.00 [PASS]

[Test 11.2.3] Non-Square Matrix (Expect Error)
[Error] Power iteration requires a square matrix.
Status: Error (Expected)

[Test 11.3] eigendecompose_jacobi() - Symmetric Matrix Decomposition

[Test 11.3.1] 2x2 Symmetric Matrix - Complete Decomposition
[Expected Results]
  Expected eigenvalues: 3.0, 1.0 (in any order)
  Expected eigenvectors (for λ=3): [0.707, 0.707] or [-0.707, -0.707] (normalized)
  Expected eigenvectors (for λ=1): [0.707, -0.707] or [-0.707, 0.707] (normalized)

[Actual Results]
Eigenvalues:
Matrix Elements >>>
        1.00       |
        3.00       |
<<< Matrix Elements

Eigenvectors (each column is an eigenvector):
Matrix Elements >>>
        0.71         0.71       |
       -0.71         0.71       |
<<< Matrix Elements

Iterations: 2
Status: OK
Eigenvalue check (should be 3.0 and 1.0): [PASS]

[Verification] Check A * v = lambda * v for first eigenvector:
A * v:
Matrix Elements >>>
        0.71       |
       -0.71       |
<<< Matrix Elements

lambda * v:
Matrix Elements >>>
        0.71       |
       -0.71       |
<<< Matrix Elements

Verification (A*v = λ*v): [PASS]

[Test 11.3.2] 3x3 Stiffness Matrix (SHM Application)
[Expected Results]
  Expected eigenvalues (approximate): 3.414, 2.000, 0.586
  Expected natural frequencies: 1.848, 1.414, 0.765 rad/s
  Note: Eigenvalues may appear in any order

[Actual Results]
Eigenvalues (natural frequencies squared):
Matrix Elements >>>
        3.41       |
        0.59       |
        2.00       |
<<< Matrix Elements

Natural frequencies (rad/s):
  Mode 0: 1.85 rad/s (Expected: ~1.85 rad/s) [PASS]
  Mode 1: 0.77 rad/s (Expected: ~0.76 rad/s) [PASS]
  Mode 2: 1.41 rad/s (Expected: ~1.41 rad/s) [PASS]
Eigenvectors (mode shapes):
Matrix Elements >>>
        0.50         0.50        -0.71       |
       -0.71         0.71         0.00       |
        0.50         0.50         0.71       |
<<< Matrix Elements

Iterations: 9
Status: OK

[Test 11.3.3] Diagonal Matrix (Eigenvalues on diagonal)
Matrix:
Matrix Elements >>>
        5.00         0.00         0.00       |
        0.00         3.00         0.00       |
        0.00         0.00         1.00       |
<<< Matrix Elements


[Expected Results]
  Expected eigenvalues: 5.0, 3.0, 1.0 (diagonal elements, may be in any order)
  Expected eigenvectors: standard basis vectors [1,0,0], [0,1,0], [0,0,1] (or their negatives)
  Expected iterations: 1 (diagonal matrix should converge immediately)

[Actual Results]
Eigenvalues:
Matrix Elements >>>
        5.00       |
        3.00       |
        1.00       |
<<< Matrix Elements

Eigenvectors:
Matrix Elements >>>
        1.00         0.00         0.00       |
        0.00         1.00         0.00       |
        0.00         0.00         1.00       |
<<< Matrix Elements

Iterations: 1 (Expected: 1)
Eigenvalue check (should be 5.0, 3.0, 1.0): [PASS]

[Test 11.4] eigendecompose_qr() - General Matrix Decomposition

[Test 11.4.1] General 2x2 Matrix
Matrix:
Matrix Elements >>>
        1.00         2.00       |
        3.00         4.00       |
<<< Matrix Elements


[Expected Results]
  Expected eigenvalues: (5+√33)/2 ≈ 5.372, (5-√33)/2 ≈ -0.372
  Note: This is a non-symmetric matrix, eigenvalues are real but may have complex eigenvectors

[Actual Results]
Eigenvalues:
Matrix Elements >>>
        5.37       |
       -0.37       |
<<< Matrix Elements

Eigenvectors:
Matrix Elements >>>
        0.42         0.91       |
        0.91        -0.42       |
<<< Matrix Elements

Iterations: 6
Status: OK
Eigenvalue 1: 5.37 (Expected: 5.37, Error: 0.00, Rel Error: 0.01%) [PASS]
Eigenvalue 2: -0.37 (Expected: -0.37, Error: 0.00, Rel Error: 0.08%) [PASS]
Overall eigenvalue check: [PASS]

[Test 11.4.2] Non-Symmetric 3x3 Matrix
Matrix [1,2,3; 4,5,6; 7,8,9]:
Matrix Elements >>>
        1.00         2.00         3.00       |
        4.00         5.00         6.00       |
        7.00         8.00         9.00       |
<<< Matrix Elements


[Expected Results]
  Expected eigenvalues (theoretical): 16.12, -1.12, 0.00
  Note: This matrix is rank-deficient (determinant = 0), so one eigenvalue is 0
  Note: QR algorithm may have numerical errors, especially for non-symmetric matrices
  Acceptable range: largest eigenvalue ~15-18, smallest eigenvalue near 0

[Actual Results]
Eigenvalues:
Matrix Elements >>>
       17.33       |
       -2.94       |
        0.00       |
<<< Matrix Elements

Eigenvectors:
Matrix Elements >>>
        0.35         1.19         0.36       |
        0.57         0.36         0.08       |
        0.79        -0.46        -0.20       |
<<< Matrix Elements

Iterations: 8
Status: OK
Eigenvalue 0: 17.33 (Expected: 16.12, Error: 1.21, Rel Error: 7.49%) [PASS]
Eigenvalue 1: -2.94 (Expected: -1.12, Error: 1.82, Rel Error: 162.13%) [FAIL - error too large]
Eigenvalue 2: 0.00 (Expected: 0.00, Error: 0.00, Rel Error: 0.38%) [PASS]
Overall eigenvalue check: [FAIL - some eigenvalues have large errors]

[Test 11.5] eigendecompose() - Automatic Method Selection

[Test 11.5.1] Symmetric Matrix (Auto-select: Jacobi)
Matrix:
Matrix Elements >>>
        4.00         1.00         2.00       |
        1.00         3.00         0.00       |
        2.00         0.00         5.00       |
<<< Matrix Elements


[Expected Results]
  Method: Should automatically use Jacobi (symmetric matrix detected)
  Expected eigenvalues (approximate): 6.67, 3.48, 1.85
  Note: Eigenvalues may appear in any order

[Actual Results]
Eigenvalues:
Matrix Elements >>>
        1.85       |
        3.48       |
        6.67       |
<<< Matrix Elements

Iterations: 8
Status: OK
Method used: Jacobi (auto-selected for symmetric matrix)

[Test 11.5.2] Non-Symmetric Matrix (Auto-select: QR)
[Expected Results]
  Method: Should automatically use QR (non-symmetric matrix detected)
  Expected eigenvalues (theoretical): 16.12, -1.12, 0.00
  Note: One eigenvalue should be near 0 (rank-deficient matrix)
  Note: QR algorithm may have numerical errors for non-symmetric matrices
  Acceptable: largest ~15-18, smallest near 0, one near -1 to -3

[Actual Results]
Eigenvalues:
Matrix Elements >>>
       17.33       |
       -2.94       |
        0.00       |
<<< Matrix Elements

Iterations: 8
Status: OK
Method used: QR (auto-selected for non-symmetric matrix)
Eigenvalue 0: 17.33 (Expected: 16.12, Error: 1.21, Rel Error: 7.49%) [PASS]
Eigenvalue 1: -2.94 (Expected: -1.12, Error: 1.82, Rel Error: 162.13%) [FAIL - error too large]
Eigenvalue 2: 0.00 (Expected: 0.00, Error: 0.00, Rel Error: 0.38%) [PASS]
Overall eigenvalue check: [FAIL - some eigenvalues have large errors]

[Test 11.6] SHM Application - Structural Dynamics Analysis

[Test 11.6.1] 4-DOF Mass-Spring System
Stiffness Matrix K:
Matrix Elements >>>
        2.00        -1.00         0.00         0.00       |
       -1.00         2.00        -1.00         0.00       |
        0.00        -1.00         2.00        -1.00       |
        0.00         0.00        -1.00         1.00       |
<<< Matrix Elements

Is symmetric: Yes

[Quick Analysis] Primary frequency using power_iteration():
[Expected Results]
  Expected primary eigenvalue: ~3.53 (largest eigenvalue)
  Expected primary frequency: sqrt(3.53) ≈ 1.88 rad/s

[Actual Results]
  Primary eigenvalue: 3.53 (Expected: ~3.53)
  Primary frequency: 1.88 rad/s (Expected: ~1.88 rad/s)
  Iterations: 13
  Error from expected: 0.00 [PASS]

[Complete Analysis] Full modal analysis using eigendecompose_jacobi():
[Expected Results]
  Expected eigenvalues (approximate): 3.53, 2.35, 1.00, 0.12
  Expected natural frequencies: 1.88, 1.53, 1.00, 0.35 rad/s
  Note: These are approximate values for the 4-DOF system

[Actual Results]
All eigenvalues (natural frequencies squared):
Matrix Elements >>>
        3.53       |
        1.00       |
        2.35       |
        0.12       |
<<< Matrix Elements

Natural frequencies (rad/s):
  Mode 0: 1.88 rad/s (Expected: ~1.88 rad/s) [PASS]
  Mode 1: 1.00 rad/s (Expected: ~1.00 rad/s) [PASS]
  Mode 2: 1.53 rad/s (Expected: ~1.53 rad/s) [PASS]
  Mode 3: 0.35 rad/s (Expected: ~0.35 rad/s) [PASS]
Mode shapes (eigenvectors):
Matrix Elements >>>
        0.43         0.58        -0.66         0.23       |
       -0.66         0.58         0.23         0.43       |
        0.58        -0.00         0.58         0.58       |
       -0.23        -0.58        -0.43         0.66       |
<<< Matrix Elements

Total iterations: 17

[Test 11.7] Edge Cases and Error Handling

[Test 11.7.1] 1x1 Matrix
Matrix: [5.0]
[Expected Results]
  Expected eigenvalue: 5.0 (the matrix element itself)
  Expected eigenvector: [1.0] (normalized)

[Actual Results]
Eigenvalue: 5.00 (Expected: 5.0)
Eigenvector:
Matrix Elements >>>
        1.00       |
<<< Matrix Elements

Error from expected: 0.00 [PASS]

[Test 11.7.2] Zero Matrix
[Error] Power iteration: computed vector norm too small.
Status: Error (Expected)

[Test 11.7.3] Identity Matrix
Matrix (3x3 Identity):
Matrix Elements >>>
        1.00         0.00         0.00       |
        0.00         1.00         0.00       |
        0.00         0.00         1.00       |
<<< Matrix Elements


[Expected Results]
  Expected eigenvalues: 1.0, 1.0, 1.0 (all eigenvalues are 1)
  Expected eigenvectors: Any orthonormal basis (e.g., standard basis vectors)
  Expected iterations: 1 (should converge immediately)

[Actual Results]
Eigenvalues (should all be 1.0):
Matrix Elements >>>
        1.00       |
        1.00       |
        1.00       |
<<< Matrix Elements

Eigenvectors:
Matrix Elements >>>
        1.00         0.00         0.00       |
        0.00         1.00         0.00       |
        0.00         0.00         1.00       |
<<< Matrix Elements

Iterations: 1 (Expected: 1)
All eigenvalues = 1.0: [PASS]

[Test 11.8] Performance Test for SHM Applications

[Test 11.8.1] Power Iteration Performance (Real-time SHM)
[Performance] Power Iteration (3x3 matrix): 89.00 us

[Test 11.8.2] Jacobi Method Performance
[Performance] Jacobi Decomposition (3x3 symmetric matrix): 128.00 us

[Test 11.8.3] QR Method Performance
[Performance] QR Decomposition (3x3 general matrix): 554.00 us

[Eigenvalue Decomposition Tests Complete]
============ [tiny_matrix_test end] ============
```

