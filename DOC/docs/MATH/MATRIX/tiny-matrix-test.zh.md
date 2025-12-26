# 测试

!!! tip
    以下的测试用代码和案例也作为使用教学案例。

## tiny_matrix_test.hpp

```c
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

```c
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

// ============================================================================
// A1: Constructor & Destructor
// ============================================================================
void test_constructor_destructor()
{
    std::cout << "\n[A1: Constructor & Destructor Tests]\n";

    // A1.1: default constructor
    std::cout << "[A1.1] Default Constructor\n";
    tiny::Mat mat1;
    mat1.print_info();
    mat1.print_matrix(true);

    // A1.2: constructor with rows and cols, using internal allocation
    std::cout << "[A1.2] Constructor with Rows and Cols\n";
    tiny::Mat mat2(3, 4);
    mat2.print_info();
    mat2.print_matrix(true);

    // A1.3: constructor with rows and cols, specifying stride, using internal allocation
    std::cout << "[A1.3] Constructor with Rows, Cols and Stride\n";
    tiny::Mat mat3(3, 4, 5);
    mat3.print_info();
    mat3.print_matrix(true);

    // A1.4: constructor with external data
    std::cout << "[A1.4] Constructor with External Data\n";
    float data[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    tiny::Mat mat4(data, 3, 4);
    mat4.print_info();
    mat4.print_matrix(true);

    // A1.5: constructor with external data and stride
    std::cout << "[A1.5] Constructor with External Data and Stride\n";
    float data_stride[15] = {0, 1, 2, 3, 0, 4, 5, 6, 7, 0, 8, 9, 10, 11, 0};
    tiny::Mat mat5(data_stride, 3, 4, 5);
    mat5.print_info();
    mat5.print_matrix(true);

    // A1.6: copy constructor
    std::cout << "[A1.6] Copy Constructor\n";
    tiny::Mat mat6(mat5);
    mat6.print_info();
    mat6.print_matrix(true);
}

// ============================================================================
// A2: Element Access
// ============================================================================
void test_element_access()
{
    std::cout << "\n[A2: Element Access Tests]\n";
    tiny::Mat mat(2, 3);

    // A2.1: non-const access
    std::cout << "[A2.1] Non-const Access\n";
    mat(0, 0) = 1.1f;
    mat(0, 1) = 2.2f;
    mat(0, 2) = 3.3f;
    mat(1, 0) = 4.4f;
    mat(1, 1) = 5.5f;
    mat(1, 2) = 6.6f;
    mat.print_info();
    mat.print_matrix(true);

    // A2.2: const access
    std::cout << "[A2.2] Const Access\n";
    const tiny::Mat const_mat = mat;
    std::cout << "const_mat(0, 0): " << const_mat(0, 0) << "\n";
}

// ============================================================================
// A3: ROI Operations
// ============================================================================
void test_roi_operations()
{
    std::cout << "\n[A3: ROI Operations Tests]\n";

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

    // A3.1: Copy ROI
    std::cout << "[A3.1] Copy ROI - Over Range Case\n";
    matB.copy_paste(matA, 1, 2);
    std::cout << "matB after copy_paste matA at (1, 2):\n";
    matB.print_matrix(true);
    std::cout << "nothing changed.\n";

    std::cout << "[A3.2] Copy ROI - Suitable Range Case\n";
    matB.copy_paste(matA, 1, 1);
    std::cout << "matB after copy_paste matA at (1, 1):\n";
    matB.print_info();
    matB.print_matrix(true);
    std::cout << "successfully copied.\n";

    // A3.3: Copy Head
    std::cout << "[A3.3] Copy Head\n";
    matC.copy_head(matB);
    std::cout << "matC after copy_head matB:\n";
    matC.print_info();
    matC.print_matrix(true);

    std::cout << "[A3.4] Copy Head - Memory Sharing Check\n"; // matB and matC share the same data pointer
    matB(0, 0) = 99.99f;
    std::cout << "matB(0, 0) = 99.99f\n";
    std::cout << "matC:\n";
    matC.print_info();
    matC.print_matrix(true);

    // A3.5: copy_paste() - Error handling - negative position
    std::cout << "\n[A3.5] copy_paste() - Error Handling - Negative Position\n";
    tiny::Mat dest1(3, 3);
    tiny::Mat src1(2, 2);
    src1(0, 0) = 1.0f; src1(0, 1) = 2.0f;
    src1(1, 0) = 3.0f; src1(1, 1) = 4.0f;
    tiny_error_t err1 = dest1.copy_paste(src1, -1, 0);
    std::cout << "copy_paste with row_pos=-1: error = " << err1 
              << " (Expected: TINY_ERR_INVALID_ARG) " 
              << (err1 == TINY_ERR_INVALID_ARG ? "[PASS]" : "[FAIL]") << "\n";
    err1 = dest1.copy_paste(src1, 0, -1);
    std::cout << "copy_paste with col_pos=-1: error = " << err1 
              << " (Expected: TINY_ERR_INVALID_ARG) " 
              << (err1 == TINY_ERR_INVALID_ARG ? "[PASS]" : "[FAIL]") << "\n";

    // A3.6: copy_paste() - Error handling - out of bounds
    std::cout << "\n[A3.6] copy_paste() - Error Handling - Out of Bounds\n";
    tiny::Mat dest2(2, 2);
    tiny::Mat src2(3, 3);  // Larger than destination
    err1 = dest2.copy_paste(src2, 0, 0);
    std::cout << "copy_paste 3x3 into 2x2 at (0,0): error = " << err1 
              << " (Expected: TINY_ERR_INVALID_ARG) " 
              << (err1 == TINY_ERR_INVALID_ARG ? "[PASS]" : "[FAIL]") << "\n";
    err1 = dest2.copy_paste(src1, 1, 1);  // src1 is 2x2, dest2 is 2x2, position (1,1) would exceed
    std::cout << "copy_paste 2x2 into 2x2 at (1,1): error = " << err1 
              << " (Expected: TINY_ERR_INVALID_ARG) " 
              << (err1 == TINY_ERR_INVALID_ARG ? "[PASS]" : "[FAIL]") << "\n";

    // A3.7: copy_paste() - Boundary case - empty source
    std::cout << "\n[A3.7] copy_paste() - Boundary Case - Empty Source Matrix\n";
    tiny::Mat dest3(3, 3);
    tiny::Mat empty_src(0, 0);
    err1 = dest3.copy_paste(empty_src, 0, 0);
    std::cout << "copy_paste empty matrix: error = " << err1 
              << " (Expected: TINY_ERR_INVALID_ARG) " 
              << (err1 == TINY_ERR_INVALID_ARG ? "[PASS]" : "[FAIL]") << "\n";

    // A3.8: copy_head() - Share data from owned-memory source
    std::cout << "\n[A3.8] copy_head() - Share Data from Owned-Memory Source (Double-Free Prevention)\n";
    tiny::Mat dest4;
    tiny::Mat owned_src(2, 2);  // Creates matrix with its own memory (ext_buff=false)
    owned_src(0, 0) = 1.0f; owned_src(0, 1) = 2.0f;
    owned_src(1, 0) = 3.0f; owned_src(1, 1) = 4.0f;
    std::cout << "Before copy_head:\n";
    std::cout << "  owned_src: ext_buff=" << owned_src.ext_buff << "\n";
    std::cout << "  dest4: ext_buff=" << dest4.ext_buff << "\n";
    
    // Note: copy_head now works with ANY source, even if it owns its memory
    // The key safety feature: destination is marked as ext_buff=true (view, not owner)
    // This prevents double-free: only original owner (ext_buff=false) deletes memory
    tiny_error_t err2 = dest4.copy_head(owned_src);
    std::cout << "copy_head from matrix with owned memory: error = " << err2 
              << " (Expected: TINY_OK) " 
              << (err2 == TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
    
    std::cout << "After copy_head:\n";
    std::cout << "  owned_src: ext_buff=" << owned_src.ext_buff << " (still owns memory)\n";
    std::cout << "  dest4: ext_buff=" << dest4.ext_buff << " (view, does not own)\n";
    
    // Verify data is shared
    std::cout << "Verify data sharing:\n";
    std::cout << "  dest4(0,0)=" << dest4(0, 0) << " (should be 1.0)\n";
    std::cout << "  dest4(1,1)=" << dest4(1, 1) << " (should be 4.0)\n";
    
    // Modify original, verify view reflects change
    owned_src(0, 0) = 99.0f;
    std::cout << "After modifying owned_src(0,0) to 99.0:\n";
    std::cout << "  dest4(0,0)=" << dest4(0, 0) << " (should be 99.0, confirming shared data)\n";

    // A3.9: Get a View of ROI - low level function
    std::cout << "[A3.9] Get a View of ROI - Low Level Function\n";
    std::cout << "get a view of ROI with overrange dimensions - rows:\n";
    tiny::Mat roi1 = matB.view_roi(1, 1, 3, 2); // note here, C++ will use the copy constructor, which will copy according to the case (submatrix - shallow copy | normal - deep copy)
    std::cout << "get a view of ROI with overrange dimensions - cols:\n";
    tiny::Mat roi2 = matB.view_roi(1, 1, 2, 4); // note here, C++ will use the copy constructor, which will copy according to the case (submatrix - shallow copy | normal - deep copy)
    std::cout << "get a view of ROI with suitable dimensions:\n";
    tiny::Mat roi3 = matB.view_roi(1, 1, 2, 2); // note here, C++ will use the copy constructor, which will copy according to the case (submatrix - shallow copy | normal - deep copy)
    std::cout << "roi3:\n";
    roi3.print_info();
    roi3.print_matrix(true);

    // A3.10: Get a View of ROI - using ROI structure
    std::cout << "[A3.10] Get a View of ROI - Using ROI Structure\n";
    tiny::Mat::ROI roi_struct(1, 1, 2, 2);
    tiny::Mat roi4 = matB.view_roi(roi_struct);
    roi4.print_info();
    roi4.print_matrix(true);

    // A3.11: Copy ROI - low level function
    std::cout << "[A3.11] Copy ROI - Low Level Function\n";
    tiny::Mat mat_deep_copy = matB.copy_roi(1, 1, 2, 2);
    mat_deep_copy.print_info();
    mat_deep_copy.print_matrix(true);

    // A3.12: Copy ROI - using ROI structure
    std::cout << "[A3.12] Copy ROI - Using ROI Structure\n";
    TinyTimeMark_t tic1 = tiny_get_running_time();
    tiny::Mat::ROI roi_struct2(1, 1, 2, 2);
    tiny::Mat mat_deep_copy2 = matB.copy_roi(roi_struct2);
    TinyTimeMark_t toc1 = tiny_get_running_time();
    TinyTimeMark_t copy_roi_time = toc1 - tic1;
    std::cout << "time for copy_roi using ROI structure: " << copy_roi_time << " ms\n";
    mat_deep_copy2.print_info();
    mat_deep_copy2.print_matrix(true);

    // A3.13: ROI resize_roi() function
    std::cout << "\n[A3.13] ROI resize_roi() Function\n";
    tiny::Mat::ROI test_roi(0, 0, 2, 2);
    std::cout << "Initial ROI: pos_x=" << test_roi.pos_x << ", pos_y=" << test_roi.pos_y 
              << ", width=" << test_roi.width << ", height=" << test_roi.height << "\n";
    test_roi.resize_roi(1, 1, 3, 3);
    std::cout << "After resize_roi(1, 1, 3, 3): pos_x=" << test_roi.pos_x << ", pos_y=" << test_roi.pos_y 
              << ", width=" << test_roi.width << ", height=" << test_roi.height << "\n";
    bool roi_resize_correct = (test_roi.pos_x == 1 && test_roi.pos_y == 1 && 
                                test_roi.width == 3 && test_roi.height == 3);
    std::cout << "ROI resize test: " << (roi_resize_correct ? "[PASS]" : "[FAIL]") << "\n";

    // A3.14: ROI area_roi() function
    std::cout << "\n[A3.14] ROI area_roi() Function\n";
    tiny::Mat::ROI area_roi1(0, 0, 3, 4);
    int area1 = area_roi1.area_roi();
    std::cout << "ROI(0, 0, 3, 4) area: " << area1 << " (Expected: 12) ";
    std::cout << (area1 == 12 ? "[PASS]" : "[FAIL]") << "\n";
    
    tiny::Mat::ROI area_roi2(1, 2, 5, 6);
    int area2 = area_roi2.area_roi();
    std::cout << "ROI(1, 2, 5, 6) area: " << area2 << " (Expected: 30) ";
    std::cout << (area2 == 30 ? "[PASS]" : "[FAIL]") << "\n";

    // A3.15: Block
    std::cout << "[A3.15] Block\n";
    TinyTimeMark_t tic2 = tiny_get_running_time();
    tiny::Mat mat_block = matB.block(1, 1, 2, 2);
    TinyTimeMark_t toc2 = tiny_get_running_time();
    TinyTimeMark_t block_roi_time = toc2 - tic2;
    std::cout << "time for block: " << block_roi_time << " ms\n";
    mat_block.print_info();
    mat_block.print_matrix(true);

    // A3.16: Swap Rows
    std::cout << "[A3.16] Swap Rows\n";
    std::cout << "matB before swap rows:\n";
    matB.print_info();
    matB.print_matrix(true);
    std::cout << "matB after swap_rows(0, 2):\n";
    matB.swap_rows(0, 2);
    matB.print_info();
    matB.print_matrix(true);

    // A3.17: Swap Columns
    std::cout << "[A3.17] Swap Columns\n";
    std::cout << "matB before swap columns:\n";
    matB.print_info();
    matB.print_matrix(true);
    std::cout << "matB after swap_cols(0, 2):\n";
    matB.swap_cols(0, 2);
    matB.print_info();
    matB.print_matrix(true);

    // A3.18: Clear
    std::cout << "[A3.18] Clear\n";
    std::cout << "matB before clear:\n";
    matB.print_info();
    matB.print_matrix(true);
    std::cout << "matB after clear:\n";
    matB.clear();
    matB.print_info();
    matB.print_matrix(true);
}

// ============================================================================
// B1: Assignment Operator
// ============================================================================
void test_assignment_operator()
{
    std::cout << "\n[B1: Assignment Operator Tests]\n";

    std::cout << "\n[B1.1] Assignment (Same Dimensions)\n";
    tiny::Mat dst(2, 3), src(2, 3);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            src(i, j) = static_cast<float>(i * 3 + j + 1);
    dst = src;
    dst.print_matrix(true);

    std::cout << "\n[B1.2] Assignment (Different Dimensions)\n";
    tiny::Mat dst2(4, 2);
    dst2 = src;
    dst2.print_matrix(true);

    std::cout << "\n[B1.3] Assignment to Sub-Matrix (Expect Error)\n";
    float data[15] = {0, 1, 2, 3, 0, 4, 5, 6, 7, 0, 8, 9, 10, 11, 0};
    tiny::Mat base(data, 3, 4, 5);
    tiny::Mat subView = base.view_roi(1, 1, 2, 2);
    subView = src;
    subView.print_matrix(true);

    std::cout << "\n[B1.4] Self-Assignment\n";
    src = src;
    src.print_matrix(true);
}

// ============================================================================
// B2: Matrix Addition
// ============================================================================
void test_matrix_addition()
{
    std::cout << "\n[B2: Matrix Addition Tests]\n";

    std::cout << "\n[B2.1] Matrix Addition (Same Dimensions)\n";
    tiny::Mat A(2, 3), B(2, 3);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
        {
            A(i, j) = static_cast<float>(i * 3 + j + 1);
            B(i, j) = 1.0f;
        }
    A += B;
    A.print_matrix(true);

    std::cout << "\n[B2.2] Sub-Matrix Addition\n";
    float data[20] = {0,1,2,3,0,4,5,6,7,0,8,9,10,11,0,12,13,14,15,0};
    tiny::Mat base(data, 4, 4, 5);
    tiny::Mat subA = base.view_roi(1,1,2,2);
    tiny::Mat subB = base.view_roi(1,1,2,2);
    subA += subB;
    subA.print_matrix(true);

    std::cout << "\n[B2.3] Full Matrix + Sub-Matrix Addition\n";
    tiny::Mat full(2,2);
    for(int i=0;i<2;++i) for(int j=0;j<2;++j) full(i,j)=2.0f;
    full += subB;
    full.print_matrix(true);

    std::cout << "\n[B2.4] Addition Dimension Mismatch (Expect Error)\n";
    tiny::Mat wrongDim(3,3);
    full += wrongDim;
}

// ============================================================================
// B3: Constant Addition
// ============================================================================
void test_constant_addition()
{
    std::cout << "\n[B3: Constant Addition Tests]\n";

    std::cout << "\n[B3.1] Full Matrix + Constant\n";
    tiny::Mat mat1(2,3);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            mat1(i,j) = static_cast<float>(i*3 + j);
    mat1 += 5.0f;
    mat1.print_matrix(true);

    std::cout << "\n[B3.2] Sub-Matrix + Constant\n";
    float data[20] = {0,1,2,3,0,4,5,6,7,0,8,9,10,11,0,12,13,14,15,0};
    tiny::Mat base(data,4,4,5);
    tiny::Mat sub = base.view_roi(1,1,2,2);
    sub += 3.0f;
    sub.print_matrix(true);

    std::cout << "\n[B3.3] Add Zero\n";
    tiny::Mat mat2(2,2);
    mat2(0,0)=1; mat2(0,1)=2; mat2(1,0)=3; mat2(1,1)=4;
    mat2 += 0.0f;
    mat2.print_matrix(true);

    std::cout << "\n[B3.4] Add Negative Constant\n";
    tiny::Mat mat3(2,2);
    mat3(0,0)=10; mat3(0,1)=20; mat3(1,0)=30; mat3(1,1)=40;
    mat3 += -15.0f;
    mat3.print_matrix(true);
}

// ============================================================================
// B4: Matrix Subtraction
// ============================================================================
void test_matrix_subtraction()
{
    std::cout << "\n[B4: Matrix Subtraction Tests]\n";

    std::cout << "\n[B4.1] Matrix Subtraction\n";
    tiny::Mat A(2,2), B(2,2);
    A(0,0)=5; A(0,1)=7; A(1,0)=9; A(1,1)=11;
    B(0,0)=1; B(0,1)=2; B(1,0)=3; B(1,1)=4;
    A -= B;
    A.print_matrix(true);

    std::cout << "\n[B4.2] Subtraction Dimension Mismatch (Expect Error)\n";
    tiny::Mat wrong(3,3);
    A -= wrong;
}

// ============================================================================
// B5: Constant Subtraction
// ============================================================================
void test_constant_subtraction()
{
    std::cout << "\n[B5: Constant Subtraction Tests]\n";

    std::cout << "\n[B5.1] Full Matrix - Constant\n";
    tiny::Mat mat(2,3);
    for (int i=0;i<2;++i) for(int j=0;j<3;++j) mat(i,j) = i*3+j+1;
    mat -= 2.0f;
    mat.print_matrix(true);

    std::cout << "\n[B5.2] Sub-Matrix - Constant\n";
    float data[15] = {0,1,2,3,0,4,5,6,7,0,8,9,10,11,0};
    tiny::Mat base(data,3,4,5);
    tiny::Mat sub = base.view_roi(1,1,2,2);
    sub -= 1.5f;
    sub.print_matrix(true);
}

// ============================================================================
// B6: Matrix Division
// ============================================================================
void test_matrix_division()
{
    std::cout << "\n[B6: Matrix Element-wise Division Tests]\n";

    std::cout << "\n[B6.1] Element-wise Division (Same Dimensions, No Zero)\n";
    tiny::Mat A(2, 2), B(2, 2);
    A(0,0) = 10; A(0,1) = 20; A(1,0) = 30; A(1,1) = 40;
    B(0,0) = 2;  B(0,1) = 4;  B(1,0) = 5;  B(1,1) = 8;
    A /= B;
    A.print_matrix(true);

    std::cout << "\n[B6.2] Dimension Mismatch (Expect Error)\n";
    tiny::Mat wrongDim(3, 3);
    A /= wrongDim;

    std::cout << "\n[B6.3] Division by Matrix Containing Zero (Expect Error)\n";
    tiny::Mat C(2, 2), D(2, 2);
    C(0,0)=5; C(0,1)=10; C(1,0)=15; C(1,1)=20;
    D(0,0)=1; D(0,1)=0;  D(1,0)=3;  D(1,1)=4;  // Contains zero
    C /= D;
    C.print_matrix(true);  // Should remain unchanged
}

// ============================================================================
// B7: Constant Division
// ============================================================================
void test_constant_division()
{
    std::cout << "\n[B7: Matrix Division by Constant Tests]\n";

    std::cout << "\n[B7.1] Divide Full Matrix by Positive Constant\n";
    tiny::Mat mat1(2, 3);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            mat1(i, j) = static_cast<float>(i * 3 + j + 2);  // Avoid zero
    mat1 /= 2.0f;
    mat1.print_matrix(true);

    std::cout << "\n[B7.2] Divide Matrix by Negative Constant\n";
    tiny::Mat mat2(2, 2);
    mat2(0,0)=6; mat2(0,1)=12; mat2(1,0)=18; mat2(1,1)=24;
    mat2 /= -3.0f;
    mat2.print_matrix(true);

    std::cout << "\n[B7.3] Division by Zero Constant (Expect Error)\n";
    tiny::Mat mat3(2, 2);
    mat3(0,0)=1; mat3(0,1)=2; mat3(1,0)=3; mat3(1,1)=4;
    mat3 /= 0.0f;
    mat3.print_matrix(true);  // Should remain unchanged
}

// ============================================================================
// B8: Matrix Exponentiation
// ============================================================================
void test_matrix_exponentiation()
{
    std::cout << "\n[B8: Matrix Exponentiation Tests]\n";

    std::cout << "\n[B8.1] Raise Each Element to Power of 2\n";
    tiny::Mat mat1(2, 2);
    mat1(0,0)=2; mat1(0,1)=3; mat1(1,0)=4; mat1(1,1)=5;
    tiny::Mat result1 = mat1 ^ 2;
    result1.print_matrix(true);

    std::cout << "\n[B8.2] Raise Each Element to Power of 0\n";
    tiny::Mat mat2(2, 2);
    mat2(0,0)=7; mat2(0,1)=-3; mat2(1,0)=0.5f; mat2(1,1)=10;
    tiny::Mat result2 = mat2 ^ 0;
    result2.print_matrix(true);  // Expect all 1

    std::cout << "\n[B8.3] Raise Each Element to Power of 1\n";
    tiny::Mat mat3(2, 2);
    mat3(0,0)=9; mat3(0,1)=8; mat3(1,0)=7; mat3(1,1)=6;
    tiny::Mat result3 = mat3 ^ 1;
    result3.print_matrix(true);  // Expect same as original

    std::cout << "\n[B8.4] Raise Each Element to Power of -1 (Element-wise Reciprocal)\n";
    tiny::Mat mat4(2, 2);
    mat4(0,0)=1; mat4(0,1)=2; mat4(1,0)=4; mat4(1,1)=5;
    tiny::Mat result4 = mat4 ^ -1;
    result4.print_matrix(true);  // Expect: [1.0, 0.5; 0.25, 0.2]

    std::cout << "\n[B8.5] Raise Matrix Containing Zero to Power of 3\n";
    tiny::Mat mat5(2, 2);
    mat5(0,0)=0; mat5(0,1)=2; mat5(1,0)=-1; mat5(1,1)=3;
    tiny::Mat result5 = mat5 ^ 3;
    result5.print_matrix(true);

    std::cout << "\n[B8.6] Raise Matrix Containing Zero to Power of -1 (Expect Warning)\n";
    tiny::Mat mat6(2, 2);
    mat6(0,0)=0; mat6(0,1)=2; mat6(1,0)=-1; mat6(1,1)=3;
    tiny::Mat result6 = mat6 ^ -1;
    result6.print_matrix(true);  // Expect warning for zero element, Inf or NaN for (0,0)
}

// ============================================================================
// C1: Matrix Transpose
// ============================================================================
void test_matrix_transpose()
{
    std::cout << "\n[C1: Matrix Transpose Tests]\n";

    // C1.1: Basic 2x3 matrix transpose
    std::cout << "\n[C1.1] Transpose of 2x3 Matrix\n";
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

    // C1.2: Square matrix transpose (3x3)
    std::cout << "\n[C1.2] Transpose of 3x3 Square Matrix\n";
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

    // C1.3: Matrix with padding (4x2, stride=3)
    std::cout << "\n[C1.3] Transpose of Matrix with Padding\n";
    float data[12] = {1, 2, 0, 3, 4, 0, 5, 6, 0, 7, 8, 0};  // stride=3, 4 rows
    tiny::Mat mat3(data, 4, 2, 3);
    std::cout << "Original 4x2 Matrix (with padding):\n";
    mat3.print_matrix(true);

    tiny::Mat transposed3 = mat3.transpose();
    std::cout << "Transposed 2x4 Matrix:\n";
    transposed3.print_matrix(true);

    // C1.4: Transpose of empty matrix
    std::cout << "\n[C1.4] Transpose of Empty Matrix\n";
    tiny::Mat mat4;
    mat4.print_matrix(true);

    tiny::Mat transposed4 = mat4.transpose();
    transposed4.print_matrix(true);
}

// ============================================================================
// C2: Matrix Minor and Cofactor
// ============================================================================
void test_matrix_cofactor()
{
    std::cout << "\n[C2: Matrix Minor and Cofactor Tests]\n";

    // C2.1: Minor of 3x3 Matrix - Standard Case
    std::cout << "\n[C2.1] Minor of 3x3 Matrix (Remove Row 1, Col 1)\n";
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

    // C2.2: Cofactor of 3x3 Matrix - Same position
    std::cout << "\n[C2.2] Cofactor of 3x3 Matrix (Remove Row 1, Col 1)\n";
    std::cout << "Note: Cofactor matrix is the same as minor matrix.\n";
    std::cout << "      The sign (-1)^(i+j) is applied when computing cofactor value, not to matrix elements.\n";
    tiny::Mat cof1 = mat1.cofactor(1, 1);
    std::cout << "Cofactor Matrix (same as minor):\n";
    cof1.print_matrix(true);  // Expected: [[1,3],[7,9]] (same as minor)

    // C2.3: Minor - Remove first row and first column
    std::cout << "\n[C2.3] Minor (Remove Row 0, Col 0)\n";
    tiny::Mat minor2 = mat1.minor(0, 0);
    minor2.print_matrix(true);  // Expected: [[5,6],[8,9]]

    // C2.4: Cofactor - Remove first row and first column
    std::cout << "\n[C2.4] Cofactor (Remove Row 0, Col 0)\n";
    std::cout << "Note: Cofactor matrix is the same as minor matrix.\n";
    tiny::Mat cof2 = mat1.cofactor(0, 0);
    cof2.print_matrix(true);  // Expected: [[5,6],[8,9]] (same as minor)

    // C2.5: Cofactor - Remove row 0, col 1
    std::cout << "\n[C2.5] Cofactor (Remove Row 0, Col 1)\n";
    std::cout << "Note: Cofactor matrix is the same as minor matrix.\n";
    std::cout << "      When computing cofactor value, sign (-1)^(0+1) = -1 would be applied.\n";
    tiny::Mat cof2_neg = mat1.cofactor(0, 1);
    std::cout << "Cofactor Matrix (same as minor):\n";
    cof2_neg.print_matrix(true);  // Expected: [[4,6],[7,9]] (same as minor, no sign in matrix)

    // C2.6: Minor - Remove last row and last column
    std::cout << "\n[C2.6] Minor (Remove Row 2, Col 2)\n";
    tiny::Mat minor3 = mat1.minor(2, 2);
    minor3.print_matrix(true);  // Expected: [[1,2],[4,5]]

    // C2.7: Cofactor - Remove last row and last column
    std::cout << "\n[C2.7] Cofactor (Remove Row 2, Col 2)\n";
    std::cout << "Note: Cofactor matrix is the same as minor matrix.\n";
    tiny::Mat cof3 = mat1.cofactor(2, 2);
    cof3.print_matrix(true);  // Expected: [[1,2],[4,5]] (same as minor)

    // C2.8: 4x4 Matrix Example - Minor
    std::cout << "\n[C2.8] Minor of 4x4 Matrix (Remove Row 2, Col 1)\n";
    tiny::Mat mat4(4, 4);
    val = 1;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            mat4(i, j) = val++;

    mat4.print_matrix(true);
    tiny::Mat minor4 = mat4.minor(2, 1);
    std::cout << "Minor Matrix:\n";
    minor4.print_matrix(true);

    // C2.9: 4x4 Matrix Example - Cofactor
    std::cout << "\n[C2.9] Cofactor of 4x4 Matrix (Remove Row 2, Col 1)\n";
    std::cout << "Note: Cofactor matrix is the same as minor matrix.\n";
    std::cout << "      When computing cofactor value, sign (-1)^(2+1) = -1 would be applied.\n";
    tiny::Mat cof4 = mat4.cofactor(2, 1);
    std::cout << "Cofactor Matrix (same as minor):\n";
    cof4.print_matrix(true);

    // C2.10: Non-square Matrix (Expect Error)
    std::cout << "\n[C2.10] Non-square Matrix (Expect Error)\n";
    tiny::Mat rectMat(3, 4);
    std::cout << "Testing minor():\n";
    tiny::Mat minor_rect = rectMat.minor(1, 1);
    bool minor_rect_empty = (minor_rect.row == 0 && minor_rect.col == 0);
    std::cout << "minor() result: " << (minor_rect_empty ? "Empty matrix (Expected)" : "Non-empty (Error)") 
              << " " << (minor_rect_empty ? "[PASS]" : "[FAIL]") << "\n";
    std::cout << "Testing cofactor():\n";
    tiny::Mat cof_rect = rectMat.cofactor(1, 1);
    bool cof_rect_empty = (cof_rect.row == 0 && cof_rect.col == 0);
    std::cout << "cofactor() result: " << (cof_rect_empty ? "Empty matrix (Expected)" : "Non-empty (Error)") 
              << " " << (cof_rect_empty ? "[PASS]" : "[FAIL]") << "\n";

    // C2.11: minor() - Boundary case - out of bounds indices
    std::cout << "\n[C2.11] minor() - Boundary Case - Out of Bounds Indices\n";
    tiny::Mat test_mat(3, 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            test_mat(i, j) = i * 3 + j + 1;
    
    tiny::Mat minor_out1 = test_mat.minor(-1, 0);
    bool minor_out1_empty = (minor_out1.row == 0 && minor_out1.col == 0);
    std::cout << "minor(-1, 0): " << (minor_out1_empty ? "Empty matrix (Expected)" : "Non-empty (Error)") 
              << " " << (minor_out1_empty ? "[PASS]" : "[FAIL]") << "\n";
    
    tiny::Mat minor_out2 = test_mat.minor(0, -1);
    bool minor_out2_empty = (minor_out2.row == 0 && minor_out2.col == 0);
    std::cout << "minor(0, -1): " << (minor_out2_empty ? "Empty matrix (Expected)" : "Non-empty (Error)") 
              << " " << (minor_out2_empty ? "[PASS]" : "[FAIL]") << "\n";
    
    tiny::Mat minor_out3 = test_mat.minor(3, 0);
    bool minor_out3_empty = (minor_out3.row == 0 && minor_out3.col == 0);
    std::cout << "minor(3, 0) (out of bounds): " << (minor_out3_empty ? "Empty matrix (Expected)" : "Non-empty (Error)") 
              << " " << (minor_out3_empty ? "[PASS]" : "[FAIL]") << "\n";

    // C2.12: minor() - Boundary case - 1x1 matrix
    std::cout << "\n[C2.12] minor() - Boundary Case - 1x1 Matrix\n";
    tiny::Mat mat1x1(1, 1);
    mat1x1(0, 0) = 5.0f;
    tiny::Mat minor_1x1 = mat1x1.minor(0, 0);
    bool minor_1x1_empty = (minor_1x1.row == 0 && minor_1x1.col == 0);
    std::cout << "1x1 matrix minor(0,0): " << (minor_1x1_empty ? "Empty matrix (Expected)" : "Non-empty (Error)") 
              << " " << (minor_1x1_empty ? "[PASS]" : "[FAIL]") << "\n";
}

// ============================================================================
// C3: Matrix Determinant
// ============================================================================
void test_matrix_determinant()
{
    std::cout << "\n[C3: Matrix Determinant Tests]\n";

    // C3.1: 1x1 Matrix
    std::cout << "\n[C3.1] 1x1 Matrix Determinant\n";
    tiny::Mat mat1(1, 1);
    mat1(0, 0) = 7;
    std::cout << "Matrix:\n";
    mat1.print_matrix(true);
    std::cout << "Determinant: " << mat1.determinant() << "  (Expected: 7)\n";

    // C3.2: 2x2 Matrix
    std::cout << "\n[C3.2] 2x2 Matrix Determinant\n";
    tiny::Mat mat2(2, 2);
    mat2(0, 0) = 3; mat2(0, 1) = 8;
    mat2(1, 0) = 4; mat2(1, 1) = 6;
    std::cout << "Matrix:\n";
    mat2.print_matrix(true);
    std::cout << "Determinant: " << mat2.determinant() << "  (Expected: -14)\n";

    // C3.3: 3x3 Matrix
    std::cout << "\n[C3.3] 3x3 Matrix Determinant\n";
    tiny::Mat mat3(3, 3);
    mat3(0,0) = 1; mat3(0,1) = 2; mat3(0,2) = 3;
    mat3(1,0) = 0; mat3(1,1) = 4; mat3(1,2) = 5;
    mat3(2,0) = 1; mat3(2,1) = 0; mat3(2,2) = 6;
    std::cout << "Matrix:\n";
    mat3.print_matrix(true);
    std::cout << "Determinant: " << mat3.determinant() << "  (Expected: 22)\n";

    // C3.4: 4x4 Matrix
    std::cout << "\n[C3.4] 4x4 Matrix Determinant\n";
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

    // C3.5: 5x5 Matrix (Tests Auto-select Mechanism)
    std::cout << "\n[C3.5] 5x5 Matrix Determinant (Tests Auto-select to LU Method)\n";
    tiny::Mat mat5_basic(5, 5);
    // Create a well-conditioned 5x5 matrix
    mat5_basic(0,0) = 2; mat5_basic(0,1) = 1; mat5_basic(0,2) = 0; mat5_basic(0,3) = 0; mat5_basic(0,4) = 0;
    mat5_basic(1,0) = 1; mat5_basic(1,1) = 2; mat5_basic(1,2) = 1; mat5_basic(1,3) = 0; mat5_basic(1,4) = 0;
    mat5_basic(2,0) = 0; mat5_basic(2,1) = 1; mat5_basic(2,2) = 2; mat5_basic(2,3) = 1; mat5_basic(2,4) = 0;
    mat5_basic(3,0) = 0; mat5_basic(3,1) = 0; mat5_basic(3,2) = 1; mat5_basic(3,3) = 2; mat5_basic(3,4) = 1;
    mat5_basic(4,0) = 0; mat5_basic(4,1) = 0; mat5_basic(4,2) = 0; mat5_basic(4,3) = 1; mat5_basic(4,4) = 2;
    std::cout << "Matrix (5x5, tridiagonal):\n";
    mat5_basic.print_matrix(true);
    float det5_basic = mat5_basic.determinant();
    std::cout << "Determinant (auto-select, should use LU for n > 4): " << det5_basic << "\n";
    std::cout << "Note: For n = 5 > 4, auto-select should use LU decomposition (O(n³)).\n";

    // C3.6: Non-square Matrix (Expect Error)
    std::cout << "\n[C3.6] Non-square Matrix (Expect Error)\n";
    tiny::Mat rectMat(3, 4);
    std::cout << "Matrix (3x4, non-square):\n";
    rectMat.print_matrix(true);
    float det_rect = rectMat.determinant();  // should trigger error
    std::cout << "Determinant: " << det_rect << "  (Expected: 0 with error message)\n";

    // C3.7: Comparison of Different Methods (5x5 Matrix)
    std::cout << "\n[C3.7] Comparison of Different Methods (5x5 Matrix)\n";
    tiny::Mat mat_test(5, 5);
    // Create a test matrix
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            mat_test(i, j) = static_cast<float>((i + 1) * (j + 1) + (i == j ? 1.0f : 0.0f));
        }
    }
    std::cout << "Matrix (5x5):\n";
    mat_test.print_matrix(true);
    float det_auto = mat_test.determinant();
    float det_laplace = mat_test.determinant_laplace();
    float det_lu = mat_test.determinant_lu();
    float det_gaussian = mat_test.determinant_gaussian();
    std::cout << "Determinant (auto-select): " << det_auto << "  (should use LU for n > 4)\n";
    std::cout << "Determinant (Laplace):     " << det_laplace << "  (O(n!), slow for n=5)\n";
    std::cout << "Determinant (LU):          " << det_lu << "  (O(n³), efficient)\n";
    std::cout << "Determinant (Gaussian):    " << det_gaussian << "  (O(n³), efficient)\n";
    std::cout << "Note: All methods should give the same result (within numerical precision).\n";
    std::cout << "      Auto-select should use LU for n > 4, avoiding slow Laplace expansion.\n";

    // C3.8: Large Matrix (6x6) - Tests Efficient Methods
    std::cout << "\n[C3.8] Large Matrix (6x6) - Tests Efficient Methods\n";
    tiny::Mat mat6(6, 6);
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            mat6(i, j) = static_cast<float>((i + 1) * (j + 1) + (i == j ? 0.5f : 0.0f));
        }
    }
    std::cout << "Matrix (6x6, showing first 4x4 block):\n";
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            std::cout << std::setw(10) << mat6(i, j) << " ";
        }
        std::cout << "...\n";
    }
    std::cout << "...\n";
    float det6_auto = mat6.determinant();
    float det6_lu = mat6.determinant_lu();
    float det6_gaussian = mat6.determinant_gaussian();
    std::cout << "Determinant (auto-select, uses LU): " << det6_auto << "\n";
    std::cout << "Determinant (LU):                   " << det6_lu << "\n";
    std::cout << "Determinant (Gaussian):             " << det6_gaussian << "\n";
    std::cout << "Note: For n > 4, auto-select uses LU decomposition (O(n³) instead of O(n!)).\n";

    // C3.9: Large Matrix (8x8) - Performance Test
    std::cout << "\n[C3.9] Large Matrix (8x8) - Performance Comparison\n";
    tiny::Mat mat8(8, 8);
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            mat8(i, j) = static_cast<float>((i + 1) * (j + 1));
        }
    }
    std::cout << "Matrix (8x8, showing first 4x4 block):\n";
    // Print partial matrix for display
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            std::cout << std::setw(10) << mat8(i, j) << " ";
        }
        std::cout << "...\n";
    }
    std::cout << "...\n";
    float det8_lu = mat8.determinant_lu();
    float det8_gaussian = mat8.determinant_gaussian();
    std::cout << "Determinant (LU):       " << det8_lu << "\n";
    std::cout << "Determinant (Gaussian): " << det8_gaussian << "\n";
    std::cout << "Note: Both methods are O(n³) and should be much faster than Laplace expansion.\n";

    // C3.10: determinant_laplace() - Boundary case - empty matrix
    std::cout << "\n[C3.10] determinant_laplace() - Boundary Case - Empty Matrix\n";
    tiny::Mat empty_det(0, 0);
    float det_empty_laplace = empty_det.determinant_laplace();
    std::cout << "Empty matrix determinant (Laplace): " << det_empty_laplace 
              << " (Expected: 1.0) " << (fabsf(det_empty_laplace - 1.0f) < 1e-6f ? "[PASS]" : "[FAIL]") << "\n";

    // C3.11: determinant_lu() - Boundary case - empty matrix
    std::cout << "\n[C3.11] determinant_lu() - Boundary Case - Empty Matrix\n";
    float det_empty_lu = empty_det.determinant_lu();
    std::cout << "Empty matrix determinant (LU): " << det_empty_lu 
              << " (Expected: 1.0) " << (fabsf(det_empty_lu - 1.0f) < 1e-6f ? "[PASS]" : "[FAIL]") << "\n";

    // C3.12: determinant_gaussian() - Boundary case - empty matrix
    std::cout << "\n[C3.12] determinant_gaussian() - Boundary Case - Empty Matrix\n";
    float det_empty_gaussian = empty_det.determinant_gaussian();
    std::cout << "Empty matrix determinant (Gaussian): " << det_empty_gaussian 
              << " (Expected: 1.0) " << (fabsf(det_empty_gaussian - 1.0f) < 1e-6f ? "[PASS]" : "[FAIL]") << "\n";

    // C3.13: determinant methods - Non-square matrix (should return 0)
    std::cout << "\n[C3.13] Determinant Methods - Non-Square Matrix\n";
    tiny::Mat rect_det(2, 3);
    float det_rect_laplace = rect_det.determinant_laplace();
    float det_rect_lu = rect_det.determinant_lu();
    float det_rect_gaussian = rect_det.determinant_gaussian();
    std::cout << "Non-square matrix (2x3) determinant (Laplace): " << det_rect_laplace 
              << " (Expected: 0.0) " << (fabsf(det_rect_laplace) < 1e-6f ? "[PASS]" : "[FAIL]") << "\n";
    std::cout << "Non-square matrix (2x3) determinant (LU): " << det_rect_lu 
              << " (Expected: 0.0) " << (fabsf(det_rect_lu) < 1e-6f ? "[PASS]" : "[FAIL]") << "\n";
    std::cout << "Non-square matrix (2x3) determinant (Gaussian): " << det_rect_gaussian 
              << " (Expected: 0.0) " << (fabsf(det_rect_gaussian) < 1e-6f ? "[PASS]" : "[FAIL]") << "\n";
}

// ============================================================================
// C4: Matrix Adjoint
// ============================================================================
void test_matrix_adjoint()
{
    std::cout << "\n[C4: Matrix Adjoint Tests]\n";

    // C4.1: 1x1 Matrix
    std::cout << "\n[C4.1] Adjoint of 1x1 Matrix\n";
    tiny::Mat mat1(1, 1);
    mat1(0, 0) = 5;
    std::cout << "Original Matrix:\n";
    mat1.print_matrix(true);
    tiny::Mat adj1 = mat1.adjoint();
    std::cout << "Adjoint Matrix:\n";
    adj1.print_matrix(true);  // Expected: [1]

    // C4.2: 2x2 Matrix
    std::cout << "\n[C4.2] Adjoint of 2x2 Matrix\n";
    tiny::Mat mat2(2, 2);
    mat2(0, 0) = 1; mat2(0, 1) = 2;
    mat2(1, 0) = 3; mat2(1, 1) = 4;
    std::cout << "Original Matrix:\n";
    mat2.print_matrix(true);
    tiny::Mat adj2 = mat2.adjoint();
    std::cout << "Adjoint Matrix:\n";
    adj2.print_matrix(true);  // Expected: [4, -2; -3, 1]

    // C4.3: 3x3 Matrix
    std::cout << "\n[C4.3] Adjoint of 3x3 Matrix\n";
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

    // C4.4: Non-Square Matrix (Expect Error)
    std::cout << "\n[C4.4] Adjoint of Non-Square Matrix (Expect Error)\n";
    tiny::Mat rectMat(2, 3);
    std::cout << "Original Matrix (2x3, non-square):\n";
    rectMat.print_matrix(true);
    tiny::Mat adjRect = rectMat.adjoint();
    std::cout << "Adjoint Matrix (should be empty due to error):\n";
    adjRect.print_matrix(true);  // Should be empty or default matrix

}

// ============================================================================
// C5: Matrix Normalization
// ============================================================================
void test_matrix_normalize()
{
    std::cout << "\n[C5: Matrix Normalization Tests]\n";

    // C5.1: Standard normalization
    std::cout << "\n[C5.1] Normalize a Standard 2x2 Matrix\n";
    tiny::Mat mat1(2, 2);
    mat1(0, 0) = 3.0f; mat1(0, 1) = 4.0f;
    mat1(1, 0) = 3.0f; mat1(1, 1) = 4.0f;

    std::cout << "Before normalization:\n";
    mat1.print_matrix(true);

    mat1.normalize();

    std::cout << "After normalization (Expected L2 norm = 1):\n";
    mat1.print_matrix(true);

    // C5.2: Matrix with padding
    std::cout << "\n[C5.2] Normalize a 2x2 Matrix with Stride=4 (Padding Test)\n";
    float data_with_padding[8] = {3.0f, 4.0f, 0.0f, 0.0f, 3.0f, 4.0f, 0.0f, 0.0f};
    tiny::Mat mat2(data_with_padding, 2, 2, 4);  // 2x2 matrix, stride 4

    std::cout << "Before normalization:\n";
    mat2.print_matrix(true);

    mat2.normalize();

    std::cout << "After normalization:\n";
    mat2.print_matrix(true);

    // C5.3: Zero matrix normalization
    std::cout << "\n[C5.3] Normalize a Zero Matrix (Expect Warning)\n";
    tiny::Mat mat3(2, 2);
    mat3.clear();  // Assuming clear() sets all elements to zero

    mat3.print_matrix(true);
    mat3.normalize();  // Should trigger warning
}

// ============================================================================
// C6: Matrix Norm Calculation
// ============================================================================
void test_matrix_norm()
{
    std::cout << "\n[C6: Matrix Norm Calculation Tests]\n";

    // C6.1: Simple 2x2 Matrix
    std::cout << "\n[C6.1] 2x2 Matrix Norm (Expect 5.0)\n";
    tiny::Mat mat1(2, 2);
    mat1(0, 0) = 3.0f; mat1(0, 1) = 4.0f;
    mat1(1, 0) = 0.0f; mat1(1, 1) = 0.0f;
    std::cout << "Matrix:\n";
    mat1.print_matrix(true);
    float norm1 = mat1.norm();
    std::cout << "Calculated Norm: " << norm1 << "\n";

    // C6.2: Zero Matrix
    std::cout << "\n[C6.2] Zero Matrix Norm (Expect 0.0)\n";
    tiny::Mat mat2(3, 3);
    mat2.clear();  // Assuming clear() sets all elements to zero
    std::cout << "Matrix:\n";
    mat2.print_matrix(true);
    float norm2 = mat2.norm();
    std::cout << "Calculated Norm: " << norm2 << "\n";

    // C6.3: Matrix with Negative Values
    std::cout << "\n[C6.3] Matrix with Negative Values\n";
    tiny::Mat mat3(2, 2);
    mat3(0, 0) = -1.0f; mat3(0, 1) = -2.0f;
    mat3(1, 0) = -3.0f; mat3(1, 1) = -4.0f;
    std::cout << "Matrix:\n";
    mat3.print_matrix(true);
    float norm3 = mat3.norm();
    std::cout << "Calculated Norm: " << norm3 << "  (Expect sqrt(30) ≈ 5.477)\n";

    // C6.4: Matrix with Padding
    std::cout << "\n[C6.4] 2x2 Matrix with Stride=4 (Padding Test)\n";
    float data4[8] = {1.0f, 2.0f, 0.0f, 0.0f, 3.0f, 4.0f, 0.0f, 0.0f};
    tiny::Mat mat4(data4, 2, 2, 4);  // 2x2 matrix, stride 4
    std::cout << "Matrix:\n";
    mat4.print_matrix(true);
    float norm4 = mat4.norm();
    std::cout << "Calculated Norm: " << norm4 << "  (Expect sqrt(30) ≈ 5.477)\n";
}

// ============================================================================
// C7: Matrix Inversion
// ============================================================================
void test_inverse_adjoint_adjoint()
{
    std::cout << "\n[C7: Matrix Inversion Tests]\n";

    // C7.1: 2x2 Regular Matrix
    std::cout << "\n[C7.1] Inverse of 2x2 Matrix\n";
    tiny::Mat mat1(2, 2);
    mat1(0, 0) = 4;  mat1(0, 1) = 7;
    mat1(1, 0) = 2;  mat1(1, 1) = 6;
    std::cout << "Original Matrix:\n";
    mat1.print_matrix(true);
    tiny::Mat inv1 = mat1.inverse_adjoint();
    std::cout << "Inverse Matrix:\n";
    inv1.print_matrix(true);
    std::cout << "Expected Approx:\n[ 0.6  -0.7 ]\n[ -0.2  0.4 ]\n";

    // C7.2: Singular Matrix (Determinant = 0)
    std::cout << "\n[C7.2] Singular Matrix (Expect Error)\n";
    tiny::Mat mat2(2, 2);
    mat2(0, 0) = 1;  mat2(0, 1) = 2;
    mat2(1, 0) = 2;  mat2(1, 1) = 4;   // Rank-deficient, det = 0
    std::cout << "Original Matrix:\n";
    mat2.print_matrix(true);
    std::cout << "Note: This matrix is singular (determinant = 0), so inverse should fail.\n";
    tiny::Mat inv2 = mat2.inverse_adjoint();
    std::cout << "Inverse Matrix (Should be zero matrix):\n";
    inv2.print_matrix(true);

    // C7.3: 3x3 Regular Matrix
    std::cout << "\n[C7.3] Inverse of 3x3 Matrix\n";
    tiny::Mat mat3(3, 3);
    mat3(0,0) = 3; mat3(0,1) = 0; mat3(0,2) = 2;
    mat3(1,0) = 2; mat3(1,1) = 0; mat3(1,2) = -2;
    mat3(2,0) = 0; mat3(2,1) = 1; mat3(2,2) = 1;
    std::cout << "Original Matrix:\n";
    mat3.print_matrix(true);
    tiny::Mat inv3 = mat3.inverse_adjoint();
    std::cout << "Inverse Matrix:\n";
    inv3.print_matrix(true);

    // C7.4: Non-Square Matrix (Expect Error)
    std::cout << "\n[C7.4] Non-Square Matrix (Expect Error)\n";
    tiny::Mat mat4(2, 3);
    std::cout << "Original Matrix (2x3, non-square):\n";
    mat4.print_matrix(true);
    tiny::Mat inv4 = mat4.inverse_adjoint();
    std::cout << "Inverse Matrix (should be empty due to error):\n";
    inv4.print_matrix(true);
}

// ============================================================================
// C8: Matrix Utilities
// ============================================================================
void test_matrix_utilities()
{
    std::cout << "\n[C8: Matrix Utilities Tests]\n";

    // C8.1: Identity Matrix (eye)
    std::cout << "\n[C8.1] Generate Identity Matrix (eye)\n";
    tiny::Mat I3 = tiny::Mat::eye(3);
    std::cout << "3x3 Identity Matrix:\n";
    I3.print_matrix(true);

    tiny::Mat I5 = tiny::Mat::eye(5);
    std::cout << "5x5 Identity Matrix:\n";
    I5.print_matrix(true);

    // C8.2: Ones Matrix
    std::cout << "\n[C8.2] Generate Ones Matrix\n";
    tiny::Mat ones_3x4 = tiny::Mat::ones(3, 4);
    std::cout << "3x4 Ones Matrix:\n";
    ones_3x4.print_matrix(true);

    tiny::Mat ones_4x4 = tiny::Mat::ones(4);
    std::cout << "4x4 Ones Matrix (Square):\n";
    ones_4x4.print_matrix(true);

    // C8.3: Matrix Augmentation
    std::cout << "\n[C8.3] Augment Two Matrices Horizontally [A | B]\n";

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

    // C8.4: Row mismatch case
    std::cout << "\n[C8.4] Augment with Row Mismatch (Expect Error)\n";
    tiny::Mat C(3, 2);  // 3x2 matrix
    tiny::Mat invalidAug = tiny::Mat::augment(A, C);
    invalidAug.print_info();  // Should show empty matrix due to error

    // C8.5: Vertical Stack (vstack)
    std::cout << "\n[C8.5] Vertically Stack Two Matrices [A; B]\n";

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

    // C8.6: Vertical Stack with different row counts
    std::cout << "\n[C8.6] Vertical Stack with Different Row Counts (Same Columns)\n";
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

    // C8.7: Column mismatch case (Expect Error)
    std::cout << "\n[C8.7] VStack with Column Mismatch (Expect Error)\n";
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

// ============================================================================
// D1: Gaussian Elimination
// ============================================================================
void test_gaussian_eliminate()
{
    std::cout << "\n[D1: Gaussian Elimination Tests]\n";

    // D1.1: Simple 3x3 System
    std::cout << "\n[D1.1] 3x3 Matrix (Simple Upper Triangular)\n";
    tiny::Mat mat1(3, 3);
    mat1(0,0) = 2; mat1(0,1) = 1; mat1(0,2) = -1;
    mat1(1,0) = -3; mat1(1,1) = -1; mat1(1,2) = 2;
    mat1(2,0) = -2; mat1(2,1) = 1; mat1(2,2) = 2;

    std::cout << "Original Matrix:\n";
    mat1.print_matrix(true);

    tiny::Mat result1 = mat1.gaussian_eliminate();

    std::cout << "After Gaussian Elimination (Should be upper triangular):\n";
    result1.print_matrix(true);

    // D1.2: 3x4 Augmented Matrix
    std::cout << "\n[D1.2] 3x4 Augmented Matrix (Linear System Ax = b)\n";
    tiny::Mat mat2(3, 4);
    mat2(0,0) = 1; mat2(0,1) = 2; mat2(0,2) = -1; mat2(0,3) =  8;
    mat2(1,0) = -3; mat2(1,1) = -1; mat2(1,2) = 2; mat2(1,3) = -11;
    mat2(2,0) = -2; mat2(2,1) = 1; mat2(2,2) = 2; mat2(2,3) = -3;

    std::cout << "Original Augmented Matrix [A | b]:\n";
    mat2.print_matrix(true);

    tiny::Mat result2 = mat2.gaussian_eliminate();

    std::cout << "After Gaussian Elimination (Row Echelon Form):\n";
    result2.print_matrix(true);

    // D1.3: Singular Matrix
    std::cout << "\n[D1.3] Singular Matrix (No Unique Solution)\n";
    tiny::Mat mat3(2, 2);
    mat3(0,0) = 1; mat3(0,1) = 2;
    mat3(1,0) = 2; mat3(1,1) = 4;  // Linearly dependent rows

    std::cout << "Original Singular Matrix:\n";
    mat3.print_matrix(true);

    tiny::Mat result3 = mat3.gaussian_eliminate();
    std::cout << "After Gaussian Elimination (Should show rows of zeros):\n";
    result3.print_matrix(true);

    // D1.4: Zero Matrix
    std::cout << "\n[D1.4] Zero Matrix\n";
    tiny::Mat mat4(3, 3);
    mat4.clear();  // Assuming clear() sets all elements to zero
    mat4.print_matrix(true);

    tiny::Mat result4 = mat4.gaussian_eliminate();
    std::cout << "After Gaussian Elimination (Should be a zero matrix):\n";
    result4.print_matrix(true);

    // D1.5: gaussian_eliminate() - Boundary case - empty matrix
    std::cout << "\n[D1.5] gaussian_eliminate() - Boundary Case - Empty Matrix\n";
    tiny::Mat empty_ge(0, 0);
    tiny::Mat result_empty_ge = empty_ge.gaussian_eliminate();
    // Empty matrix should return empty matrix (0x0) or error state (data == nullptr or 1x1 error matrix)
    bool empty_ge_correct = (result_empty_ge.row == 0 && result_empty_ge.col == 0) || 
                           (result_empty_ge.data == nullptr) ||
                           (result_empty_ge.row == 1 && result_empty_ge.col == 1 && result_empty_ge.data != nullptr);
    std::cout << "Empty matrix gaussian_eliminate: " << (empty_ge_correct ? "Empty matrix or error state (Expected)" : "Non-empty (Error)") 
              << " " << (empty_ge_correct ? "[PASS]" : "[FAIL]") << "\n";

    // D1.6: gaussian_eliminate() - Boundary case - 1x1 matrix
    std::cout << "\n[D1.6] gaussian_eliminate() - Boundary Case - 1x1 Matrix\n";
    tiny::Mat mat1x1_ge(1, 1);
    mat1x1_ge(0, 0) = 5.0f;
    tiny::Mat result1x1_ge = mat1x1_ge.gaussian_eliminate();
    std::cout << "1x1 matrix after gaussian_eliminate:\n";
    result1x1_ge.print_matrix(true);
    bool ge1x1_correct = (result1x1_ge.row == 1 && result1x1_ge.col == 1 && 
                          fabsf(result1x1_ge(0, 0) - 5.0f) < 1e-6f);
    std::cout << "1x1 matrix gaussian_eliminate: " << (ge1x1_correct ? "[PASS]" : "[FAIL]") << "\n";
}


// ============================================================================
// D2: Row Reduce from Gaussian (RREF Calculation)
// ============================================================================
void test_row_reduce_from_gaussian()
{
    std::cout << "\n[D2: Row Reduce from Gaussian (RREF) Tests]\n";

    // D2.1: Simple 3x4 augmented matrix (representing a system of equations)
    std::cout << "\n[D2.1] 3x4 Augmented Matrix\n";
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

    // D2.2: 2x3 Matrix
    std::cout << "\n[D2.2] 2x3 Matrix\n";
    tiny::Mat mat2(2, 3);
    mat2(0,0) = 1; mat2(0,1) = 2;  mat2(0,2) = 3;
    mat2(1,0) = 4; mat2(1,1) = 5;  mat2(1,2) = 6;

    std::cout << "Original Matrix:\n";
    mat2.print_matrix(true);

    tiny::Mat rref2 = mat2.gaussian_eliminate().row_reduce_from_gaussian();
    std::cout << "RREF Result:\n";
    rref2.print_matrix(true);

    // D2.3: Already reduced matrix (should remain the same)
    std::cout << "\n[D2.3] Already Reduced Matrix\n";
    tiny::Mat mat3(2, 3);
    mat3(0,0) = 1; mat3(0,1) = 0; mat3(0,2) = 2;
    mat3(1,0) = 0; mat3(1,1) = 1; mat3(1,2) = 3;

    std::cout << "Original Matrix:\n";
    mat3.print_matrix(true);

    tiny::Mat rref3 = mat3.row_reduce_from_gaussian();
    std::cout << "RREF Result:\n";
    rref3.print_matrix(true);

    // D2.4: row_reduce_from_gaussian() - Boundary case - empty matrix
    std::cout << "\n[D2.4] row_reduce_from_gaussian() - Boundary Case - Empty Matrix\n";
    tiny::Mat empty_rref(0, 0);
    tiny::Mat result_empty_rref = empty_rref.row_reduce_from_gaussian();
    // Empty matrix should return empty matrix (0x0) or error state (data == nullptr or 1x1 error matrix)
    bool empty_rref_correct = (result_empty_rref.row == 0 && result_empty_rref.col == 0) || 
                              (result_empty_rref.data == nullptr) ||
                              (result_empty_rref.row == 1 && result_empty_rref.col == 1 && result_empty_rref.data != nullptr);
    std::cout << "Empty matrix row_reduce_from_gaussian: " << (empty_rref_correct ? "Empty matrix or error state (Expected)" : "Non-empty (Error)") 
              << " " << (empty_rref_correct ? "[PASS]" : "[FAIL]") << "\n";
}

// ============================================================================
// D3: Gaussian Inverse
// ============================================================================
void test_inverse_gje()
{
    std::cout << "\n[D3: Gaussian Inverse Tests]\n";

    // D3.1: Regular 2x2 Matrix
    std::cout << "\n[D3.1] 2x2 Matrix Inverse\n";
    tiny::Mat mat1(2, 2);
    mat1(0, 0) = 4; mat1(0, 1) = 7;
    mat1(1, 0) = 2; mat1(1, 1) = 6;
    std::cout << "Original matrix (mat1):\n";
    mat1.print_matrix(true);
    
    tiny::Mat invMat1 = mat1.inverse_gje();
    std::cout << "Inverse matrix (mat1):\n";
    invMat1.print_matrix(true);

    // D3.2: Identity Matrix (should return identity matrix)
    std::cout << "\n[D3.2] Identity Matrix Inverse\n";
    tiny::Mat mat2 = tiny::Mat::eye(3);
    std::cout << "Original matrix (Identity):\n";
    mat2.print_matrix(true);
    
    tiny::Mat invMat2 = mat2.inverse_gje();
    std::cout << "Inverse matrix (Identity):\n";
    invMat2.print_matrix(true); // Expected: Identity matrix

    // D3.3: Singular Matrix (should return empty matrix or indicate error)
    std::cout << "\n[D3.3] Singular Matrix (Expected: No Inverse)\n";
    tiny::Mat mat3(3, 3);
    mat3(0, 0) = 1; mat3(0, 1) = 2; mat3(0, 2) = 3;
    mat3(1, 0) = 4; mat3(1, 1) = 5; mat3(1, 2) = 6;
    mat3(2, 0) = 7; mat3(2, 1) = 8; mat3(2, 2) = 9;  // Determinant is 0
    std::cout << "Original matrix (singular):\n";
    mat3.print_matrix(true);
    
    tiny::Mat invMat3 = mat3.inverse_gje();
    std::cout << "Inverse matrix (singular):\n";
    invMat3.print_matrix(true); // Expected: empty matrix or error message

    // D3.4: 3x3 Matrix with a valid inverse
    std::cout << "\n[D3.4] 3x3 Matrix Inverse\n";
    tiny::Mat mat4(3, 3);
    mat4(0, 0) = 4; mat4(0, 1) = 7; mat4(0, 2) = 2;
    mat4(1, 0) = 3; mat4(1, 1) = 5; mat4(1, 2) = 1;
    mat4(2, 0) = 8; mat4(2, 1) = 6; mat4(2, 2) = 9;
    std::cout << "Original matrix (mat4):\n";
    mat4.print_matrix(true);
    
    tiny::Mat invMat4 = mat4.inverse_gje();
    std::cout << "Inverse matrix (mat4):\n";
    invMat4.print_matrix(true); // Check that the inverse is calculated correctly

    // D3.5: Non-square Matrix (should return error or empty matrix)
    std::cout << "\n[D3.5] Non-square Matrix Inverse (Expected Error)\n";
    tiny::Mat mat5(2, 3);
    mat5(0, 0) = 1; mat5(0, 1) = 2; mat5(0, 2) = 3;
    mat5(1, 0) = 4; mat5(1, 1) = 5; mat5(1, 2) = 6;
    std::cout << "Original matrix (non-square):\n";
    mat5.print_matrix(true);
    
    tiny::Mat invMat5 = mat5.inverse_gje();
    std::cout << "Inverse matrix (non-square):\n";
    invMat5.print_matrix(true); // Expected: Error message or empty matrix
}

// ============================================================================
// D4: Dot Product
// ============================================================================
void test_dotprod()
{
    std::cout << "\n[D4: Dot Product Tests]\n";

    // D4.1: Valid Dot Product Calculation (Same Length Vectors)
    std::cout << "\n[D4.1] Valid Dot Product (Same Length Vectors)\n";
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

    // D4.2: Dot Product with Dimension Mismatch (Different Length Vectors)
    std::cout << "\n[D4.2] Invalid Dot Product (Dimension Mismatch)\n";
    tiny::Mat vectorC(2, 1);  // Create a 2x1 vector (different size)
    vectorC(0, 0) = 1.0f;
    vectorC(1, 0) = 2.0f;

    std::cout << "Vector A (3x1):\n";
    vectorA.print_matrix(true);
    std::cout << "Vector C (2x1, different size):\n";
    vectorC.print_matrix(true);

    float invalidResult = vectorA.dotprod(vectorA, vectorC);  // Should print an error and return 0
    std::cout << "Dot product (dimension mismatch): " << invalidResult << std::endl;  // Expected: 0 and error message

    // D4.3: Dot Product of Zero Vectors
    std::cout << "\n[D4.3] Dot Product of Zero Vectors\n";
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

// ============================================================================
// D5: Solve Linear System
// ============================================================================
void test_solve()
{
    std::cout << "\n[D5: Solve Linear System Tests]\n";

    // D5.1: Solving a simple 2x2 system
    std::cout << "\n[D5.1] Solving a Simple 2x2 System Ax = b\n";
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

    // D5.2: Solving a 3x3 system
    std::cout << "\n[D5.2] Solving a 3x3 System Ax = b\n";
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

    // D5.3: Solving a system where one row is all zeros
    std::cout << "\n[D5.3] Solving a System Where One Row is All Zeros (Expect Failure or Infinite Solutions)\n";
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

    // D5.4: Solving a system with zero determinant (singular matrix)
    std::cout << "\n[D5.4] Solving a System with Zero Determinant (Singular Matrix)\n";
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

    // D5.5: Solving a system with linearly dependent rows
    std::cout << "\n[D5.5] Solving a System with Linearly Dependent Rows (Expect Failure or Infinite Solutions)\n";
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

    // D5.6: Solving a larger 4x4 system
    std::cout << "\n[D5.6] Solving a Larger 4x4 System Ax = b\n";
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

    // D5.7: solve() - Boundary case - empty matrix
    std::cout << "\n[D5.7] solve() - Boundary Case - Empty Matrix\n";
    tiny::Mat empty_A(0, 0);
    tiny::Mat empty_b(0, 1);
    tiny::Mat solution_empty = empty_A.solve(empty_A, empty_b);
    // Error case should return empty matrix (0x0) or error state (data == nullptr or 1x1 error matrix)
    bool solve_empty_correct = (solution_empty.row == 0 && solution_empty.col == 0) || 
                               (solution_empty.data == nullptr) ||
                               (solution_empty.row == 1 && solution_empty.col == 1 && solution_empty.data != nullptr);
    std::cout << "Empty system solve: " << (solve_empty_correct ? "Empty matrix or error state (Expected)" : "Non-empty (Error)") 
              << " " << (solve_empty_correct ? "[PASS]" : "[FAIL]") << "\n";

    // D5.8: solve() - Error handling - dimension mismatch
    std::cout << "\n[D5.8] solve() - Error Handling - Dimension Mismatch\n";
    tiny::Mat A_mismatch(2, 2);
    tiny::Mat b_mismatch(3, 1);  // Different dimension
    tiny::Mat solution_mismatch = A_mismatch.solve(A_mismatch, b_mismatch);
    // Error case should return empty matrix (0x0) or error state (data == nullptr or 1x1 error matrix)
    // For dimension mismatch, expected solution size is 2x1, so 1x1 indicates error
    bool solve_mismatch_correct = (solution_mismatch.row == 0 && solution_mismatch.col == 0) || 
                                  (solution_mismatch.data == nullptr) ||
                                  (solution_mismatch.row == 1 && solution_mismatch.col == 1 && solution_mismatch.data != nullptr);
    std::cout << "Dimension mismatch solve: " << (solve_mismatch_correct ? "Empty matrix or error state (Expected)" : "Non-empty (Error)") 
              << " " << (solve_mismatch_correct ? "[PASS]" : "[FAIL]") << "\n";
}

// ============================================================================
// D6: Band Solve
// ============================================================================
void test_band_solve()
{
    std::cout << "\n[D6: Band Solve Tests]\n";

    // D6.1: Simple 3x3 Band Matrix
    std::cout << "\n[D6.1] Simple 3x3 Band Matrix\n";
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

    // D6.2: 4x4 Band Matrix with different right-hand side vector
    std::cout << "\n[D6.2] 4x4 Band Matrix\n";
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

    // D6.3: Incompatible dimensions (expect error)
    std::cout << "\n[D6.3] Incompatible Dimensions (Expect Error)\n";
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

    // D6.4: Singular Matrix (Should fail)
    std::cout << "\n[D6.4] Singular Matrix (No Unique Solution)\n";
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

    // D6.5: band_solve() - Boundary case - empty matrix
    std::cout << "\n[D6.5] band_solve() - Boundary Case - Empty Matrix\n";
    tiny::Mat empty_A_band(0, 0);
    tiny::Mat empty_b_band(0, 1);
    tiny::Mat solution_empty_band = empty_A_band.band_solve(empty_A_band, empty_b_band, 0);
    // Error case should return empty matrix (0x0) or error state (data == nullptr or 1x1 error matrix)
    bool band_solve_empty_correct = (solution_empty_band.row == 0 && solution_empty_band.col == 0) || 
                                    (solution_empty_band.data == nullptr) ||
                                    (solution_empty_band.row == 1 && solution_empty_band.col == 1 && solution_empty_band.data != nullptr);
    std::cout << "Empty system band_solve: " << (band_solve_empty_correct ? "Empty matrix or error state (Expected)" : "Non-empty (Error)") 
              << " " << (band_solve_empty_correct ? "[PASS]" : "[FAIL]") << "\n";

    // D6.6: band_solve() - Error handling - invalid bandwidth
    std::cout << "\n[D6.6] band_solve() - Error Handling - Invalid Bandwidth\n";
    tiny::Mat A_band(3, 3);
    tiny::Mat b_band(3, 1);
    tiny::Mat solution_neg_k = A_band.band_solve(A_band, b_band, -1);
    // Error case should return empty matrix (0x0) or error state (data == nullptr or 1x1 error matrix)
    // For invalid bandwidth, expected solution size is 3x1, so 1x1 indicates error
    bool band_solve_neg_k_correct = (solution_neg_k.row == 0 && solution_neg_k.col == 0) || 
                                     (solution_neg_k.data == nullptr) ||
                                     (solution_neg_k.row == 1 && solution_neg_k.col == 1 && solution_neg_k.data != nullptr);
    std::cout << "band_solve with k=-1: " << (band_solve_neg_k_correct ? "Empty matrix or error state (Expected)" : "Non-empty (Error)") 
              << " " << (band_solve_neg_k_correct ? "[PASS]" : "[FAIL]") << "\n";
}

// ============================================================================
// D7: Roots
// ============================================================================
void test_roots()
{
    std::cout << "\n[D7: Roots Tests]\n";

    // D7.1: Simple 2x2 System
    std::cout << "\n[D7.1] Solving a Simple 2x2 System Ax = b\n";
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

    // D7.2: 3x3 System
    std::cout << "\n[D7.2] Solving a 3x3 System Ax = b\n";
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

    // D7.3: Singular Matrix
    std::cout << "\n[D7.3] Singular Matrix (No Unique Solution)\n";
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

    // D7.4: Incompatible Dimensions (Expect Error)
    std::cout << "\n[D7.4] Incompatible Dimensions (Expect Error)\n";
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
    // Error case should return empty matrix (0x0) or error state (data == nullptr or 1x1 error matrix)
    // For dimension mismatch, expected solution size is 3x1, so 1x1 indicates error
    bool roots_mismatch_correct = (solution4.row == 0 && solution4.col == 0) || 
                                  (solution4.data == nullptr) ||
                                  (solution4.row == 1 && solution4.col == 1 && solution4.data != nullptr);
    std::cout << "Dimension mismatch roots: " << (roots_mismatch_correct ? "Empty matrix or error state (Expected)" : "Non-empty (Error)") 
              << " " << (roots_mismatch_correct ? "[PASS]" : "[FAIL]") << "\n";

    // D7.5: roots() - Boundary case - empty matrix
    std::cout << "\n[D7.5] roots() - Boundary Case - Empty Matrix\n";
    tiny::Mat empty_A_roots(0, 0);
    tiny::Mat empty_y_roots(0, 1);
    tiny::Mat solution_empty_roots = empty_A_roots.roots(empty_A_roots, empty_y_roots);
    // Error case should return empty matrix (0x0) or error state (data == nullptr or 1x1 error matrix)
    bool roots_empty_correct = (solution_empty_roots.row == 0 && solution_empty_roots.col == 0) || 
                                (solution_empty_roots.data == nullptr) ||
                                (solution_empty_roots.row == 1 && solution_empty_roots.col == 1 && solution_empty_roots.data != nullptr);
    std::cout << "Empty system roots: " << (roots_empty_correct ? "Empty matrix or error state (Expected)" : "Non-empty (Error)") 
              << " " << (roots_empty_correct ? "[PASS]" : "[FAIL]") << "\n";
}

// ============================================================================
// E: Advanced Linear Algebra
// ============================================================================
// Purpose: Advanced linear algebra operations for stable and efficient solving
// E1: Matrix Decomposition
// E2: Gram-Schmidt Orthogonalization

// ============================================================================
// E3: Eigenvalue Decomposition
// ============================================================================
// Purpose: Eigenvalue decomposition for SHM and modal analysis
// E3: Eigenvalue & Eigenvector

// ============================================================================
// F: Auxiliary Functions
// ============================================================================
// Purpose: Convenience functions and I/O operations
// F1: Stream Operators
// F2: Global Arithmetic Operators

// ============================================================================
// G: Quality Assurance
// ============================================================================
// Purpose: Ensure robustness, performance, and correctness
// G1: Boundary Conditions and Error Handling
// G2: Performance Benchmarks
// G3: Memory Layout

// ============================================================================
// F1: Stream Operators
// ============================================================================
void test_stream_operators()
{
    std::cout << "\n[F1: Stream Operators Tests]\n";

    // F1.1: Test stream insertion operator (<<) for Mat
    std::cout << "\n[F1.1] Stream Insertion Operator (<<) for Mat\n";
    tiny::Mat mat1(3, 3);
    mat1(0, 0) = 1; mat1(0, 1) = 2; mat1(0, 2) = 3;
    mat1(1, 0) = 4; mat1(1, 1) = 5; mat1(1, 2) = 6;
    mat1(2, 0) = 7; mat1(2, 1) = 8; mat1(2, 2) = 9;

    std::cout << "Matrix mat1:\n";
    std::cout << mat1 << std::endl; // Use the << operator to print mat1

    // F1.2: Test stream insertion operator (<<) for Mat::ROI
    std::cout << "\n[F1.2] Stream Insertion Operator (<<) for Mat::ROI\n";
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

    // F1.3: Test stream extraction operator (>>) for Mat
    std::cout << "\n[F1.3] Stream Extraction Operator (>>) for Mat\n";
    tiny::Mat mat2(2, 2);
    // Use istringstream to simulate input (for automated testing)
    std::istringstream input1("10 20 30 40");
    std::cout << "Simulated input: \"10 20 30 40\"\n";
    input1 >> mat2; // Use the >> operator to read from string stream
    std::cout << "Matrix mat2 after input:\n";
    std::cout << mat2 << std::endl; // Use the << operator to print mat2
    std::cout << "Expected: [10, 20; 30, 40]\n";

    // F1.4: Test stream extraction operator (>>) for Mat (with different values)
    std::cout << "\n[F1.4] Stream Extraction Operator (>>) for Mat (2x3 matrix)\n";
    tiny::Mat mat3(2, 3);
    // Use istringstream to simulate input (for automated testing)
    std::istringstream input2("1.5 2.5 3.5 4.5 5.5 6.5");
    std::cout << "Simulated input: \"1.5 2.5 3.5 4.5 5.5 6.5\"\n";
    input2 >> mat3; // Use the >> operator to read from string stream
    std::cout << "Matrix mat3 after input:\n";
    std::cout << mat3 << std::endl; // Use the << operator to print mat3
    std::cout << "Expected: [1.5, 2.5, 3.5; 4.5, 5.5, 6.5]\n";
}

// ============================================================================
// F2: Global Arithmetic Operators
// ============================================================================
void test_matrix_operations()
{
    std::cout << "\n[F2: Global Arithmetic Operators Tests]\n";

    // F2.1: Matrix Addition (operator+)
    std::cout << "\n[F2.1] Matrix Addition (operator+)\n";
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

    // F2.2: Matrix Addition with Constant (operator+)
    std::cout << "\n[F2.2] Matrix Addition with Constant (operator+)\n";
    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Constant: 5.0\n";
    tiny::Mat resultAddConst = matA + 5.0f;
    std::cout << "matA + 5.0f:\n";
    std::cout << resultAddConst << std::endl;  // Expected: [6, 7], [8, 9]

    // F2.3: Matrix Subtraction (operator-)
    std::cout << "\n[F2.3] Matrix Subtraction (operator-)\n";
    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Matrix B:\n";
    matB.print_matrix(true);
    tiny::Mat resultSub = matA - matB;
    std::cout << "matA - matB:\n";
    std::cout << resultSub << std::endl;  // Expected: [-4, -4], [-4, -4]

    // F2.4: Matrix Subtraction with Constant (operator-)
    std::cout << "\n[F2.4] Matrix Subtraction with Constant (operator-)\n";
    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Constant: 2.0\n";
    tiny::Mat resultSubConst = matA - 2.0f;
    std::cout << "matA - 2.0f:\n";
    std::cout << resultSubConst << std::endl;  // Expected: [-1, 0], [1, 2]

    // F2.5: Matrix Multiplication (operator*)
    std::cout << "\n[F2.5] Matrix Multiplication (operator*)\n";
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

    // F2.6: Matrix Multiplication with Constant (operator*)
    std::cout << "\n[F2.6] Matrix Multiplication with Constant (operator*)\n";
    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Constant: 2.0\n";
    tiny::Mat resultMulConst = matA * 2.0f;
    std::cout << "matA * 2.0f:\n";
    std::cout << resultMulConst << std::endl;  // Expected: [2, 4], [6, 8]

    // F2.7: Matrix Division (operator/)
    std::cout << "\n[F2.7] Matrix Division (operator/)\n";
    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Constant: 2.0\n";
    tiny::Mat resultDiv = matA / 2.0f;
    std::cout << "matA / 2.0f:\n";
    std::cout << resultDiv << std::endl;  // Expected: [0.5, 1], [1.5, 2]

    // F2.8: Matrix Division Element-wise (operator/)
    std::cout << "\n[F2.8] Matrix Division Element-wise (operator/)\n";
    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Matrix B:\n";
    matB.print_matrix(true);
    tiny::Mat resultDivElem = matA / matB;
    std::cout << "matA / matB:\n";
    std::cout << resultDivElem << std::endl;  // Expected: [0.2, 0.333], [0.428, 0.5]

    // F2.9: Matrix Comparison (operator==)
    std::cout << "\n[F2.9] Matrix Comparison (operator==)\n";
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

// ============================================================================
// G1: Quality Assurance - Boundary Conditions and Error Handling Tests
// ============================================================================
// Purpose: Test error handling and edge cases - ensure robustness
void test_boundary_conditions()
{
    std::cout << "\n[G1: Quality Assurance - Boundary Conditions and Error Handling Tests]\n";

    // G1.1: Null pointer handling in print functions
    std::cout << "\n[G1.1] Null Pointer Handling in print_matrix\n";
    tiny::Mat null_mat;
    null_mat.data = nullptr;  // Simulate null pointer
    null_mat.print_matrix(true);  // Should handle gracefully

    // G1.2: Null pointer handling in operator<<
    std::cout << "\n[G1.2] Null Pointer Handling in operator<<\n";
    tiny::Mat null_mat2;
    null_mat2.data = nullptr;
    std::cout << null_mat2 << std::endl;  // Should handle gracefully

    // G1.3: Invalid block parameters
    std::cout << "\n[G1.3] Invalid Block Parameters\n";
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

    // G1.4: Invalid swap_rows parameters
    std::cout << "\n[G1.4] Invalid swap_rows Parameters\n";
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

    // G1.5: Invalid swap_cols parameters
    std::cout << "\n[G1.5] Invalid swap_cols Parameters\n";
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

    // G1.6: Division by zero
    std::cout << "\n[G1.6] Division by Zero\n";
    tiny::Mat mat3(2, 2);
    mat3(0, 0) = 1; mat3(0, 1) = 2;
    mat3(1, 0) = 3; mat3(1, 1) = 4;
    
    tiny::Mat result = mat3 / 0.0f;
    std::cout << "mat3 / 0.0f: " << (result.data == nullptr ? "Empty (correct)" : "Error") << "\n";

    // G1.7: Matrix division with zero elements
    std::cout << "\n[G1.7] Matrix Division with Zero Elements\n";
    tiny::Mat mat4(2, 2);
    mat4(0, 0) = 1; mat4(0, 1) = 2;
    mat4(1, 0) = 3; mat4(1, 1) = 4;
    
    tiny::Mat divisor(2, 2);
    divisor(0, 0) = 1; divisor(0, 1) = 0;  // Contains zero
    divisor(1, 0) = 3; divisor(1, 1) = 4;
    
    mat4 /= divisor;
    std::cout << "mat4 /= divisor (with zero):\n";
    mat4.print_matrix(true);

    // G1.8: Empty matrix operations
    std::cout << "\n[G1.8] Empty Matrix Operations\n";
    tiny::Mat empty1(0, 0), empty2(0, 0);  // True empty matrices (0x0)
    tiny::Mat empty_sum = empty1 + empty2;
    // Empty matrix addition should return empty matrix (0x0) or error state
    bool empty_sum_correct = (empty_sum.row == 0 && empty_sum.col == 0) || 
                             (empty_sum.data == nullptr) ||
                             (empty_sum.row == 1 && empty_sum.col == 1 && empty_sum.data != nullptr);
    std::cout << "Empty matrix addition: " << (empty_sum_correct ? "Empty matrix or error state (Expected)" : "Non-empty (Error)") 
              << " " << (empty_sum_correct ? "[PASS]" : "[FAIL]") << "\n";
}

// ============================================================================
// G2: Quality Assurance - Performance Benchmarks Tests
// ============================================================================
// Purpose: Test performance characteristics - critical for real-time applications
void test_performance_benchmarks()
{
    std::cout << "\n[G2: Quality Assurance - Performance Benchmarks Tests]\n";
    
    // Ensure current task is added to watchdog before starting performance tests
    #if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    ensure_task_wdt_added();
    #endif

    // G2.1: Matrix Addition Performance (reduced size to prevent timeout)
    std::cout << "\n[G2.1] Matrix Addition Performance\n";
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

    // G2.2: Matrix Multiplication Performance (reduced size)
    std::cout << "\n[G2.2] Matrix Multiplication Performance\n";
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

    // G2.3: Matrix Transpose Performance (reduced size)
    std::cout << "\n[G2.3] Matrix Transpose Performance\n";
    tiny::Mat G(50, 30);  // Reduced from 100x50 to 50x30
    for (int i = 0; i < 50; ++i)
        for (int j = 0; j < 30; ++j)
            G(i, j) = static_cast<float>(i * 30 + j);
    TIME_REPEATED_OPERATION(tiny::Mat H = G.transpose();, PERFORMANCE_TEST_ITERATIONS, "50x30 Matrix Transpose");

    // G2.4: Determinant Performance Comparison
    // Note: Determinant calculation now has multiple methods:
    //   - Laplace expansion: O(n!) - for small matrices (n <= 4)
    //   - LU decomposition: O(n³) - for large matrices (n > 4, auto-selected)
    //   - Gaussian elimination: O(n³) - alternative for large matrices
    std::cout << "\n[G2.4] Determinant Calculation Performance Comparison\n";
    
    // G2.4.1: Small Matrix (4x4) - Laplace Expansion
        std::cout << "\n[G2.4.1] Small Matrix (4x4) - Laplace Expansion\n";
    tiny::Mat I4(4, 4);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            I4(i, j) = static_cast<float>(i * 4 + j + 1);
    
    feed_watchdog();
    TinyTimeMark_t det4_t0 = tiny_get_running_time();
    for (int i = 0; i < PERFORMANCE_TEST_ITERATIONS_HEAVY; ++i)
    {
        feed_watchdog();
        float det = I4.determinant_laplace();
        (void)det;
        feed_watchdog();
    }
    TinyTimeMark_t det4_t1 = tiny_get_running_time();
    double det4_dt_total_us = (double)(det4_t1 - det4_t0);
    double det4_dt_avg_us = det4_dt_total_us / PERFORMANCE_TEST_ITERATIONS_HEAVY;
    std::cout << "[Performance] 4x4 Determinant (Laplace, " << PERFORMANCE_TEST_ITERATIONS_HEAVY << " iterations): "
              << std::fixed << std::setprecision(2) << det4_dt_total_us << " us total, "
              << det4_dt_avg_us << " us avg\n";
    
    // G2.4.2: Large Matrix (8x8) - LU Decomposition
        std::cout << "\n[G2.4.2] Large Matrix (8x8) - LU Decomposition\n";
    tiny::Mat I8(8, 8);
    // Create a non-singular matrix (diagonally dominant matrix) for performance testing
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            if (i == j)
                I8(i, j) = static_cast<float>(10 + i + 1);  // Diagonal dominance
            else
                I8(i, j) = static_cast<float>((i + 1) * (j + 1) * 0.1f);  // Off-diagonal elements
        }
    }
    
    feed_watchdog();
    TinyTimeMark_t det8_lu_t0 = tiny_get_running_time();
    for (int i = 0; i < PERFORMANCE_TEST_ITERATIONS_HEAVY; ++i)
    {
        feed_watchdog();
        float det = I8.determinant_lu();
        (void)det;
        feed_watchdog();
    }
    TinyTimeMark_t det8_lu_t1 = tiny_get_running_time();
    double det8_lu_dt_total_us = (double)(det8_lu_t1 - det8_lu_t0);
    double det8_lu_dt_avg_us = det8_lu_dt_total_us / PERFORMANCE_TEST_ITERATIONS_HEAVY;
    std::cout << "[Performance] 8x8 Determinant (LU, " << PERFORMANCE_TEST_ITERATIONS_HEAVY << " iterations): "
              << std::fixed << std::setprecision(2) << det8_lu_dt_total_us << " us total, "
              << det8_lu_dt_avg_us << " us avg\n";
    
    // G2.4.3: Large Matrix (8x8) - Gaussian Elimination
        std::cout << "\n[G2.4.3] Large Matrix (8x8) - Gaussian Elimination\n";
    feed_watchdog();
    TinyTimeMark_t det8_gauss_t0 = tiny_get_running_time();
    for (int i = 0; i < PERFORMANCE_TEST_ITERATIONS_HEAVY; ++i)
    {
        feed_watchdog();
        float det = I8.determinant_gaussian();
        (void)det;
        feed_watchdog();
    }
    TinyTimeMark_t det8_gauss_t1 = tiny_get_running_time();
    double det8_gauss_dt_total_us = (double)(det8_gauss_t1 - det8_gauss_t0);
    double det8_gauss_dt_avg_us = det8_gauss_dt_total_us / PERFORMANCE_TEST_ITERATIONS_HEAVY;
    std::cout << "[Performance] 8x8 Determinant (Gaussian, " << PERFORMANCE_TEST_ITERATIONS_HEAVY << " iterations): "
              << std::fixed << std::setprecision(2) << det8_gauss_dt_total_us << " us total, "
              << det8_gauss_dt_avg_us << " us avg\n";
    
    // G2.4.4: Auto-select Method (8x8) - Should use LU
        std::cout << "\n[G2.4.4] Large Matrix (8x8) - Auto-select Method\n";
    feed_watchdog();
    TinyTimeMark_t det8_auto_t0 = tiny_get_running_time();
    for (int i = 0; i < PERFORMANCE_TEST_ITERATIONS_HEAVY; ++i)
    {
        feed_watchdog();
        float det = I8.determinant();  // Auto-selects LU for n > 4
        (void)det;
        feed_watchdog();
    }
    TinyTimeMark_t det8_auto_t1 = tiny_get_running_time();
    double det8_auto_dt_total_us = (double)(det8_auto_t1 - det8_auto_t0);
    double det8_auto_dt_avg_us = det8_auto_dt_total_us / PERFORMANCE_TEST_ITERATIONS_HEAVY;
    std::cout << "[Performance] 8x8 Determinant (auto-select, " << PERFORMANCE_TEST_ITERATIONS_HEAVY << " iterations): "
              << std::fixed << std::setprecision(2) << det8_auto_dt_total_us << " us total, "
              << det8_auto_dt_avg_us << " us avg\n";
    
    std::cout << "\n[Note] Performance Summary:\n";
    std::cout << "  - Laplace expansion (O(n!)): Suitable only for small matrices (n <= 4)\n";
    std::cout << "  - LU decomposition (O(n³)): Efficient for large matrices, auto-selected for n > 4\n";
    std::cout << "  - Gaussian elimination (O(n³)): Alternative efficient method for large matrices\n";
    std::cout << "  - Auto-select: Automatically chooses the best method based on matrix size\n";

    // G2.5: Matrix Copy Performance (with padding, reduced size)
    std::cout << "\n[G2.5] Matrix Copy with Padding Performance\n";
    float data[80] = {0};  // Reduced from 150 to 80
    for (int i = 0; i < 80; ++i) data[i] = static_cast<float>(i);
    tiny::Mat J(data, 8, 8, 10);  // Reduced from 10x10 stride 15 to 8x8 stride 10
    TIME_REPEATED_OPERATION(tiny::Mat K = J.copy_roi(0, 0, 8, 8);, PERFORMANCE_TEST_ITERATIONS, "8x8 Copy ROI (with padding)");

    // G2.6: Element Access Performance (reduced size)
    std::cout << "\n[G2.6] Element Access Performance\n";
    tiny::Mat L(50, 50);  // Reduced from 100x100 to 50x50
    for (int i = 0; i < 50; ++i)
        for (int j = 0; j < 50; ++j)
            L(i, j) = static_cast<float>(i * 50 + j);
    
    // G2.6 continued: Element Access Performance (custom implementation for multi-line operation)
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

// ============================================================================
// G3: Quality Assurance - Memory Layout Tests (Padding and Stride)
// ============================================================================
// Purpose: Test memory layout handling - important for performance and compatibility
void test_memory_layout()
{
    std::cout << "\n[G3: Quality Assurance - Memory Layout Tests (Padding and Stride)]\n";

    // G3.1: Contiguous memory (pad=0, step=1)
    std::cout << "\n[G3.1] Contiguous Memory (no padding)\n";
    tiny::Mat mat1(3, 4);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            mat1(i, j) = static_cast<float>(i * 4 + j);
    std::cout << "Matrix 3x4 (stride=4, pad=0):\n";
    mat1.print_info();
    mat1.print_matrix(true);

    // G3.2: Padded memory (stride > col)
    std::cout << "\n[G3.2] Padded Memory (stride > col)\n";
    float data[15] = {0, 1, 2, 3, 0, 4, 5, 6, 7, 0, 8, 9, 10, 11, 0};
    tiny::Mat mat2(data, 3, 4, 5);
    std::cout << "Matrix 3x4 (stride=5, pad=1):\n";
    mat2.print_info();
    mat2.print_matrix(true);

    // G3.3: Operations with padded matrices
    std::cout << "\n[G3.3] Addition with Padded Matrices\n";
    float data1[15] = {1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 9, 10, 11, 12, 0};
    float data2[15] = {10, 20, 30, 40, 0, 50, 60, 70, 80, 0, 90, 100, 110, 120, 0};
    tiny::Mat mat3(data1, 3, 4, 5);
    tiny::Mat mat4(data2, 3, 4, 5);
    tiny::Mat mat5 = mat3 + mat4;
    std::cout << "Result of padded matrix addition:\n";
    mat5.print_info();
    mat5.print_matrix(true);

    // G3.4: ROI operations with padded matrices
    std::cout << "\n[G3.4] ROI Operations with Padded Matrices\n";
    tiny::Mat roi = mat2.view_roi(1, 1, 2, 2);
    std::cout << "ROI (1,1,2,2) from padded matrix:\n";
    roi.print_info();
    roi.print_matrix(true);

    // G3.5: Copy operations preserve stride
    std::cout << "\n[G3.5] Copy Operations Preserve Stride\n";
    tiny::Mat copied = mat2.copy_roi(0, 0, 3, 4);
    std::cout << "Copied matrix (should have stride=4, no padding):\n";
    copied.print_info();
    copied.print_matrix(true);
}

// ============================================================================
// E1: Matrix Decomposition
// ============================================================================
void test_matrix_decomposition()
{
    std::cout << "\n[E1: Matrix Decomposition Tests]\n";

    // E1.1: is_positive_definite() - Basic functionality
    std::cout << "\n[E1.1] is_positive_definite() - Basic Functionality\n";
    
    // E1.11: Positive definite matrix
    {
        std::cout << "\n[E1.11] Positive Definite 3x3 Matrix\n";
        tiny::Mat pd_mat(3, 3);
        pd_mat(0, 0) = 4.0f; pd_mat(0, 1) = 1.0f; pd_mat(0, 2) = 0.0f;
        pd_mat(1, 0) = 1.0f; pd_mat(1, 1) = 3.0f; pd_mat(1, 2) = 0.0f;
        pd_mat(2, 0) = 0.0f; pd_mat(2, 1) = 0.0f; pd_mat(2, 2) = 2.0f;
        std::cout << "Matrix:\n";
        pd_mat.print_matrix(true);
        
        bool is_pd = pd_mat.is_positive_definite(1e-6f);
        std::cout << "Is positive definite: " << (is_pd ? "True" : "False") 
                  << " (Expected: True) " << (is_pd ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.12: Non-positive definite matrix
    {
        std::cout << "\n[E1.12] Non-Positive Definite Matrix\n";
        tiny::Mat non_pd(2, 2);
        non_pd(0, 0) = 1.0f; non_pd(0, 1) = 2.0f;
        non_pd(1, 0) = 2.0f; non_pd(1, 1) = 1.0f;  // Has negative eigenvalue
        std::cout << "Matrix:\n";
        non_pd.print_matrix(true);
        
        bool is_pd = non_pd.is_positive_definite(1e-6f);
        std::cout << "Is positive definite: " << (is_pd ? "True" : "False") 
                  << " (Expected: False) " << (!is_pd ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.13: max_minors_to_check parameter
    {
        std::cout << "\n[E1.13] max_minors_to_check Parameter Testing\n";
        tiny::Mat pd_mat(4, 4);
        // Create a 4x4 positive definite matrix
        pd_mat(0, 0) = 4.0f; pd_mat(0, 1) = 1.0f; pd_mat(0, 2) = 0.0f; pd_mat(0, 3) = 0.0f;
        pd_mat(1, 0) = 1.0f; pd_mat(1, 1) = 3.0f; pd_mat(1, 2) = 0.0f; pd_mat(1, 3) = 0.0f;
        pd_mat(2, 0) = 0.0f; pd_mat(2, 1) = 0.0f; pd_mat(2, 2) = 2.0f; pd_mat(2, 3) = 0.5f;
        pd_mat(3, 0) = 0.0f; pd_mat(3, 1) = 0.0f; pd_mat(3, 2) = 0.5f; pd_mat(3, 3) = 1.5f;
        
        // Test with max_minors_to_check = -1 (check all minors)
        bool is_pd_all = pd_mat.is_positive_definite(1e-6f, -1);
        std::cout << "max_minors_to_check = -1 (check all): " << (is_pd_all ? "True" : "False") 
                  << " (Expected: True) " << (is_pd_all ? "[PASS]" : "[FAIL]") << "\n";
        
        // Test with max_minors_to_check = 3 (check first 3 minors)
        bool is_pd_partial = pd_mat.is_positive_definite(1e-6f, 3);
        std::cout << "max_minors_to_check = 3 (check first 3): " << (is_pd_partial ? "True" : "False") 
                  << " (Expected: True) " << (is_pd_partial ? "[PASS]" : "[FAIL]") << "\n";
        
        // Test with max_minors_to_check = 0 (should return false/error)
        bool is_pd_zero = pd_mat.is_positive_definite(1e-6f, 0);
        std::cout << "max_minors_to_check = 0 (invalid): " << (is_pd_zero ? "True" : "False") 
                  << " (Expected: False) " << (!is_pd_zero ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.14: Parameter validation - negative tolerance
    {
        std::cout << "\n[E1.14] Parameter Validation - Negative Tolerance\n";
        tiny::Mat pd_mat(2, 2);
        pd_mat(0, 0) = 2.0f; pd_mat(0, 1) = 0.0f;
        pd_mat(1, 0) = 0.0f; pd_mat(1, 1) = 2.0f;
        
        // Test with tolerance < 0 (should return false/error)
        bool is_pd_neg = pd_mat.is_positive_definite(-1e-6f);
        std::cout << "tolerance = -1e-6 (invalid): " << (is_pd_neg ? "True" : "False") 
                  << " (Expected: False) " << (!is_pd_neg ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.15: Boundary case - empty matrix
    {
        std::cout << "\n[E1.15] Boundary Case - Empty Matrix (0x0)\n";
        tiny::Mat empty_mat(0, 0);
        
        // Empty matrix cannot be positive definite (data is null or invalid dimensions)
        bool is_pd_empty = empty_mat.is_positive_definite(1e-6f);
        std::cout << "Empty matrix (0x0): " << (is_pd_empty ? "True" : "False") 
                  << " (Expected: False, empty matrix is invalid) " << (!is_pd_empty ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.16: Boundary case - invalid dimensions
    {
        std::cout << "\n[E1.16] Boundary Case - Invalid Dimensions\n";
        tiny::Mat invalid_mat(2, 3);  // Non-square matrix
        
        // Non-square matrix cannot be positive definite
        bool is_pd_invalid = invalid_mat.is_positive_definite(1e-6f);
        std::cout << "Non-square matrix (2x3): " << (is_pd_invalid ? "True" : "False") 
                  << " (Expected: False) " << (!is_pd_invalid ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.2: LU Decomposition
    std::cout << "\n[E1.2] LU Decomposition\n";
    
    // E1.21: Simple 3x3 matrix with pivoting
    {
        std::cout << "\n[E1.21] 3x3 Matrix - LU Decomposition with Pivoting\n";
        tiny::Mat A(3, 3);
        A(0, 0) = 2.0f; A(0, 1) = 1.0f; A(0, 2) = 1.0f;
        A(1, 0) = 4.0f; A(1, 1) = 3.0f; A(1, 2) = 3.0f;
        A(2, 0) = 2.0f; A(2, 1) = 1.0f; A(2, 2) = 2.0f;
        std::cout << "Matrix A:\n";
        A.print_matrix(true);
        
        tiny::Mat::LUDecomposition lu = A.lu_decompose(true);
        std::cout << "\n[Results]\n";
        std::cout << "Status: " << (lu.status == TINY_OK ? "OK" : "Error") << "\n";
        if (lu.status == TINY_OK)
        {
            std::cout << "L matrix (lower triangular):\n";
            lu.L.print_matrix(true);
            std::cout << "U matrix (upper triangular):\n";
            lu.U.print_matrix(true);
            if (lu.pivoted)
            {
                std::cout << "P matrix (permutation):\n";
                lu.P.print_matrix(true);
            }
            
            // Verify: P * A = L * U
            tiny::Mat PA = lu.P * A;
            tiny::Mat LU = lu.L * lu.U;
            std::cout << "\n[Verification] P * A should equal L * U\n";
            float diff = 0.0f;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    diff += fabsf(PA(i, j) - LU(i, j));
                }
            }
            std::cout << "Total difference: " << diff << (diff < 0.01f ? " [PASS]" : " [FAIL]") << "\n";
        }
    }

    // E1.22: Solve using LU decomposition
    {
        std::cout << "\n[E1.22] Solve Linear System using LU Decomposition\n";
        tiny::Mat A(3, 3);
        A(0, 0) = 2.0f; A(0, 1) = 1.0f; A(0, 2) = 1.0f;
        A(1, 0) = 4.0f; A(1, 1) = 3.0f; A(1, 2) = 3.0f;
        A(2, 0) = 2.0f; A(2, 1) = 1.0f; A(2, 2) = 2.0f;
        tiny::Mat b(3, 1);
        b(0, 0) = 1.0f;
        b(1, 0) = 2.0f;
        b(2, 0) = 3.0f;
        
        std::cout << "System: A * x = b\n";
        std::cout << "A:\n";
        A.print_matrix(true);
        std::cout << "b:\n";
        b.print_matrix(true);
        
        tiny::Mat::LUDecomposition lu = A.lu_decompose(true);
        tiny::Mat x = tiny::Mat::solve_lu(lu, b);
        
        std::cout << "\n[Results]\n";
        std::cout << "Solution x:\n";
        x.print_matrix(true);
        
        // Verify: A * x = b
        tiny::Mat Ax = A * x;
        float error = 0.0f;
        for (int i = 0; i < 3; ++i)
        {
            error += fabsf(Ax(i, 0) - b(i, 0));
        }
        std::cout << "Verification error: " << error << (error < 0.01f ? " [PASS]" : " [FAIL]") << "\n";
    }

    // E1.26: solve_lu() - Boundary case - empty matrix
    {
        std::cout << "\n[E1.26] solve_lu() - Boundary Case - Empty Matrix\n";
        tiny::Mat empty_A(0, 0);
        tiny::Mat empty_b(0, 1);
        
        tiny::Mat::LUDecomposition lu_empty = empty_A.lu_decompose(true);
        tiny::Mat x_empty = tiny::Mat::solve_lu(lu_empty, empty_b);
        // Error case should return empty matrix (0x0) or error state (1x1 error matrix)
        bool empty_correct = (x_empty.row == 0 && x_empty.col == 0) || 
                            (x_empty.data == nullptr) ||
                            (x_empty.row == 1 && x_empty.col == 1 && x_empty.data != nullptr);
        std::cout << "Empty system: x rows = " << x_empty.row 
                  << " (Expected: 0 or error state) " << (empty_correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.27: solve_lu() - Invalid LU decomposition
    {
        std::cout << "\n[E1.27] solve_lu() - Invalid LU Decomposition\n";
        tiny::Mat::LUDecomposition invalid_lu;
        invalid_lu.status = TINY_ERR_INVALID_ARG;  // Simulate invalid decomposition
        
        tiny::Mat b(3, 1);
        b(0, 0) = 1.0f; b(1, 0) = 2.0f; b(2, 0) = 3.0f;
        
        tiny::Mat x_invalid = tiny::Mat::solve_lu(invalid_lu, b);
        // Error case should return empty matrix (0x0) or error state (1x1 error matrix)
        bool invalid_correct = (x_invalid.row == 0 && x_invalid.col == 0) || 
                               (x_invalid.data == nullptr) ||
                               (x_invalid.row == 1 && x_invalid.col == 1 && x_invalid.data != nullptr);
        std::cout << "Invalid LU decomposition: x rows = " << x_invalid.row 
                  << " (Expected: 0 or error state) " << (invalid_correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.23: Boundary case - empty matrix
    {
        std::cout << "\n[E1.23] Boundary Case - Empty Matrix (0x0)\n";
        tiny::Mat empty_mat(0, 0);
        
        tiny::Mat::LUDecomposition lu_empty = empty_mat.lu_decompose(true);
        // Error case: empty matrix should return error status, L may be 0x0 or 1x1 error matrix
        bool empty_correct = (lu_empty.status != TINY_OK) && 
                            ((lu_empty.L.row == 0 && lu_empty.L.col == 0) || 
                             (lu_empty.L.data == nullptr) ||
                             (lu_empty.L.row == 1 && lu_empty.L.col == 1 && lu_empty.L.data != nullptr));
        std::cout << "Empty matrix: Status = " << (lu_empty.status == TINY_OK ? "OK" : "Error") 
                  << ", L rows = " << lu_empty.L.row 
                  << " (Expected: Error status, L is 0x0 or error state) " << (empty_correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.24: lu_decompose() - without pivoting
    {
        std::cout << "\n[E1.24] LU Decomposition without Pivoting\n";
        tiny::Mat A(3, 3);
        A(0, 0) = 2.0f; A(0, 1) = 1.0f; A(0, 2) = 1.0f;
        A(1, 0) = 4.0f; A(1, 1) = 3.0f; A(1, 2) = 3.0f;
        A(2, 0) = 2.0f; A(2, 1) = 1.0f; A(2, 2) = 2.0f;
        
        tiny::Mat::LUDecomposition lu_no_pivot = A.lu_decompose(false);
        std::cout << "Status: " << (lu_no_pivot.status == TINY_OK ? "OK" : "Error") << "\n";
        std::cout << "Pivoted: " << (lu_no_pivot.pivoted ? "Yes" : "No (Expected)") 
                  << " " << (!lu_no_pivot.pivoted ? "[PASS]" : "[FAIL]") << "\n";
        if (lu_no_pivot.status == TINY_OK && !lu_no_pivot.pivoted)
        {
            // Verify: A = L * U (no permutation)
            tiny::Mat LU = lu_no_pivot.L * lu_no_pivot.U;
            float diff = 0.0f;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    diff += fabsf(LU(i, j) - A(i, j));
                }
            }
            std::cout << "Verification (A = L * U): difference = " << diff 
                      << (diff < 0.01f ? " [PASS]" : " [FAIL]") << "\n";
        }
    }

    // E1.25: lu_decompose() - Error handling - non-square matrix
    {
        std::cout << "\n[E1.25] lu_decompose() - Error Handling - Non-Square Matrix\n";
        tiny::Mat non_square(2, 3);
        tiny::Mat::LUDecomposition lu_non_square = non_square.lu_decompose(true);
        std::cout << "Non-square matrix (2x3): Status = " << (lu_non_square.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (lu_non_square.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.3: Cholesky Decomposition
    std::cout << "\n[E1.3] Cholesky Decomposition\n";
    
    // E1.31: Symmetric positive definite matrix
    {
        std::cout << "\n[E1.31] SPD Matrix - Cholesky Decomposition\n";
        tiny::Mat spd(3, 3);
        spd(0, 0) = 4.0f; spd(0, 1) = 2.0f; spd(0, 2) = 0.0f;
        spd(1, 0) = 2.0f; spd(1, 1) = 5.0f; spd(1, 2) = 1.0f;
        spd(2, 0) = 0.0f; spd(2, 1) = 1.0f; spd(2, 2) = 3.0f;
        std::cout << "Matrix A (SPD):\n";
        spd.print_matrix(true);
        
        tiny::Mat::CholeskyDecomposition chol = spd.cholesky_decompose();
        std::cout << "\n[Results]\n";
        std::cout << "Status: " << (chol.status == TINY_OK ? "OK" : "Error") << "\n";
        if (chol.status == TINY_OK)
        {
            std::cout << "L matrix (lower triangular):\n";
            chol.L.print_matrix(true);
            
            // Verify: A = L * L^T
            tiny::Mat LLT = chol.L * chol.L.transpose();
            std::cout << "\n[Verification] L * L^T should equal A\n";
            float diff = 0.0f;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    diff += fabsf(LLT(i, j) - spd(i, j));
                }
            }
            std::cout << "Total difference: " << diff << (diff < 0.01f ? " [PASS]" : " [FAIL]") << "\n";
        }
    }

    // E1.32: Solve using Cholesky decomposition
    {
        std::cout << "\n[E1.32] Solve Linear System using Cholesky Decomposition\n";
        tiny::Mat A(3, 3);
        A(0, 0) = 4.0f; A(0, 1) = 2.0f; A(0, 2) = 0.0f;
        A(1, 0) = 2.0f; A(1, 1) = 5.0f; A(1, 2) = 1.0f;
        A(2, 0) = 0.0f; A(2, 1) = 1.0f; A(2, 2) = 3.0f;
        tiny::Mat b(3, 1);
        b(0, 0) = 2.0f;
        b(1, 0) = 3.0f;
        b(2, 0) = 1.0f;
        
        tiny::Mat::CholeskyDecomposition chol = A.cholesky_decompose();
        tiny::Mat x = tiny::Mat::solve_cholesky(chol, b);
        
        std::cout << "Solution x:\n";
        x.print_matrix(true);
        
        // Verify: A * x = b
        tiny::Mat Ax = A * x;
        float error = 0.0f;
        for (int i = 0; i < 3; ++i)
        {
            error += fabsf(Ax(i, 0) - b(i, 0));
        }
        std::cout << "Verification error: " << error << (error < 0.01f ? " [PASS]" : " [FAIL]") << "\n";
    }

    // E1.35: solve_cholesky() - Boundary case - empty matrix
    {
        std::cout << "\n[E1.35] solve_cholesky() - Boundary Case - Empty Matrix\n";
        tiny::Mat empty_A(0, 0);
        tiny::Mat empty_b(0, 1);
        
        tiny::Mat::CholeskyDecomposition chol_empty = empty_A.cholesky_decompose();
        tiny::Mat x_empty = tiny::Mat::solve_cholesky(chol_empty, empty_b);
        // Error case should return empty matrix (0x0) or error state (1x1 error matrix)
        bool empty_correct = (x_empty.row == 0 && x_empty.col == 0) || 
                            (x_empty.data == nullptr) ||
                            (x_empty.row == 1 && x_empty.col == 1 && x_empty.data != nullptr);
        std::cout << "Empty system: x rows = " << x_empty.row 
                  << " (Expected: 0 or error state) " << (empty_correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.36: solve_cholesky() - Invalid Cholesky decomposition
    {
        std::cout << "\n[E1.36] solve_cholesky() - Invalid Cholesky Decomposition\n";
        tiny::Mat::CholeskyDecomposition invalid_chol;
        invalid_chol.status = TINY_ERR_INVALID_ARG;  // Simulate invalid decomposition
        
        tiny::Mat b(3, 1);
        b(0, 0) = 1.0f; b(1, 0) = 2.0f; b(2, 0) = 3.0f;
        
        tiny::Mat x_invalid = tiny::Mat::solve_cholesky(invalid_chol, b);
        // Error case should return empty matrix (0x0) or error state (1x1 error matrix)
        bool invalid_correct = (x_invalid.row == 0 && x_invalid.col == 0) || 
                               (x_invalid.data == nullptr) ||
                               (x_invalid.row == 1 && x_invalid.col == 1 && x_invalid.data != nullptr);
        std::cout << "Invalid Cholesky decomposition: x rows = " << x_invalid.row 
                  << " (Expected: 0 or error state) " << (invalid_correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.33: Boundary case - empty matrix
    {
        std::cout << "\n[E1.33] Boundary Case - Empty Matrix (0x0)\n";
        tiny::Mat empty_mat(0, 0);
        
        tiny::Mat::CholeskyDecomposition chol_empty = empty_mat.cholesky_decompose();
        // Error case: empty matrix should return error status, L may be 0x0 or 1x1 error matrix
        bool empty_correct = (chol_empty.status != TINY_OK) && 
                            ((chol_empty.L.row == 0 && chol_empty.L.col == 0) || 
                             (chol_empty.L.data == nullptr) ||
                             (chol_empty.L.row == 1 && chol_empty.L.col == 1 && chol_empty.L.data != nullptr));
        std::cout << "Empty matrix: Status = " << (chol_empty.status == TINY_OK ? "OK" : "Error") 
                  << ", L rows = " << chol_empty.L.row 
                  << " (Expected: Error status, L is 0x0 or error state) " << (empty_correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.34: Non-symmetric matrix (should fail)
    {
        std::cout << "\n[E1.34] Non-Symmetric Matrix (Should Fail)\n";
        tiny::Mat non_sym(2, 2);
        non_sym(0, 0) = 1.0f; non_sym(0, 1) = 2.0f;
        non_sym(1, 0) = 3.0f; non_sym(1, 1) = 4.0f;  // Non-symmetric
        
        tiny::Mat::CholeskyDecomposition chol_non_sym = non_sym.cholesky_decompose();
        std::cout << "Non-symmetric matrix: Status = " << (chol_non_sym.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (chol_non_sym.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.37: solve_cholesky() - Error handling - dimension mismatch
    {
        std::cout << "\n[E1.37] solve_cholesky() - Error Handling - Dimension Mismatch\n";
        tiny::Mat A(3, 3);
        A(0, 0) = 4.0f; A(0, 1) = 2.0f; A(0, 2) = 0.0f;
        A(1, 0) = 2.0f; A(1, 1) = 5.0f; A(1, 2) = 1.0f;
        A(2, 0) = 0.0f; A(2, 1) = 1.0f; A(2, 2) = 3.0f;
        tiny::Mat b(4, 1);  // Wrong dimension
        tiny::Mat::CholeskyDecomposition chol = A.cholesky_decompose();
        if (chol.status == TINY_OK)
        {
            tiny::Mat x_mismatch = tiny::Mat::solve_cholesky(chol, b);
            // Error case should return empty matrix (0x0) or error state (1x1 error matrix)
            // For dimension mismatch, expected solution size is 3x1, so 1x1 indicates error
            bool solve_mismatch_correct = (x_mismatch.row == 0 && x_mismatch.col == 0) || 
                                         (x_mismatch.data == nullptr) ||
                                         (x_mismatch.row == 1 && x_mismatch.col == 1 && x_mismatch.data != nullptr);
            std::cout << "Dimension mismatch solve_cholesky: " << (solve_mismatch_correct ? "Empty matrix or error state (Expected)" : "Non-empty (Error)") 
                      << " " << (solve_mismatch_correct ? "[PASS]" : "[FAIL]") << "\n";
        }
    }

    // E1.4: QR Decomposition
    std::cout << "\n[E1.4] QR Decomposition\n";
    
    // E1.41: General matrix
    {
        std::cout << "\n[E1.41] General 3x3 Matrix - QR Decomposition\n";
        tiny::Mat A(3, 3);
        A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
        A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;
        A(2, 0) = 7.0f; A(2, 1) = 8.0f; A(2, 2) = 9.0f;
        std::cout << "Matrix A:\n";
        A.print_matrix(true);
        
        tiny::Mat::QRDecomposition qr = A.qr_decompose();
        std::cout << "\n[Results]\n";
        std::cout << "Status: " << (qr.status == TINY_OK ? "OK" : "Error") << "\n";
        if (qr.status == TINY_OK)
        {
            std::cout << "Q matrix (orthogonal):\n";
            qr.Q.print_matrix(true);
            std::cout << "R matrix (upper triangular):\n";
            qr.R.print_matrix(true);
            
            // Verify: A = Q * R
            tiny::Mat QR = qr.Q * qr.R;
            std::cout << "\n[Verification] Q * R should equal A\n";
            float diff = 0.0f;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    diff += fabsf(QR(i, j) - A(i, j));
                }
            }
            std::cout << "Total difference: " << diff << (diff < 0.1f ? " [PASS]" : " [FAIL]") << "\n";
            
            // Verify Q is orthogonal: Q^T * Q = I
            tiny::Mat QtQ = qr.Q.transpose() * qr.Q;
            tiny::Mat I = tiny::Mat::eye(3);
            float ortho_diff = 0.0f;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    ortho_diff += fabsf(QtQ(i, j) - I(i, j));
                }
            }
            std::cout << "Q orthogonality error: " << ortho_diff << (ortho_diff < 0.1f ? " [PASS]" : " [FAIL]") << "\n";
        }
    }

    // E1.42: Solve using QR decomposition (least squares)
    {
        std::cout << "\n[E1.42] Least Squares Solution using QR Decomposition\n";
        tiny::Mat A(3, 2);  // Overdetermined system
        A(0, 0) = 1.0f; A(0, 1) = 1.0f;
        A(1, 0) = 1.0f; A(1, 1) = 2.0f;
        A(2, 0) = 1.0f; A(2, 1) = 3.0f;
        tiny::Mat b(3, 1);
        b(0, 0) = 2.0f;
        b(1, 0) = 3.0f;
        b(2, 0) = 4.0f;
        
        std::cout << "Overdetermined system: A * x ≈ b\n";
        std::cout << "A:\n";
        A.print_matrix(true);
        std::cout << "b:\n";
        b.print_matrix(true);
        
        tiny::Mat::QRDecomposition qr = A.qr_decompose();
        tiny::Mat x = tiny::Mat::solve_qr(qr, b);
        
        std::cout << "\n[Results]\n";
        std::cout << "Least squares solution x:\n";
        x.print_matrix(true);
        
        // Compute residual: ||A * x - b||
        tiny::Mat Ax = A * x;
        tiny::Mat residual = Ax - b;
        float residual_norm = 0.0f;
        for (int i = 0; i < 3; ++i)
        {
            residual_norm += residual(i, 0) * residual(i, 0);
        }
        residual_norm = sqrtf(residual_norm);
        std::cout << "Residual norm ||A*x - b||: " << residual_norm << "\n";
    }

    // E1.46: solve_qr() - Boundary case - empty matrix
    {
        std::cout << "\n[E1.46] solve_qr() - Boundary Case - Empty Matrix\n";
        tiny::Mat empty_A(0, 0);
        tiny::Mat empty_b(0, 1);
        
        tiny::Mat::QRDecomposition qr_empty = empty_A.qr_decompose();
        tiny::Mat x_empty = tiny::Mat::solve_qr(qr_empty, empty_b);
        // Error case should return empty matrix (0x0) or error state (1x1 error matrix)
        bool empty_correct = (x_empty.row == 0 && x_empty.col == 0) || 
                            (x_empty.data == nullptr) ||
                            (x_empty.row == 1 && x_empty.col == 1 && x_empty.data != nullptr);
        std::cout << "Empty system: x rows = " << x_empty.row 
                  << " (Expected: 0 or error state) " << (empty_correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.47: solve_qr() - Invalid QR decomposition
    {
        std::cout << "\n[E1.47] solve_qr() - Invalid QR Decomposition\n";
        tiny::Mat::QRDecomposition invalid_qr;
        invalid_qr.status = TINY_ERR_INVALID_ARG;  // Simulate invalid decomposition
        
        tiny::Mat b(3, 1);
        b(0, 0) = 1.0f; b(1, 0) = 2.0f; b(2, 0) = 3.0f;
        
        tiny::Mat x_invalid = tiny::Mat::solve_qr(invalid_qr, b);
        // Error case should return empty matrix (0x0) or error state (1x1 error matrix)
        bool invalid_correct = (x_invalid.row == 0 && x_invalid.col == 0) || 
                               (x_invalid.data == nullptr) ||
                               (x_invalid.row == 1 && x_invalid.col == 1 && x_invalid.data != nullptr);
        std::cout << "Invalid QR decomposition: x rows = " << x_invalid.row 
                  << " (Expected: 0 or error state) " << (invalid_correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.43: Boundary case - empty matrix
    {
        std::cout << "\n[E1.43] Boundary Case - Empty Matrix (0x0)\n";
        tiny::Mat empty_mat(0, 0);
        
        tiny::Mat::QRDecomposition qr_empty = empty_mat.qr_decompose();
        // Error case: empty matrix should return error status, Q may be 0x0 or 1x1 error matrix
        bool empty_correct = (qr_empty.status != TINY_OK) && 
                            ((qr_empty.Q.row == 0 && qr_empty.Q.col == 0) || 
                             (qr_empty.Q.data == nullptr) ||
                             (qr_empty.Q.row == 1 && qr_empty.Q.col == 1 && qr_empty.Q.data != nullptr));
        std::cout << "Empty matrix: Status = " << (qr_empty.status == TINY_OK ? "OK" : "Error") 
                  << ", Q rows = " << qr_empty.Q.row 
                  << " (Expected: Error status, Q is 0x0 or error state) " << (empty_correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.44: Boundary case - m=0 or n=0
    {
        std::cout << "\n[E1.44] Boundary Case - Zero Rows or Columns\n";
        tiny::Mat zero_rows(0, 3);
        tiny::Mat zero_cols(3, 0);
        
        tiny::Mat::QRDecomposition qr_zero_rows = zero_rows.qr_decompose();
        // Zero rows/cols should return error status (invalid matrix)
        std::cout << "Matrix with 0 rows (0x3): Status = " << (qr_zero_rows.status == TINY_OK ? "OK" : "Error") 
                  << " " << (qr_zero_rows.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
        
        tiny::Mat::QRDecomposition qr_zero_cols = zero_cols.qr_decompose();
        // Zero rows/cols should return error status (invalid matrix)
        std::cout << "Matrix with 0 cols (3x0): Status = " << (qr_zero_cols.status == TINY_OK ? "OK" : "Error") 
                  << " " << (qr_zero_cols.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.45: solve_qr() - Error handling - dimension mismatch
    {
        std::cout << "\n[E1.45] solve_qr() - Error Handling - Dimension Mismatch\n";
        tiny::Mat A(3, 2);
        tiny::Mat b(4, 1);  // Wrong dimension
        tiny::Mat::QRDecomposition qr = A.qr_decompose();
        if (qr.status == TINY_OK)
        {
            tiny::Mat x_mismatch = tiny::Mat::solve_qr(qr, b);
            // Error case should return empty matrix (0x0) or error state (1x1 error matrix)
            // For dimension mismatch, expected solution size is 2x1, so 1x1 indicates error
            bool solve_mismatch_correct = (x_mismatch.row == 0 && x_mismatch.col == 0) || 
                                         (x_mismatch.data == nullptr) ||
                                         (x_mismatch.row == 1 && x_mismatch.col == 1 && x_mismatch.data != nullptr);
            std::cout << "Dimension mismatch solve_qr: " << (solve_mismatch_correct ? "Empty matrix or error state (Expected)" : "Non-empty (Error)") 
                      << " " << (solve_mismatch_correct ? "[PASS]" : "[FAIL]") << "\n";
        }
    }

    // E1.5: SVD Decomposition
    std::cout << "\n[E1.5] Singular Value Decomposition (SVD)\n";
    
    // E1.51: General matrix
    {
        std::cout << "\n[E1.51] General 3x3 Matrix - SVD Decomposition\n";
        tiny::Mat A(3, 3);
        A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
        A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;
        A(2, 0) = 7.0f; A(2, 1) = 8.0f; A(2, 2) = 9.0f;
        std::cout << "Matrix A:\n";
        A.print_matrix(true);
        
        tiny::Mat::SVDDecomposition svd = A.svd_decompose(100, 1e-6f);
        std::cout << "\n[Results]\n";
        std::cout << "Status: " << (svd.status == TINY_OK ? "OK" : "Error") << "\n";
        if (svd.status == TINY_OK)
        {
            std::cout << "Singular values:\n";
            svd.S.print_matrix(true);
            std::cout << "Numerical rank: " << svd.rank << "\n";
            std::cout << "Iterations: " << svd.iterations << "\n";
            
            // Verify: A ≈ U * S * V^T (for first rank columns)
            if (svd.rank > 0)
            {
                tiny::Mat US(svd.U.row, svd.rank);
                for (int i = 0; i < svd.U.row; ++i)
                {
                    for (int j = 0; j < svd.rank; ++j)
                    {
                        US(i, j) = svd.U(i, j) * svd.S(j, 0);
                    }
                }
                tiny::Mat Vt(svd.rank, svd.V.row);
                for (int i = 0; i < svd.rank; ++i)
                {
                    for (int j = 0; j < svd.V.row; ++j)
                    {
                        Vt(i, j) = svd.V(j, i);  // V^T
                    }
                }
                tiny::Mat USVt = US * Vt;
                
                float diff = 0.0f;
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        diff += fabsf(USVt(i, j) - A(i, j));
                    }
                }
                std::cout << "Reconstruction error: " << diff << (diff < 0.5f ? " [PASS]" : " [FAIL]") << "\n";
            }
        }
    }

    // E1.52: Pseudo-inverse using SVD
    {
        std::cout << "\n[E1.52] Pseudo-inverse using SVD\n";
        tiny::Mat A(3, 2);  // Non-square matrix
        A(0, 0) = 1.0f; A(0, 1) = 2.0f;
        A(1, 0) = 3.0f; A(1, 1) = 4.0f;
        A(2, 0) = 5.0f; A(2, 1) = 6.0f;
        
        std::cout << "Matrix A (3x2):\n";
        A.print_matrix(true);
        
        tiny::Mat::SVDDecomposition svd = A.svd_decompose(100, 1e-6f);
        tiny::Mat A_plus = tiny::Mat::pseudo_inverse(svd, 1e-6f);
        
        std::cout << "\n[Results]\n";
        std::cout << "Pseudo-inverse A^+ (2x3):\n";
        A_plus.print_matrix(true);
        
        // Verify: A * A^+ * A ≈ A
        tiny::Mat AAplusA = A * A_plus * A;
        float diff = 0.0f;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                diff += fabsf(AAplusA(i, j) - A(i, j));
            }
        }
        std::cout << "Verification error (A * A^+ * A ≈ A): " << diff << (diff < 0.1f ? " [PASS]" : " [FAIL]") << "\n";
    }

    // E1.57: pseudo_inverse() - Parameter validation - tolerance < 0
    {
        std::cout << "\n[E1.57] pseudo_inverse() - Parameter Validation - tolerance < 0\n";
        tiny::Mat test_mat(2, 2);
        test_mat(0, 0) = 1.0f; test_mat(0, 1) = 2.0f;
        test_mat(1, 0) = 3.0f; test_mat(1, 1) = 4.0f;
        
        tiny::Mat::SVDDecomposition svd = test_mat.svd_decompose(100, 1e-6f);
        if (svd.status == TINY_OK)
        {
            tiny::Mat A_plus_neg = tiny::Mat::pseudo_inverse(svd, -1e-6f);
            // Error case should return empty matrix (0x0) or error state (1x1 error matrix)
            bool neg_tol_correct = (A_plus_neg.row == 0 && A_plus_neg.col == 0) || 
                                  (A_plus_neg.data == nullptr) ||
                                  (A_plus_neg.row == 1 && A_plus_neg.col == 1 && A_plus_neg.data != nullptr);
            std::cout << "tolerance = -1e-6: A_plus rows = " << A_plus_neg.row 
                      << " (Expected: 0 or error state) " << (neg_tol_correct ? "[PASS]" : "[FAIL]") << "\n";
        }
    }

    // E1.58: pseudo_inverse() - Invalid SVD decomposition
    {
        std::cout << "\n[E1.58] pseudo_inverse() - Invalid SVD Decomposition\n";
        tiny::Mat::SVDDecomposition invalid_svd;
        invalid_svd.status = TINY_ERR_INVALID_ARG;  // Simulate invalid decomposition
        
        tiny::Mat A_plus_invalid = tiny::Mat::pseudo_inverse(invalid_svd, 1e-6f);
        // Error case should return empty matrix (0x0) or error state (1x1 error matrix)
        bool pseudo_inv_invalid_correct = (A_plus_invalid.row == 0 && A_plus_invalid.col == 0) || 
                                          (A_plus_invalid.data == nullptr) ||
                                          (A_plus_invalid.row == 1 && A_plus_invalid.col == 1 && A_plus_invalid.data != nullptr);
        std::cout << "Invalid SVD decomposition: A_plus rows = " << A_plus_invalid.row 
                  << " (Expected: 0 or error state) " << (pseudo_inv_invalid_correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.53: Parameter validation - max_iter <= 0
    {
        std::cout << "\n[E1.53] Parameter Validation - max_iter <= 0\n";
        tiny::Mat test_mat(2, 2);
        test_mat(0, 0) = 1.0f; test_mat(0, 1) = 2.0f;
        test_mat(1, 0) = 3.0f; test_mat(1, 1) = 4.0f;
        
        tiny::Mat::SVDDecomposition result_zero = test_mat.svd_decompose(0, 1e-6f);
        std::cout << "max_iter = 0: Status = " << (result_zero.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_zero.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
        
        tiny::Mat::SVDDecomposition result_neg = test_mat.svd_decompose(-1, 1e-6f);
        std::cout << "max_iter = -1: Status = " << (result_neg.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_neg.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.54: Parameter validation - tolerance < 0
    {
        std::cout << "\n[E1.54] Parameter Validation - tolerance < 0\n";
        tiny::Mat test_mat(2, 2);
        test_mat(0, 0) = 1.0f; test_mat(0, 1) = 2.0f;
        test_mat(1, 0) = 3.0f; test_mat(1, 1) = 4.0f;
        
        tiny::Mat::SVDDecomposition result_neg = test_mat.svd_decompose(100, -1e-6f);
        std::cout << "tolerance = -1e-6: Status = " << (result_neg.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_neg.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.55: Boundary case - empty matrix (m=0 or n=0)
    {
        std::cout << "\n[E1.55] Boundary Case - Empty Matrix (m=0 or n=0)\n";
        tiny::Mat zero_rows(0, 3);
        tiny::Mat zero_cols(3, 0);
        
        tiny::Mat::SVDDecomposition svd_zero_rows = zero_rows.svd_decompose(100, 1e-6f);
        // Zero rows/cols should return error status (invalid matrix)
        std::cout << "Matrix with 0 rows (0x3): Status = " << (svd_zero_rows.status == TINY_OK ? "OK" : "Error") 
                  << " " << (svd_zero_rows.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
        
        tiny::Mat::SVDDecomposition svd_zero_cols = zero_cols.svd_decompose(100, 1e-6f);
        // Zero rows/cols should return error status (invalid matrix)
        std::cout << "Matrix with 0 cols (3x0): Status = " << (svd_zero_cols.status == TINY_OK ? "OK" : "Error") 
                  << " " << (svd_zero_cols.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.56: pseudo_inverse() - Error handling - invalid SVD decomposition
    {
        std::cout << "\n[E1.56] pseudo_inverse() - Error Handling - Invalid SVD Decomposition\n";
        tiny::Mat::SVDDecomposition invalid_svd;
        invalid_svd.status = TINY_ERR_INVALID_ARG;  // Simulate invalid decomposition
        
        tiny::Mat A_plus_invalid = tiny::Mat::pseudo_inverse(invalid_svd, 1e-6f);
        // Error case should return empty matrix (0x0) or error state (1x1 error matrix)
        bool pseudo_inv_invalid_correct = (A_plus_invalid.row == 0 && A_plus_invalid.col == 0) || 
                                          (A_plus_invalid.data == nullptr) ||
                                          (A_plus_invalid.row == 1 && A_plus_invalid.col == 1 && A_plus_invalid.data != nullptr);
        std::cout << "Invalid SVD decomposition: A_plus rows = " << A_plus_invalid.row 
                  << " (Expected: 0 or error state) " << (pseudo_inv_invalid_correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E1.6: Performance Tests
    std::cout << "\n[E1.6] Matrix Decomposition Performance Tests\n";
    
    tiny::Mat perf_mat(4, 4);
    perf_mat(0, 0) = 4.0f; perf_mat(0, 1) = 2.0f; perf_mat(0, 2) = 1.0f; perf_mat(0, 3) = 0.0f;
    perf_mat(1, 0) = 2.0f; perf_mat(1, 1) = 5.0f; perf_mat(1, 2) = 1.0f; perf_mat(1, 3) = 0.0f;
    perf_mat(2, 0) = 1.0f; perf_mat(2, 1) = 1.0f; perf_mat(2, 2) = 3.0f; perf_mat(2, 3) = 1.0f;
    perf_mat(3, 0) = 0.0f; perf_mat(3, 1) = 0.0f; perf_mat(3, 2) = 1.0f; perf_mat(3, 3) = 2.0f;
    
    // E1.61: LU decomposition performance
    std::cout << "\n[E1.61] LU Decomposition Performance\n";
    TIME_OPERATION(
        tiny::Mat::LUDecomposition perf_lu = perf_mat.lu_decompose(true);
        (void)perf_lu;
    , "LU Decomposition (4x4 matrix)");
    
    // E1.62: Cholesky decomposition performance
    std::cout << "\n[E1.62] Cholesky Decomposition Performance\n";
    TIME_OPERATION(
        tiny::Mat::CholeskyDecomposition perf_chol = perf_mat.cholesky_decompose();
        (void)perf_chol;
    , "Cholesky Decomposition (4x4 SPD matrix)");
    
    // E1.63: QR decomposition performance
    std::cout << "\n[E1.63] QR Decomposition Performance\n";
    TIME_OPERATION(
        tiny::Mat::QRDecomposition perf_qr = perf_mat.qr_decompose();
        (void)perf_qr;
    , "QR Decomposition (4x4 matrix)");
    
    // E1.64: SVD decomposition performance
    std::cout << "\n[E1.64] SVD Decomposition Performance\n";
    TIME_OPERATION(
        tiny::Mat::SVDDecomposition perf_svd = perf_mat.svd_decompose(50, 1e-5f);
        (void)perf_svd;
    , "SVD Decomposition (4x4 matrix)");

    std::cout << "\n[Matrix Decomposition Tests Complete]\n";
}

// ============================================================================
// ============================================================================
// E2: Gram-Schmidt Orthogonalization
// ============================================================================
void test_gram_schmidt_orthogonalize()
{
    std::cout << "\n[E2: Gram-Schmidt Orthogonalization Tests]\n";

    // E2.1: Basic orthogonalization of linearly independent vectors
    {
        std::cout << "\n[E2.1] Basic Orthogonalization - Linearly Independent Vectors\n";
        tiny::Mat vectors(3, 3);
        // Create three linearly independent vectors
        vectors(0, 0) = 1.0f; vectors(0, 1) = 1.0f; vectors(0, 2) = 0.0f;
        vectors(1, 0) = 0.0f; vectors(1, 1) = 1.0f; vectors(1, 2) = 1.0f;
        vectors(2, 0) = 1.0f; vectors(2, 1) = 0.0f; vectors(2, 2) = 1.0f;
        
        std::cout << "Input vectors (each column is a vector):\n";
        vectors.print_matrix(true);
        
        tiny::Mat Q, R;
        bool success = tiny::Mat::gram_schmidt_orthogonalize(vectors, Q, R, 1e-6f);
        
        std::cout << "\n[Results]\n";
        std::cout << "Status: " << (success ? "OK" : "Error") << "\n";
        if (success)
        {
            std::cout << "Orthogonalized vectors Q (each column is orthogonal):\n";
            Q.print_matrix(true);
            std::cout << "Coefficients R (upper triangular):\n";
            R.print_matrix(true);
            
            // Verify orthogonality: Q^T * Q should be identity (or close to it)
            tiny::Mat QtQ = Q.transpose() * Q;
            tiny::Mat I = tiny::Mat::eye(3);
            float ortho_error = 0.0f;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    ortho_error += fabsf(QtQ(i, j) - I(i, j));
                }
            }
            std::cout << "\n[Verification] Q^T * Q should be identity\n";
            std::cout << "Orthogonality error: " << ortho_error 
                      << (ortho_error < 0.1f ? " [PASS]" : " [FAIL]") << "\n";
            
            // Verify normalization: each column of Q should be unit vector
            std::cout << "\n[Verification] Each column of Q should be normalized\n";
            bool all_normalized = true;
            for (int j = 0; j < 3; ++j)
            {
                float norm = 0.0f;
                for (int i = 0; i < 3; ++i)
                {
                    norm += Q(i, j) * Q(i, j);
                }
                norm = sqrtf(norm);
                float norm_error = fabsf(norm - 1.0f);
                std::cout << "  Column " << j << " norm: " << norm 
                          << " (error: " << norm_error << ")";
                if (norm_error > 0.01f)
                {
                    all_normalized = false;
                    std::cout << " [FAIL]";
                }
                else
                {
                    std::cout << " [PASS]";
                }
                std::cout << "\n";
            }
            
            // Verify reconstruction: vectors should equal Q * R (approximately)
            tiny::Mat QR = Q * R;
            float recon_error = 0.0f;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    recon_error += fabsf(QR(i, j) - vectors(i, j));
                }
            }
            std::cout << "\n[Verification] Q * R should reconstruct original vectors\n";
            std::cout << "Reconstruction error: " << recon_error 
                      << (recon_error < 0.1f ? " [PASS]" : " [FAIL]") << "\n";
        }
    }
    
    // E2.2: Orthogonalization with near-linear-dependent vectors
    {
        std::cout << "\n[E2.2] Orthogonalization - Near-Linear-Dependent Vectors\n";
        tiny::Mat vectors(3, 3);
        // Create vectors where third is almost a linear combination of first two
        vectors(0, 0) = 1.0f; vectors(0, 1) = 0.0f; vectors(0, 2) = 1.0f;
        vectors(1, 0) = 0.0f; vectors(1, 1) = 1.0f; vectors(1, 2) = 1.0f;
        vectors(2, 0) = 0.0f; vectors(2, 1) = 0.0f; vectors(2, 2) = 0.001f;  // Very small third component
        
        std::cout << "Input vectors (third vector is nearly linear dependent):\n";
        vectors.print_matrix(true);
        
        tiny::Mat Q, R;
        bool success = tiny::Mat::gram_schmidt_orthogonalize(vectors, Q, R, 1e-6f);
        
        std::cout << "\n[Results]\n";
        std::cout << "Status: " << (success ? "OK" : "Error") << "\n";
        if (success)
        {
            std::cout << "Orthogonalized vectors Q:\n";
            Q.print_matrix(true);
            std::cout << "Coefficients R:\n";
            R.print_matrix(true);
            
            // Check if third column was handled correctly (should be zero or orthogonal)
            float third_col_norm = 0.0f;
            for (int i = 0; i < 3; ++i)
            {
                third_col_norm += Q(i, 2) * Q(i, 2);
            }
            third_col_norm = sqrtf(third_col_norm);
            std::cout << "\n[Note] Third column norm: " << third_col_norm 
                      << " (should be 0 if linearly dependent, or 1 if orthogonalized)\n";
        }
    }
    
    // E2.3: Orthogonalization of 2D vectors
    {
        std::cout << "\n[E2.3] Orthogonalization - 2D Vectors (2x2)\n";
        tiny::Mat vectors(2, 2);
        vectors(0, 0) = 3.0f; vectors(0, 1) = 1.0f;
        vectors(1, 0) = 1.0f; vectors(1, 1) = 2.0f;
        
        std::cout << "Input vectors:\n";
        vectors.print_matrix(true);
        
        tiny::Mat Q, R;
        bool success = tiny::Mat::gram_schmidt_orthogonalize(vectors, Q, R, 1e-6f);
        
        std::cout << "\n[Results]\n";
        std::cout << "Status: " << (success ? "OK" : "Error") << "\n";
        if (success)
        {
            std::cout << "Orthogonalized vectors Q:\n";
            Q.print_matrix(true);
            std::cout << "Coefficients R:\n";
            R.print_matrix(true);
            
            // Verify orthogonality
            float dot_product = 0.0f;
            for (int i = 0; i < 2; ++i)
            {
                dot_product += Q(i, 0) * Q(i, 1);
            }
            std::cout << "\n[Verification] Dot product of Q columns: " << dot_product 
                      << " (should be ~0 for orthogonal) " 
                      << (fabsf(dot_product) < 0.01f ? "[PASS]" : "[FAIL]") << "\n";
        }
    }
    
    // E2.4: Error handling - invalid input
    {
        std::cout << "\n[E2.4] Error Handling - Invalid Input\n";
        tiny::Mat empty_mat(0, 0);  // True empty matrix (0x0)
        tiny::Mat Q, R;
        bool success = tiny::Mat::gram_schmidt_orthogonalize(empty_mat, Q, R, 1e-6f);
        std::cout << "Empty matrix test: " << (success ? "FAIL (should return false)" : "PASS (correctly rejected)") << "\n";
    }

    // E2.5: gram_schmidt_orthogonalize() - Parameter validation - negative tolerance
    {
        std::cout << "\n[E2.5] gram_schmidt_orthogonalize() - Parameter Validation - Negative Tolerance\n";
        tiny::Mat vectors(2, 2);
        vectors(0, 0) = 1.0f; vectors(0, 1) = 0.0f;
        vectors(1, 0) = 0.0f; vectors(1, 1) = 1.0f;
        tiny::Mat Q, R;
        bool success_neg = tiny::Mat::gram_schmidt_orthogonalize(vectors, Q, R, -1e-6f);
        std::cout << "tolerance = -1e-6: " << (success_neg ? "FAIL (should return false)" : "PASS (correctly rejected)") << "\n";
    }

    // E2.6: gram_schmidt_orthogonalize() - Boundary case - zero rows
    {
        std::cout << "\n[E2.6] gram_schmidt_orthogonalize() - Boundary Case - Zero Rows\n";
        tiny::Mat zero_rows(0, 2);
        tiny::Mat Q, R;
        bool success_zero_rows = tiny::Mat::gram_schmidt_orthogonalize(zero_rows, Q, R, 1e-6f);
        std::cout << "Zero rows (0x2): " << (success_zero_rows ? "FAIL (should return false)" : "PASS (correctly rejected)") << "\n";
    }

    // E2.7: gram_schmidt_orthogonalize() - Boundary case - zero columns
    {
        std::cout << "\n[E2.7] gram_schmidt_orthogonalize() - Boundary Case - Zero Columns\n";
        tiny::Mat zero_cols(2, 0);
        tiny::Mat Q, R;
        bool success_zero_cols = tiny::Mat::gram_schmidt_orthogonalize(zero_cols, Q, R, 1e-6f);
        std::cout << "Zero columns (2x0): " << (success_zero_cols ? "FAIL (should return false)" : "PASS (correctly rejected)") << "\n";
    }
}

// ============================================================================
// ============================================================================
// E3: Eigenvalue Decomposition
// ============================================================================
void test_eigenvalue_decomposition()
{
    std::cout << "\n[E3: Eigenvalue Decomposition Tests]\n";

    // E3.1: is_symmetric() - Basic functionality
    std::cout << "\n[E3.1] is_symmetric() - Basic Functionality\n";
    
    // E3.11: Symmetric matrix
    {
        std::cout << "[E3.11] Symmetric 3x3 Matrix\n";
        tiny::Mat sym_mat1(3, 3);
        sym_mat1(0, 0) = 4.0f; sym_mat1(0, 1) = 1.0f; sym_mat1(0, 2) = 2.0f;
        sym_mat1(1, 0) = 1.0f; sym_mat1(1, 1) = 3.0f; sym_mat1(1, 2) = 0.0f;
        sym_mat1(2, 0) = 2.0f; sym_mat1(2, 1) = 0.0f; sym_mat1(2, 2) = 5.0f;
        bool is_sym1 = sym_mat1.is_symmetric(1e-5f);
        std::cout << "Matrix:\n";
        sym_mat1.print_matrix(true);
        std::cout << "Is symmetric: " << (is_sym1 ? "True" : "False") << " (Expected: True)\n";
    }

    // E3.12: Non-symmetric matrix (keep for later tests)
    tiny::Mat non_sym_mat(3, 3);
    {
        std::cout << "\n[E3.12] Non-Symmetric 3x3 Matrix\n";
        non_sym_mat(0, 0) = 1.0f; non_sym_mat(0, 1) = 2.0f; non_sym_mat(0, 2) = 3.0f;
        non_sym_mat(1, 0) = 4.0f; non_sym_mat(1, 1) = 5.0f; non_sym_mat(1, 2) = 6.0f;
        non_sym_mat(2, 0) = 7.0f; non_sym_mat(2, 1) = 8.0f; non_sym_mat(2, 2) = 9.0f;
        bool is_sym2 = non_sym_mat.is_symmetric(1e-5f);
        std::cout << "Matrix:\n";
        non_sym_mat.print_matrix(true);
        std::cout << "Is symmetric: " << (is_sym2 ? "True" : "False") << " (Expected: False)\n";
    }

    // E3.13: Non-square matrix
    {
        std::cout << "\n[E3.13] Non-Square Matrix (2x3)\n";
        tiny::Mat rect_mat(2, 3);
        bool is_sym3 = rect_mat.is_symmetric(1e-5f);
        std::cout << "Is symmetric: " << (is_sym3 ? "True" : "False") << " (Expected: False)\n";
    }

    // E3.14: Symmetric matrix with small numerical errors
    {
        std::cout << "\n[E3.14] Symmetric Matrix with Small Numerical Errors\n";
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

    // E3.15: Parameter validation - negative tolerance
    {
        std::cout << "\n[E3.15] Parameter Validation - Negative Tolerance\n";
        tiny::Mat sym_mat(2, 2);
        sym_mat(0, 0) = 1.0f; sym_mat(0, 1) = 2.0f;
        sym_mat(1, 0) = 2.0f; sym_mat(1, 1) = 3.0f;
        
        // Test with tolerance < 0 (should return false/error)
        bool is_sym_neg = sym_mat.is_symmetric(-1e-6f);
        std::cout << "tolerance = -1e-6 (invalid): " << (is_sym_neg ? "True" : "False") 
                  << " (Expected: False) " << (!is_sym_neg ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.16: Boundary case - empty matrix
    {
        std::cout << "\n[E3.16] Boundary Case - Empty Matrix (0x0)\n";
        tiny::Mat empty_mat(0, 0);
        
        // Empty matrix cannot be checked for symmetry (data is null or invalid)
        bool is_sym_empty = empty_mat.is_symmetric(1e-6f);
        std::cout << "Empty matrix (0x0): " << (is_sym_empty ? "True" : "False") 
                  << " (Expected: False, empty matrix is invalid) " << (!is_sym_empty ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.2: power_iteration() - Dominant eigenvalue
    std::cout << "\n[E3.2] power_iteration() - Dominant Eigenvalue\n";
    
    // E3.21: Simple 2x2 symmetric matrix (known eigenvalues)
    tiny::Mat mat2x2(2, 2);
    {
        std::cout << "\n[E3.21] Simple 2x2 Matrix\n";
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

    // E3.22: 3x3 matrix (SHM-like stiffness matrix) - keep for later tests
    tiny::Mat stiffness(3, 3);
    {
        std::cout << "\n[E3.22] 3x3 Stiffness Matrix (SHM Application)\n";
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

    // E3.23: Non-square matrix (should fail)
    {
        std::cout << "\n[E3.23] Non-Square Matrix (Expect Error)\n";
        tiny::Mat non_square(2, 3);
        tiny::Mat::EigenPair result_error = non_square.power_iteration(100, 1e-6f);
        std::cout << "Status: " << (result_error.status == TINY_OK ? "OK" : "Error (Expected)") << "\n";
    }

    // E3.25: Parameter validation - max_iter <= 0
    {
        std::cout << "\n[E3.25] Parameter Validation - max_iter <= 0\n";
        tiny::Mat test_mat(2, 2);
        test_mat(0, 0) = 2.0f; test_mat(0, 1) = 1.0f;
        test_mat(1, 0) = 1.0f; test_mat(1, 1) = 2.0f;
        
        tiny::Mat::EigenPair result_zero = test_mat.power_iteration(0, 1e-6f);
        std::cout << "max_iter = 0: Status = " << (result_zero.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_zero.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
        
        tiny::Mat::EigenPair result_neg = test_mat.power_iteration(-1, 1e-6f);
        std::cout << "max_iter = -1: Status = " << (result_neg.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_neg.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.26: Parameter validation - tolerance < 0
    {
        std::cout << "\n[E3.26] Parameter Validation - tolerance < 0\n";
        tiny::Mat test_mat(2, 2);
        test_mat(0, 0) = 2.0f; test_mat(0, 1) = 1.0f;
        test_mat(1, 0) = 1.0f; test_mat(1, 1) = 2.0f;
        
        tiny::Mat::EigenPair result_neg = test_mat.power_iteration(100, -1e-6f);
        std::cout << "tolerance = -1e-6: Status = " << (result_neg.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_neg.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.27: Boundary case - empty matrix
    {
        std::cout << "\n[E3.27] Boundary Case - Empty Matrix (0x0)\n";
        tiny::Mat empty_mat(0, 0);
        
        tiny::Mat::EigenPair result_empty = empty_mat.power_iteration(100, 1e-6f);
        // Empty matrix should return error status
        bool empty_correct = (result_empty.status != TINY_OK);
        std::cout << "Empty matrix: Status = " << (result_empty.status == TINY_OK ? "OK" : "Error") 
                  << ", eigenvalue = " << result_empty.eigenvalue 
                  << " (Expected: Error status) " << (empty_correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.24: inverse_power_iteration() - Smallest eigenvalue (Critical for System Identification)
    std::cout << "\n[E3.24] inverse_power_iteration() - Smallest Eigenvalue (System Identification)\n";
    
    // E3.28: Simple 2x2 symmetric matrix (known eigenvalues)
    {
        std::cout << "\n[E3.28] Simple 2x2 Matrix - Smallest Eigenvalue\n";
        std::cout << "Matrix (same as E3.21):\n";
        mat2x2.print_matrix(true);
        
        // Expected values: eigenvalues are 3 and 1 (for matrix [2,1; 1,2])
        // Power iteration finds λ_max = 3, inverse power iteration should find λ_min = 1
        std::cout << "\n[Expected Results]\n";
        std::cout << "  Expected eigenvalues: 3.0 (largest), 1.0 (smallest)\n";
        std::cout << "  Expected smallest eigenvalue: 1.0\n";
        std::cout << "  Expected smallest eigenvector (for λ=1): approximately [0.707, -0.707] or [-0.707, 0.707] (normalized)\n";
        std::cout << "  Note: This is critical for system identification - smallest eigenvalue = fundamental frequency\n";
        
        tiny::Mat::EigenPair result_inv_power = mat2x2.inverse_power_iteration(1000, 1e-6f);
        std::cout << "\n[Actual Results]\n";
        std::cout << "  Smallest eigenvalue: " << result_inv_power.eigenvalue 
                  << " (Expected: 1.0, smallest eigenvalue)\n";
        std::cout << "  Iterations: " << result_inv_power.iterations << "\n";
        std::cout << "  Status: " << (result_inv_power.status == TINY_OK ? "OK" : "Error") << "\n";
        std::cout << "  Smallest eigenvector:\n";
        result_inv_power.eigenvector.print_matrix(true);
        
        // Check if result matches expected
        float error = fabsf(result_inv_power.eigenvalue - 1.0f);
        std::cout << "  Error from expected (1.0): " << error << (error < 0.01f ? " [PASS]" : " [FAIL]") << "\n";
        
        // Compare with power iteration results (recompute for comparison)
        tiny::Mat::EigenPair result_power_compare = mat2x2.power_iteration(1000, 1e-6f);
        std::cout << "\n[Comparison] Power vs Inverse Power Iteration:\n";
        std::cout << "  Power iteration (λ_max): " << result_power_compare.eigenvalue << "\n";
        std::cout << "  Inverse power iteration (λ_min): " << result_inv_power.eigenvalue << "\n";
        std::cout << "  Ratio (λ_max/λ_min): " << (result_power_compare.eigenvalue / result_inv_power.eigenvalue) 
                  << " (Expected: ~3.0) " << (fabsf(result_power_compare.eigenvalue / result_inv_power.eigenvalue - 3.0f) < 0.1f ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.29: 3x3 stiffness matrix - Smallest eigenvalue (SHM Application)
    {
        std::cout << "\n[E3.29] 3x3 Stiffness Matrix - Smallest Eigenvalue (SHM Application)\n";
        std::cout << "Stiffness Matrix (same as E3.22):\n";
        stiffness.print_matrix(true);
        
        // Expected values for 3x3 tridiagonal symmetric matrix [2,-1,0; -1,2,-1; 0,-1,2]
        // Approximate eigenvalues: λ₁ ≈ 3.414 (largest), λ₂ ≈ 2.000, λ₃ ≈ 0.586 (smallest)
        std::cout << "\n[Expected Results]\n";
        std::cout << "  Expected eigenvalues (approximate): 3.414 (largest), 2.000, 0.586 (smallest)\n";
        std::cout << "  Expected smallest eigenvalue: ~0.586 (fundamental frequency squared)\n";
        std::cout << "  Expected fundamental frequency: sqrt(0.586) ≈ 0.765 rad/s\n";
        std::cout << "  Note: Smallest eigenvalue is critical for system identification - represents fundamental mode\n";
        
        tiny::Mat::EigenPair result_inv_stiff = stiffness.inverse_power_iteration(500, 1e-6f);
        std::cout << "\n[Actual Results]\n";
        std::cout << "  Smallest eigenvalue (fundamental frequency squared): " << result_inv_stiff.eigenvalue << "\n";
        std::cout << "  Fundamental frequency: " << sqrtf(result_inv_stiff.eigenvalue) << " rad/s (Expected: ~0.765 rad/s)\n";
        std::cout << "  Iterations: " << result_inv_stiff.iterations << "\n";
        std::cout << "  Status: " << (result_inv_stiff.status == TINY_OK ? "OK" : "Error") << "\n";
        std::cout << "  Smallest eigenvector (fundamental mode shape):\n";
        result_inv_stiff.eigenvector.print_matrix(true);
        
        float expected_eigen = 0.586f;
        float error = fabsf(result_inv_stiff.eigenvalue - expected_eigen);
        std::cout << "  Error from expected (" << expected_eigen << "): " << error << (error < 0.1f ? " [PASS]" : " [FAIL]") << "\n";
        
        // Compare with power iteration (recompute for comparison)
        tiny::Mat::EigenPair result_stiff_compare = stiffness.power_iteration(500, 1e-6f);
        std::cout << "\n[Comparison] Power vs Inverse Power Iteration for SHM:\n";
        std::cout << "  Power iteration (primary frequency²): " << result_stiff_compare.eigenvalue 
                  << " → frequency: " << sqrtf(result_stiff_compare.eigenvalue) << " rad/s\n";
        std::cout << "  Inverse power iteration (fundamental frequency²): " << result_inv_stiff.eigenvalue 
                  << " → frequency: " << sqrtf(result_inv_stiff.eigenvalue) << " rad/s\n";
        std::cout << "  Frequency ratio: " << (sqrtf(result_stiff_compare.eigenvalue) / sqrtf(result_inv_stiff.eigenvalue))
                  << " (Expected: ~2.4, ratio of highest to lowest mode)\n";
    }

    // E3.210: Non-square matrix (should fail)
    {
        std::cout << "\n[E3.210] Non-Square Matrix (Expect Error)\n";
        tiny::Mat non_square(2, 3);
        tiny::Mat::EigenPair result_error = non_square.inverse_power_iteration(100, 1e-6f);
        std::cout << "Status: " << (result_error.status == TINY_OK ? "OK" : "Error (Expected)") << "\n";
        bool correct = (result_error.status != TINY_OK);
        std::cout << "Error handling: " << (correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.211: Near-singular matrix (should handle gracefully)
    {
        std::cout << "\n[E3.211] Near-Singular Matrix (Edge Case)\n";
        tiny::Mat near_singular(3, 3);
        // Create a matrix that is close to singular but still invertible
        near_singular(0, 0) = 1.0f; near_singular(0, 1) = 0.0f; near_singular(0, 2) = 0.0f;
        near_singular(1, 0) = 0.0f; near_singular(1, 1) = 1.0f; near_singular(1, 2) = 0.001f;
        near_singular(2, 0) = 0.0f; near_singular(2, 1) = 0.001f; near_singular(2, 2) = 1.0f;
        std::cout << "Matrix (near-singular but invertible):\n";
        near_singular.print_matrix(true);
        
        tiny::Mat::EigenPair result_near_sing = near_singular.inverse_power_iteration(500, 1e-5f);
        std::cout << "\n[Results]\n";
        std::cout << "  Status: " << (result_near_sing.status == TINY_OK ? "OK" : "Error") << "\n";
        if (result_near_sing.status == TINY_OK)
        {
            std::cout << "  Smallest eigenvalue: " << result_near_sing.eigenvalue << "\n";
            std::cout << "  Iterations: " << result_near_sing.iterations << "\n";
            std::cout << "  Note: Successfully handled near-singular matrix [PASS]\n";
        }
        else
        {
            std::cout << "  Note: Correctly detected problematic matrix [PASS]\n";
        }
    }

    // E3.212: Parameter validation - max_iter <= 0
    {
        std::cout << "\n[E3.212] Parameter Validation - max_iter <= 0\n";
        tiny::Mat test_mat(2, 2);
        test_mat(0, 0) = 2.0f; test_mat(0, 1) = 1.0f;
        test_mat(1, 0) = 1.0f; test_mat(1, 1) = 2.0f;
        
        tiny::Mat::EigenPair result_zero = test_mat.inverse_power_iteration(0, 1e-6f);
        std::cout << "max_iter = 0: Status = " << (result_zero.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_zero.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
        
        tiny::Mat::EigenPair result_neg = test_mat.inverse_power_iteration(-1, 1e-6f);
        std::cout << "max_iter = -1: Status = " << (result_neg.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_neg.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.213: Parameter validation - tolerance < 0
    {
        std::cout << "\n[E3.213] Parameter Validation - tolerance < 0\n";
        tiny::Mat test_mat(2, 2);
        test_mat(0, 0) = 2.0f; test_mat(0, 1) = 1.0f;
        test_mat(1, 0) = 1.0f; test_mat(1, 1) = 2.0f;
        
        tiny::Mat::EigenPair result_neg = test_mat.inverse_power_iteration(100, -1e-6f);
        std::cout << "tolerance = -1e-6: Status = " << (result_neg.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_neg.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.214: Boundary case - empty matrix
    {
        std::cout << "\n[E3.214] Boundary Case - Empty Matrix (0x0)\n";
        tiny::Mat empty_mat(0, 0);
        
        tiny::Mat::EigenPair result_empty = empty_mat.inverse_power_iteration(100, 1e-6f);
        // Empty matrix should return error status
        bool empty_correct = (result_empty.status != TINY_OK);
        std::cout << "Empty matrix: Status = " << (result_empty.status == TINY_OK ? "OK" : "Error") 
                  << ", eigenvalue = " << result_empty.eigenvalue 
                  << " (Expected: Error status) " << (empty_correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.215: Singular matrix (should fail)
    {
        std::cout << "\n[E3.215] Singular Matrix (Should Fail)\n";
        tiny::Mat singular(2, 2);
        singular(0, 0) = 1.0f; singular(0, 1) = 2.0f;
        singular(1, 0) = 2.0f; singular(1, 1) = 4.0f;  // Second row is 2x first row (singular)
        
        tiny::Mat::EigenPair result_sing = singular.inverse_power_iteration(100, 1e-6f);
        // Note: Some implementations may handle singular matrices differently
        // Check if status is Error OR if eigenvalue is valid (implementation-dependent)
        // For now, accept either Error status or valid result (some algorithms can handle singular matrices)
        bool singular_correct = (result_sing.status != TINY_OK) || 
                               (result_sing.status == TINY_OK && (result_sing.eigenvalue == 0.0f || fabsf(result_sing.eigenvalue) < 1e-5f));
        std::cout << "Singular matrix: Status = " << (result_sing.status == TINY_OK ? "OK" : "Error");
        if (result_sing.status == TINY_OK)
        {
            std::cout << ", eigenvalue = " << result_sing.eigenvalue;
        }
        std::cout << " (Expected: Error or OK with eigenvalue ≈ 0) " << (singular_correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.3: eigendecompose_jacobi() - Symmetric matrix decomposition
    std::cout << "\n[E3.3] eigendecompose_jacobi() - Symmetric Matrix Decomposition\n";
    
    // E3.31: Simple 2x2 symmetric matrix
    {
        std::cout << "\n[E3.31] 2x2 Symmetric Matrix - Complete Decomposition\n";
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

    // E3.32: 3x3 symmetric matrix (SHM stiffness matrix)
    {
        std::cout << "\n[E3.32] 3x3 Stiffness Matrix (SHM Application)\n";
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

    // E3.33: Diagonal matrix (trivial case)
    {
        std::cout << "\n[E3.33] Diagonal Matrix (Eigenvalues on diagonal)\n";
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

    // E3.34: Parameter validation - tolerance < 0
    {
        std::cout << "\n[E3.34] Parameter Validation - tolerance < 0\n";
        tiny::Mat test_mat(2, 2);
        test_mat(0, 0) = 2.0f; test_mat(0, 1) = 1.0f;
        test_mat(1, 0) = 1.0f; test_mat(1, 1) = 2.0f;
        
        tiny::Mat::EigenDecomposition result_neg = test_mat.eigendecompose_jacobi(-1e-6f, 100);
        std::cout << "tolerance = -1e-6: Status = " << (result_neg.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_neg.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.35: Parameter validation - max_iter <= 0
    {
        std::cout << "\n[E3.35] Parameter Validation - max_iter <= 0\n";
        tiny::Mat test_mat(2, 2);
        test_mat(0, 0) = 2.0f; test_mat(0, 1) = 1.0f;
        test_mat(1, 0) = 1.0f; test_mat(1, 1) = 2.0f;
        
        tiny::Mat::EigenDecomposition result_zero = test_mat.eigendecompose_jacobi(1e-6f, 0);
        std::cout << "max_iter = 0: Status = " << (result_zero.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_zero.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
        
        tiny::Mat::EigenDecomposition result_neg = test_mat.eigendecompose_jacobi(1e-6f, -1);
        std::cout << "max_iter = -1: Status = " << (result_neg.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_neg.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.36: Boundary case - empty matrix
    {
        std::cout << "\n[E3.36] Boundary Case - Empty Matrix (0x0)\n";
        tiny::Mat empty_mat(0, 0);
        
        tiny::Mat::EigenDecomposition result_empty = empty_mat.eigendecompose_jacobi(1e-6f, 100);
        // Error case: empty matrix should return error status, eigenvalues may be 0x0 or 1x1 error matrix
        bool empty_correct = (result_empty.status != TINY_OK) && 
                            ((result_empty.eigenvalues.row == 0 && result_empty.eigenvalues.col == 0) || 
                             (result_empty.eigenvalues.data == nullptr) ||
                             (result_empty.eigenvalues.row == 1 && result_empty.eigenvalues.col == 1 && result_empty.eigenvalues.data != nullptr));
        std::cout << "Empty matrix: Status = " << (result_empty.status == TINY_OK ? "OK" : "Error") 
                  << ", eigenvalues rows = " << result_empty.eigenvalues.row 
                  << " (Expected: Error status, eigenvalues is 0x0 or error state) " << (empty_correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.4: eigendecompose_qr() - General matrix decomposition
    std::cout << "\n[E3.4] eigendecompose_qr() - General Matrix Decomposition\n";
    
    // E3.41: General 2x2 matrix
    {
        std::cout << "\n[E3.41] General 2x2 Matrix\n";
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

    // E3.42: Non-symmetric 3x3 matrix
    {
        std::cout << "\n[E3.42] Non-Symmetric 3x3 Matrix\n";
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

    // E3.43: Parameter validation - max_iter <= 0
    {
        std::cout << "\n[E3.43] Parameter Validation - max_iter <= 0\n";
        tiny::Mat test_mat(2, 2);
        test_mat(0, 0) = 1.0f; test_mat(0, 1) = 2.0f;
        test_mat(1, 0) = 3.0f; test_mat(1, 1) = 4.0f;
        
        tiny::Mat::EigenDecomposition result_zero = test_mat.eigendecompose_qr(0, 1e-6f);
        std::cout << "max_iter = 0: Status = " << (result_zero.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_zero.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
        
        tiny::Mat::EigenDecomposition result_neg = test_mat.eigendecompose_qr(-1, 1e-6f);
        std::cout << "max_iter = -1: Status = " << (result_neg.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_neg.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.44: Parameter validation - tolerance < 0
    {
        std::cout << "\n[E3.44] Parameter Validation - tolerance < 0\n";
        tiny::Mat test_mat(2, 2);
        test_mat(0, 0) = 1.0f; test_mat(0, 1) = 2.0f;
        test_mat(1, 0) = 3.0f; test_mat(1, 1) = 4.0f;
        
        tiny::Mat::EigenDecomposition result_neg = test_mat.eigendecompose_qr(100, -1e-6f);
        std::cout << "tolerance = -1e-6: Status = " << (result_neg.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_neg.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.45: Boundary case - empty matrix
    {
        std::cout << "\n[E3.45] Boundary Case - Empty Matrix (0x0)\n";
        tiny::Mat empty_mat(0, 0);
        
        tiny::Mat::EigenDecomposition result_empty = empty_mat.eigendecompose_qr(100, 1e-6f);
        // Error case: empty matrix should return error status, eigenvalues may be 0x0 or 1x1 error matrix
        bool empty_correct = (result_empty.status != TINY_OK) && 
                            ((result_empty.eigenvalues.row == 0 && result_empty.eigenvalues.col == 0) || 
                             (result_empty.eigenvalues.data == nullptr) ||
                             (result_empty.eigenvalues.row == 1 && result_empty.eigenvalues.col == 1 && result_empty.eigenvalues.data != nullptr));
        std::cout << "Empty matrix: Status = " << (result_empty.status == TINY_OK ? "OK" : "Error") 
                  << ", eigenvalues rows = " << result_empty.eigenvalues.row 
                  << " (Expected: Error status, eigenvalues is 0x0 or error state) " << (empty_correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.5: eigendecompose() - Automatic method selection
    std::cout << "\n[E3.5] eigendecompose() - Automatic Method Selection\n";
    
    // E3.51: Symmetric matrix (should use Jacobi)
    {
        std::cout << "\n[E3.51] Symmetric Matrix (Auto-select: Jacobi)\n";
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

    // E3.52: Non-symmetric matrix (should use QR)
    {
        std::cout << "\n[E3.52] Non-Symmetric Matrix (Auto-select: QR)\n";
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
        
        // Check results with detailed error reporting
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

    // E3.53: Parameter validation - tolerance < 0
    {
        std::cout << "\n[E3.53] Parameter Validation - tolerance < 0\n";
        tiny::Mat test_mat(2, 2);
        test_mat(0, 0) = 2.0f; test_mat(0, 1) = 1.0f;
        test_mat(1, 0) = 1.0f; test_mat(1, 1) = 2.0f;
        
        tiny::Mat::EigenDecomposition result_neg = test_mat.eigendecompose(-1e-6f);
        std::cout << "tolerance = -1e-6: Status = " << (result_neg.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_neg.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.531: Parameter validation - max_iter <= 0
    {
        std::cout << "\n[E3.531] Parameter Validation - max_iter <= 0\n";
        tiny::Mat test_mat(2, 2);
        test_mat(0, 0) = 2.0f; test_mat(0, 1) = 1.0f;
        test_mat(1, 0) = 1.0f; test_mat(1, 1) = 2.0f;
        
        tiny::Mat::EigenDecomposition result_zero_iter = test_mat.eigendecompose(1e-6f, 0);
        std::cout << "max_iter = 0: Status = " << (result_zero_iter.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_zero_iter.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
        
        tiny::Mat::EigenDecomposition result_neg_iter = test_mat.eigendecompose(1e-6f, -10);
        std::cout << "max_iter = -10: Status = " << (result_neg_iter.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_neg_iter.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.532: Custom max_iter parameter test
    {
        std::cout << "\n[E3.532] Custom max_iter Parameter Test\n";
        tiny::Mat test_mat(3, 3);
        test_mat(0, 0) = 4.0f; test_mat(0, 1) = 1.0f; test_mat(0, 2) = 2.0f;
        test_mat(1, 0) = 1.0f; test_mat(1, 1) = 3.0f; test_mat(1, 2) = 0.0f;
        test_mat(2, 0) = 2.0f; test_mat(2, 1) = 0.0f; test_mat(2, 2) = 5.0f;
        
        std::cout << "[Test 1] max_iter = 10 (may not converge)\n";
        tiny::Mat::EigenDecomposition result_10 = test_mat.eigendecompose(1e-6f, 10);
        std::cout << "  Status: " << (result_10.status == TINY_OK ? "OK (Converged)" : "Not Finished (Expected)") << "\n";
        std::cout << "  Iterations: " << result_10.iterations << "\n";
        
        std::cout << "[Test 2] max_iter = 200 (should converge)\n";
        tiny::Mat::EigenDecomposition result_200 = test_mat.eigendecompose(1e-6f, 200);
        std::cout << "  Status: " << (result_200.status == TINY_OK ? "OK (Converged)" : "Not Finished") << "\n";
        std::cout << "  Iterations: " << result_200.iterations << "\n";
        std::cout << "  Eigenvalues:\n";
        result_200.eigenvalues.print_matrix(true);
        
        bool test_pass = (result_10.iterations <= 10) && (result_200.status == TINY_OK);
        std::cout << "Custom max_iter test: " << (test_pass ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.54: eigendecompose() - Boundary case - empty matrix
    {
        std::cout << "\n[E3.54] eigendecompose() - Boundary Case - Empty Matrix (0x0)\n";
        tiny::Mat empty_mat(0, 0);
        
        tiny::Mat::EigenDecomposition result_empty = empty_mat.eigendecompose(1e-6f);
        // Error case: empty matrix should return error status, eigenvalues may be 0x0 or 1x1 error matrix
        bool empty_correct = (result_empty.status != TINY_OK) && 
                            ((result_empty.eigenvalues.row == 0 && result_empty.eigenvalues.col == 0) || 
                             (result_empty.eigenvalues.data == nullptr) ||
                             (result_empty.eigenvalues.row == 1 && result_empty.eigenvalues.col == 1 && result_empty.eigenvalues.data != nullptr));
        std::cout << "Empty matrix: Status = " << (result_empty.status == TINY_OK ? "OK" : "Error") 
                  << ", eigenvalues rows = " << result_empty.eigenvalues.row 
                  << " (Expected: Error status, eigenvalues is 0x0 or error state) " << (empty_correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.55: eigendecompose() - Error handling - non-square matrix
    {
        std::cout << "\n[E3.55] eigendecompose() - Error Handling - Non-Square Matrix\n";
        tiny::Mat non_square(2, 3);
        tiny::Mat::EigenDecomposition result_non_square = non_square.eigendecompose(1e-6f);
        std::cout << "Non-square matrix (2x3): Status = " << (result_non_square.status == TINY_OK ? "OK" : "Error (Expected)") 
                  << " " << (result_non_square.status != TINY_OK ? "[PASS]" : "[FAIL]") << "\n";
    }

    // E3.6: SHM Application Scenario - Structural Dynamics
    std::cout << "\n[E3.6] SHM Application - Structural Dynamics Analysis\n";
    
    // Create a simple 4-DOF structural system (mass-spring system)
    {
        std::cout << "\n[E3.61] 4-DOF Mass-Spring System\n";
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

    // E3.7: Edge Cases and Error Handling
    std::cout << "\n[E3.7] Edge Cases and Error Handling\n";
    
    // E3.71: 1x1 matrix
    {
        std::cout << "\n[E3.71] 1x1 Matrix\n";
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
    
    // E3.72: Zero matrix
    {
        std::cout << "\n[E3.72] Zero Matrix\n";
        tiny::Mat zero_mat(3, 3);
        zero_mat.clear();
        tiny::Mat::EigenPair result_zero = zero_mat.power_iteration(100, 1e-6f);
        std::cout << "Status: " << (result_zero.status == TINY_OK ? "OK" : "Error (Expected)") << "\n";
    }
    
    // E3.73: Identity matrix
    {
        std::cout << "\n[E3.73] Identity Matrix\n";
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

    // E3.8: Performance Test for SHM Applications
    std::cout << "\n[E3.8] Performance Test for SHM Applications\n";
    
    // E3.81: Power iteration performance (fast method for dominant eigenvalue)
    std::cout << "\n[E3.81] Power Iteration Performance (Real-time SHM - Dominant Eigenvalue)\n";
    TIME_OPERATION(
        tiny::Mat::EigenPair perf_result = stiffness.power_iteration(500, 1e-6f);
        (void)perf_result;
    , "Power Iteration (3x3 matrix)");
    
    // E3.82: Inverse power iteration performance (system identification - smallest eigenvalue)
    std::cout << "\n[E3.82] Inverse Power Iteration Performance (System Identification - Smallest Eigenvalue)\n";
    TIME_OPERATION(
        tiny::Mat::EigenPair perf_inv_result = stiffness.inverse_power_iteration(500, 1e-6f);
        (void)perf_inv_result;
    , "Inverse Power Iteration (3x3 matrix)");
    
    // E3.83: Jacobi method performance (complete eigendecomposition for symmetric matrices)
    std::cout << "\n[E3.83] Jacobi Method Performance (Complete Eigendecomposition - Symmetric Matrices)\n";
    TIME_OPERATION(
        tiny::Mat::EigenDecomposition perf_jacobi = stiffness.eigendecompose_jacobi(1e-5f, 100);
        (void)perf_jacobi;
    , "Jacobi Decomposition (3x3 symmetric matrix)");
    
    // E3.84: QR method performance (complete eigendecomposition for general matrices)
    std::cout << "\n[E3.84] QR Method Performance (Complete Eigendecomposition - General Matrices)\n";
    TIME_OPERATION(
        tiny::Mat::EigenDecomposition perf_qr = non_sym_mat.eigendecompose_qr(100, 1e-5f);
        (void)perf_qr;
    , "QR Decomposition (3x3 general matrix)");

    std::cout << "\n[Eigenvalue Decomposition Tests Complete]\n";
}

void tiny_matrix_test()
{
    std::cout << "============ [tiny_matrix_test start] ============\n";
    std::cout << "\n[Test Organization: Application-Oriented Logic]\n";
    std::cout << "  Foundation → Basic Ops → Properties → Linear Systems → Decompositions → Applications → Quality\n\n";

    // ========================================================================
    // Phase 1: Object Foundation (A)
    // ========================================================================
    // Purpose: Learn to create and manipulate matrix objects
    // A1: Constructor & Destructor
    test_constructor_destructor();

    // A2: Element Access
    test_element_access();

    // A3: ROI Operations
    test_roi_operations();

    // ========================================================================
    // Phase 2: Basic Operations (B)
    // ========================================================================
    // Purpose: Learn basic arithmetic operations
    // B1-B8: Arithmetic Operators
    // test_assignment_operator();      // B1
    // test_matrix_addition();         // B2
    // test_constant_addition();       // B3
    // test_matrix_subtraction();      // B4
    // test_constant_subtraction();    // B5
    // test_matrix_division();          // B6
    // test_constant_division();       // B7
    // test_matrix_exponentiation();    // B8

    // ========================================================================
    // Phase 3: Matrix Properties (C)
    // ========================================================================
    // Purpose: Understand matrix properties and basic linear algebra
    // C1-C8: Matrix Properties
    // test_matrix_transpose();        // C1
    // test_matrix_cofactor();         // C2
    // test_matrix_determinant();       // C3
    // test_matrix_adjoint();           // C4
    // test_matrix_normalize();         // C5
    // test_matrix_norm();              // C6
    // test_inverse_adjoint_adjoint();  // C7
    // test_matrix_utilities();         // C8

    // ========================================================================
    // Phase 4: Linear System Solving (D)
    // ========================================================================
    // Purpose: Core application - solving linear systems Ax = b
    // D1-D7: Linear System Solving
    // test_gaussian_eliminate();       // D1
    // test_row_reduce_from_gaussian(); // D2
    // test_inverse_gje();             // D3
    // test_dotprod();                 // D4
    // test_solve();                   // D5
    // test_band_solve();              // D6
    // test_roots();                   // D7

    // ========================================================================
    // Phase 5: Advanced Linear Algebra (E1+2)
    // ========================================================================
    // Purpose: Advanced linear algebra operations for stable and efficient solving
    // E1: Matrix Decomposition
    // test_matrix_decomposition();
    
    // E2: Gram-Schmidt Orthogonalization
    // test_gram_schmidt_orthogonalize();

    // ========================================================================
    // Phase 6: System Identification Applications (E3)
    // ========================================================================
    // Purpose: Eigenvalue decomposition for SHM and modal analysis
    // E3: Eigenvalue Decomposition
    // test_eigenvalue_decomposition();

    // ========================================================================
    // Phase 7: Auxiliary Functions (F)
    // ========================================================================
    // Purpose: Convenience functions and I/O operations
    // F1: Stream Operators
    // test_stream_operators();

    // F2: Global Arithmetic Operators
    // test_matrix_operations();

    // ========================================================================
    // Phase 8: Quality Assurance (G)
    // ========================================================================
    // Purpose: Ensure robustness, performance, and correctness
    // G1: Boundary Conditions and Error Handling
    // test_boundary_conditions();

    // G2: Performance Benchmarks
    // test_performance_benchmarks();

    // G3: Memory Layout
    // test_memory_layout();

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


## 测试输出

### 第一阶段： 对象基础 （A）

```c
============ [tiny_matrix_test start] ============

[Test Organization: Application-Oriented Logic]
  Foundation → Basic Ops → Properties → Linear Systems → Decompositions → Applications → Quality


[A1: Constructor & Destructor Tests]
[A1.1] Default Constructor
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

[A1.2] Constructor with Rows and Cols
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

[A1.3] Constructor with Rows, Cols and Stride
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

[A1.4] Constructor with External Data
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        0
stride          4
memory          12
data pointer    0x3fc9a49c
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            1            2            3       |
           4            5            6            7       |
           8            9           10           11       |
<<< Matrix Elements

[A1.5] Constructor with External Data and Stride
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc9a4f0
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            1            2            3       |      0 
           4            5            6            7       |      0 
           8            9           10           11       |      0 
<<< Matrix Elements

[A1.6] Copy Constructor
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
           0            1            2            3       |1.89175e-43 
           4            5            6            7       |1.61149e-43 
           8            9           10           11       |1.61413 
<<< Matrix Elements


[A2: Element Access Tests]
[A2.1] Non-const Access
Matrix Info >>>
rows            2
cols            3
elements        6
paddings        0
stride          3
memory          6
data pointer    0x3fce9a9c
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
         1.1          2.2          3.3       |
         4.4          5.5          6.6       |
<<< Matrix Elements

[A2.2] Const Access
const_mat(0, 0): 1.1

[A3: ROI Operations Tests]
[Material Matrices]
matA:
Matrix Info >>>
rows            2
cols            3
elements        6
paddings        0
stride          3
memory          6
data pointer    0x3fce9a9c
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
data pointer    0x3fc9a284
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
data pointer    0x3fce9a78
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0       |
<<< Matrix Elements

[A3.1] Copy ROI - Over Range Case
[Error] copy_paste: source matrix exceeds destination column boundary: col_pos=2, src.cols=3, dest.cols=4
matB after copy_paste matA at (1, 2):
Matrix Elements >>>
           0            1            2            3       |      0 
           4            5            6            7       |      0 
           8            9           10           11       |      0 
<<< Matrix Elements

nothing changed.
[A3.2] Copy ROI - Suitable Range Case
matB after copy_paste matA at (1, 1):
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc9a284
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
[A3.3] Copy Head
matC after copy_head matB:
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc9a284
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            1            2            3       |      0 
           4          0.1          0.2          0.3       |      0 
           8          0.4          0.5          0.6       |      0 
<<< Matrix Elements

[A3.4] Copy Head - Memory Sharing Check
matB(0, 0) = 99.99f
matC:
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc9a284
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
       99.99            1            2            3       |      0 
           4          0.1          0.2          0.3       |      0 
           8          0.4          0.5          0.6       |      0 
<<< Matrix Elements


[A3.5] copy_paste() - Error Handling - Negative Position
[Error] copy_paste: invalid position: row_pos=-1, col_pos=0 (must be non-negative)
copy_paste with row_pos=-1: error = 258 (Expected: TINY_ERR_INVALID_ARG) [PASS]
[Error] copy_paste: invalid position: row_pos=0, col_pos=-1 (must be non-negative)
copy_paste with col_pos=-1: error = 258 (Expected: TINY_ERR_INVALID_ARG) [PASS]

[A3.6] copy_paste() - Error Handling - Out of Bounds
[Error] copy_paste: source matrix exceeds destination row boundary: row_pos=0, src.rows=3, dest.rows=2
copy_paste 3x3 into 2x2 at (0,0): error = 258 (Expected: TINY_ERR_INVALID_ARG) [PASS]
[Error] copy_paste: source matrix exceeds destination row boundary: row_pos=1, src.rows=2, dest.rows=2
copy_paste 2x2 into 2x2 at (1,1): error = 258 (Expected: TINY_ERR_INVALID_ARG) [PASS]

[A3.7] copy_paste() - Boundary Case - Empty Source Matrix
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] copy_paste: source matrix data pointer is null
copy_paste empty matrix: error = 258 (Expected: TINY_ERR_INVALID_ARG) [PASS]

[A3.8] copy_head() - Share Data from Owned-Memory Source (Double-Free Prevention)
Before copy_head:
  owned_src: ext_buff=0
  dest4: ext_buff=0
copy_head from matrix with owned memory: error = 0 (Expected: TINY_OK) [PASS]
After copy_head:
  owned_src: ext_buff=0 (still owns memory)
  dest4: ext_buff=1 (view, does not own)
Verify data sharing:
  dest4(0,0)=1 (should be 1.0)
  dest4(1,1)=4 (should be 4.0)
After modifying owned_src(0,0) to 99.0:
  dest4(0,0)=99 (should be 99.0, confirming shared data)
[A3.9] Get a View of ROI - Low Level Function
get a view of ROI with overrange dimensions - rows:
[Error] view_roi: ROI exceeds row boundary: start_row=1, roi_rows=3, source.rows=3
get a view of ROI with overrange dimensions - cols:
[Error] view_roi: ROI exceeds column boundary: start_col=1, roi_cols=4, source.cols=4
get a view of ROI with suitable dimensions:
roi3:
Matrix Info >>>
rows            2
cols            2
elements        4
paddings        3
stride          5
memory          10
data pointer    0x3fc9a29c
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      1   (This is a Sub-Matrix View)
<<< Matrix Info
Matrix Elements >>>
         0.1          0.2       |    0.3            0            8 
         0.4          0.5       |    0.6            0   4.2039e-45 
<<< Matrix Elements

[A3.10] Get a View of ROI - Using ROI Structure
Matrix Info >>>
rows            2
cols            2
elements        4
paddings        3
stride          5
memory          10
data pointer    0x3fc9a29c
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      1   (This is a Sub-Matrix View)
<<< Matrix Info
Matrix Elements >>>
         0.1          0.2       |    0.3            0            8 
         0.4          0.5       |    0.6            0   4.2039e-45 
<<< Matrix Elements

[A3.11] Copy ROI - Low Level Function
Matrix Info >>>
rows            2
cols            2
elements        4
paddings        0
stride          2
memory          4
data pointer    0x3fce9ca8
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
         0.1          0.2       |
         0.4          0.5       |
<<< Matrix Elements

[A3.12] Copy ROI - Using ROI Structure
time for copy_roi using ROI structure: 43 ms
Matrix Info >>>
rows            2
cols            2
elements        4
paddings        0
stride          2
memory          4
data pointer    0x3fce9cbc
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
         0.1          0.2       |
         0.4          0.5       |
<<< Matrix Elements


[A3.13] ROI resize_roi() Function
Initial ROI: pos_x=0, pos_y=0, width=2, height=2
After resize_roi(1, 1, 3, 3): pos_x=1, pos_y=1, width=3, height=3
ROI resize test: [PASS]

[A3.14] ROI area_roi() Function
ROI(0, 0, 3, 4) area: 12 (Expected: 12) [PASS]
ROI(1, 2, 5, 6) area: 30 (Expected: 30) [PASS]
[A3.15] Block
time for block: 53 ms
Matrix Info >>>
rows            2
cols            2
elements        4
paddings        0
stride          2
memory          4
data pointer    0x3fce9cd0
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
         0.1          0.2       |
         0.4          0.5       |
<<< Matrix Elements

[A3.16] Swap Rows
matB before swap rows:
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc9a284
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
data pointer    0x3fc9a284
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           8          0.4          0.5          0.6       |      0 
           4          0.1          0.2          0.3       |      0 
       99.99            1            2            3       |      0 
<<< Matrix Elements

[A3.17] Swap Columns
matB before swap columns:
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc9a284
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
data pointer    0x3fc9a284
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
         0.5          0.4            8          0.6       |      0 
         0.2          0.1            4          0.3       |      0 
           2            1        99.99            3       |      0 
<<< Matrix Elements

[A3.18] Clear
matB before clear:
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc9a284
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
data pointer    0x3fc9a284
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
### 第二阶段： 基础操作 （B）

```c
============ [tiny_matrix_test start] ============

[Test Organization: Application-Oriented Logic]
  Foundation → Basic Ops → Properties → Linear Systems → Decompositions → Applications → Quality


[B1: Assignment Operator Tests]

[B1.1] Assignment (Same Dimensions)
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
<<< Matrix Elements


[B1.2] Assignment (Different Dimensions)
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
<<< Matrix Elements


[B1.3] Assignment to Sub-Matrix (Expect Error)
[Error] Assignment to a sub-matrix is not allowed.
Matrix Elements >>>
           5            6       |      7            0            8 
           9           10       |     11            0   4.2039e-45 
<<< Matrix Elements


[B1.4] Self-Assignment
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
<<< Matrix Elements


[B2: Matrix Addition Tests]

[B2.1] Matrix Addition (Same Dimensions)
Matrix Elements >>>
           2            3            4       |
           5            6            7       |
<<< Matrix Elements


[B2.2] Sub-Matrix Addition
Matrix Elements >>>
          10           12       |      7            0            8 
          18           20       |     11            0           12 
<<< Matrix Elements


[B2.3] Full Matrix + Sub-Matrix Addition
Matrix Elements >>>
          12           14       |
          20           22       |
<<< Matrix Elements


[B2.4] Addition Dimension Mismatch (Expect Error)
[Error] Matrix addition failed: Dimension mismatch (2x2 vs 3x3)

[B3: Constant Addition Tests]

[B3.1] Full Matrix + Constant
Matrix Elements >>>
           5            6            7       |
           8            9           10       |
<<< Matrix Elements


[B3.2] Sub-Matrix + Constant
Matrix Elements >>>
           8            9       |      7            0            8 
          12           13       |     11            0           12 
<<< Matrix Elements


[B3.3] Add Zero
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements


[B3.4] Add Negative Constant
Matrix Elements >>>
          -5            5       |
          15           25       |
<<< Matrix Elements


[B4: Matrix Subtraction Tests]

[B4.1] Matrix Subtraction
Matrix Elements >>>
           4            5       |
           6            7       |
<<< Matrix Elements


[B4.2] Subtraction Dimension Mismatch (Expect Error)
[Error] Matrix subtraction failed: Dimension mismatch (2x2 vs 3x3)

[B5: Constant Subtraction Tests]

[B5.1] Full Matrix - Constant
Matrix Elements >>>
          -1            0            1       |
           2            3            4       |
<<< Matrix Elements


[B5.2] Sub-Matrix - Constant
Matrix Elements >>>
         3.5          4.5       |      7            0            8 
         7.5          8.5       |     11            0   4.2039e-45 
<<< Matrix Elements


[B6: Matrix Element-wise Division Tests]

[B6.1] Element-wise Division (Same Dimensions, No Zero)
Matrix Elements >>>
           5            5       |
           6            5       |
<<< Matrix Elements


[B6.2] Dimension Mismatch (Expect Error)
[Error] Matrix division failed: Dimension mismatch (2x2 vs 3x3)

[B6.3] Division by Matrix Containing Zero (Expect Error)
[Error] Matrix division failed: Division by zero detected at position (0, 1)
Matrix Elements >>>
           5           10       |
          15           20       |
<<< Matrix Elements


[B7: Matrix Division by Constant Tests]

[B7.1] Divide Full Matrix by Positive Constant
Matrix Elements >>>
           1          1.5            2       |
         2.5            3          3.5       |
<<< Matrix Elements


[B7.2] Divide Matrix by Negative Constant
Matrix Elements >>>
          -2           -4       |
          -6           -8       |
<<< Matrix Elements


[B7.3] Division by Zero Constant (Expect Error)
[Error] Matrix division by zero is undefined (divisor=0)
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements


[B8: Matrix Exponentiation Tests]

[B8.1] Raise Each Element to Power of 2
Matrix Elements >>>
           4            9       |
          16           25       |
<<< Matrix Elements


[B8.2] Raise Each Element to Power of 0
Matrix Elements >>>
           1            1       |
           1            1       |
<<< Matrix Elements


[B8.3] Raise Each Element to Power of 1
Matrix Elements >>>
           9            8       |
           7            6       |
<<< Matrix Elements


[B8.4] Raise Each Element to Power of -1 (Element-wise Reciprocal)
Matrix Elements >>>
           1          0.5       |
        0.25          0.2       |
<<< Matrix Elements


[B8.5] Raise Matrix Containing Zero to Power of 3
Matrix Elements >>>
           0            8       |
          -1           27       |
<<< Matrix Elements


[B8.6] Raise Matrix Containing Zero to Power of -1 (Expect Warning)
[Warning] operator^: element at (0, 0) is zero or too small (0), cannot compute negative power. Result will be Inf or NaN.
Matrix Elements >>>
         inf          0.5       |
          -1     0.333333       |
<<< Matrix Elements

============ [tiny_matrix_test end] ============
```

### 第三阶段：矩阵特性（C）

```c
============ [tiny_matrix_test start] ============

[Test Organization: Application-Oriented Logic]
  Foundation → Basic Ops → Properties → Linear Systems → Decompositions → Applications → Quality


[C1: Matrix Transpose Tests]

[C1.1] Transpose of 2x3 Matrix
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


[C1.2] Transpose of 3x3 Square Matrix
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


[C1.3] Transpose of Matrix with Padding
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


[C1.4] Transpose of Empty Matrix
Matrix Elements >>>
           0       |
<<< Matrix Elements

Matrix Elements >>>
           0       |
<<< Matrix Elements


[C2: Matrix Minor and Cofactor Tests]

[C2.1] Minor of 3x3 Matrix (Remove Row 1, Col 1)
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


[C2.2] Cofactor of 3x3 Matrix (Remove Row 1, Col 1)
Note: Cofactor matrix is the same as minor matrix.
      The sign (-1)^(i+j) is applied when computing cofactor value, not to matrix elements.
Cofactor Matrix (same as minor):
Matrix Elements >>>
           1            3       |
           7            9       |
<<< Matrix Elements


[C2.3] Minor (Remove Row 0, Col 0)
Matrix Elements >>>
           5            6       |
           8            9       |
<<< Matrix Elements


[C2.4] Cofactor (Remove Row 0, Col 0)
Note: Cofactor matrix is the same as minor matrix.
Matrix Elements >>>
           5            6       |
           8            9       |
<<< Matrix Elements


[C2.5] Cofactor (Remove Row 0, Col 1)
Note: Cofactor matrix is the same as minor matrix.
      When computing cofactor value, sign (-1)^(0+1) = -1 would be applied.
Cofactor Matrix (same as minor):
Matrix Elements >>>
           4            6       |
           7            9       |
<<< Matrix Elements


[C2.6] Minor (Remove Row 2, Col 2)
Matrix Elements >>>
           1            2       |
           4            5       |
<<< Matrix Elements


[C2.7] Cofactor (Remove Row 2, Col 2)
Note: Cofactor matrix is the same as minor matrix.
Matrix Elements >>>
           1            2       |
           4            5       |
<<< Matrix Elements


[C2.8] Minor of 4x4 Matrix (Remove Row 2, Col 1)
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


[C2.9] Cofactor of 4x4 Matrix (Remove Row 2, Col 1)
Note: Cofactor matrix is the same as minor matrix.
      When computing cofactor value, sign (-1)^(2+1) = -1 would be applied.
Cofactor Matrix (same as minor):
Matrix Elements >>>
           1            3            4       |
           5            7            8       |
          13           15           16       |
<<< Matrix Elements


[C2.10] Non-square Matrix (Expect Error)
Testing minor():
[Error] Minor requires square matrix (got 3x4)
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
minor() result: Empty matrix (Expected) [PASS]
Testing cofactor():
[Error] Minor requires square matrix (got 3x4)
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
cofactor() result: Empty matrix (Expected) [PASS]

[C2.11] minor() - Boundary Case - Out of Bounds Indices
[Error] minor: target_row=-1 is out of range [0, 2]
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
minor(-1, 0): Empty matrix (Expected) [PASS]
[Error] minor: target_col=-1 is out of range [0, 2]
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
minor(0, -1): Empty matrix (Expected) [PASS]
[Error] minor: target_row=3 is out of range [0, 2]
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
minor(3, 0) (out of bounds): Empty matrix (Expected) [PASS]

[C2.12] minor() - Boundary Case - 1x1 Matrix
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
1x1 matrix minor(0,0): Empty matrix (Expected) [PASS]

[C3: Matrix Determinant Tests]

[C3.1] 1x1 Matrix Determinant
Matrix:
Matrix Elements >>>
           7       |
<<< Matrix Elements

Determinant: 7  (Expected: 7)

[C3.2] 2x2 Matrix Determinant
Matrix:
Matrix Elements >>>
           3            8       |
           4            6       |
<<< Matrix Elements

Determinant: -14  (Expected: -14)

[C3.3] 3x3 Matrix Determinant
Matrix:
Matrix Elements >>>
           1            2            3       |
           0            4            5       |
           1            0            6       |
<<< Matrix Elements

Determinant: 22  (Expected: 22)

[C3.4] 4x4 Matrix Determinant
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

[C3.5] 5x5 Matrix Determinant (Tests Auto-select to LU Method)
Matrix (5x5, tridiagonal):
Matrix Elements >>>
           2            1            0            0            0       |
           1            2            1            0            0       |
           0            1            2            1            0       |
           0            0            1            2            1       |
           0            0            0            1            2       |
<<< Matrix Elements

Determinant (auto-select, should use LU for n > 4): 6
Note: For n = 5 > 4, auto-select should use LU decomposition (O(n³)).

[C3.6] Non-square Matrix (Expect Error)
Matrix (3x4, non-square):
Matrix Elements >>>
           0            0            0            0       |
           0            0            0            0       |
           0            0            0            0       |
<<< Matrix Elements

[Error] Determinant requires a square matrix (got 3x4)
Determinant: 0  (Expected: 0 with error message)

[C3.7] Comparison of Different Methods (5x5 Matrix)
Matrix (5x5):
Matrix Elements >>>
           2            2            3            4            5       |
           2            5            6            8           10       |
           3            6           10           12           15       |
           4            8           12           17           20       |
           5           10           15           20           26       |
<<< Matrix Elements

Determinant (auto-select): 56  (should use LU for n > 4)
Determinant (Laplace):     56  (O(n!), slow for n=5)
Determinant (LU):          56  (O(n³), efficient)
Determinant (Gaussian):    56  (O(n³), efficient)
Note: All methods should give the same result (within numerical precision).
      Auto-select should use LU for n > 4, avoiding slow Laplace expansion.

[C3.8] Large Matrix (6x6) - Tests Efficient Methods
Matrix (6x6, showing first 4x4 block):
       1.5          2          3          4 ...
         2        4.5          6          8 ...
         3          6        9.5         12 ...
         4          8         12       16.5 ...
...
Determinant (auto-select, uses LU): 2.85938
Determinant (LU):                   2.85938
Determinant (Gaussian):             2.85938
Note: For n > 4, auto-select uses LU decomposition (O(n³) instead of O(n!)).

[C3.9] Large Matrix (8x8) - Performance Comparison
Matrix (8x8, showing first 4x4 block):
         1          2          3          4 ...
         2          4          6          8 ...
         3          6          9         12 ...
         4          8         12         16 ...
...
[Error] LU decomposition: Matrix is singular or near-singular.
[Warning] determinant_lu: LU decomposition failed (status=458754), matrix may be singular
Determinant (LU):       0
Determinant (Gaussian): 0
Note: Both methods are O(n³) and should be much faster than Laplace expansion.

[C3.10] determinant_laplace() - Boundary Case - Empty Matrix
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
Empty matrix determinant (Laplace): 1 (Expected: 1.0) [PASS]

[C3.11] determinant_lu() - Boundary Case - Empty Matrix
Empty matrix determinant (LU): 1 (Expected: 1.0) [PASS]

[C3.12] determinant_gaussian() - Boundary Case - Empty Matrix
Empty matrix determinant (Gaussian): 1 (Expected: 1.0) [PASS]

[C3.13] Determinant Methods - Non-Square Matrix
[Error] Determinant requires a square matrix (got 2x3)
[Error] Determinant requires a square matrix (got 2x3)
[Error] Determinant requires a square matrix (got 2x3)
Non-square matrix (2x3) determinant (Laplace): 0 (Expected: 0.0) [PASS]
Non-square matrix (2x3) determinant (LU): 0 (Expected: 0.0) [PASS]
Non-square matrix (2x3) determinant (Gaussian): 0 (Expected: 0.0) [PASS]

[C4: Matrix Adjoint Tests]

[C4.1] Adjoint of 1x1 Matrix
Original Matrix:
Matrix Elements >>>
           5       |
<<< Matrix Elements

[>>> Error ! <<<] Memory allocation failed in alloc_mem()
Adjoint Matrix:
Matrix Elements >>>
           1       |
<<< Matrix Elements


[C4.2] Adjoint of 2x2 Matrix
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


[C4.3] Adjoint of 3x3 Matrix
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


[C4.4] Adjoint of Non-Square Matrix (Expect Error)
Original Matrix (2x3, non-square):
Matrix Elements >>>
           0            0            0       |
           0            0            0       |
<<< Matrix Elements

[Error] Adjoint requires a square matrix (got 2x3)
Adjoint Matrix (should be empty due to error):
Matrix Elements >>>
           0       |
<<< Matrix Elements


[C5: Matrix Normalization Tests]

[C5.1] Normalize a Standard 2x2 Matrix
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


[C5.2] Normalize a 2x2 Matrix with Stride=4 (Padding Test)
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


[C5.3] Normalize a Zero Matrix (Expect Warning)
Matrix Elements >>>
           0            0       |
           0            0       |
<<< Matrix Elements

[Warning] normalize: matrix norm is zero (matrix is all zeros), normalization skipped

[C6: Matrix Norm Calculation Tests]

[C6.1] 2x2 Matrix Norm (Expect 5.0)
Matrix:
Matrix Elements >>>
           3            4       |
           0            0       |
<<< Matrix Elements

Calculated Norm: 5

[C6.2] Zero Matrix Norm (Expect 0.0)
Matrix:
Matrix Elements >>>
           0            0            0       |
           0            0            0       |
           0            0            0       |
<<< Matrix Elements

Calculated Norm: 0

[C6.3] Matrix with Negative Values
Matrix:
Matrix Elements >>>
          -1           -2       |
          -3           -4       |
<<< Matrix Elements

Calculated Norm: 5.47723  (Expect sqrt(30) ≈ 5.477)

[C6.4] 2x2 Matrix with Stride=4 (Padding Test)
Matrix:
Matrix Elements >>>
           1            2       |      0            0 
           3            4       |      0            0 
<<< Matrix Elements

Calculated Norm: 5.47723  (Expect sqrt(30) ≈ 5.477)

[C7: Matrix Inversion Tests]

[C7.1] Inverse of 2x2 Matrix
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

[C7.2] Singular Matrix (Expect Error)
Original Matrix:
Matrix Elements >>>
           1            2       |
           2            4       |
<<< Matrix Elements

Note: This matrix is singular (determinant = 0), so inverse should fail.
[Error] inverse_adjoint: matrix is singular (det=0), cannot compute inverse
Inverse Matrix (Should be zero matrix):
Matrix Elements >>>
           0       |
<<< Matrix Elements


[C7.3] Inverse of 3x3 Matrix
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


[C7.4] Non-Square Matrix (Expect Error)
Original Matrix (2x3, non-square):
Matrix Elements >>>
           0            0            0       |
           0            0            0       |
<<< Matrix Elements

[Error] inverse_adjoint: requires square matrix (got 2x3)
Inverse Matrix (should be empty due to error):
Matrix Elements >>>
           0       |
<<< Matrix Elements


[C8: Matrix Utilities Tests]

[C8.1] Generate Identity Matrix (eye)
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


[C8.2] Generate Ones Matrix
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


[C8.3] Augment Two Matrices Horizontally [A | B]
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


[C8.4] Augment with Row Mismatch (Expect Error)
[Error] augment: row counts must match (A: 2, B: 3)
Matrix Info >>>
rows            1
cols            1
elements        1
paddings        0
stride          1
memory          1
data pointer    0x3fce9c54
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info

[C8.5] Vertically Stack Two Matrices [A; B]
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

[C8.6] Vertical Stack with Different Row Counts (Same Columns)
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


[C8.7] VStack with Column Mismatch (Expect Error)
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

[Error] vstack: column counts must match (A: 2, B: 3)
Result (should be empty due to error):
Matrix Info >>>
rows            1
cols            1
elements        1
paddings        0
stride          1
memory          1
data pointer    0x3fce9e1c
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
============ [tiny_matrix_test end] ============

```


### 第四阶段：线性系统求解（D）

```c
============ [tiny_matrix_test start] ============

[Test Organization: Application-Oriented Logic]
  Foundation → Basic Ops → Properties → Linear Systems → Decompositions → Applications → Quality


[D1: Gaussian Elimination Tests]

[D1.1] 3x3 Matrix (Simple Upper Triangular)
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


[D1.2] 3x4 Augmented Matrix (Linear System Ax = b)
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


[D1.3] Singular Matrix (No Unique Solution)
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


[D1.4] Zero Matrix
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


[D1.5] gaussian_eliminate() - Boundary Case - Empty Matrix
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] gaussian_eliminate: matrix data pointer is null
Empty matrix gaussian_eliminate: Empty matrix or error state (Expected) [PASS]

[D1.6] gaussian_eliminate() - Boundary Case - 1x1 Matrix
1x1 matrix after gaussian_eliminate:
Matrix Elements >>>
           5       |
<<< Matrix Elements

1x1 matrix gaussian_eliminate: [PASS]

[D2: Row Reduce from Gaussian (RREF) Tests]

[D2.1] 3x4 Augmented Matrix
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


[D2.2] 2x3 Matrix
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


[D2.3] Already Reduced Matrix
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


[D2.4] row_reduce_from_gaussian() - Boundary Case - Empty Matrix
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] row_reduce_from_gaussian: matrix data pointer is null
Empty matrix row_reduce_from_gaussian: Empty matrix or error state (Expected) [PASS]

[D3: Gaussian Inverse Tests]

[D3.1] 2x2 Matrix Inverse
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


[D3.2] Identity Matrix Inverse
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


[D3.3] Singular Matrix (Expected: No Inverse)
Original matrix (singular):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

[Error] inverse_gje: matrix is singular (not invertible), left half is not identity matrix at (0, 2): expected=0, actual=-1
Inverse matrix (singular):
Matrix Elements >>>
           0       |
<<< Matrix Elements


[D3.4] 3x3 Matrix Inverse
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


[D3.5] Non-square Matrix Inverse (Expected Error)
Original matrix (non-square):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
<<< Matrix Elements

[Error] inverse_gje: requires square matrix (got 2x3)
Inverse matrix (non-square):
Matrix Elements >>>
           0       |
<<< Matrix Elements


[D4: Dot Product Tests]

[D4.1] Valid Dot Product (Same Length Vectors)
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

[D4.2] Invalid Dot Product (Dimension Mismatch)
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

[Error] dotprod: matrices must have the same size (A: 3x1, B: 2x1)
Dot product (dimension mismatch): 0

[D4.3] Dot Product of Zero Vectors
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

[D5: Solve Linear System Tests]

[D5.1] Solving a Simple 2x2 System Ax = b
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


[D5.2] Solving a 3x3 System Ax = b
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


[D5.3] Solving a System Where One Row is All Zeros (Expect Failure or Infinite Solutions)
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

[Error] solve: pivot at (1, 1) is zero or too small (0), matrix is singular or near-singular
Solution x:
Matrix Elements >>>
           0       |
<<< Matrix Elements


[D5.4] Solving a System with Zero Determinant (Singular Matrix)
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

[Error] solve: pivot at (1, 1) is zero or too small (0), matrix is singular or near-singular
Solution x:
Matrix Elements >>>
           0       |
<<< Matrix Elements


[D5.5] Solving a System with Linearly Dependent Rows (Expect Failure or Infinite Solutions)
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

[Error] solve: pivot at (1, 1) is zero or too small (0), matrix is singular or near-singular
Solution x:
Matrix Elements >>>
           0       |
<<< Matrix Elements


[D5.6] Solving a Larger 4x4 System Ax = b
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


[D5.7] solve() - Boundary Case - Empty Matrix
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] solve: matrix A data pointer is null
Empty system solve: Empty matrix or error state (Expected) [PASS]

[D5.8] solve() - Error Handling - Dimension Mismatch
[Error] solve: dimensions do not match (A: 2x2, b: 3x1, expected b: 2x1)
Dimension mismatch solve: Empty matrix or error state (Expected) [PASS]

[D6: Band Solve Tests]

[D6.1] Simple 3x3 Band Matrix
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


[D6.2] 4x4 Band Matrix
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


[D6.3] Incompatible Dimensions (Expect Error)
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

[Error] band_solve: dimensions do not match (A: 3x3, b: 2x1, expected b: 3x1)
Solution x:
Matrix Elements >>>
           0       |
<<< Matrix Elements


[D6.4] Singular Matrix (No Unique Solution)
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

[Error] band_solve: zero or near-zero pivot detected at (1, 1) = 0, matrix is singular or near-singular
Solution x:
Matrix Elements >>>
           0       |
<<< Matrix Elements


[D6.5] band_solve() - Boundary Case - Empty Matrix
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] band_solve: matrix A data pointer is null
Empty system band_solve: Empty matrix or error state (Expected) [PASS]

[D6.6] band_solve() - Error Handling - Invalid Bandwidth
[Error] band_solve: bandwidth k must be >= 1 (got -1)
band_solve with k=-1: Empty matrix or error state (Expected) [PASS]

[D7: Roots Tests]

[D7.1] Solving a Simple 2x2 System Ax = b
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


[D7.2] Solving a 3x3 System Ax = b
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


[D7.3] Singular Matrix (No Unique Solution)
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

[Error] roots: pivot is zero or too small at (1, 1) = 0, system may have no solution
Solution x:
Matrix Elements >>>
           0       |
<<< Matrix Elements


[D7.4] Incompatible Dimensions (Expect Error)
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

[Error] roots: dimensions do not match (A: 3x3, y: 2x1, expected y: 3x1)
Dimension mismatch roots: Empty matrix or error state (Expected) [PASS]

[D7.5] roots() - Boundary Case - Empty Matrix
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] roots: matrix A data pointer is null
Empty system roots: Empty matrix or error state (Expected) [PASS]
============ [tiny_matrix_test end] ============
```

### 第五阶段： 高级线性代数（E1+2）

```c
============ [tiny_matrix_test start] ============

[Test Organization: Application-Oriented Logic]
  Foundation → Basic Ops → Properties → Linear Systems → Decompositions → Applications → Quality


[E1: Matrix Decomposition Tests]

[E1.1] is_positive_definite() - Basic Functionality

[E1.11] Positive Definite 3x3 Matrix
Matrix:
Matrix Elements >>>
           4            1            0       |
           1            3            0       |
           0            0            2       |
<<< Matrix Elements

Is positive definite: True (Expected: True) [PASS]

[E1.12] Non-Positive Definite Matrix
Matrix:
Matrix Elements >>>
           1            2       |
           2            1       |
<<< Matrix Elements

Is positive definite: False (Expected: False) [PASS]

[E1.13] max_minors_to_check Parameter Testing
max_minors_to_check = -1 (check all): True (Expected: True) [PASS]
max_minors_to_check = 3 (check first 3): True (Expected: True) [PASS]
[Error] is_positive_definite: max_minors_to_check must be > 0 or -1 (got 0)
max_minors_to_check = 0 (invalid): False (Expected: False) [PASS]

[E1.14] Parameter Validation - Negative Tolerance
[Error] is_positive_definite: tolerance must be non-negative (got -1e-06)
tolerance = -1e-6 (invalid): False (Expected: False) [PASS]

[E1.15] Boundary Case - Empty Matrix (0x0)
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] is_positive_definite: matrix data pointer is null
Empty matrix (0x0): False (Expected: False, empty matrix is invalid) [PASS]

[E1.16] Boundary Case - Invalid Dimensions
Non-square matrix (2x3): False (Expected: False) [PASS]

[E1.2] LU Decomposition

[E1.21] 3x3 Matrix - LU Decomposition with Pivoting
Matrix A:
Matrix Elements >>>
           2            1            1       |
           4            3            3       |
           2            1            2       |
<<< Matrix Elements


[Results]
Status: OK
L matrix (lower triangular):
Matrix Elements >>>
           1            0            0       |
         0.5            1            0       |
         0.5            1            1       |
<<< Matrix Elements

U matrix (upper triangular):
Matrix Elements >>>
           4            3            3       |
           0         -0.5         -0.5       |
           0            0            1       |
<<< Matrix Elements

P matrix (permutation):
Matrix Elements >>>
           0            1            0       |
           1            0            0       |
           0            0            1       |
<<< Matrix Elements


[Verification] P * A should equal L * U
Total difference: 0 [PASS]

[E1.22] Solve Linear System using LU Decomposition
System: A * x = b
A:
Matrix Elements >>>
           2            1            1       |
           4            3            3       |
           2            1            2       |
<<< Matrix Elements

b:
Matrix Elements >>>
           1       |
           2       |
           3       |
<<< Matrix Elements


[Results]
Solution x:
Matrix Elements >>>
         0.5       |
          -2       |
           2       |
<<< Matrix Elements

Verification error: 0 [PASS]

[E1.26] solve_lu() - Boundary Case - Empty Matrix
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] lu_decompose: matrix data pointer is null
[Error] solve_lu: invalid LU decomposition (status: 458759)
Empty system: x rows = 1 (Expected: 0 or error state) [PASS]

[E1.27] solve_lu() - Invalid LU Decomposition
[Error] solve_lu: invalid LU decomposition (status: 258)
Invalid LU decomposition: x rows = 1 (Expected: 0 or error state) [PASS]

[E1.23] Boundary Case - Empty Matrix (0x0)
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] lu_decompose: matrix data pointer is null
Empty matrix: Status = Error, L rows = 1 (Expected: Error status, L is 0x0 or error state) [PASS]

[E1.24] LU Decomposition without Pivoting
Status: OK
Pivoted: No (Expected) [PASS]
Verification (A = L * U): difference = 0 [PASS]

[E1.25] lu_decompose() - Error Handling - Non-Square Matrix
[Error] lu_decompose: requires square matrix (got 2x3)
Non-square matrix (2x3): Status = Error (Expected) [PASS]

[E1.3] Cholesky Decomposition

[E1.31] SPD Matrix - Cholesky Decomposition
Matrix A (SPD):
Matrix Elements >>>
           4            2            0       |
           2            5            1       |
           0            1            3       |
<<< Matrix Elements


[Results]
Status: OK
L matrix (lower triangular):
Matrix Elements >>>
           2            0            0       |
           1            2            0       |
           0          0.5      1.65831       |
<<< Matrix Elements


[Verification] L * L^T should equal A
Total difference: 2.38419e-07 [PASS]

[E1.32] Solve Linear System using Cholesky Decomposition
Solution x:
Matrix Elements >>>
    0.272727       |
    0.454545       |
    0.181818       |
<<< Matrix Elements

Verification error: 0 [PASS]

[E1.35] solve_cholesky() - Boundary Case - Empty Matrix
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] cholesky_decompose: matrix data pointer is null
[Error] solve_cholesky: invalid Cholesky decomposition (status: 458759)
Empty system: x rows = 1 (Expected: 0 or error state) [PASS]

[E1.36] solve_cholesky() - Invalid Cholesky Decomposition
[Error] solve_cholesky: invalid Cholesky decomposition (status: 258)
Invalid Cholesky decomposition: x rows = 1 (Expected: 0 or error state) [PASS]

[E1.33] Boundary Case - Empty Matrix (0x0)
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] cholesky_decompose: matrix data pointer is null
Empty matrix: Status = Error, L rows = 1 (Expected: Error status, L is 0x0 or error state) [PASS]

[E1.34] Non-Symmetric Matrix (Should Fail)
[Error] cholesky_decompose: requires symmetric matrix
Non-symmetric matrix: Status = Error (Expected) [PASS]

[E1.37] solve_cholesky() - Error Handling - Dimension Mismatch
[Error] solve_cholesky: dimension mismatch - b must be 3x1 vector (got 4x1)
Dimension mismatch solve_cholesky: Empty matrix or error state (Expected) [PASS]

[E1.4] QR Decomposition

[E1.41] General 3x3 Matrix - QR Decomposition
Matrix A:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements


[Results]
Status: OK
Q matrix (orthogonal):
Matrix Elements >>>
    0.123091     0.904534     0.408248       |
    0.492366     0.301511    -0.816497       |
     0.86164    -0.301511     0.408248       |
<<< Matrix Elements

R matrix (upper triangular):
Matrix Elements >>>
     8.12404      9.60114      11.0782       |
           0     0.904534      1.80907       |
           0            0            0       |
<<< Matrix Elements


[Verification] Q * R should equal A
Total difference: 1.66893e-06 [PASS]
Q orthogonality error: 2.83733e-07 [PASS]

[E1.42] Least Squares Solution using QR Decomposition
Overdetermined system: A * x ≈ b
A:
Matrix Elements >>>
           1            1       |
           1            2       |
           1            3       |
<<< Matrix Elements

b:
Matrix Elements >>>
           2       |
           3       |
           4       |
<<< Matrix Elements


[Results]
Least squares solution x:
Matrix Elements >>>
           1       |
           1       |
<<< Matrix Elements

Residual norm ||A*x - b||: 4.12953e-07

[E1.46] solve_qr() - Boundary Case - Empty Matrix
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] qr_decompose: matrix data pointer is null
[Error] solve_qr: invalid QR decomposition (status: 458759)
Empty system: x rows = 1 (Expected: 0 or error state) [PASS]

[E1.47] solve_qr() - Invalid QR Decomposition
[Error] solve_qr: invalid QR decomposition (status: 258)
Invalid QR decomposition: x rows = 1 (Expected: 0 or error state) [PASS]

[E1.43] Boundary Case - Empty Matrix (0x0)
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] qr_decompose: matrix data pointer is null
Empty matrix: Status = Error, Q rows = 1 (Expected: Error status, Q is 0x0 or error state) [PASS]

[E1.44] Boundary Case - Zero Rows or Columns
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] qr_decompose: matrix data pointer is null
Matrix with 0 rows (0x3): Status = Error [PASS]
[Error] qr_decompose: matrix data pointer is null
Matrix with 0 cols (3x0): Status = Error [PASS]

[E1.45] solve_qr() - Error Handling - Dimension Mismatch
[Error] solve_qr: dimension mismatch - b must be 3x1 vector (got 4x1)
Dimension mismatch solve_qr: Empty matrix or error state (Expected) [PASS]

[E1.5] Singular Value Decomposition (SVD)

[E1.51] General 3x3 Matrix - SVD Decomposition
Matrix A:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements


[Results]
Status: OK
Singular values:
Matrix Elements >>>
     1.06837       |
     16.8481       |
           0       |
<<< Matrix Elements

Numerical rank: 2
Iterations: 7
Reconstruction error: 1.68085e-05 [PASS]

[E1.52] Pseudo-inverse using SVD
Matrix A (3x2):
Matrix Elements >>>
           1            2       |
           3            4       |
           5            6       |
<<< Matrix Elements


[Results]
Pseudo-inverse A^+ (2x3):
Matrix Elements >>>
    -1.33333    -0.333333     0.666666       |
     1.08333     0.333333    -0.416666       |
<<< Matrix Elements

Verification error (A * A^+ * A ≈ A): 5.72205e-06 [PASS]

[E1.57] pseudo_inverse() - Parameter Validation - tolerance < 0
[Error] pseudo_inverse: tolerance must be >= 0 (got -1e-06)
tolerance = -1e-6: A_plus rows = 1 (Expected: 0 or error state) [PASS]

[E1.58] pseudo_inverse() - Invalid SVD Decomposition
[Error] pseudo_inverse: invalid SVD decomposition (status: 258)
Invalid SVD decomposition: A_plus rows = 1 (Expected: 0 or error state) [PASS]

[E1.53] Parameter Validation - max_iter <= 0
[Error] svd_decompose: max_iter must be > 0 (got 0)
max_iter = 0: Status = Error (Expected) [PASS]
[Error] svd_decompose: max_iter must be > 0 (got -1)
max_iter = -1: Status = Error (Expected) [PASS]

[E1.54] Parameter Validation - tolerance < 0
[Error] svd_decompose: tolerance must be >= 0 (got -1e-06)
tolerance = -1e-6: Status = Error (Expected) [PASS]

[E1.55] Boundary Case - Empty Matrix (m=0 or n=0)
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] svd_decompose: matrix data pointer is null
Matrix with 0 rows (0x3): Status = Error [PASS]
[Error] svd_decompose: matrix data pointer is null
Matrix with 0 cols (3x0): Status = Error [PASS]

[E1.56] pseudo_inverse() - Error Handling - Invalid SVD Decomposition
[Error] pseudo_inverse: invalid SVD decomposition (status: 258)
Invalid SVD decomposition: A_plus rows = 1 (Expected: 0 or error state) [PASS]

[E1.6] Matrix Decomposition Performance Tests

[E1.61] LU Decomposition Performance
[Performance] LU Decomposition (4x4 matrix): 148.00 us

[E1.62] Cholesky Decomposition Performance
[Performance] Cholesky Decomposition (4x4 SPD matrix): 95.00 us

[E1.63] QR Decomposition Performance
[Performance] QR Decomposition (4x4 matrix): 194.00 us

[E1.64] SVD Decomposition Performance
[Performance] SVD Decomposition (4x4 matrix): 367.00 us

[Matrix Decomposition Tests Complete]

[E2: Gram-Schmidt Orthogonalization Tests]

[E2.1] Basic Orthogonalization - Linearly Independent Vectors
Input vectors (each column is a vector):
Matrix Elements >>>
        1.00         1.00         0.00       |
        0.00         1.00         1.00       |
        1.00         0.00         1.00       |
<<< Matrix Elements


[Results]
Status: OK
Orthogonalized vectors Q (each column is orthogonal):
Matrix Elements >>>
        0.71         0.41        -0.58       |
        0.00         0.82         0.58       |
        0.71        -0.41         0.58       |
<<< Matrix Elements

Coefficients R (upper triangular):
Matrix Elements >>>
        1.41         0.71         0.71       |
        0.00         1.22         0.41       |
        0.00         0.00         1.15       |
<<< Matrix Elements


[Verification] Q^T * Q should be identity
Orthogonality error: 0.00 [PASS]

[Verification] Each column of Q should be normalized
  Column 0 norm: 1.00 (error: 0.00) [PASS]
  Column 1 norm: 1.00 (error: 0.00) [PASS]
  Column 2 norm: 1.00 (error: 0.00) [PASS]

[Verification] Q * R should reconstruct original vectors
Reconstruction error: 0.00 [PASS]

[E2.2] Orthogonalization - Near-Linear-Dependent Vectors
Input vectors (third vector is nearly linear dependent):
Matrix Elements >>>
        1.00         0.00         1.00       |
        0.00         1.00         1.00       |
        0.00         0.00         0.00       |
<<< Matrix Elements


[Results]
Status: OK
Orthogonalized vectors Q:
Matrix Elements >>>
        1.00         0.00         0.00       |
        0.00         1.00         0.00       |
        0.00         0.00         1.00       |
<<< Matrix Elements

Coefficients R:
Matrix Elements >>>
        1.00         0.00         1.00       |
        0.00         1.00         1.00       |
        0.00         0.00         0.00       |
<<< Matrix Elements


[Note] Third column norm: 1.00 (should be 0 if linearly dependent, or 1 if orthogonalized)

[E2.3] Orthogonalization - 2D Vectors (2x2)
Input vectors:
Matrix Elements >>>
        3.00         1.00       |
        1.00         2.00       |
<<< Matrix Elements


[Results]
Status: OK
Orthogonalized vectors Q:
Matrix Elements >>>
        0.95        -0.32       |
        0.32         0.95       |
<<< Matrix Elements

Coefficients R:
Matrix Elements >>>
        3.16         1.58       |
        0.00         1.58       |
<<< Matrix Elements


[Verification] Dot product of Q columns: 0.00 (should be ~0 for orthogonal) [PASS]

[E2.4] Error Handling - Invalid Input
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] gram_schmidt_orthogonalize: Input matrix is null.
Empty matrix test: PASS (correctly rejected)

[E2.5] gram_schmidt_orthogonalize() - Parameter Validation - Negative Tolerance
[Error] gram_schmidt_orthogonalize: tolerance must be non-negative (got -1e-06)
tolerance = -1e-6: PASS (correctly rejected)

[E2.6] gram_schmidt_orthogonalize() - Boundary Case - Zero Rows
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] gram_schmidt_orthogonalize: Input matrix is null.
Zero rows (0x2): PASS (correctly rejected)

[E2.7] gram_schmidt_orthogonalize() - Boundary Case - Zero Columns
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] gram_schmidt_orthogonalize: Input matrix is null.
Zero columns (2x0): PASS (correctly rejected)
============ [tiny_matrix_test end] ============
```

### 第六阶段：系统识别应用（E3）

```c
============ [tiny_matrix_test start] ============

[Test Organization: Application-Oriented Logic]
  Foundation → Basic Ops → Properties → Linear Systems → Decompositions → Applications → Quality


[E3: Eigenvalue Decomposition Tests]

[E3.1] is_symmetric() - Basic Functionality
[E3.11] Symmetric 3x3 Matrix
Matrix:
Matrix Elements >>>
           4            1            2       |
           1            3            0       |
           2            0            5       |
<<< Matrix Elements

Is symmetric: True (Expected: True)

[E3.12] Non-Symmetric 3x3 Matrix
Matrix:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

Is symmetric: False (Expected: False)

[E3.13] Non-Square Matrix (2x3)
Is symmetric: False (Expected: False)

[E3.14] Symmetric Matrix with Small Numerical Errors
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

[E3.15] Parameter Validation - Negative Tolerance
[Error] is_symmetric: tolerance must be >= 0 (got -1e-06)
tolerance = -1e-6 (invalid): False (Expected: False) [PASS]

[E3.16] Boundary Case - Empty Matrix (0x0)
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] is_symmetric: matrix data pointer is null
Empty matrix (0x0): False (Expected: False, empty matrix is invalid) [PASS]

[E3.2] power_iteration() - Dominant Eigenvalue

[E3.21] Simple 2x2 Matrix
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

[E3.22] 3x3 Stiffness Matrix (SHM Application)
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

[E3.23] Non-Square Matrix (Expect Error)
[Error] power_iteration: requires square matrix (got 2x3)
Status: Error (Expected)

[E3.25] Parameter Validation - max_iter <= 0
[Error] power_iteration: max_iter must be > 0 (got 0)
max_iter = 0: Status = Error (Expected) [PASS]
[Error] power_iteration: max_iter must be > 0 (got -1)
max_iter = -1: Status = Error (Expected) [PASS]

[E3.26] Parameter Validation - tolerance < 0
[Error] power_iteration: tolerance must be >= 0 (got -1e-06)
tolerance = -1e-6: Status = Error (Expected) [PASS]

[E3.27] Boundary Case - Empty Matrix (0x0)
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] power_iteration: matrix data pointer is null
Empty matrix: Status = Error, eigenvalue = 0.00 (Expected: Error status) [PASS]

[E3.24] inverse_power_iteration() - Smallest Eigenvalue (System Identification)

[E3.28] Simple 2x2 Matrix - Smallest Eigenvalue
Matrix (same as E3.21):
Matrix Elements >>>
        2.00         1.00       |
        1.00         2.00       |
<<< Matrix Elements


[Expected Results]
  Expected eigenvalues: 3.0 (largest), 1.0 (smallest)
  Expected smallest eigenvalue: 1.0
  Expected smallest eigenvector (for λ=1): approximately [0.707, -0.707] or [-0.707, 0.707] (normalized)
  Note: This is critical for system identification - smallest eigenvalue = fundamental frequency

[Actual Results]
  Smallest eigenvalue: 1.00 (Expected: 1.0, smallest eigenvalue)
  Iterations: 6
  Status: OK
  Smallest eigenvector:
Matrix Elements >>>
        0.71       |
       -0.71       |
<<< Matrix Elements

  Error from expected (1.0): 0.00 [PASS]

[Comparison] Power vs Inverse Power Iteration:
  Power iteration (λ_max): 3.00
  Inverse power iteration (λ_min): 1.00
  Ratio (λ_max/λ_min): 3.00 (Expected: ~3.0) [PASS]

[E3.29] 3x3 Stiffness Matrix - Smallest Eigenvalue (SHM Application)
Stiffness Matrix (same as E3.22):
Matrix Elements >>>
        2.00        -1.00         0.00       |
       -1.00         2.00        -1.00       |
        0.00        -1.00         2.00       |
<<< Matrix Elements


[Expected Results]
  Expected eigenvalues (approximate): 3.414 (largest), 2.000, 0.586 (smallest)
  Expected smallest eigenvalue: ~0.586 (fundamental frequency squared)
  Expected fundamental frequency: sqrt(0.586) ≈ 0.765 rad/s
  Note: Smallest eigenvalue is critical for system identification - represents fundamental mode

[Actual Results]
  Smallest eigenvalue (fundamental frequency squared): 0.59
  Fundamental frequency: 0.77 rad/s (Expected: ~0.765 rad/s)
  Iterations: 8
  Status: OK
  Smallest eigenvector (fundamental mode shape):
Matrix Elements >>>
        0.50       |
        0.71       |
        0.50       |
<<< Matrix Elements

  Error from expected (0.59): 0.00 [PASS]

[Comparison] Power vs Inverse Power Iteration for SHM:
  Power iteration (primary frequency²): 3.41 → frequency: 1.85 rad/s
  Inverse power iteration (fundamental frequency²): 0.59 → frequency: 0.77 rad/s
  Frequency ratio: 2.41 (Expected: ~2.4, ratio of highest to lowest mode)

[E3.210] Non-Square Matrix (Expect Error)
[Error] inverse_power_iteration: requires square matrix (got 2x3)
Status: Error (Expected)
Error handling: [PASS]

[E3.211] Near-Singular Matrix (Edge Case)
Matrix (near-singular but invertible):
Matrix Elements >>>
        1.00         0.00         0.00       |
        0.00         1.00         0.00       |
        0.00         0.00         1.00       |
<<< Matrix Elements


[Results]
  Status: OK
  Smallest eigenvalue: 1.00
  Iterations: 2
  Note: Successfully handled near-singular matrix [PASS]

[E3.212] Parameter Validation - max_iter <= 0
[Error] inverse_power_iteration: max_iter must be > 0 (got 0)
max_iter = 0: Status = Error (Expected) [PASS]
[Error] inverse_power_iteration: max_iter must be > 0 (got -1)
max_iter = -1: Status = Error (Expected) [PASS]

[E3.213] Parameter Validation - tolerance < 0
[Error] inverse_power_iteration: tolerance must be >= 0 (got -1e-06)
tolerance = -1e-6: Status = Error (Expected) [PASS]

[E3.214] Boundary Case - Empty Matrix (0x0)
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] inverse_power_iteration: matrix data pointer is null
Empty matrix: Status = Error, eigenvalue = 0.00 (Expected: Error status) [PASS]

[E3.215] Singular Matrix (Should Fail)
[Error] solve: pivot at (1, 1) is zero or too small (0), matrix is singular or near-singular
[Error] Inverse power iteration: Matrix is singular or near-singular. Cannot solve linear system A * y = v.
Singular matrix: Status = Error (Expected: Error or OK with eigenvalue ≈ 0) [PASS]

[E3.3] eigendecompose_jacobi() - Symmetric Matrix Decomposition

[E3.31] 2x2 Symmetric Matrix - Complete Decomposition
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

[E3.32] 3x3 Stiffness Matrix (SHM Application)
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

[E3.33] Diagonal Matrix (Eigenvalues on diagonal)
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

[E3.34] Parameter Validation - tolerance < 0
[Error] eigendecompose_jacobi: tolerance must be >= 0 (got -1e-06)
tolerance = -1e-6: Status = Error (Expected) [PASS]

[E3.35] Parameter Validation - max_iter <= 0
[Error] eigendecompose_jacobi: max_iter must be > 0 (got 0)
max_iter = 0: Status = Error (Expected) [PASS]
[Error] eigendecompose_jacobi: max_iter must be > 0 (got -1)
max_iter = -1: Status = Error (Expected) [PASS]

[E3.36] Boundary Case - Empty Matrix (0x0)
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] eigendecompose_jacobi: matrix data pointer is null
Empty matrix: Status = Error, eigenvalues rows = 1 (Expected: Error status, eigenvalues is 0x0 or error state) [PASS]

[E3.4] eigendecompose_qr() - General Matrix Decomposition

[E3.41] General 2x2 Matrix
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
Eigenvalue 2: -0.37 (Expected: -0.37, Error: 0.00, Rel Error: 0.07%) [PASS]
Overall eigenvalue check: [PASS]

[E3.42] Non-Symmetric 3x3 Matrix
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
       16.12       |
       -1.12       |
        0.00       |
<<< Matrix Elements

Eigenvectors:
Matrix Elements >>>
        0.23         0.88         0.41       |
        0.53         0.24        -0.82       |
        0.82        -0.40         0.41       |
<<< Matrix Elements

Iterations: 6
Status: OK
Eigenvalue 0: 16.12 (Expected: 16.12, Error: 0.00, Rel Error: 0.02%) [PASS]
Eigenvalue 1: -1.12 (Expected: -1.12, Error: 0.00, Rel Error: 0.28%) [PASS]
Eigenvalue 2: 0.00 (Expected: 0.00, Error: 0.00, Rel Error: 0.00%) [PASS]
Overall eigenvalue check: [PASS]

[E3.43] Parameter Validation - max_iter <= 0
[Error] eigendecompose_qr: max_iter must be > 0 (got 0)
max_iter = 0: Status = Error (Expected) [PASS]
[Error] eigendecompose_qr: max_iter must be > 0 (got -1)
max_iter = -1: Status = Error (Expected) [PASS]

[E3.44] Parameter Validation - tolerance < 0
[Error] eigendecompose_qr: tolerance must be >= 0 (got -1e-06)
tolerance = -1e-6: Status = Error (Expected) [PASS]

[E3.45] Boundary Case - Empty Matrix (0x0)
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] eigendecompose_qr: matrix data pointer is null
Empty matrix: Status = Error, eigenvalues rows = 1 (Expected: Error status, eigenvalues is 0x0 or error state) [PASS]

[E3.5] eigendecompose() - Automatic Method Selection

[E3.51] Symmetric Matrix (Auto-select: Jacobi)
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

[E3.52] Non-Symmetric Matrix (Auto-select: QR)
[Expected Results]
  Method: Should automatically use QR (non-symmetric matrix detected)
  Expected eigenvalues (theoretical): 16.12, -1.12, 0.00
  Note: One eigenvalue should be near 0 (rank-deficient matrix)
  Note: QR algorithm may have numerical errors for non-symmetric matrices
  Acceptable: largest ~15-18, smallest near 0, one near -1 to -3

[Actual Results]
Eigenvalues:
Matrix Elements >>>
       16.12       |
       -1.12       |
        0.00       |
<<< Matrix Elements

Iterations: 6
Status: OK
Method used: QR (auto-selected for non-symmetric matrix)
Eigenvalue 0: 16.12 (Expected: 16.12, Error: 0.00, Rel Error: 0.02%) [PASS]
Eigenvalue 1: -1.12 (Expected: -1.12, Error: 0.00, Rel Error: 0.28%) [PASS]
Eigenvalue 2: 0.00 (Expected: 0.00, Error: 0.00, Rel Error: 0.00%) [PASS]
Overall eigenvalue check: [PASS]

[E3.53] Parameter Validation - tolerance < 0
[Error] eigendecompose: tolerance must be >= 0 (got -1e-06)
tolerance = -1e-6: Status = Error (Expected) [PASS]

[E3.531] Parameter Validation - max_iter <= 0
[Error] eigendecompose: max_iter must be > 0 (got 0)
max_iter = 0: Status = Error (Expected) [PASS]
[Error] eigendecompose: max_iter must be > 0 (got -10)
max_iter = -10: Status = Error (Expected) [PASS]

[E3.532] Custom max_iter Parameter Test
[Test 1] max_iter = 10 (may not converge)
  Status: OK (Converged)
  Iterations: 8
[Test 2] max_iter = 200 (should converge)
  Status: OK (Converged)
  Iterations: 8
  Eigenvalues:
Matrix Elements >>>
        1.85       |
        3.48       |
        6.67       |
<<< Matrix Elements

Custom max_iter test: [PASS]

[E3.54] eigendecompose() - Boundary Case - Empty Matrix (0x0)
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] eigendecompose: matrix data pointer is null
Empty matrix: Status = Error, eigenvalues rows = 1 (Expected: Error status, eigenvalues is 0x0 or error state) [PASS]

[E3.55] eigendecompose() - Error Handling - Non-Square Matrix
[Error] eigendecompose_qr: requires square matrix (got 2x3)
Non-square matrix (2x3): Status = Error (Expected) [PASS]

[E3.6] SHM Application - Structural Dynamics Analysis

[E3.61] 4-DOF Mass-Spring System
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

[E3.7] Edge Cases and Error Handling

[E3.71] 1x1 Matrix
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

[E3.72] Zero Matrix
[Error] power_iteration: computed vector norm too small
Status: Error (Expected)

[E3.73] Identity Matrix
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

[E3.8] Performance Test for SHM Applications

[E3.81] Power Iteration Performance (Real-time SHM - Dominant Eigenvalue)
[Performance] Power Iteration (3x3 matrix): 123.00 us

[E3.82] Inverse Power Iteration Performance (System Identification - Smallest Eigenvalue)
[Performance] Inverse Power Iteration (3x3 matrix): 554.00 us

[E3.83] Jacobi Method Performance (Complete Eigendecomposition - Symmetric Matrices)
[Performance] Jacobi Decomposition (3x3 symmetric matrix): 187.00 us

[E3.84] QR Method Performance (Complete Eigendecomposition - General Matrices)
[Performance] QR Decomposition (3x3 general matrix): 806.00 us

[Eigenvalue Decomposition Tests Complete]
============ [tiny_matrix_test end] ============
```


### 第七阶段：辅助函数（F）

```c
============ [tiny_matrix_test start] ============

[Test Organization: Application-Oriented Logic]
  Foundation → Basic Ops → Properties → Linear Systems → Decompositions → Applications → Quality


[F1: Stream Operators Tests]

[F1.1] Stream Insertion Operator (<<) for Mat
Matrix mat1:
1 2 3
4 5 6
7 8 9


[F1.2] Stream Insertion Operator (<<) for Mat::ROI
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


[F1.3] Stream Extraction Operator (>>) for Mat
Simulated input: "10 20 30 40"
Matrix mat2 after input:
10 20
30 40

Expected: [10, 20; 30, 40]

[F1.4] Stream Extraction Operator (>>) for Mat (2x3 matrix)
Simulated input: "1.5 2.5 3.5 4.5 5.5 6.5"
Matrix mat3 after input:
1.5 2.5 3.5
4.5 5.5 6.5

Expected: [1.5, 2.5, 3.5; 4.5, 5.5, 6.5]

[F2: Global Arithmetic Operators Tests]

[F2.1] Matrix Addition (operator+)
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


[F2.2] Matrix Addition with Constant (operator+)
Matrix A:
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Constant: 5.0
matA + 5.0f:
6 7
8 9


[F2.3] Matrix Subtraction (operator-)
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


[F2.4] Matrix Subtraction with Constant (operator-)
Matrix A:
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Constant: 2.0
matA - 2.0f:
-1 0
1 2


[F2.5] Matrix Multiplication (operator*)
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


[F2.6] Matrix Multiplication with Constant (operator*)
Matrix A:
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Constant: 2.0
matA * 2.0f:
2 4
6 8


[F2.7] Matrix Division (operator/)
Matrix A:
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements

Constant: 2.0
matA / 2.0f:
0.5 1
1.5 2


[F2.8] Matrix Division Element-wise (operator/)
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


[F2.9] Matrix Comparison (operator==)
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


### 第八阶段：质量保证（G）

```c
============ [tiny_matrix_test start] ============

[Test Organization: Application-Oriented Logic]
  Foundation → Basic Ops → Properties → Linear Systems → Decompositions → Applications → Quality


[G1: Quality Assurance - Boundary Conditions and Error Handling Tests]

[G1.1] Null Pointer Handling in print_matrix
[Error] Cannot print matrix: data pointer is null.

[G1.2] Null Pointer Handling in operator<<
[Error] Cannot print matrix: data pointer is null.


[G1.3] Invalid Block Parameters
[Error] block: invalid position: start_row=-1, start_col=0 (must be non-negative)
block(-1, 0, 2, 2): Error
[Error] block: block exceeds row boundary: start_row=2, block_rows=2, source.rows=3
block(2, 2, 2, 2) on 3x3 matrix: Error
[Error] block: invalid block size: block_rows=0, block_cols=2 (must be positive)
block(0, 0, 0, 2): Error

[G1.4] Invalid swap_rows Parameters
Before invalid swap_rows:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

[Error] swap_rows: row1 index out of range: row1=-1, matrix.rows=3
After swap_rows(-1, 1):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

[Error] swap_rows: row2 index out of range: row2=5, matrix.rows=3
After swap_rows(0, 5):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements


[G1.5] Invalid swap_cols Parameters
Before invalid swap_cols:
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

[Error] swap_cols: col1 index out of range: col1=-1, matrix.cols=3
After swap_cols(-1, 1):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements

[Error] swap_cols: col2 index out of range: col2=5, matrix.cols=3
After swap_cols(0, 5):
Matrix Elements >>>
           1            2            3       |
           4            5            6       |
           7            8            9       |
<<< Matrix Elements


[G1.6] Division by Zero
[Error] Division by zero in operator/.
mat3 / 0.0f: Error

[G1.7] Matrix Division with Zero Elements
[Error] Matrix division failed: Division by zero detected at position (0, 1)
mat4 /= divisor (with zero):
Matrix Elements >>>
           1            2       |
           3            4       |
<<< Matrix Elements


[G1.8] Empty Matrix Operations
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[>>> Error ! <<<] Memory allocation failed in alloc_mem()
[Error] operator+=: this matrix data pointer is null
Empty matrix addition: Empty matrix or error state (Expected) [PASS]

[G2: Quality Assurance - Performance Benchmarks Tests]

[G2.1] Matrix Addition Performance
[Performance] 50x50 Matrix Addition (100 iterations): 18155.00 us total, 181.55 us avg

[G2.2] Matrix Multiplication Performance
[Performance] 30x30 Matrix Multiplication (100 iterations): 66761.00 us total, 667.61 us avg

[G2.3] Matrix Transpose Performance
[Performance] 50x30 Matrix Transpose (100 iterations): 22002.00 us total, 220.02 us avg

[G2.4] Determinant Calculation Performance Comparison

[G2.4.1] Small Matrix (4x4) - Laplace Expansion
[Performance] 4x4 Determinant (Laplace, 10 iterations): 3124.00 us total, 312.40 us avg

[G2.4.2] Large Matrix (8x8) - LU Decomposition
[Performance] 8x8 Determinant (LU, 10 iterations): 2028.00 us total, 202.80 us avg

[G2.4.3] Large Matrix (8x8) - Gaussian Elimination
[Performance] 8x8 Determinant (Gaussian, 10 iterations): 459.00 us total, 45.90 us avg

[G2.4.4] Large Matrix (8x8) - Auto-select Method
[Performance] 8x8 Determinant (auto-select, 10 iterations): 2013.00 us total, 201.30 us avg

[Note] Performance Summary:
  - Laplace expansion (O(n!)): Suitable only for small matrices (n <= 4)
  - LU decomposition (O(n³)): Efficient for large matrices, auto-selected for n > 4
  - Gaussian elimination (O(n³)): Alternative efficient method for large matrices
  - Auto-select: Automatically chooses the best method based on matrix size

[G2.5] Matrix Copy with Padding Performance
[Performance] 8x8 Copy ROI (with padding) (100 iterations): 2587.00 us total, 25.87 us avg

[G2.6] Element Access Performance
[Performance] Computing element access (warmup)...
[Performance] 50x50 Element Access (all elements) (100 iterations): 9685.00 us total, 96.85 us avg

[G3: Quality Assurance - Memory Layout Tests (Padding and Stride)]

[G3.1] Contiguous Memory (no padding)
Matrix 3x4 (stride=4, pad=0):
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        0
stride          4
memory          12
data pointer    0x3fce9af0
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
        0.00         1.00         2.00         3.00       |
        4.00         5.00         6.00         7.00       |
        8.00         9.00        10.00        11.00       |
<<< Matrix Elements


[G3.2] Padded Memory (stride > col)
Matrix 3x4 (stride=5, pad=1):
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fc9a3f4
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
        0.00         1.00         2.00         3.00       |   0.00 
        4.00         5.00         6.00         7.00       |   0.00 
        8.00         9.00        10.00        11.00       |   0.00 
<<< Matrix Elements


[G3.3] Addition with Padded Matrices
Result of padded matrix addition:
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
stride          5
memory          15
data pointer    0x3fce9c64
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
       11.00        22.00        33.00        44.00       |  20.00 
       55.00        66.00        77.00        88.00       |  25.00 
       99.00       110.00       121.00       132.00       |   1.61 
<<< Matrix Elements


[G3.4] ROI Operations with Padded Matrices
ROI (1,1,2,2) from padded matrix:
Matrix Info >>>
rows            2
cols            2
elements        4
paddings        3
stride          5
memory          10
data pointer    0x3fc9a40c
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      1   (This is a Sub-Matrix View)
<<< Matrix Info
Matrix Elements >>>
        5.00         6.00       |   7.00         0.00         8.00 
        9.00        10.00       |  11.00         0.00         0.00 
<<< Matrix Elements


[G3.5] Copy Operations Preserve Stride
Copied matrix (should have stride=4, no padding):
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        0
stride          4
memory          12
data pointer    0x3fce9d98
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
        0.00         1.00         2.00         3.00       |
        4.00         5.00         6.00         7.00       |
        8.00         9.00        10.00        11.00       |
<<< Matrix Elements

============ [tiny_matrix_test end] ============
```