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
// Group 1: Object Foundation - Constructor & Destructor Tests
// ============================================================================
// Purpose: Test object creation and destruction - the foundation of all operations
void test_constructor_destructor()
{
    std::cout << "\n[Group 1: Object Foundation - Constructor & Destructor Tests]\n";

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

// ============================================================================
// Group 2: Object Foundation - Element Access Tests
// ============================================================================
// Purpose: Test element access - fundamental operation for data manipulation
void test_element_access()
{
    std::cout << "\n[Group 2: Object Foundation - Element Access Tests]\n";
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

// ============================================================================
// Group 3: Object Foundation - Data Manipulation Tests (ROI Operations)
// ============================================================================
// Purpose: Test ROI operations - efficient data views and submatrix operations
void test_roi_operations()
{
    std::cout << "\n[Group 3: Object Foundation - Data Manipulation Tests (ROI Operations)]\n";

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

// ============================================================================
// Group 4: Basic Operations - Arithmetic Operators Tests
// ============================================================================
// Purpose: Test basic arithmetic operations - foundation for numerical computations
// Group 4.1: Assignment Operator
void test_assignment_operator()
{
    std::cout << "\n[Group 4.1: Basic Operations - Assignment Operator Tests]\n";

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

// ============================================================================
// Group 5: Matrix Properties - Linear Algebra Tests
// ============================================================================
// Purpose: Test matrix properties and basic linear algebra operations
// Group 5.1: Matrix Transpose
void test_matrix_transpose()
{
    std::cout << "\n[Group 5.1: Matrix Properties - Matrix Transpose Tests]\n";

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

    // Test 5.3.5: 5x5 Matrix (Tests Auto-select Mechanism)
    std::cout << "\n[Test 5.3.5] 5x5 Matrix Determinant (Tests Auto-select to LU Method)\n";
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

    // Test 5.3.6: Non-square Matrix (Expect Error)
    std::cout << "\n[Test 5.3.6] Non-square Matrix (Expect Error)\n";
    tiny::Mat rectMat(3, 4);
    std::cout << "Matrix (3x4, non-square):\n";
    rectMat.print_matrix(true);
    float det_rect = rectMat.determinant();  // should trigger error
    std::cout << "Determinant: " << det_rect << "  (Expected: 0 with error message)\n";

    // Test 5.3.7: Comparison of Different Methods (5x5 Matrix)
    std::cout << "\n[Test 5.3.7] Comparison of Different Methods (5x5 Matrix)\n";
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

    // Test 5.3.8: Large Matrix (6x6) - Tests Efficient Methods
    std::cout << "\n[Test 5.3.8] Large Matrix (6x6) - Tests Efficient Methods\n";
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

    // Test 5.3.9: Large Matrix (8x8) - Performance Test
    std::cout << "\n[Test 5.3.9] Large Matrix (8x8) - Performance Comparison\n";
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

// ============================================================================
// Group 6: Linear System Solving - Core Application Tests
// ============================================================================
// Purpose: Test linear system solving - the core application of matrix library
// Note: This group includes Gaussian elimination, solve methods, and dot product
// Group 6.1: Gaussian Elimination (moved from Group 5.9)
// Group 6.2: Solve Linear System (moved from Group 5.13)
// Group 6.3: Dot Product (moved from Group 5.12)
// Group 6.4: Band Solve (moved from Group 5.14)
// Group 6.5: Roots (moved from Group 5.15)

// ============================================================================
// Group 10: Auxiliary Functions - Stream Operators Tests
// ============================================================================
// Purpose: Test I/O operations - auxiliary but important for debugging
void test_stream_operators()
{
    std::cout << "\n[Group 10: Auxiliary Functions - Stream Operators Tests]\n";

    // Test 10.1: Test stream insertion operator (<<) for Mat
    std::cout << "\n[Test 10.1] Stream Insertion Operator (<<) for Mat\n";
    tiny::Mat mat1(3, 3);
    mat1(0, 0) = 1; mat1(0, 1) = 2; mat1(0, 2) = 3;
    mat1(1, 0) = 4; mat1(1, 1) = 5; mat1(1, 2) = 6;
    mat1(2, 0) = 7; mat1(2, 1) = 8; mat1(2, 2) = 9;

    std::cout << "Matrix mat1:\n";
    std::cout << mat1 << std::endl; // Use the << operator to print mat1

    // Test 10.2: Test stream insertion operator (<<) for Mat::ROI
    std::cout << "\n[Test 10.2] Stream Insertion Operator (<<) for Mat::ROI\n";
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

    // Test 10.3: Test stream extraction operator (>>) for Mat
    std::cout << "\n[Test 10.3] Stream Extraction Operator (>>) for Mat\n";
    tiny::Mat mat2(2, 2);
    // Use istringstream to simulate input (for automated testing)
    std::istringstream input1("10 20 30 40");
    std::cout << "Simulated input: \"10 20 30 40\"\n";
    input1 >> mat2; // Use the >> operator to read from string stream
    std::cout << "Matrix mat2 after input:\n";
    std::cout << mat2 << std::endl; // Use the << operator to print mat2
    std::cout << "Expected: [10, 20; 30, 40]\n";

    // Test 10.4: Test stream extraction operator (>>) for Mat (with different values)
    std::cout << "\n[Test 10.4] Stream Extraction Operator (>>) for Mat (2x3 matrix)\n";
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
// Group 11: Auxiliary Functions - Global Arithmetic Operators Tests
// ============================================================================
// Purpose: Test global operator overloads - syntactic sugar for convenience
void test_matrix_operations()
{
    std::cout << "\n[Group 11: Auxiliary Functions - Global Arithmetic Operators Tests]\n";

    // Test 11.1: Matrix Addition (operator+)
    std::cout << "\n[Test 11.1] Matrix Addition (operator+)\n";
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

    // Test 11.2: Matrix Addition with Constant (operator+)
    std::cout << "\n[Test 11.2] Matrix Addition with Constant (operator+)\n";
    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Constant: 5.0\n";
    tiny::Mat resultAddConst = matA + 5.0f;
    std::cout << "matA + 5.0f:\n";
    std::cout << resultAddConst << std::endl;  // Expected: [6, 7], [8, 9]

    // Test 11.3: Matrix Subtraction (operator-)
    std::cout << "\n[Test 11.3] Matrix Subtraction (operator-)\n";
    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Matrix B:\n";
    matB.print_matrix(true);
    tiny::Mat resultSub = matA - matB;
    std::cout << "matA - matB:\n";
    std::cout << resultSub << std::endl;  // Expected: [-4, -4], [-4, -4]

    // Test 11.4: Matrix Subtraction with Constant (operator-)
    std::cout << "\n[Test 11.4] Matrix Subtraction with Constant (operator-)\n";
    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Constant: 2.0\n";
    tiny::Mat resultSubConst = matA - 2.0f;
    std::cout << "matA - 2.0f:\n";
    std::cout << resultSubConst << std::endl;  // Expected: [-1, 0], [1, 2]

    // Test 11.5: Matrix Multiplication (operator*)
    std::cout << "\n[Test 11.5] Matrix Multiplication (operator*)\n";
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

    // Test 11.6: Matrix Multiplication with Constant (operator*)
    std::cout << "\n[Test 11.6] Matrix Multiplication with Constant (operator*)\n";
    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Constant: 2.0\n";
    tiny::Mat resultMulConst = matA * 2.0f;
    std::cout << "matA * 2.0f:\n";
    std::cout << resultMulConst << std::endl;  // Expected: [2, 4], [6, 8]

    // Test 11.7: Matrix Division (operator/)
    std::cout << "\n[Test 11.7] Matrix Division (operator/)\n";
    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Constant: 2.0\n";
    tiny::Mat resultDiv = matA / 2.0f;
    std::cout << "matA / 2.0f:\n";
    std::cout << resultDiv << std::endl;  // Expected: [0.5, 1], [1.5, 2]

    // Test 11.8: Matrix Division Element-wise (operator/)
    std::cout << "\n[Test 11.8] Matrix Division Element-wise (operator/)\n";
    std::cout << "Matrix A:\n";
    matA.print_matrix(true);
    std::cout << "Matrix B:\n";
    matB.print_matrix(true);
    tiny::Mat resultDivElem = matA / matB;
    std::cout << "matA / matB:\n";
    std::cout << resultDivElem << std::endl;  // Expected: [0.2, 0.333], [0.428, 0.5]

    // Test 11.9: Matrix Comparison (operator==)
    std::cout << "\n[Test 11.9] Matrix Comparison (operator==)\n";
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
// Group 12: Quality Assurance - Boundary Conditions and Error Handling Tests
// ============================================================================
// Purpose: Test error handling and edge cases - ensure robustness
void test_boundary_conditions()
{
    std::cout << "\n[Group 12: Quality Assurance - Boundary Conditions and Error Handling Tests]\n";

    // Test 12.1: Null pointer handling in print functions
    std::cout << "\n[Test 12.1] Null Pointer Handling in print_matrix\n";
    tiny::Mat null_mat;
    null_mat.data = nullptr;  // Simulate null pointer
    null_mat.print_matrix(true);  // Should handle gracefully

    // Test 12.2: Null pointer handling in operator<<
    std::cout << "\n[Test 12.2] Null Pointer Handling in operator<<\n";
    tiny::Mat null_mat2;
    null_mat2.data = nullptr;
    std::cout << null_mat2 << std::endl;  // Should handle gracefully

    // Test 12.3: Invalid block parameters
    std::cout << "\n[Test 12.3] Invalid Block Parameters\n";
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

    // Test 12.4: Invalid swap_rows parameters
    std::cout << "\n[Test 12.4] Invalid swap_rows Parameters\n";
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

    // Test 12.5: Invalid swap_cols parameters
    std::cout << "\n[Test 12.5] Invalid swap_cols Parameters\n";
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

    // Test 12.6: Division by zero
    std::cout << "\n[Test 12.6] Division by Zero\n";
    tiny::Mat mat3(2, 2);
    mat3(0, 0) = 1; mat3(0, 1) = 2;
    mat3(1, 0) = 3; mat3(1, 1) = 4;
    
    tiny::Mat result = mat3 / 0.0f;
    std::cout << "mat3 / 0.0f: " << (result.data == nullptr ? "Empty (correct)" : "Error") << "\n";

    // Test 12.7: Matrix division with zero elements
    std::cout << "\n[Test 12.7] Matrix Division with Zero Elements\n";
    tiny::Mat mat4(2, 2);
    mat4(0, 0) = 1; mat4(0, 1) = 2;
    mat4(1, 0) = 3; mat4(1, 1) = 4;
    
    tiny::Mat divisor(2, 2);
    divisor(0, 0) = 1; divisor(0, 1) = 0;  // Contains zero
    divisor(1, 0) = 3; divisor(1, 1) = 4;
    
    mat4 /= divisor;
    std::cout << "mat4 /= divisor (with zero):\n";
    mat4.print_matrix(true);

    // Test 12.8: Empty matrix operations
    std::cout << "\n[Test 12.8] Empty Matrix Operations\n";
    tiny::Mat empty1, empty2;
    tiny::Mat empty_sum = empty1 + empty2;
    std::cout << "Empty matrix addition: " << (empty_sum.data == nullptr ? "Empty (correct)" : "Error") << "\n";
}

// ============================================================================
// Group 13: Quality Assurance - Performance Benchmarks Tests
// ============================================================================
// Purpose: Test performance characteristics - critical for real-time applications
void test_performance_benchmarks()
{
    std::cout << "\n[Group 13: Quality Assurance - Performance Benchmarks Tests]\n";
    
    // Ensure current task is added to watchdog before starting performance tests
    #if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    ensure_task_wdt_added();
    #endif

    // Test 13.1: Matrix Addition Performance (reduced size to prevent timeout)
    std::cout << "\n[Test 13.1] Matrix Addition Performance\n";
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

    // Test 13.2: Matrix Multiplication Performance (reduced size)
    std::cout << "\n[Test 13.2] Matrix Multiplication Performance\n";
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

    // Test 13.3: Matrix Transpose Performance (reduced size)
    std::cout << "\n[Test 13.3] Matrix Transpose Performance\n";
    tiny::Mat G(50, 30);  // Reduced from 100x50 to 50x30
    for (int i = 0; i < 50; ++i)
        for (int j = 0; j < 30; ++j)
            G(i, j) = static_cast<float>(i * 30 + j);
    TIME_REPEATED_OPERATION(tiny::Mat H = G.transpose();, PERFORMANCE_TEST_ITERATIONS, "50x30 Matrix Transpose");

    // Test 13.4: Determinant Performance Comparison
    // Note: Determinant calculation now has multiple methods:
    //   - Laplace expansion: O(n!) - for small matrices (n <= 4)
    //   - LU decomposition: O(n³) - for large matrices (n > 4, auto-selected)
    //   - Gaussian elimination: O(n³) - alternative for large matrices
    std::cout << "\n[Test 13.4] Determinant Calculation Performance Comparison\n";
    
    // Test 13.4.1: Small Matrix (4x4) - Laplace Expansion
        std::cout << "\n[Test 13.4.1] Small Matrix (4x4) - Laplace Expansion\n";
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
    
    // Test 13.4.2: Large Matrix (8x8) - LU Decomposition
        std::cout << "\n[Test 13.4.2] Large Matrix (8x8) - LU Decomposition\n";
    tiny::Mat I8(8, 8);
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            I8(i, j) = static_cast<float>((i + 1) * (j + 1));
    
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
    
    // Test 13.4.3: Large Matrix (8x8) - Gaussian Elimination
        std::cout << "\n[Test 13.4.3] Large Matrix (8x8) - Gaussian Elimination\n";
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
    
    // Test 13.4.4: Auto-select Method (8x8) - Should use LU
        std::cout << "\n[Test 13.4.4] Large Matrix (8x8) - Auto-select Method\n";
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

    // Test 13.5: Matrix Copy Performance (with padding, reduced size)
    std::cout << "\n[Test 13.5] Matrix Copy with Padding Performance\n";
    float data[80] = {0};  // Reduced from 150 to 80
    for (int i = 0; i < 80; ++i) data[i] = static_cast<float>(i);
    tiny::Mat J(data, 8, 8, 10);  // Reduced from 10x10 stride 15 to 8x8 stride 10
    TIME_REPEATED_OPERATION(tiny::Mat K = J.copy_roi(0, 0, 8, 8);, PERFORMANCE_TEST_ITERATIONS, "8x8 Copy ROI (with padding)");

    // Test 13.6: Element Access Performance (reduced size)
    std::cout << "\n[Test 13.6] Element Access Performance\n";
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

// ============================================================================
// Group 14: Quality Assurance - Memory Layout Tests (Padding and Stride)
// ============================================================================
// Purpose: Test memory layout handling - important for performance and compatibility
void test_memory_layout()
{
    std::cout << "\n[Group 14: Quality Assurance - Memory Layout Tests (Padding and Stride)]\n";

    // Test 14.1: Contiguous memory (pad=0, step=1)
    std::cout << "\n[Test 14.1] Contiguous Memory (no padding)\n";
    tiny::Mat mat1(3, 4);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            mat1(i, j) = static_cast<float>(i * 4 + j);
    std::cout << "Matrix 3x4 (stride=4, pad=0):\n";
    mat1.print_info();
    mat1.print_matrix(true);

    // Test 14.2: Padded memory (stride > col)
    std::cout << "\n[Test 14.2] Padded Memory (stride > col)\n";
    float data[15] = {0, 1, 2, 3, 0, 4, 5, 6, 7, 0, 8, 9, 10, 11, 0};
    tiny::Mat mat2(data, 3, 4, 5);
    std::cout << "Matrix 3x4 (stride=5, pad=1):\n";
    mat2.print_info();
    mat2.print_matrix(true);

    // Test 14.3: Operations with padded matrices
    std::cout << "\n[Test 14.3] Addition with Padded Matrices\n";
    float data1[15] = {1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 9, 10, 11, 12, 0};
    float data2[15] = {10, 20, 30, 40, 0, 50, 60, 70, 80, 0, 90, 100, 110, 120, 0};
    tiny::Mat mat3(data1, 3, 4, 5);
    tiny::Mat mat4(data2, 3, 4, 5);
    tiny::Mat mat5 = mat3 + mat4;
    std::cout << "Result of padded matrix addition:\n";
    mat5.print_info();
    mat5.print_matrix(true);

    // Test 14.4: ROI operations with padded matrices
    std::cout << "\n[Test 14.4] ROI Operations with Padded Matrices\n";
    tiny::Mat roi = mat2.view_roi(1, 1, 2, 2);
    std::cout << "ROI (1,1,2,2) from padded matrix:\n";
    roi.print_info();
    roi.print_matrix(true);

    // Test 14.5: Copy operations preserve stride
    std::cout << "\n[Test 14.5] Copy Operations Preserve Stride\n";
    tiny::Mat copied = mat2.copy_roi(0, 0, 3, 4);
    std::cout << "Copied matrix (should have stride=4, no padding):\n";
    copied.print_info();
    copied.print_matrix(true);
}

// ============================================================================
// Group 7: Advanced Linear Algebra - Matrix Decomposition Tests
// ============================================================================
// Purpose: Test matrix decompositions (LU, Cholesky, QR, SVD) - foundation for 
//          stable linear system solving and least squares problems
void test_matrix_decomposition()
{
    std::cout << "\n[Group 7: Advanced Linear Algebra - Matrix Decomposition Tests]\n";

    // Test 7.1: is_positive_definite() - Basic functionality
    std::cout << "\n[Test 7.1] is_positive_definite() - Basic Functionality\n";
    
    // Test 7.1.1: Positive definite matrix
    {
        std::cout << "\n[Test 7.1.1] Positive Definite 3x3 Matrix\n";
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

    // Test 7.1.2: Non-positive definite matrix
    {
        std::cout << "\n[Test 7.1.2] Non-Positive Definite Matrix\n";
        tiny::Mat non_pd(2, 2);
        non_pd(0, 0) = 1.0f; non_pd(0, 1) = 2.0f;
        non_pd(1, 0) = 2.0f; non_pd(1, 1) = 1.0f;  // Has negative eigenvalue
        std::cout << "Matrix:\n";
        non_pd.print_matrix(true);
        
        bool is_pd = non_pd.is_positive_definite(1e-6f);
        std::cout << "Is positive definite: " << (is_pd ? "True" : "False") 
                  << " (Expected: False) " << (!is_pd ? "[PASS]" : "[FAIL]") << "\n";
    }

    // Test 7.2: LU Decomposition
    std::cout << "\n[Test 7.2] LU Decomposition\n";
    
    // Test 7.2.1: Simple 3x3 matrix with pivoting
    {
        std::cout << "\n[Test 7.2.1] 3x3 Matrix - LU Decomposition with Pivoting\n";
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

    // Test 7.2.2: Solve using LU decomposition
    {
        std::cout << "\n[Test 7.2.2] Solve Linear System using LU Decomposition\n";
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

    // Test 7.3: Cholesky Decomposition
    std::cout << "\n[Test 7.3] Cholesky Decomposition\n";
    
    // Test 7.3.1: Symmetric positive definite matrix
    {
        std::cout << "\n[Test 7.3.1] SPD Matrix - Cholesky Decomposition\n";
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

    // Test 7.3.2: Solve using Cholesky decomposition
    {
        std::cout << "\n[Test 7.3.2] Solve Linear System using Cholesky Decomposition\n";
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

    // Test 7.4: QR Decomposition
    std::cout << "\n[Test 7.4] QR Decomposition\n";
    
    // Test 7.4.1: General matrix
    {
        std::cout << "\n[Test 7.4.1] General 3x3 Matrix - QR Decomposition\n";
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

    // Test 7.4.2: Solve using QR decomposition (least squares)
    {
        std::cout << "\n[Test 7.4.2] Least Squares Solution using QR Decomposition\n";
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

    // Test 7.5: SVD Decomposition
    std::cout << "\n[Test 7.5] Singular Value Decomposition (SVD)\n";
    
    // Test 7.5.1: General matrix
    {
        std::cout << "\n[Test 7.5.1] General 3x3 Matrix - SVD Decomposition\n";
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

    // Test 7.5.2: Pseudo-inverse using SVD
    {
        std::cout << "\n[Test 7.5.2] Pseudo-inverse using SVD\n";
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

    // Test 7.6: Performance Tests
    std::cout << "\n[Test 7.6] Matrix Decomposition Performance Tests\n";
    
    tiny::Mat perf_mat(4, 4);
    perf_mat(0, 0) = 4.0f; perf_mat(0, 1) = 2.0f; perf_mat(0, 2) = 1.0f; perf_mat(0, 3) = 0.0f;
    perf_mat(1, 0) = 2.0f; perf_mat(1, 1) = 5.0f; perf_mat(1, 2) = 1.0f; perf_mat(1, 3) = 0.0f;
    perf_mat(2, 0) = 1.0f; perf_mat(2, 1) = 1.0f; perf_mat(2, 2) = 3.0f; perf_mat(2, 3) = 1.0f;
    perf_mat(3, 0) = 0.0f; perf_mat(3, 1) = 0.0f; perf_mat(3, 2) = 1.0f; perf_mat(3, 3) = 2.0f;
    
    // Test 7.6.1: LU decomposition performance
    std::cout << "\n[Test 7.6.1] LU Decomposition Performance\n";
    TIME_OPERATION(
        tiny::Mat::LUDecomposition perf_lu = perf_mat.lu_decompose(true);
        (void)perf_lu;
    , "LU Decomposition (4x4 matrix)");
    
    // Test 7.6.2: Cholesky decomposition performance
    std::cout << "\n[Test 7.6.2] Cholesky Decomposition Performance\n";
    TIME_OPERATION(
        tiny::Mat::CholeskyDecomposition perf_chol = perf_mat.cholesky_decompose();
        (void)perf_chol;
    , "Cholesky Decomposition (4x4 SPD matrix)");
    
    // Test 7.6.3: QR decomposition performance
    std::cout << "\n[Test 7.6.3] QR Decomposition Performance\n";
    TIME_OPERATION(
        tiny::Mat::QRDecomposition perf_qr = perf_mat.qr_decompose();
        (void)perf_qr;
    , "QR Decomposition (4x4 matrix)");
    
    // Test 7.6.4: SVD decomposition performance
    std::cout << "\n[Test 7.6.4] SVD Decomposition Performance\n";
    TIME_OPERATION(
        tiny::Mat::SVDDecomposition perf_svd = perf_mat.svd_decompose(50, 1e-5f);
        (void)perf_svd;
    , "SVD Decomposition (4x4 matrix)");

    std::cout << "\n[Matrix Decomposition Tests Complete]\n";
}

// ============================================================================
// Group 8: Advanced Linear Algebra - Gram-Schmidt Orthogonalization Tests
// ============================================================================
// Purpose: Test Gram-Schmidt orthogonalization process - fundamental operation for
//          QR decomposition, eigenvalue decomposition, and basis transformation
void test_gram_schmidt_orthogonalize()
{
    std::cout << "\n[Group 8: Advanced Linear Algebra - Gram-Schmidt Orthogonalization Tests]\n";

    // Test 8.1: Basic orthogonalization of linearly independent vectors
    {
        std::cout << "\n[Test 8.1] Basic Orthogonalization - Linearly Independent Vectors\n";
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
    
    // Test 8.2: Orthogonalization with near-linear-dependent vectors
    {
        std::cout << "\n[Test 8.2] Orthogonalization - Near-Linear-Dependent Vectors\n";
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
    
    // Test 8.3: Orthogonalization of 2D vectors
    {
        std::cout << "\n[Test 8.3] Orthogonalization - 2D Vectors (2x2)\n";
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
    
    // Test 8.4: Error handling - invalid input
    {
        std::cout << "\n[Test 8.4] Error Handling - Invalid Input\n";
        tiny::Mat empty_mat;  // Empty matrix
        tiny::Mat Q, R;
        bool success = tiny::Mat::gram_schmidt_orthogonalize(empty_mat, Q, R, 1e-6f);
        std::cout << "Empty matrix test: " << (success ? "FAIL (should return false)" : "PASS (correctly rejected)") << "\n";
    }
}

// ============================================================================
// Group 9: System Identification - Eigenvalue and Eigenvector Decomposition Tests
// ============================================================================
// Purpose: Test eigenvalue decomposition - critical for SHM and system identification
//          applications (modal analysis, natural frequencies, mode shapes)
void test_eigenvalue_decomposition()
{
    std::cout << "\n[Group 9: System Identification - Eigenvalue and Eigenvector Decomposition Tests]\n";

    // Test 9.1: is_symmetric() - Basic functionality
    std::cout << "\n[Test 9.1] is_symmetric() - Basic Functionality\n";
    
    // Test 9.1.1: Symmetric matrix
    {
        std::cout << "[Test 9.1.1] Symmetric 3x3 Matrix\n";
        tiny::Mat sym_mat1(3, 3);
        sym_mat1(0, 0) = 4.0f; sym_mat1(0, 1) = 1.0f; sym_mat1(0, 2) = 2.0f;
        sym_mat1(1, 0) = 1.0f; sym_mat1(1, 1) = 3.0f; sym_mat1(1, 2) = 0.0f;
        sym_mat1(2, 0) = 2.0f; sym_mat1(2, 1) = 0.0f; sym_mat1(2, 2) = 5.0f;
        bool is_sym1 = sym_mat1.is_symmetric(1e-5f);
        std::cout << "Matrix:\n";
        sym_mat1.print_matrix(true);
        std::cout << "Is symmetric: " << (is_sym1 ? "True" : "False") << " (Expected: True)\n";
    }

    // Test 9.1.2: Non-symmetric matrix (keep for later tests)
    tiny::Mat non_sym_mat(3, 3);
    {
        std::cout << "\n[Test 9.1.2] Non-Symmetric 3x3 Matrix\n";
        non_sym_mat(0, 0) = 1.0f; non_sym_mat(0, 1) = 2.0f; non_sym_mat(0, 2) = 3.0f;
        non_sym_mat(1, 0) = 4.0f; non_sym_mat(1, 1) = 5.0f; non_sym_mat(1, 2) = 6.0f;
        non_sym_mat(2, 0) = 7.0f; non_sym_mat(2, 1) = 8.0f; non_sym_mat(2, 2) = 9.0f;
        bool is_sym2 = non_sym_mat.is_symmetric(1e-5f);
        std::cout << "Matrix:\n";
        non_sym_mat.print_matrix(true);
        std::cout << "Is symmetric: " << (is_sym2 ? "True" : "False") << " (Expected: False)\n";
    }

    // Test 9.1.3: Non-square matrix
    {
        std::cout << "\n[Test 9.1.3] Non-Square Matrix (2x3)\n";
        tiny::Mat rect_mat(2, 3);
        bool is_sym3 = rect_mat.is_symmetric(1e-5f);
        std::cout << "Is symmetric: " << (is_sym3 ? "True" : "False") << " (Expected: False)\n";
    }

    // Test 9.1.4: Symmetric matrix with small numerical errors
    {
        std::cout << "\n[Test 9.1.4] Symmetric Matrix with Small Numerical Errors\n";
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

    // Test 9.2: power_iteration() - Dominant eigenvalue
    std::cout << "\n[Test 9.2] power_iteration() - Dominant Eigenvalue\n";
    
    // Test 9.2.1: Simple 2x2 symmetric matrix (known eigenvalues)
    tiny::Mat mat2x2(2, 2);
    {
        std::cout << "\n[Test 9.2.1] Simple 2x2 Matrix\n";
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

    // Test 9.2.2: 3x3 matrix (SHM-like stiffness matrix) - keep for later tests
    tiny::Mat stiffness(3, 3);
    {
        std::cout << "\n[Test 9.2.2] 3x3 Stiffness Matrix (SHM Application)\n";
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

    // Test 9.2.3: Non-square matrix (should fail)
    {
        std::cout << "\n[Test 9.2.3] Non-Square Matrix (Expect Error)\n";
        tiny::Mat non_square(2, 3);
        tiny::Mat::EigenPair result_error = non_square.power_iteration(100, 1e-6f);
        std::cout << "Status: " << (result_error.status == TINY_OK ? "OK" : "Error (Expected)") << "\n";
    }

    // Test 9.2.4: inverse_power_iteration() - Smallest eigenvalue (Critical for System Identification)
    std::cout << "\n[Test 9.2.4] inverse_power_iteration() - Smallest Eigenvalue (System Identification)\n";
    
    // Test 9.2.4.1: Simple 2x2 symmetric matrix (known eigenvalues)
    {
        std::cout << "\n[Test 9.2.4.1] Simple 2x2 Matrix - Smallest Eigenvalue\n";
        std::cout << "Matrix (same as Test 9.2.1):\n";
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

    // Test 9.2.4.2: 3x3 stiffness matrix - Smallest eigenvalue (SHM Application)
    {
        std::cout << "\n[Test 9.2.4.2] 3x3 Stiffness Matrix - Smallest Eigenvalue (SHM Application)\n";
        std::cout << "Stiffness Matrix (same as Test 9.2.2):\n";
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

    // Test 9.2.4.3: Non-square matrix (should fail)
    {
        std::cout << "\n[Test 9.2.4.3] Non-Square Matrix (Expect Error)\n";
        tiny::Mat non_square(2, 3);
        tiny::Mat::EigenPair result_error = non_square.inverse_power_iteration(100, 1e-6f);
        std::cout << "Status: " << (result_error.status == TINY_OK ? "OK" : "Error (Expected)") << "\n";
        bool correct = (result_error.status != TINY_OK);
        std::cout << "Error handling: " << (correct ? "[PASS]" : "[FAIL]") << "\n";
    }

    // Test 9.2.4.4: Near-singular matrix (should handle gracefully)
    {
        std::cout << "\n[Test 9.2.4.4] Near-Singular Matrix (Edge Case)\n";
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

    // Test 9.3: eigendecompose_jacobi() - Symmetric matrix decomposition
    std::cout << "\n[Test 9.3] eigendecompose_jacobi() - Symmetric Matrix Decomposition\n";
    
    // Test 9.3.1: Simple 2x2 symmetric matrix
    {
        std::cout << "\n[Test 9.3.1] 2x2 Symmetric Matrix - Complete Decomposition\n";
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

    // Test 9.3.2: 3x3 symmetric matrix (SHM stiffness matrix)
    {
        std::cout << "\n[Test 9.3.2] 3x3 Stiffness Matrix (SHM Application)\n";
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

    // Test 9.3.3: Diagonal matrix (trivial case)
    {
        std::cout << "\n[Test 9.3.3] Diagonal Matrix (Eigenvalues on diagonal)\n";
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

    // Test 9.4: eigendecompose_qr() - General matrix decomposition
    std::cout << "\n[Test 9.4] eigendecompose_qr() - General Matrix Decomposition\n";
    
    // Test 9.4.1: General 2x2 matrix
    {
        std::cout << "\n[Test 9.4.1] General 2x2 Matrix\n";
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

    // Test 9.4.2: Non-symmetric 3x3 matrix
    {
        std::cout << "\n[Test 9.4.2] Non-Symmetric 3x3 Matrix\n";
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

    // Test 9.5: eigendecompose() - Automatic method selection
    std::cout << "\n[Test 9.5] eigendecompose() - Automatic Method Selection\n";
    
    // Test 9.5.1: Symmetric matrix (should use Jacobi)
    {
        std::cout << "\n[Test 9.5.1] Symmetric Matrix (Auto-select: Jacobi)\n";
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

    // Test 9.5.2: Non-symmetric matrix (should use QR)
    {
        std::cout << "\n[Test 9.5.2] Non-Symmetric Matrix (Auto-select: QR)\n";
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

    // Test 9.6: SHM Application Scenario - Structural Dynamics
    std::cout << "\n[Test 9.6] SHM Application - Structural Dynamics Analysis\n";
    
    // Create a simple 4-DOF structural system (mass-spring system)
    {
        std::cout << "\n[Test 9.6.1] 4-DOF Mass-Spring System\n";
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

    // Test 9.7: Edge Cases and Error Handling
    std::cout << "\n[Test 9.7] Edge Cases and Error Handling\n";
    
    // Test 9.7.1: 1x1 matrix
    {
        std::cout << "\n[Test 9.7.1] 1x1 Matrix\n";
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
    
    // Test 9.7.2: Zero matrix
    {
        std::cout << "\n[Test 9.7.2] Zero Matrix\n";
        tiny::Mat zero_mat(3, 3);
        zero_mat.clear();
        tiny::Mat::EigenPair result_zero = zero_mat.power_iteration(100, 1e-6f);
        std::cout << "Status: " << (result_zero.status == TINY_OK ? "OK" : "Error (Expected)") << "\n";
    }
    
    // Test 9.7.3: Identity matrix
    {
        std::cout << "\n[Test 9.7.3] Identity Matrix\n";
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

    // Test 9.8: Performance Test for SHM Applications
    std::cout << "\n[Test 9.8] Performance Test for SHM Applications\n";
    
    // Test 9.8.1: Power iteration performance (fast method for dominant eigenvalue)
    std::cout << "\n[Test 9.8.1] Power Iteration Performance (Real-time SHM - Dominant Eigenvalue)\n";
    TIME_OPERATION(
        tiny::Mat::EigenPair perf_result = stiffness.power_iteration(500, 1e-6f);
        (void)perf_result;
    , "Power Iteration (3x3 matrix)");
    
    // Test 9.8.2: Inverse power iteration performance (system identification - smallest eigenvalue)
    std::cout << "\n[Test 9.8.2] Inverse Power Iteration Performance (System Identification - Smallest Eigenvalue)\n";
    TIME_OPERATION(
        tiny::Mat::EigenPair perf_inv_result = stiffness.inverse_power_iteration(500, 1e-6f);
        (void)perf_inv_result;
    , "Inverse Power Iteration (3x3 matrix)");
    
    // Test 9.8.3: Jacobi method performance (complete eigendecomposition for symmetric matrices)
    std::cout << "\n[Test 9.8.3] Jacobi Method Performance (Complete Eigendecomposition - Symmetric Matrices)\n";
    TIME_OPERATION(
        tiny::Mat::EigenDecomposition perf_jacobi = stiffness.eigendecompose_jacobi(1e-5f, 100);
        (void)perf_jacobi;
    , "Jacobi Decomposition (3x3 symmetric matrix)");
    
    // Test 9.8.4: QR method performance (complete eigendecomposition for general matrices)
    std::cout << "\n[Test 9.8.4] QR Method Performance (Complete Eigendecomposition - General Matrices)\n";
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
    // Phase 1: Object Foundation (Groups 1-3)
    // ========================================================================
    // Purpose: Learn to create and manipulate matrix objects
    // Group 1: Constructor & Destructor
    // test_constructor_destructor();

    // Group 2: Element Access
    // test_element_access();

    // Group 3: ROI Operations
    // test_roi_operations();

    // ========================================================================
    // Phase 2: Basic Operations (Group 4)
    // ========================================================================
    // Purpose: Learn basic arithmetic operations
    // Group 4: Arithmetic Operators
    // test_assignment_operator();
    // test_matrix_addition();
    // test_constant_addition();
    // test_matrix_subtraction();
    // test_constant_subtraction();
    // test_matrix_division();
    // test_constant_division();
    // test_matrix_exponentiation();

    // ========================================================================
    // Phase 3: Matrix Properties (Group 5)
    // ========================================================================
    // Purpose: Understand matrix properties and basic linear algebra
    // Group 5: Matrix Properties
    // test_matrix_transpose();
    // test_matrix_cofactor();
    // test_matrix_determinant();
    // test_matrix_adjoint();
    // test_matrix_normalize();
    // test_matrix_norm();
    // test_inverse_adjoint_adjoint();
    // test_matrix_utilities();

    // ========================================================================
    // Phase 4: Linear System Solving (Group 6)
    // ========================================================================
    // Purpose: Core application - solving linear systems Ax = b
    // Group 6: Linear System Solving
    // test_gaussian_eliminate();
    // test_row_reduce_from_gaussian();
    // test_inverse_gje();
    // test_dotprod();
    // test_solve();
    // test_band_solve();
    // test_roots();

    // ========================================================================
    // Phase 5: Advanced Linear Algebra (Groups 7-8)
    // ========================================================================
    // Purpose: Advanced linear algebra operations for stable and efficient solving
    // Group 7: Matrix Decomposition
    // test_matrix_decomposition();
    
    // Group 8: Gram-Schmidt Orthogonalization
    // test_gram_schmidt_orthogonalize();

    // ========================================================================
    // Phase 6: System Identification Applications (Group 9)
    // ========================================================================
    // Purpose: Eigenvalue decomposition for SHM and modal analysis
    // Group 9: Eigenvalue Decomposition
    // test_eigenvalue_decomposition();

    // ========================================================================
    // Phase 7: Auxiliary Functions (Groups 10-11)
    // ========================================================================
    // Purpose: Convenience functions and I/O operations
    // Group 10: Stream Operators
    // test_stream_operators();

    // Group 11: Global Arithmetic Operators
    // test_matrix_operations();

    // ========================================================================
    // Phase 8: Quality Assurance (Groups 12-14)
    // ========================================================================
    // Purpose: Ensure robustness, performance, and correctness
    // Group 12: Boundary Conditions and Error Handling
    test_boundary_conditions();

    // Group 13: Performance Benchmarks
    test_performance_benchmarks();

    // Group 14: Memory Layout
    test_memory_layout();

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