/**
 * @file tiny_matrix.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the source file for the submodule matrix (advanced matrix operations) of the tiny_math middleware.
 * @version 1.0
 * @date 2025-04-17
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
// TinyMath
#include "tiny_matrix.hpp"

// Standard Libraries
#include <cstring>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cinttypes>
#include <iomanip>
#include <vector>
#include <limits>

/* LIBRARY CONTENTS */
namespace tiny
{
    // ============================================================================
    // Rectangular ROI Structure
    // ============================================================================
    /**
     * @brief Construct a new Mat::ROI object.
     *
     * @param pos_x
     * @param pos_y
     * @param width
     * @param height
     */
    Mat::ROI::ROI(int pos_x, int pos_y, int width, int height)
    {
        this->pos_x = pos_x;
        this->pos_y = pos_y;
        this->width = width;
        this->height = height;
    }

    /**
     * @brief resize the ROI structure
     * 
     * @param pos_x starting column
     * @param pos_y starting row
     * @param width number of columns
     * @param height number of rows
     */
    void Mat::ROI::resize_roi(int pos_x, int pos_y, int width, int height)
    {
        this->pos_x = pos_x;
        this->pos_y = pos_y;
        this->width = width;
        this->height = height;
    }

    /**
     * @brief calculate the area of the ROI structure - how many elements covered
     * 
     * @return int 
     */
    int Mat::ROI::area_roi(void) const
    {
        if (this->width < 0 || this->height < 0)
        {
            return 0;
        }
        return this->width * this->height;
    }

    // ============================================================================
    // Printing Functions
    // ============================================================================
    /**
     * @name Mat::print_info()
     * @brief Print the header of the matrix.
     */
    void Mat::print_info() const
    {
        std::cout << "Matrix Info >>>\n";

        // Basic matrix metadata
        std::cout << "rows            " << this->row << "\n";
        std::cout << "cols            " << this->col << "\n";
        std::cout << "elements        " << this->element;

        // Check if elements match rows * cols
        if (this->element != this->row * this->col)
        {
            std::cout << "   [Warning] Mismatch! Expected: " << (this->row * this->col);
        }
        std::cout << "\n";

        std::cout << "paddings        " << this->pad << "\n";
        std::cout << "step            " << this->step << "\n";
        std::cout << "memory          " << this->memory << "\n";

        // Pointer information
        std::cout << "data pointer    " << static_cast<const void *>(this->data) << "\n";
        std::cout << "temp pointer    " << static_cast<const void *>(this->temp) << "\n";

        // Flags information
        std::cout << "ext_buff        " << this->ext_buff;
        if (this->ext_buff)
        {
            std::cout << "   (External buffer or View)";
        }
        std::cout << "\n";

        std::cout << "sub_matrix      " << this->sub_matrix;
        if (this->sub_matrix)
        {
            std::cout << "   (This is a Sub-Matrix View)";
        }
        std::cout << "\n";

        // State warnings
        if (this->sub_matrix && !this->ext_buff)
        {
            std::cout << "[Warning] Sub-matrix is marked but ext_buff is false! Potential logic error.\n";
        }

        if (this->data == nullptr)
        {
            std::cout << "[Info] No data buffer assigned to this matrix.\n";
        }

        std::cout << "<<< Matrix Info\n";
    }

    /**
     * @name Mat::print_matrix()
     * @brief Print the matrix elements.
     *
     * @param show_padding If true, print the padding elements as well.
     */
    void Mat::print_matrix(bool show_padding) const
    {
        if (this->data == nullptr)
        {
            std::cout << "[Error] Cannot print matrix: data pointer is null.\n";
            return;
        }
        
        if (this->row < 0 || this->col < 0 || this->step < 0)
        {
            std::cout << "[Error] Invalid matrix dimensions\n";
            return;
        }
        
        const int print_cols = (this->step < this->col) ? this->step : this->col;
        if (this->step < this->col)
        {
            std::cout << "[Warning] step < cols; printing only the first " << print_cols
                      << " column(s) per row to avoid out-of-bounds access.\n";
        }

        std::cout << "Matrix Elements >>>\n";
        for (int i = 0; i < this->row; ++i)
        {
            // print the non-padding elements (bounded by step)
            for (int j = 0; j < print_cols; ++j)
            {
                std::cout << std::setw(12) << this->data[i * this->step + j] << " ";
            }

            // if padding is enabled, print the padding elements
            if (show_padding)
            {
                // print a separator first
                std::cout << "      |";

                // print the padding elements
                for (int j = this->col; j < this->step; ++j)
                {
                    if (j == this->col)
                    {
                        std::cout << std::setw(7) << this->data[i * this->step + j] << " ";
                    }
                    else
                    {
                        // print the padding elements
                        std::cout << std::setw(12) << this->data[i * this->step + j] << " ";
                    }
                }
            }

            // print a new line after each row
            std::cout << "\n";
        }

        std::cout << "<<< Matrix Elements\n";
        std::cout << std::endl;
    }

    // ============================================================================
    // Constructors & Destructor
    // ============================================================================
    // memory allocation
    /**
     * @name Mat::alloc_mem()
     * @brief Allocate memory for the matrix according to the memory required.
     * @note For ESP32, it will automatically determine if using RAM or PSRAM based on the size of the matrix.
     * @note This function sets ext_buff to false and allocates memory based on row * step.
     *       If allocation fails or parameters are invalid, data will be set to nullptr.
     */
    void Mat::alloc_mem()
    {
        // Parameter validation: check if row and step are non-negative
        if (this->row < 0 || this->step < 0)
        {
            std::cerr << "[Error] Invalid matrix dimensions in alloc_mem(): row=" << this->row 
                      << ", step=" << this->step << "\n";
            this->data = nullptr;
            this->ext_buff = false;
            this->memory = 0;
            return;
        }
        
        // Check for integer overflow: row * step might overflow
        if (this->row > 0 && this->step > INT_MAX / this->row)
        {
            std::cerr << "[Error] Matrix size too large, integer overflow: row=" << this->row 
                      << ", step=" << this->step << "\n";
            this->data = nullptr;
            this->ext_buff = false;
            this->memory = 0;
            return;
        }
        
        this->ext_buff = false;
        this->memory = this->row * this->step;
        
        // Handle empty matrix case (memory = 0)
        if (this->memory == 0)
        {
            this->data = nullptr;
            return;
        }
        
        // Use nothrow new to return nullptr on failure instead of throwing exception
        // This allows callers to check for nullptr, which is consistent with existing code
        this->data = new(std::nothrow) float[this->memory];
        
        // If allocation failed, data will be nullptr (caller should check)
        if (this->data == nullptr)
        {
            this->memory = 0;
        }
    }

    /**
     * @name Mat::Mat()
     * @brief Constructor - default constructor: create a 1x1 matrix with only a zero element.
     * @note If memory allocation fails, the object will be in an invalid state (data = nullptr).
     *       Caller should check the data pointer before using the matrix.
     */
    Mat::Mat()
        : row(1), col(1), pad(0), step(1), element(1), memory(1),
          data(nullptr), temp(nullptr),
          ext_buff(false), sub_matrix(false)
    {
        // memory will be recalculated by alloc_mem() based on row * step
        alloc_mem();
        if (this->data == nullptr && this->memory > 0)
        {
            std::cerr << "[>>> Error ! <<<] Memory allocation failed in alloc_mem()\n";
            // Memory allocation failed, object is in invalid state (data = nullptr)
            // Caller should check data pointer before using the matrix
            return;
        }
        // Initialize all elements to zero
        std::memset(this->data, 0, this->memory * sizeof(float));
    }

    /**
     * @name Mat::Mat(int rows, int cols)
     * @brief Constructor - create a matrix with the specified number of rows and columns.
     * @param rows Number of rows (must be non-negative)
     * @param cols Number of columns (must be non-negative)
     * @note If rows or cols is negative, the object will be in an invalid state.
     * @note If memory allocation fails, the object will be in an invalid state (data = nullptr).
     *       Caller should check the data pointer before using the matrix.
     */
    Mat::Mat(int rows, int cols)
        : row(rows), col(cols), pad(0), step(cols),
          element(rows * cols), memory(rows * cols),
          data(nullptr), temp(nullptr),
          ext_buff(false), sub_matrix(false)
    {
        // Parameter validation: check if rows and cols are non-negative
        if (rows < 0 || cols < 0)
        {
            std::cerr << "[Error] Invalid matrix dimensions: rows=" << rows 
                      << ", cols=" << cols << " (must be non-negative)\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        // Check for integer overflow: rows * cols might overflow
        if (rows > 0 && cols > INT_MAX / rows)
        {
            std::cerr << "[Error] Matrix size too large, integer overflow: rows=" << rows 
                      << ", cols=" << cols << "\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        // Empty matrix is a valid state; keep data == nullptr without allocation.
        if (rows == 0 || cols == 0)
        {
            this->element = 0;
            this->memory = 0;
            this->data = nullptr;
            return;
        }

        // memory will be recalculated by alloc_mem() based on row * step
        alloc_mem();
        if (this->data == nullptr && this->memory > 0)
        {
            std::cerr << "[>>> Error ! <<<] Memory allocation failed in alloc_mem()\n";
            // Memory allocation failed, object is in invalid state (data = nullptr)
            // Caller should check data pointer before using the matrix
            return;
        }
        // Initialize all elements to zero
        std::memset(this->data, 0, this->memory * sizeof(float));
    }
    /**
     * @name Mat::Mat(int rows, int cols, int step)
     * @brief Constructor - create a matrix with the specified number of rows, columns and step.
     * @param rows Number of rows (must be non-negative)
     * @param cols Number of columns (must be non-negative)
     * @param step Total number of elements in a row (must be >= cols)
     * @note If rows, cols is negative, or step < cols, the object will be in an invalid state.
     * @note If memory allocation fails, the object will be in an invalid state (data = nullptr).
     *       Caller should check the data pointer before using the matrix.
     */
    Mat::Mat(int rows, int cols, int step)
        : row(rows), col(cols), pad(step - cols), step(step),
          element(rows * cols), memory(rows * step),
          data(nullptr), temp(nullptr),
          ext_buff(false), sub_matrix(false)
    {
        // Parameter validation: check if rows, cols are non-negative
        if (rows < 0 || cols < 0)
        {
            std::cerr << "[Error] Invalid matrix dimensions: rows=" << rows 
                      << ", cols=" << cols << " (must be non-negative)\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        // Validate step: must be >= cols (padding cannot be negative)
        if (step < cols)
        {
            std::cerr << "[Error] Invalid step: step=" << step 
                      << " must be >= cols=" << cols << "\n";
            this->data = nullptr;
            this->memory = 0;
            this->pad = 0;  // Reset pad to avoid negative value
            return;
        }
        
        // Check for integer overflow: rows * cols and rows * step might overflow
        if (rows > 0 && cols > INT_MAX / rows)
        {
            std::cerr << "[Error] Matrix size too large, integer overflow: rows=" << rows 
                      << ", cols=" << cols << "\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        if (rows > 0 && step > INT_MAX / rows)
        {
            std::cerr << "[Error] Matrix size too large, integer overflow: rows=" << rows 
                      << ", step=" << step << "\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        // Empty matrix is a valid state; keep data == nullptr without allocation.
        if (rows == 0 || cols == 0)
        {
            this->element = 0;
            this->memory = 0;
            this->data = nullptr;
            return;
        }

        // memory will be recalculated by alloc_mem() based on row * step
        alloc_mem();
        if (this->data == nullptr && this->memory > 0)
        {
            std::cerr << "[>>> Error ! <<<] Memory allocation failed in alloc_mem()\n";
            // Memory allocation failed, object is in invalid state (data = nullptr)
            // Caller should check data pointer before using the matrix
            return;
        }
        // Initialize all elements to zero
        std::memset(this->data, 0, this->memory * sizeof(float));
    }

    /**
     * @name Mat::Mat(float *data, int rows, int cols)
     * @brief Constructor - create a matrix with the specified number of rows, columns and external data.
     * @param data Pointer to external data buffer (can be nullptr for empty matrix)
     * @param rows Number of rows (must be non-negative)
     * @param cols Number of columns (must be non-negative)
     * @note This constructor does not allocate memory. The matrix uses the external buffer.
     * @note If rows or cols is negative, the object will be in an invalid state.
     * @note The caller is responsible for ensuring the buffer is large enough and valid.
     */
    Mat::Mat(float *data, int rows, int cols)
        : row(rows), col(cols), pad(0), step(cols),
          element(rows * cols), memory(rows * cols),
          data(data), temp(nullptr),
          ext_buff(true), sub_matrix(false)
    {
        // Parameter validation: check if rows and cols are non-negative
        if (rows < 0 || cols < 0)
        {
            std::cerr << "[Error] Invalid matrix dimensions: rows=" << rows 
                      << ", cols=" << cols << " (must be non-negative)\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        // Check for integer overflow: rows * cols might overflow
        if (rows > 0 && cols > INT_MAX / rows)
        {
            std::cerr << "[Error] Matrix size too large, integer overflow: rows=" << rows 
                      << ", cols=" << cols << "\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        // Note: data can be nullptr for empty matrix, but caller should ensure buffer validity
    }

    /**
     * @name Mat::Mat(float *data, int rows, int cols, int step)
     * @brief Constructor - create a matrix with the specified number of rows, columns and external data.
     * @param data Pointer to external data buffer (can be nullptr for empty matrix)
     * @param rows Number of rows (must be non-negative)
     * @param cols Number of columns (must be non-negative)
     * @param step Total number of elements in a row (must be >= cols)
     * @note This constructor does not allocate memory. The matrix uses the external buffer.
     * @note If rows, cols is negative, or step < cols, the object will be in an invalid state.
     * @note The caller is responsible for ensuring the buffer is large enough and valid.
     */
    Mat::Mat(float *data, int rows, int cols, int step)
        : row(rows), col(cols), pad(step - cols), step(step),
          element(rows * cols), memory(rows * step),
          data(data), temp(nullptr),
          ext_buff(true), sub_matrix(false)
    {
        // Parameter validation: check if rows, cols are non-negative
        if (rows < 0 || cols < 0)
        {
            std::cerr << "[Error] Invalid matrix dimensions: rows=" << rows 
                      << ", cols=" << cols << " (must be non-negative)\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        // Validate step: must be >= cols (padding cannot be negative)
        if (step < cols)
        {
            std::cerr << "[Error] Invalid step: step=" << step 
                      << " must be >= cols=" << cols << "\n";
            this->data = nullptr;
            this->memory = 0;
            this->pad = 0;  // Reset pad to avoid negative value
            return;
        }
        
        // Check for integer overflow: rows * cols and rows * step might overflow
        if (rows > 0 && cols > INT_MAX / rows)
        {
            std::cerr << "[Error] Matrix size too large, integer overflow: rows=" << rows 
                      << ", cols=" << cols << "\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        if (rows > 0 && step > INT_MAX / rows)
        {
            std::cerr << "[Error] Matrix size too large, integer overflow: rows=" << rows 
                      << ", step=" << step << "\n";
            this->data = nullptr;
            this->memory = 0;
            return;
        }
        
        // Note: data can be nullptr for empty matrix, but caller should ensure buffer validity
    }

    /**
     * @name Mat::Mat(const Mat &src)
     * @brief Copy constructor - create a matrix with the same properties as the source matrix.
     * @param src Source matrix
     * @note If source is a submatrix view (sub_matrix && ext_buff), performs shallow copy (shares data pointer).
     *       Otherwise, performs deep copy (allocates new memory and copies data).
     * @note If memory allocation fails, the object will be in an invalid state (data = nullptr).
     *       Caller should check the data pointer before using the matrix.
     * @warning Shallow copy: If source is destroyed, the copied matrix's data pointer will be invalid.
     */
    Mat::Mat(const Mat &src)
        : row(src.row), col(src.col), pad(src.pad), step(src.step),
          element(src.element), memory(src.memory),
          data(nullptr), temp(nullptr),
          ext_buff(false), sub_matrix(false)
    {
        if (src.sub_matrix && src.ext_buff)
        {
            // if the source is a view (submatrix), do shallow copy
            // WARNING: This creates a shared reference. If source is destroyed, this pointer becomes invalid.
            this->data = src.data;
            this->ext_buff = true;
            this->sub_matrix = true;
        }
        else
        {
            // otherwise do deep copy
            this->ext_buff = false;
            this->sub_matrix = false;

            if (src.data != nullptr)
            {
                alloc_mem();
                if (this->data == nullptr)
                {
                    std::cerr << "[Error] Memory allocation failed in alloc_mem()\n";
                    // Memory allocation failed, object is in invalid state (data = nullptr)
                    // Caller should check data pointer before using the matrix
                    return;
                }
                
                // Copy data row by row to handle different strides correctly
                // This ensures correct copying even if source and destination have different strides
                for (int i = 0; i < this->row; ++i)
                {
                    std::memcpy(
                        &this->data[i * this->step],
                        &src.data[i * src.step],
                        this->col * sizeof(float)
                    );
                }
            }
            // If src.data == nullptr, this->data remains nullptr (empty matrix)
        }
    }

    /**
     * @name ~Mat()
     * @brief Destructor - free the memory allocated for the matrix.
     * @note Only deletes memory if it was allocated by this object (ext_buff == false).
     *       External buffers are not deleted.
     * @note temp buffer is always deleted if it exists (assumed to be allocated by this object).
     */
    Mat::~Mat()
    {
        // Only delete data if it was allocated by this object (not external buffer)
        if (!this->ext_buff && this->data != nullptr)
        {
            delete[] this->data;
            this->data = nullptr;  // Set to nullptr after deletion (good practice)
        }
        
        // Delete temporary buffer if it exists
        // Note: temp is assumed to be allocated by this object, not external
        if (this->temp != nullptr)
        {
            delete[] this->temp;
            this->temp = nullptr;  // Set to nullptr after deletion (good practice)
        }
    }

    // ============================================================================
    // Element Access
    // ============================================================================
    // Already defined by inline functions in the header file

    // ============================================================================
    // Data Manipulation
    // ============================================================================

    /**
     * @name Mat::copy_paste(const Mat &src, int row_pos, int col_pos)
     * @brief Copy the elements of the source matrix into the destination matrix. 
     *        The dimension of the current matrix must be larger than the source matrix.
     * @brief This one does not share memory with the source matrix.
     * @param src Source matrix (must be valid and non-empty)
     * @param row_pos Start row position of the destination matrix (must be non-negative)
     * @param col_pos Start column position of the destination matrix (must be non-negative)
     * @return TINY_OK on success, TINY_ERR_INVALID_ARG on error
     */
    tiny_error_t Mat::copy_paste(const Mat &src, int row_pos, int col_pos)
    {
        // Check for null pointers
        if (this->data == nullptr)
        {
            std::cerr << "[Error] copy_paste: destination matrix data pointer is null\n";
            return TINY_ERR_INVALID_ARG;
        }
        if (src.data == nullptr)
        {
            std::cerr << "[Error] copy_paste: source matrix data pointer is null\n";
            return TINY_ERR_INVALID_ARG;
        }
        
        // Validate source matrix dimensions
        if (src.row <= 0 || src.col <= 0)
        {
            std::cerr << "[Error] copy_paste: invalid source matrix dimensions: rows=" 
                      << src.row << ", cols=" << src.col << "\n";
            return TINY_ERR_INVALID_ARG;
        }
        
        // Validate position parameters (must be non-negative)
        if (row_pos < 0 || col_pos < 0)
        {
            std::cerr << "[Error] copy_paste: invalid position: row_pos=" << row_pos 
                      << ", col_pos=" << col_pos << " (must be non-negative)\n";
            return TINY_ERR_INVALID_ARG;
        }
        
        // Validate destination matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] copy_paste: invalid destination matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return TINY_ERR_INVALID_ARG;
        }
        
        // Check if source matrix fits in destination at the specified position
        if ((row_pos + src.row) > this->row)
        {
            std::cerr << "[Error] copy_paste: source matrix exceeds destination row boundary: "
                      << "row_pos=" << row_pos << ", src.rows=" << src.row 
                      << ", dest.rows=" << this->row << "\n";
            return TINY_ERR_INVALID_ARG;
        }
        if ((col_pos + src.col) > this->col)
        {
            std::cerr << "[Error] copy_paste: source matrix exceeds destination column boundary: "
                      << "col_pos=" << col_pos << ", src.cols=" << src.col 
                      << ", dest.cols=" << this->col << "\n";
            return TINY_ERR_INVALID_ARG;
        }
        
        // Copy data row by row (handles different strides correctly)
        for (int r = 0; r < src.row; r++)
        {
            memcpy(&this->data[(r + row_pos) * this->step + col_pos], 
                   &src.data[r * src.step], 
                   src.col * sizeof(float));
        }

        return TINY_OK;
    }

    /**
     * @name Mat::copy_head(const Mat &src)
     * @brief Copy the header (metadata) of the source matrix into the destination matrix. 
     *        The data pointer is shared (shallow copy).
     * @param src Source matrix (must be valid)
     * @return TINY_OK on success, TINY_ERR_INVALID_ARG on error
     * @warning This function performs a SHALLOW COPY. The destination matrix shares the 
     *          data pointer with the source matrix. If the source matrix is destroyed, 
     *          the destination matrix's data pointer will become invalid.
     * @note The destination matrix marks ext_buff=true to indicate it does NOT own the data
     *       buffer, preventing double-free issues. Only the original owner (ext_buff=false)
     *       is responsible for memory deallocation.
     * @note The temp pointer is NOT shared (set to nullptr) to prevent double-free issues.
     *       Each object manages its own temp buffer independently.
     */
    tiny_error_t Mat::copy_head(const Mat &src)
    {
        // Check for null pointer
        if (src.data == nullptr)
        {
            std::cerr << "[Error] copy_head: source matrix data pointer is null\n";
            return TINY_ERR_INVALID_ARG;
        }
        
        // Delete current data if it was allocated by this object
        if (!this->ext_buff && this->data != nullptr)
        {
            delete[] this->data;
            this->data = nullptr;
        }
        
        // Delete current temp if it exists (assuming it was allocated by this object)
        if (this->temp != nullptr)
        {
            delete[] this->temp;
            this->temp = nullptr;
        }
        
        // Copy all metadata from source matrix
        this->row = src.row;
        this->col = src.col;
        this->element = src.element;
        this->pad = src.pad;
        this->step = src.step;
        this->memory = src.memory;
        
        // Share data pointer regardless of source ownership
        // This allows multiple matrices to reference the same data buffer
        this->data = src.data;
        this->sub_matrix = src.sub_matrix;
        
        // Key difference: ALWAYS set ext_buff=true for destination
        // This ensures only the original owner (src.ext_buff=false) will deallocate memory
        // The destination matrix becomes a view that does NOT own the data
        // This design eliminates double-free issues:
        // - Original owner: ext_buff=false, deletes memory on destruction
        // - All views: ext_buff=true, does NOT delete memory on destruction
        this->ext_buff = true;
        
        // Do NOT share temp pointer - temp is a temporary buffer that should not be shared
        // Setting temp to nullptr prevents double-free issues when either object is destroyed
        // Each object should manage its own temp buffer if needed
        this->temp = nullptr;

        return TINY_OK;
    }

    /**
     * @name Mat::view_roi(int start_row, int start_col, int roi_rows, int roi_cols)
     * @brief Make a shallow copy of ROI matrix. Create a view of the ROI matrix. 
     *        Low level function. Unlike ESP-DSP, it is not allowed to setup step here, 
     *        step is automatically calculated inside the function.
     * @param start_row Start row position of source matrix (must be non-negative)
     * @param start_col Start column position of source matrix (must be non-negative)
     * @param roi_rows Size of row elements of source matrix to copy (must be positive)
     * @param roi_cols Size of column elements of source matrix to copy (must be positive)
     * @return result matrix size roi_rows x roi_cols, or empty matrix on error
     * @warning The returned matrix is a VIEW (shallow copy) that shares data with the source matrix.
     *          If the source matrix is destroyed, the view's data pointer will become invalid.
     * @note The step of the result matrix is inherited from the source matrix.
     */
    Mat Mat::view_roi(int start_row, int start_col, int roi_rows, int roi_cols) const
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] view_roi: source matrix data pointer is null\n";
            return Mat();  // Return empty matrix as error indicator
        }
        
        // Validate source matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] view_roi: invalid source matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return Mat();
        }
        
        // Validate position parameters (must be non-negative)
        if (start_row < 0 || start_col < 0)
        {
            std::cerr << "[Error] view_roi: invalid position: start_row=" << start_row 
                      << ", start_col=" << start_col << " (must be non-negative)\n";
            return Mat();
        }
        
        // Validate ROI size parameters (must be positive)
        if (roi_rows <= 0 || roi_cols <= 0)
        {
            std::cerr << "[Error] view_roi: invalid ROI size: roi_rows=" << roi_rows 
                      << ", roi_cols=" << roi_cols << " (must be positive)\n";
            return Mat();
        }
        
        // Check if ROI fits within source matrix boundaries
        if ((start_row + roi_rows) > this->row)
        {
            std::cerr << "[Error] view_roi: ROI exceeds row boundary: start_row=" << start_row 
                      << ", roi_rows=" << roi_rows << ", source.rows=" << this->row << "\n";
            return Mat();
        }
        if ((start_col + roi_cols) > this->col)
        {
            std::cerr << "[Error] view_roi: ROI exceeds column boundary: start_col=" << start_col 
                      << ", roi_cols=" << roi_cols << ", source.cols=" << this->col << "\n";
            return Mat();
        }
        
        // Validate step: must be >= roi_cols (padding cannot be negative)
        if (this->step < roi_cols)
        {
            std::cerr << "[Error] view_roi: step < roi_cols: step=" << this->step 
                      << ", roi_cols=" << roi_cols << "\n";
            return Mat();
        }
        
        // Check for integer overflow
        if (roi_rows > 0 && this->step > INT_MAX / roi_rows)
        {
            std::cerr << "[Error] view_roi: integer overflow: roi_rows=" << roi_rows 
                      << ", step=" << this->step << "\n";
            return Mat();
        }
        if (roi_rows > 0 && roi_cols > INT_MAX / roi_rows)
        {
            std::cerr << "[Error] view_roi: integer overflow: roi_rows=" << roi_rows 
                      << ", roi_cols=" << roi_cols << "\n";
            return Mat();
        }
        
        // Create ROI view (shallow copy)
        Mat result;
        result.row = roi_rows;
        result.col = roi_cols;
        result.step = this->step;
        result.pad = this->step - roi_cols;  // Now guaranteed to be non-negative
        result.element = roi_rows * roi_cols;
        result.memory = roi_rows * this->step;
        result.data = this->data + (start_row * this->step + start_col);
        result.temp = nullptr;
        result.ext_buff = true;
        result.sub_matrix = true;

        return result;
    }

    /**
     * @name Mat::view_roi(const Mat::ROI &roi)
     * @brief Make a shallow copy of ROI matrix. Create a view of the ROI matrix using ROI structure.
     * @param roi Rectangular area of interest (roi.pos_x, roi.pos_y must be non-negative, 
     *            roi.width, roi.height must be positive)
     * @return result matrix size roi.height x roi.width, or empty matrix on error
     * @warning The returned matrix is a VIEW (shallow copy) that shares data with the source matrix.
     *          If the source matrix is destroyed, the view's data pointer will become invalid.
     * @note This is a convenience wrapper that calls view_roi(roi.pos_y, roi.pos_x, roi.height, roi.width).
     */
    Mat Mat::view_roi(const Mat::ROI &roi) const
    {
        return view_roi(roi.pos_y, roi.pos_x, roi.height, roi.width);
    }

    /**
     * @name Mat::copy_roi(int start_row, int start_col, int height, int width)
     * @brief Make a deep copy of matrix. Compared to view_roi(), this one is a deep copy, 
     *        not sharing memory with the source matrix.
     * @param start_row Start row position of source matrix to copy (must be non-negative)
     * @param start_col Start column position of source matrix to copy (must be non-negative)
     * @param height Size of row elements of source matrix to copy (must be positive)
     * @param width Size of column elements of source matrix to copy (must be positive)
     * @return result matrix size height x width, or empty matrix on error
     * @note The returned matrix is a DEEP COPY with its own memory. It is independent 
     *       of the source matrix and can be safely used after the source is destroyed.
     */
    Mat Mat::copy_roi(int start_row, int start_col, int height, int width)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] copy_roi: source matrix data pointer is null\n";
            return Mat();  // Return empty matrix as error indicator
        }
        
        // Validate source matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] copy_roi: invalid source matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return Mat();
        }
        
        // Validate position parameters (must be non-negative)
        if (start_row < 0 || start_col < 0)
        {
            std::cerr << "[Error] copy_roi: invalid position: start_row=" << start_row 
                      << ", start_col=" << start_col << " (must be non-negative)\n";
            return Mat();
        }
        
        // Validate size parameters (must be positive)
        if (height <= 0 || width <= 0)
        {
            std::cerr << "[Error] copy_roi: invalid size: height=" << height 
                      << ", width=" << width << " (must be positive)\n";
            return Mat();
        }
        
        // Check if ROI fits within source matrix boundaries
        if ((start_row + height) > this->row)
        {
            std::cerr << "[Error] copy_roi: ROI exceeds row boundary: start_row=" << start_row 
                      << ", height=" << height << ", source.rows=" << this->row << "\n";
            return Mat();
        }
        if ((start_col + width) > this->col)
        {
            std::cerr << "[Error] copy_roi: ROI exceeds column boundary: start_col=" << start_col 
                      << ", width=" << width << ", source.cols=" << this->col << "\n";
            return Mat();
        }

        // Create result matrix (deep copy)
        Mat result(height, width);
        
        // Check if result matrix was created successfully
        if (result.data == nullptr)
        {
            std::cerr << "[Error] copy_roi: failed to allocate memory for result matrix\n";
            return Mat();
        }

        // Deep copy the data from the source matrix row by row
        // This handles different step values correctly
        for (int r = 0; r < result.row; r++)
        {
            memcpy(&result.data[r * result.step], 
                   &this->data[(r + start_row) * this->step + start_col], 
                   result.col * sizeof(float));
        }

        return result;
    }

    /**
     * @name Mat::copy_roi(const Mat::ROI &roi)
     * @brief Make a deep copy of matrix using ROI structure. Compared to view_roi(), 
     *        this one is a deep copy, not sharing memory with the source matrix.
     * @param roi Rectangular area of interest (roi.pos_x, roi.pos_y must be non-negative, 
     *            roi.width, roi.height must be positive)
     * @return result matrix size roi.height x roi.width, or empty matrix on error
     * @note The returned matrix is a DEEP COPY with its own memory. It is independent 
     *       of the source matrix and can be safely used after the source is destroyed.
     * @note This is a convenience wrapper that calls copy_roi(roi.pos_y, roi.pos_x, roi.height, roi.width).
     */
    Mat Mat::copy_roi(const Mat::ROI &roi)
    {
        return copy_roi(roi.pos_y, roi.pos_x, roi.height, roi.width);
    }

    /**
     * @name Mat::block(int start_row, int start_col, int block_rows, int block_cols)
     * @brief Get a block (submatrix) of the matrix. This is a deep copy operation.
     * @param start_row Start row position of the block (must be non-negative)
     * @param start_col Start column position of the block (must be non-negative)
     * @param block_rows Number of rows in the block (must be positive)
     * @param block_cols Number of columns in the block (must be positive)
     * @return result matrix size block_rows x block_cols, or empty matrix on error
     * @note The returned matrix is a DEEP COPY with its own memory. It is independent 
     *       of the source matrix and can be safely used after the source is destroyed.
     * @note This function is similar to copy_roi(), but uses element-by-element access.
     *       For better performance with large blocks, consider using copy_roi() instead.
     */
    Mat Mat::block(int start_row, int start_col, int block_rows, int block_cols)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] block: source matrix data pointer is null\n";
            return Mat();  // Return empty matrix as error indicator
        }
        
        // Validate source matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] block: invalid source matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return Mat();
        }
        
        // Boundary check: validate position parameters (must be non-negative)
        if (start_row < 0 || start_col < 0)
        {
            std::cerr << "[Error] block: invalid position: start_row=" << start_row 
                      << ", start_col=" << start_col << " (must be non-negative)\n";
            return Mat();
        }
        
        // Boundary check: validate block size parameters (must be positive)
        if (block_rows <= 0 || block_cols <= 0)
        {
            std::cerr << "[Error] block: invalid block size: block_rows=" << block_rows 
                      << ", block_cols=" << block_cols << " (must be positive)\n";
            return Mat();
        }
        
        // Check if block fits within source matrix boundaries
        if ((start_row + block_rows) > this->row)
        {
            std::cerr << "[Error] block: block exceeds row boundary: start_row=" << start_row 
                      << ", block_rows=" << block_rows << ", source.rows=" << this->row << "\n";
            return Mat();
        }
        if ((start_col + block_cols) > this->col)
        {
            std::cerr << "[Error] block: block exceeds column boundary: start_col=" << start_col 
                      << ", block_cols=" << block_cols << ", source.cols=" << this->col << "\n";
            return Mat();
        }
        
        // Create result matrix
        Mat result(block_rows, block_cols);
        
        // Check if result matrix was created successfully
        if (result.data == nullptr)
        {
            std::cerr << "[Error] block: failed to allocate memory for result matrix\n";
            return Mat();
        }
        
        // Copy block data element by element
        // Note: This uses operator() which handles step correctly, but is slower than memcpy
        // For better performance, consider using copy_roi() which uses memcpy
        for (int i = 0; i < block_rows; ++i)
        {
            for (int j = 0; j < block_cols; ++j)
            {
                result(i, j) = (*this)(start_row + i, start_col + j);
            }
        }
        
        return result;
    }

    /**
     * @name Mat::swap_rows(int row1, int row2)
     * @brief Swap two rows of the matrix.
     * @param row1 The index of the first row to swap (must be in range [0, row-1])
     * @param row2 The index of the second row to swap (must be in range [0, row-1])
     * @note If row1 == row2, the function returns immediately without doing anything.
     * @note This function is commonly used in matrix operations like Gaussian elimination with row pivoting.
     */
    void Mat::swap_rows(int row1, int row2)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] swap_rows: matrix data pointer is null\n";
            return;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] swap_rows: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return;
        }
        
        // Validate row indices
        if (row1 < 0 || row1 >= this->row)
        {
            std::cerr << "[Error] swap_rows: row1 index out of range: row1=" << row1 
                      << ", matrix.rows=" << this->row << "\n";
            return;
        }
        if (row2 < 0 || row2 >= this->row)
        {
            std::cerr << "[Error] swap_rows: row2 index out of range: row2=" << row2 
                      << ", matrix.rows=" << this->row << "\n";
            return;
        }
        
        // Optimization: if same row, no need to swap
        if (row1 == row2)
        {
            return;
        }
        
        // Allocate temporary buffer for row swap
        // Note: Using new/delete here is acceptable for this operation,
        // but could be optimized by using the matrix's temp buffer if available
        float *temp_row = new(std::nothrow) float[this->col];
        if (temp_row == nullptr)
        {
            std::cerr << "[Error] swap_rows: failed to allocate temporary buffer\n";
            return;
        }
        
        // Swap rows using memcpy (handles step correctly)
        memcpy(temp_row, &this->data[row1 * this->step], this->col * sizeof(float));
        memcpy(&this->data[row1 * this->step], &this->data[row2 * this->step], this->col * sizeof(float));
        memcpy(&this->data[row2 * this->step], temp_row, this->col * sizeof(float));
        
        delete[] temp_row;
    }

    /**
     * @name Mat::swap_cols(int col1, int col2)
     * @brief Swap two columns of the matrix.
     * @param col1 The index of the first column to swap (must be in range [0, col-1])
     * @param col2 The index of the second column to swap (must be in range [0, col-1])
     * @note If col1 == col2, the function returns immediately without doing anything.
     * @note Useful for column pivoting in algorithms like Gaussian elimination with column pivoting.
     * @note This function swaps columns element by element, which correctly handles step.
     */
    void Mat::swap_cols(int col1, int col2)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] swap_cols: matrix data pointer is null\n";
            return;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] swap_cols: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return;
        }
        
        // Validate column indices
        if (col1 < 0 || col1 >= this->col)
        {
            std::cerr << "[Error] swap_cols: col1 index out of range: col1=" << col1 
                      << ", matrix.cols=" << this->col << "\n";
            return;
        }
        if (col2 < 0 || col2 >= this->col)
        {
            std::cerr << "[Error] swap_cols: col2 index out of range: col2=" << col2 
                      << ", matrix.cols=" << this->col << "\n";
            return;
        }
        
        // Optimization: if same column, no need to swap
        if (col1 == col2)
        {
            return;
        }
        
        // Swap columns element by element (considering step)
        // Note: This approach correctly handles different step values
        for (int i = 0; i < this->row; ++i)
        {
            float temp = (*this)(i, col1);
            (*this)(i, col1) = (*this)(i, col2);
            (*this)(i, col2) = temp;
        }
    }

    /**
     * @name Mat::clear()
     * @brief Clear the matrix by setting all elements to zero.
     * @note Only clears the actual matrix elements (col elements per row), not the padding area.
     * @note If the matrix has padding (step > col), the padding elements are not cleared.
     */
    void Mat::clear(void)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] clear: matrix data pointer is null\n";
            return;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            // Empty matrix, nothing to clear
            return;
        }
        
        // Clear matrix row by row (handles step correctly)
        // Only clear the actual matrix elements (col elements), not the padding
        for (int row = 0; row < this->row; row++)
        {
            memset(this->data + (row * this->step), 0, this->col * sizeof(float));
        }
    }

    // ============================================================================
    // Arithmetic Operators
    // ============================================================================
    /**
     * @name Mat::operator=(const Mat &src)
     * @brief Copy assignment operator - copy the elements of the source matrix into the destination matrix.
     * @param src Source matrix to copy from
     * @return Reference to this matrix
     * @note Compared to the copy constructor, this operator is used for existing matrices.
     *       If dimensions differ, memory will be reallocated.
     * @note Assignment to sub-matrix views is not allowed.
     * @warning If memory allocation fails, the matrix may be in an invalid state.
     */
    Mat &Mat::operator=(const Mat &src)
    {
        // 1. Self-assignment check
        if (this == &src)
        {
            return *this;
        }

        // 2. Forbid assignment to sub-matrix views
        if (this->sub_matrix)
        {
            std::cerr << "[Error] Assignment to a sub-matrix is not allowed.\n";
            return *this;
        }
        
        // Check source matrix validity
        if (src.data == nullptr)
        {
            std::cerr << "[Error] operator=: source matrix data pointer is null\n";
            return *this;
        }

        // 3. If dimensions differ, reallocate memory
        if (this->row != src.row || this->col != src.col)
        {
            if (!this->ext_buff && this->data != nullptr)
            {
                delete[] this->data;
                this->data = nullptr;
            }

            // Update dimensions and memory info
            this->row = src.row;
            this->col = src.col;
            this->step = src.col; // Follow source's logical step (no padding)
            this->pad = 0;
            
            // Check for integer overflow
            if (this->row > 0 && this->col > INT_MAX / this->row)
            {
                std::cerr << "[Error] operator=: integer overflow in element calculation\n";
                this->data = nullptr;
                return *this;
            }
            this->element = this->row * this->col;
            
            if (this->row > 0 && this->step > INT_MAX / this->row)
            {
                std::cerr << "[Error] operator=: integer overflow in memory calculation\n";
                this->data = nullptr;
                return *this;
            }
            this->memory = this->row * this->step;

            this->ext_buff = false;
            this->sub_matrix = false;

            alloc_mem();
            
            // Check if memory allocation succeeded
            if (this->data == nullptr)
            {
                std::cerr << "[Error] operator=: memory allocation failed\n";
                return *this;
            }
        }
        
        // Check if this->data is valid before copying
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator=: destination matrix data pointer is null\n";
            return *this;
        }

        // 4. Data copy (row-wise, handles different strides correctly)
        for (int r = 0; r < this->row; ++r)
        {
            std::memcpy(this->data + r * this->step, 
                       src.data + r * src.step, 
                       this->col * sizeof(float));
        }

        return *this;
    }

    /**
     * @name Mat::operator+=(const Mat &A)
     * @brief Element-wise addition of another matrix to this matrix.
     * @param A The matrix to add (must have same dimensions as this matrix)
     * @return Mat& Reference to the current matrix
     * @note This function performs element-wise addition: this[i,j] += A[i,j]
     * @note The function automatically handles padding and uses optimized vectorized operations when possible.
     */
    Mat &Mat::operator+=(const Mat &A)
    {
        // Check for null pointers
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator+=: this matrix data pointer is null\n";
            return *this;
        }
        if (A.data == nullptr)
        {
            std::cerr << "[Error] operator+=: source matrix data pointer is null\n";
            return *this;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] operator+=: invalid this matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return *this;
        }
        if (A.row <= 0 || A.col <= 0)
        {
            std::cerr << "[Error] operator+=: invalid source matrix dimensions: rows=" 
                      << A.row << ", cols=" << A.col << "\n";
            return *this;
        }

        // 1. Dimension check - matrices must have same dimensions
        if ((this->row != A.row) || (this->col != A.col))
        {
            std::cerr << "[Error] Matrix addition failed: Dimension mismatch ("
                      << this->row << "x" << this->col << " vs "
                      << A.row << "x" << A.col << ")\n";
            return *this;
        }

        // 2. Determine if padding handling is needed
        bool need_padding_handling = (this->pad > 0) || (A.pad > 0);

        if (need_padding_handling)
        {
            // Padding-aware addition
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_add_f32(this->data, A.data, this->data,
                         this->row, this->col,
                         this->pad, A.pad, this->pad,
                         1, 1, 1);
#else
            tiny_mat_add_f32(this->data, A.data, this->data,
                             this->row, this->col,
                             this->pad, A.pad, this->pad,
                             1, 1, 1);
#endif
        }
        else
        {
            // Vectorized addition for contiguous memory
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dsps_add_f32(this->data, A.data, this->data, this->memory, 1, 1, 1);
#else
            tiny_vec_add_f32(this->data, A.data, this->data, this->memory, 1, 1, 1);
#endif
        }

        return *this;
    }

    /**
     * @name Mat::operator+=(float C)
     * @brief Element-wise addition of a constant to this matrix.
     * @param C The constant to add to each element
     * @return Mat& Reference to the current matrix
     * @note This function performs element-wise addition: this[i,j] += C
     * @note The function automatically handles padding and uses optimized vectorized operations when possible.
     */
    Mat &Mat::operator+=(float C)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator+=(float): matrix data pointer is null\n";
            return *this;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] operator+=(float): invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return *this;
        }

        // Check whether padding is present
        bool need_padding_handling = (this->pad > 0);

        if (need_padding_handling)
        {
            // Padding-aware constant addition
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_addc_f32(this->data, this->data, C,
                          this->row, this->col,
                          this->pad, this->pad,
                          1, 1);
#else
            tiny_mat_addc_f32(this->data, this->data, C,
                              this->row, this->col,
                              this->pad, this->pad,
                              1, 1);
#endif
        }
        else
        {
            // Vectorized constant addition
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dsps_addc_f32(this->data, this->data, this->memory, C, 1, 1);
#else
            tiny_vec_addc_f32(this->data, this->data, this->memory, C, 1, 1);
#endif
        }

        return *this;
    }

    /**
     * @name Mat::operator-=(const Mat &A)
     * @brief Element-wise subtraction of another matrix from this matrix.
     * @param A The matrix to subtract (must have same dimensions as this matrix)
     * @return Mat& Reference to the current matrix
     * @note This function performs element-wise subtraction: this[i,j] -= A[i,j]
     * @note The function automatically handles padding and uses optimized vectorized operations when possible.
     */
    Mat &Mat::operator-=(const Mat &A)
    {
        // Check for null pointers
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator-=: this matrix data pointer is null\n";
            return *this;
        }
        if (A.data == nullptr)
        {
            std::cerr << "[Error] operator-=: source matrix data pointer is null\n";
            return *this;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] operator-=: invalid this matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return *this;
        }
        if (A.row <= 0 || A.col <= 0)
        {
            std::cerr << "[Error] operator-=: invalid source matrix dimensions: rows=" 
                      << A.row << ", cols=" << A.col << "\n";
            return *this;
        }

        // 1. Dimension check - matrices must have same dimensions
        if ((this->row != A.row) || (this->col != A.col))
        {
            std::cerr << "[Error] Matrix subtraction failed: Dimension mismatch ("
                      << this->row << "x" << this->col << " vs "
                      << A.row << "x" << A.col << ")\n";
            return *this;
        }

        // 2. Determine if padding handling is needed
        bool need_padding_handling = (this->pad > 0) || (A.pad > 0);

        if (need_padding_handling)
        {
            // Padding-aware subtraction
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_sub_f32(this->data, A.data, this->data,
                         this->row, this->col,
                         this->pad, A.pad, this->pad,
                         1, 1, 1);
#else
            tiny_mat_sub_f32(this->data, A.data, this->data,
                             this->row, this->col,
                             this->pad, A.pad, this->pad,
                             1, 1, 1);
#endif
        }
        else
        {
            // Vectorized subtraction for contiguous memory
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dsps_sub_f32(this->data, A.data, this->data, this->memory, 1, 1, 1);
#else
            tiny_vec_sub_f32(this->data, A.data, this->data, this->memory, 1, 1, 1);
#endif
        }

        return *this;
    }

    /**
     * @name Mat::operator-=(float C)
     * @brief Element-wise subtraction of a constant from this matrix.
     * @param C The constant to subtract from each element
     * @return Mat& Reference to the current matrix
     * @note This function performs element-wise subtraction: this[i,j] -= C
     * @note The function automatically handles padding and uses optimized vectorized operations when possible.
     * @note On ESP32, this uses addc with -C since subc is not available in DSP library.
     */
    Mat &Mat::operator-=(float C)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator-=(float): matrix data pointer is null\n";
            return *this;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] operator-=(float): invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return *this;
        }

        bool need_padding_handling = (this->pad > 0);

        if (need_padding_handling)
        {
            // Padding-aware constant subtraction
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            // Note: ESP32 DSP does not provide dspm_subc_f32, using dspm_addc_f32 with -C
            dspm_addc_f32(this->data, this->data, -C,
                          this->row, this->col,
                          this->pad, this->pad,
                          1, 1);
#else
            tiny_mat_subc_f32(this->data, this->data, C,
                              this->row, this->col,
                              this->pad, this->pad,
                              1, 1);
#endif
        }
        else
        {
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            // Note: ESP32 DSP does not provide dsps_subc_f32, using dsps_addc_f32 with -C
            dsps_addc_f32(this->data, this->data, this->memory, -C, 1, 1);
#else
            tiny_vec_subc_f32(this->data, this->data, this->memory, C, 1, 1);
#endif
        }

        return *this;
    }

    /**
     * @name Mat::operator*=(const Mat &m)
     * @brief Matrix multiplication: this = this * m
     * @param m The matrix to multiply with (must have compatible dimensions: this.col == m.row)
     * @return Mat& Reference to the current matrix
     * @note Matrix multiplication requires: this.col == m.row
     *       Result dimensions: this.row x m.col
     * @note This function creates a temporary copy to avoid overwriting data during computation.
     */
    Mat &Mat::operator*=(const Mat &m)
    {
        // Check for null pointers
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator*=: this matrix data pointer is null\n";
            return *this;
        }
        if (m.data == nullptr)
        {
            std::cerr << "[Error] operator*=: source matrix data pointer is null\n";
            return *this;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] operator*=: invalid this matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return *this;
        }
        if (m.row <= 0 || m.col <= 0)
        {
            std::cerr << "[Error] operator*=: invalid source matrix dimensions: rows=" 
                      << m.row << ", cols=" << m.col << "\n";
            return *this;
        }

        // 1. Dimension check - matrix multiplication requires: this.col == m.row
        if (this->col != m.row)
        {
            std::cerr << "[Error] Matrix multiplication failed: incompatible dimensions ("
                      << this->row << "x" << this->col << " * "
                      << m.row << "x" << m.col << ")\n";
            return *this;
        }

        // 1.5. In-place matrix multiplication may change output width (this->col -> m.col).
        // For views/external buffers, resizing is unsafe because ownership/capacity is unknown.
        if ((this->sub_matrix || this->ext_buff) && (this->col != m.col))
        {
            std::cerr << "[Error] operator*=: cannot reshape sub-matrix/external-buffer matrix ("
                      << this->row << "x" << this->col << " -> "
                      << this->row << "x" << m.col << ")\n";
            return *this;
        }

        // 2. Prepare temp matrix (in case overwriting the original data)
        // Create a copy of this matrix to avoid overwriting during computation
        Mat temp = this->copy_roi(0, 0, this->row, this->col);
        
        // Check if copy_roi succeeded
        if (temp.data == nullptr)
        {
            std::cerr << "[Error] operator*=: failed to create temporary matrix copy\n";
            return *this;
        }

        // 2.5. Ensure destination buffer/metadata can hold the multiplication result.
        // Result shape: this->row x m.col
        const int result_col = m.col;
        if (this->col != result_col)
        {
            // We only reach here when this matrix owns its buffer (checked above).
            if (!this->ext_buff && this->data != nullptr)
            {
                delete[] this->data;
                this->data = nullptr;
            }

            this->col = result_col;
            this->step = result_col;  // keep destination contiguous after reshape
            this->pad = 0;

            if (this->row > 0 && this->col > INT_MAX / this->row)
            {
                std::cerr << "[Error] operator*=: integer overflow in element calculation\n";
                this->data = nullptr;
                this->element = 0;
                this->memory = 0;
                return *this;
            }
            this->element = this->row * this->col;

            if (this->row > 0 && this->step > INT_MAX / this->row)
            {
                std::cerr << "[Error] operator*=: integer overflow in memory calculation\n";
                this->data = nullptr;
                this->memory = 0;
                return *this;
            }
            this->memory = this->row * this->step;

            alloc_mem();
            if (this->data == nullptr)
            {
                std::cerr << "[Error] operator*=: memory allocation failed for result matrix\n";
                return *this;
            }
        }

        // 3. Check whether padding is present in either matrix
        bool need_padding_handling = (this->pad > 0) || (m.pad > 0);

        if (need_padding_handling)
        {
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_mult_ex_f32(temp.data, m.data, this->data, temp.row, temp.col, m.col, temp.pad, m.pad, this->pad);
#else
            tiny_mat_mult_ex_f32(temp.data, m.data, this->data, temp.row, temp.col, m.col, temp.pad, m.pad, this->pad);
#endif
        }
        else
        {
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_mult_f32(temp.data, m.data, this->data, temp.row, temp.col, m.col);
#else
            tiny_mat_mult_f32(temp.data, m.data, this->data, temp.row, temp.col, m.col);
#endif
        }

        return *this;
    }

    /**
     * @name Mat::operator*=(float num)
     * @brief Element-wise multiplication by a constant.
     * @param num The constant multiplier
     * @return Mat& Reference to the current matrix
     * @note This function performs element-wise multiplication: this[i,j] *= num
     * @note The function automatically handles padding and uses optimized vectorized operations when possible.
     */
    Mat &Mat::operator*=(float num)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator*=(float): matrix data pointer is null\n";
            return *this;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] operator*=(float): invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return *this;
        }

        // Check whether padding is present
        bool need_padding_handling = (this->pad > 0);

        if (need_padding_handling)
        {
            // Padding-aware multiplication
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_mulc_f32(this->data, this->data, num,
                          this->row, this->col,
                          this->pad, this->pad,
                          1, 1);
#else
            tiny_mat_multc_f32(this->data, this->data, num, this->row, this->col, this->pad, this->pad, 1, 1);
#endif
        }
        else
        {
            // No padding, use vectorized multiplication
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dsps_mulc_f32(this->data, this->data, this->memory, num, 1, 1);
#else
            tiny_vec_mulc_f32(this->data, this->data, this->memory, num, 1, 1);
#endif
        }

        return *this;
    }

    /**
     * @name Mat::operator/=(const Mat &B)
     * @brief Element-wise division: this = this / B
     * @param B The matrix divisor (must have same dimensions as this matrix, and no zero elements)
     * @return Mat& Reference to the current matrix
     * @note This function performs element-wise division: this[i,j] /= B[i,j]
     * @warning Division by zero will cause an error. All elements of B must be non-zero.
     */
    Mat &Mat::operator/=(const Mat &B)
    {
        // Check for null pointers
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator/=: this matrix data pointer is null\n";
            return *this;
        }
        if (B.data == nullptr)
        {
            std::cerr << "[Error] operator/=: divisor matrix data pointer is null\n";
            return *this;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] operator/=: invalid this matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return *this;
        }
        if (B.row <= 0 || B.col <= 0)
        {
            std::cerr << "[Error] operator/=: invalid divisor matrix dimensions: rows=" 
                      << B.row << ", cols=" << B.col << "\n";
            return *this;
        }

        // 1. Dimension check - matrices must have same dimensions
        if ((this->row != B.row) || (this->col != B.col))
        {
            std::cerr << "[Error] Matrix division failed: Dimension mismatch ("
                      << this->row << "x" << this->col << " vs "
                      << B.row << "x" << B.col << ")\n";
            return *this;
        }

        // 2. Zero division check - scan for near-zero elements
        bool zero_found = false;
        const float epsilon = 1e-9f;
        for (int i = 0; i < B.row; ++i)
        {
            for (int j = 0; j < B.col; ++j)
            {
                if (fabs(B(i, j)) < epsilon)
                {
                    zero_found = true;
                    std::cerr << "[Error] Matrix division failed: Division by zero detected at position ("
                              << i << ", " << j << ")\n";
                    break;
                }
            }
            if (zero_found)
                break;
        }

        if (zero_found)
        {
            return *this;
        }

        // 3. Element-wise division
        for (int i = 0; i < this->row; ++i)
        {
            for (int j = 0; j < this->col; ++j)
            {
                (*this)(i, j) /= B(i, j);
            }
        }

        return *this;
    }

    /**
     * @name Mat::operator/=(float num)
     * @brief Element-wise division of this matrix by a constant.
     * @param num The constant divisor (must be non-zero)
     * @return Mat& Reference to the current matrix
     * @note This function performs element-wise division: this[i,j] /= num
     * @note The function uses multiplication by 1/num for efficiency (division is slower than multiplication).
     * @note The function automatically handles padding and uses optimized vectorized operations when possible.
     * @warning Division by zero will cause an error.
     */
    Mat &Mat::operator/=(float num)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator/=(float): matrix data pointer is null\n";
            return *this;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] operator/=(float): invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return *this;
        }

        // 1. Check division by zero
        const float epsilon = 1e-9f;
        if (fabs(num) < epsilon)
        {
            std::cerr << "[Error] Matrix division by zero is undefined (divisor=" << num << ")\n";
            return *this;
        }

        // 2. Determine if padding handling is needed
        bool need_padding_handling = (this->pad > 0);

        // Use multiplication by inverse for better performance (division is slower)
        float inv_num = 1.0f / num;

        if (need_padding_handling)
        {
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_mulc_f32(this->data, this->data, inv_num,
                          this->row, this->col,
                          this->pad, this->pad,
                          1, 1);
#else
            tiny_mat_multc_f32(this->data, this->data, inv_num,
                              this->row, this->col,
                              this->pad, this->pad,
                              1, 1);
#endif
        }
        else
        {
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dsps_mulc_f32(this->data, this->data, this->memory, inv_num, 1, 1);
#else
            tiny_vec_mulc_f32(this->data, this->data, this->memory, inv_num, 1, 1);
#endif
        }

        return *this;
    }

    /**
     * @name Mat::operator^(int num)
     * @brief Element-wise integer exponentiation. Returns a new matrix where each element is raised to the given power.
     * @param num The exponent (integer, can be positive, negative, or zero)
     * @return Mat New matrix after exponentiation
     * @note This function performs element-wise exponentiation: result[i,j] = this[i,j]^num
     * @note For num=0, all elements become 1.0. For num=1, returns a copy of the matrix.
     * @note For num=-1, returns element-wise reciprocal: result[i,j] = 1.0 / this[i,j]
     * @note For num < 0, performs: result[i,j] = 1.0 / (this[i,j]^(-num))
     * @warning For negative exponents, elements that are zero or too close to zero will result in
     *          infinity or NaN. The function will print warnings for such cases.
     */
    Mat Mat::operator^(int num)
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] operator^: matrix data pointer is null\n";
            return Mat();  // Return empty matrix as error indicator
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] operator^: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return Mat();
        }

        // Handle special cases
        if (num == 0)
        {
            // Any number to the power of 0 is 1
            Mat result(this->row, this->col, this->step);
            if (result.data == nullptr)
            {
                std::cerr << "[Error] operator^: failed to allocate memory for result matrix\n";
                return Mat();
            }
            for (int i = 0; i < this->row; ++i)
            {
                for (int j = 0; j < this->col; ++j)
                {
                    result(i, j) = 1.0f;
                }
            }
            return result;
        }

        if (num == 1)
        {
            // Return a copy of current matrix
            return Mat(*this);
        }

        // Handle negative exponents
        if (num < 0)
        {
            // For negative exponent: A^(-n) = 1 / A^n (element-wise)
            // Compute positive power first, then take reciprocal
            int abs_num = -num;  // Positive exponent value
            
            Mat result(this->row, this->col, this->step);
            if (result.data == nullptr)
            {
                std::cerr << "[Error] operator^: failed to allocate memory for result matrix\n";
                return Mat();
            }
            
            // Element-wise exponentiation with negative exponent
            // result[i,j] = 1.0 / (this[i,j]^abs_num)
            for (int i = 0; i < this->row; ++i)
            {
                for (int j = 0; j < this->col; ++j)
                {
                    float base = (*this)(i, j);
                    
                    // Check for zero or near-zero base (for negative exponents)
                    if (fabsf(base) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                    {
                        std::cerr << "[Warning] operator^: element at (" << i << ", " << j 
                                  << ") is zero or too small (" << base 
                                  << "), cannot compute negative power. Result will be Inf or NaN.\n";
                        result(i, j) = (base == 0.0f) ? 
                                      (std::numeric_limits<float>::infinity()) : 
                                      (std::numeric_limits<float>::quiet_NaN());
                        continue;
                    }
                    
                    // Compute base^abs_num (positive power)
                    float value = 1.0f;
                    for (int k = 0; k < abs_num; ++k)
                    {
                        value *= base;
                    }
                    
                    // Take reciprocal: 1 / value
                    if (fabsf(value) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                    {
                        std::cerr << "[Warning] operator^: computed power value at (" << i << ", " << j 
                                  << ") is too small (" << value 
                                  << "), reciprocal may be invalid.\n";
                        result(i, j) = std::numeric_limits<float>::infinity();
                    }
                    else
                    {
                        result(i, j) = 1.0f / value;
                    }
                }
            }
            
            return result;
        }

        // General case: positive exponent > 1
        Mat result(this->row, this->col, this->step);
        if (result.data == nullptr)
        {
            std::cerr << "[Error] operator^: failed to allocate memory for result matrix\n";
            return Mat();
        }
        
        // Element-wise exponentiation using iterative multiplication
        // Note: For large exponents, this could be optimized using fast exponentiation,
        // but for typical use cases (small exponents), this is acceptable
        for (int i = 0; i < this->row; ++i)
        {
            for (int j = 0; j < this->col; ++j)
            {
                float base = (*this)(i, j);
                float value = 1.0f;
                for (int k = 0; k < num; ++k)
                {
                    value *= base;
                }
                result(i, j) = value;
            }
        }

        return result;
    }

    // ============================================================================
    // Linear Algebra - Basic Operations
    // ============================================================================
    /**
     * @name Mat::transpose()
     * @brief Transpose the matrix. Returns a new matrix with rows and columns swapped.
     * @return Transposed matrix (col x row), or empty matrix on error
     * @note If this matrix is m x n, the result will be n x m.
     * @note The transpose operation: result[j][i] = this[i][j]
     */
    Mat Mat::transpose()
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] transpose: matrix data pointer is null\n";
            return Mat();  // Return empty matrix as error indicator
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] transpose: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return Mat();
        }

        // Create result matrix with swapped dimensions
        Mat result(this->col, this->row);
        
        // Check if result matrix was created successfully
        if (result.data == nullptr)
        {
            std::cerr << "[Error] transpose: failed to allocate memory for result matrix\n";
            return Mat();
        }
        
        // Transpose: swap rows and columns
        for (int i = 0; i < this->row; ++i)
        {
            for (int j = 0; j < this->col; ++j)
            {
                result(j, i) = this->data[i * this->step + j];
            }
        }
        
        return result;
    }

    /**
     * @name Mat::determinant()
     * @brief Compute the determinant of the matrix (auto-selects method based on size).
     * @return float The determinant value, or 0.0f on error
     * @note For small matrices (n <= 4), uses Laplace expansion (more accurate).
     *       For larger matrices, uses LU decomposition (O(n³)) for better efficiency.
     * @note Determinant can be 0.0f for singular matrices, which is a valid result.
     *       Use error checking to distinguish between error and zero determinant.
     */
    float Mat::determinant()
    {
        // Special case: 0×0 matrix (empty matrix)
        // By mathematical convention, det(empty matrix) = 1 (similar to empty product)
        if (this->row == 0 && this->col == 0)
        {
            return 1.0f;
        }
        
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] determinant: matrix data pointer is null\n";
            return 0.0f;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] determinant: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return 0.0f;
        }

        // Check if matrix is square
        if (this->row != this->col)
        {
            std::cerr << "[Error] Determinant requires a square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            return 0.0f;
        }

        int n = this->row;
        
        // For small matrices, use Laplace expansion (more accurate for small sizes)
        if (n <= 4)
        {
            return this->determinant_laplace();
        }
        
        // For larger matrices, use LU decomposition (much faster, O(n³) vs O(n!))
        return this->determinant_lu();
    }

    /**
     * @name Mat::determinant_laplace()
     * @brief Compute the determinant using Laplace expansion (cofactor expansion).
     * @return float The determinant value, or 0.0f on error
     * @note Time complexity: O(n!) - suitable only for small matrices (n <= 4).
     *       Uses recursive method with first row expansion.
     * @note For n=1 and n=2, uses direct formulas for efficiency.
     * @note Determinant can be 0.0f for singular matrices, which is a valid result.
     */
    float Mat::determinant_laplace()
    {
        // Special case: 0×0 matrix (empty matrix)
        // By mathematical convention, det(empty matrix) = 1 (similar to empty product)
        if (this->row == 0 && this->col == 0)
        {
            return 1.0f;
        }
        
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] determinant_laplace: matrix data pointer is null\n";
            return 0.0f;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] determinant_laplace: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return 0.0f;
        }

        // Check if matrix is square
        if (this->row != this->col)
        {
            std::cerr << "[Error] Determinant requires a square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            return 0.0f;
        }

        int n = this->row;
        
        // Base case: 1x1 matrix
        if (n == 1)
        {
            return this->data[0];
        }
        
        // Base case: 2x2 matrix (direct formula)
        if (n == 2)
        {
            return this->data[0] * this->data[this->step + 1] - 
                   this->data[1] * this->data[this->step];
        }

        // Recursive case: use Laplace expansion along first row
        float det = 0.0f;
        for (int j = 0; j < n; ++j)
        {
            Mat minor_mat = this->minor(0, j);
            
            // Check if minor matrix was created successfully
            if (minor_mat.data == nullptr)
            {
                std::cerr << "[Error] determinant_laplace: failed to create minor matrix at (0, " << j << ")\n";
                return 0.0f;
            }
            
            // Compute cofactor: (-1)^(i+j) * det(minor)
            float cofactor_val = ((j % 2 == 0) ? 1.0f : -1.0f) * minor_mat.determinant_laplace();
            det += this->data[j] * cofactor_val;
        }
        
        return det;
    }

    /**
     * @name Mat::determinant_lu()
     * @brief Compute the determinant using LU decomposition.
     * @return float The determinant value, or 0.0f on error
     * @note Time complexity: O(n³) - efficient for large matrices.
     *       Formula: det(A) = det(P) * det(L) * det(U) = det(P) * 1 * (product of U diagonal)
     *       where det(P) = (-1)^(number of row swaps)
     * @note Uses pivoting for numerical stability.
     * @note Determinant can be 0.0f for singular matrices, which is a valid result.
     */
    float Mat::determinant_lu()
    {
        // Special case: 0×0 matrix (empty matrix)
        // By mathematical convention, det(empty matrix) = 1 (similar to empty product)
        if (this->row == 0 && this->col == 0)
        {
            return 1.0f;
        }
        
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] determinant_lu: matrix data pointer is null\n";
            return 0.0f;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] determinant_lu: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return 0.0f;
        }

        // Check if matrix is square
        if (this->row != this->col)
        {
            std::cerr << "[Error] Determinant requires a square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            return 0.0f;
        }

        // Perform LU decomposition with pivoting for numerical stability
        LUDecomposition lu = this->lu_decompose(true);
        
        if (lu.status != TINY_OK)
        {
            // Matrix is singular or near-singular, or decomposition failed
            std::cerr << "[Warning] determinant_lu: LU decomposition failed (status=" 
                      << lu.status << "), matrix may be singular\n";
            return 0.0f;
        }
        
        // Check if decomposition matrices are valid
        if (lu.U.data == nullptr)
        {
            std::cerr << "[Error] determinant_lu: LU decomposition U matrix is null\n";
            return 0.0f;
        }

        int n = this->row;
        
        // Compute det(P): permutation matrix determinant = (-1)^(permutation signature)
        float det_P = 1.0f;
        if (lu.pivoted && lu.P.data != nullptr)
        {
            // Compute permutation signature by finding cycles in P
            // P is a permutation matrix, so each row/column has exactly one 1
            // We can compute the sign by decomposing into transpositions
            std::vector<bool> visited(n, false);
            int cycle_count = 0;
            
            for (int i = 0; i < n; ++i)
            {
                if (visited[i]) continue;
                
                // Find the cycle starting at i
                int current = i;
                int cycle_length = 0;
                bool found_mapping = false;
                while (!visited[current])
                {
                    visited[current] = true;
                    cycle_length++;
                    found_mapping = false;
                    
                    // Find where P maps current row
                    for (int j = 0; j < n; ++j)
                    {
                        if (fabsf(lu.P(current, j) - 1.0f) < 1e-6f)
                        {
                            current = j;
                            found_mapping = true;
                            break;
                        }
                    }
                    
                    // Safety check: if no mapping found, break to avoid infinite loop
                    // This should not happen for a valid permutation matrix
                    if (!found_mapping)
                    {
                        std::cerr << "[Warning] determinant_lu: Could not find mapping for row " 
                                  << current << " in permutation matrix P. Matrix may be invalid.\n";
                        break;
                    }
                }
                
                // A cycle of length k contributes (k-1) transpositions
                if (cycle_length > 1)
                {
                    cycle_count += (cycle_length - 1);
                }
            }
            
            // det(P) = (-1)^(number of transpositions)
            det_P = (cycle_count % 2 == 0) ? 1.0f : -1.0f;
        }

        // Compute det(L): lower triangular with unit diagonal = 1
        float det_L = 1.0f;  // L has unit diagonal, so det(L) = 1

        // Compute det(U): product of diagonal elements
        float det_U = 1.0f;
        for (int i = 0; i < n; ++i)
        {
            det_U *= lu.U(i, i);
        }

        // det(A) = det(P) * det(L) * det(U)
        return det_P * det_L * det_U;
    }

    /**
     * @name Mat::determinant_gaussian()
     * @brief Compute the determinant using Gaussian elimination.
     * @return float The determinant value, or 0.0f on error
     * @note Time complexity: O(n³) - efficient for large matrices.
     *       Converts matrix to upper triangular form, then multiplies diagonal elements.
     *       Tracks row swaps to account for sign changes.
     * @note Uses partial pivoting for numerical stability.
     * @note Determinant can be 0.0f for singular matrices, which is a valid result.
     */
    float Mat::determinant_gaussian()
    {
        // Special case: 0×0 matrix (empty matrix)
        // By mathematical convention, det(empty matrix) = 1 (similar to empty product)
        if (this->row == 0 && this->col == 0)
        {
            return 1.0f;
        }
        
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] determinant_gaussian: matrix data pointer is null\n";
            return 0.0f;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] determinant_gaussian: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return 0.0f;
        }

        // Check if matrix is square
        if (this->row != this->col)
        {
            std::cerr << "[Error] Determinant requires a square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            return 0.0f;
        }

        int n = this->row;
        Mat A = Mat(*this);  // Working copy
        
        // Check if copy was successful
        if (A.data == nullptr)
        {
            std::cerr << "[Error] determinant_gaussian: failed to create working copy\n";
            return 0.0f;
        }
        
        int swap_count = 0;  // Track number of row swaps

        // Gaussian elimination to upper triangular form
        for (int k = 0; k < n - 1; ++k)
        {
            // Partial pivoting: find row with largest element in column k
            int max_row = k;
            float max_val = fabsf(A(k, k));
            for (int i = k + 1; i < n; ++i)
            {
                if (fabsf(A(i, k)) > max_val)
                {
                    max_val = fabsf(A(i, k));
                    max_row = i;
                }
            }

            // Swap rows if necessary
            if (max_row != k)
            {
                A.swap_rows(k, max_row);
                swap_count++;
            }

            // Check for singular matrix (near-zero pivot)
            if (fabsf(A(k, k)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                // Matrix is singular or near-singular
                return 0.0f;
            }

            // Eliminate below diagonal
            for (int i = k + 1; i < n; ++i)
            {
                float factor = A(i, k) / A(k, k);
                for (int j = k; j < n; ++j)
                {
                    A(i, j) -= factor * A(k, j);
                }
            }
        }

        // Compute determinant: product of diagonal elements
        float det = 1.0f;
        for (int i = 0; i < n; ++i)
        {
            det *= A(i, i);
        }

        // Account for row swaps: each swap multiplies determinant by -1
        if (swap_count % 2 == 1)
        {
            det = -det;
        }

        return det;
    }

    /**
     * @name Mat::adjoint()
     * @brief Compute the adjoint (adjugate) matrix.
     * @note The adjoint matrix is the transpose of the cofactor matrix.
     *       adj(A)_ij = (-1)^(i+j) * det(minor_ji)
     *       Note: The result is stored at (j,i) to achieve transpose.
     * @return Mat The adjoint matrix, or empty Mat() on error
     * @note Time complexity: O(n² * O(det)) - expensive for large matrices.
     *       For n×n matrix, requires computing n² determinants of (n-1)×(n-1) matrices.
     */
    Mat Mat::adjoint()
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] adjoint: matrix data pointer is null\n";
            return Mat();
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] adjoint: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return Mat();
        }
        
        // Check if matrix is square
        if (this->row != this->col)
        {
            std::cerr << "[Error] Adjoint requires a square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            return Mat();
        }

        int n = this->row;
        Mat result(n, n);
        
        // Check if result matrix was created successfully
        if (result.data == nullptr)
        {
            std::cerr << "[Error] adjoint: failed to create result matrix\n";
            return Mat();
        }

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                Mat cofactor_mat = this->cofactor(i, j);
                
                // For 1×1 matrix, cofactor is a 0×0 empty matrix (valid result)
                // The determinant of a 0×0 matrix is 1.0 by mathematical convention
                // So cofactor_mat.data == nullptr is acceptable for this case
                
                // Compute cofactor value: (-1)^(i+j) * det(minor)
                float sign = ((i + j) % 2 == 0) ? 1.0f : -1.0f;
                float cofactor_val = sign * cofactor_mat.determinant();
                
                // Store at (j,i) to achieve transpose: adj(A) = C^T
                result(j, i) = cofactor_val;
            }
        }

        return result;
    }

    /**
     * @name Mat::normalize()
     * @brief Normalize the matrix by dividing each element by the matrix norm (Frobenius norm).
     * @note Normalization: M_normalized = M / ||M||_F
     *       where ||M||_F = sqrt(sum of squares of all elements)
     * @note If the matrix norm is zero or too small, normalization is skipped
     *       and a warning is printed. The matrix remains unchanged.
     * @note This function modifies the matrix in-place.
     */
    void Mat::normalize()
    {
        // Validate matrix dimensions
        if (this->row < 0 || this->col < 0)
        {
            std::cerr << "[Error] normalize: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return;
        }
        
        // Check for empty matrix
        if (this->row == 0 || this->col == 0)
        {
            // Empty matrix, nothing to normalize
            return;
        }

        // For non-empty matrices, data pointer must be valid
        if (this->data == nullptr)
        {
            std::cerr << "[Error] normalize: matrix data pointer is null\n";
            return;
        }
        
        float n = this->norm();
        
        // Check for invalid norm (NaN, Inf, or too small)
        if (!(n > TINY_MATH_MIN_POSITIVE_INPUT_F32))
        {
            if (n == 0.0f)
            {
                std::cerr << "[Warning] normalize: matrix norm is zero (matrix is all zeros), "
                          << "normalization skipped\n";
            }
            else if (std::isnan(n) || std::isinf(n))
            {
                std::cerr << "[Error] normalize: matrix norm is invalid (NaN or Inf), "
                          << "normalization skipped\n";
            }
            else
            {
                std::cerr << "[Warning] normalize: matrix norm is too small (" << n 
                          << "), normalization skipped\n";
            }
            return;
        }
        
        // Normalize each element
        for (int i = 0; i < this->row; ++i)
        {
            for (int j = 0; j < this->col; ++j)
            {
                (*this)(i, j) /= n;
            }
        }
    }

    /**
     * @name Mat::norm()
     * @brief Compute the Frobenius norm (Euclidean norm) of the matrix.
     * @note Frobenius norm: ||M||_F = sqrt(Σ M_ij²)
     *       This is the square root of the sum of squares of all matrix elements.
     * @note For empty matrices, returns 0.0f.
     * @note For zero matrices, returns 0.0f.
     * 
     * @return float The Frobenius norm of the matrix, or 0.0f on error
     */
    float Mat::norm() const
    {
        // Validate matrix dimensions
        if (this->row < 0 || this->col < 0)
        {
            std::cerr << "[Error] norm: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return 0.0f;
        }
        
        // Handle empty matrix
        if (this->row == 0 || this->col == 0)
        {
            return 0.0f;
        }

        // For non-empty matrices, data pointer must be valid
        if (this->data == nullptr)
        {
            std::cerr << "[Error] norm: matrix data pointer is null\n";
            return 0.0f;
        }
        
        float sum_sq = 0.0f;
        for (int i = 0; i < this->row; ++i)
        {
            for (int j = 0; j < this->col; ++j)
            {
                float val = (*this)(i, j);
                sum_sq += val * val;
            }
        }
        
        // Compute square root
        // Note: sum_sq should always be non-negative (sum of squares)
        float result = sqrtf(sum_sq);
        
        // Safety check: if result is invalid, return 0.0f
        if (std::isnan(result) || std::isinf(result))
        {
            std::cerr << "[Warning] norm: computed norm is invalid (NaN or Inf), returning 0.0f\n";
            return 0.0f;
        }
        
        return result;
    }

    /**
     * @name Mat::inverse_adjoint()
     * @brief Compute the inverse matrix using the adjoint method.
     * @note Formula: A^(-1) = (1 / det(A)) * adj(A)
     *       where adj(A) is the adjoint (transpose of cofactor matrix).
     * @note WARNING: This method is SLOW for large matrices!
     *       - Time complexity: O(n² × n!) - exponential growth with matrix size
     *       - For n×n matrix, requires computing n² determinants of (n-1)×(n-1) matrices
     *       - Each determinant calculation uses Laplace expansion (O(n!) complexity)
     *       - Example: 4×4 matrix needs 16 determinants of 3×3 matrices
     *       - Example: 5×5 matrix needs 25 determinants of 4×4 matrices (very slow!)
     * @note Performance comparison:
     *       - 2×2 matrix: Fast (direct formula)
     *       - 3×3 matrix: Acceptable
     *       - 4×4 matrix: Slow but usable
     *       - 5×5+ matrix: VERY SLOW - use other methods instead
     * @note For larger matrices (n >= 4), strongly recommend using:
     *       - inverse_gje() (Gauss-Jordan elimination, O(n³))
     *       - LU decomposition methods (O(n³), more stable)
     * @note This method is mainly useful for:
     *       - Small matrices (n <= 3) where simplicity is preferred
     *       - Educational purposes to understand the adjoint formula
     *       - Cases where you need the adjoint matrix anyway
     * 
     * @return Mat The inverse matrix, or empty Mat() on error
     */
    Mat Mat::inverse_adjoint()
    {
        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] inverse_adjoint: matrix data pointer is null\n";
            return Mat();
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] inverse_adjoint: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            return Mat();
        }
        
        // Check if matrix is square
        if (this->row != this->col)
        {
            std::cerr << "[Error] inverse_adjoint: requires square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            return Mat();
        }
        
        // Compute determinant
        float det = this->determinant();
        
        // Check if determinant is valid
        if (std::isnan(det) || std::isinf(det))
        {
            std::cerr << "[Error] inverse_adjoint: determinant is invalid (NaN or Inf)\n";
            return Mat();
        }
        
        // Check if matrix is singular (determinant is zero or too small)
        if (fabsf(det) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
        {
            std::cerr << "[Error] inverse_adjoint: matrix is singular (det=" << det 
                      << "), cannot compute inverse\n";
            return Mat();
        }

        // Compute adjoint matrix
        Mat adj = this->adjoint();
        
        // Check if adjoint computation was successful
        if (adj.data == nullptr)
        {
            std::cerr << "[Error] inverse_adjoint: failed to compute adjoint matrix\n";
            return Mat();
        }
        
        // Check if adjoint matrix has correct dimensions
        if (adj.row != this->row || adj.col != this->col)
        {
            std::cerr << "[Error] inverse_adjoint: adjoint matrix has incorrect dimensions: " 
                      << adj.row << "x" << adj.col << " (expected " << this->row << "x" << this->col << ")\n";
            return Mat();
        }
        
        // Compute inverse: A^(-1) = (1 / det(A)) * adj(A)
        float inv_det = 1.0f / det;
        Mat result = adj * inv_det;
        
        // Check if matrix multiplication was successful
        if (result.data == nullptr)
        {
            std::cerr << "[Error] inverse_adjoint: failed to compute inverse matrix (multiplication failed)\n";
            return Mat();
        }
        
        return result;
    }

    /**
     * @name Mat::dotprod(const Mat &A, const Mat &B)
     * @brief Compute the dot product (Frobenius inner product) of two matrices.
     * @note Mathematical definition: A · B = Σ A_ij * B_ij (sum over all elements)
     *       This is the element-wise multiplication followed by summation.
     * @note Also known as:
     *       - Frobenius inner product
     *       - Hadamard product sum
     *       - Element-wise dot product
     * @note For vectors, this is equivalent to the standard dot product.
     * @note For empty matrices, returns 0.0f.
     *
     * @param A First matrix
     * @param B Second matrix
     * @return float Dot product value, or 0.0f on error
     */
    float Mat::dotprod(const Mat &A, const Mat &B)
    {
        // Check for null pointers
        if (A.data == nullptr)
        {
            std::cerr << "[Error] dotprod: matrix A data pointer is null\n";
            return 0.0f;
        }
        
        if (B.data == nullptr)
        {
            std::cerr << "[Error] dotprod: matrix B data pointer is null\n";
            return 0.0f;
        }
        
        // Validate matrix dimensions
        if (A.row <= 0 || A.col <= 0)
        {
            std::cerr << "[Error] dotprod: invalid matrix A dimensions: rows=" 
                      << A.row << ", cols=" << A.col << "\n";
            return 0.0f;
        }
        
        if (B.row <= 0 || B.col <= 0)
        {
            std::cerr << "[Error] dotprod: invalid matrix B dimensions: rows=" 
                      << B.row << ", cols=" << B.col << "\n";
            return 0.0f;
        }
        
        // Check if matrices have the same size
        if (A.row != B.row || A.col != B.col)
        {
            std::cerr << "[Error] dotprod: matrices must have the same size (A: " 
                      << A.row << "x" << A.col << ", B: " << B.row << "x" << B.col << ")\n";
            return 0.0f;
        }
        
        // Handle empty matrices
        if (A.row == 0 || A.col == 0)
        {
            return 0.0f;
        }

        // Compute dot product: sum of element-wise products
        float result = 0.0f;
        for (int i = 0; i < A.row; ++i)
        {
            for (int j = 0; j < A.col; ++j)
            {
                result += A(i, j) * B(i, j);
            }
        }
        
        return result;
    }

    // ============================================================================
    // Linear Algebra - Matrix Utilities
    // ============================================================================
    /**
     * @name Mat::eye(int size)
     * @brief Create an identity matrix (unit matrix) of specified size.
     * @note Identity matrix I has:
     *       - I_ij = 1 if i == j (diagonal elements)
     *       - I_ij = 0 if i != j (off-diagonal elements)
     * @note Properties:
     *       - I * A = A (left identity)
     *       - A * I = A (right identity)
     *       - I^(-1) = I (self-inverse)
     * @note For size = 0, returns empty matrix.
     *
     * @param size Size of the square identity matrix (must be >= 0)
     * @return Mat Identity matrix, or empty Mat() on error
     */
    Mat Mat::eye(int size)
    {
        // Validate parameter
        if (size < 0)
        {
            std::cerr << "[Error] eye: size must be non-negative (got " << size << ")\n";
            return Mat();
        }
        
        // Handle size = 0 (empty matrix)
        if (size == 0)
        {
            return Mat(0, 0);
        }
        
        Mat identity(size, size);
        
        // Check if matrix was created successfully
        if (identity.data == nullptr)
        {
            std::cerr << "[Error] eye: failed to create identity matrix of size " << size << "\n";
            return Mat();
        }
        
        // Initialize identity matrix: diagonal = 1, others = 0
        // Note: Mat constructor already initializes to 0, so we only need to set diagonal
        for (int i = 0; i < size; ++i)
        {
            identity(i, i) = 1.0f;  // Set diagonal elements to 1
            // Off-diagonal elements are already 0 from constructor
        }
        
        return identity;
    }

    /**
     * @name Mat::ones(int rows, int cols)
     * @brief Create a matrix filled with ones (all elements = 1.0f).
     * @note Creates a matrix where every element is 1.0f.
     * @note For rows = 0 or cols = 0, returns empty matrix.
     *
     * @param rows Number of rows (must be >= 0)
     * @param cols Number of columns (must be >= 0)
     * @return Mat Matrix filled with ones, or empty Mat() on error
     */
    Mat Mat::ones(int rows, int cols)
    {
        // Validate parameters
        if (rows < 0 || cols < 0)
        {
            std::cerr << "[Error] ones: dimensions must be non-negative (got rows=" 
                      << rows << ", cols=" << cols << ")\n";
            return Mat();
        }
        
        // Handle empty matrix
        if (rows == 0 || cols == 0)
        {
            return Mat(rows, cols);
        }
        
        Mat result(rows, cols);
        
        // Check if matrix was created successfully
        if (result.data == nullptr)
        {
            std::cerr << "[Error] ones: failed to create matrix of size " 
                      << rows << "x" << cols << "\n";
            return Mat();
        }
        
        // Fill all elements with 1.0f
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                result(i, j) = 1.0f;
            }
        }
        
        return result;
    }

    /**
     * @name Mat::ones(int size)
     * @brief Create a square matrix filled with ones (all elements = 1.0f).
     * @note Convenience function that creates a size×size matrix filled with ones.
     *       Equivalent to ones(size, size).
     * @note For size = 0, returns empty matrix.
     *
     * @param size Size of the square matrix (rows = cols, must be >= 0)
     * @return Mat Square matrix [size x size] with all elements = 1, or empty Mat() on error
     */
    Mat Mat::ones(int size)
    {
        // All validation is handled by ones(size, size)
        return Mat::ones(size, size);
    }

    /**
     * @name Mat::augment(const Mat &A, const Mat &B)
     * @brief Augment two matrices horizontally [A | B] (concatenate columns).
     * @note Creates a new matrix by placing B to the right of A.
     *       Result: [A | B] with dimensions (rows, A.cols + B.cols)
     *       where rows must be the same for both matrices.
     * @note Common use case: Creating augmented matrix [A | b] for solving Ax = b
     *       using Gaussian elimination.
     * @note For empty matrices, returns empty matrix if dimensions are valid.
     *
     * @param A Left matrix
     * @param B Right matrix
     * @return Mat Augmented matrix [A B], or empty Mat() on error
     */
    Mat Mat::augment(const Mat &A, const Mat &B)
    {
        // Validate matrix dimensions
        if (A.row < 0 || A.col < 0)
        {
            std::cerr << "[Error] augment: invalid matrix A dimensions: rows=" 
                      << A.row << ", cols=" << A.col << "\n";
            return Mat();
        }
        
        if (B.row < 0 || B.col < 0)
        {
            std::cerr << "[Error] augment: invalid matrix B dimensions: rows=" 
                      << B.row << ", cols=" << B.col << "\n";
            return Mat();
        }
        
        // Check if row counts match
        if (A.row != B.row)
        {
            std::cerr << "[Error] augment: row counts must match (A: " 
                      << A.row << ", B: " << B.row << ")\n";
            return Mat();
        }
        
        // Check for integer overflow in column sum
        if (A.col > INT_MAX - B.col)
        {
            std::cerr << "[Error] augment: combined column count too large, integer overflow "
                      << "(A.col=" << A.col << ", B.col=" << B.col << ")\n";
            return Mat();
        }

        // For non-empty matrices, data pointers must be valid.
        // Empty matrices (rows==0 or cols==0) are allowed to have nullptr data.
        if (A.row > 0 && A.col > 0 && A.data == nullptr)
        {
            std::cerr << "[Error] augment: matrix A data pointer is null\n";
            return Mat();
        }
        if (B.row > 0 && B.col > 0 && B.data == nullptr)
        {
            std::cerr << "[Error] augment: matrix B data pointer is null\n";
            return Mat();
        }
        
        // Create new matrix with combined columns
        Mat AB(A.row, A.col + B.col);
        
        // Check if matrix was created successfully
        if (AB.data == nullptr)
        {
            std::cerr << "[Error] augment: failed to create augmented matrix of size " 
                      << A.row << "x" << (A.col + B.col) << "\n";
            return Mat();
        }
        
        // Handle empty matrices
        if (A.row == 0)
        {
            // Empty result matrix already created, return it
            return AB;
        }
        
        // Copy data from A and B
        for (int i = 0; i < A.row; ++i)
        {
            // Copy A (left part)
            for (int j = 0; j < A.col; ++j)
            {
                AB(i, j) = A(i, j);
            }
            // Copy B (right part)
            for (int j = 0; j < B.col; ++j)
            {
                AB(i, A.col + j) = B(i, j);
            }
        }

        return AB;
    }

    /**
     * @name Mat::vstack(const Mat &A, const Mat &B)
     * @brief Vertically stack two matrices [A; B] (concatenate rows).
     * @note Creates a new matrix by placing B below A.
     *       Result: [A; B] with dimensions (A.rows + B.rows, cols)
     *       where cols must be the same for both matrices.
     * @note Common use case: Combining data from multiple sources vertically,
     *       or building block matrices in linear algebra operations.
     * @note For empty matrices, returns empty matrix if dimensions are valid.
     *
     * @param A Top matrix
     * @param B Bottom matrix
     * @return Mat Vertically stacked matrix [A; B], or empty Mat() on error
     */
    Mat Mat::vstack(const Mat &A, const Mat &B)
    {
        // Validate matrix dimensions
        if (A.row < 0 || A.col < 0)
        {
            std::cerr << "[Error] vstack: invalid matrix A dimensions: rows=" 
                      << A.row << ", cols=" << A.col << "\n";
            return Mat();
        }
        
        if (B.row < 0 || B.col < 0)
        {
            std::cerr << "[Error] vstack: invalid matrix B dimensions: rows=" 
                      << B.row << ", cols=" << B.col << "\n";
            return Mat();
        }
        
        // Check if column counts match
        if (A.col != B.col)
        {
            std::cerr << "[Error] vstack: column counts must match (A: " 
                      << A.col << ", B: " << B.col << ")\n";
            return Mat();
        }
        
        // Check for integer overflow in row sum
        if (A.row > INT_MAX - B.row)
        {
            std::cerr << "[Error] vstack: combined row count too large, integer overflow "
                      << "(A.row=" << A.row << ", B.row=" << B.row << ")\n";
            return Mat();
        }

        // For non-empty matrices, data pointers must be valid.
        // Empty matrices (rows==0 or cols==0) are allowed to have nullptr data.
        if (A.row > 0 && A.col > 0 && A.data == nullptr)
        {
            std::cerr << "[Error] vstack: matrix A data pointer is null\n";
            return Mat();
        }
        if (B.row > 0 && B.col > 0 && B.data == nullptr)
        {
            std::cerr << "[Error] vstack: matrix B data pointer is null\n";
            return Mat();
        }
        
        // Create new matrix with combined rows
        Mat AB(A.row + B.row, A.col);
        
        // Check if matrix was created successfully
        if (AB.data == nullptr)
        {
            std::cerr << "[Error] vstack: failed to create stacked matrix of size " 
                      << (A.row + B.row) << "x" << A.col << "\n";
            return Mat();
        }
        
        // Handle empty matrices
        if (A.col == 0)
        {
            // Empty result matrix already created, return it
            return AB;
        }
        
        // Copy data from A and B
        // Copy A (top rows)
        for (int i = 0; i < A.row; ++i)
        {
            for (int j = 0; j < A.col; ++j)
            {
                AB(i, j) = A(i, j);
            }
        }
        // Copy B (bottom rows)
        for (int i = 0; i < B.row; ++i)
        {
            for (int j = 0; j < B.col; ++j)
            {
                AB(A.row + i, j) = B(i, j);
            }
        }

        return AB;
    }

    /**
     * @name Mat::gram_schmidt_orthogonalize()
     * @brief QR-style orthogonalization of a column-vector set via Modified
     *        Gram-Schmidt (MGS) with conditional twice-reorthogonalization.
     *
     * Given an input matrix A whose columns are the vectors a_0..a_{n-1}, this
     * function produces:
     *   - Q (orthogonal_vectors, m x n) : columns q_0..q_{n-1} are orthonormal
     *                                      (Q^T * Q = I to working precision).
     *   - R (coefficients,       n x n) : upper triangular such that A = Q * R,
     *                                      where R(k,j) is the projection of
     *                                      a_j onto q_k.
     *
     * Algorithmic notes
     * -----------------
     *  - We use Modified Gram-Schmidt rather than Classical GS: each projection
     *    is subtracted immediately, which is much more numerically stable.
     *  - A second MGS sweep (Kahan/Parlett "twice is enough") is performed
     *    *only* when pass-1 cancelled away more than ~30 % of the original
     *    norm. This recovers full orthogonality on ill-conditioned inputs while
     *    saving roughly half the FLOPs on well-conditioned inputs.
     *  - When a column is linearly dependent on the previous q's, we set
     *    R(j,j) = 0 and synthesize a substitute q_j by orthogonalizing the
     *    first standard basis vector e_b that survives the projection. This
     *    keeps Q a complete orthonormal set without disturbing A = Q*R
     *    (because the dependent column is fully described by R(0..j-1, j)).
     *
     * @param vectors             Input matrix; columns are the vectors a_j.
     * @param orthogonal_vectors  [out] Q (m x n), orthonormal columns.
     * @param coefficients        [out] R (n x n), upper triangular.
     * @param tolerance           Rank threshold; columns whose post-orthogonal
     *                            norm falls below max(tolerance, 1e-5f) are
     *                            treated as linearly dependent. Must be >= 0.
     *                            The 1e-5f floor is enforced to keep
     *                            single-precision noise from being mistaken
     *                            for a real direction.
     * @return true on success; false on invalid input.
     */
    bool Mat::gram_schmidt_orthogonalize(const Mat &vectors, Mat &orthogonal_vectors,
                                         Mat &coefficients, float tolerance)
    {
        // ----------------------------------------------------------------
        // 1) Input validation
        // ----------------------------------------------------------------
        if (vectors.data == nullptr)
        {
            std::cerr << "[Error] gram_schmidt_orthogonalize: input matrix is null.\n";
            return false;
        }

        const int m = vectors.row;  // dimension of each column vector
        const int n = vectors.col;  // number of column vectors

        if (m <= 0 || n <= 0)
        {
            std::cerr << "[Error] gram_schmidt_orthogonalize: invalid dimensions (m="
                      << m << ", n=" << n << ")\n";
            return false;
        }
        if (tolerance < 0.0f)
        {
            std::cerr << "[Error] gram_schmidt_orthogonalize: tolerance must be non-negative (got "
                      << tolerance << ")\n";
            return false;
        }

        // ----------------------------------------------------------------
        // 2) Allocate outputs
        //    Q : m x n  -- orthonormal columns
        //    R : n x n  -- upper-triangular coefficients (A = Q * R)
        // ----------------------------------------------------------------
        orthogonal_vectors = Mat(m, n);
        coefficients       = Mat(n, n);
        if (orthogonal_vectors.data == nullptr || coefficients.data == nullptr)
        {
            std::cerr << "[Error] gram_schmidt_orthogonalize: output allocation failed.\n";
            return false;
        }
        coefficients.clear();  // ensure strictly-lower & off-band entries are 0

        // ----------------------------------------------------------------
        // 3) Effective thresholds (kept in single source of truth)
        // ----------------------------------------------------------------
        // Single-precision rank floor: below this we cannot trust the value
        // of ||q_j|| against rounding noise, so we always clamp here.
        constexpr float kRankFloor    = 1e-5f;
        // Kahan's "twice is enough" trigger: 1/sqrt(2). If pass-1 left the
        // working column shorter than this fraction of its original length,
        // significant cancellation occurred and a second MGS pass is needed.
        constexpr float kReorthRatio  = 0.7071068f;

        const float rank_tol = (tolerance > kRankFloor) ? tolerance : kRankFloor;

        // ----------------------------------------------------------------
        // 4) Local column-wise primitives on Q
        //    The matrix is row-major with stride 'step', so columns are not
        //    contiguous; we rely on the inline operator() in the header.
        //    Lambdas below are zero-overhead after inlining.
        // ----------------------------------------------------------------
        Mat &Q = orthogonal_vectors;
        Mat &R = coefficients;

        // <Q(:,k), Q(:,j)>
        auto col_dot = [&](int k, int j) -> float {
            float s = 0.0f;
            for (int i = 0; i < m; ++i) s += Q(i, k) * Q(i, j);
            return s;
        };
        // ||Q(:,j)||_2
        auto col_norm = [&](int j) -> float {
            float s = 0.0f;
            for (int i = 0; i < m; ++i) { const float v = Q(i, j); s += v * v; }
            return sqrtf(s);
        };
        // Q(:,j) <- Q(:,j) - c * Q(:,k)
        auto col_axpy = [&](int j, int k, float c) {
            for (int i = 0; i < m; ++i) Q(i, j) -= c * Q(i, k);
        };
        // One MGS sweep: orthogonalize Q(:,j) against Q(:,0..j-1).
        //   write_R == true  -> dot products are recorded in R(k,j)
        //   accumulate       -> add to R(k,j) (used by pass-2) instead of overwrite
        auto mgs_sweep = [&](int j, bool write_R, bool accumulate) {
            for (int k = 0; k < j; ++k)
            {
                const float dot = col_dot(k, j);
                if (write_R)
                {
                    if (accumulate) R(k, j) += dot;
                    else            R(k, j)  = dot;
                }
                col_axpy(j, k, dot);
            }
        };

        // ----------------------------------------------------------------
        // 5) Main orthogonalization loop
        // ----------------------------------------------------------------
        for (int j = 0; j < n; ++j)
        {
            // 5.1 Initialize working column: Q(:,j) <- A(:,j)
            for (int i = 0; i < m; ++i) Q(i, j) = vectors(i, j);

            // 5.2 First MGS pass (always); records R(k,j) for k < j
            const float norm_before = col_norm(j);   // = ||A(:,j)||
            mgs_sweep(j, /*write_R=*/true, /*accumulate=*/false);
            float norm_after = col_norm(j);

            // 5.3 Conditional second MGS pass (Kahan)
            //     Triggered only when significant cancellation occurred.
            if (norm_after < kReorthRatio * norm_before)
            {
                mgs_sweep(j, /*write_R=*/true, /*accumulate=*/true);
                norm_after = col_norm(j);
            }

            // 5.4 Regular case: column is independent -> normalize and store R(j,j)
            if (norm_after >= rank_tol)
            {
                R(j, j) = norm_after;
                const float inv = 1.0f / norm_after;
                for (int i = 0; i < m; ++i) Q(i, j) *= inv;
                continue;
            }

            // 5.5 Linearly-dependent case: synthesize a substitute q_j
            //
            // Mathematically a_j ∈ span{q_0..q_{j-1}}, so R(j,j) = 0 and the
            // already-written R(0..j-1, j) reconstruct a_j via Q*R. We still
            // need a unit q_j orthogonal to all previous columns so that Q
            // remains a complete orthonormal set (consumers such as SVD or
            // null-space queries rely on this).
            //
            // Strategy: try standard basis vectors e_0, e_1, ..., e_{m-1};
            // keep the first one whose projected residual is large enough.
            // Re-orthogonalization is applied unconditionally here because an
            // e_b that lies almost inside span{q_0..q_{j-1}} loses most of
            // its norm in pass-1 and is numerically fragile.
            R(j, j) = 0.0f;
            bool found = false;

            for (int b = 0; b < m && !found; ++b)
            {
                for (int i = 0; i < m; ++i) Q(i, j) = (i == b) ? 1.0f : 0.0f;

                mgs_sweep(j, /*write_R=*/false, /*accumulate=*/false);  // pass 1
                mgs_sweep(j, /*write_R=*/false, /*accumulate=*/false);  // pass 2

                const float new_norm = col_norm(j);
                if (new_norm > kRankFloor)
                {
                    const float inv = 1.0f / new_norm;
                    for (int i = 0; i < m; ++i) Q(i, j) *= inv;
                    found = true;
                }
            }

            // 5.6 Defensive fallback: only reachable when n > m, i.e. the
            //     caller asked for more orthonormal columns than the ambient
            //     dimension can supply. Leave Q(:,j) as a zero vector.
            if (!found)
            {
                for (int i = 0; i < m; ++i) Q(i, j) = 0.0f;
            }
        }

        return true;
    }

    // ============================================================================
    // Linear Algebra - Matrix Operations
    // ============================================================================
    /**
     * @name Mat::minor(int target_row, int target_col) const
     * @brief Submatrix obtained by deleting one row and one column.
     *
     * For an m x n input, the result is (m-1) x (n-1). This is purely a
     * geometric "delete a row, delete a column" operation and is therefore
     * defined on ANY rectangular matrix; it does NOT require squareness.
     * (Squareness is only needed when the caller subsequently takes a
     * determinant, e.g. for the minor value or cofactor value -- which is
     * why cofactor() is the function that enforces it.)
     *
     * Edge case: if either dimension would collapse to 0 after the deletion
     * (i.e. m == 1 or n == 1), the result is the canonical empty Mat(0, 0).
     *
     * Naming reference:
     *   - minor   matrix  M_ij = this function's return value.
     *   - minor   value   m_ij = det(M_ij)             [requires square]
     *   - cofactor value  C_ij = (-1)^(i+j) * det(M_ij) [requires square]
     *
     * @param target_row Row index to remove (0-based, in [0, this->row)).
     * @param target_col Column index to remove (0-based, in [0, this->col)).
     * @return Mat The (m-1)x(n-1) submatrix; Mat(0, 0) on invalid input or
     *             when an output dimension would be zero.
     */
    Mat Mat::minor(int target_row, int target_col) const
    {
        // ----------------------------------------------------------------
        // 1) Input validation
        // ----------------------------------------------------------------
        if (this->data == nullptr)
        {
            std::cerr << "[Error] minor: matrix data pointer is null\n";
            return Mat(0, 0);
        }
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] minor: invalid matrix dimensions (rows="
                      << this->row << ", cols=" << this->col << ")\n";
            return Mat(0, 0);
        }

        const int rows = this->row;
        const int cols = this->col;

        if (target_row < 0 || target_row >= rows)
        {
            std::cerr << "[Error] minor: target_row=" << target_row
                      << " is out of range [0, " << (rows - 1) << "]\n";
            return Mat(0, 0);
        }
        if (target_col < 0 || target_col >= cols)
        {
            std::cerr << "[Error] minor: target_col=" << target_col
                      << " is out of range [0, " << (cols - 1) << "]\n";
            return Mat(0, 0);
        }

        // Degenerate cases: deleting the only row or the only column leaves
        // a matrix with a zero dimension. Collapse all such results to the
        // canonical Mat(0, 0) for predictable downstream handling.
        if (rows == 1 || cols == 1)
        {
            return Mat(0, 0);
        }

        // ----------------------------------------------------------------
        // 2) Allocate result : (rows-1) x (cols-1)
        // ----------------------------------------------------------------
        Mat result(rows - 1, cols - 1);
        if (result.data == nullptr)
        {
            std::cerr << "[Error] minor: failed to create result matrix\n";
            return Mat(0, 0);
        }

        // ----------------------------------------------------------------
        // 3) Bulk copy with row-skip and column-skip
        //
        //    Each surviving row is copied as at most TWO contiguous chunks:
        //
        //      src row i :  [ 0 .. tc-1 | tc | tc+1 .. cols-1 ]
        //                    └─ left ─┘  skip  └──  right  ──┘
        //      dst row   :  [ 0 .. tc-1     | tc .. cols-2    ]
        //
        //    This eliminates the inner per-element branch (j == target_col)
        //    and lets std::memcpy use the platform's optimized block copy
        //    (vectorized / aligned). Asymptotic cost is still O(rows*cols)
        //    but the per-iteration constant drops several-fold for large
        //    matrices. (Especially valuable inside O(n!) Laplace recursion.)
        // ----------------------------------------------------------------
        const int src_step    = this->step;
        const int dst_step    = result.step;
        const int left_count  = target_col;                  // cols [0, tc)
        const int right_count = (cols - 1) - target_col;     // cols (tc, cols)

        int res_i = 0;
        for (int i = 0; i < rows; ++i)
        {
            if (i == target_row)
            {
                continue;
            }

            const float *src_row = this->data  + i     * src_step;
            float       *dst_row = result.data + res_i * dst_step;

            // Left chunk: columns [0, target_col)
            if (left_count > 0)
            {
                std::memcpy(dst_row,
                            src_row,
                            sizeof(float) * static_cast<size_t>(left_count));
            }
            // Right chunk: columns (target_col, cols), shifted left by 1 in dst
            if (right_count > 0)
            {
                std::memcpy(dst_row + left_count,
                            src_row + target_col + 1,
                            sizeof(float) * static_cast<size_t>(right_count));
            }

            ++res_i;
        }

        return result;
    }

    /**
     * @name Mat::cofactor(int target_row, int target_col) const
     * @brief Cofactor submatrix (same shape/contents as the minor matrix).
     *
     * The (i,j) cofactor matrix C_ij equals the minor matrix M_ij; the only
     * difference is semantic: cofactor() is the entry point used when the
     * caller is going to apply the sign (-1)^(i+j) and take a determinant
     * to obtain the cofactor VALUE. Because the cofactor value requires
     * det(M_ij), the input must be SQUARE -- this function therefore
     * enforces squareness, while minor() allows any m x n shape.
     *
     * @note Cofactor VALUE computation:
     *       float val = ((i + j) % 2 == 0 ? 1.0f : -1.0f) *
     *                   A.cofactor(i, j).determinant();
     *
     * @param target_row Row index to remove (0-based, in [0, n)).
     * @param target_col Column index to remove (0-based, in [0, n)).
     * @return Mat The (n-1) x (n-1) cofactor submatrix; Mat(0, 0) on error
     *             (non-square, out-of-range index, or 1x1 input).
     */
    Mat Mat::cofactor(int target_row, int target_col) const
    {
        // Cofactor is only well-defined on square matrices, since downstream
        // it is consumed via det(M_ij). Validate squareness here, then
        // delegate the actual submatrix extraction to minor().
        if (this->row != this->col)
        {
            std::cerr << "[Error] cofactor: requires a square matrix (got "
                      << this->row << "x" << this->col << ")\n";
            return Mat(0, 0);
        }
        return this->minor(target_row, target_col);
    }

    /**
     * @name Mat::gaussian_eliminate
     * @brief Gaussian elimination with partial pivoting -> Row Echelon Form.
     *
     * Reduces the matrix to upper-triangular Row Echelon Form (REF) via
     * elementary row operations (row swaps + row additions).
     *
     * Numerical strategy
     * ------------------
     *  - **Partial pivoting**: at each step, the row with the LARGEST
     *    |element| in the current column (within rows [r, rows)) is chosen
     *    as the pivot row and swapped into row r. This bounds the
     *    multiplier |factor| <= 1 and keeps roundoff under control. This
     *    is the standard textbook recipe used by LAPACK's *getrf, and is
     *    materially more stable than picking the first non-zero entry.
     *  - **Exact zero in the eliminated column**: after subtracting the
     *    pivot row, the entry directly under the pivot is FORCED to 0.0f
     *    (it is mathematically zero; this avoids visible roundoff in REF
     *    structure without polluting the rest of the row).
     *  - Other entries are left as they actually compute -- any small
     *    "denormal noise" floor decision is left to the caller.
     *
     * Algorithmic complexity: O(rows * cols * min(rows, cols))
     *
     * @return Mat REF copy of the input. Mat(0, 0) on invalid input;
     *             a copy of the (empty) input when row == 0 or col == 0.
     */
    Mat Mat::gaussian_eliminate() const
    {
        // ----------------------------------------------------------------
        // 1) Input validation
        // ----------------------------------------------------------------
        if (this->data == nullptr)
        {
            std::cerr << "[Error] gaussian_eliminate: matrix data pointer is null\n";
            return Mat(0, 0);
        }
        if (this->row < 0 || this->col < 0)
        {
            std::cerr << "[Error] gaussian_eliminate: invalid matrix dimensions (rows="
                      << this->row << ", cols=" << this->col << ")\n";
            return Mat(0, 0);
        }
        // Valid empty matrix (one or both dims == 0): nothing to do.
        if (this->row == 0 || this->col == 0)
        {
            return Mat(*this);
        }

        // ----------------------------------------------------------------
        // 2) Working copy
        // ----------------------------------------------------------------
        Mat result(*this);
        if (result.data == nullptr)
        {
            std::cerr << "[Error] gaussian_eliminate: failed to create working copy\n";
            return Mat(0, 0);
        }

        const int rows = result.row;
        const int cols = result.col;
        const int rstep = result.step;

        // Pivot acceptance threshold. TINY_MATH_MIN_POSITIVE_INPUT_F32 is
        // the project-wide "non-zero" floor. With partial pivoting we
        // declare the column "pivot-less" only if even its LARGEST entry
        // sits below this threshold, so this check is much less aggressive
        // than the old "first non-zero" pattern.
        const float pivot_tol = TINY_MATH_MIN_POSITIVE_INPUT_F32;

        // ----------------------------------------------------------------
        // 3) Main elimination loop
        //    Independently advance row index 'r' and column index 'lead'.
        //    When the current column has no usable pivot, advance 'lead'
        //    only and try again with the same 'r' (these rows then take a
        //    "free" position, contributing zeros to the rank/echelon).
        // ----------------------------------------------------------------
        int r    = 0;
        int lead = 0;

        while (r < rows && lead < cols)
        {
            // 3.1 Find pivot row: argmax_{i in [r, rows)} |A(i, lead)|
            int   pivot_row = r;
            float pivot_abs = fabsf(result.data[r * rstep + lead]);
            for (int i = r + 1; i < rows; ++i)
            {
                const float v = fabsf(result.data[i * rstep + lead]);
                if (v > pivot_abs)
                {
                    pivot_abs = v;
                    pivot_row = i;
                }
            }

            // 3.2 No usable pivot in this column -> skip to next column,
            //     keep r unchanged so this row tries again later.
            if (pivot_abs < pivot_tol)
            {
                ++lead;
                continue;
            }

            // 3.3 Bring pivot row to position r
            if (pivot_row != r)
            {
                result.swap_rows(pivot_row, r);
            }

            // 3.4 Eliminate all rows below the pivot
            //     Row pointers hoisted out of the inner loop to avoid the
            //     repeated (i*step + j) address calculation; this also
            //     lets the compiler vectorize the AXPY across columns.
            float       *row_r  = result.data + r * rstep;
            const float  pivot  = row_r[lead];

            for (int j = r + 1; j < rows; ++j)
            {
                float *row_j = result.data + j * rstep;

                // Skip rows whose entry in the pivot column is already ~0
                if (fabsf(row_j[lead]) < pivot_tol)
                {
                    row_j[lead] = 0.0f;   // tidy any tiny residue
                    continue;
                }

                const float factor = row_j[lead] / pivot;
                // The eliminated column is mathematically zero after the
                // subtraction below; assign it directly to avoid visible
                // roundoff and shave one FMA per row.
                row_j[lead] = 0.0f;

                for (int k = lead + 1; k < cols; ++k)
                {
                    row_j[k] -= factor * row_r[k];
                }
            }

            // 3.5 Advance both indices: this pivot is consumed.
            ++r;
            ++lead;
        }

        return result;
    }

    /**
     * @name Mat::row_reduce_from_gaussian() const
     * @brief Convert a Row Echelon Form (REF) matrix to Reduced Row Echelon
     *        Form (RREF) by back-substitution.
     *
     * Contract: input is assumed to already be in REF (zeros below each
     * pivot, pivots strictly stair-step to the right). Calling this on
     * non-REF inputs is undefined behaviour by contract, though for
     * "almost REF" inputs it still degrades gracefully because each row's
     * leading non-zero is taken as that row's pivot.
     *
     * Algorithm
     * ---------
     *   For each row from the bottom up:
     *     a) Locate the pivot = first |entry| >= pivot_tol in that row.
     *        (Since the input is REF, this IS the pivot column.)
     *     b) Scale the row so the pivot becomes exactly 1.0.
     *     c) Subtract appropriate multiples of the row from every row
     *        above to zero out their entry in the pivot column.
     *
     * Numerical strategy
     * ------------------
     *  - The pivot column entry is set to its mathematically exact value
     *    (1.0 after normalize, 0.0 after eliminate-above) DIRECTLY,
     *    avoiding visible roundoff while keeping the rest of each row
     *    untouched (no per-element noise floor).
     *  - Inner loops are written against raw row pointers to skip the
     *    operator() index math and to give the compiler a clean AXPY/
     *    scaled-assign shape suitable for auto-vectorization.
     *
     * @return Mat RREF copy of the input. Mat(0, 0) on invalid input;
     *             a copy of the (empty) input when row == 0 or col == 0.
     */
    Mat Mat::row_reduce_from_gaussian() const
    {
        // ----------------------------------------------------------------
        // 1) Input validation
        // ----------------------------------------------------------------
        if (this->data == nullptr)
        {
            std::cerr << "[Error] row_reduce_from_gaussian: matrix data pointer is null\n";
            return Mat(0, 0);
        }
        if (this->row < 0 || this->col < 0)
        {
            std::cerr << "[Error] row_reduce_from_gaussian: invalid matrix dimensions (rows="
                      << this->row << ", cols=" << this->col << ")\n";
            return Mat(0, 0);
        }
        // Valid empty matrix: nothing to reduce.
        if (this->row == 0 || this->col == 0)
        {
            return Mat(*this);
        }

        // ----------------------------------------------------------------
        // 2) Working copy
        // ----------------------------------------------------------------
        Mat R(*this);
        if (R.data == nullptr)
        {
            std::cerr << "[Error] row_reduce_from_gaussian: failed to create working copy\n";
            return Mat(0, 0);
        }

        const int rows  = R.row;
        const int cols  = R.col;
        const int rstep = R.step;

        const float pivot_tol = TINY_MATH_MIN_POSITIVE_INPUT_F32;

        // ----------------------------------------------------------------
        // 3) Back-substitution loop: bottom row -> top row
        // ----------------------------------------------------------------
        for (int p = rows - 1; p >= 0; --p)
        {
            float *row_p = R.data + p * rstep;

            // 3.1 Locate pivot in row p (first |entry| >= pivot_tol).
            int   pivot_col = -1;
            float pivot_val = 0.0f;
            for (int k = 0; k < cols; ++k)
            {
                if (fabsf(row_p[k]) >= pivot_tol)
                {
                    pivot_col = k;
                    pivot_val = row_p[k];
                    break;
                }
            }
            if (pivot_col < 0)
            {
                // All-zero row -> nothing to do.
                continue;
            }

            // 3.2 Normalize the pivot row so that row_p[pivot_col] == 1.0.
            //     The pivot column is set to exact 1.0 directly; only
            //     columns to its right need the actual division.
            const float inv_pivot = 1.0f / pivot_val;
            row_p[pivot_col] = 1.0f;
            for (int s = pivot_col + 1; s < cols; ++s)
            {
                row_p[s] *= inv_pivot;
            }

            // 3.3 Eliminate above the pivot:
            //     row_t[s] -= factor * row_p[s]   for each row t < p,
            //     where factor = row_t[pivot_col] (the entry to clear).
            for (int t = p - 1; t >= 0; --t)
            {
                float       *row_t = R.data + t * rstep;
                const float  factor = row_t[pivot_col];

                if (fabsf(factor) < pivot_tol)
                {
                    row_t[pivot_col] = 0.0f;   // tidy any tiny residue
                    continue;
                }

                // Eliminated column is mathematically zero; assign
                // directly and start the inner loop one column later.
                row_t[pivot_col] = 0.0f;
                for (int s = pivot_col + 1; s < cols; ++s)
                {
                    row_t[s] -= factor * row_p[s];
                }
            }
        }

        return R;
    }

    /**
     * @name Mat::inverse_gje()
     * @brief Compute the inverse of a square matrix via Gauss-Jordan elimination.
     *
     * @details
     *   Algorithm (classical Gauss-Jordan on the augmented matrix):
     *     1. Build the augmented matrix M = [A | I] of size n x 2n.
     *     2. Reduce M to row-echelon form (REF) via gaussian_eliminate(), which
     *        applies partial pivoting (largest |pivot| in column).
     *     3. Reduce REF to reduced row-echelon form (RREF) via
     *        row_reduce_from_gaussian(), which back-substitutes to clear entries
     *        above each pivot and normalizes pivots to 1.
     *     4. If A is invertible, M reduces to [I | A^{-1}]; the right block is
     *        the inverse. Otherwise the left block is not identity and we abort.
     *
     *   Singularity test: after RREF, gaussian_eliminate + row_reduce_from_gaussian
     *   force pivot rows / columns to exact 1.0 / 0.0 (no roundoff). For an
     *   invertible A, every left-block column is a pivot column, so the left
     *   block equals I exactly. For a singular A, at least one column is a
     *   non-pivot column whose entries are structurally non-zero, far above
     *   TINY_MATH_MIN_POSITIVE_INPUT_F32. A small "any non-zero" threshold is
     *   therefore both necessary and sufficient.
     *
     * @note Complexity: O(n^3) time, O(n^2) extra memory (one n x 2n scratch
     *       matrix). Preferred over the adjoint method for n >= 4.
     * @note On any error (null data, non-square, singular, allocation failure)
     *       returns an empty matrix Mat(0, 0).
     *
     * @return Mat n x n inverse if A is invertible; empty Mat(0, 0) otherwise.
     */
    Mat Mat::inverse_gje() const
    {
        // ---------------------------------------------------------------------
        // 1. Validate input
        // ---------------------------------------------------------------------
        if (this->data == nullptr)
        {
            std::cerr << "[Error] inverse_gje: matrix data pointer is null.\n";
            return Mat(0, 0);
        }
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] inverse_gje: invalid matrix dimensions: "
                      << this->row << "x" << this->col << ".\n";
            return Mat(0, 0);
        }
        if (this->row != this->col)
        {
            std::cerr << "[Error] inverse_gje: requires a square matrix (got "
                      << this->row << "x" << this->col << ").\n";
            return Mat(0, 0);
        }

        const int n = this->row;

        // ---------------------------------------------------------------------
        // 2. Build the augmented matrix M = [A | I]
        // ---------------------------------------------------------------------
        Mat I = Mat::eye(n);
        if (I.data == nullptr)
        {
            std::cerr << "[Error] inverse_gje: failed to allocate identity matrix.\n";
            return Mat(0, 0);
        }

        Mat augmented = Mat::augment(*this, I);
        if (augmented.data == nullptr)
        {
            std::cerr << "[Error] inverse_gje: failed to build augmented matrix.\n";
            return Mat(0, 0);
        }
        // augment() guarantees augmented.col == n + n on success; no extra check needed.

        // ---------------------------------------------------------------------
        // 3. Reduce [A | I] to RREF via REF -> RREF
        // ---------------------------------------------------------------------
        Mat rref = augmented.gaussian_eliminate();
        if (rref.data == nullptr)
        {
            std::cerr << "[Error] inverse_gje: gaussian_eliminate failed.\n";
            return Mat(0, 0);
        }

        rref = rref.row_reduce_from_gaussian();
        if (rref.data == nullptr)
        {
            std::cerr << "[Error] inverse_gje: row_reduce_from_gaussian failed.\n";
            return Mat(0, 0);
        }

        // ---------------------------------------------------------------------
        // 4. Singularity test: left block must equal identity
        //
        //    Because GE/RREF set pivot columns to exact 1.0 / 0.0, a fully
        //    pivoted (i.e. invertible) matrix yields an exact identity on the
        //    left. Any deviation > MIN_POSITIVE_INPUT signals a missing pivot.
        // ---------------------------------------------------------------------
        for (int i = 0; i < n; ++i)
        {
            const float* row_i = rref.data + i * rref.step;
            for (int j = 0; j < n; ++j)
            {
                const float expected = (i == j) ? 1.0f : 0.0f;
                if (fabsf(row_i[j] - expected) > TINY_MATH_MIN_POSITIVE_INPUT_F32)
                {
                    std::cerr << "[Error] inverse_gje: matrix is singular "
                              << "(left block not identity at (" << i << ", " << j
                              << "): expected " << expected << ", got " << row_i[j] << ").\n";
                    return Mat(0, 0);
                }
            }
        }

        // ---------------------------------------------------------------------
        // 5. Extract the right block as A^{-1} (one memcpy per row)
        // ---------------------------------------------------------------------
        Mat result(n, n);
        if (result.data == nullptr)
        {
            std::cerr << "[Error] inverse_gje: failed to allocate result matrix.\n";
            return Mat(0, 0);
        }

        const size_t row_bytes = sizeof(float) * static_cast<size_t>(n);
        for (int i = 0; i < n; ++i)
        {
            std::memcpy(result.data + i * result.step,
                        rref.data + i * rref.step + n,
                        row_bytes);
        }

        return result;
    }

    /**
     * @name Mat::solve
     * @brief Solve the linear system Ax = b using Gaussian elimination.
     *
     * @details
     *   The method builds the augmented matrix [A | b], reduces it to REF with
     *   gaussian_eliminate() (partial pivoting), then performs back-substitution.
     *   This keeps solve() numerically aligned with the shared elimination path
     *   used by inverse_gje().
     *
     * @note Time complexity: O(n^3). On invalid input, singular systems, or
     *       allocation failure, returns Mat(0, 0).
     *
     * @param A Coefficient matrix (N×N, must be square)
     * @param b Result vector (N×1)
     * @return Mat Solution vector (N×1) containing x such that Ax = b, or Mat(0, 0) on error.
     */
    Mat Mat::solve(const Mat &A, const Mat &b) const
    {
        // ---------------------------------------------------------------------
        // 1. Validate input
        // ---------------------------------------------------------------------
        if (A.data == nullptr)
        {
            std::cerr << "[Error] solve: matrix A data pointer is null.\n";
            return Mat(0, 0);
        }
        if (b.data == nullptr)
        {
            std::cerr << "[Error] solve: vector b data pointer is null.\n";
            return Mat(0, 0);
        }
        if (A.row <= 0 || A.col <= 0)
        {
            std::cerr << "[Error] solve: invalid matrix A dimensions: "
                      << A.row << "x" << A.col << ".\n";
            return Mat(0, 0);
        }
        if (b.row <= 0 || b.col <= 0)
        {
            std::cerr << "[Error] solve: invalid vector b dimensions: "
                      << b.row << "x" << b.col << ".\n";
            return Mat(0, 0);
        }
        if (A.row != A.col)
        {
            std::cerr << "[Error] solve: matrix A must be square (got "
                      << A.row << "x" << A.col << ").\n";
            return Mat(0, 0);
        }
        if (A.row != b.row || b.col != 1)
        {
            std::cerr << "[Error] solve: dimensions do not match (A: "
                      << A.row << "x" << A.col << ", b: " << b.row << "x" << b.col
                      << ", expected b: " << A.row << "x1).\n";
            return Mat(0, 0);
        }

        const int n = A.row;

        // ---------------------------------------------------------------------
        // 2. Build [A | b] and reduce it to REF
        // ---------------------------------------------------------------------
        Mat augmented = Mat::augment(A, b);
        if (augmented.data == nullptr)
        {
            std::cerr << "[Error] solve: failed to build augmented matrix.\n";
            return Mat(0, 0);
        }

        Mat ref = augmented.gaussian_eliminate();
        if (ref.data == nullptr)
        {
            std::cerr << "[Error] solve: gaussian_eliminate failed.\n";
            return Mat(0, 0);
        }

        // ---------------------------------------------------------------------
        // 3. Back-substitution on the upper triangular REF
        // ---------------------------------------------------------------------
        Mat solution(n, 1);
        if (solution.data == nullptr)
        {
            std::cerr << "[Error] solve: failed to allocate solution vector.\n";
            return Mat(0, 0);
        }

        const int ref_step = ref.step;
        const int rhs_col = n;
        for (int i = n - 1; i >= 0; --i)
        {
            const float* row_i = ref.data + i * ref_step;
            const float pivot = row_i[i];
            if (fabsf(pivot) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] solve: zero or near-zero pivot at (" << i << ", " << i
                          << "), system is singular or rank-deficient.\n";
                return Mat(0, 0);
            }

            float sum = row_i[rhs_col];
            for (int j = i + 1; j < n; ++j)
            {
                sum -= row_i[j] * solution.data[j * solution.step];
            }
            solution.data[i * solution.step] = sum / pivot;
        }

        return solution;
    }

    /**
     * @name Mat::band_solve
     * @brief Solve Ax = b using band-limited Gaussian elimination.
     *
     * @details
     *   This routine assumes A is a square banded matrix whose non-zero entries
     *   lie within k total diagonals centered on the main diagonal. It avoids
     *   touching known-zero regions during elimination and back-substitution.
     *
     *   No pivoting is applied: row swaps generally widen the band and can
     *   destroy the purpose of a banded solver. If a diagonal pivot is tiny,
     *   the method fails fast; use solve() for the more robust partial-pivoting
     *   path.
     *
     * @note Time complexity: O(n * half_band^2) for symmetric narrow bands.
     * @note k is the total band width including the diagonal. For a tridiagonal
     *       matrix, k = 3 and half_band = 1.
     *
     * @param A Coefficient matrix (N×N) - banded matrix (passed by value, will be modified)
     * @param b Result vector (N×1) (passed by value, will be modified)
     * @param k Bandwidth of the matrix (must be >= 1 and odd, typically 3, 5, 7, ...)
     * @return Mat Solution vector (N×1) containing x such that Ax = b, or Mat(0, 0) on error.
     */
    Mat Mat::band_solve(Mat A, Mat b, int k) const
    {
        // ---------------------------------------------------------------------
        // 1. Validate input
        // ---------------------------------------------------------------------
        if (A.data == nullptr)
        {
            std::cerr << "[Error] band_solve: matrix A data pointer is null.\n";
            return Mat(0, 0);
        }
        if (b.data == nullptr)
        {
            std::cerr << "[Error] band_solve: vector b data pointer is null.\n";
            return Mat(0, 0);
        }
        if (A.row <= 0 || A.col <= 0)
        {
            std::cerr << "[Error] band_solve: invalid matrix A dimensions: "
                      << A.row << "x" << A.col << ".\n";
            return Mat(0, 0);
        }
        if (b.row <= 0 || b.col <= 0)
        {
            std::cerr << "[Error] band_solve: invalid vector b dimensions: "
                      << b.row << "x" << b.col << ".\n";
            return Mat(0, 0);
        }
        if (A.row != A.col)
        {
            std::cerr << "[Error] band_solve: matrix A must be square (got "
                      << A.row << "x" << A.col << ").\n";
            return Mat(0, 0);
        }
        if (A.row != b.row || b.col != 1)
        {
            std::cerr << "[Error] band_solve: dimensions do not match (A: "
                      << A.row << "x" << A.col << ", b: " << b.row << "x" << b.col
                      << ", expected b: " << A.row << "x1).\n";
            return Mat(0, 0);
        }
        if (k < 1)
        {
            std::cerr << "[Error] band_solve: bandwidth k must be >= 1 (got " << k << ").\n";
            return Mat(0, 0);
        }

        const int n = A.row;
        if (k > A.row)
        {
            std::cerr << "[Warning] band_solve: bandwidth k=" << k
                      << " is larger than matrix size " << n
                      << "; using general solve may be more efficient.\n";
        }

        const int half_band = (k - 1) / 2;
        const float pivot_tol = TINY_MATH_MIN_POSITIVE_INPUT_F32;

        // ---------------------------------------------------------------------
        // 2. Forward elimination within the band
        // ---------------------------------------------------------------------
        for (int i = 0; i < n; ++i)
        {
            float* row_i = A.data + i * A.step;
            const float pivot = row_i[i];
            if (fabsf(pivot) < pivot_tol)
            {
                std::cerr << "[Error] band_solve: zero or near-zero pivot at ("
                          << i << ", " << i << ") = " << pivot
                          << "; matrix is singular or requires pivoting.\n";
                return Mat(0, 0);
            }

            const float inv_pivot = 1.0f / pivot;
            const int last_row = std::min(n, i + half_band + 1);
            const int last_col = std::min(n, i + half_band + 1);

            for (int j = i + 1; j < last_row; ++j)
            {
                float* row_j = A.data + j * A.step;
                if (fabsf(row_j[i]) < pivot_tol)
                {
                    row_j[i] = 0.0f;
                    continue;
                }

                const float factor = row_j[i] * inv_pivot;
                row_j[i] = 0.0f; // eliminated exactly by construction

                for (int col_idx = i + 1; col_idx < last_col; ++col_idx)
                {
                    row_j[col_idx] -= row_i[col_idx] * factor;
                }
                b.data[j * b.step] -= b.data[i * b.step] * factor;
            }
        }

        // ---------------------------------------------------------------------
        // 3. Back-substitution within the upper band
        // ---------------------------------------------------------------------
        Mat x(n, 1);
        if (x.data == nullptr)
        {
            std::cerr << "[Error] band_solve: failed to allocate solution vector.\n";
            return Mat(0, 0);
        }

        for (int i = n - 1; i >= 0; --i)
        {
            const float* row_i = A.data + i * A.step;
            const float pivot = row_i[i];
            if (fabsf(pivot) < pivot_tol)
            {
                std::cerr << "[Error] band_solve: zero pivot at (" << i << ", " << i
                          << ") during back-substitution.\n";
                return Mat(0, 0);
            }

            float sum = 0.0f;
            const int max_j = std::min(n, i + half_band + 1);
            for (int j = i + 1; j < max_j; ++j)
            {
                sum += row_i[j] * x.data[j * x.step];
            }

            x.data[i * x.step] = (b.data[i * b.step] - sum) / pivot;
        }

        return x;
    }

    /**
     * @name Mat::roots(Mat A, Mat y)
     * @brief Solve the linear system A * x = y.
     *
     * @details
     *   roots() is kept as a compatibility wrapper around solve(). Older code
     *   had a second, independent elimination implementation here; delegating
     *   to solve() avoids duplicated numerical logic and gives roots() the same
     *   partial-pivoting behavior.
     *
     * @note Time complexity: O(n^3), same as solve().
     *
     * @param A Coefficient matrix (N×N, must be square)
     * @param y Result vector (N×1)
     * @return Mat Solution vector (N×1) containing x such that Ax = y, or Mat(0, 0) on error.
     */
    Mat Mat::roots(Mat A, Mat y) const
    {
        return solve(A, y);
    }

    // ============================================================================
    // Matrix Decomposition
    // ============================================================================
    /**
     * @name Mat::LUDecomposition::LUDecomposition()
     * @brief Default constructor for LUDecomposition structure
     */
    Mat::LUDecomposition::LUDecomposition()
        : L(0, 0), U(0, 0), P(0, 0), pivoted(false), status(TINY_OK)
    {
    }

    /**
     * @name Mat::CholeskyDecomposition::CholeskyDecomposition()
     * @brief Default constructor for CholeskyDecomposition structure
     */
    Mat::CholeskyDecomposition::CholeskyDecomposition()
        : L(0, 0), status(TINY_OK)
    {
    }

    /**
     * @name Mat::QRDecomposition::QRDecomposition()
     * @brief Default constructor for QRDecomposition structure
     */
    Mat::QRDecomposition::QRDecomposition()
        : Q(0, 0), R(0, 0), status(TINY_OK)
    {
    }

    /**
     * @name Mat::SVDDecomposition::SVDDecomposition()
     * @brief Default constructor for SVDDecomposition structure
     */
    Mat::SVDDecomposition::SVDDecomposition()
        : U(0, 0), S(0, 0), V(0, 0), rank(0), iterations(0), status(TINY_OK)
    {
    }

    /**
     * @name Mat::is_positive_definite()
     * @brief Check whether a symmetric matrix is positive definite.
     *
     * @details
     *   A symmetric matrix A is positive definite iff its Cholesky decomposition
     *   A = L * L^T exists with strictly positive diagonal entries on L. This is
     *   equivalent to Sylvester's criterion (every leading principal minor is
     *   positive) but **much cheaper to evaluate**: a partial Cholesky factor
     *   built up to depth k passes if and only if the first k leading principal
     *   minors are positive.
     *
     *   Algorithm (partial Cholesky):
     *       for i = 0 .. depth-1:
     *           sum = sum_{k<i} L(i,k)^2
     *           diag = A(i,i) - sum                  // == det of (i+1)x(i+1) leading minor
     *                                                 //    divided by det of i x i leading minor
     *           if diag <= tolerance: not PD, return false
     *           L(i,i) = sqrt(diag)
     *           for j = i+1 .. depth-1:
     *               L(j,i) = (A(j,i) - sum_{k<i} L(j,k)*L(i,k)) / L(i,i)
     *
     *   Complexity: O(depth^3 / 3), versus the previous O(n^4) Sylvester+Laplace
     *   path. Storage: depth^2 floats (one scratch lower-triangular L).
     *
     * @param tolerance Lower bound for each Cholesky pivot diag (must be >= 0).
     *                  A pivot <= tolerance is treated as a failure (i.e. A is
     *                  not strictly PD at that level).
     * @param max_minors_to_check Depth of the leading principal block to test.
     *                            - If < 0: test the whole matrix (depth = n).
     *                            - If == 0: invalid; returns false with an error.
     *                            - If > 0: clamped to [1, n]; only the first
     *                              `max_minors_to_check` leading principal minors
     *                              are tested (early-exit semantics).
     *
     * @return true if every tested leading principal minor is positive.
     */
    bool Mat::is_positive_definite(float tolerance, int max_minors_to_check) const
    {
        // ---------------------------------------------------------------------
        // 1. Validate input
        // ---------------------------------------------------------------------
        if (this->data == nullptr)
        {
            std::cerr << "[Error] is_positive_definite: matrix data pointer is null.\n";
            return false;
        }
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] is_positive_definite: invalid matrix dimensions: "
                      << this->row << "x" << this->col << ".\n";
            return false;
        }
        if (tolerance < 0.0f)
        {
            std::cerr << "[Error] is_positive_definite: tolerance must be >= 0 (got "
                      << tolerance << ").\n";
            return false;
        }
        if (this->row != this->col)
        {
            return false;
        }
        if (!this->is_symmetric(tolerance))
        {
            return false;
        }

        const int n = this->row;
        if (n == 0)
        {
            return true; // empty matrix trivially PD
        }

        // ---------------------------------------------------------------------
        // 2. Resolve check depth
        // ---------------------------------------------------------------------
        int depth;
        if (max_minors_to_check < 0)
        {
            depth = n;
        }
        else if (max_minors_to_check == 0)
        {
            std::cerr << "[Error] is_positive_definite: max_minors_to_check must be > 0 or -1 (got 0).\n";
            return false;
        }
        else
        {
            depth = (max_minors_to_check > n) ? n : max_minors_to_check;
        }

        // ---------------------------------------------------------------------
        // 3. Partial Cholesky to depth
        //    L is a (depth x depth) lower-triangular working buffer. We store
        //    only the lower triangle, which is the same memory layout the
        //    Cholesky-factor convention uses.
        // ---------------------------------------------------------------------
        Mat L(depth, depth);
        if (L.data == nullptr)
        {
            std::cerr << "[Error] is_positive_definite: failed to allocate scratch L.\n";
            return false;
        }

        const int Lstep = L.step;
        const int Astep = this->step;

        for (int i = 0; i < depth; ++i)
        {
            float* row_Li = L.data + i * Lstep;
            const float* row_Ai = this->data + i * Astep;

            // Diagonal: L(i,i) = sqrt(A(i,i) - sum_{k<i} L(i,k)^2)
            float diag_sum = 0.0f;
            for (int k = 0; k < i; ++k)
            {
                diag_sum += row_Li[k] * row_Li[k];
            }
            const float diag = row_Ai[i] - diag_sum;
            if (diag <= tolerance || std::isnan(diag) || std::isinf(diag))
            {
                return false; // i-th leading principal minor not PD
            }
            const float pivot = sqrtf(diag);
            row_Li[i] = pivot;
            const float inv_pivot = 1.0f / pivot;

            // Below-diagonal column i: L(j,i) = (A(j,i) - <Lj,Li>) / L(i,i)
            for (int j = i + 1; j < depth; ++j)
            {
                float* row_Lj = L.data + j * Lstep;
                const float* row_Aj = this->data + j * Astep;

                float off_sum = 0.0f;
                for (int k = 0; k < i; ++k)
                {
                    off_sum += row_Lj[k] * row_Li[k];
                }
                row_Lj[i] = (row_Aj[i] - off_sum) * inv_pivot;
            }
        }

        return true;
    }

    /**
     * @name Mat::lu_decompose()
     * @brief Compute LU decomposition: A = L * U (with optional pivoting).
     * @note LU decomposition factors a matrix A into:
     *       - L: Lower triangular matrix (with unit diagonal)
     *       - U: Upper triangular matrix
     *       - P: Permutation matrix (if pivoting used)
     * @note With pivoting: P * A = L * U
     *       Without pivoting: A = L * U
     * @note Algorithm: Modified Gaussian elimination that stores multipliers in L.
     *       Uses partial pivoting for numerical stability.
     * @note Time complexity: O(n³) - efficient for large matrices.
     *       More efficient than Gaussian elimination when solving multiple systems
     *       with the same coefficient matrix.
     * 
     * @param use_pivoting Whether to use partial pivoting (default: true).
     *                     Pivoting improves numerical stability but requires P matrix.
     * @return LUDecomposition containing L, U, P matrices and status
     */
    Mat::LUDecomposition Mat::lu_decompose(bool use_pivoting) const
    {
        LUDecomposition result;

        // ---------------------------------------------------------------------
        // 1. Validate input
        // ---------------------------------------------------------------------
        if (this->data == nullptr)
        {
            std::cerr << "[Error] lu_decompose: matrix data pointer is null.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] lu_decompose: invalid matrix dimensions: "
                      << this->row << "x" << this->col << ".\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        if (this->row != this->col)
        {
            std::cerr << "[Error] lu_decompose: requires a square matrix (got "
                      << this->row << "x" << this->col << ").\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        const int n = this->row;
        result.pivoted = use_pivoting;

        if (n == 0)
        {
            result.L = Mat(0, 0);
            result.U = Mat(0, 0);
            if (use_pivoting) result.P = Mat(0, 0);
            result.status = TINY_OK;
            return result;
        }

        // ---------------------------------------------------------------------
        // 2. Allocate working / output buffers
        //    A : working copy that ends as packed L|U after the loop
        //    L : initialized to I, fills lower-triangular multipliers
        //    U : initialized to 0, fills upper-triangular at the end
        //    P : (optional) permutation matrix, init to I
        // ---------------------------------------------------------------------
        Mat A(*this);
        if (A.data == nullptr)
        {
            std::cerr << "[Error] lu_decompose: failed to allocate working copy.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        result.L = Mat::eye(n);
        result.U = Mat(n, n);
        if (result.L.data == nullptr || result.U.data == nullptr)
        {
            std::cerr << "[Error] lu_decompose: failed to allocate L or U.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        if (use_pivoting)
        {
            result.P = Mat::eye(n);
            if (result.P.data == nullptr)
            {
                std::cerr << "[Error] lu_decompose: failed to allocate P.\n";
                result.status = TINY_ERR_MATH_NULL_POINTER;
                return result;
            }
        }

        const int Astep = A.step;
        const int Lstep = result.L.step;
        float* const Ldata = result.L.data;
        float* const Adata = A.data;

        // ---------------------------------------------------------------------
        // 3. LU with optional partial pivoting (Doolittle: unit-diagonal L)
        // ---------------------------------------------------------------------
        for (int k = 0; k < n; ++k)
        {
            if (use_pivoting)
            {
                // 3.1 Find row with the largest |A(i, k)| for i in [k, n)
                int   max_row = k;
                float max_val = fabsf(Adata[k * Astep + k]);
                for (int i = k + 1; i < n; ++i)
                {
                    const float abs_val = fabsf(Adata[i * Astep + k]);
                    if (abs_val > max_val)
                    {
                        max_val = abs_val;
                        max_row = i;
                    }
                }

                // 3.2 Swap rows in A, in P, and in the already-written part of L
                if (max_row != k)
                {
                    A.swap_rows(k, max_row);
                    result.P.swap_rows(k, max_row);
                    float* row_Lk = Ldata + k       * Lstep;
                    float* row_Lm = Ldata + max_row * Lstep;
                    for (int j = 0; j < k; ++j)
                    {
                        const float t = row_Lk[j];
                        row_Lk[j] = row_Lm[j];
                        row_Lm[j] = t;
                    }
                }
            }

            float* row_Ak = Adata + k * Astep;

            // 3.3 Singularity guard
            const float pivot = row_Ak[k];
            if (fabsf(pivot) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] lu_decompose: matrix is singular or near-singular at column "
                          << k << " (pivot = " << pivot << ").\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }
            const float inv_pivot = 1.0f / pivot;

            // 3.4 Copy U row k from A
            float* row_Uk = result.U.data + k * result.U.step;
            for (int j = k; j < n; ++j)
            {
                row_Uk[j] = row_Ak[j];
            }

            // 3.5 Eliminate below the pivot, store multipliers in L(i, k)
            for (int i = k + 1; i < n; ++i)
            {
                float* row_Ai = Adata + i * Astep;
                const float multiplier = row_Ai[k] * inv_pivot;
                Ldata[i * Lstep + k] = multiplier;
                if (multiplier == 0.0f) continue; // already eliminated
                for (int j = k + 1; j < n; ++j)
                {
                    row_Ai[j] -= multiplier * row_Ak[j];
                }
            }
        }

        result.status = TINY_OK;
        return result;
    }

    /**
     * @name Mat::cholesky_decompose()
     * @brief Compute Cholesky decomposition: A = L * L^T (for symmetric positive definite matrices).
     * @note Cholesky decomposition factors a symmetric positive definite matrix A into:
     *       A = L * L^T
     *       where L is a lower triangular matrix with positive diagonal elements.
     * @note Algorithm: For each row i and column j (j <= i):
     *       - Diagonal (j == i): L[i][i] = sqrt(A[i][i] - sum(L[i][k]^2))
     *       - Off-diagonal (j < i): L[i][j] = (A[i][j] - sum(L[i][k]*L[j][k])) / L[j][j]
     * @note Requirements:
     *       - Matrix must be square
     *       - Matrix must be symmetric: A^T = A
     *       - Matrix must be positive definite: all eigenvalues > 0
     * @note Advantages over LU decomposition:
     *       - Faster: O(n³/3) vs O(n³) for LU
     *       - More stable: no pivoting needed
     *       - Uses less memory: only stores L (not L and U)
     * @note Applications: Structural dynamics, optimization, Kalman filtering, Monte Carlo simulation
     * @note Time complexity: O(n³/3) - about 2x faster than LU decomposition
     * 
     * @return CholeskyDecomposition containing L matrix and status
     */
    Mat::CholeskyDecomposition Mat::cholesky_decompose() const
    {
        CholeskyDecomposition result;

        // Check for null pointer
        if (this->data == nullptr)
        {
            std::cerr << "[Error] cholesky_decompose: matrix data pointer is null\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        
        // Validate matrix dimensions
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] cholesky_decompose: invalid matrix dimensions: rows=" 
                      << this->row << ", cols=" << this->col << "\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        // Validation: must be square matrix
        if (this->row != this->col)
        {
            std::cerr << "[Error] cholesky_decompose: requires square matrix (got " 
                      << this->row << "x" << this->col << ")\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        int n = this->row;
        
        // Handle empty matrix
        if (n == 0)
        {
            // Empty matrix: L is also empty
            result.L = Mat(0, 0);
            result.status = TINY_OK;
            return result;
        }

        // Check if symmetric (use standard tolerance)
        if (!this->is_symmetric(TINY_MATH_MIN_POSITIVE_INPUT_F32))
        {
            std::cerr << "[Error] cholesky_decompose: requires symmetric matrix\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        result.L = Mat(n, n);
        if (result.L.data == nullptr)
        {
            std::cerr << "[Error] cholesky_decompose: failed to allocate L matrix.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }

        // ---------------------------------------------------------------------
        // Cholesky factorization (column-by-column, packed lower triangular).
        //
        //   For column j running 0 .. n-1:
        //       diag = A(j,j) - <L(j, 0:j), L(j, 0:j)>
        //       if diag <= eps  -> A is not positive definite
        //       L(j,j)  = sqrt(diag)
        //       inv_d  = 1 / L(j,j)
        //       for i in j+1 .. n-1:
        //           L(i,j) = (A(i,j) - <L(i, 0:j), L(j, 0:j)>) * inv_d
        //
        // Strict lower triangle is the only part written; the upper triangle
        // stays zero from Mat(n, n).  Hot inner loops use raw row pointers.
        // ---------------------------------------------------------------------
        const int Lstep = result.L.step;
        const int Astep = this->step;
        float* const Ldata = result.L.data;
        const float* const Adata = this->data;

        for (int j = 0; j < n; ++j)
        {
            float* row_Lj = Ldata + j * Lstep;
            const float* row_Aj = Adata + j * Astep;

            // Diagonal pivot
            float diag_sum = 0.0f;
            for (int k = 0; k < j; ++k)
            {
                diag_sum += row_Lj[k] * row_Lj[k];
            }
            const float diag = row_Aj[j] - diag_sum;
            if (diag <= TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] cholesky_decompose: matrix is not positive definite "
                          << "(diagonal residual " << diag << " at position ["
                          << j << "][" << j << "]).\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }
            const float pivot = sqrtf(diag);
            row_Lj[j] = pivot;
            const float inv_pivot = 1.0f / pivot;

            // Below-diagonal entries in column j
            for (int i = j + 1; i < n; ++i)
            {
                float* row_Li = Ldata + i * Lstep;
                const float* row_Ai = Adata + i * Astep;

                float off_sum = 0.0f;
                for (int k = 0; k < j; ++k)
                {
                    off_sum += row_Li[k] * row_Lj[k];
                }
                row_Li[j] = (row_Ai[j] - off_sum) * inv_pivot;
            }
        }

        result.status = TINY_OK;
        return result;
    }

    /**
     * @name Mat::qr_decompose()
     * @brief Compute QR decomposition: A = Q * R (Q orthogonal, R upper triangular).
     * @note QR decomposition factors a matrix A into:
     *       A = Q * R
     *       where:
     *       - Q: Orthogonal matrix (Q^T * Q = I, columns are orthonormal)
     *       - R: Upper triangular matrix
     * @note Algorithm: Uses Modified Gram-Schmidt process for orthogonalization.
     *       The Gram-Schmidt process orthogonalizes the columns of A to form Q,
     *       and stores the projection coefficients in R.
     * @note Mathematical relationship:
     *       - Q is m×min(m,n) matrix with orthonormal columns
     *       - R is min(m,n)×n upper triangular matrix
     *       - For m >= n: A = Q * R (full QR)
     *       - For m < n: A = Q * R (reduced QR, Q is m×m, R is m×n)
     * @note Applications:
     *       - Least squares problems: min ||A*x - b|| → solve R*x = Q^T*b
     *       - Solving overdetermined systems
     *       - Eigenvalue computation (QR algorithm)
     *       - Matrix rank estimation
     * @note Time complexity: O(m*n²) - efficient for tall matrices (m >> n)
     * @note Numerical stability: Modified Gram-Schmidt is more stable than classical GS
     * 
     * @return QRDecomposition containing Q and R matrices and status
     */
    Mat::QRDecomposition Mat::qr_decompose() const
    {
        QRDecomposition result;

        // ---------------------------------------------------------------------
        // 1. Validate input
        // ---------------------------------------------------------------------
        if (this->data == nullptr)
        {
            std::cerr << "[Error] qr_decompose: matrix data pointer is null.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] qr_decompose: invalid matrix dimensions: "
                      << this->row << "x" << this->col << ".\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        const int m = this->row;
        const int n = this->col;
        const int min_dim = (m < n) ? m : n;

        if (m == 0 || n == 0)
        {
            result.Q = Mat(0, 0);
            result.R = Mat(0, 0);
            result.status = TINY_OK;
            return result;
        }

        // ---------------------------------------------------------------------
        // 2. Run Modified Gram-Schmidt (reuses the optimized helper)
        //    R_coeff is already a fully populated n x n upper-triangular block:
        //      - For k <= j: R_coeff(k, j) is the projection r_kj computed by MGS
        //      - For k >  j: zero (kept by gram_schmidt_orthogonalize)
        //    Therefore there is no need to recompute Q^T * A here.
        // ---------------------------------------------------------------------
        Mat Q_ortho, R_coeff;
        if (!Mat::gram_schmidt_orthogonalize(*this, Q_ortho, R_coeff, TINY_MATH_MIN_POSITIVE_INPUT_F32))
        {
            std::cerr << "[Error] qr_decompose: gram_schmidt_orthogonalize failed.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        if (Q_ortho.data == nullptr || R_coeff.data == nullptr)
        {
            std::cerr << "[Error] qr_decompose: gram_schmidt_orthogonalize returned null buffers.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        if (Q_ortho.row != m || Q_ortho.col != n ||
            R_coeff.row != n || R_coeff.col != n)
        {
            std::cerr << "[Error] qr_decompose: unexpected dimensions from gram_schmidt_orthogonalize.\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        // ---------------------------------------------------------------------
        // 3. Materialize Q and R in the QR convention
        //    R is sized (m x n) so the bottom (m - min_dim) rows are zero.
        //    The top min_dim rows are exactly R_coeff's upper-triangular block,
        //    plus, for wide matrices (m < n), the extra columns j >= m which
        //    MGS already filled (r_kj = q_k^T * a_j for k < m).
        // ---------------------------------------------------------------------
        result.Q = Q_ortho;
        result.R = Mat(m, n);
        if (result.R.data == nullptr)
        {
            std::cerr << "[Error] qr_decompose: failed to allocate R matrix.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }

        const int Rstep = result.R.step;
        const int Cstep = R_coeff.step;
        for (int j = 0; j < n; ++j)
        {
            const int k_end = (j + 1 < min_dim) ? (j + 1) : min_dim;
            float* col_R = result.R.data + j;
            const float* col_C = R_coeff.data + j;
            for (int k = 0; k < k_end; ++k)
            {
                col_R[k * Rstep] = col_C[k * Cstep];
            }
            // remaining rows in this column stay zero (Mat ctor zero-inits)
        }

        result.status = TINY_OK;
        return result;
    }

    /**
     * @name Mat::svd_decompose()
     * @brief Compute the (thin/economy) Singular Value Decomposition: A = U * S * V^T.
     *
     * Output dimensions:
     *   - U: m x min(m,n)  — orthonormal columns (left singular vectors)
     *   - S: min(m,n) x 1  — singular values, sorted in *descending* order
     *   - V: n x n         — orthogonal matrix; first `rank` columns are the
     *                        right singular vectors corresponding to the non-zero
     *                        singular values (sorted), remaining columns form an
     *                        orthonormal basis of the null space of A.
     *
     * Algorithm (normal-equations / Jacobi):
     *   1. Form M = A^T A  (n x n, symmetric positive-semidefinite)
     *   2. eig(M) = V diag(lambda) V^T  via Jacobi
     *   3. sigma_i = sqrt(lambda_i)     (lambda_i is clamped at 0 to absorb tiny
     *                                    negative roundoff)
     *   4. Sort (sigma_i, V(:, i)) by sigma_i descending
     *   5. U(:, i) = A * V(:, i) / sigma_i, for i with sigma_i > tolerance
     *      (Note: orthogonality of U columns relies on Jacobi's accuracy on
     *       symmetric matrices; small floating-point drift may exist when
     *       several sigma_i are nearly equal. If strict U^T U == I is needed,
     *       apply gram_schmidt_orthogonalize() on the result externally.)
     *
     * Numerical caveat:
     *   Forming A^T A squares the condition number of A, so singular values
     *   smaller than ~sqrt(eps)*sigma_max are unreliable. For accuracy near the
     *   rank cliff, prefer a bidiagonalization-based SVD (not implemented here).
     *
     * Tolerance semantics:
     *   `tolerance` is used both as the Jacobi off-diagonal convergence threshold
     *   and as the *singular-value* zero threshold (sigma <= tolerance is treated
     *   as zero). It is therefore in the *same scale as the entries of A*. This
     *   matches the convention used by `pseudo_inverse(svd, tolerance)`.
     *
     * Properties guaranteed:
     *   - sigma_i >= 0, sorted descending
     *   - rank   = number of sigma_i strictly greater than tolerance
     *   - U columns 0..rank-1 are orthonormal; columns rank..min_dim-1 are zero
     *   - V is fully populated (n columns) and is orthogonal up to floating-point
     *     drift inherited from Jacobi.
     *
     * Time complexity: O(m*n^2 + n^3) — dominated by Jacobi on the n x n matrix.
     *
     * @param max_iter   Maximum Jacobi iterations (must be > 0).
     * @param tolerance  Convergence threshold (Jacobi) and singular-value zero
     *                   threshold (must be >= 0).
     * @return SVDDecomposition with U, S, V, rank, iterations, status.
     */
    Mat::SVDDecomposition Mat::svd_decompose(int max_iter, float tolerance) const
    {
        SVDDecomposition result;

        // ---------------------------------------------------------------------
        // 1. Validate input
        // ---------------------------------------------------------------------
        if (this->data == nullptr)
        {
            std::cerr << "[Error] svd_decompose: matrix data pointer is null.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] svd_decompose: invalid matrix dimensions: "
                      << this->row << "x" << this->col << ".\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        if (max_iter <= 0)
        {
            std::cerr << "[Error] svd_decompose: max_iter must be > 0 (got " << max_iter << ").\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        if (tolerance < 0.0f)
        {
            std::cerr << "[Error] svd_decompose: tolerance must be >= 0 (got " << tolerance << ").\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        const int m = this->row;
        const int n = this->col;
        const int min_dim = (m < n) ? m : n;
        // Note: m == 0 || n == 0 cases are already rejected by the dimension
        // check above (this->row <= 0 || this->col <= 0).

        // ---------------------------------------------------------------------
        // 2. Form M = A^T * A (n x n, symmetric PSD).
        //    Compute the upper triangle, then mirror, so that M is *exactly*
        //    symmetric (Jacobi assumes symmetry).
        // ---------------------------------------------------------------------
        Mat AtA(n, n);
        if (AtA.data == nullptr)
        {
            std::cerr << "[Error] svd_decompose: failed to allocate A^T*A.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }

        const int Astep = this->step;
        const float* const Adata = this->data;
        const int AtAstep = AtA.step;

        for (int i = 0; i < n; ++i)
        {
            float* row_AtA_i = AtA.data + i * AtAstep;
            for (int j = i; j < n; ++j)
            {
                float s = 0.0f;
                for (int k = 0; k < m; ++k)
                {
                    const float* row_Ak = Adata + k * Astep;
                    s += row_Ak[i] * row_Ak[j];
                }
                row_AtA_i[j] = s;
                if (j != i)
                {
                    AtA.data[j * AtAstep + i] = s;
                }
            }
        }

        // ---------------------------------------------------------------------
        // 3. Eigendecomposition of A^T A.
        //    eigenvectors are columns of `eig.eigenvectors`; eigenvalues are
        //    delivered in *unsorted* (natural diagonal) order.
        // ---------------------------------------------------------------------
        Mat::EigenDecomposition eig = AtA.eigendecompose_jacobi(tolerance, max_iter);
        if (eig.status != TINY_OK)
        {
            std::cerr << "[Error] svd_decompose: eigendecomposition failed with status "
                      << eig.status << ".\n";
            result.status = eig.status;
            return result;
        }
        if (eig.eigenvalues.data == nullptr || eig.eigenvectors.data == nullptr)
        {
            std::cerr << "[Error] svd_decompose: eigendecomposition returned null buffers.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        if (eig.eigenvalues.row != n || eig.eigenvalues.col != 1 ||
            eig.eigenvectors.row != n || eig.eigenvectors.col != n)
        {
            std::cerr << "[Error] svd_decompose: unexpected dimensions from eigendecomposition.\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        // ---------------------------------------------------------------------
        // 4. Build a permutation that sorts eigenvalues in descending order.
        //    Negative roundoff on PSD spectra is clamped to 0 here so that the
        //    sqrt is always well defined.
        // ---------------------------------------------------------------------
        const int evalStep = eig.eigenvalues.step;
        const int EVstep   = eig.eigenvectors.step;

        std::vector<float> lambda(n);
        std::vector<int>   perm(n);
        for (int i = 0; i < n; ++i)
        {
            float lam = eig.eigenvalues.data[i * evalStep];
            if (std::isnan(lam) || lam < 0.0f) lam = 0.0f;   // clamp PSD roundoff
            lambda[i] = lam;
            perm[i] = i;
        }
        // Insertion sort by lambda descending — n is typically small.
        for (int i = 1; i < n; ++i)
        {
            const int   key_idx = perm[i];
            const float key_lam = lambda[i];
            int j = i - 1;
            while (j >= 0 && lambda[j] < key_lam)
            {
                lambda[j + 1] = lambda[j];
                perm[j + 1]   = perm[j];
                --j;
            }
            lambda[j + 1] = key_lam;
            perm[j + 1]   = key_idx;
        }

        // ---------------------------------------------------------------------
        // 5. Allocate outputs and populate V (full n x n) and S (min_dim x 1).
        //    V's columns are *all* eigenvectors of A^T A in sorted order, so V
        //    is an orthogonal matrix (the trailing columns span the null space
        //    of A and have sigma_i = 0).
        //    rank counts only sigma_i > tolerance.
        // ---------------------------------------------------------------------
        result.S = Mat(min_dim, 1);
        result.V = Mat(n, n);
        result.U = Mat(m, min_dim);
        if (result.S.data == nullptr || result.V.data == nullptr || result.U.data == nullptr)
        {
            std::cerr << "[Error] svd_decompose: failed to allocate U/S/V.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }

        const int Vstep = result.V.step;
        const int Sstep = result.S.step;
        const int Ustep = result.U.step;

        // Populate every column of V (n columns total) using the sort permutation.
        for (int col = 0; col < n; ++col)
        {
            const int src = perm[col];
            for (int row_idx = 0; row_idx < n; ++row_idx)
            {
                result.V.data[row_idx * Vstep + col] =
                    eig.eigenvectors.data[row_idx * EVstep + src];
            }
        }

        // Fill S; count rank as sigma_i > tolerance (NOT >= — treat == tolerance
        // as numerically zero, matching pseudo_inverse).
        int rank = 0;
        for (int i = 0; i < min_dim; ++i)
        {
            const float sigma = sqrtf(lambda[i]); // lambda already clamped >= 0
            result.S.data[i * Sstep] = sigma;
            if (sigma > tolerance) ++rank;
        }
        result.rank = rank;

        // ---------------------------------------------------------------------
        // 6. Recover U from A * V = U * S, column by column:
        //        U(:, i) = (A * V(:, i)) / sigma_i,   for sigma_i > tolerance
        //    Columns rank..min_dim-1 are left at 0 (zero-initialized).
        //
        //    In exact arithmetic ||A * V_i|| == sigma_i, so U(:, i) is unit
        //    norm. Across i, U columns are orthogonal because V columns are
        //    (Jacobi guarantees orthonormal eigenvectors of A^T A). Floating-
        //    point drift may slightly violate U^T U == I; downstream consumers
        //    that need strict orthogonality should re-orthogonalize on the
        //    result.
        // ---------------------------------------------------------------------
        for (int i = 0; i < rank; ++i)
        {
            const float sigma = result.S.data[i * Sstep];
            const float inv_sigma = 1.0f / sigma;
            for (int j = 0; j < m; ++j)
            {
                const float* row_Aj = Adata + j * Astep;
                float s = 0.0f;
                for (int k = 0; k < n; ++k)
                {
                    s += row_Aj[k] * result.V.data[k * Vstep + i];
                }
                result.U.data[j * Ustep + i] = s * inv_sigma;
            }
        }

        result.iterations = eig.iterations;
        result.status = TINY_OK;
        return result;
    }

    /**
     * @name Mat::solve_lu()
     * @brief Solve linear system using LU decomposition (more efficient for multiple RHS).
     * @note Solves A * x = b using precomputed LU decomposition.
     *       Algorithm: P * A = L * U, so A * x = b becomes:
     *       1. Apply permutation: P * A * x = P * b → (L * U) * x = P * b
     *       2. Forward substitution: L * y = P * b → solve for y
     *       3. Backward substitution: U * x = y → solve for x
     * @note Advantages over direct solve():
     *       - More efficient when solving multiple systems with same A
     *       - LU decomposition computed once, reused for different b vectors
     *       - Time complexity: O(n²) vs O(n³) for direct solve
     * @note Requirements:
     *       - LU decomposition must be valid (status == TINY_OK)
     *       - Matrix A must be square and non-singular
     *       - Vector b must have same dimension as A
     * 
     * @param lu LU decomposition of coefficient matrix A
     * @param b Right-hand side vector (must be column vector)
     * @return Solution vector x such that A * x = b, or empty matrix on error
     */
    Mat Mat::solve_lu(const LUDecomposition &lu, const Mat &b)
    {
        // ---------------------------------------------------------------------
        // 1. Validate input
        // ---------------------------------------------------------------------
        if (lu.status != TINY_OK)
        {
            std::cerr << "[Error] solve_lu: invalid LU decomposition (status="
                      << lu.status << ").\n";
            return Mat(0, 0);
        }
        if (lu.L.data == nullptr || lu.U.data == nullptr)
        {
            std::cerr << "[Error] solve_lu: LU has null L or U buffer.\n";
            return Mat(0, 0);
        }
        if (b.data == nullptr)
        {
            std::cerr << "[Error] solve_lu: right-hand side b is null.\n";
            return Mat(0, 0);
        }
        if (lu.L.row != lu.L.col || lu.U.row != lu.U.col || lu.L.row != lu.U.row)
        {
            std::cerr << "[Error] solve_lu: L and U must be square and same-sized "
                      << "(got L=" << lu.L.row << "x" << lu.L.col
                      << ", U=" << lu.U.row << "x" << lu.U.col << ").\n";
            return Mat(0, 0);
        }

        const int n = lu.L.row;

        // Legitimate empty system: 0 equations -> trivial 0x1 solution.
        if (n == 0)
        {
            return Mat(0, 1);
        }

        if (b.row != n || b.col != 1)
        {
            std::cerr << "[Error] solve_lu: b must be " << n << "x1 (got "
                      << b.row << "x" << b.col << ").\n";
            return Mat(0, 0);
        }
        if (lu.pivoted && (lu.P.data == nullptr || lu.P.row != n || lu.P.col != n))
        {
            std::cerr << "[Error] solve_lu: pivoting enabled but P is invalid "
                      << "(got " << lu.P.row << "x" << lu.P.col << ").\n";
            return Mat(0, 0);
        }

        // ---------------------------------------------------------------------
        // 2. Build b_perm = P * b (or just a copy of b if no pivoting)
        //    P has exactly one 1.0 per row; we scan each row to find it.
        //    This is O(n^2) but only runs once and avoids mutating b.
        // ---------------------------------------------------------------------
        Mat b_perm(n, 1);
        if (b_perm.data == nullptr)
        {
            std::cerr << "[Error] solve_lu: failed to allocate permuted RHS.\n";
            return Mat(0, 0);
        }
        const int b_step = b.step;
        const int bp_step = b_perm.step;
        if (lu.pivoted)
        {
            const int P_step = lu.P.step;
            for (int i = 0; i < n; ++i)
            {
                const float* row_P = lu.P.data + i * P_step;
                int src = -1;
                for (int j = 0; j < n; ++j)
                {
                    if (fabsf(row_P[j] - 1.0f) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                    {
                        src = j;
                        break;
                    }
                }
                b_perm.data[i * bp_step] =
                    (src >= 0) ? b.data[src * b_step] : 0.0f;
            }
        }
        else
        {
            for (int i = 0; i < n; ++i)
            {
                b_perm.data[i * bp_step] = b.data[i * b_step];
            }
        }

        // ---------------------------------------------------------------------
        // 3. Forward substitution: L * y = b_perm
        //    L is unit-diagonal (Doolittle), so no divisions are needed.
        //    y is stored in-place by reusing b_perm: y(i) = b_perm(i) - sum_{j<i} L(i,j) y(j)
        // ---------------------------------------------------------------------
        const int L_step = lu.L.step;
        for (int i = 0; i < n; ++i)
        {
            const float* row_L = lu.L.data + i * L_step;
            float s = b_perm.data[i * bp_step];
            for (int j = 0; j < i; ++j)
            {
                s -= row_L[j] * b_perm.data[j * bp_step];
            }
            b_perm.data[i * bp_step] = s; // == y(i)
        }

        // ---------------------------------------------------------------------
        // 4. Backward substitution: U * x = y
        // ---------------------------------------------------------------------
        Mat x(n, 1);
        if (x.data == nullptr)
        {
            std::cerr << "[Error] solve_lu: failed to allocate solution vector.\n";
            return Mat(0, 0);
        }
        const int U_step = lu.U.step;
        const int x_step = x.step;
        for (int i = n - 1; i >= 0; --i)
        {
            const float* row_U = lu.U.data + i * U_step;
            const float pivot = row_U[i];
            if (fabsf(pivot) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] solve_lu: singular U at index " << i
                          << " (pivot=" << pivot << ").\n";
                return Mat(0, 0);
            }
            float s = b_perm.data[i * bp_step]; // y(i)
            for (int j = i + 1; j < n; ++j)
            {
                s -= row_U[j] * x.data[j * x_step];
            }
            x.data[i * x_step] = s / pivot;
        }

        return x;
    }

    /**
     * @name Mat::solve_cholesky()
     * @brief Solve linear system using Cholesky decomposition (for SPD matrices).
     * @note Solves A * x = b using precomputed Cholesky decomposition.
     *       Algorithm: A = L * L^T, so A * x = b becomes:
     *       1. Forward substitution: L * y = b → solve for y
     *       2. Backward substitution: L^T * x = y → solve for x
     * @note Advantages over LU decomposition:
     *       - Faster: O(n²) vs O(n²) but with better numerical stability
     *       - More stable: no pivoting needed for SPD matrices
     *       - Uses less memory: only stores L (not L and U)
     *       - Guaranteed to work for positive definite matrices
     * @note Requirements:
     *       - Cholesky decomposition must be valid (status == TINY_OK)
     *       - Matrix A must be symmetric positive definite (SPD)
     *       - Vector b must have same dimension as A
     * @note Time complexity: O(n²) - efficient for SPD systems
     * 
     * @param chol Cholesky decomposition of coefficient matrix A (A = L * L^T)
     * @param b Right-hand side vector (must be column vector)
     * @return Solution vector x such that A * x = b, or empty matrix on error
     */
    Mat Mat::solve_cholesky(const CholeskyDecomposition &chol, const Mat &b)
    {
        // ---------------------------------------------------------------------
        // 1. Validate input
        // ---------------------------------------------------------------------
        if (chol.status != TINY_OK)
        {
            std::cerr << "[Error] solve_cholesky: invalid decomposition (status="
                      << chol.status << ").\n";
            return Mat(0, 0);
        }
        if (chol.L.data == nullptr)
        {
            std::cerr << "[Error] solve_cholesky: L is null.\n";
            return Mat(0, 0);
        }
        if (b.data == nullptr)
        {
            std::cerr << "[Error] solve_cholesky: right-hand side b is null.\n";
            return Mat(0, 0);
        }
        if (chol.L.row != chol.L.col)
        {
            std::cerr << "[Error] solve_cholesky: L must be square (got "
                      << chol.L.row << "x" << chol.L.col << ").\n";
            return Mat(0, 0);
        }

        const int n = chol.L.row;

        if (n == 0)
        {
            return Mat(0, 1);
        }
        if (b.row != n || b.col != 1)
        {
            std::cerr << "[Error] solve_cholesky: b must be " << n << "x1 (got "
                      << b.row << "x" << b.col << ").\n";
            return Mat(0, 0);
        }

        // ---------------------------------------------------------------------
        // 2. Forward substitution: L * y = b
        //    L lower-triangular with positive diagonal (guaranteed by
        //    cholesky_decompose). We allocate one (n x 1) buffer and reuse it
        //    in-place: it starts as b, becomes y, then becomes x.
        // ---------------------------------------------------------------------
        Mat x(n, 1);
        if (x.data == nullptr)
        {
            std::cerr << "[Error] solve_cholesky: failed to allocate solution.\n";
            return Mat(0, 0);
        }
        const int b_step = b.step;
        const int x_step = x.step;
        for (int i = 0; i < n; ++i)
        {
            x.data[i * x_step] = b.data[i * b_step];
        }

        const int L_step = chol.L.step;
        for (int i = 0; i < n; ++i)
        {
            const float* row_L = chol.L.data + i * L_step;
            const float pivot = row_L[i];
            if (fabsf(pivot) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                std::cerr << "[Error] solve_cholesky: zero/near-zero L diagonal at "
                          << i << " (= " << pivot << ").\n";
                return Mat(0, 0);
            }
            float s = x.data[i * x_step];
            for (int j = 0; j < i; ++j)
            {
                s -= row_L[j] * x.data[j * x_step];
            }
            x.data[i * x_step] = s / pivot; // == y(i)
        }

        // ---------------------------------------------------------------------
        // 3. Backward substitution: L^T * x = y
        //    L^T(i, j) = L(j, i), so we walk *column* i of L below the diagonal,
        //    which is non-contiguous. We use the row+stride access pattern.
        // ---------------------------------------------------------------------
        for (int i = n - 1; i >= 0; --i)
        {
            const float pivot = chol.L.data[i * L_step + i];
            float s = x.data[i * x_step]; // == y(i)
            for (int j = i + 1; j < n; ++j)
            {
                s -= chol.L.data[j * L_step + i] * x.data[j * x_step];
            }
            x.data[i * x_step] = s / pivot;
        }

        return x;
    }

    /**
     * @name Mat::solve_qr()
     * @brief Solve linear system using QR decomposition (least squares solution).
     * @note Solves A * x = b using precomputed QR decomposition.
     *       Algorithm: A = Q * R, so A * x = b becomes:
     *       1. Compute Q^T * b (project b onto column space of Q)
     *       2. Solve R * x = Q^T * b using backward substitution
     * @note This method provides least squares solution:
     *       - For overdetermined systems (m > n): minimizes ||A * x - b||
     *       - For determined systems (m = n): exact solution
     *       - For underdetermined systems (m < n): minimum norm solution
     * @note Advantages:
     *       - Numerically stable (no pivoting needed)
     *       - Works for rectangular matrices
     *       - Handles rank-deficient matrices gracefully
     * @note Requirements:
     *       - QR decomposition must be valid (status == TINY_OK)
     *       - Vector b must have same number of rows as Q
     * @note Time complexity: O(m*n) - efficient for least squares problems
     * 
     * @param qr QR decomposition of coefficient matrix A (A = Q * R)
     * @param b Right-hand side vector (must be column vector)
     * @return Least squares solution vector x such that ||A * x - b|| is minimized,
     *         or empty matrix on error
     */
    Mat Mat::solve_qr(const QRDecomposition &qr, const Mat &b)
    {
        // ---------------------------------------------------------------------
        // 1. Validate input
        // ---------------------------------------------------------------------
        if (qr.status != TINY_OK)
        {
            std::cerr << "[Error] solve_qr: invalid QR decomposition (status="
                      << qr.status << ").\n";
            return Mat(0, 0);
        }
        if (qr.Q.data == nullptr || qr.R.data == nullptr)
        {
            std::cerr << "[Error] solve_qr: Q or R is null.\n";
            return Mat(0, 0);
        }
        if (b.data == nullptr)
        {
            std::cerr << "[Error] solve_qr: right-hand side b is null.\n";
            return Mat(0, 0);
        }
        if (qr.Q.row <= 0 || qr.Q.col <= 0 || qr.R.row <= 0 || qr.R.col <= 0)
        {
            std::cerr << "[Error] solve_qr: invalid Q/R dimensions.\n";
            return Mat(0, 0);
        }

        const int m = qr.Q.row;
        const int n = qr.R.col;
        const int min_dim = (m < n) ? m : n;

        // Q is expected to be m x n and R is m x n (qr_decompose convention).
        // We only require enough capacity to address Q(:, 0..min_dim-1) and
        // R(0..min_dim-1, :), so check that conservatively.
        if (qr.Q.col < min_dim || qr.R.row < min_dim)
        {
            std::cerr << "[Error] solve_qr: Q/R shape inconsistent (Q="
                      << qr.Q.row << "x" << qr.Q.col
                      << ", R=" << qr.R.row << "x" << qr.R.col << ").\n";
            return Mat(0, 0);
        }

        // Legitimate empty system: 0 equations or 0 unknowns.
        if (m == 0 || n == 0)
        {
            return Mat(n, 1);
        }

        if (b.row != m || b.col != 1)
        {
            std::cerr << "[Error] solve_qr: b must be " << m << "x1 (got "
                      << b.row << "x" << b.col << ").\n";
            return Mat(0, 0);
        }

        // ---------------------------------------------------------------------
        // 2. Compute c = Q^T * b
        //    Q is m x n; only the first min_dim columns are guaranteed to be
        //    a valid orthonormal basis for col(A). The remaining components
        //    are 0 (left at the value set by the zero-initialized constructor).
        //
        //    Inner-product accumulation walks Q column-wise, which is the
        //    cache-unfriendly direction for row-major storage. We still hoist
        //    the row pointers to avoid re-multiplying the stride per element.
        // ---------------------------------------------------------------------
        Mat c(n, 1); // x will reuse this buffer after backward substitution
        if (c.data == nullptr)
        {
            std::cerr << "[Error] solve_qr: failed to allocate working vector.\n";
            return Mat(0, 0);
        }
        const int Q_step = qr.Q.step;
        const int b_step = b.step;
        const int c_step = c.step;
        for (int i = 0; i < min_dim; ++i)
        {
            float s = 0.0f;
            for (int j = 0; j < m; ++j)
            {
                s += qr.Q.data[j * Q_step + i] * b.data[j * b_step];
            }
            c.data[i * c_step] = s;
        }
        // c[min_dim .. n-1] left at 0 by Mat(n,1) zero-init.

        // ---------------------------------------------------------------------
        // 3. Backward substitution: R x = c
        //    R is upper-triangular (m >= n) or upper-trapezoidal (m < n).
        //    We only solve the first min_dim rows; for rank-deficient or
        //    under-determined cases (R(i,i) ~ 0), set x(i)=0 (minimum-norm-ish).
        //    Final x[min_dim..n-1] are 0 (zero-init), giving the basic least
        //    squares solution that lives in the span of the leading min_dim cols.
        // ---------------------------------------------------------------------
        Mat x(n, 1);
        if (x.data == nullptr)
        {
            std::cerr << "[Error] solve_qr: failed to allocate solution vector.\n";
            return Mat(0, 0);
        }
        const int R_step = qr.R.step;
        const int x_step = x.step;
        for (int i = min_dim - 1; i >= 0; --i)
        {
            const float* row_R = qr.R.data + i * R_step;
            const float pivot = row_R[i];
            if (fabsf(pivot) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                // Rank deficient: take 0 as the i-th unknown (already zero).
                continue;
            }
            float s = c.data[i * c_step];
            for (int j = i + 1; j < n; ++j)
            {
                s -= row_R[j] * x.data[j * x_step];
            }
            x.data[i * x_step] = s / pivot;
        }

        return x;
    }

    /**
     * @name Mat::pseudo_inverse()
     * @brief Compute pseudo-inverse using SVD: A^+ = V * S^+ * U^T.
     * @note Pseudo-inverse (Moore-Penrose inverse) is the generalization of matrix inverse
     *       for rectangular or singular matrices.
     * @note Algorithm: A = U * S * V^T, so A^+ = V * S^+ * U^T
     *       where S^+ is the pseudo-inverse of S (diagonal matrix):
     *       - If σ_i > tolerance: S^+[i][i] = 1/σ_i
     *       - If σ_i <= tolerance: S^+[i][i] = 0 (treat as zero)
     * @note Mathematical properties:
     *       - A * A^+ * A = A
     *       - A^+ * A * A^+ = A^+
     *       - (A * A^+)^T = A * A^+
     *       - (A^+ * A)^T = A^+ * A
     * @note Applications:
     *       - Solving least squares problems: x = A^+ * b
     *       - Solving underdetermined systems (minimum norm solution)
     *       - Solving overdetermined systems (least squares solution)
     *       - Rank-deficient matrix inversion
     * @note Time complexity: O(m*n*rank) - dominated by matrix multiplications
     * 
     * @param svd SVD decomposition of matrix A (A = U * S * V^T)
     * @param tolerance Tolerance for singular values (must be >= 0).
     *                  Singular values <= tolerance are treated as zero.
     * @return Pseudo-inverse matrix A^+ (n×m matrix), or empty matrix on error
     */
    Mat Mat::pseudo_inverse(const SVDDecomposition &svd, float tolerance)
    {
        if (svd.status != TINY_OK)
        {
            std::cerr << "[Error] pseudo_inverse: invalid SVD decomposition (status: "
                      << svd.status << ")\n";
            return Mat(0, 0);
        }
        if (svd.U.data == nullptr || svd.V.data == nullptr || svd.S.data == nullptr)
        {
            std::cerr << "[Error] pseudo_inverse: SVD decomposition matrices have null pointers.\n";
            return Mat(0, 0);
        }
        if (tolerance < 0.0f)
        {
            std::cerr << "[Error] pseudo_inverse: tolerance must be >= 0 (got "
                      << tolerance << ").\n";
            return Mat(0, 0);
        }
        if (svd.U.row <= 0 || svd.U.col <= 0 ||
            svd.V.row <= 0 || svd.V.col <= 0 ||
            svd.S.row <= 0 || svd.S.col != 1)
        {
            std::cerr << "[Error] pseudo_inverse: invalid SVD decomposition dimensions.\n";
            return Mat(0, 0);
        }

        const int m = svd.U.row;       // A has m rows
        const int n = svd.V.row;       // A has n columns
        const int min_dim = svd.S.row; // S is min(m,n) x 1

        // Validate SVD shape consistency with svd_decompose() contract.
        if (min_dim != ((m < n) ? m : n) || svd.U.col < min_dim || svd.V.col < min_dim)
        {
            std::cerr << "[Error] pseudo_inverse: inconsistent SVD dimensions.\n";
            return Mat(0, 0);
        }
        if (svd.rank < 0 || svd.rank > min_dim)
        {
            std::cerr << "[Error] pseudo_inverse: invalid rank " << svd.rank
                      << " (expected 0 to " << min_dim << ").\n";
            return Mat(0, 0);
        }

        // Precompute active reciprocal singular values based on this call's tolerance.
        std::vector<int> active_idx;
        std::vector<float> active_inv_sigma;
        active_idx.reserve(min_dim);
        active_inv_sigma.reserve(min_dim);
        for (int k = 0; k < min_dim; ++k)
        {
            const float sigma = svd.S(k, 0);
            if (!std::isfinite(sigma))
            {
                std::cerr << "[Error] pseudo_inverse: non-finite singular value at index "
                          << k << ".\n";
                return Mat(0, 0);
            }
            if (sigma > tolerance)
            {
                active_idx.push_back(k);
                active_inv_sigma.push_back(1.0f / sigma);
            }
        }

        Mat A_plus(n, m);
        if (A_plus.data == nullptr)
        {
            std::cerr << "[Error] pseudo_inverse: failed to allocate result matrix.\n";
            return Mat(0, 0);
        }

        const int AplusStep = A_plus.step;
        const int Ustep = svd.U.step;
        const int Vstep = svd.V.step;

        // Explicitly zero-initialize in case Mat constructor does not.
        for (int i = 0; i < n; ++i)
        {
            float *row_out = A_plus.data + i * AplusStep;
            for (int j = 0; j < m; ++j) row_out[j] = 0.0f;
        }

        // A^+ = sum_k V(:,k) * (1/sigma_k) * U(:,k)^T, over sigma_k > tolerance.
        // Loop order (k -> i -> j) improves reuse of V(i,k)/sigma_k.
        for (size_t t = 0; t < active_idx.size(); ++t)
        {
            const int k = active_idx[t];
            const float inv_sigma = active_inv_sigma[t];
            for (int i = 0; i < n; ++i)
            {
                float *row_out = A_plus.data + i * AplusStep;
                const float vik_scaled = svd.V.data[i * Vstep + k] * inv_sigma;
                const float *u_col_base = svd.U.data + k; // U(j,k) = u_col_base[j*Ustep]
                for (int j = 0; j < m; ++j)
                {
                    row_out[j] += vik_scaled * u_col_base[j * Ustep];
                }
            }
        }

        return A_plus;
    }

    // ============================================================================
    // Eigenvalue & Eigenvector Decomposition
    // ============================================================================
    /**
     * @name Mat::EigenPair::EigenPair()
     * @brief Default constructor for EigenPair structure
     */
    Mat::EigenPair::EigenPair() : eigenvalue(0.0f), iterations(0), status(TINY_OK)
    {
    }

    /**
     * @name Mat::EigenDecomposition::EigenDecomposition()
     * @brief Default constructor for EigenDecomposition structure
     */
    Mat::EigenDecomposition::EigenDecomposition() : iterations(0), status(TINY_OK)
    {
    }

    /**
     * @name Mat::is_symmetric()
     * @brief Check whether A^T = A within a tolerance.
     *
     * @note Semantics:
     *       - Non-square            -> false (silent; not a bug, just not symmetric)
     *       - 0x0                   -> true  (vacuously symmetric)
     *       - Negative dimensions   -> false (logged: invalid input)
     *       - Null data on a sized
     *         matrix                -> false (logged: invalid input)
     *       - NaN element           -> false (NaN never compares <= tolerance,
     *                                  so it falls through naturally)
     * @note Hot loop hoists the row pointer; the (j, i) probe is column-strided
     *       (unavoidable without an explicit transpose).
     * @note Time complexity: O(n^2) (upper triangle only).
     *
     * @param tolerance Max allowed |A(i,j) - A(j,i)|; must be >= 0.
     * @return true if symmetric within tolerance.
     */
    bool Mat::is_symmetric(float tolerance) const
    {
        if (this->data == nullptr && (this->row > 0 && this->col > 0))
        {
            std::cerr << "[Error] is_symmetric: matrix data pointer is null.\n";
            return false;
        }
        if (this->row < 0 || this->col < 0)
        {
            std::cerr << "[Error] is_symmetric: invalid matrix dimensions: "
                      << this->row << "x" << this->col << ".\n";
            return false;
        }
        if (tolerance < 0.0f)
        {
            std::cerr << "[Error] is_symmetric: tolerance must be >= 0 (got "
                      << tolerance << ").\n";
            return false;
        }
        if (this->row != this->col)
        {
            return false;
        }

        const int n = this->row;
        if (n == 0)
        {
            return true; // 0x0 is trivially symmetric
        }

        // Hot loop: hoist row pointer for row i; column-direction access for
        // (j, i) is strided but unavoidable without a transpose.
        // Using `!(diff <= tolerance)` causes NaN to fall through to false.
        const int   stride = this->step;
        const float* const base = this->data;
        for (int i = 0; i < n; ++i)
        {
            const float* row_i = base + i * stride;
            for (int j = i + 1; j < n; ++j)
            {
                const float diff = fabsf(row_i[j] - base[j * stride + i]);
                if (!(diff <= tolerance))
                {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * @name Mat::power_iteration()
     * @brief Compute the dominant (largest magnitude) eigenvalue and eigenvector using power iteration.
     * @note Power iteration finds the eigenvalue with the largest absolute value and its corresponding eigenvector.
     *       Algorithm: v_{k+1} = A * v_k / ||A * v_k||, converges to dominant eigenvector.
     * @note Fast method suitable for real-time SHM applications to quickly identify primary frequency.
     * @note Time complexity: O(n² * iterations) - efficient for sparse or large matrices
     * @note Convergence: Requires that the dominant eigenvalue is unique and has larger magnitude than others.
     * 
     * @param max_iter Maximum number of iterations (must be > 0)
     * @param tolerance Convergence tolerance (must be >= 0). Convergence when |λ_k - λ_{k-1}| < tolerance * |λ_k|
     * @return EigenPair containing the dominant eigenvalue, eigenvector, and status
     */
    Mat::EigenPair Mat::power_iteration(int max_iter, float tolerance) const
    {
        EigenPair result;

        // -----------------------------------------------------------------
        // 1. Validate input
        // -----------------------------------------------------------------
        if (this->data == nullptr)
        {
            std::cerr << "[Error] power_iteration: matrix data pointer is null.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] power_iteration: invalid matrix dimensions: "
                      << this->row << "x" << this->col << ".\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        if (this->row != this->col)
        {
            std::cerr << "[Error] power_iteration: requires square matrix (got "
                      << this->row << "x" << this->col << ").\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        if (max_iter <= 0)
        {
            std::cerr << "[Error] power_iteration: max_iter must be > 0 (got "
                      << max_iter << ").\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        if (tolerance < 0.0f)
        {
            std::cerr << "[Error] power_iteration: tolerance must be >= 0 (got "
                      << tolerance << ").\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        const int n = this->row;
        if (n == 0)
        {
            result.eigenvalue  = 0.0f;
            result.eigenvector = Mat(0, 1);
            result.iterations  = 0;
            result.status      = TINY_OK;
            return result;
        }

        // -----------------------------------------------------------------
        // 2. Allocate working buffers (eigenvector + scratch).
        // -----------------------------------------------------------------
        result.eigenvector = Mat(n, 1);
        Mat temp_vec(n, 1);
        if (result.eigenvector.data == nullptr || temp_vec.data == nullptr)
        {
            std::cerr << "[Error] power_iteration: failed to allocate buffers.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }

        // -----------------------------------------------------------------
        // 3. Initialize the seed vector.
        //    We bias by the column-1-norms so directions with more "mass"
        //    in A start with a larger projection along that axis. The +1.0f
        //    floor guarantees a positive result, so the well-known fallback
        //    branch is unreachable; a uniform [1..1] start is fine for any
        //    realistic matrix.
        // -----------------------------------------------------------------
        const int Astep = this->step;
        const int Vstep = result.eigenvector.step;
        const int Tstep = temp_vec.step;
        const float* const Adata = this->data;
        float* const Vdata = result.eigenvector.data;
        float* const Tdata = temp_vec.data;

        {
            float norm_sq = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                float col_sum = 0.0f;
                for (int j = 0; j < n; ++j)
                    col_sum += fabsf(Adata[j * Astep + i]);
                const float v = col_sum + 1.0f;
                Vdata[i * Vstep] = v;
                norm_sq += v * v;
            }
            const float sqrt_norm = sqrtf(norm_sq);
            if (!(sqrt_norm > TINY_MATH_MIN_POSITIVE_INPUT_F32))
            {
                std::cerr << "[Error] power_iteration: invalid initial eigenvector norm.\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }
            const float inv_norm = 1.0f / sqrt_norm;
            for (int i = 0; i < n; ++i)
                Vdata[i * Vstep] *= inv_norm;
        }

        // -----------------------------------------------------------------
        // 4. Power iteration loop.
        //    Invariant: at the top of each iteration, ||v||_2 == 1, so the
        //    Rayleigh quotient denominator (v^T v) is exactly 1 and we skip
        //    computing it. This saves n FLOPs per iteration.
        //    Order each iteration:
        //       temp = A * v
        //       lambda = v^T * temp        (denominator = 1)
        //       v <- temp / ||temp||       (re-establish unit-norm invariant)
        // -----------------------------------------------------------------
        float prev_eigenvalue = 0.0f;
        for (int iter = 0; iter < max_iter; ++iter)
        {
            // 4.1 temp = A * v
            for (int i = 0; i < n; ++i)
            {
                const float* row_Ai = Adata + i * Astep;
                float sum = 0.0f;
                for (int j = 0; j < n; ++j)
                    sum += row_Ai[j] * Vdata[j * Vstep];
                Tdata[i * Tstep] = sum;
            }

            // 4.2 lambda = v^T * temp  (||v|| == 1 at this point)
            float lambda = 0.0f;
            float new_norm_sq = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                const float vi = Vdata[i * Vstep];
                const float ti = Tdata[i * Tstep];
                lambda      += vi * ti;
                new_norm_sq += ti * ti;
            }

            if (!(new_norm_sq > TINY_MATH_MIN_POSITIVE_INPUT_F32))
            {
                std::cerr << "[Error] power_iteration: matrix-vector product collapsed to zero.\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }
            result.eigenvalue = lambda;

            // 4.3 v = temp / ||temp||
            const float inv_norm = 1.0f / sqrtf(new_norm_sq);
            for (int i = 0; i < n; ++i)
                Vdata[i * Vstep] = Tdata[i * Tstep] * inv_norm;

            // 4.4 Convergence check.
            //     Use absolute change for tiny eigenvalues, relative otherwise.
            if (iter > 0)
            {
                const float change = fabsf(result.eigenvalue - prev_eigenvalue);
                const float abs_l  = fabsf(result.eigenvalue);
                const float thresh = (abs_l < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                                     ? tolerance
                                     : tolerance * abs_l;
                if (change < thresh)
                {
                    result.iterations = iter + 1;
                    result.status     = TINY_OK;
                    return result;
                }
            }
            prev_eigenvalue = result.eigenvalue;
        }

        result.iterations = max_iter;
        result.status     = TINY_ERR_NOT_FINISHED;
        std::cerr << "[Warning] power_iteration: did not converge within "
                  << max_iter << " iterations.\n";
        return result;
    }

    /**
     * @name Mat::inverse_power_iteration()
     * @brief Compute the smallest (minimum magnitude) eigenvalue and eigenvector using inverse power iteration.
     * @note Inverse power iteration finds the eigenvalue with the smallest absolute value and its eigenvector.
     *       Algorithm: v_{k+1} = A^(-1) * v_k / ||A^(-1) * v_k||, converges to smallest eigenvector.
     * @note Critical for system identification - finds fundamental frequency/lowest mode in structural dynamics.
     *       This method is essential for SHM applications where the smallest eigenvalue corresponds to the
     *       fundamental frequency of the system.
     * @note Time complexity: O(n³ * iterations) - each iteration requires solving a linear system
     * @note The matrix must be invertible (non-singular) for this method to work.
     *       If the matrix is singular or near-singular, the method will fail gracefully.
     * 
     * @param max_iter Maximum number of iterations (must be > 0)
     * @param tolerance Convergence tolerance (must be >= 0). Convergence when |λ_k - λ_{k-1}| < tolerance * max(|λ_k|, 1)
     * @return EigenPair containing the smallest eigenvalue, eigenvector, and status
     */
    Mat::EigenPair Mat::inverse_power_iteration(int max_iter, float tolerance) const
    {
        EigenPair result;

        // -----------------------------------------------------------------
        // 1. Validate input
        // -----------------------------------------------------------------
        if (this->data == nullptr)
        {
            std::cerr << "[Error] inverse_power_iteration: matrix data pointer is null.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] inverse_power_iteration: invalid matrix dimensions: "
                      << this->row << "x" << this->col << ".\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        if (this->row != this->col)
        {
            std::cerr << "[Error] inverse_power_iteration: requires square matrix (got "
                      << this->row << "x" << this->col << ").\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        if (max_iter <= 0)
        {
            std::cerr << "[Error] inverse_power_iteration: max_iter must be > 0 (got "
                      << max_iter << ").\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        if (tolerance < 0.0f)
        {
            std::cerr << "[Error] inverse_power_iteration: tolerance must be >= 0 (got "
                      << tolerance << ").\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        const int n = this->row;
        if (n == 0)
        {
            result.eigenvalue  = 0.0f;
            result.eigenvector = Mat(0, 1);
            result.iterations  = 0;
            result.status      = TINY_OK;
            return result;
        }

        // -----------------------------------------------------------------
        // 2. Factorise A once (PA = LU). Each iteration then solves the
        //    linear system in O(n^2) instead of O(n^3) -- this is the whole
        //    point of inverse power iteration in practice.
        // -----------------------------------------------------------------
        LUDecomposition lu = this->lu_decompose(/*use_pivoting=*/true);
        if (lu.status != TINY_OK)
        {
            std::cerr << "[Error] inverse_power_iteration: matrix is singular or near-singular "
                      << "(LU status=" << lu.status << ").\n";
            result.status = TINY_ERR_MATH_INVALID_PARAM;
            return result;
        }

        // -----------------------------------------------------------------
        // 3. Allocate eigenvector buffer and seed it with an alternating
        //    +/-1 pattern to break alignment with the dominant eigenvector
        //    direction; small perturbation keeps it generic.
        // -----------------------------------------------------------------
        result.eigenvector = Mat(n, 1);
        if (result.eigenvector.data == nullptr)
        {
            std::cerr << "[Error] inverse_power_iteration: failed to allocate eigenvector.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }

        const int Vstep = result.eigenvector.step;
        float* const Vdata = result.eigenvector.data;
        {
            float norm_sq = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                const float v = ((i & 1) == 0 ? 1.0f : -1.0f)
                                + 0.1f * static_cast<float>(i) / static_cast<float>(n);
                Vdata[i * Vstep] = v;
                norm_sq += v * v;
            }
            const float sqrt_norm = sqrtf(norm_sq);
            if (!(sqrt_norm > TINY_MATH_MIN_POSITIVE_INPUT_F32))
            {
                std::cerr << "[Error] inverse_power_iteration: invalid initial eigenvector norm.\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }
            const float inv_norm = 1.0f / sqrt_norm;
            for (int i = 0; i < n; ++i)
                Vdata[i * Vstep] *= inv_norm;
        }

        // -----------------------------------------------------------------
        // 4. Iterate.
        //
        //    Per iteration we want the smallest-magnitude eigenvalue of A,
        //    which is the largest-magnitude eigenvalue of A^{-1}. The
        //    classical recurrence is:
        //        y      = A^{-1} v        (here: solve_lu(lu, v))
        //        mu     = v^T y           (Rayleigh quotient on A^{-1};
        //                                  v^T v == 1 by invariant)
        //        v_next = y / ||y||       (re-normalise)
        //        lambda = 1 / mu          (eigenvalue of A)
        //
        //    This avoids the explicit second matvec A * v that the previous
        //    implementation used, saving ~n^2 FLOPs per iteration.
        // -----------------------------------------------------------------
        float prev_eigenvalue = 0.0f;
        for (int iter = 0; iter < max_iter; ++iter)
        {
            // 4.1 y = A^{-1} v  via reusable LU.
            Mat y = Mat::solve_lu(lu, result.eigenvector);
            if (y.data == nullptr || y.row != n || y.col != 1)
            {
                std::cerr << "[Error] inverse_power_iteration: solve_lu failed at iter "
                          << iter << ".\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }

            // 4.2 Rayleigh quotient on A^{-1} and ||y||^2 in one sweep.
            const int Ystep = y.step;
            const float* const Ydata = y.data;

            float mu = 0.0f;          // v^T y
            float norm_sq = 0.0f;     // y^T y
            for (int i = 0; i < n; ++i)
            {
                const float vi = Vdata[i * Vstep];
                const float yi = Ydata[i * Ystep];
                mu      += vi * yi;
                norm_sq += yi * yi;
            }

            if (!(norm_sq > TINY_MATH_MIN_POSITIVE_INPUT_F32))
            {
                std::cerr << "[Error] inverse_power_iteration: solve produced a zero vector.\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }
            if (fabsf(mu) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
            {
                // mu = v^T A^{-1} v ~ 0 => smallest |eigenvalue| of A is huge
                // (or v happens to be orthogonal to the smallest mode). In
                // either case, 1/mu would overflow; abort gracefully.
                std::cerr << "[Error] inverse_power_iteration: Rayleigh quotient on A^{-1} "
                          << "underflowed (eigenvalue would diverge).\n";
                result.status = TINY_ERR_MATH_INVALID_PARAM;
                return result;
            }
            result.eigenvalue = 1.0f / mu;

            // 4.3 v <- y / ||y||
            const float inv_norm = 1.0f / sqrtf(norm_sq);
            for (int i = 0; i < n; ++i)
                Vdata[i * Vstep] = Ydata[i * Ystep] * inv_norm;

            // 4.4 Convergence: relative change with a unit floor for tiny
            //     eigenvalues. Matches the previous semantics.
            if (iter > 0)
            {
                const float change   = fabsf(result.eigenvalue - prev_eigenvalue);
                const float rel_tol  = tolerance * fmaxf(fabsf(result.eigenvalue), 1.0f);
                if (change < rel_tol)
                {
                    result.iterations = iter + 1;
                    result.status     = TINY_OK;
                    return result;
                }
            }
            prev_eigenvalue = result.eigenvalue;
        }

        result.iterations = max_iter;
        result.status     = TINY_ERR_NOT_FINISHED;
        std::cerr << "[Warning] inverse_power_iteration: did not converge within "
                  << max_iter << " iterations.\n";
        return result;
    }

    /**
     * @name Mat::eigendecompose_jacobi()
     * @brief Eigendecomposition of a symmetric matrix via classical Jacobi rotations.
     *
     * Algorithm:
     *   1. Symmetrize input: A := (A + A^T) / 2  (defends against tiny FP asymmetry)
     *   2. Iterate:
     *        - Find largest off-diagonal |A(p,q)|.
     *        - Build a Givens rotation that zeros A(p,q) (and A(q,p) by symmetry).
     *        - Apply the rotation to A from both sides; accumulate it into V.
     *      Stop when max |A(p,q)| < tolerance, or after `max_iter` sweeps.
     *
     * @note Best for small/medium symmetric matrices (n < ~100). For SHM
     *       structural matrices this is ideal.
     * @note Convergence is tested with an absolute threshold; callers that
     *       need scale-invariance should pre-scale `tolerance` accordingly.
     * @note Time complexity: O(n^3 * iterations); convergence is quadratic
     *       near the solution, so iteration count is typically O(n^2).
     *
     * @param tolerance  Off-diagonal magnitude below which we stop (>= 0).
     * @param max_iter   Maximum sweeps (must be > 0).
     * @return EigenDecomposition with eigenvalues (n x 1), eigenvectors
     *         (n x n, columns are unit eigenvectors), iteration count, status.
     */
    Mat::EigenDecomposition Mat::eigendecompose_jacobi(float tolerance, int max_iter) const
    {
        EigenDecomposition result;

        // -----------------------------------------------------------------
        // 1. Validate input
        // -----------------------------------------------------------------
        if (this->data == nullptr)
        {
            std::cerr << "[Error] eigendecompose_jacobi: matrix data pointer is null.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] eigendecompose_jacobi: invalid matrix dimensions: "
                      << this->row << "x" << this->col << ".\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        if (this->row != this->col)
        {
            std::cerr << "[Error] eigendecompose_jacobi: requires square matrix (got "
                      << this->row << "x" << this->col << ").\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        if (tolerance < 0.0f)
        {
            std::cerr << "[Error] eigendecompose_jacobi: tolerance must be >= 0 (got "
                      << tolerance << ").\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        if (max_iter <= 0)
        {
            std::cerr << "[Error] eigendecompose_jacobi: max_iter must be > 0 (got "
                      << max_iter << ").\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        const int n = this->row;

        // -----------------------------------------------------------------
        // 2. Allocate outputs once
        // -----------------------------------------------------------------
        result.eigenvalues  = Mat(n, 1);
        result.eigenvectors = Mat::eye(n);
        if (n > 0 && (result.eigenvalues.data == nullptr ||
                      result.eigenvectors.data == nullptr))
        {
            std::cerr << "[Error] eigendecompose_jacobi: failed to allocate outputs.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        if (n == 0)
        {
            result.iterations = 0;
            result.status = TINY_OK;
            return result;
        }

        // -----------------------------------------------------------------
        // 3. Working copy with explicit symmetrization: A := (T + T^T) / 2.
        //    Doing this here lets callers pass slightly non-symmetric inputs
        //    safely, and removes the need for a separate is_symmetric() probe
        //    whose tolerance is hard to set scale-invariantly.
        // -----------------------------------------------------------------
        Mat A(n, n);
        if (A.data == nullptr)
        {
            std::cerr << "[Error] eigendecompose_jacobi: failed to allocate working copy.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        {
            const int Astep_init = A.step;
            const int Tstep      = this->step;
            float* const Adata_init = A.data;
            const float* const Tdata = this->data;
            for (int i = 0; i < n; ++i)
            {
                float* row_Ai = Adata_init + i * Astep_init;
                const float* row_Ti = Tdata + i * Tstep;
                row_Ai[i] = row_Ti[i];
                for (int j = i + 1; j < n; ++j)
                {
                    const float v = 0.5f * (row_Ti[j] + Tdata[j * Tstep + i]);
                    row_Ai[j] = v;
                    Adata_init[j * Astep_init + i] = v;
                }
            }
        }

        // -----------------------------------------------------------------
        // 4. Jacobi sweeps (classical: pick the largest off-diagonal entry).
        //    Hot loops use hoisted row pointers to skip operator() overhead.
        // -----------------------------------------------------------------
        const int Astep = A.step;
        const int Vstep = result.eigenvectors.step;
        float* const Adata = A.data;
        float* const Vdata = result.eigenvectors.data;

        bool converged  = false;
        int  iter_count = max_iter;

        for (int iter = 0; iter < max_iter; ++iter)
        {
            // 4.1 Locate the largest |A(p,q)| with p < q.
            float max_off = 0.0f;
            int   p = 0, q = 1;
            for (int i = 0; i < n - 1; ++i)
            {
                const float* row_Ai = Adata + i * Astep;
                for (int j = i + 1; j < n; ++j)
                {
                    const float a = fabsf(row_Ai[j]);
                    if (a > max_off)
                    {
                        max_off = a;
                        p = i;
                        q = j;
                    }
                }
            }

            // 4.2 Convergence check.
            if (max_off < tolerance)
            {
                converged  = true;
                iter_count = iter + 1;
                break;
            }

            // 4.3 Compute rotation (c, s) that zeros A(p,q).
            //     The branch on tau picks the smaller root of the quadratic
            //     and so keeps |t| <= 1, avoiding cancellation.
            float* const row_Ap = Adata + p * Astep;
            float* const row_Aq = Adata + q * Astep;
            const float app = row_Ap[p];
            const float aqq = row_Aq[q];
            const float apq = row_Ap[q];

            const float tau = (aqq - app) / (2.0f * apq);
            const float t = (tau >= 0.0f)
                            ?  1.0f / ( tau + sqrtf(1.0f + tau * tau))
                            : -1.0f / (-tau + sqrtf(1.0f + tau * tau));
            const float c = 1.0f / sqrtf(1.0f + t * t);
            const float s = t * c;

            // 4.4 Apply rotation to rows/cols (j, p) and (j, q), j != p, q.
            for (int j = 0; j < n; ++j)
            {
                if (j == p || j == q) continue;
                float* const row_Aj = Adata + j * Astep;
                const float apj = row_Ap[j];
                const float aqj = row_Aq[j];
                const float new_apj = c * apj - s * aqj;
                const float new_aqj = s * apj + c * aqj;
                row_Ap[j] = new_apj;
                row_Aq[j] = new_aqj;
                row_Aj[p] = new_apj; // maintain symmetry
                row_Aj[q] = new_aqj;
            }

            // 4.5 Update the (p,p), (q,q), (p,q), (q,p) block.
            row_Ap[p] = c * c * app - 2.0f * c * s * apq + s * s * aqq;
            row_Aq[q] = s * s * app + 2.0f * c * s * apq + c * c * aqq;
            row_Ap[q] = 0.0f;
            row_Aq[p] = 0.0f;

            // 4.6 Accumulate the rotation into V: V := V * G.
            for (int i = 0; i < n; ++i)
            {
                float* const row_Vi = Vdata + i * Vstep;
                const float vip = row_Vi[p];
                const float viq = row_Vi[q];
                row_Vi[p] = c * vip - s * viq;
                row_Vi[q] = s * vip + c * viq;
            }
        }

        // -----------------------------------------------------------------
        // 5. Read eigenvalues off the diagonal (single allocation, single pass).
        // -----------------------------------------------------------------
        const int evStep = result.eigenvalues.step;
        for (int i = 0; i < n; ++i)
        {
            result.eigenvalues.data[i * evStep] = Adata[i * Astep + i];
        }

        result.iterations = iter_count;
        if (converged)
        {
            result.status = TINY_OK;
        }
        else
        {
            result.status = TINY_ERR_NOT_FINISHED;
            std::cerr << "[Warning] eigendecompose_jacobi: did not converge within "
                      << max_iter << " iterations.\n";
        }
        return result;
    }

    /**
     * @name Mat::eigendecompose_qr()
     * @brief Unshifted QR algorithm for general (non-symmetric) real matrices.
     *
     * Algorithm (per iteration):
     *   1. A_k = Q_k * R_k         (via Modified Gram-Schmidt; R is full upper)
     *   2. A_{k+1} = R_k * Q_k     (similar transform; preserves spectrum)
     *   3. V_{k+1} = V_k * Q_k     (accumulates eigenvectors)
     *
     * Convergence:
     *   Stop when every sub-diagonal entry is below `tolerance` scaled by the
     *   neighbouring diagonal magnitudes (with a 1.0 floor for tiny matrices).
     *
     * Caveats:
     *   - No shifts and no Hessenberg reduction. Convergence is linear in the
     *     general case; for fast convergence on real-world matrices add a
     *     Wilkinson shift in a future revision.
     *   - Pairs of complex eigenvalues will leave 2x2 blocks on the diagonal
     *     and will *not* converge. Only the real part of the diagonal is
     *     reported, with `status = TINY_ERR_NOT_FINISHED`.
     *
     * Performance:
     *   - Two-buffer ping-pong avoids the per-iteration n*n allocations of
     *     A_new / V_new that the previous implementation did.
     *   - The R*Q product exploits R's upper-triangular structure
     *     (k starts at i), halving its FLOPs on average.
     *
     * @param max_iter  Max QR sweeps (must be > 0).
     * @param tolerance Sub-diagonal convergence threshold (must be >= 0).
     * @return EigenDecomposition with diagonal eigenvalues, accumulated V,
     *         iteration count, and status.
     */
    Mat::EigenDecomposition Mat::eigendecompose_qr(int max_iter, float tolerance) const
    {
        EigenDecomposition result;

        // -----------------------------------------------------------------
        // 1. Validate input
        // -----------------------------------------------------------------
        if (this->data == nullptr)
        {
            std::cerr << "[Error] eigendecompose_qr: matrix data pointer is null.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        if (this->row <= 0 || this->col <= 0)
        {
            std::cerr << "[Error] eigendecompose_qr: invalid matrix dimensions: "
                      << this->row << "x" << this->col << ".\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        if (this->row != this->col)
        {
            std::cerr << "[Error] eigendecompose_qr: requires square matrix (got "
                      << this->row << "x" << this->col << ").\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        if (max_iter <= 0)
        {
            std::cerr << "[Error] eigendecompose_qr: max_iter must be > 0 (got "
                      << max_iter << ").\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        if (tolerance < 0.0f)
        {
            std::cerr << "[Error] eigendecompose_qr: tolerance must be >= 0 (got "
                      << tolerance << ").\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        const int n = this->row;

        // -----------------------------------------------------------------
        // 2. Allocate eigenvalues output (filled at the end). Empty matrix
        //    short-circuits with valid-empty results.
        // -----------------------------------------------------------------
        result.eigenvalues = Mat(n, 1);
        if (n > 0 && result.eigenvalues.data == nullptr)
        {
            std::cerr << "[Error] eigendecompose_qr: failed to allocate eigenvalues.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        if (n == 0)
        {
            result.eigenvectors = Mat(0, 0);
            result.iterations   = 0;
            result.status       = TINY_OK;
            return result;
        }

        // -----------------------------------------------------------------
        // 3. Two-buffer ping-pong for A and V, so each iteration only swaps
        //    pointers instead of allocating fresh n*n matrices.
        // -----------------------------------------------------------------
        Mat A_a(*this);          // working copy, drifts toward upper triangular
        Mat A_b(n, n);
        Mat V_a = Mat::eye(n);   // accumulated eigenvectors
        Mat V_b(n, n);
        if (A_a.data == nullptr || A_b.data == nullptr ||
            V_a.data == nullptr || V_b.data == nullptr)
        {
            std::cerr << "[Error] eigendecompose_qr: failed to allocate working buffers.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        Mat *A_cur = &A_a, *A_nxt = &A_b;
        Mat *V_cur = &V_a, *V_nxt = &V_b;

        // gram_schmidt_orthogonalize() reallocates these on every call, so
        // declare them once and let the inner function manage memory.
        Mat Q, R;

        bool converged  = false;
        int  iter_count = max_iter;

        for (int iter = 0; iter < max_iter; ++iter)
        {
            // 3.1 Convergence: every sub-diagonal entry below the (relative)
            //     tolerance. Bail as soon as one entry fails the test.
            {
                bool not_converged = false;
                const int Astep = A_cur->step;
                const float* const Adata = A_cur->data;
                for (int i = 1; i < n && !not_converged; ++i)
                {
                    const float* row_Ai = Adata + i * Astep;
                    for (int j = 0; j < i; ++j)
                    {
                        const float abs_val  = fabsf(row_Ai[j]);
                        const float diag_scl = fmaxf(fabsf(Adata[i * Astep + i]),
                                                     fabsf(Adata[j * Astep + j]));
                        const float rel_tol  = tolerance * fmaxf(1.0f, diag_scl);
                        if (abs_val > rel_tol)
                        {
                            not_converged = true;
                            break;
                        }
                    }
                }
                if (!not_converged)
                {
                    converged  = true;
                    iter_count = iter;
                    break;
                }
            }

            // 3.2 A = Q * R via Modified Gram-Schmidt.
            //     R is delivered as a *full* upper-triangular matrix, so we
            //     can use it directly with no patch-up step.
            if (!Mat::gram_schmidt_orthogonalize(*A_cur, Q, R,
                                                 TINY_MATH_MIN_POSITIVE_INPUT_F32))
            {
                std::cerr << "[Error] eigendecompose_qr: Gram-Schmidt failed.\n";
                result.status = TINY_ERR_MATH_NULL_POINTER;
                return result;
            }

            // 3.3 A_nxt = R * Q  (R is upper-triangular, so k starts at i)
            {
                const int Rstep  = R.step;
                const int Qstep  = Q.step;
                const int Anstep = A_nxt->step;
                const float* const Rdata = R.data;
                const float* const Qdata = Q.data;
                float* const       Andata = A_nxt->data;
                for (int i = 0; i < n; ++i)
                {
                    const float* row_Ri = Rdata + i * Rstep;
                    float* row_Ani = Andata + i * Anstep;
                    for (int j = 0; j < n; ++j)
                    {
                        float sum = 0.0f;
                        for (int k = i; k < n; ++k) // R(i,k) = 0 for k < i
                            sum += row_Ri[k] * Qdata[k * Qstep + j];
                        row_Ani[j] = sum;
                    }
                }
            }

            // 3.4 V_nxt = V_cur * Q
            {
                const int Vcstep = V_cur->step;
                const int Vnstep = V_nxt->step;
                const int Qstep  = Q.step;
                const float* const Vcdata = V_cur->data;
                const float* const Qdata  = Q.data;
                float* const       Vndata = V_nxt->data;
                for (int i = 0; i < n; ++i)
                {
                    const float* row_Vi  = Vcdata + i * Vcstep;
                    float*       row_Vni = Vndata + i * Vnstep;
                    for (int j = 0; j < n; ++j)
                    {
                        float sum = 0.0f;
                        for (int k = 0; k < n; ++k)
                            sum += row_Vi[k] * Qdata[k * Qstep + j];
                        row_Vni[j] = sum;
                    }
                }
            }

            // 3.5 Swap the ping-pong pointers (zero copy).
            {
                Mat *tmp = A_cur; A_cur = A_nxt; A_nxt = tmp;
                tmp = V_cur; V_cur = V_nxt; V_nxt = tmp;
            }
            iter_count = iter + 1;
        }

        // -----------------------------------------------------------------
        // 4. Read eigenvalues off the (converged or last) diagonal and copy
        //    out the accumulated eigenvectors.
        // -----------------------------------------------------------------
        {
            const int Astep        = A_cur->step;
            const float* const Adata = A_cur->data;
            const int evStep       = result.eigenvalues.step;
            for (int i = 0; i < n; ++i)
                result.eigenvalues.data[i * evStep] = Adata[i * Astep + i];
        }
        result.eigenvectors = *V_cur;
        result.iterations   = iter_count;
        if (converged)
        {
            result.status = TINY_OK;
        }
        else
        {
            result.status = TINY_ERR_NOT_FINISHED;
            std::cerr << "[Warning] eigendecompose_qr: did not converge within "
                      << max_iter << " iterations.\n";
        }
        return result;
    }

    /**
     * @name Mat::eigendecompose()
     * @brief Automatic eigenvalue decomposition with method selection.
     * @note Convenient interface for edge computing: automatically selects the best method.
     *       - Symmetric matrices: Uses Jacobi method (more efficient and stable)
     *       - General matrices: Uses QR algorithm
     * @note Time complexity: Depends on selected method
     *       - Jacobi: O(n³ * iterations) for symmetric matrices
     *       - QR: O(n³ * iterations) for general matrices
     * 
     * @param tolerance Convergence tolerance (must be >= 0)
     * @param max_iter Maximum number of iterations (must be > 0, default = 100)
     * @return EigenDecomposition containing eigenvalues, eigenvectors, and status
     */
    Mat::EigenDecomposition Mat::eigendecompose(float tolerance, int max_iter) const
    {
        if (this->data == nullptr)
        {
            EigenDecomposition result;
            std::cerr << "[Error] eigendecompose: matrix data pointer is null.\n";
            result.status = TINY_ERR_MATH_NULL_POINTER;
            return result;
        }
        if (tolerance < 0.0f)
        {
            EigenDecomposition result;
            std::cerr << "[Error] eigendecompose: tolerance must be >= 0 (got "
                      << tolerance << ").\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }
        if (max_iter <= 0)
        {
            EigenDecomposition result;
            std::cerr << "[Error] eigendecompose: max_iter must be > 0 (got "
                      << max_iter << ").\n";
            result.status = TINY_ERR_INVALID_ARG;
            return result;
        }

        // Symmetry-test margin: treat the matrix as symmetric when off-diag
        // mismatches stay within `kSymmetryToleranceFactor * tolerance`. This
        // is intentionally loose so matrices built from differences of
        // symmetric ops (which carry small FP asymmetry) still pick the
        // faster, more-stable Jacobi path.
        constexpr float kSymmetryToleranceFactor = 10.0f;
        const float sym_tol = kSymmetryToleranceFactor * tolerance;
        if (this->is_symmetric(sym_tol))
        {
            return this->eigendecompose_jacobi(tolerance, max_iter);
        }
        return this->eigendecompose_qr(max_iter, tolerance);
    }

    // ============================================================================
    // Stream Operators
    // ============================================================================
    /**
     * @name operator<<
     * @brief Stream insertion operator for printing matrix to the output stream (e.g., std::cout).
     *
     * This function allows printing the contents of a matrix to an output stream.
     * It prints each row of the matrix on a new line, with elements separated by spaces.
     *
     * @param os Output stream where the matrix will be printed (e.g., std::cout)
     * @param m Matrix to be printed
     *
     * @return os The output stream after printing the matrix
     */
    std::ostream &operator<<(std::ostream &os, const Mat &m)
    {
        // A 0x0 / 0xN / Nx0 matrix is a legal empty value (see Mat(int,int));
        // print nothing for it. A truly bad pointer (sized but null) is the
        // only real error case.
        if (m.row == 0 || m.col == 0)
        {
            return os;
        }
        if (m.data == nullptr)
        {
            os << "[Error] Cannot print matrix: data pointer is null.\n";
            return os;
        }

        for (int i = 0; i < m.row; ++i)
        {
            os << m(i, 0);
            for (int j = 1; j < m.col; ++j)
            {
                os << " " << m(i, j);
            }
            os << std::endl;
        }
        return os;
    }

    /**
     * @name operator<<
     * @brief Stream insertion operator for printing the Rectangular ROI structure to the output stream.
     *
     * This function prints the details of the ROI (Region of Interest) including the start row and column,
     * and the width and height of the rectangular region.
     *
     * @param os Output stream where the ROI will be printed (e.g., std::cout)
     * @param roi The ROI structure to be printed
     *
     * @return os The output stream after printing the ROI details
     */
    std::ostream &operator<<(std::ostream &os, const Mat::ROI &roi)
    {
        os << "row start " << roi.pos_y << std::endl;
        os << "col start " << roi.pos_x << std::endl;
        os << "row count " << roi.height << std::endl;
        os << "col count " << roi.width << std::endl;

        return os;
    }

    /**
     * @name operator>>
     * @brief Stream extraction operator for reading matrix from the input stream (e.g., std::cin).
     *
     * This function reads the contents of a matrix from an input stream.
     * The matrix elements are read row by row, with elements separated by spaces or newlines.
     *
     * @param is Input stream from which the matrix will be read (e.g., std::cin)
     * @param m Matrix to store the read data
     *
     * @return is The input stream after reading the matrix
     */
    std::istream &operator>>(std::istream &is, Mat &m)
    {
        if (m.row == 0 || m.col == 0)
        {
            return is; // nothing to read into an empty matrix
        }
        if (m.data == nullptr)
        {
            std::cerr << "[Error] operator>>: target matrix has null data buffer.\n";
            is.setstate(std::ios::failbit);
            return is;
        }

        for (int i = 0; i < m.row; ++i)
        {
            for (int j = 0; j < m.col; ++j)
            {
                is >> m(i, j);
            }
        }
        return is;
    }

    // ============================================================================
    // Global Arithmetic Operators
    // ============================================================================
    /**
     * + operator, sum of two matrices
     * The operator use DSP optimized implementation of multiplication.
     *
     * @param[in] A: Input matrix A
     * @param[in] B: Input matrix B
     *
     * @return
     *     - result matrix A+B
     */
    Mat operator+(const Mat &m1, const Mat &m2)
    {
        if (m1.data == nullptr || m2.data == nullptr)
        {
            std::cerr << "[Error] operator+: null matrix data pointer.\n";
            return Mat(0, 0);
        }
        if ((m1.row != m2.row) || (m1.col != m2.col))
        {
            std::cerr << "[Error] operator+: matrices do not have equal dimensions ("
                      << m1.row << "x" << m1.col << " vs "
                      << m2.row << "x" << m2.col << ").\n";
            return Mat(0, 0);
        }

        if (m1.sub_matrix || m2.sub_matrix)
        {
            Mat temp(m1.row, m1.col);
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_add_f32(m1.data, m2.data, temp.data, m1.row, m1.col, m1.pad, m2.pad, temp.pad, 1, 1, 1);
#else
            tiny_mat_add_f32(m1.data, m2.data, temp.data, m1.row, m1.col, m1.pad, m2.pad, temp.pad, 1, 1, 1);
#endif
            return temp;
        }
        else
        {
            Mat temp(m1);
            return (temp += m2);
        }
    }

    /**
     * + operator, sum of matrix with constant
     * The operator use DSP optimized implementation of multiplication.
     *
     * @param[in] A: Input matrix A
     * @param[in] C: Input constant
     *
     * @return
     *     - result matrix A+C
     */
    Mat operator+(const Mat &m, float C)
    {
        if (m.sub_matrix)
        {
            Mat temp(m.row, m.col);
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_addc_f32(m.data, temp.data, C, m.row, m.col, m.pad, temp.pad, 1, 1);
#else
            tiny_mat_addc_f32(m.data, temp.data, C, m.row, m.col, m.pad, temp.pad, 1, 1);
#endif
            return temp;
        }
        else
        {
            Mat temp(m);
            return (temp += C);
        }
    }

    /**
     * - operator, subtraction of two matrices
     * The operator use DSP optimized implementation of multiplication.
     *
     * @param[in] A: Input matrix A
     * @param[in] B: Input matrix B
     *
     * @return
     *     - result matrix A-B
     */
    Mat operator-(const Mat &m1, const Mat &m2)
    {
        if (m1.data == nullptr || m2.data == nullptr)
        {
            std::cerr << "[Error] operator-: null matrix data pointer.\n";
            return Mat(0, 0);
        }
        if ((m1.row != m2.row) || (m1.col != m2.col))
        {
            std::cerr << "[Error] operator-: matrices do not have equal dimensions ("
                      << m1.row << "x" << m1.col << " vs "
                      << m2.row << "x" << m2.col << ").\n";
            return Mat(0, 0);
        }

        if (m1.sub_matrix || m2.sub_matrix)
        {
            Mat temp(m1.row, m1.col);
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_sub_f32(m1.data, m2.data, temp.data, m1.row, m1.col, m1.pad, m2.pad, temp.pad, 1, 1, 1);
#else
            tiny_mat_sub_f32(m1.data, m2.data, temp.data, m1.row, m1.col, m1.pad, m2.pad, temp.pad, 1, 1, 1);
#endif
            return temp;
        }
        else
        {
            Mat temp(m1);
            return (temp -= m2);
        }
    }


    /**
     * - operator, subtraction of matrix with constant
     * The operator use DSP optimized implementation of multiplication.
     *
     * @param[in] A: Input matrix A
     * @param[in] C: Input constant
     *
     * @return
     *     - result matrix A-C
     */
    Mat operator-(const Mat &m, float C)
    {
        if (m.sub_matrix)
        {
            Mat temp(m.row, m.col);
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_addc_f32(m.data, temp.data, -C, m.row, m.col, m.pad, temp.pad, 1, 1);
#else
            tiny_mat_addc_f32(m.data, temp.data, -C, m.row, m.col, m.pad, temp.pad, 1, 1);
#endif
            return temp;
        }
        else
        {
            Mat temp(m);
            return (temp -= C);
        }
    }


    /**
     * * operator, multiplication of two matrices.
     * The operator use DSP optimized implementation of multiplication.
     *
     * @param[in] A: Input matrix A
     * @param[in] B: Input matrix B
     *
     * @return
     *     - result matrix A*B
     */
    Mat operator*(const Mat &m1, const Mat &m2)
    {
        if (m1.data == nullptr || m2.data == nullptr)
        {
            std::cerr << "[Error] operator*: null matrix data pointer.\n";
            return Mat(0, 0);
        }
        if (m1.col != m2.row)
        {
            std::cerr << "[Error] operator*: incompatible inner dimensions ("
                      << m1.row << "x" << m1.col << " * "
                      << m2.row << "x" << m2.col << ").\n";
            return Mat(0, 0);
        }
        Mat temp(m1.row, m2.col);

        if (m1.sub_matrix || m2.sub_matrix)
        {
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_mult_ex_f32(m1.data, m2.data, temp.data, m1.row, m1.col, m2.col, m1.pad, m2.pad, temp.pad);
#else
            tiny_mat_mult_ex_f32(m1.data, m2.data, temp.data, m1.row, m1.col, m2.col, m1.pad, m2.pad, temp.pad);
#endif
        }
        else
        {
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_mult_f32(m1.data, m2.data, temp.data, m1.row, m1.col, m2.col);
#else
            tiny_mat_mult_f32(m1.data, m2.data, temp.data, m1.row, m1.col, m2.col);
#endif
        }

        return temp;
    }

    /**
     * * operator, multiplication of matrix with constant
     * The operator use DSP optimized implementation of multiplication.
     *
     * @param[in] A: Input matrix A
     * @param[in] C: floating point value
     *
     * @return
     *     - result matrix A*B
     */
    Mat operator*(const Mat &m, float num)
    {
        if (m.sub_matrix)
        {
            Mat temp(m.row, m.col);
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_mulc_f32(m.data, temp.data, num, m.row, m.col, m.pad, temp.pad, 1, 1);
#else
            tiny_mat_multc_f32(m.data, temp.data, num, m.row, m.col, m.pad, temp.pad, 1, 1);
#endif
            return temp;
        }
        else
        {
            Mat temp(m);
            return (temp *= num);
        }
    }
    
    /**
     * * operator, multiplication of matrix with constant
     * The operator use DSP optimized implementation of multiplication.
     *
     * @param[in] C: floating point value
     * @param[in] A: Input matrix A
     *
     * @return
     *     - result matrix C*A
     */
    Mat operator*(float num, const Mat &m)
    {
        return (m * num);
    }

    /**
     * / operator, divide of matrix by constant
     * The operator use DSP optimized implementation of multiplication.
     *
     * @param[in] A: Input matrix A
     * @param[in] C: floating point value
     *
     * @return
     *     - result matrix A/C
     */
    Mat operator/(const Mat &m, float num)
    {
        if (m.data == nullptr)
        {
            std::cerr << "[Error] operator/: null matrix data pointer.\n";
            return Mat(0, 0);
        }
        if (!(fabsf(num) > TINY_MATH_MIN_POSITIVE_INPUT_F32))
        {
            std::cerr << "[Error] operator/: division by zero (num=" << num << ").\n";
            return Mat(0, 0);
        }

        if (m.sub_matrix)
        {
            Mat temp(m.row, m.col);
            float inv_num = 1.0f / num;
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_mulc_f32(m.data, temp.data, inv_num, m.row, m.col, m.pad, temp.pad, 1, 1);
#else
            tiny_mat_multc_f32(m.data, temp.data, inv_num, m.row, m.col, m.pad, temp.pad, 1, 1);
#endif
            return temp;
        }
        else
        {
            Mat temp(m);
            return (temp /= num);
        }
    }


    /**
     * / operator, divide matrix A by matrix B (element-wise)
     *
     * @param[in] A: Input matrix A
     * @param[in] B: Input matrix B
     *
     * @return
     *     - result matrix C, where C[i,j] = A[i,j]/B[i,j]
     */
    Mat operator/(const Mat &A, const Mat &B)
    {
        if (A.data == nullptr || B.data == nullptr)
        {
            std::cerr << "[Error] operator/: null matrix data pointer.\n";
            return Mat(0, 0);
        }
        if (A.row <= 0 || A.col <= 0 || B.row <= 0 || B.col <= 0)
        {
            std::cerr << "[Error] operator/: invalid matrix dimensions.\n";
            return Mat(0, 0);
        }
        if ((A.row != B.row) || (A.col != B.col))
        {
            std::cerr << "[Error] operator/: matrices do not have equal dimensions ("
                      << A.row << "x" << A.col << " vs "
                      << B.row << "x" << B.col << ").\n";
            return Mat(0, 0);
        }

        Mat temp(A.row, A.col);
        if (temp.data == nullptr)
        {
            std::cerr << "[Error] operator/: failed to allocate result.\n";
            return Mat(0, 0);
        }

        const int Astep = A.step;
        const int Bstep = B.step;
        const int Tstep = temp.step;
        const float* const Adata = A.data;
        const float* const Bdata = B.data;
        float* const       Tdata = temp.data;
        const int rows = A.row;
        const int cols = A.col;

        for (int r = 0; r < rows; ++r)
        {
            const float* row_A = Adata + r * Astep;
            const float* row_B = Bdata + r * Bstep;
            float*       row_T = Tdata + r * Tstep;
            for (int c = 0; c < cols; ++c)
            {
                if (!(fabsf(row_B[c]) > TINY_MATH_MIN_POSITIVE_INPUT_F32))
                {
                    std::cerr << "[Error] operator/: division by zero at ("
                              << r << ", " << c << ").\n";
                    return Mat(0, 0);
                }
                row_T[c] = row_A[c] / row_B[c];
            }
        }
        return temp;
    }

    
    /**
     * == operator, compare two matrices
     *
     * @param[in] A: Input matrix A
     * @param[in] B: Input matrix B
     *
     * @return
     *      - true if matrices are the same
     *      - false if matrices are different
     */
    bool operator==(const Mat &m1, const Mat &m2)
    {
        // Shape mismatch -> not equal.
        if ((m1.col != m2.col) || (m1.row != m2.row))
        {
            return false;
        }
        // Two empty matrices of the same shape are equal.
        if (m1.row == 0 || m1.col == 0)
        {
            return true;
        }
        // Pointer hazards: a sized matrix with a null buffer cannot be
        // compared meaningfully. Treat as not-equal rather than UB.
        if (m1.data == nullptr || m2.data == nullptr)
        {
            return false;
        }

        // Use `!(diff <= epsilon)` so that NaN on either side falls through
        // to "not equal" (NaN compares unordered against everything).
        // This is the IEEE-correct semantics that `diff > epsilon` misses.
        constexpr float epsilon = 1e-5f;
        const int s1 = m1.step;
        const int s2 = m2.step;
        const float* const d1 = m1.data;
        const float* const d2 = m2.data;
        const int rows = m1.row;
        const int cols = m1.col;

        for (int r = 0; r < rows; ++r)
        {
            const float* row1 = d1 + r * s1;
            const float* row2 = d2 + r * s2;
            for (int c = 0; c < cols; ++c)
            {
                const float diff = fabsf(row1[c] - row2[c]);
                if (!(diff <= epsilon))
                {
                    return false;
                }
            }
        }
        return true;
    }
}

