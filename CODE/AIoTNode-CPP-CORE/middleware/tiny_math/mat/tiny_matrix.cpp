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
#include <iostream>
#include <stdexcept>
#include <string.h>
#include <math.h>
#include <cmath>
#include <inttypes.h>
#include <iomanip>

/* LIBRARIE CONTENTS */
namespace tiny
{
    /* === Rectangular ROI Structure === */
    /**
     * @brief Construct a new Mat:: R O I:: R O I object
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
        return this->width * this->height;
    }

    /* === Printing Functions === */
    /**
     * @name Mat::PrintHead()
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
        std::cout << "stride          " << this->stride << "\n";
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
    void Mat::print_matrix(bool show_padding)
    {
        std::cout << "Matrix Elements >>>\n";
        for (int i = 0; i < this->row; ++i)
        {
            // print the non-padding elements
            for (int j = 0; j < this->col; ++j)
            {
                std::cout << std::setw(12) << this->data[i * this->stride + j] << " ";
            }

            // if padding is enabled, print the padding elements
            if (show_padding)
            {
                // print a separator first
                std::cout << "      |";

                // print the padding elements
                for (int j = this->col; j < this->stride; ++j)
                {
                    if (j == this->col)
                    {
                        std::cout << std::setw(7) << this->data[i * this->stride + j] << " ";
                    }
                    else
                    {
                        // print the padding elements
                        std::cout << std::setw(12) << this->data[i * this->stride + j] << " ";
                    }
                }
            }

            // print a new line after each row
            std::cout << "\n";
        }

        std::cout << "<<< Matrix Elements\n";
        std::cout << std::endl;
    }

    /* === Constructors & Destructor === */
    // memory allocation
    /**
     * @name Mat::allocate()
     * @brief Allocate memory for the matrix according to the memory required.
     */
    void Mat::alloc_mem()
    {
        this->ext_buff = false;
        this->memory = this->row * this->stride;
        this->data = new float[this->memory];
    }

    /**
     * @name Mat::Mat()
     * @brief Constructor - default constructor: create a 1x1 matrix with only a zero element.
     */
    Mat::Mat()
    {
        this->row = 1;
        this->col = 1;
        this->pad = 0;
        this->stride = 1;
        this->element = 1;
        this->memory = 1;
        this->data = nullptr;
        this->temp = nullptr;
        this->ext_buff = false;
        this->sub_matrix = false;
        alloc_mem();
        if (this->data == nullptr)
        {
            std::cerr << "[>>> Error ! <<<] Memory allocation failed in alloc_mem()\n";
        }
        std::memset(this->data, 0, this->memory * sizeof(float));
    }

    /**
     * @name Mat::Mat(int rows, int cols)
     * @brief Constructor - create a matrix with the specified number of rows and columns.
     *
     * @param rows Number of rows
     * @param cols Number of columns
     */
    Mat::Mat(int rows, int cols)
    {
        this->row = rows;
        this->col = cols;
        this->pad = 0;
        this->stride = cols;
        this->element = rows * cols;
        this->memory = rows * cols;
        this->data = nullptr;
        this->temp = nullptr;
        this->ext_buff = false;
        this->sub_matrix = false;
        alloc_mem();
        if (this->data == nullptr)
        {
            std::cerr << "[>>> Error ! <<<] Memory allocation failed in alloc_mem()\n";
        }
        std::memset(this->data, 0, this->memory * sizeof(float));
    }
    /**
     * @name Mat::Mat(int rows, int cols, int stride)
     * @brief Constructor - create a matrix with the specified number of rows, columns and stride.
     *
     * @param rows Number of rows
     * @param cols Number of columns
     * @param stride Stride (number of elements in a row)
     */
    Mat::Mat(int rows, int cols, int stride)
    {
        this->row = rows;
        this->col = cols;
        this->pad = stride - cols;
        this->stride = stride;
        this->element = rows * cols;
        this->memory = rows * stride;
        this->data = nullptr;
        this->temp = nullptr;
        this->ext_buff = false;
        this->sub_matrix = false;
        alloc_mem();
        if (this->data == nullptr)
        {
            std::cerr << "[>>> Error ! <<<] Memory allocation failed in alloc_mem()\n";
        }
        std::memset(this->data, 0, this->memory * sizeof(float));
    }

    /**
     * @name Mat::Mat(float *data, int rows, int cols)
     * @brief Constructor - create a matrix with the specified number of rows, columns and external data.
     *
     * @param data Pointer to external data buffer
     * @param rows Number of rows
     * @param cols Number of columns
     */
    Mat::Mat(float *data, int rows, int cols)
    {
        this->row = rows;
        this->col = cols;
        this->pad = 0;
        this->stride = cols;
        this->element = rows * cols;
        this->memory = rows * cols; // for external data, this item is actually not used
        this->data = data;
        this->temp = nullptr;
        this->ext_buff = true;
        this->sub_matrix = false;
    }

    /**
     * @name Mat::Mat(float *data, int rows, int cols, int stride)
     * @brief Constructor - create a matrix with the specified number of rows, columns and external data.
     *
     * @param data Pointer to external data buffer
     * @param rows Number of rows
     * @param cols Number of columns
     * @param stride Stride (number of elements in a row)
     */
    Mat::Mat(float *data, int rows, int cols, int stride)
    {
        this->row = rows;
        this->col = cols;
        this->pad = stride - cols;
        this->stride = stride;
        this->element = rows * cols;
        this->memory = rows * stride; // for external data, this item is actually not used
        this->data = data;
        this->temp = nullptr;
        this->ext_buff = true;
        this->sub_matrix = false;
    }

    /**
     * @name Mat::Mat(const Mat &src)
     * @brief Copy constructor - create a matrix with the same properties as the source matrix.
     *
     * @param src Source matrix
     */
    Mat::Mat(const Mat &src)
    {
        this->row = src.row;
        this->col = src.col;
        this->pad = src.pad;
        this->stride = src.stride;
        this->element = src.element;
        this->memory = src.memory;

        if (src.sub_matrix && src.ext_buff)
        {
            // if the source is a view (submatrix), do shallow copy
            this->data = src.data;
            this->temp = nullptr;
            this->ext_buff = true;
            this->sub_matrix = true;
        }
        else
        {
            // otherwise do deep copy
            this->data = nullptr;
            this->temp = nullptr;
            this->ext_buff = false;
            this->sub_matrix = false;

            if (src.data != nullptr)
            {
                alloc_mem();
                if (this->data == nullptr)
                {
                    std::cerr << "[Error] Memory allocation failed in alloc_mem()\n";
                }
                std::memcpy(this->data, src.data, this->memory * sizeof(float));
            }
        }
    }

    /**
     * @name ~Mat()
     * @brief Destructor - free the memory allocated for the matrix.
     */
    Mat::~Mat()
    {
        if (!this->ext_buff && this->data)
        {
            delete[] this->data;
        }
        if (this->temp)
        {
            delete[] this->temp;
        }
    }

    /* === Element Access === */
    // Already defined by inline functions in the header file

    /* === Data Manipulation === */

    /**
     * @name Mat::copy_paste(const Mat &src, int row_pos, int col_pos)
     * @brief Copy the elements of the source matrix into the destination matrix. The dimension of the current matrix must be larger than the source matrix.
     * @brief This one does not share memory with the source matrix.
     *
     * @param src Source matrix
     * @param row_pos Start row position of the destination matrix
     * @param col_pos Start column position of the destination matrix
     */
    tiny_error_t Mat::copy_paste(const Mat &src, int row_pos, int col_pos)
    {
        if ((row_pos + src.row) > this->row)
        {
            std::cerr << "[>>> Error ! <<<] Invalid row position " << std::endl;
            return TINY_ERR_INVALID_ARG;
        }
        if ((col_pos + src.col) > this->col)
        {
            std::cerr << "[>>> Error ! <<<] Invalid column position " << std::endl;
            return TINY_ERR_INVALID_ARG;
        }
        for (size_t r = 0; r < src.row; r++)
        {
            memcpy(&this->data[(r + row_pos) * this->stride + col_pos], &src.data[r * src.col], src.col * sizeof(float));
        }

        return TINY_OK;
    }

    /**
     * @name Mat::copy_head(const Mat &src)
     * @brief Copy the header of the source matrix into the destination matrix. The data pointer is shared.
     *
     * @param src Source matrix
     */
    tiny_error_t Mat::copy_head(const Mat &src)
    {
        if (!this->ext_buff)
        {
            delete[] this->data;
        }
        this->row = src.row;
        this->col = src.col;
        this->element = src.element;
        this->pad = src.pad;
        this->stride = src.stride;
        this->memory = src.memory;
        this->data = src.data;
        this->temp = src.temp;
        this->ext_buff = src.ext_buff;
        this->sub_matrix = src.sub_matrix;

        return TINY_OK;
    }

    /**
     * @name Mat::view_roi(int start_row, int start_col, int roi_rows, int roi_cols)
     * @brief Make a shallow copy of ROI matrix. | Make a view of the ROI matrix. Low level function. Unlike ESP-DSP, it is not allowed to setup stride here, stride is automatically calculated inside the function.
     *
     * @param start_row Start row position of source matrix to copy
     * @param start_col Start column position of source matrix to copy
     * @param roi_rows Size of row elements of source matrix to copy
     * @param roi_cols Size of column elements of source matrix to copy
     *
     * @todo the pointer address is changing every time access, but the result is correct.
     *
     * @return result matrix size row_size x col_size
     */
    Mat Mat::view_roi(int start_row, int start_col, int roi_rows, int roi_cols) const
    {
        if ((start_row + roi_rows) > this->row || (start_col + roi_cols) > this->col)
        {
            std::cerr << "[Error] Invalid ROI request.\n";
            return Mat();
        }

        Mat result;
        result.row = roi_rows;
        result.col = roi_cols;
        result.stride = this->stride;
        result.pad = this->stride - roi_cols;
        result.element = roi_rows * roi_cols;
        result.memory = roi_rows * this->stride;
        result.data = this->data + (start_row * this->stride + start_col);
        result.temp = nullptr;
        result.ext_buff = true;
        result.sub_matrix = true;

        return result;
    }

    /**
     * @name Mat::view_roi(const Mat::ROI &roi)
     * @brief Make a shallow copy of ROI matrix. | Make a view of the ROI matrix. Using ROI structure.
     *
     * @param roi Rectangular area of interest
     *
     * @return result matrix size row_size x col_size
     */
    Mat Mat::view_roi(const Mat::ROI &roi) const
    {
        return view_roi(roi.pos_y, roi.pos_x, roi.height, roi.width);
    }

    /**
     * @name Mat::copy_roi(int start_row, int start_col, int height, int width)
     * @brief Make a deep copy of matrix. Copared to view_roi(), this one is a deep copy, not sharing memory with the source matrix.
     *
     * @param start_row Start row position of source matrix to copy
     * @param start_col Start column position of source matrix to copy
     * @param height Size of row elements of source matrix to copy
     * @param width Size of column elements of source matrix to copy
     *
     * @return result matrix size row_size x col_size
     */
    Mat Mat::copy_roi(int start_row, int start_col, int height, int width)
    {
        if ((start_row + height) > this->row)
        {
            std::cerr << "[>>> Error ! <<<] Invalid row position " << std::endl;
            return Mat();
        }
        if ((start_col + width) > this->col)
        {
            std::cerr << "[>>> Error ! <<<] Invalid columnn position " << std::endl;
            return Mat();
        }

        // initiate the result matrix
        Mat result(height, width);

        // deep copy the data from the source matrix
        for (size_t r = 0; r < result.row; r++)
        {
            memcpy(&result.data[r * result.col], &this->data[(r + start_row) * this->stride + start_col], result.col * sizeof(float));
        }

        // return result;
        return result;
    }

    /**
     * @name Mat::copy_roi(const Mat::ROI &roi)
     * @brief Make a deep copy of matrix. Using ROI structure. Copared to view_roi(), this one is a deep copy, not sharing memory with the source matrix.
     *
     * @param roi Rectangular area of interest
     *
     * @return result matrix size row_size x col_size
     */
    Mat Mat::copy_roi(const Mat::ROI &roi)
    {
        return (copy_roi(roi.pos_y, roi.pos_x, roi.height, roi.width));
    }

    /**
     * @name Mat::block(int start_row, int start_col, int block_rows, int block_cols)
     * @brief Get a block of matrix.
     *
     * @param start_row
     * @param start_col
     * @param block_rows
     * @param block_cols
     * @return Mat
     */
    Mat Mat::block(int start_row, int start_col, int block_rows, int block_cols)
    {
        Mat result(block_rows, block_cols);
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
     *
     * @param row1 The index of the first row to swap
     * @param row2 The index of the second row to swap
     */
    void Mat::swap_rows(int row1, int row2)
    {
        if ((this->row <= row1) || (this->row <= row2))
        {
            std::cerr << "Error: row index out of range" << std::endl;
        }
        else
        {
            float *temp_row = new float[this->col];
            memcpy(temp_row, &this->data[row1 * this->stride], this->col * sizeof(float));
            memcpy(&this->data[row1 * this->stride], &this->data[row2 * this->stride], this->col * sizeof(float));
            memcpy(&this->data[row2 * this->stride], temp_row, this->col * sizeof(float));
            delete[] temp_row;
        }
    }

    /**
     * @name Mat::clear()
     * @brief Clear the matrix by setting all elements to zero.
     */
    void Mat::clear(void)
    {
        for (int row = 0; row < this->row; row++)
        {
            memset(this->data + (row * this->stride), 0, this->col * sizeof(float));
        }
    }

    /* === Arithmetic Operators === */
    /**
     * @name &Mat::operator=(const Mat &src)
     * @brief Copy assignment operator - copy the elements of the source matrix into the destination matrix. Compared to the copy constructor, this one is used for existing matrix to copy the elements. The copy constructor is used for the first time to create a new matrix and copy the elements at the same time.
     *
     * @param src
     * @return Mat&
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

        // 3. If dimensions differ, reallocate memory
        if (this->row != src.row || this->col != src.col)
        {
            if (!this->ext_buff && this->data != nullptr)
            {
                delete[] this->data;
            }

            // Update dimensions and memory info
            this->row = src.row;
            this->col = src.col;
            this->stride = src.col; // Follow source's logical stride
            this->pad = 0;
            this->element = this->row * this->col;
            this->memory = this->row * this->stride;

            this->ext_buff = false;
            this->sub_matrix = false;

            alloc_mem();
        }

        // 4. Data copy (row-wise)
        for (int r = 0; r < this->row; ++r)
        {
            std::memcpy(this->data + r * this->stride, src.data + r * src.stride, this->col * sizeof(float));
        }

        return *this;
    }

    /**
     * @name Mat::operator+=(const Mat &A)
     * @brief Element-wise addition of another matrix to this matrix.
     *
     * @param A The matrix to add
     * @return Mat& Reference to the current matrix
     */
    /**
     * @name Mat::operator+=(const Mat &A)
     * @brief Element-wise addition of another matrix to this matrix.
     *
     * @param A The matrix to add
     * @return Mat& Reference to the current matrix
     */
    Mat &Mat::operator+=(const Mat &A)
    {
        // 1. Dimension check
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
     *
     * @param C The constant to add
     */
    /**
     * @name Mat::operator+=(float C)
     * @brief Element-wise addition of a constant to this matrix.
     *
     * @param C The constant to add
     * @return Mat& Reference to the current matrix
     */
    Mat &Mat::operator+=(float C)
    {
        // check whether padding is presented
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
     *
     * @param A The matrix to subtract
     * @return Mat& Reference to the current matrix
     */
    /**
     * @name Mat::operator-=(const Mat &A)
     * @brief Element-wise subtraction of another matrix from this matrix.
     *
     * @param A The matrix to subtract
     * @return Mat& Reference to the current matrix
     */
    Mat &Mat::operator-=(const Mat &A)
    {
        // 1. Dimension check
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
     *
     * @param C The constant to subtract
     */
    /**
     * @name Mat::operator-=(float C)
     * @brief Element-wise subtraction of a constant from this matrix.
     *
     * @param C The constant to subtract
     * @return Mat& Reference to the current matrix
     */
    Mat &Mat::operator-=(float C)
    {
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
     *
     * @param m The matrix to multiply with
     * @return Mat& Reference to the current matrix
     */
    Mat &Mat::operator*=(const Mat &m)
    {
        // 1. Dimension check
        if (this->col != m.row)
        {
            std::cerr << "[Error] Matrix multiplication failed: incompatible dimensions ("
                      << this->row << "x" << this->col << " * "
                      << m.row << "x" << m.col << ")\n";
            return *this;
        }

        // 2. Prepare temp matrix (incase overwriting the original data)
        Mat temp = this->copy_roi(0, 0, this->row, this->col);

        // 3. check whether padding is present in either matrix
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
     * @brief Element-wise multiplication by a constant
     *
     * @param num The constant multiplier
     * @return Mat& Reference to the current matrix
     */
    Mat &Mat::operator*=(float num)
    {
        // check whether padding is present
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
     *
     * @param B The matrix divisor
     * @return Mat& Reference to the current matrix
     */
    Mat &Mat::operator/=(const Mat &B)
    {
        // 1. Dimension check
        if ((this->row != B.row) || (this->col != B.col))
        {
            std::cerr << "[Error] Matrix division failed: Dimension mismatch ("
                      << this->row << "x" << this->col << " vs "
                      << B.row << "x" << B.col << ")\n";
            return *this;
        }

        // 2. Zero division check
        bool zero_found = false;
        for (int i = 0; i < B.row; ++i)
        {
            for (int j = 0; j < B.col; ++j)
            {
                if (B(i, j) == 0.0f)
                {
                    zero_found = true;
                    break;
                }
            }
            if (zero_found)
                break;
        }

        if (zero_found)
        {
            std::cerr << "[Error] Matrix division failed: Division by zero detected.\n";
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
     *
     * @param num The constant divisor
     * @return Mat& Reference to the current matrix
     */
    Mat &Mat::operator/=(float num)
    {
        // 1. Check division by zero
        if (num == 0.0f)
        {
            std::cerr << "[Error] Matrix division by zero is undefined.\n";
            return *this;
        }

        // 2. Determine if padding handling is needed
        bool need_padding_handling = (this->pad > 0);

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
     *
     * @param num The exponent (integer)
     * @return Mat New matrix after exponentiation
     */
    Mat Mat::operator^(int num)
    {
        // Handle special cases
        if (num == 0)
        {
            // Any number to the power of 0 is 1
            Mat result(this->row, this->col, this->stride);
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

        if (num < 0)
        {
            std::cerr << "[Error] Negative exponent not supported in operator^.\n";
            return Mat(*this); // Return a copy without modification
        }

        // General case: positive exponent > 1
        Mat result(this->row, this->col, this->stride);
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

    /* === Linear Algebra === */
    /**
     * @name Mat::transpose
     * @brief Transpose the matrix.
     *
     * @return Transposed matrix
     */
    Mat Mat::transpose()
    {
        Mat result(this->col, this->row);
        for (int i = 0; i < this->row; ++i)
        {
            for (int j = 0; j < this->col; ++j)
            {
                result(j, i) = this->data[i * this->stride + j];
            }
        }
        return result;
    }

    /**
     * @brief Calculate the cofactor matrix by removing specified row and column.
     *
     * @param target_row Row index to remove
     * @param target_col Column index to remove
     * @return Mat The (n-1)x(n-1) cofactor matrix
     */
    Mat Mat::cofactor(int target_row, int target_col)
    {
        if (this->row != this->col)
        {
            std::cerr << "[Error] Cofactor requires square matrix.\n";
            return Mat();
        }

        int n = this->row;
        Mat result(n - 1, n - 1);

        for (int i = 0, res_i = 0; i < n; ++i)
        {
            if (i == target_row)
                continue;

            for (int j = 0, res_j = 0; j < n; ++j)
            {
                if (j == target_col)
                    continue;

                result.data[res_i * result.stride + res_j] = this->data[i * this->stride + j];
                res_j++;
            }
            res_i++;
        }

        return result;
    }

    /**
     * @name Mat::determinant()
     * @brief Calculate the determinant of a square matrix using Laplace Expansion.
     * @brief Low efficiency, only suitable for small matrices!!!
     *
     * @return Determinant value (float)
     */
    float Mat::determinant()
    {
        if (this->row != this->col)
        {
            std::cerr << "[Error] Determinant can only be calculated for square matrices.\n";
            return 0.0f;
        }

        int n = this->row;

        // Base case: 1x1 matrix
        if (n == 1)
            return this->data[0];

        // Base case: 2x2 matrix (optimized)
        if (n == 2)
            return this->data[0] * this->data[3] - this->data[1] * this->data[2];

        float D = 0.0f;
        int sign = 1;

        for (int f = 0; f < n; ++f)
        {
            Mat minor = this->cofactor(0, f);                // Get cofactor matrix
            D += sign * (*this)(0, f) * minor.determinant(); // Recursive call to calculate determinant of the cofactor matrix
            sign = -sign;                                    // Alternate sign
        }

        return D;
    }

    /**
     * @name Mat::adjoint()
     * @brief Calculate the adjoint (adjugate) matrix of a square matrix.
     *
     * @return Mat The adjoint matrix
     */
    Mat Mat::adjoint()
    {
        if (this->row != this->col)
        {
            std::cerr << "[Error] Adjoint can only be computed for square matrices.\n";
            return Mat();
        }

        int n = this->row;
        Mat adj(n, n);

        // Special case for 1x1 matrix
        if (n == 1)
        {
            adj(0, 0) = 1.0f;
            return adj;
        }

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                // Calculate cofactor matrix of element (i, j)
                Mat cof = this->cofactor(i, j);

                int sign = ((i + j) % 2 == 0) ? 1 : -1;

                // Adjoint is transpose of cofactor matrix
                adj(j, i) = sign * cof.determinant();
            }
        }

        return adj;
    }

    /**
     * @brief Normalize the matrix using L2 norm (Frobenius norm).
     *        After normalization, ||Matrix|| = 1
     */
    void Mat::normalize()
    {
        float norm_sq = 0.0f;

        for (int i = 0; i < this->row; ++i)
        {
            for (int j = 0; j < this->col; ++j)
            {
                float val = (*this)(i, j);
                norm_sq += val * val;
            }
        }

        if (norm_sq == 0.0f)
        {
            std::cerr << "[Warning] Cannot normalize a zero matrix.\n";
            return;
        }

        float inv_norm = 1.0f / sqrtf(norm_sq);
        *this *= inv_norm;
    }

    /**
     * @name Mat::norm() const
     * @brief Calculate the Frobenius norm (L2 norm) of the matrix.
     *
     * @return float The computed matrix norm
     */
    float Mat::norm() const
    {
        float sum_sq = 0.0f;

        for (int i = 0; i < this->row; ++i)
        {
            for (int j = 0; j < this->col; ++j)
            {
                float val = (*this)(i, j); // Access valid matrix element
                sum_sq += val * val;
            }
        }

        return sqrtf(sum_sq);
    }

    /**
     * @name Mat::inverse_adjoint()
     * @brief Compute the inverse of a square matrix using adjoint method.
     *
     * @return Mat The inverse matrix. If singular, returns a zero matrix.
     */
    Mat Mat::inverse_adjoint()
    {
        if (this->row != this->col)
        {
            std::cerr << "[Error] Inverse can only be computed for square matrices.\n";
            return Mat();
        }

        int n = this->row;

        float det_val = this->determinant();
        if (det_val == 0.0f)
        {
            std::cerr << "[Error] Singular matrix, inverse does not exist.\n";
            return Mat(n, n); // Return zero matrix
        }

        Mat adj = this->adjoint();

        // Inverse = adjoint / determinant
        Mat inv(n, n);
        float inv_det = 1.0f / det_val;

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                inv(i, j) = adj(i, j) * inv_det;
            }
        }

        return inv;
    }

    /**
     * @name Mat::eye(int size)
     * @brief Generate an identity matrix of given size.
     *
     * @param size Dimension of the square identity matrix
     * @return Mat Identity matrix (size x size)
     */
    Mat Mat::eye(int size)
    {
        Mat identity(size, size);

        // Set diagonal elements to 1, rest are initialized as 0
        for (int i = 0; i < size; ++i)
        {
            identity(i, i) = 1.0f;
        }

        return identity;
    }

    /**
     * @name Mat::augment(const Mat &A, const Mat &B)
     * @brief Augment two matrices horizontally [A | B].
     *
     * @param A Left matrix
     * @param B Right matrix
     * @return Mat Augmented matrix [A B]
     */
    Mat Mat::augment(const Mat &A, const Mat &B)
    {
        // 1. Check if row counts match
        if (A.row != B.row)
        {
            std::cerr << "[Error] Cannot augment matrices: Row counts do not match ("
                      << A.row << " vs " << B.row << ")\n";
            return Mat();
        }

        // 2. Create new matrix with combined columns
        Mat AB(A.row, A.col + B.col);

        // 3. Copy data from A and B
        for (int i = 0; i < A.row; ++i)
        {
            // Copy A
            for (int j = 0; j < A.col; ++j)
            {
                AB(i, j) = A(i, j);
            }
            // Copy B
            for (int j = 0; j < B.col; ++j)
            {
                AB(i, A.col + j) = B(i, j);
            }
        }

        return AB;
    }

    /**
     * @name Mat::ones(int size)
     * @brief Create a square matrix filled with ones.
     *
     * @param size Size of the square matrix (rows = cols)
     * @return Mat Square matrix [size x size] with all elements = 1
     */
    Mat Mat::ones(int size)
    {
        return Mat::ones(size, size);
    }

    /**
     * @name Mat::ones(int rows, int cols)
     * @brief Create a matrix of specified size filled with ones.
     *
     * @param rows Number of rows
     * @param cols Number of columns
     * @return Mat Matrix [rows x cols] with all elements = 1
     */
    Mat Mat::ones(int rows, int cols)
    {
        Mat result(rows, cols);

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
     * @name Mat::gaussian_eliminate
     * @brief Perform Gaussian Elimination to convert matrix to Row Echelon Form (REF).
     *
     * @return Mat The upper triangular matrix (REF form)
     */
    Mat Mat::gaussian_eliminate() const
    {
        Mat result(*this); // Create a copy of the original matrix
        int rows = result.row;
        int cols = result.col;

        int lead = 0; // Leading column tracker

        for (int r = 0; r < rows; ++r)
        {
            if (lead >= cols)
                break;

            int i = r;

            // Find pivot row (partial pivoting)
            while (result(i, lead) == 0)
            {
                i++;
                if (i == rows)
                {
                    i = r;
                    lead++;
                    if (lead == cols)
                        return result; // Return the result matrix (upper triangular)
                }
            }

            // Swap rows if pivot is not in current row
            if (i != r)
                result.swap_rows(i, r);

            // Eliminate rows below
            for (int j = r + 1; j < rows; ++j)
            {
                if (result(j, lead) == 0)
                    continue;

                float factor = result(j, lead) / result(r, lead);
                for (int k = lead; k < cols; ++k)
                {
                    result(j, k) -= factor * result(r, k);

                    // Numerical precision handling (set near-zero values to zero)
                    if (fabs(result(j, k)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                        result(j, k) = 0.0f;
                }
            }

            lead++;
        }

        return result; // Return the upper triangular matrix
    }

    /**
     * @name Mat::row_reduce_from_gaussian()
     * @brief Convert a matrix (assumed in row echelon form) to Reduced Row Echelon Form (RREF).
     *
     * @return Mat The matrix in RREF form
     */
    Mat Mat::row_reduce_from_gaussian()
    {
        Mat R(*this); // Make a copy to preserve original matrix
        int rows = R.row;
        int cols = R.col;

        int pivot_row = rows - 1;
        int pivot_col = cols - 2;

        while (pivot_row >= 0)
        {
            // Locate pivot in current row
            int current_pivot_col = -1;
            for (int k = 0; k < cols; ++k)
            {
                if (R(pivot_row, k) != 0)
                {
                    current_pivot_col = k;
                    break;
                }
            }

            if (current_pivot_col != -1)
            {
                // Normalize pivot row
                float pivot_val = R(pivot_row, current_pivot_col);
                for (int s = current_pivot_col; s < cols; ++s)
                {
                    R(pivot_row, s) /= pivot_val;
                    if (fabs(R(pivot_row, s)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                    {
                        R(pivot_row, s) = 0.0f;
                    }
                }

                // Eliminate above pivot
                for (int t = pivot_row - 1; t >= 0; --t)
                {
                    float factor = R(t, current_pivot_col);
                    for (int s = current_pivot_col; s < cols; ++s)
                    {
                        R(t, s) -= factor * R(pivot_row, s);
                        if (fabs(R(t, s)) < TINY_MATH_MIN_POSITIVE_INPUT_F32)
                        {
                            R(t, s) = 0.0f;
                        }
                    }
                }
            }

            pivot_row--;
        }

        return R;
    }

    /**
     * @name Mat::inverse_gje()
     * @brief Compute the inverse of a square matrix using Gauss-Jordan elimination.
     *
     * @return Mat The inverse matrix if invertible, otherwise returns empty matrix.
     */
    Mat Mat::inverse_gje()
    {
        if (this->row != this->col)
        {
            std::cerr << "[Error] Inversion requires a square matrix.\n";
            return Mat();
        }

        // Step 1: Create augmented matrix [A | I]
        Mat I = Mat::eye(this->row);            // Identity matrix
        Mat augmented = Mat::augment(*this, I); // Augment matrix A with I

        // Step 2: Apply Gauss-Jordan elimination to get [I | A_inv]
        Mat rref = augmented.gaussian_eliminate().row_reduce_from_gaussian();

        // Check if the left half is the identity matrix
        for (int i = 0; i < this->row; ++i)
        {
            for (int j = 0; j < this->col; ++j)
            {
                if (fabs(rref(i, j) - I(i, j)) > TINY_MATH_MIN_POSITIVE_INPUT_F32)
                {
                    std::cerr << "[Error] Matrix is singular, cannot compute inverse.\n";
                    return Mat();
                }
            }
        }

        // Step 3: Extract the right half as the inverse matrix
        Mat result(this->row, this->col);
        for (int i = 0; i < this->row; ++i)
        {
            for (int j = 0; j < this->col; ++j)
            {
                result(i, j) = rref(i, j + this->col); // Extract the right part
            }
        }

        return result;
    }

    /**
     * @name Mat::dotprod(const Mat &A, const Mat &B)
     * @brief Calculate the dot product of two vectors (Nx1).
     *
     * @param[in] A Input vector A (Nx1).
     * @param[in] B Input vector B (Nx1).
     *
     * @return float The computed dot product value.
     */
    float Mat::dotprod(const Mat &A, const Mat &B)
    {
        if (A.row != B.row || A.col != 1 || B.col != 1)
        {
            std::cerr << "[Error] Dot product can only be computed for two vectors of the same length.\n";
            return 0.0f; // Return 0 in case of dimension mismatch
        }

        float sum = 0;
        for (int i = 0; i < A.row; ++i)
        {
            sum += A(i, 0) * B(i, 0);
        }

        return sum;
    }

    /**
     * @name Mat::solve
     * @brief Solve the linear system Ax = b using Gaussian elimination.
     *
     * @param A Coefficient matrix (NxN)
     * @param b Result vector (Nx1)
     * @return Mat Solution vector (Nx1) containing the roots of the equation Ax = b
     */
    Mat Mat::solve(const Mat &A, const Mat &b)
    {
        // Check if the matrix A is square
        if (A.row != A.col)
        {
            std::cerr << "[Error] Matrix A must be square for solving.\n";
            return Mat(); // Return empty matrix
        }

        // Check if A and b dimensions are compatible for solving
        if (A.row != b.row || b.col != 1)
        {
            std::cerr << "[Error] Matrix dimensions do not match for solving.\n";
            return Mat(); // Return empty matrix
        }

        // Create augmented matrix [A | b]
        Mat augmentedMatrix(A.row, A.col + 1);
        for (int i = 0; i < A.row; ++i)
        {
            for (int j = 0; j < A.col; ++j)
            {
                augmentedMatrix(i, j) = A(i, j); // Copy matrix A into augmented matrix
            }
            augmentedMatrix(i, A.col) = b(i, 0); // Copy vector b into augmented matrix
        }

        // Perform Gaussian elimination
        for (int i = 0; i < A.row; ++i)
        {
            // Find pivot and make sure it's non-zero
            if (augmentedMatrix(i, i) == 0)
            {
                std::cerr << "[Error] Pivot is zero, matrix is singular.\n";
                return Mat(); // Return empty matrix
            }

            // Normalize the pivot row
            float pivot = augmentedMatrix(i, i);
            for (int j = i; j < augmentedMatrix.col; ++j)
            {
                augmentedMatrix(i, j) /= pivot; // Normalize the pivot row
            }

            // Eliminate the entries below the pivot
            for (int j = i + 1; j < A.row; ++j)
            {
                float factor = augmentedMatrix(j, i);
                for (int k = i; k < augmentedMatrix.col; ++k)
                {
                    augmentedMatrix(j, k) -= factor * augmentedMatrix(i, k);
                }
            }
        }

        // Back-substitution to find the solution
        Mat solution(A.row, 1);
        for (int i = A.row - 1; i >= 0; --i)
        {
            float sum = augmentedMatrix(i, A.col);
            for (int j = i + 1; j < A.row; ++j)
            {
                sum -= augmentedMatrix(i, j) * solution(j, 0);
            }
            solution(i, 0) = sum;
        }

        return solution;
    }

    /**
     * @name Mat::band_solve
     * @brief Solve the system of equations Ax = b using optimized Gaussian elimination for banded matrices.
     *
     * @param A Coefficient matrix (NxN) - banded matrix
     * @param b Result vector (Nx1)
     * @param k Bandwidth of the matrix (the width of the non-zero bands)
     * @return Mat Solution vector (Nx1) containing the roots of the equation Ax = b
     */
    Mat Mat::band_solve(Mat A, Mat b, int k)
    {
        // Dimension compatibility check
        if (A.row != A.col) // Check if A is a square matrix
        {
            std::cerr << "[Error] Matrix A must be square for solving.\n";
            return Mat(); // Return an empty matrix in case of an error
        }

        if (A.row != b.row || b.col != 1) // Check if dimensions of A and b are compatible
        {
            std::cerr << "[Error] Matrix dimensions are not compatible for solving.\n";
            return Mat(); // Return an empty matrix in case of an error
        }

        int bandsBelow = (k - 1) / 2; // Number of bands below the main diagonal

        // Perform forward elimination to reduce the matrix
        for (int i = 0; i < A.row; ++i)
        {
            if (A(i, i) == 0)
            {
                // Pivot 0 - error
                std::cerr << "[Error] Zero pivot detected in bandSolve. Cannot proceed.\n";
                Mat err_result(b.row, 1);
                memset(err_result.data, 0, b.row * sizeof(float));
                return err_result;
            }

            float a_ii = 1 / A(i, i); // Inverse of the pivot element

            // Eliminate elements below the pivot in the current column
            for (int j = i + 1; j < A.row && j <= i + bandsBelow; ++j)
            {
                if (A(j, i) != 0)
                {
                    float factor = A(j, i) * a_ii;
                    for (int k = i; k < A.col; ++k)
                    {
                        A(j, k) -= A(i, k) * factor; // Eliminate the element
                    }
                    b(j, 0) -= b(i, 0) * factor; // Update the result vector
                    A(j, i) = 0;                 // Set the element to zero as it has been eliminated
                }
            }
        }

        // Back substitution to solve for x
        Mat x(b.row, 1);
        x(x.row - 1, 0) = b(x.row - 1, 0) / A(x.row - 1, x.row - 1); // Solve the last variable

        for (int i = x.row - 2; i >= 0; --i)
        {
            float sum = 0;
            for (int j = i + 1; j < x.row; ++j)
            {
                sum += A(i, j) * x(j, 0); // Sum of the known terms
            }
            x(i, 0) = (b(i, 0) - sum) / A(i, i); // Solve for the current variable
        }

        return x; // Return the solution vector
    }

    /**
     * @name Mat::roots(Mat A, Mat y)
     * @brief   Solve the matrix using a different method. Another implementation of the 'solve' function, no difference in principle.
     *
     * This method solves the linear system A * x = y using Gaussian elimination.
     *
     * @param[in] A: matrix [N]x[N] with input coefficients
     * @param[in] y: vector [N]x[1] with result values
     *
     * @return
     *      - matrix [N]x[1] with roots
     */
    Mat Mat::roots(Mat A, Mat y)
    {
        int n = A.col; // Number of rows and columns in A (assuming A is square)

        // Create augmented matrix [A | y]
        Mat augmentedMatrix = Mat::augment(A, y);

        // Perform Gaussian elimination
        for (int j = 0; j < n; j++)
        {
            // Normalize the pivot row (make pivot element equal to 1)
            float pivot = augmentedMatrix(j, j);
            if (pivot == 0)
            {
                std::cerr << "[Error] Pivot is zero, system may have no solution." << std::endl;
                return Mat(); // Return an empty matrix in case of an error
            }

            for (int k = 0; k < augmentedMatrix.col; k++)
            {
                augmentedMatrix(j, k) /= pivot;
            }

            // Eliminate the column below the pivot (set other elements in the column to zero)
            for (int i = j + 1; i < n; i++)
            {
                float factor = augmentedMatrix(i, j);
                for (int k = 0; k < augmentedMatrix.col; k++)
                {
                    augmentedMatrix(i, k) -= factor * augmentedMatrix(j, k);
                }
            }
        }

        // Perform back-substitution
        Mat result(n, 1);
        for (int i = n - 1; i >= 0; i--)
        {
            float sum = augmentedMatrix(i, n); // Right-hand side of the augmented matrix
            for (int j = i + 1; j < n; j++)
            {
                sum -= augmentedMatrix(i, j) * result(j, 0); // Subtract the known terms
            }
            result(i, 0) = sum; // Solve for the current variable
        }

        return result;
    }

    /* === Stream Operators === */
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
        for (int i = 0; i < m.row; ++i)
        {
            for (int j = 0; j < m.col; ++j)
            {
                is >> m(i, j);
            }
        }
        return is;
    }

    /* === Global Arithmetic Operators === */
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
        if ((m1.row != m2.row) || (m1.col != m2.col))
        {
            std::cerr << "operator + Error: matrices do not have equal dimensions" << std::endl;
            Mat err_ret;
            return err_ret;
        }

        if (m1.sub_matrix || m2.sub_matrix)
        {
            Mat temp(m1.row, m2.col);
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
        if ((m1.row != m2.row) || (m1.col != m2.col))
        {
            std::cerr << "operator - Error: matrices do not have equal dimensions" << std::endl;
            Mat err_ret;
            return err_ret;
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
        if (m1.col != m2.row)
        {
            std::cerr << "operator * Error: matrices do not have correct dimensions" << std::endl;
            Mat err_ret;
            return err_ret;
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
        if (m.sub_matrix)
        {
            Mat temp(m.row, m.col);
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
            dspm_mulc_f32(m.data, temp.data, 1 / num, m.row, m.col, m.pad, temp.pad, 1, 1);
#else
            tiny_mat_multc_f32(m.data, temp.data, 1 / num, m.row, m.col, m.pad, temp.pad, 1, 1);
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
        if ((A.row != B.row) || (A.col != B.col))
        {
            std::cerr << "operator / Error: matrices do not have equal dimensions" << std::endl;
            Mat err_ret;
            return err_ret;
        }

        Mat temp(A.row, A.col);
        for (int row = 0; row < A.row; row++)
        {
            for (int col = 0; col < A.col; col++)
            {
                temp(row, col) = A(row, col) / B(row, col);
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
        if ((m1.col != m2.col) || (m1.row != m2.row))
        {
            return false;
        }

        for (int row = 0; row < m1.row; row++)
        {
            for (int col = 0; col < m1.col; col++)
            {
                if (m1(row, col) != m2(row, col))
                {
                    std::cout << "operator == Error: " << row << " " << col << ", m1.data=" << m1(row, col) << ", m2.data=" << m2(row, col) << std::endl;
                    return false;
                }
            }
        }

        return true;
    }
}
