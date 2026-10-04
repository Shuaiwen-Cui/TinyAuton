# Data manipulation and operators

APIs follow the MATH project’s `middleware/tiny_math/mat/tiny_matrix.hpp`. Elements are `float`; check dimensions, layout and result status before use.

[Overview](../OVERVIEW/api.md) · [Historical tests](../TESTS/overview.md)

## DATA MANIPULATION {#data-manipulation}

### Copy other matrix into this matrix as a sub-matrix {#copy-other-matrix-into-this-matrix-as-a-sub-matrix}
```cpp
tiny_error_t Mat::copy_paste(const Mat &src, int row_pos, int col_pos);
```

**Description**:

Copies the specified source matrix into this matrix as a sub-matrix starting from the specified row and column positions, not sharing the data buffer.

**Parameters**:

- `const Mat &src` : Source matrix.

- `int row_pos` : Starting row position.

- `int col_pos` : Starting column position.

**Returns**:

tiny_error_t - Error code (TINY_OK on success).

### Copy header of other matrix to this matrix {#copy-header-of-other-matrix-to-this-matrix}
```cpp
tiny_error_t Mat::copy_head(const Mat &src);
```

**Description**:

Copies the header of the specified source matrix to this matrix, sharing the data buffer. All items copy the source matrix.

**Parameters**:

- `const Mat &src` : Source matrix.

**Returns**:

tiny_error_t - Error code.

### Get a view (shallow copy) of sub-matrix (ROI) from this matrix {#get-a-view-shallow-copy-of-sub-matrix-roi-from-this-matrix}
```cpp
Mat Mat::view_roi(int start_row, int start_col, int roi_rows, int roi_cols) const;
```

**Description**:

Gets a view (shallow copy) of the sub-matrix (ROI) from this matrix starting from the specified row and column positions.

**Parameters**:

- `int start_row` : Starting row position.

- `int start_col` : Starting column position.

- `int roi_rows` : Number of rows in the ROI.

- `int roi_cols` : Number of columns in the ROI.

!!! warning
    Unlike ESP-DSP, view_roi does not allow to setup stride as it will automatically calculate the stride based on the number of columns and paddings. The function will also refuse illegal requests, i.e., out of bound requests. 

### Get a view (shallow copy) of sub-matrix (ROI) from this matrix using ROI structure {#get-a-view-shallow-copy-of-sub-matrix-roi-from-this-matrix-using-roi-structure}
```cpp
Mat Mat::view_roi(const Mat::ROI &roi) const;
```

**Description**:

Gets a view (shallow copy) of the sub-matrix (ROI) from this matrix using the specified ROI structure. This function will call the previous function in low level by passing the ROI structure to the parameters.

**Parameters**:

- `const Mat::ROI &roi` : ROI structure.

### Get a replica (deep copy) of sub-matrix (ROI) {#get-a-replica-deep-copy-of-sub-matrix-roi}
```cpp
Mat Mat::copy_roi(int start_row, int start_col, int height, int width);
```

**Description**:

Gets a replica (deep copy) of the sub-matrix (ROI) from this matrix starting from the specified row and column positions. This function will return a new matrix object that does not share the data buffer with the original matrix.

**Parameters**:

- `int start_row` : Starting row position.

- `int start_col` : Starting column position.

- `int height` : Number of rows in the ROI.

- `int width` : Number of columns in the ROI.

### Get a replica (deep copy) of sub-matrix (ROI) using ROI structure {#get-a-replica-deep-copy-of-sub-matrix-roi-using-roi-structure}
```cpp
Mat Mat::copy_roi(const Mat::ROI &roi);
```

**Description**:

Gets a replica (deep copy) of the sub-matrix (ROI) from this matrix using the specified ROI structure. This function will call the previous function in low level by passing the ROI structure to the parameters.

**Parameters**:

- `const Mat::ROI &roi` : ROI structure.

### Get a block of matrix {#get-a-block-of-matrix}
```cpp
Mat Mat::block(int start_row, int start_col, int block_rows, int block_cols);
```

**Description**:

Gets a block of the matrix starting from the specified row and column positions.

**Parameters**:

- `int start_row` : Starting row position.

- `int start_col` : Starting column position.

- `int block_rows` : Number of rows in the block.

- `int block_cols` : Number of columns in the block.

!!! tip "Differences between view_roi | copy_roi | block"

    - `view_roi` : Shallow copy of the sub-matrix (ROI) from this matrix.

    - `copy_roi` : Deep copy of the sub-matrix (ROI) from this matrix. Rigid and faster.

    - `block` : Deep copy of the block from this matrix. Flexible and slower.

### Swap rows {#swap-rows}

```cpp
void Mat::swap_rows(int row1, int row2);
```

**Description**:

Swaps the specified rows in the matrix.

**Parameters**:

- `int row1` : First row index.

- `int row2` : Second row index.

**Returns**:

void

### Swap columns {#swap-columns}

```cpp
void Mat::swap_cols(int col1, int col2);
```

**Description**:

Swaps the specified columns in the matrix. 

**Parameters**:

- `int col1` : First column index.

- `int col2` : Second column index.

**Returns**:

void

### Clear matrix {#clear-matrix}

```cpp
void Mat::clear(void);
```

**Description**:

Clears the matrix by setting all elements to zero.

**Parameters**:

void

**Returns**:

void

## ARITHMETIC OPERATORS {#arithmetic-operators}

!!! INFO "In-Place Operations"
    This section defines the arithmetic operators that act on the current matrix itself (in-place operations). These operators modify the matrix and return a reference to it, enabling chained operations like `A += B += C`. The operators are optimized to handle padding and use DSP-accelerated functions when available.

### Copy assignment {#copy-assignment}
```cpp
Mat &operator=(const Mat &src);
```

**Description**:

Copy assignment operator for the matrix. Copies elements from source matrix to current matrix. Handles dimension changes by reallocating memory if necessary. Prevents assignment to sub-matrix views for safety.

**Mathematical Principle**:

Creates an independent copy of the source matrix. Unlike copy constructor, this is used for existing matrices.

**Parameters**:

- `const Mat &src` : Source matrix.

**Returns**:

Mat& - Reference to the current matrix (enables chaining).

**Usage Insights**:

- **Memory Management**: Automatically reallocates memory if dimensions differ. Frees old memory if it was internally allocated.

- **Sub-Matrix Protection**: Assignment to sub-matrix views is forbidden to prevent accidental data corruption.

- **Self-Assignment**: Handles self-assignment safely (A = A).

- **Performance**: O(n²) for n×n matrices. For large matrices, consider if a view would suffice.

### Add matrix {#add-matrix}
```cpp
Mat &operator+=(const Mat &A);
```

**Description**:

Adds the specified matrix to this matrix.

**Parameters**:

- `const Mat &A` : Matrix to be added.

### Add constant {#add-constant}
```cpp
Mat &operator+=(float C);
```

**Description**:

Element-wise addition of a constant to this matrix.

**Parameters**:

- `float C` : The constant to add.

**Returns**:

Mat& - Reference to the current matrix.

### Subtract matrix {#subtract-matrix}
```cpp
Mat &operator-=(const Mat &A);
```

**Description**:

Subtracts the specified matrix from this matrix.

**Parameters**:

- `const Mat &A` : Matrix to be subtracted.

### Subtract constant {#subtract-constant}
```cpp
Mat &operator-=(float C);
```

**Description**:

Element-wise subtraction of a constant from this matrix.

**Parameters**:

- `float C` : The constant to subtract.

**Returns**:

Mat& - Reference to the current matrix.

### Multiply matrix {#multiply-matrix}
```cpp
Mat &operator*=(const Mat &A);
```

**Description**:

Matrix multiplication: this = this * A. Performs standard matrix multiplication (not element-wise). The number of columns of the current matrix must equal the number of rows of A.

**Mathematical Principle**:

Matrix multiplication C = A * B where Cᵢⱼ = Σₖ Aᵢₖ * Bₖⱼ. This is the standard matrix product, not element-wise multiplication.

**Dimension Requirements**: 
- Current matrix: m × n

- Matrix A: n × p

- Result: m × p

**Parameters**:

- `const Mat &A` : Matrix to be multiplied (must have n rows, where n = current matrix columns).

**Returns**:

Mat& - Reference to the current matrix.

**Usage Insights**:

- **Memory Efficiency**: Creates a temporary copy to avoid overwriting data during computation, then updates the current matrix.

- **Padding Support**: Handles matrices with padding using specialized DSP functions when available.

- **Performance**: O(mnp) for m×n * n×p multiplication. Uses optimized DSP functions on ESP32 platform.

- **Common Mistake**: This is matrix multiplication, not element-wise. For element-wise, use a loop with `operator()()`.

### Multiply constant {#multiply-constant}
```cpp
Mat &operator*=(float C);
```

**Description**:

Element-wise multiplication by a constant.

**Parameters**:

- `float C` : The constant multiplier.

**Returns**:

Mat& - Reference to the current matrix.

### Divide matrix (element-wise) {#divide-matrix-element-wise}
```cpp
Mat &operator/=(const Mat &B);
```

**Description**:

Element-wise division: this = this / B.

**Parameters**:

- `const Mat &B` : The matrix divisor.

**Returns**:

Mat& - Reference to the current matrix.

### Divide constant {#divide-constant}
```cpp
Mat &operator/=(float C);
```

**Description**:

Element-wise division of this matrix by a constant.

**Parameters**:

- `float C` : The constant divisor.

**Returns**:

Mat& - Reference to the current matrix.

### Exponentiation {#exponentiation}
```cpp
Mat operator^(int C);
```

**Description**:

Element-wise integer exponentiation. Returns a new matrix where each element is raised to the given power.

**Parameters**:

- `int C` : The exponent (integer).

**Returns**:

Mat - New matrix after exponentiation.


## STREAM OPERATORS {#stream-operators}

### Matrix output stream operator {#matrix-output-stream-operator}
```cpp
std::ostream &operator<<(std::ostream &os, const Mat &m);
```

**Description**:

Overloaded output stream operator for the matrix.

**Parameters**:

- `std::ostream &os` : Output stream.

- `const Mat &m` : Matrix to be output.

### ROI output stream operator {#roi-output-stream-operator}
```cpp
std::ostream &operator<<(std::ostream &os, const Mat::ROI &roi);
```

**Description**:

Overloaded output stream operator for the ROI structure.

**Parameters**:

- `std::ostream &os` : Output stream.

- `const Mat::ROI &roi` : ROI structure.

### Matrix input stream operator {#matrix-input-stream-operator}
```cpp
std::istream &operator>>(std::istream &is, Mat &m);
```

**Description**:

Overloaded input stream operator for the matrix.

**Parameters**:

- `std::istream &is` : Input stream.

- `Mat &m` : Matrix to be input.

!!! tip 
    This section is actually kind of overlapping with print function in terms of showing the matrix.

## GLOBAL ARITHMETIC OPERATORS {#global-arithmetic-operators}

!!! INFO "Non-Modifying Operations"
    The operators in this section return a new matrix object, which is the result of the operation. The original matrices remain unchanged. These are functional-style operations that don't modify their operands, making them safe for use with const references and temporary objects.
    
!!! TIP "When to Use"
    - Use global operators (A + B) when you want to preserve original matrices
    - Use member operators (A += B) when you want to modify the matrix in-place (more memory efficient)


### Add matrix {#add-matrix_1}
```cpp
Mat operator+(const Mat &A, const Mat &B);
```

**Description**:

Adds two matrices element-wise.

**Parameters**:

- `const Mat &A` : First matrix.

- `const Mat &B` : Second matrix.

**Returns**:

Mat - Result matrix A+B.

### Add constant {#add-constant_1}
```cpp
Mat operator+(const Mat &A, float C);
```

**Description**:

Adds a constant to a matrix element-wise.

**Parameters**:

- `const Mat &A` : Input matrix A.

- `float C` : Input constant.

**Returns**:

Mat - Result matrix A+C.

### Subtract matrix {#subtract-matrix_1}
```cpp
Mat operator-(const Mat &A, const Mat &B);
```

**Description**:

Subtracts two matrices element-wise.

**Parameters**:

- `const Mat &A` : First matrix.

- `const Mat &B` : Second matrix.

**Returns**:

Mat - Result matrix A-B.

### Subtract constant {#subtract-constant_1}
```cpp
Mat operator-(const Mat &A, float C);
```

**Description**:

Subtracts a constant from a matrix element-wise.

**Parameters**:

- `const Mat &A` : Input matrix A.

- `float C` : Input constant.

**Returns**:

Mat - Result matrix A-C.

### Multiply matrix {#multiply-matrix_1}
```cpp
Mat operator*(const Mat &A, const Mat &B);
```

**Description**:

Multiplies two matrices (matrix multiplication).

**Parameters**:

- `const Mat &A` : First matrix.

- `const Mat &B` : Second matrix.

**Returns**:

Mat - Result matrix A*B.

### Multiply constant {#multiply-constant_1}
```cpp
Mat operator*(const Mat &A, float C);
```

**Description**:

Multiplies a matrix by a constant element-wise.

**Parameters**:

- `const Mat &A` : Input matrix A.

- `float C` : Floating point value.

**Returns**:

Mat - Result matrix A*C.

### Multiply constant (left side) {#multiply-constant-left-side}
```cpp
Mat operator*(float C, const Mat &A);
```

**Description**:

Multiplies a constant by a matrix element-wise.

**Parameters**:

- `float C` : Floating point value.

- `const Mat &A` : Input matrix A.

**Returns**:

Mat - Result matrix C*A.


### Divide matrix (by constant) {#divide-matrix-by-constant}
```cpp
Mat operator/(const Mat &A, float C);
```

**Description**:

Divides a matrix by a constant element-wise.

**Parameters**:

- `const Mat &A` : Input matrix A.

- `float C` : Floating point value.

**Returns**:

Mat - Result matrix A/C.

### Divide matrix (element-wise) {#divide-matrix-element-wise_1}
```cpp
Mat operator/(const Mat &A, const Mat &B);
```

**Description**:

Divides matrix A by matrix B element-wise.

**Parameters**:

- `const Mat &A` : Input matrix A.

- `const Mat &B` : Input matrix B.

**Returns**:

Mat - Result matrix C, where C[i,j] = A[i,j]/B[i,j].

### Equality check {#equality-check}
```cpp
bool operator==(const Mat &A, const Mat &B);
```

**Description**:

Checks if the specified matrices are equal.

**Parameters**:

- `const Mat &A` : First matrix.

- `const Mat &B` : Second matrix.

**Returns**:

bool - true if equal, false otherwise.
