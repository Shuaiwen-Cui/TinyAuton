# Objects, memory and element access

APIs follow the MATH project’s `middleware/tiny_math/mat/tiny_matrix.hpp`. Elements are `float`; check dimensions, layout and result status before use.

[Overview](../OVERVIEW/api.md) · [Historical tests](../TESTS/overview.md)

## MATRIX METADATA {#matrix-metadata}

!!! INFO "Matrix Structure"
    The Mat class uses a row-major storage layout with support for padding and step. In this class, `stride` is a backward-compatible alias of `step`. This design enables efficient memory access patterns and compatibility with DSP libraries.

### Core Dimensions {#core-dimensions}

- **`int row`** : Number of rows in the matrix.

- **`int col`** : Number of columns in the matrix.

- **`int element`** : Total number of elements = rows × cols.

### Memory Layout {#memory-layout}

- **`int step` / `int stride`** : Total elements per row (logical columns + padding). In this class they are equivalent, and `stride` is an alias of `step`.

- **`int pad`** : Number of padding elements between two rows. Padding is used for memory alignment and DSP optimization.

- **`int memory`** : Size of the data buffer = rows × step (in number of float elements).

### Data Pointers {#data-pointers}

- **`float *data`** : Pointer to the data buffer containing matrix elements. Elements are stored in row-major order: element at (i, j) is at `data[i * step + j]`.

- **`float *temp`** : Pointer to the temporary data buffer (if allocated). Used internally for certain operations.

### Memory Management Flags {#memory-management-flags}

- **`bool ext_buff`** : Flag indicating that the matrix uses an external buffer. When `true`, the destructor will not free the memory (caller is responsible).

- **`bool sub_matrix`** : Flag indicating that the matrix is a subset/view of another matrix. When `true`, the matrix shares data with the parent matrix.

!!! TIP "Memory Layout Example"
    For a 3×4 matrix with stride=4 (no padding):
    ```
    [a b c d]   row 0: data[0*4+0] to data[0*4+3]
    [e f g h]   row 1: data[1*4+0] to data[1*4+3]
    [i j k l]   row 2: data[2*4+0] to data[2*4+3]
    ```
    
    For a 3×4 matrix with stride=6 (padding=2):
    ```
    [a b c d _ _]   row 0: data[0*6+0] to data[0*6+3], padding at data[0*6+4,5]
    [e f g h _ _]   row 1: data[1*6+0] to data[1*6+3], padding at data[1*6+4,5]
    [i j k l _ _]   row 2: data[2*6+0] to data[2*6+3], padding at data[2*6+4,5]
    ```

## ROI STRUCTURE {#roi-structure}

!!! INFO "Region of Interest"
    The ROI (Region of Interest) structure represents a rectangular subregion of a matrix. It's used with `view_roi()` and `copy_roi()` functions to extract or reference submatrices efficiently.

### ROI Metadata {#roi-metadata}

- **`int pos_x`** : Starting column index (x-coordinate of the top-left corner).

- **`int pos_y`** : Starting row index (y-coordinate of the top-left corner).

- **`int width`** : Width of the ROI in columns.

- **`int height`** : Height of the ROI in rows.

### ROI Constructor {#roi-constructor}

```cpp
Mat::ROI::ROI(int pos_x = 0, int pos_y = 0, int width = 0, int height = 0);
```

**Description**: 

ROI constructor initializes the ROI with the specified position and size.

**Parameters**:

- `int pos_x` : Starting column index.

- `int pos_y` : Starting row index.

- `int width` : Width of the ROI (columns).

- `int height` : Height of the ROI (rows).

### ROI RESIZE {#roi-resize}

```cpp
void Mat::ROI::resize_roi(int pos_x, int pos_y, int width, int height);
```

**Description**: 

Resizes the ROI to the specified position and size.

**Parameters**:

- `int pos_x` : Starting column index.

- `int pos_y` : Starting row index.

- `int width` : Width of the ROI (columns).

- `int height` : Height of the ROI (rows).

**Returns**:

void

### AREA ROI {#area-roi}

```cpp
int Mat::ROI::area_roi(void) const;
```

**Description**: 

Calculates the area of the ROI.

**Parameters**:

void

**Returns**:

int - Area of the ROI.

## PRINTING FUNCTIONS {#printing-functions}

!!! TIP "Debugging Tools"
    These functions are essential for debugging and understanding matrix state. Use them to verify matrix dimensions, memory layout, and data values during development.

### Print Matrix Information {#print-matrix-information}

```cpp
void Mat::print_info() const;
```

**Description**: 

Prints comprehensive matrix information including:

- Dimensions: rows, columns, elements

- Memory layout: paddings, stride, memory size

- Pointers: data buffer address, temporary buffer address

- Flags: external buffer usage, sub-matrix status

- Warnings: dimension mismatches, invalid states

**Parameters**:

void

**Returns**:

void

**Usage Insights**:

- **Debugging**: Essential for verifying matrix state and detecting memory issues.

- **Memory Analysis**: Shows actual memory usage vs. logical size, helping identify memory inefficiencies.

- **Sub-Matrix Detection**: Clearly indicates if a matrix is a view, which affects memory management.

### Print Matrix Elements {#print-matrix-elements}

```cpp
   void Mat::print_matrix(bool show_padding) const;
```

**Description**: 

Prints the matrix elements in a formatted table. Optionally displays padding elements separated by a visual separator.

**Parameters**: 

- `bool show_padding` : If `true`, displays padding values with a separator `|`. If `false`, only shows actual matrix elements.

**Returns**:

void

**Usage Insights**:

- **Formatting**: Elements are formatted with fixed width (12 characters) for alignment.

- **Padding Visualization**: The `show_padding` option helps understand memory layout and verify padding values.

- **Large Matrices**: For very large matrices, consider using `view_roi()` to print specific regions.

## CONSTRUCTORS & DESTRUCTOR {#constructors-destructor}

!!! INFO "Memory Management"
    Constructors handle memory allocation automatically. The destructor safely frees memory only if it was internally allocated (not external buffers or views). Always check the `data` pointer after construction to ensure successful allocation.

### Memory Allocation {#memory-allocation}

```cpp
void Mat::alloc_mem();
```

**Description**: 

Internal function that allocates memory for the matrix according to the computed memory requirements. Sets `ext_buff = false` and allocates `row * step` float elements.

**Parameters**:

void

**Returns**:

void

**Usage Insights**:

- **Automatic Call**: Called automatically by constructors. Rarely needs manual invocation.

- **Memory Calculation**: Allocates `row * step` elements, which may include padding.

- **Error Handling**: If allocation fails, `data` remains `nullptr`. Always check `data` after construction.

### Default Constructor {#default-constructor}

```cpp
Mat::Mat();
```

**Description**: 

Default constructor creates a 1×1 zero matrix. This is useful for initialization and as a return value for error cases.

**Mathematical Principle**:

Creates the identity element for matrix operations in some contexts, though typically you'll want to specify dimensions.

**Parameters**:

void

**Returns**:

Mat - A 1×1 matrix with element = 0.

### Constructor - Mat(int rows, int cols) {#constructor-matint-rows-int-cols}

```cpp
Mat::Mat(int rows, int cols);
```

**Description**: 

Constructor creates a matrix with specified dimensions. All elements are initialized to zero. This is the most commonly used constructor.

**Parameters**:

- `int rows` : Number of rows (must be > 0).

- `int cols` : Number of columns (must be > 0).

**Returns**:

Mat - A rows×cols matrix with all elements initialized to 0.

**Usage Insights**:

- **Zero Initialization**: All elements are set to zero using `memset`, ensuring clean state.

- **Memory Layout**: Creates a contiguous memory layout with no padding (stride = cols).

- **Error Handling**: If memory allocation fails, `data` will be `nullptr`. Always verify allocation success.

### Constructor - Mat(int rows, int cols, int step) {#constructor-matint-rows-int-cols-int-step}

```cpp
Mat::Mat(int rows, int cols, int step);
```

**Description**: 

Constructor creates a matrix with specified dimensions and step. Useful when you need padding for memory alignment or DSP optimization.

**Parameters**:

- `int rows` : Number of rows.

- `int cols` : Number of columns.

- `int step` : Total elements per row (must be ≥ cols). Padding = step - cols.

**Returns**:

Mat - A rows×cols matrix with step, all elements initialized to 0.

**Usage Insights**:

- **DSP Optimization**: Some DSP libraries require aligned memory. Use stride to ensure proper alignment.

- **Memory Efficiency**: Padding allows efficient vectorized operations on aligned boundaries.

- **Compatibility**: Enables compatibility with external libraries that use strided memory layouts.

### Constructor - Mat(float *data, int rows, int cols) {#constructor-matfloat-data-int-rows-int-cols}

```cpp
Mat::Mat(float *data, int rows, int cols);
```

**Description**: 

Constructor creates a matrix view over an external data buffer. The matrix does not own the memory; the caller is responsible for managing it. Useful for interfacing with existing data arrays.

**Parameters**:

- `float *data` : Pointer to external data buffer (must remain valid for matrix lifetime).

- `int rows` : Number of rows.

- `int cols` : Number of columns.

**Returns**:

Mat - A matrix view with `ext_buff = true`.

**Usage Insights**:

- **Zero-Copy**: No memory copy occurs; the matrix directly references external data.

- **Lifetime Management**: The external buffer must remain valid while the matrix exists. The destructor will not free this memory.

- **Data Layout**: Assumes row-major layout with no padding (stride = cols).

- **Use Cases**: 

  - Wrapping C arrays

  - Interfacing with other libraries

  - Avoiding unnecessary copies

### Constructor - Mat(float *data, int rows, int cols, int step) {#constructor-matfloat-data-int-rows-int-cols-int-step}

```cpp
Mat::Mat(float *data, int rows, int cols, int step);
```

**Description**: 

Constructor creates a matrix view over an external data buffer with specified step. Supports padded row-major layouts for DSP compatibility.

**Parameters**:

- `float *data` : Pointer to external data buffer (must remain valid for matrix lifetime).

- `int rows` : Number of rows.

- `int cols` : Number of columns.

- `int step` : Total elements per row (must be ≥ cols).

**Returns**:

Mat - A matrix view with `ext_buff = true` and specified step.

**Usage Insights**:

- **Strided Layouts**: Essential for working with DSP libraries that use strided memory layouts.

- **Memory Safety**: Same lifetime requirements as the previous constructor - external buffer must remain valid.

- **Padding Support**: Can handle buffers with padding between rows.

### Copy Constructor - Mat(const Mat &src) {#copy-constructor-matconst-mat-src}

```cpp
Mat::Mat(const Mat &src);
```

**Description**: 

Copy constructor creates a new matrix from a source matrix. Uses intelligent copying: deep copy for regular matrices, shallow copy for sub-matrix views.

**Copy Strategy**:

- **Regular matrices**: Deep copy - allocates new memory and copies all data

- **Sub-matrix views**: Shallow copy - shares data with source (creates another view)

**Parameters**:

- `const Mat &src` : Source matrix.

**Returns**:

Mat - A new matrix with copied or shared data depending on source type.

**Usage Insights**:

- **Automatic Selection**: Automatically chooses deep or shallow copy based on source matrix type.

- **Memory Efficiency**: Sub-matrix views are copied shallowly to avoid unnecessary memory allocation.

- **Independence**: Deep copies are independent; modifications don't affect the source.

### Destructor {#destructor}

```cpp
Mat::~Mat();
```

**Description**: 

Destructor safely releases allocated memory. Only frees memory if it was internally allocated (`ext_buff = false`). External buffers and views are not freed.

**Memory Management**:

- Frees `data` buffer if `ext_buff = false`

- Frees `temp` buffer if allocated

- Does nothing for external buffers or views

**Parameters**:

void

**Returns**:

void

!!! note "Constructor and Destructor Rules"
    - Constructor functions must have the same name as the class and no return type
    - C++ allows function overloading by changing parameter number/order
    - The destructor is automatically called when the object goes out of scope
    - Always check `data != nullptr` after construction to verify successful allocation

## ELEMENT ACCESS {#element-access}

!!! INFO "Matrix Indexing"
    The Mat class uses operator overloading to provide intuitive matrix element access. The `operator()` allows natural syntax like `A(i, j)` instead of `A.data[i * step + j]`. The implementation automatically handles step and padding.

### Access Matrix Elements (Non-Const) {#access-matrix-elements-non-const}

```cpp
inline float &operator()(int row, int col);
```

**Description**:

Accesses matrix elements with read-write capability. Returns a reference to the element, allowing both reading and modification.

**Mathematical Principle**:

Element at position (row, col) is accessed as `data[row * step + col]`, where step accounts for padding.

**Parameters**：

- `int row` : Row index (0-based, must be in range [0, row-1]).

- `int col` : Column index (0-based, must be in range [0, col-1]).

**Returns**:

`float&` - Reference to the matrix element, enabling modification.

**Usage Insights**:

- **Bounds Checking**: No automatic bounds checking for performance. Ensure indices are valid.

- **Stride Handling**: Automatically accounts for stride, so it works correctly with padded matrices.

- **Performance**: Inline function with minimal overhead, suitable for tight loops.

- **Example**: `A(2, 3) = 5.0f;` sets element at row 2, column 3 to 5.0.

### Access Matrix Elements (Const) {#access-matrix-elements-const}

```cpp
inline const float &operator()(int row, int col) const;
```

**Description**:

Accesses matrix elements in read-only mode. Returns a const reference, preventing modification. Used when the matrix is const.

**Parameters**：

- `int row` : Row index (0-based).

- `int col` : Column index (0-based).

**Returns**:

`const float&` - Const reference to the matrix element (read-only).

**Usage Insights**:

- **Const Correctness**: Enables proper const-correct code. Use this version in const member functions.

- **Safety**: Prevents accidental modification of const matrices.

!!! note "Operator Overloading"
    These functions overload the `()` operator, enabling natural matrix indexing syntax:
    ```cpp
    Mat A(3, 4);
    A(1, 2) = 3.14f;        // Write access
    float val = A(1, 2);    // Read access
    const Mat& B = A;
    float val2 = B(1, 2);   // Read-only access (uses const version)
    ```
