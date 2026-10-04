# Basic operations

**Duplicate historical record:** the original Basic Operations output is identical to Object Foundation and contains A-group tests. It does not verify the B-group operations. The original block remains intact below.

Execution date, device and firmware commit are unrecorded. The complete stage output is preserved below. Label counts are not counts of independent test cases.

[Overview](overview.md) · [Full source and record](../tiny-matrix-test.md)

```text
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
step            1
memory          1
data pointer    0x3fce9a7c
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
step            4
memory          12
data pointer    0x3fce9a8c
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            0            0            0       |
           0            0            0            0       |
           0            0            0            0       |
<<< Matrix Elements

[A1.3] Constructor with Rows, Cols and Step
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
step            5
memory          15
data pointer    0x3fce9ac0
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
step            4
memory          12
data pointer    0x3fc9a4ac
temp pointer    0
ext_buff        1   (External buffer or View)
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            1            2            3       |
           4            5            6            7       |
           8            9           10           11       |
<<< Matrix Elements

[A1.5] Constructor with External Data and Step
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
step            5
memory          15
data pointer    0x3fc9a500
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
step            5
memory          15
data pointer    0x3fce9b00
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
           0            1            2            3       |      7 
           4            5            6            7       |2.97902e-41 
           8            9           10           11       | 1.6141 
<<< Matrix Elements


[A2: Element Access Tests]
[A2.1] Non-const Access
Matrix Info >>>
rows            2
cols            3
elements        6
paddings        0
step            3
memory          6
data pointer    0x3fce9a7c
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
step            3
memory          6
data pointer    0x3fce9a7c
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
step            5
memory          15
data pointer    0x3fc9a254
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
step            1
memory          1
data pointer    0x3fce9a98
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
step            5
memory          15
data pointer    0x3fc9a254
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
step            5
memory          15
data pointer    0x3fc9a254
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
step            5
memory          15
data pointer    0x3fc9a254
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
step            5
memory          10
data pointer    0x3fc9a26c
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
step            5
memory          10
data pointer    0x3fc9a26c
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
step            2
memory          4
data pointer    0x3fce9bd0
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
Matrix Elements >>>
         0.1          0.2       |
         0.4          0.5       |
<<< Matrix Elements

[A3.12] Copy ROI - Using ROI Structure
time for copy_roi using ROI structure: 34 ms
Matrix Info >>>
rows            2
cols            2
elements        4
paddings        0
step            2
memory          4
data pointer    0x3fce9be4
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

[A3.14.1] ROI area_roi() - Negative Dimensions
ROI(0, 0, -3, 4) area: 0 (Expected: 0) [PASS]
ROI(0, 0, 3, -4) area: 0 (Expected: 0) [PASS]

[A3.14.2] print_matrix() - step < col Safety Branch
Expect warning and bounded printing without out-of-bounds access:
Matrix Info >>>
rows            2
cols            4
elements        8
paddings        0
step            3
memory          8
data pointer    0x3fce9bf8
temp pointer    0
ext_buff        0
sub_matrix      0
<<< Matrix Info
[Warning] step < cols; printing only the first 3 column(s) per row to avoid out-of-bounds access.
Matrix Elements >>>
          10           11           12       |
          13           20           21       |
<<< Matrix Elements

step < col safety print completed [PASS]
[A3.15] Block
time for block: 43 ms
Matrix Info >>>
rows            2
cols            2
elements        4
paddings        0
step            2
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

[A3.16] Swap Rows
matB before swap rows:
Matrix Info >>>
rows            3
cols            4
elements        12
paddings        1
step            5
memory          15
data pointer    0x3fc9a254
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
step            5
memory          15
data pointer    0x3fc9a254
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
step            5
memory          15
data pointer    0x3fc9a254
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
step            5
memory          15
data pointer    0x3fc9a254
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
step            5
memory          15
data pointer    0x3fc9a254
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
step            5
memory          15
data pointer    0x3fc9a254
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
