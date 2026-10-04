# 辅助功能

原始输出的运行日期、硬件和固件提交未记录；下方完整保留该阶段输出。标签数量不是独立测试用例数量。

[测试概览](overview.md) · [完整源码与记录](../tiny-matrix-test.md)

```text
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

matE == matF after modification: False

[F2.10] In-place Matrix Multiplication Shape Change (operator*=)
matG *= matH -> shape: 2x4 (expected 2x4)
38 44 50 56
83 98 113 128


[F2.11] Element-wise Matrix Division Zero Check (operator/)
[Error] operator/: division by zero at (0, 1).
divA / divB with zero element: Empty (expected)
============ [tiny_matrix_test end] ============
```
