# C++ 矩阵接口概览

`tiny::Mat` 使用 `float`、行主序和可选行尾填充。对象拥有的缓冲区随析构释放；外部缓冲区和 ROI 视图必须由调用方保证生命周期。

| 阅读目标 | 页面 |
|---|---|
| 构造、复制、外部缓冲区、布局 | [对象与内存](../CORE/api.md) |
| ROI、数据操作、算术与流 | [操作与运算符](../ARITH/api.md) |
| 行列式、逆、线性系统 | [线性系统](../LINALG/api.md) |
| LU、Cholesky、QR、SVD 与求解 | [矩阵分解](../DECOMP/api.md) |
| 幂迭代、Jacobi、QR | [特征值](../EIGEN/api.md) |
| 完整头文件及旧链接 | [完整接口记录](../tiny-matrix-api.md) |
| 实现与验证 | [源码](../tiny-matrix-code.md) · [测试概览](../TESTS/overview.md) |

## 最小用法

```cpp
#include "tiny_matrix.hpp"

void matrix_demo()
{
    tiny::Mat A(2, 2);
    if (!A.data) return;
    A(0, 0) = 4.0f; A(0, 1) = 1.0f;
    A(1, 0) = 1.0f; A(1, 1) = 3.0f;
    auto lu = A.lu_decompose();
    if (lu.status != TINY_OK) return;
    lu.L.print_matrix(false);
    lu.U.print_matrix(false);
}
```

## 使用边界

- 对象字段为 `row/col/step`；在该类中 `stride` 是 `step` 的旧别名。元素位置为 `data[i * step + j]`。
- C `tiny_mat` 的 padding/stride 参数不能按 C++ 对象字段机械替换。
- 分解要求和失败状态随方法而异，QR 特征值接口仅返回复数结果的实部。
- 数值方法的时延、体积和稳定性需要按具体输入与构建测量；本页不继承未附条件的性能倍数。
