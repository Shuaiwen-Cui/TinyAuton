# 说明

!!! note "说明"
    `tiny::Tensor` 是 `tiny_ai` 中所有数据流的载体，提供最多 4 维的 float32 张量、PSRAM 感知的内存分配、与 `tiny::Mat` 的零拷贝互通。它是激活、层、损失等其它模块的输入输出类型。

## 张量概念

### 形状与 size

```cpp
class Tensor
{
    int   ndim;        // 1 ~ 4
    int   shape[4];    // 未使用的维度填 1
    int   size;        // shape[0]*shape[1]*shape[2]*shape[3]
    float *data;       // 行主序 flat 缓冲区
    bool  owns_data;   // 析构时是否释放
};
```

- **行主序**：4 维张量元素 `(i, j, k, l)` 对应缓冲区下标 `((i * shape[1] + j) * shape[2] + k) * shape[3] + l`。
- **`size = shape[0] * shape[1] * shape[2] * shape[3]`**：未使用的维度填 1，因此 `size` 始终等于元素总数。
- **所有权**：构造函数申请的张量 `owns_data = true`，析构时自动释放；`from_data()` 创建的视图 `owns_data = false`，由调用方保证生命周期。

### 常见形状约定

| 模块 | 形状 | 说明 |
| --- | --- | --- |
| Dense | `[batch, features]` | 二维 |
| Conv1D | `[batch, channels, length]` | 三维 |
| Conv2D | `[batch, channels, height, width]` | 四维 |
| Attention | `[batch, seq_len, embed_dim]` | 三维 |
| GlobalAvgPool 输入 | `[batch, seq_len, feat]` → 输出 `[batch, feat]` | 序列到向量 |

## 构造与命名构造

### 直接构造

```cpp
Tensor t1(N);              // 1-D
Tensor t2(N, F);           // 2-D
Tensor t3(N, C, L);        // 3-D
Tensor t4(N, C, H, W);     // 4-D
Tensor t0;                 // empty (no allocation)
```

构造时调用内部 `alloc()` 申请 `size * sizeof(float)` 字节，`memset` 清零。

### 命名构造

```cpp
Tensor::zeros(n0);                          // 1-D 零张量
Tensor::zeros(n0, n1);                      // 2-D 零张量
Tensor::zeros(n0, n1, n2);                  // 3-D 零张量
Tensor::zeros(n0, n1, n2, n3);              // 4-D 零张量

Tensor::zeros_like(other);                  // 与 other 形状一致的零张量

Tensor::from_data(buf, ndim, shape);        // 包装外部缓冲区；不拷贝、不接管
```

!!! warning "from_data 的所有权"
    `from_data()` 返回的 Tensor 不拥有数据，调用方必须保证 `buf` 在张量销毁前不会被释放或重定位（例如把权重放在 PSRAM，但确保它的生命周期 ≥ 模型对象）。

## 元素访问

`Tensor` 提供 `at()` 内联重载，按维度数自动计算下标：

```cpp
inline float &at(int i);
inline float &at(int i, int j);
inline float &at(int i, int j, int k);
inline float &at(int i, int j, int k, int l);
```

也提供 const 版本。`at()` 全部为 `inline`，不进行越界检查 —— 这是为了在 ESP32-S3 这样的 MCU 上提供与裸数组相当的访问速度。

```cpp
Tensor x(B, C, H, W);
for (int b = 0; b < B; b++)
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                x.at(b, c, h, w) = some_value;
```

直接读写 `x.data[idx]` 也是允许的，等价于内部行主序展平。

## 形状语义访问器

```cpp
int batch();     // ndim>=3 ? shape[0] : 1
int rows();      // ndim>=2 ? shape[ndim-2] : shape[0]
int cols();      // shape[ndim-1]
int channels();  // ndim==4 ? shape[1] : 1
```

这些访问器允许激活函数、损失函数等组件以「最后一维 = 类别 / 特征」的统一方式处理 2D/3D 张量。例如 `softmax_inplace` 用 `rows = size / cols, cls = cols` 沿最后一维做归一化。

## 原地操作

```cpp
void zero();                          // memset 0
void fill(float val);                 // 填充常量
void copy_from(const Tensor &src);    // 形状必须一致
Tensor clone() const;                 // 深拷贝（owns_data=true）
```

## 形状重塑

```cpp
tiny_error_t reshape(int ndim, const int *new_shape);   // 元素总数不变
tiny_error_t reshape_2d(int n0, int n1);
tiny_error_t reshape_3d(int n0, int n1, int n2);
```

reshape 只改 `ndim` / `shape`，不重新分配缓冲区。若 `new_shape` 总元素数 ≠ `size` 则返回 `TINY_ERR_AI_INVALID_SHAPE`。

`Attention` 例子里就利用 `reshape_3d / reshape_2d` 在 `[B, F]` 与 `[B, T, E]` 之间来回切换。

## 与 `tiny::Mat` 互通

```cpp
Mat to_mat() const;
```

仅对 2 维张量有意义：返回的 `tiny::Mat` 共享同一缓冲区（不拷贝、不拥有）。当上游需要矩阵乘法 / 行列操作时（例如 Dense 层）可借此复用 `tiny_math` 的高性能实现。

```cpp
Tensor x(B, F);
Mat xm = x.to_mat();          // [B, F] 视图，零拷贝
// 通过 xm 调用 tiny_math 的矩阵 API
```

## 形状比较与打印

```cpp
bool same_shape(const Tensor &other) const;
void print(const char *name = "") const;
```

`print()` 显示形状元数据并最多打印前 32 个元素，便于在串口里快速调试。

## 复制 / 移动

`Tensor` 实现了规则五大件：

- 拷贝构造 / 拷贝赋值 → 深拷贝（重新 `alloc()` + `memcpy`）。
- 移动构造 / 移动赋值 → 接管 `data` 指针并把对方置空。
- 析构 → 仅当 `owns_data == true` 时调用 `TINY_AI_FREE(data)`。

因此可以放心按值传递、放入 `std::vector<Tensor>` 等容器。
