# 说明

!!! note "说明"
    `Dataset` 在 `tiny_ai` 中负责把外部 float32 矩阵 / 标签数组包装成可洗牌、可拆分、可迭代的训练数据集。它仅持有索引数组，原始数据保持只读视图，便于把数据嵌入只读段或 PSRAM。

## 类定义

```cpp
class Dataset
{
public:
    Dataset(const float *X, const int *y,
            int n_samples, int n_features, int n_classes);

    Dataset();
    Dataset(const Dataset &);
    Dataset(Dataset &&) noexcept;
    Dataset &operator=(const Dataset &);
    Dataset &operator=(Dataset &&) noexcept;
    ~Dataset();

    void   shuffle(uint32_t seed = 0);
    void   reset();
    int    next_batch(Tensor &X_batch, int *y_batch, int batch_size);

    void   split(float test_ratio, Dataset &train_out, Dataset &test_out,
                 uint32_t seed = 0) const;

    int    n_samples()  const;
    int    n_features() const;
    int    n_classes()  const;

    Tensor to_tensor() const;
};
```

## 数据约定

- `X` 是行主序的 `n_samples × n_features` 浮点矩阵，存储在调用方拥有的内存（典型为 `iris_data.hpp` / `signal_data.hpp` 中的 `static const float[] = {...}`）。
- `y` 是长度为 `n_samples` 的整数数组，元素是类别下标。
- `Dataset` 仅持有 `X` / `y` 的指针视图与一个 `int *indices_` 数组，析构时只释放 `indices_`。
- 拷贝 / 移动 `Dataset` 不会复制原始数据，仅复制 / 接管 `indices_`。

## shuffle / split

```cpp
void shuffle(uint32_t seed = 0);
```

使用 LCG 随机数 + Fisher-Yates 重排 `indices_`，并把 `cursor_` 重置为 0。`seed = 0` 时用默认种子 `1234567891u`。

```cpp
void split(float test_ratio, Dataset &train_out, Dataset &test_out, uint32_t seed = 0) const;
```

- 计算 `n_test = round(n_samples * test_ratio)`，至少 1，至多 `n_samples - 1`。
- 复制并洗牌一份索引数组，前 `n_train` 个分给 `train_out`，后 `n_test` 个分给 `test_out`。
- 内部使用私有构造函数 `Dataset(X, y, n, F, C, given_indices)` 让两个子集各自拥有自己的索引副本。

```cpp
Dataset full(X, y, N, F, C);
Dataset train, test;
full.split(0.2f, train, test, 42);
```

## next_batch 迭代

```cpp
int next_batch(Tensor &X_batch, int *y_batch, int batch_size);
```

- 从 `indices_[cursor_]` 开始取 `actual = min(batch_size, n_samples - cursor_)` 个样本。
- 若 `X_batch` 的 size 不等于 `actual * n_features`，则重新分配为 `Tensor(actual, n_features)`。
- 把每行从原始 `X` `memcpy` 到 `X_batch`；同步把对应 `y_[idx]` 写入 `y_batch[i]`。
- 返回 `actual`，若返回 0 表示一个 epoch 结束。

典型循环：

```cpp
Dataset ds(X, y, N, F, C);
ds.shuffle(epoch);
ds.reset();
Tensor X_batch;
int *y_batch = (int *)TINY_AI_MALLOC(B * sizeof(int));
while (true)
{
    int actual = ds.next_batch(X_batch, y_batch, B);
    if (actual == 0) break;
    // forward / backward / step ...
}
```

`Trainer::fit()` 已经替你写好了这个循环。

## to_tensor

```cpp
Tensor to_tensor() const;
```

把当前索引顺序下的所有样本拷贝到一个 `[n_samples, n_features]` 的 Tensor 中（深拷贝）。常用于一次性推理 / `Sequential::accuracy`。

## 内存预算

- **本身**：`indices_` 占 `n_samples * sizeof(int)`，几 KB 级。
- **每 batch**：`Tensor X_batch` 占 `B * F * 4` 字节，`y_batch` 占 `B * 4`；都按需重分配。
- **划分后**：训练 + 测试集各持有自己的索引副本，但共享 `X` / `y`。

对于 ESP32-S3 的常见 IMU / 振动数据集（N ~ 几千 / F ~ 几十）来说，`Dataset` 整体只需个位数 KB，可以放在内部 SRAM 上。
