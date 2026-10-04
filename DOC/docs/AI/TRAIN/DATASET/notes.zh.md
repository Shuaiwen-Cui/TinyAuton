# Dataset：数据、索引与小批量

本页以 `tiny_dataset.hpp/.cpp` 为依据。Dataset 引用调用方的行主序 `float` 特征数组和 `int` 标签数组，自己拥有索引数组。输入数组必须保持有效；复制 Dataset 复制索引，不复制底层样本。

## 构造与划分

`Dataset(X, y, n_samples, n_features, n_classes)` 是公开构造函数，**没有公开无参构造函数**。`split(test_ratio, train, test, seed)` 的比例是测试集占比，默认种子为 42。

```cpp
#include "tiny_dataset.hpp"

void split_data(const float *X, const int *y, int N, int F, int C)
{
    tiny::Dataset full(X, y, N, F, C);
    tiny::Dataset train(full), test(full);
    full.split(0.2f, train, test, 42);
    // train/test share the original arrays and own separate index arrays.
}
```

## 小批量与元数据

- `reset()` 重置游标，`shuffle(seed)` 重排索引。
- `next_batch(X_batch, y_batch, batch_size)` 返回实际批量大小，末尾可能不足一个整批；`y_batch` 由调用方分配。
- 元数据接口为 `size()/features()/classes()/at_end()`。
- `to_tensor()` 按当前索引复制特征；`labels()` 返回底层标签指针，不能假定它与划分或洗牌后的 Tensor 顺序一致。评估优先使用 Trainer，手动评估应按同一批次获取特征与标签。

[当前训练示例](../../USAGE/usage.md) · [实现源码](code.md)

## 历史设计与接口摘录 {#auton-historical-contract}

下方包含旧版类定义、默认值和示意调用，保留供对照；当前接口以上方和工程头文件为准。

<details class="auton-source" markdown="1">
<summary>历史设计与接口摘录</summary>

# TinyAI · 训练 · 数据集 — 设计说明 {#_1}

!!! info "实现依据与记录"
    本节接口以 `CODE/AIoTNode-TinyAuton-AI/middleware/` 为依据。源码摘录与串口输出包含历史记录；是否运行某项测试，请核对工程入口和启用开关。

!!! note "说明"
    `Dataset` 在 `tiny_ai` 中负责把外部 float32 矩阵 / 标签数组包装成可洗牌、可拆分、可迭代的训练数据集。它仅持有索引数组，原始数据保持只读视图，便于把数据嵌入只读段或 PSRAM。

!!! abstract "Dataset — 数据管理：shuffle、切分、小批量"
    机器学习的第一步是把数据组织好。Dataset 做三件事：打乱、切分、按批喂给模型。

## 算法直觉 {#_2}

### 为什么需要 shuffle？ {#shuffle}

- **不打乱**：模型会学到数据的顺序模式而非真正的分类模式
- **打乱后**：每个 batch 都有各类样本的均匀分布，训练更稳定

### 为什么需要验证集？ {#_3}

- 模型可能在训练集上表现很好，但遇到新数据就失败（**过拟合**）
- `split(0.8)` 表示 80% 训练、20% 验证
- 只用验证集评估，不用来更新参数

### Mini-Batch 训练 {#mini-batch}

- 把所有数据一次性算梯度（full batch）→ 内存不够且容易陷入局部最优
- 一次一个样本（stochastic）→ 梯度噪声大，不稳定
- 小批量（mini-batch, 16/32/64）→ 折中方案：计算高效 + 梯度稳定

---

## 类定义 {#_4}

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

## 数据约定 {#_5}

- `X` 是行主序的 `n_samples × n_features` 浮点矩阵，存储在调用方拥有的内存（典型为 `iris_data.hpp` / `signal_data.hpp` 中的 `static const float[] = {...}`）。
- `y` 是长度为 `n_samples` 的整数数组，元素是类别下标。
- `Dataset` 仅持有 `X` / `y` 的指针视图与一个 `int *indices_` 数组，析构时只释放 `indices_`。
- 拷贝 / 移动 `Dataset` 不会复制原始数据，仅复制 / 接管 `indices_`。

## shuffle / split {#shuffle-split}

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

## next_batch 迭代 {#next_batch}

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

## to_tensor {#to_tensor}

```cpp
Tensor to_tensor() const;
```

把当前索引顺序下的所有样本拷贝到一个 `[n_samples, n_features]` 的 Tensor 中（深拷贝）。常用于一次性推理 / `Sequential::accuracy`。

## 内存预算 {#_6}

- **本身**：`indices_` 占 `n_samples * sizeof(int)`，几 KB 级。
- **每 batch**：`Tensor X_batch` 占 `B * F * 4` 字节，`y_batch` 占 `B * 4`；都按需重分配。
- **划分后**：训练 + 测试集各持有自己的索引副本，但共享 `X` / `y`。

对于 ESP32-S3 的常见 IMU / 振动数据集（N ~ 几千 / F ~ 几十）来说，`Dataset` 整体只需个位数 KB，可以放在内部 SRAM 上。

</details>
