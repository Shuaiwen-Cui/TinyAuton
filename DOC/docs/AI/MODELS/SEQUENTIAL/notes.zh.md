# 说明

!!! note "说明"
    `Sequential` 是 `tiny_ai` 的层堆叠容器：按 `add()` 顺序依次执行 `forward`，反向时倒序执行 `backward`。它拥有所有 `Layer*`，析构时统一释放。

## 类定义

```cpp
class Sequential
{
public:
    Sequential() = default;
    ~Sequential();   // 删除所有持有的 Layer*

    void add(Layer *layer);

    Tensor forward (const Tensor &x);

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out);
    void   collect_params(std::vector<ParamGroup> &groups);
#endif

    void  summary()   const;
    void  predict (const Tensor &x, int *labels);
    float accuracy(const Tensor &x, const int *labels, int n_samples);

    Layer *operator[](int idx);
    int    num_layers() const;

protected:
    std::vector<Layer *> layers_;
};
```

## 构建模型

```cpp
Sequential m;
m.add(new Dense(F, 128));
m.add(new ActivationLayer(ActType::RELU));
m.add(new Dense(128, num_classes));
m.add(new ActivationLayer(ActType::SOFTMAX));
```

`add()` 接收 **裸指针**：`Sequential` 接管所有权，析构时 `delete` 每一个层。这意味着调用方不能再 `delete` 这些指针。

## forward / backward

- **forward**：`Tensor out = x.clone()`（避免修改输入），循环 `out = layers_[i]->forward(out)`。
- **backward**：从最后一层往前 `g = layers_[i]->backward(g)`，最终返回 `dL/dx`。
- **collect_params**：跳过 `trainable == false` 的层，对剩余层调用各自的 `collect_params`。

## predict 与 accuracy

```cpp
void  predict (const Tensor &x, int *labels);
float accuracy(const Tensor &x, const int *labels, int n_samples);
```

- `predict`：跑一遍 forward，对每个样本的输出取 argmax 写入 `labels`。
- `accuracy`：内部调用 `predict`，与给定 `labels` 比较，返回正确率。注意 `accuracy` 是按整个 `x` 一次性 forward，因此 `n_samples` 应等于 `x.rows()`。

## summary

`summary()` 用 `printf` 列出每层的索引与名称，便于调试网络结构：

```txt
Sequential model  (4 layers)
--------------------
  [ 0] dense
  [ 1] activation
  [ 2] dense
  [ 3] activation
--------------------
```

## 与 Trainer 的关系

`Trainer` 接收一个 `Sequential *`，按 fit 流程驱动：

```cpp
Sequential model;
// add layers ...

Adam opt(1e-3f);
Trainer trainer(&model, &opt, LossType::CROSS_ENTROPY);
trainer.fit(train_ds, cfg);
```

`Trainer::ensure_params_collected()` 在第一次 fit 时调用 `model.collect_params(params_)` 与 `optimizer_->init(params_)`，因此 `Sequential` 不需要主动暴露内部参数。

## 子类化：MLP / CNN1D

`MLP` 与 `CNN1D` 都继承 `Sequential`，仅在构造函数里把对应的层加进 `layers_`。Trainer 既能直接接收 `Sequential *`，也能接收这两个子类，因为多态会保持 `forward / backward / collect_params` 的调用一致。
