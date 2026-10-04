# TinyAI · 模型 · 顺序模型 — 设计说明 {#_1}

!!! info "实现依据与记录"
    本节接口以 `CODE/AIoTNode-TinyAuton-AI/middleware/` 为依据。源码摘录与串口输出包含历史记录；是否运行某项测试，请核对工程入口和启用开关。

!!! note "说明"
    `Sequential` 是 `tiny_ai` 的层堆叠容器：按 `add()` 顺序依次执行 `forward`，反向时倒序执行 `backward`。它拥有所有 `Layer*`，析构时统一释放。

!!! abstract "Sequential — 层的容器：从前到后，依次通过"
    Sequential 把层按顺序排好：输入依次经过每层，输出就是预测结果。

## 算法直觉 {#_2}

### 模型 = 管道 {#_3}

想象一根管子：一端塞进数据，经过一层层的处理，另一端吐出预测结果。

```
输入 → Dense → ReLU → Dense → Softmax → 输出
```

Sequential 就是这根管子。

### 前向 / 反向 {#_4}

- **forward**: 输入依次通过每层的 `forward()` → 输出
- **backward**（训练时）：梯度从最后一层开始，逆序通过每层的 `backward()`

### 参数收集 {#_5}

训练前 `collect_params()` 遍历所有层，收集 `(权重张量, 梯度张量)` 对。Optimizer 初始化时根据这个列表的大小分配动量/Adam 缓冲。

!!! tip "所有权管理"
    `add()` 传入的层指针由 Sequential 接管所有权，析构时会自动 delete。

---

## 类定义 {#_6}

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

## 构建模型 {#_7}

```cpp
Sequential m;
m.add(new Dense(F, 128));
m.add(new ActivationLayer(ActType::RELU));
m.add(new Dense(128, num_classes));
m.add(new ActivationLayer(ActType::SOFTMAX));
```

`add()` 接收 **裸指针**：`Sequential` 接管所有权，析构时 `delete` 每一个层。这意味着调用方不能再 `delete` 这些指针。

## forward / backward {#forward-backward}

- **forward**：`Tensor out = x.clone()`（避免修改输入），循环 `out = layers_[i]->forward(out)`。
- **backward**：从最后一层往前 `g = layers_[i]->backward(g)`，最终返回 `dL/dx`。
- **collect_params**：跳过 `trainable == false` 的层，对剩余层调用各自的 `collect_params`。

## predict 与 accuracy {#predict-accuracy}

```cpp
void  predict (const Tensor &x, int *labels);
float accuracy(const Tensor &x, const int *labels, int n_samples);
```

- `predict`：跑一遍 forward，对每个样本的输出取 argmax 写入 `labels`。
- `accuracy`：内部调用 `predict`，与给定 `labels` 比较，返回正确率。注意 `accuracy` 是按整个 `x` 一次性 forward，因此 `n_samples` 应等于 `x.rows()`。

## summary {#summary}

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

## 与 Trainer 的关系 {#trainer}

`Trainer` 接收一个 `Sequential *`，按 fit 流程驱动：

```cpp
Sequential model;
// add layers ...

Adam opt(1e-3f);
Trainer trainer(&model, &opt, LossType::CROSS_ENTROPY);
trainer.fit(train_ds, cfg);
```

`Trainer::ensure_params_collected()` 在第一次 fit 时调用 `model.collect_params(params_)` 与 `optimizer_->init(params_)`，因此 `Sequential` 不需要主动暴露内部参数。

## 子类化：MLP / CNN1D {#mlp-cnn1d}

`MLP` 与 `CNN1D` 都继承 `Sequential`，仅在构造函数里把对应的层加进 `layers_`。Trainer 既能直接接收 `Sequential *`，也能接收这两个子类，因为多态会保持 `forward / backward / collect_params` 的调用一致。
