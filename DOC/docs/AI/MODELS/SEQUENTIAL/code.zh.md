# TinyAI · 模型 · 顺序模型 — 实现与源码 {#_1}

!!! info "实现依据与记录"
    本节接口以 `CODE/AIoTNode-TinyAuton-AI/middleware/` 为依据。源码摘录与串口输出包含历史记录；是否运行某项测试，请核对工程入口和启用开关。

本页保留完整源码摘录，按文件展开阅读。先看[设计说明](notes.md)了解数据流、接口和算法，再核对实现；源码摘录可能属于历史版本，当前实现请以所选工程为准。

## `tiny_sequential.hpp` {#tiny_sequentialhpp}

```cpp
/**
 * @file tiny_sequential.hpp
 * @brief Sequential model container for tiny_ai — stacks layers in order
 *        and runs forward/backward through them.
 */

#pragma once

#include "tiny_layer.hpp"

#ifdef __cplusplus

#include <vector>

namespace tiny
{

class Sequential
{
public:
    Sequential() = default;
    ~Sequential();

    void add(Layer *layer);

    Tensor forward(const Tensor &x);

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out);
    void   collect_params(std::vector<ParamGroup> &groups);
#endif

    void  summary() const;
    void  predict(const Tensor &x, int *labels);
    float accuracy(const Tensor &x, const int *labels, int n_samples);

    Layer *operator[](int idx)             { return layers_[idx]; }
    int    num_layers() const              { return (int)layers_.size(); }

protected:
    std::vector<Layer *> layers_;
};

} // namespace tiny

#endif // __cplusplus
```

## `tiny_sequential.cpp` {#tiny_sequentialcpp}

<details class="auton-source" markdown="1">
<summary>展开 <code>tiny_sequential.cpp</code> · 83 行</summary>

```cpp
/**
 * @file tiny_sequential.cpp
 * @brief Sequential model implementation.
 */

#include "tiny_sequential.hpp"
#include <cstdio>

#ifdef __cplusplus

namespace tiny
{

Sequential::~Sequential()
{
    for (Layer *l : layers_) delete l;
}

void Sequential::add(Layer *layer) { layers_.push_back(layer); }

Tensor Sequential::forward(const Tensor &x)
{
    Tensor out = x.clone();
    for (Layer *l : layers_) out = l->forward(out);
    return out;
}

#if TINY_AI_TRAINING_ENABLED

Tensor Sequential::backward(const Tensor &grad_out)
{
    Tensor g = grad_out.clone();
    for (int i = (int)layers_.size() - 1; i >= 0; i--)
        g = layers_[i]->backward(g);
    return g;
}

void Sequential::collect_params(std::vector<ParamGroup> &groups)
{
    for (Layer *l : layers_)
        if (l->trainable) l->collect_params(groups);
}

#endif

void Sequential::summary() const
{
    printf("Sequential model  (%d layers)\n", (int)layers_.size());
    printf("%-20s\n", "--------------------");
    for (int i = 0; i < (int)layers_.size(); i++)
        printf("  [%2d] %s\n", i, layers_[i]->name);
    printf("%-20s\n", "--------------------");
}

void Sequential::predict(const Tensor &x, int *labels)
{
    Tensor out = forward(x);
    int batch = out.rows();
    int cls   = out.cols();
    for (int b = 0; b < batch; b++)
    {
        int   best_c = 0;
        float best_v = out.at(b, 0);
        for (int c = 1; c < cls; c++)
            if (out.at(b, c) > best_v) { best_v = out.at(b, c); best_c = c; }
        labels[b] = best_c;
    }
}

float Sequential::accuracy(const Tensor &x, const int *labels, int n_samples)
{
    int *preds = (int *)TINY_AI_MALLOC((size_t)n_samples * sizeof(int));
    if (!preds) return 0.0f;
    predict(x, preds);
    int correct = 0;
    for (int i = 0; i < n_samples; i++) if (preds[i] == labels[i]) correct++;
    TINY_AI_FREE(preds);
    return (float)correct / (float)n_samples;
}

} // namespace tiny

#endif // __cplusplus
```

</details>
