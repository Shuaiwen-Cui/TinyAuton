# TinyAI · 模型 · CNN — 实现与源码 {#_1}

!!! info "实现依据与记录"
    本节接口以 `CODE/AIoTNode-TinyAuton-AI/middleware/` 为依据。源码摘录与串口输出包含历史记录；是否运行某项测试，请核对工程入口和启用开关。

本页保留完整源码摘录，按文件展开阅读。先看[设计说明](notes.md)了解数据流、接口和算法，再核对实现；源码摘录可能属于历史版本，当前实现请以所选工程为准。

## `tiny_cnn.hpp` {#tiny_cnnhpp}

```cpp
/**
 * @file tiny_cnn.hpp
 * @brief 1D Convolutional Neural Network (CNN1D) — convenience wrapper.
 */

#pragma once

#include "tiny_sequential.hpp"
#include "tiny_dense.hpp"
#include "tiny_conv.hpp"
#include "tiny_pool.hpp"

#ifdef __cplusplus

#include <vector>

namespace tiny
{

struct CNN1DConfig
{
    int signal_length;
    int in_channels = 1;
    int num_classes = 3;

    std::vector<int> filters;
    std::vector<int> kernels;
    int pool_size = 2;

    int  fc_units    = 32;
    bool use_softmax = true;
};

class CNN1D : public Sequential
{
public:
    explicit CNN1D(const CNN1DConfig &cfg);

    int flat_features() const { return flat_feat_; }

private:
    int flat_feat_;
};

} // namespace tiny

#endif // __cplusplus
```

## `tiny_cnn.cpp` {#tiny_cnncpp}

```cpp
/**
 * @file tiny_cnn.cpp
 * @brief CNN1D model implementation.
 */

#include "tiny_cnn.hpp"

#ifdef __cplusplus

namespace tiny
{

CNN1D::CNN1D(const CNN1DConfig &cfg)
{
    int in_ch    = cfg.in_channels;
    int L        = cfg.signal_length;
    int n_blocks = (int)cfg.filters.size();

    for (int i = 0; i < n_blocks; i++)
    {
        int out_ch = cfg.filters[i];
        int k      = cfg.kernels.size() > (size_t)i ? cfg.kernels[i] : 3;

        add(new Conv1D(in_ch, out_ch, k, 1, 0, true));
        add(new ActivationLayer(ActType::RELU));
        L = L - k + 1;

        add(new MaxPool1D(cfg.pool_size, cfg.pool_size));
        L     = L / cfg.pool_size;
        in_ch = out_ch;
    }

    add(new Flatten());
    flat_feat_ = in_ch * L;

    if (cfg.fc_units > 0)
    {
        add(new Dense(flat_feat_, cfg.fc_units, true));
        add(new ActivationLayer(ActType::RELU));
        add(new Dense(cfg.fc_units, cfg.num_classes, true));
    }
    else
    {
        add(new Dense(flat_feat_, cfg.num_classes, true));
    }

    if (cfg.use_softmax)
        add(new ActivationLayer(ActType::SOFTMAX));
}

} // namespace tiny

#endif // __cplusplus
```
