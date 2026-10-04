# TinyAI · 层 · 归一化 — 实现与源码 {#_1}

!!! info "实现依据与记录"
    本节接口以 `CODE/AIoTNode-TinyAuton-AI/middleware/` 为依据。源码摘录与串口输出包含历史记录；是否运行某项测试，请核对工程入口和启用开关。

本页保留完整源码摘录，按文件展开阅读。先看[设计说明](notes.md)了解数据流、接口和算法，再核对实现；源码摘录可能属于历史版本，当前实现请以所选工程为准。

## tiny_norm.hpp（节选） {#tiny_normhpp}

```cpp
class LayerNorm : public Layer
{
public:
    Tensor gamma, beta;
    LayerNorm(int feat, float epsilon = 1e-5f);
    Tensor forward(const Tensor &x) override;
    // backward / collect_params 省略
};

class BatchNorm1D : public Layer
{
public:
    Tensor gamma, beta;
    Tensor running_mean, running_var;

    BatchNorm1D(int feat, float momentum = 0.1f, float epsilon = 1e-5f);
    void set_training(bool mode) override { training_mode_ = mode; }
    Tensor forward(const Tensor &x) override;
    // backward / collect_params 省略
};

class BatchNorm2D : public Layer
{
public:
    Tensor gamma, beta;
    Tensor running_mean, running_var;

    BatchNorm2D(int num_channels, float momentum = 0.1f, float epsilon = 1e-5f);
    void set_training(bool mode) override { training_mode_ = mode; }
    Tensor forward(const Tensor &x) override;
    // backward / collect_params 省略
};
```

## tiny_norm.cpp（关键逻辑） {#tiny_normcpp}

```cpp
Tensor BatchNorm1D::forward(const Tensor &x)
{
#if TINY_AI_TRAINING_ENABLED
    if (training_mode_) {
        // 训练：按 feature 统计当前 batch 的均值方差
        // 并更新 running_mean/running_var
    }
#endif
    // 推理：使用 running stats 融合为 scale + shift
    // out = scale * x + shift
}

Tensor BatchNorm2D::forward(const Tensor &x)
{
    // x: [N, C, S...]，每个通道在 N*S 上统计
#if TINY_AI_TRAINING_ENABLED
    if (training_mode_) {
        // 训练：更新每个通道的 running_mean/running_var
    }
#endif
    // 推理：每个通道使用融合常数执行线性变换
}
```

完整实现请参考源码：`middleware/tiny_ai/layers/tiny_norm.hpp` 与 `middleware/tiny_ai/layers/tiny_norm.cpp`。
