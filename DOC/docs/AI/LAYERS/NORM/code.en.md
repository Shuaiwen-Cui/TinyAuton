# Code

## tiny_norm.hpp (excerpt)

```cpp
class LayerNorm : public Layer
{
public:
    Tensor gamma, beta;
    LayerNorm(int feat, float epsilon = 1e-5f);
    Tensor forward(const Tensor &x) override;
    // backward / collect_params omitted
};

class BatchNorm1D : public Layer
{
public:
    Tensor gamma, beta;
    Tensor running_mean, running_var;

    BatchNorm1D(int feat, float momentum = 0.1f, float epsilon = 1e-5f);
    void set_training(bool mode) override { training_mode_ = mode; }
    Tensor forward(const Tensor &x) override;
    // backward / collect_params omitted
};

class BatchNorm2D : public Layer
{
public:
    Tensor gamma, beta;
    Tensor running_mean, running_var;

    BatchNorm2D(int num_channels, float momentum = 0.1f, float epsilon = 1e-5f);
    void set_training(bool mode) override { training_mode_ = mode; }
    Tensor forward(const Tensor &x) override;
    // backward / collect_params omitted
};
```

## tiny_norm.cpp (core logic)

```cpp
Tensor BatchNorm1D::forward(const Tensor &x)
{
#if TINY_AI_TRAINING_ENABLED
    if (training_mode_) {
        // Training: compute batch mean/variance per feature
        // and update running_mean/running_var
    }
#endif
    // Inference: fuse running stats into scale + shift
    // out = scale * x + shift
}

Tensor BatchNorm2D::forward(const Tensor &x)
{
    // x: [N, C, S...], stats computed per channel over N*S
#if TINY_AI_TRAINING_ENABLED
    if (training_mode_) {
        // Training: update per-channel running_mean/running_var
    }
#endif
    // Inference: per-channel fused linear transform
}
```

See full implementation in `middleware/tiny_ai/layers/tiny_norm.hpp` and `middleware/tiny_ai/layers/tiny_norm.cpp`.
