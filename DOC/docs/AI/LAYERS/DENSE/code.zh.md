# 代码

## tiny_dense.hpp

```cpp
/**
 * @file tiny_dense.hpp
 * @brief Fully-connected (Dense / Linear) layer for tiny_ai.
 *        output = input × weight^T + bias
 */

#pragma once

#include "tiny_layer.hpp"

#ifdef __cplusplus

namespace tiny
{

class Dense : public Layer
{
public:
    Tensor weight;   // [out_features, in_features]
    Tensor bias;     // [out_features]

#if TINY_AI_TRAINING_ENABLED
    Tensor dweight;
    Tensor dbias;
#endif

    Dense(int in_features, int out_features, bool use_bias = true);

    Tensor forward(const Tensor &x) override;

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out) override;
    void   collect_params(std::vector<ParamGroup> &groups) override;
#endif

    int in_features()  const { return in_feat_; }
    int out_features() const { return out_feat_; }

private:
    int  in_feat_;
    int  out_feat_;
    bool use_bias_;

#if TINY_AI_TRAINING_ENABLED
    Tensor x_cache_;
#endif

    void xavier_init();
};

} // namespace tiny

#endif // __cplusplus
```

## tiny_dense.cpp

```cpp
/**
 * @file tiny_dense.cpp
 * @brief Dense layer implementation.
 */

#include "tiny_dense.hpp"
#include <cmath>
#include <cstdlib>

#ifdef __cplusplus

namespace tiny
{

Dense::Dense(int in_features, int out_features, bool use_bias)
    : Layer("dense", true),
      weight(out_features, in_features),
      bias(use_bias ? out_features : 0),
      in_feat_(in_features), out_feat_(out_features), use_bias_(use_bias)
{
#if TINY_AI_TRAINING_ENABLED
    dweight = Tensor::zeros_like(weight);
    if (use_bias_) dbias = Tensor(out_features);
#endif
    xavier_init();
}

void Dense::xavier_init()
{
    float limit = sqrtf(6.0f / (float)(in_feat_ + out_feat_));
    for (int i = 0; i < weight.size; i++)
    {
        float r = (float)rand() / (float)RAND_MAX;
        weight.data[i] = (r * 2.0f - 1.0f) * limit;
    }
    if (use_bias_) bias.zero();
}

Tensor Dense::forward(const Tensor &x)
{
    int batch = x.shape[0];
    Tensor out(batch, out_feat_);

#if TINY_AI_TRAINING_ENABLED
    x_cache_ = x.clone();
#endif

    for (int b = 0; b < batch; b++)
        for (int o = 0; o < out_feat_; o++)
        {
            float sum = use_bias_ ? bias.data[o] : 0.0f;
            for (int i = 0; i < in_feat_; i++)
                sum += weight.at(o, i) * x.at(b, i);
            out.at(b, o) = sum;
        }
    return out;
}

#if TINY_AI_TRAINING_ENABLED

Tensor Dense::backward(const Tensor &grad_out)
{
    int batch = x_cache_.shape[0];
    Tensor grad_input(batch, in_feat_);

    for (int o = 0; o < out_feat_; o++)
        for (int i = 0; i < in_feat_; i++)
        {
            float acc = 0.0f;
            for (int b = 0; b < batch; b++)
                acc += grad_out.at(b, o) * x_cache_.at(b, i);
            dweight.at(o, i) += acc;
        }

    if (use_bias_)
        for (int o = 0; o < out_feat_; o++)
        {
            float acc = 0.0f;
            for (int b = 0; b < batch; b++) acc += grad_out.at(b, o);
            dbias.data[o] += acc;
        }

    for (int b = 0; b < batch; b++)
        for (int i = 0; i < in_feat_; i++)
        {
            float acc = 0.0f;
            for (int o = 0; o < out_feat_; o++)
                acc += grad_out.at(b, o) * weight.at(o, i);
            grad_input.at(b, i) = acc;
        }
    return grad_input;
}

void Dense::collect_params(std::vector<ParamGroup> &groups)
{
    groups.push_back({&weight, &dweight});
    if (use_bias_) groups.push_back({&bias, &dbias});
}

#endif // TINY_AI_TRAINING_ENABLED

} // namespace tiny

#endif // __cplusplus
```
