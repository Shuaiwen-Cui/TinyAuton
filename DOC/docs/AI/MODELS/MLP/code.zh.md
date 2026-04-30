# 代码

## tiny_mlp.hpp

```cpp
/**
 * @file tiny_mlp.hpp
 * @brief Multi-Layer Perceptron (MLP) — convenience wrapper around Sequential.
 */

#pragma once

#include "tiny_sequential.hpp"
#include "tiny_dense.hpp"

#ifdef __cplusplus

#include <initializer_list>

namespace tiny
{

class MLP : public Sequential
{
public:
    MLP(std::initializer_list<int> dims,
        ActType hidden_act = ActType::RELU,
        bool    use_softmax = true,
        bool    use_bias    = true);

    int in_features()  const { return in_feat_; }
    int out_features() const { return out_feat_; }

private:
    int in_feat_;
    int out_feat_;
};

} // namespace tiny

#endif // __cplusplus
```

## tiny_mlp.cpp

```cpp
/**
 * @file tiny_mlp.cpp
 * @brief MLP model implementation.
 */

#include "tiny_mlp.hpp"

#ifdef __cplusplus

namespace tiny
{

MLP::MLP(std::initializer_list<int> dims,
         ActType hidden_act,
         bool    use_softmax,
         bool    use_bias)
{
    const int *d  = dims.begin();
    int        nd = (int)dims.size();

    in_feat_  = d[0];
    out_feat_ = d[nd - 1];

    for (int i = 0; i < nd - 1; i++)
    {
        add(new Dense(d[i], d[i + 1], use_bias));

        if (i < nd - 2)
            add(new ActivationLayer(hidden_act));
        else if (use_softmax)
            add(new ActivationLayer(ActType::SOFTMAX));
    }
}

} // namespace tiny

#endif // __cplusplus
```
