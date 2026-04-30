/**
 * @file tiny_mlp.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief MLP model implementation.
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
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

        // Hidden layers get hidden_act; last layer gets Softmax (if requested)
        if (i < nd - 2)
            add(new ActivationLayer(hidden_act));
        else if (use_softmax)
            add(new ActivationLayer(ActType::SOFTMAX));
    }
}

} // namespace tiny

#endif // __cplusplus
