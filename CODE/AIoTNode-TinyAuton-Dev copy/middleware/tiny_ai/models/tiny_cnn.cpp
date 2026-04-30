/**
 * @file tiny_cnn.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief CNN1D model implementation.
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#include "tiny_cnn.hpp"

#ifdef __cplusplus

namespace tiny
{

CNN1D::CNN1D(const CNN1DConfig &cfg)
{
    int in_ch = cfg.in_channels;
    int L     = cfg.signal_length;
    int n_blocks = (int)cfg.filters.size();

    for (int i = 0; i < n_blocks; i++)
    {
        int out_ch = cfg.filters[i];
        int k      = cfg.kernels.size() > (size_t)i ? cfg.kernels[i] : 3;

        // Conv1D (stride=1, no padding): out_L = L - k + 1
        add(new Conv1D(in_ch, out_ch, k, 1, 0, true));
        add(new ActivationLayer(ActType::RELU));
        L      = L - k + 1;

        // MaxPool1D: out_L = L / pool_size
        add(new MaxPool1D(cfg.pool_size, cfg.pool_size));
        L      = L / cfg.pool_size;
        in_ch  = out_ch;
    }

    // Flatten: [batch, in_ch, L] → [batch, in_ch*L]
    add(new Flatten());
    flat_feat_ = in_ch * L;

    // Dense head
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
