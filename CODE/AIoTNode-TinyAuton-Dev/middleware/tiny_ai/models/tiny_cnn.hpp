/**
 * @file tiny_cnn.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Convenience 1-D CNN model wrapper for tiny_ai.
 *        Builds: N × (Conv1D + ReLU + MaxPool1D) → Flatten → Dense → Softmax
 *        Suited for time-series / sensor-signal classification on ESP32-S3.
 *
 *  Example (3-class signal classifier):
 *    CNN1DConfig cfg;
 *    cfg.in_channels   = 1;
 *    cfg.signal_length = 64;
 *    cfg.num_classes   = 3;
 *    cfg.filters       = {8, 16};    // 2 conv blocks
 *    cfg.kernels       = {5, 3};
 *    cfg.pool_size     = 2;
 *    cfg.fc_units      = 32;
 *    CNN1D model(cfg);
 *
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#pragma once

#include "tiny_sequential.hpp"
#include "tiny_conv.hpp"
#include "tiny_pool.hpp"
#include "tiny_dense.hpp"
#include "tiny_activation.hpp"

#ifdef __cplusplus

#include <vector>

namespace tiny
{

struct CNN1DConfig
{
    int              in_channels   = 1;     ///< Number of input channels
    int              signal_length = 64;    ///< Length of input signal
    int              num_classes   = 3;     ///< Number of output classes
    std::vector<int> filters       = {8};   ///< Filters per conv block
    std::vector<int> kernels       = {5};   ///< Kernel size per conv block
    int              pool_size     = 2;     ///< MaxPool kernel size after each block
    int              fc_units      = 32;    ///< Units in the intermediate Dense layer
    bool             use_softmax   = true;
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
