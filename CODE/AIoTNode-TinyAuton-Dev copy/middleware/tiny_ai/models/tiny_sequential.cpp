/**
 * @file tiny_sequential.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Sequential model implementation.
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
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

void Sequential::add(Layer *layer)
{
    layers_.push_back(layer);
}

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
    // Process in batches of n_samples to keep allocations minimal
    int   *preds = (int *)TINY_AI_MALLOC((size_t)n_samples * sizeof(int));
    if (!preds) return 0.0f;

    predict(x, preds);

    int correct = 0;
    for (int i = 0; i < n_samples; i++) if (preds[i] == labels[i]) correct++;
    TINY_AI_FREE(preds);
    return (float)correct / (float)n_samples;
}

} // namespace tiny

#endif // __cplusplus
