# Code

## tiny_trainer.hpp

```cpp
/**
 * @file tiny_trainer.hpp
 * @brief Training loop helper for tiny_ai.
 */

#pragma once

#include "tiny_sequential.hpp"
#include "tiny_loss.hpp"
#include "tiny_optimizer.hpp"
#include "tiny_dataset.hpp"

#ifdef __cplusplus

namespace tiny
{

#if TINY_AI_TRAINING_ENABLED

class Trainer
{
public:
    struct Config
    {
        int  epochs      = 100;
        int  batch_size  = 16;
        bool verbose     = true;
        int  print_every = 10;
    };

    Trainer(Sequential *model, Optimizer *optimizer,
            LossType loss_type = LossType::CROSS_ENTROPY);

    void fit(Dataset &train_data, const Config &cfg = Config{},
             Dataset *val_data = nullptr);

    float evaluate_loss    (Dataset &data, int batch_size = 16);
    float evaluate_accuracy(Dataset &data, int batch_size = 16);

private:
    void ensure_params_collected();

    Sequential *model_;
    Optimizer  *optimizer_;
    LossType    loss_type_;
    std::vector<ParamGroup> params_;
    bool                    params_collected_;
};

#endif // TINY_AI_TRAINING_ENABLED

} // namespace tiny

#endif // __cplusplus
```

## tiny_trainer.cpp

```cpp
/**
 * @file tiny_trainer.cpp
 * @brief Trainer implementation.
 */

#include "tiny_trainer.hpp"
#include <cstdio>
#include <cstdlib>

#ifdef __cplusplus

namespace tiny
{

#if TINY_AI_TRAINING_ENABLED

Trainer::Trainer(Sequential *model, Optimizer *optimizer, LossType loss_type)
    : model_(model), optimizer_(optimizer),
      loss_type_(loss_type), params_collected_(false) {}

void Trainer::ensure_params_collected()
{
    if (!params_collected_)
    {
        model_->collect_params(params_);
        optimizer_->init(params_);
        params_collected_ = true;
    }
}

void Trainer::fit(Dataset &train_data, const Config &cfg, Dataset *val_data)
{
    ensure_params_collected();

    int batch_size = cfg.batch_size;
    int *y_batch = (int *)TINY_AI_MALLOC((size_t)batch_size * sizeof(int));
    if (!y_batch) return;

    for (int epoch = 0; epoch < cfg.epochs; epoch++)
    {
        train_data.shuffle(epoch + 1);
        float epoch_loss = 0.0f;
        int   n_batches_done = 0;
        Tensor X_batch;

        while (true)
        {
            int actual = train_data.next_batch(X_batch, y_batch, batch_size);
            if (actual == 0) break;

            Tensor logits = model_->forward(X_batch);
            float  loss   = loss_forward(logits, Tensor::zeros_like(logits),
                                          loss_type_, y_batch);
            epoch_loss += loss;

            optimizer_->zero_grad(params_);
            Tensor grad = loss_backward(logits, Tensor::zeros_like(logits),
                                         loss_type_, y_batch);
            model_->backward(grad);
            optimizer_->step(params_);
            n_batches_done++;
        }

        epoch_loss /= (float)(n_batches_done > 0 ? n_batches_done : 1);

        if (cfg.verbose && (epoch + 1) % cfg.print_every == 0)
        {
            if (val_data)
            {
                float val_acc = evaluate_accuracy(*val_data, batch_size);
                printf("Epoch [%3d/%3d]  loss: %.6f  val_acc: %.4f\n",
                       epoch + 1, cfg.epochs, epoch_loss, val_acc);
            }
            else
            {
                printf("Epoch [%3d/%3d]  loss: %.6f\n",
                       epoch + 1, cfg.epochs, epoch_loss);
            }
        }
    }

    TINY_AI_FREE(y_batch);
}

float Trainer::evaluate_loss(Dataset &data, int batch_size)
{
    data.reset();
    float total = 0.0f;
    int   n     = 0;
    int *y_batch = (int *)TINY_AI_MALLOC((size_t)batch_size * sizeof(int));
    if (!y_batch) return 0.0f;

    Tensor X_batch;
    while (true)
    {
        int actual = data.next_batch(X_batch, y_batch, batch_size);
        if (actual == 0) break;
        Tensor logits = model_->forward(X_batch);
        total += loss_forward(logits, Tensor::zeros_like(logits), loss_type_, y_batch);
        n++;
    }

    TINY_AI_FREE(y_batch);
    return n > 0 ? total / (float)n : 0.0f;
}

float Trainer::evaluate_accuracy(Dataset &data, int batch_size)
{
    data.reset();
    int correct = 0, total = 0;

    int *y_batch = (int *)TINY_AI_MALLOC((size_t)batch_size * sizeof(int));
    int *y_pred  = (int *)TINY_AI_MALLOC((size_t)batch_size * sizeof(int));
    if (!y_batch || !y_pred) { TINY_AI_FREE(y_batch); TINY_AI_FREE(y_pred); return 0.0f; }

    Tensor X_batch;
    while (true)
    {
        int actual = data.next_batch(X_batch, y_batch, batch_size);
        if (actual == 0) break;
        model_->predict(X_batch, y_pred);
        for (int i = 0; i < actual; i++)
            if (y_pred[i] == y_batch[i]) correct++;
        total += actual;
    }

    TINY_AI_FREE(y_batch);
    TINY_AI_FREE(y_pred);
    return total > 0 ? (float)correct / (float)total : 0.0f;
}

#endif // TINY_AI_TRAINING_ENABLED

} // namespace tiny

#endif // __cplusplus
```
