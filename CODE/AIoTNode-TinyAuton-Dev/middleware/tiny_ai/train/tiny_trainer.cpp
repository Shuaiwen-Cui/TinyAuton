/**
 * @file tiny_trainer.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Trainer implementation.
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
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
      loss_type_(loss_type), params_collected_(false)
{}

void Trainer::ensure_params_collected()
{
    if (!params_collected_)
    {
        model_->collect_params(params_);
        optimizer_->init(params_);
        params_collected_ = true;
    }
}

// ============================================================================
// Training loop
// ============================================================================

void Trainer::fit(Dataset &train_data, const Config &cfg, Dataset *val_data)
{
    ensure_params_collected();

    int batch_size = cfg.batch_size;

    // Allocate label buffers (stack is fine for typical batch sizes)
    int *y_batch  = (int *)TINY_AI_MALLOC((size_t)batch_size * sizeof(int));
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

            // ---- Forward ----
            Tensor logits = model_->forward(X_batch);

            // ---- Loss ----
            float loss = loss_forward(logits, Tensor::zeros_like(logits),
                                       loss_type_, y_batch);
            epoch_loss += loss;

            // ---- Backward ----
            optimizer_->zero_grad(params_);

            Tensor grad = loss_backward(logits, Tensor::zeros_like(logits),
                                         loss_type_, y_batch);
            model_->backward(grad);

            // ---- Update ----
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

// ============================================================================
// Evaluation
// ============================================================================

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

    int *y_batch    = (int *)TINY_AI_MALLOC((size_t)batch_size * sizeof(int));
    int *y_pred     = (int *)TINY_AI_MALLOC((size_t)batch_size * sizeof(int));
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
