/**
 * @file tiny_trainer.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Training loop for tiny_ai — wraps Sequential + Optimizer + Loss.
 * @version 1.0
 * @date 2026-04-30
 * @copyright Copyright (c) 2026
 */

#pragma once

#include "tiny_sequential.hpp"
#include "tiny_optimizer.hpp"
#include "tiny_loss.hpp"
#include "tiny_dataset.hpp"

#ifdef __cplusplus

namespace tiny
{

#if TINY_AI_TRAINING_ENABLED

class Trainer
{
public:
    // =========================================================================
    // Training configuration
    // =========================================================================
    struct Config
    {
        int   epochs      = 50;    ///< Number of training epochs
        int   batch_size  = 16;    ///< Mini-batch size
        bool  verbose     = true;  ///< Print loss / accuracy each epoch
        int   print_every = 10;    ///< Print interval in epochs
        float lr_decay    = 1.0f;  ///< Multiplicative LR decay per epoch (1 = no decay)
    };

    /**
     * @param model      Pointer to the Sequential model (not owned by Trainer)
     * @param optimizer  Pointer to the Optimizer (not owned by Trainer)
     * @param loss_type  Loss function to use during training
     */
    Trainer(Sequential *model, Optimizer *optimizer, LossType loss_type = LossType::CROSS_ENTROPY);

    // =========================================================================
    // Training
    // =========================================================================

    /**
     * @brief Run the full training loop.
     *
     * @param train_data  Training dataset
     * @param cfg         Training configuration
     * @param val_data    Optional validation dataset (nullptr to skip validation)
     */
    void fit(Dataset &train_data, const Config &cfg,
             Dataset *val_data = nullptr);

    // =========================================================================
    // Evaluation
    // =========================================================================

    /**
     * @brief Compute loss on a dataset.
     * @return Mean loss over all batches
     */
    float evaluate_loss(Dataset &data, int batch_size = 32);

    /**
     * @brief Compute classification accuracy on a dataset.
     * @return Accuracy in [0, 1]
     */
    float evaluate_accuracy(Dataset &data, int batch_size = 32);

private:
    Sequential *model_;
    Optimizer  *optimizer_;
    LossType    loss_type_;

    std::vector<ParamGroup> params_;
    bool params_collected_;

    void ensure_params_collected();
};

#endif // TINY_AI_TRAINING_ENABLED

} // namespace tiny

#endif // __cplusplus
