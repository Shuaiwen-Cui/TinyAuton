/**
 * @file example_attention.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Multi-head self-attention example for tiny_ai.
 *
 *  Dataset : Iris (150 samples, 4 features, 3 classes — from iris_data.hpp)
 *  Model   : Dense(4,32)+ReLU → reshape [B,32]→[B,4,8]
 *            → Attention(embed_dim=8, heads=2) → GlobalAvgPool
 *            → Dense(8,3)  [raw logits — cross-entropy handles softmax]
 *  Training: Adam, 100 epochs, batch=16, cross-entropy
 *
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#include "tiny_ai.h"
#include "iris_data.hpp"
#include <cstdio>
#include <cstdlib>

// ESP-IDF task watchdog feeding for long-running training loops
#if defined(ESP_PLATFORM) || defined(IDF_VER) || defined(ESP32)
#include "esp_task_wdt.h"
#endif

#ifdef __cplusplus

using namespace tiny;
using namespace tiny_data;

// ============================================================================
// Entry point
// ============================================================================

void example_attention(void)
{
    printf("\n");
    printf("========================================\n");
    printf("  tiny_ai  |  Attention Example (Iris)\n");
    printf("========================================\n");

    // ---- Dataset ----
    Dataset dataset(
        &IRIS_X[0][0], IRIS_Y,
        IRIS_N_SAMPLES, IRIS_N_FEATURES, IRIS_N_CLASSES);

    Dataset train_ds(dataset), test_ds(dataset);
    dataset.split(0.2f, train_ds, test_ds, 42);
    printf("Dataset split: %d train / %d test\n",
           train_ds.size(), test_ds.size());

    // ---- Model components ----
    //
    //  Input:         [B, 4]   (4 Iris features)
    //  embed_proj:    Dense(4, 32) → [B, 32]
    //  embed_act:     ReLU        → [B, 32]
    //  reshape:       [B, 32] → [B, 4, 8]   (4 tokens, embed_dim = 8)
    //  attn:          Attention(8, 2)        → [B, 4, 8]
    //  gap:           GlobalAvgPool          → [B, 8]
    //  classifier:    Dense(8, 3)            → [B, 3]  (raw logits)
    //
    const int SEQ_LEN  = IRIS_N_FEATURES;  // 4 tokens (one per feature)
    const int EMB_DIM  = 8;
    const int N_HEADS  = 2;

    Dense          embed_proj(IRIS_N_FEATURES, SEQ_LEN * EMB_DIM, true);
    ActivationLayer embed_act(ActType::RELU);
    Attention      attn(EMB_DIM, N_HEADS, true);
    GlobalAvgPool  gap;
    Dense          classifier(EMB_DIM, IRIS_N_CLASSES, true);

    printf("Model summary:\n");
    printf("  Dense(%d, %d) + ReLU\n", IRIS_N_FEATURES, SEQ_LEN * EMB_DIM);
    printf("  reshape [B, %d] -> [B, %d, %d]  (tokens, embed_dim)\n",
           SEQ_LEN * EMB_DIM, SEQ_LEN, EMB_DIM);
    printf("  Attention(embed_dim=%d, heads=%d, head_dim=%d)\n",
           EMB_DIM, N_HEADS, EMB_DIM / N_HEADS);
    printf("  GlobalAvgPool\n");
    printf("  Dense(%d, %d)  [raw logits]\n", EMB_DIM, IRIS_N_CLASSES);

    // ---- Optimizer ----
    Adam opt(1e-3f);

#if TINY_AI_TRAINING_ENABLED

    // Collect learnable parameters
    std::vector<ParamGroup> params;
    embed_proj.collect_params(params);
    attn.collect_params(params);
    classifier.collect_params(params);

    opt.init(params);

    const int batch_size  = 16;
    const int n_epochs    = 100;
    const int print_every = 20;

    int *y_batch = (int *)TINY_AI_MALLOC((size_t)batch_size * sizeof(int));
    if (!y_batch)
    {
        printf("  Memory allocation failed!\n");
        return;
    }

    printf("\nTraining...\n");

    for (int epoch = 0; epoch < n_epochs; epoch++)
    {
        train_ds.shuffle(epoch + 1);
        float epoch_loss = 0.0f;
        int   n_batches  = 0;
        Tensor X_batch;

        while (true)
        {
            int actual = train_ds.next_batch(X_batch, y_batch, batch_size);
            if (actual == 0) break;

            // ------ Forward ------

            // Embed: [actual, 4] → [actual, 32]
            Tensor e0 = embed_proj.forward(X_batch);
            Tensor e1 = embed_act.forward(e0);

            // Reshape: [actual, 32] → [actual, 4, 8]
            e1.reshape_3d(actual, SEQ_LEN, EMB_DIM);

            // Attention: [actual, 4, 8] → [actual, 4, 8]
            Tensor a0 = attn.forward(e1);

            // Pool: [actual, 4, 8] → [actual, 8]
            Tensor p0 = gap.forward(a0);

            // Classify: [actual, 8] → [actual, 3]
            Tensor logits = classifier.forward(p0);

            float loss = cross_entropy_forward(logits, y_batch);
            epoch_loss += loss;

            // ------ Backward ------

            opt.zero_grad(params);

            // grad w.r.t. logits: [actual, 3]
            Tensor dlogits = cross_entropy_backward(logits, y_batch);

            // classifier backward: [actual, 3] → [actual, 8]
            Tensor dp0 = classifier.backward(dlogits);

            // gap backward: [actual, 8] → [actual, 4, 8]
            Tensor da0 = gap.backward(dp0);

            // attention backward: [actual, 4, 8] → [actual, 4, 8]
            Tensor de1 = attn.backward(da0);

            // reshape gradient back: [actual, 4, 8] → [actual, 32]
            de1.reshape_2d(actual, SEQ_LEN * EMB_DIM);

            // embed backward: [actual, 32] → [actual, 32]
            Tensor de0 = embed_act.backward(de1);

            // embed_proj backward (updates dW, db internally): [actual, 32] → [actual, 4]
            embed_proj.backward(de0);

            opt.step(params);
            n_batches++;
        }

        if ((epoch + 1) % print_every == 0)
            printf("Epoch [%3d/%3d]  loss: %.6f\n",
                   epoch + 1, n_epochs, epoch_loss / (float)n_batches);
    }

    TINY_AI_FREE(y_batch);

    // ---- Evaluate ----

    auto eval_accuracy = [&](Dataset &ds, const char *tag)
    {
        ds.reset();
        int correct = 0, total = 0;
        int *yb = (int *)TINY_AI_MALLOC((size_t)batch_size * sizeof(int));
        int *yp = (int *)TINY_AI_MALLOC((size_t)batch_size * sizeof(int));
        if (!yb || !yp) { TINY_AI_FREE(yb); TINY_AI_FREE(yp); return; }
        Tensor Xb;

        while (true)
        {
            int actual = ds.next_batch(Xb, yb, batch_size);
            if (actual == 0) break;

            Tensor e0 = embed_proj.forward(Xb);
            Tensor e1 = embed_act.forward(e0);
            e1.reshape_3d(actual, SEQ_LEN, EMB_DIM);
            Tensor a0     = attn.forward(e1);
            Tensor p0     = gap.forward(a0);
            Tensor logits = classifier.forward(p0);

            int n_cls = logits.shape[1];
            for (int i = 0; i < actual; i++)
            {
                float best = logits.at(i, 0);
                int   pred = 0;
                for (int c = 1; c < n_cls; c++)
                    if (logits.at(i, c) > best) { best = logits.at(i, c); pred = c; }
                yp[i] = pred;
            }
            for (int i = 0; i < actual; i++) if (yp[i] == yb[i]) correct++;
            total += actual;
        }

        TINY_AI_FREE(yb);
        TINY_AI_FREE(yp);
        printf("  %s accuracy: %.2f%%\n", tag, 100.0f * correct / total);
    };

    printf("\n--- Float32 Results ---\n");
    eval_accuracy(train_ds, "Train");
    eval_accuracy(test_ds,  "Test ");

#else
    printf("(Training disabled)\n");
#endif

    printf("\nexample_attention  DONE\n");
}

#endif // __cplusplus
