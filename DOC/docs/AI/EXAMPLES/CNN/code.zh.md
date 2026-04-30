# 代码

## example_cnn.cpp

```cpp
/**
 * @file example_cnn.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief CNN example for tiny_ai.
 *
 *  Dataset : Synthetic 1-D signals (3 classes, 50 samples each, 64 pts/sample)
 *            Generated on-device from signal_data.hpp — no file I/O needed.
 *  Model   : Conv1D(1,8,5)+ReLU+MaxPool1D(2) →
 *            Conv1D(8,16,3)+ReLU+MaxPool1D(2) →
 *            Flatten → Dense(224,32)+ReLU → Dense(32,3)+Softmax
 *  Training: Adam, 50 epochs, batch=16, cross-entropy
 *  Post    : FP8 E4M3FN weight compression demo
 *
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#include "tiny_ai.h"
#include "signal_data.hpp"
#include <cstdio>
#include <cstdlib>

#ifdef __cplusplus

using namespace tiny;
using namespace tiny_data;

// ============================================================================
// FP8 compression demo
// ============================================================================

static void fp8_compression_demo(void)
{
    printf("\n--- FP8 E4M3FN Compression Demo ---\n");

    // Create a small representative weight tensor
    Tensor w(16, 8, 3);  // simulates Conv1D weight [16, 8, 3]
    for (int i = 0; i < w.size; i++)
        w.data[i] = (float)(i - w.size / 2) * 0.02f;

    // Calibrate
    QuantParams qp = calibrate(w, TINY_DTYPE_FP8_E4M3);
    printf("  E4M3 scale = %.6f\n", qp.scale);

    // Quantise
    uint8_t *fp8_buf = (uint8_t *)TINY_AI_MALLOC((size_t)w.size);
    if (!fp8_buf) return;

    quantize(w, fp8_buf, qp);

    // Dequantise and check error
    Tensor w_recon = Tensor::zeros_like(w);
    dequantize(fp8_buf, w_recon, qp);

    float max_err = 0.0f, mse = 0.0f;
    for (int i = 0; i < w.size; i++)
    {
        float e = w.data[i] - w_recon.data[i];
        if (e < 0.0f) e = -e;
        if (e > max_err) max_err = e;
        mse += (w.data[i] - w_recon.data[i]) * (w.data[i] - w_recon.data[i]);
    }
    mse /= (float)w.size;

    printf("  Size: FP32=%d bytes  FP8=%d bytes  (4:1 compression)\n",
           w.size * 4, w.size);
    printf("  Max abs error = %.6f   MSE = %.8f\n", max_err, mse);

    // Show a few values
    for (int i = 0; i < 4; i++)
        printf("  w[%d] orig=%.4f  fp8=0x%02X  recon=%.4f\n",
               i, w.data[i], fp8_buf[i], w_recon.data[i]);

    TINY_AI_FREE(fp8_buf);
}

// ============================================================================
// Entry point
// ============================================================================

void example_cnn(void)
{
    printf("\n");
    printf("========================================\n");
    printf("  tiny_ai  |  CNN Example (Signal)\n");
    printf("========================================\n");

    // ---- Generate dataset ----
    float *X_raw = (float *)TINY_AI_MALLOC_PSRAM(
        (size_t)SIG_N_SAMPLES * SIG_SIGNAL_LEN * sizeof(float));
    int   *Y_raw = (int   *)TINY_AI_MALLOC(
        (size_t)SIG_N_SAMPLES * sizeof(int));

    if (!X_raw || !Y_raw)
    {
        printf("  Memory allocation failed!\n");
        TINY_AI_FREE(X_raw); TINY_AI_FREE(Y_raw);
        return;
    }

    generate_signal_dataset(X_raw, Y_raw);
    printf("Generated %d signals (%d classes × %d samples, %d pts each)\n",
           SIG_N_SAMPLES, SIG_N_CLASSES, SIG_SAMPLES_PER_CLASS, SIG_SIGNAL_LEN);

    // Wrap in Dataset (features are [n, signal_len] — treated as 1-D)
    Dataset dataset(X_raw, Y_raw, SIG_N_SAMPLES, SIG_SIGNAL_LEN, SIG_N_CLASSES);

    Dataset train_ds(dataset), test_ds(dataset);
    dataset.split(0.2f, train_ds, test_ds, 7);
    printf("Split: %d train / %d test\n", train_ds.size(), test_ds.size());

    // ---- Model ----
    // Input to CNN: [batch, 1, 64]
    // The Dataset returns [batch, 64]; we reshape in the forward-pass wrapper below.
    CNN1DConfig cfg;
    cfg.in_channels   = 1;
    cfg.signal_length = SIG_SIGNAL_LEN;
    cfg.num_classes   = SIG_N_CLASSES;
    cfg.filters       = {8, 16};
    cfg.kernels       = {5, 3};
    cfg.pool_size     = 2;
    cfg.fc_units      = 32;

    CNN1D model(cfg);
    model.summary();
    printf("  Flat features after conv blocks: %d\n", model.flat_features());

    // ---- Optimiser + Trainer ----
    Adam opt(1e-3f);

#if TINY_AI_TRAINING_ENABLED
    Trainer trainer(&model, &opt, LossType::CROSS_ENTROPY);

    Trainer::Config tcfg;
    tcfg.epochs      = 50;
    tcfg.batch_size  = 16;
    tcfg.verbose     = true;
    tcfg.print_every = 10;

    printf("\nTraining...\n");

    // Custom training loop (to handle the [B,L] → [B,1,L] reshape)
    std::vector<ParamGroup> params;
    model.collect_params(params);
    opt.init(params);

    int batch_size = tcfg.batch_size;
    int *y_batch   = (int *)TINY_AI_MALLOC((size_t)batch_size * sizeof(int));

    for (int epoch = 0; epoch < tcfg.epochs; epoch++)
    {
        train_ds.shuffle(epoch + 1);
        float epoch_loss = 0.0f;
        int n_batches = 0;
        Tensor X_batch;

        while (true)
        {
            int actual = train_ds.next_batch(X_batch, y_batch, batch_size);
            if (actual == 0) break;

            // Reshape [B, L] → [B, 1, L]
            Tensor X3(actual, 1, SIG_SIGNAL_LEN);
            for (int b = 0; b < actual; b++)
                for (int l = 0; l < SIG_SIGNAL_LEN; l++)
                    X3.at(b, 0, l) = X_batch.at(b, l);

            Tensor logits = model.forward(X3);
            float loss = cross_entropy_forward(logits, y_batch);
            epoch_loss += loss;

            opt.zero_grad(params);
            Tensor grad = cross_entropy_backward(logits, y_batch);
            model.backward(grad);
            opt.step(params);
            n_batches++;
            }

        if (tcfg.verbose && (epoch + 1) % tcfg.print_every == 0)
            printf("Epoch [%3d/%3d]  loss: %.6f\n",
                   epoch + 1, tcfg.epochs, epoch_loss / (float)n_batches);
    }

    TINY_AI_FREE(y_batch);

    // ---- Evaluate ----
    auto eval_accuracy = [&](Dataset &ds, const char *tag) {
        ds.reset();
        int correct = 0, total = 0;
        int *yb  = (int *)TINY_AI_MALLOC((size_t)batch_size * sizeof(int));
        int *yp  = (int *)TINY_AI_MALLOC((size_t)batch_size * sizeof(int));
        Tensor Xb;
        while (true)
        {
            int actual = ds.next_batch(Xb, yb, batch_size);
            if (actual == 0) break;
            Tensor X3(actual, 1, SIG_SIGNAL_LEN);
            for (int b = 0; b < actual; b++)
                for (int l = 0; l < SIG_SIGNAL_LEN; l++)
                    X3.at(b, 0, l) = Xb.at(b, l);
            model.predict(X3, yp);
            for (int i = 0; i < actual; i++) if (yp[i] == yb[i]) correct++;
            total += actual;
        }
        TINY_AI_FREE(yb); TINY_AI_FREE(yp);
        printf("  %s accuracy: %.2f%%\n", tag, 100.0f * correct / total);
    };

    printf("\n--- Float32 Results ---\n");
    eval_accuracy(train_ds, "Train");
    eval_accuracy(test_ds,  "Test ");
#else
    printf("(Training disabled)\n");
#endif

    // ---- FP8 compression demo ----
    fp8_compression_demo();

    TINY_AI_FREE(X_raw);
    TINY_AI_FREE(Y_raw);

    printf("\nexample_cnn  DONE\n");
}

#endif // __cplusplus
```

## 新增片段：BatchNorm2D Demo

```cpp
#if TINY_AI_TRAINING_ENABLED
static void batchnorm2d_demo(Dataset &train_ds, Dataset &test_ds)
{
    auto *bn0 = new BatchNorm2D(8);
    auto *bn1 = new BatchNorm2D(16);

    Sequential model;
    model.add(new Conv1D(1, 8, 5));
    model.add(bn0);
    model.add(new ActivationLayer(ActType::RELU));
    model.add(new MaxPool1D(2));
    model.add(new Conv1D(8, 16, 3));
    model.add(bn1);
    model.add(new ActivationLayer(ActType::RELU));
    model.add(new MaxPool1D(2));
    model.add(new Flatten());
    model.add(new Dense(224, 32));
    model.add(new ActivationLayer(ActType::RELU));
    model.add(new Dense(32, 3));
    model.add(new ActivationLayer(ActType::SOFTMAX));

    model.set_training_mode(true);   // batch stats
    // ... 训练 ...

    model.set_training_mode(false);  // running stats
    // ... 推理评估 ...

    // 打印通道 running_mean / running_var
}
#endif
```
