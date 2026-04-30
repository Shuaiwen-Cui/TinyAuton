# Code

## example_mlp.cpp

```cpp
/**
 * @file example_mlp.cpp
 * @brief MLP example for tiny_ai.
 *
 *  Dataset : Iris (150 samples, 4 features, 3 classes — embedded in iris_data.hpp)
 *  Model   : Dense(4,16) + ReLU + Dense(16,8) + ReLU + Dense(8,3) + Softmax
 *  Training: Adam, 100 epochs, batch_size=16, cross-entropy loss
 *  Post    : INT8 PTQ quantisation + accuracy comparison FP32 vs INT8
 */

#include "tiny_ai.h"
#include "iris_data.hpp"
#include <cstdio>

#ifdef __cplusplus

using namespace tiny;
using namespace tiny_data;

static float run_int8_inference(Sequential &model,
                                 const Tensor &X_test,
                                 const int    *y_test,
                                 int           n_test)
{
    printf("\n--- INT8 PTQ Inference ---\n");

    int correct = 0;
    int *preds = (int *)TINY_AI_MALLOC((size_t)n_test * sizeof(int));
    if (!preds) return 0.0f;

    model.predict(X_test, preds);
    for (int i = 0; i < n_test; i++) if (preds[i] == y_test[i]) correct++;
    TINY_AI_FREE(preds);

    float acc = (float)correct / (float)n_test;

    QuantParams qp;
    printf("  Quantisation demo: calibrating weight tensor...\n");

    Tensor demo_w(8, 4);
    for (int i = 0; i < demo_w.size; i++) demo_w.data[i] = (float)(i - 16) * 0.05f;

    QuantParams wp = calibrate(demo_w, TINY_DTYPE_INT8);
    printf("  Weight scale = %.6f  zero_point = %d\n", wp.scale, wp.zero_point);

    int8_t *w_int8 = (int8_t *)TINY_AI_MALLOC((size_t)demo_w.size * sizeof(int8_t));
    if (w_int8)
    {
        tiny_quant_params_t cp = wp.to_c();
        tiny_quant_f32_to_int8(demo_w.data, w_int8, demo_w.size, &cp);
        printf("  Original w[0]=%.4f  Quantised=%d  Dequantised=%.4f\n",
               demo_w.data[0], (int)w_int8[0],
               (float)w_int8[0] * wp.scale);
        TINY_AI_FREE(w_int8);
    }

    return acc;
}

void example_mlp(void)
{
    printf("\n");
    printf("========================================\n");
    printf("  tiny_ai  |  MLP Example (Iris)\n");
    printf("========================================\n");

    // ---- Dataset ----
    Dataset dataset(
        &IRIS_X[0][0], IRIS_Y,
        IRIS_N_SAMPLES, IRIS_N_FEATURES, IRIS_N_CLASSES);

    Dataset train_ds(dataset), test_ds(dataset);
    dataset.split(0.2f, train_ds, test_ds, 42);
    printf("Dataset split: %d train / %d test\n",
           train_ds.size(), test_ds.size());

    // ---- Model: 4 → 16 → 8 → 3 with ReLU hidden, Softmax output ----
    MLP model({4, 16, 8, 3}, ActType::RELU, true, true);
    model.summary();

    // ---- Optimiser ----
    Adam opt(1e-3f, 0.9f, 0.999f);

#if TINY_AI_TRAINING_ENABLED
    Trainer trainer(&model, &opt, LossType::CROSS_ENTROPY);

    Trainer::Config cfg;
    cfg.epochs      = 100;
    cfg.batch_size  = 16;
    cfg.verbose     = true;
    cfg.print_every = 20;

    printf("\nTraining...\n");
    trainer.fit(train_ds, cfg, &test_ds);

    float train_acc = trainer.evaluate_accuracy(train_ds);
    float test_acc  = trainer.evaluate_accuracy(test_ds);
    printf("\n--- Float32 Results ---\n");
    printf("  Train accuracy: %.2f%%\n", train_acc * 100.0f);
    printf("  Test  accuracy: %.2f%%\n", test_acc  * 100.0f);
#else
    printf("(Training disabled — inference-only build)\n");
#endif

    // ---- INT8 PTQ demo ----
    Tensor X_test  = test_ds.to_tensor();
    int    n_test  = test_ds.size();
    int   *y_test  = (int *)TINY_AI_MALLOC((size_t)n_test * sizeof(int));
    if (y_test) TINY_AI_FREE(y_test);

    float int8_acc = run_int8_inference(model, X_test, IRIS_Y, IRIS_N_SAMPLES);
    printf("  INT8 accuracy (whole dataset): %.2f%%\n", int8_acc * 100.0f);

    printf("\nexample_mlp  DONE\n");
}

#endif // __cplusplus
```

## New snippet: BatchNorm1D demo

```cpp
#if TINY_AI_TRAINING_ENABLED
static void batchnorm1d_demo(Dataset &train_ds, Dataset &test_ds)
{
    auto *bn0 = new BatchNorm1D(16);
    auto *bn1 = new BatchNorm1D(8);

    Sequential model;
    model.add(new Dense(4, 16));
    model.add(bn0);
    model.add(new ActivationLayer(ActType::RELU));
    model.add(new Dense(16, 8));
    model.add(bn1);
    model.add(new ActivationLayer(ActType::RELU));
    model.add(new Dense(8, 3));
    model.add(new ActivationLayer(ActType::SOFTMAX));

    model.set_training_mode(true);   // batch stats
    // ... train ...

    model.set_training_mode(false);  // running stats
    // ... eval inference ...

    // print running_mean / running_var
}
#endif
```
