tiny_ai Library Structure
==========================

Directory Structure:
--------------------
tiny_ai/
├── include/              [✓] Header files
│   ├── tiny_ai.h         [✓] Main entry
│   └── tiny_ai_config.h  [✓] Configuration
├── core/                 [~] Core infrastructure
│   ├── tensor/           [✓] Tensor data structure
│   ├── graph/            [~] Computation graph
│   └── memory/           [ ] Memory management
├── operators/            [✓] Operators (forward + backward)
│   ├── activations/      [✓] Activation functions
│   ├── fc/               [✓] Fully connected layer
│   ├── conv/             [✓] Convolution layer
│   ├── pool/             [✓] Pooling layer
│   └── norm/             [✓] Normalization layer
├── loss/                 [ ] Loss functions
├── model/                [✓] Model wrappers
│   ├── mlp/              [✓] MLP
│   └── cnn/              [✓] CNN
├── train/                [~] Training components
│   ├── optimizer/        [✓] Optimizers
│   ├── trainer/          [✓] Training loop
│   └── dataloader/       [✓] Data loading
└── utils/                [ ] Utility functions
    ├── preprocess/        [ ] Preprocessing
    └── postprocess/       [ ] Postprocessing

Status Legend:
--------------
[ ] Not started
[~] In progress
[✓] Completed

Notes:
------
- Supports forward and backward propagation (training)
- Based on computation graph architecture
- Lightweight design for MCU
