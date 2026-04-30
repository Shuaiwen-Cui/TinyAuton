# Notes

!!! note "Notes"
    `Sequential` is the layer-stack container of `tiny_ai`: it runs `forward` in insertion order and `backward` in reverse. It owns every `Layer*` and frees them in its destructor.

## CLASS DEFINITION

```cpp
class Sequential
{
public:
    Sequential() = default;
    ~Sequential();   // delete every owned Layer*

    void add(Layer *layer);

    Tensor forward (const Tensor &x);

#if TINY_AI_TRAINING_ENABLED
    Tensor backward(const Tensor &grad_out);
    void   collect_params(std::vector<ParamGroup> &groups);
#endif

    void  summary()   const;
    void  predict (const Tensor &x, int *labels);
    float accuracy(const Tensor &x, const int *labels, int n_samples);

    Layer *operator[](int idx);
    int    num_layers() const;

protected:
    std::vector<Layer *> layers_;
};
```

## BUILDING A MODEL

```cpp
Sequential m;
m.add(new Dense(F, 128));
m.add(new ActivationLayer(ActType::RELU));
m.add(new Dense(128, num_classes));
m.add(new ActivationLayer(ActType::SOFTMAX));
```

`add()` takes **raw pointers** — `Sequential` assumes ownership and `delete`s each layer in its destructor. The caller must not `delete` those pointers themselves.

## FORWARD / BACKWARD

- **forward**: starts with `Tensor out = x.clone()` (so the input is not mutated), then loops `out = layers_[i]->forward(out)`.
- **backward**: walks from the last layer to the first, `g = layers_[i]->backward(g)`, returning `dL/dx`.
- **collect_params**: skips layers with `trainable == false`, otherwise dispatches to their own `collect_params`.

## predict / accuracy

```cpp
void  predict (const Tensor &x, int *labels);
float accuracy(const Tensor &x, const int *labels, int n_samples);
```

- `predict`: runs one forward pass and writes argmax labels.
- `accuracy`: calls `predict` internally and compares with the given labels. It runs one forward over the entire `x`, so `n_samples` should equal `x.rows()`.

## summary

`summary()` prints the layer index and name to stdout for quick inspection:

```txt
Sequential model  (4 layers)
--------------------
  [ 0] dense
  [ 1] activation
  [ 2] dense
  [ 3] activation
--------------------
```

## TRAINER INTEGRATION

```cpp
Sequential model;
// add layers ...

Adam opt(1e-3f);
Trainer trainer(&model, &opt, LossType::CROSS_ENTROPY);
trainer.fit(train_ds, cfg);
```

`Trainer::ensure_params_collected()` lazily calls `model.collect_params(params_)` and `optimizer_->init(params_)` on the first `fit`, so `Sequential` never has to expose its internal parameter list manually.

## SUBCLASSING: MLP / CNN1D

Both `MLP` and `CNN1D` inherit from `Sequential` and merely populate `layers_` in their constructors. `Trainer` can accept either the base `Sequential *` or these subclasses thanks to virtual dispatch on `forward / backward / collect_params`.
