# Notes

!!! note "Notes"
    `Dataset` wraps an external float32 matrix + label array into a shuffleable, splittable, iterable training dataset. It only owns an index array — the underlying data stays as a read-only view, which makes it natural to pin the matrix in read-only flash or PSRAM.

## CLASS DEFINITION

```cpp
class Dataset
{
public:
    Dataset(const float *X, const int *y,
            int n_samples, int n_features, int n_classes);

    Dataset();
    Dataset(const Dataset &);
    Dataset(Dataset &&) noexcept;
    Dataset &operator=(const Dataset &);
    Dataset &operator=(Dataset &&) noexcept;
    ~Dataset();

    void   shuffle(uint32_t seed = 0);
    void   reset();
    int    next_batch(Tensor &X_batch, int *y_batch, int batch_size);

    void   split(float test_ratio, Dataset &train_out, Dataset &test_out,
                 uint32_t seed = 0) const;

    int    n_samples()  const;
    int    n_features() const;
    int    n_classes()  const;

    Tensor to_tensor() const;
};
```

## DATA CONTRACT

- `X` is a row-major `n_samples × n_features` float matrix owned by the caller (typically a `static const float[]` in `iris_data.hpp` / `signal_data.hpp`).
- `y` is an `n_samples`-long array of class indices.
- `Dataset` keeps view pointers to `X` / `y` and an `int *indices_` array; the destructor only frees `indices_`.
- Copying / moving a `Dataset` never copies the underlying data, only the index array.

## shuffle / split

```cpp
void shuffle(uint32_t seed = 0);
```

Uses an LCG + Fisher–Yates pass to permute `indices_`, then resets `cursor_`. When `seed == 0`, the default seed `1234567891u` is used.

```cpp
void split(float test_ratio, Dataset &train_out, Dataset &test_out, uint32_t seed = 0) const;
```

- `n_test = round(n_samples * test_ratio)`, clamped to `[1, n_samples - 1]`.
- Copy + shuffle the index array; the first `n_train` indices go to `train_out`, the rest to `test_out`.
- Internally it uses a private `Dataset(X, y, n, F, C, given_indices)` constructor so each subset owns an independent copy of the indices.

```cpp
Dataset full(X, y, N, F, C);
Dataset train, test;
full.split(0.2f, train, test, 42);
```

## next_batch ITERATION

```cpp
int next_batch(Tensor &X_batch, int *y_batch, int batch_size);
```

- Pulls `actual = min(batch_size, n_samples - cursor_)` samples starting from `indices_[cursor_]`.
- If `X_batch.size != actual * n_features`, the tensor is reallocated as `Tensor(actual, n_features)`.
- Copies each row from `X` into `X_batch`; copies `y_[idx]` into `y_batch[i]`.
- Returns `actual`. A return value of 0 indicates end-of-epoch.

Typical loop:

```cpp
Dataset ds(X, y, N, F, C);
ds.shuffle(epoch);
ds.reset();
Tensor X_batch;
int *y_batch = (int *)TINY_AI_MALLOC(B * sizeof(int));
while (true)
{
    int actual = ds.next_batch(X_batch, y_batch, B);
    if (actual == 0) break;
    // forward / backward / step ...
}
```

`Trainer::fit()` already implements this loop.

## to_tensor

```cpp
Tensor to_tensor() const;
```

Copies all currently-indexed samples into a `[n_samples, n_features]` tensor (deep copy). Handy for one-shot inference / `Sequential::accuracy`.

## MEMORY BUDGET

- **Self**: `indices_` is `n_samples * sizeof(int)` — a few KB.
- **Per-batch**: `Tensor X_batch` is `B * F * 4` bytes, `y_batch` is `B * 4` — both reallocated on demand.
- **After split**: train + test each carry their own index copy but share `X` / `y`.

For typical ESP32-S3 IMU / vibration datasets (N ~ thousands, F ~ tens), the full `Dataset` overhead is in the single-digit KB range and lives comfortably in internal SRAM.
