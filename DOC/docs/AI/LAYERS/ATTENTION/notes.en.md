# Notes

!!! note "Notes"
    `Attention` implements multi-head self-attention — the heart of the Transformer architecture. The implementation supports any `embed_dim` / `num_heads` (must divide evenly) and ships with a complete backward pass, enabling on-device fine-tuning.

## MATH

Input `x` of shape `[batch, seq_len, embed_dim]`, denoted `B / S / E`. Per-head dimension `D = E / H`.

### Q / K / V projections

\[
Q = x W_q^\top + b_q,\quad
K = x W_k^\top + b_k,\quad
V = x W_v^\top + b_v
\]

Shapes: all `[B, S, E]`.

### Multi-head split

For each head `h = 0..H-1`, slice Q / K / V along the last dim into `[B, S, D]`.

### Scaled dot-product attention

Per head:

\[
A_{b, s_1, s_2} = \frac{Q_{b, s_1, :}\,K_{b, s_2, :}^\top}{\sqrt{D}}
\]

\[
A = \mathrm{softmax}_{s_2}(A)
\]

\[
\mathrm{ctx}_{b, s_1, :} = \sum_{s_2} A_{b, s_1, s_2}\,V_{b, s_2, :}
\]

### Concat + output projection

Write each head's `ctx_h` back to `ctx[:, :, h*D : (h+1)*D]`, then apply the final linear:

\[
y = \mathrm{ctx}\,W_o^\top + b_o
\]

Output shape `[B, S, E]` matches the input.

## CLASS DEFINITION

```cpp
class Attention : public Layer
{
public:
    Tensor Wq, Wk, Wv, Wo;     // [E, E] projection weights
    Tensor bq, bk, bv, bo;     // [E]    biases (optional)
#if TINY_AI_TRAINING_ENABLED
    Tensor dWq, dWk, dWv, dWo;
    Tensor dbq, dbk, dbv, dbo;
#endif

    Attention(int embed_dim, int num_heads, bool use_bias = true);

    Tensor forward (const Tensor &x) override;
    Tensor backward(const Tensor &grad_out) override;
    void   collect_params(std::vector<ParamGroup> &groups) override;
};
```

`embed_dim % num_heads == 0` is required; `head_dim = embed_dim / num_heads`.

## TRAINING CACHES

- `x_cache_`: the original input, used in backward to compute `dWq/k/v` and `dx`.
- `Q_cache_ / K_cache_ / V_cache_`: full projection outputs, sliced per head in backward.
- `A_cache_`: attention weights of shape `[B*H, S, S]`, required by the softmax backward.
- `ctx_cache_`: concatenated context, used by `Wo` backward.

## BACKWARD STEPS

1. **Wo backward**
    - `dWo += ctx^T @ grad_out` (accumulated over batch×seq).
    - `dctx  = grad_out @ Wo`.
2. **Multi-head attention backward** (per head)
    - `dV  = A^T @ dctx_h`
    - `dA  = dctx_h @ V^T`
    - Softmax backward: `dS = A * (dA - rowsum(dA*A)) * scale`
    - `dQ  = dS @ K`, `dK = dS^T @ Q`, then accumulate the per-head gradient back into the full `[B, S, E]` tensor.
3. **Wq / Wk / Wv backward**
    - For each weight: `dW += x^T @ dProj`, `dx += dProj @ W`.

## EXAMPLE

### Tiny Transformer on Iris

```cpp
const int SEQ_LEN = 4;       // treat 4 features as 4 tokens
const int EMB_DIM = 8;
const int N_HEADS = 2;

Dense           embed_proj(IRIS_N_FEATURES, SEQ_LEN * EMB_DIM, true);
ActivationLayer embed_act(ActType::RELU);
Attention       attn(EMB_DIM, N_HEADS, true);
GlobalAvgPool   gap;
Dense           classifier(EMB_DIM, IRIS_N_CLASSES, true);

// forward
Tensor e0 = embed_proj.forward(X_batch);     // [B, 32]
Tensor e1 = embed_act.forward(e0);           // [B, 32]
e1.reshape_3d(B, SEQ_LEN, EMB_DIM);          // [B, 4, 8]
Tensor a0     = attn.forward(e1);            // [B, 4, 8]
Tensor p0     = gap.forward(a0);             // [B, 8]
Tensor logits = classifier.forward(p0);      // [B, 3]
```

Full source under [EXAMPLES/ATTENTION](../../EXAMPLES/ATTENTION/notes.md).

## RESOURCES

- **Parameters**: `4 * E^2 + 4 * E` (with biases).
- **Activation memory**: `A_cache_` size `B * H * S^2` — quadratic in sequence length.
- **Complexity**: `O(B * H * S^2 * D + B * S * E^2)`.

!!! warning "ESP32-S3 budget"
    With limited PSRAM, keep `seq_len ≤ 64` or partition into smaller attention blocks. `embed_dim ≤ 64` plus `num_heads = 2~4` is comfortably feasible on ESP32-S3.
