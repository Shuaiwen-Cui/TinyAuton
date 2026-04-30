# 说明

!!! note "说明"
    `tiny_fp8` 在纯软件层面实现了 OCP 规范的 8-bit 浮点格式：E4M3FN 适合权重 / 激活，E5M2 适合梯度。ESP32-S3 没有 FP8 硬件，所有数据按 `uint8_t` 存储，运算时升回 `float32` 完成。

## 格式总览

| 格式 | 位排列 | bias | 最大值 | 最小正常值 | 特殊编码 |
| --- | --- | --- | --- | --- | --- |
| E4M3FN | `S EEEE MMM` | 7 | ±448.0 | 2⁻⁶ | NaN = 0x7F / 0xFF；**无 ±inf** |
| E5M2 | `S EEEEE MM` | 15 | ±57344.0 | 2⁻¹⁴ | ±Inf = 0x7C / 0xFC；NaN = 0x7D-0x7F / 0xFD-0xFF |

E4M3FN 牺牲了 ±inf 来换取额外的 4 个数值，是 OCP 推荐的「fitting normal」格式，用于权重 / 激活。E5M2 与 IEEE 754 子集结构一致，保留 ±inf / NaN，被推荐用于梯度的反向传播。

## E4M3FN

```cpp
uint8_t fp32_to_fp8_e4m3 (float val);
float   fp8_e4m3_to_fp32 (uint8_t fp8);
void    fp32_to_fp8_e4m3_batch(const float *src, uint8_t *dst, int n);
void    fp8_e4m3_to_fp32_batch(const uint8_t *src, float *dst, int n);
```

编码流程：

1. 抓取 sign / exponent / mantissa。
2. 若 `exp > 8` → clamp 到 ±448（编码 `0x7E`）；若 `exp < -9` → flush 到 ±0。
3. 重新偏置 `new_exp = exp + 7`。
4. 若 `new_exp <= 0`：subnormal，按 `(1 - new_exp)` 位右移；否则做 round-to-nearest-even 到 3-bit 尾数。
5. 若四舍五入溢出尾数最大值 → 增加指数，再次检查上溢。
6. 拼接 `S EEEE MMM`。

解码流程：

- `0x7F / 0xFF` → NaN。
- `exp == 0` → subnormal：`val = (-1)^S · 2⁻⁶ · (mant / 8)`。
- 否则 → normal：`val = (-1)^S · 2^(exp - 7) · (1 + mant / 8)`。

## E5M2

```cpp
uint8_t fp32_to_fp8_e5m2 (float val);
float   fp8_e5m2_to_fp32 (uint8_t fp8);
void    fp32_to_fp8_e5m2_batch(const float *src, uint8_t *dst, int n);
void    fp8_e5m2_to_fp32_batch(const uint8_t *src, float *dst, int n);
```

与 E4M3 类似，区别：

- bias = 15，新指数 `new_exp = exp + 15`。
- 尾数 round-to-nearest-even 到 2-bit。
- 大于 `±57344` → 编码 `±Inf`。
- IEEE 形式上的 ±Inf / NaN 都直接保留。

## 格式 dispatch 助手

```cpp
uint8_t fp32_to_fp8(float val, tiny_dtype_t dtype);   // 按 dtype 选 E4M3 / E5M2
float   fp8_to_fp32(uint8_t fp8, tiny_dtype_t dtype);
void    fp32_to_fp8_batch(const float *src, uint8_t *dst, int n, tiny_dtype_t dtype);
void    fp8_to_fp32_batch(const uint8_t *src, float *dst, int n, tiny_dtype_t dtype);
```

`tiny::quantize / dequantize`（`tiny_quant.hpp`）会根据 `params.dtype` 自动调用上述 dispatch 函数，所以应用层往往不需要直接接触 batch 函数。

## 使用模式

### 权重压缩存档

```cpp
QuantParams qp = calibrate(weight, TINY_DTYPE_FP8_E4M3);

uint8_t *buf = (uint8_t *)TINY_AI_MALLOC(weight.size);
quantize(weight, buf, qp);

// ... 写入 SPIFFS / NVS / 部署文件 ...

// 加载 + 解压
Tensor restored = Tensor::zeros_like(weight);
dequantize(buf, restored, qp);
```

`example_cnn.cpp` 演示了 4× 内存节省（`fp32 → e4m3`）+ 误差统计的流程，详见 [EXAMPLES/CNN](../../EXAMPLES/CNN/notes.md)。

### 梯度通信 / 检查点

将梯度先压成 E5M2 再写入 PSRAM 备份，可显著缩减检查点体积：

```cpp
QuantParams qp_g = calibrate(grad, TINY_DTYPE_FP8_E5M2);
uint8_t *gb = (uint8_t *)TINY_AI_MALLOC_PSRAM(grad.size);
quantize(grad, gb, qp_g);
```

需要时再 `dequantize` 回 fp32 继续训练。

## 精度与误差

- **E4M3FN**：相对误差约 1/8 = 12.5%（最低）；适合 ReLU 后已经稀疏化的权重 / 激活。
- **E5M2**：相对误差约 1/4 = 25%（更低），但范围大 128 倍，适合「分布尾部很长」的梯度。
- **建议**：与 INT8 互补使用 —— 当 INT8 由于过大动态范围（典型见 attention 权重）丢精度时，切换到 E4M3 往往更稳。

## 软件实现的代价

ESP32-S3 没有 FP8 ALU，因此每次量化 / 反量化都要走纯 C++ 的浮点位拼接（包含 `expf / powf`）。建议：

- 把 FP8 当作**存储格式**而非**计算格式**：解压一次，用 fp32 计算。
- 量化 / 反量化可以放到非热路径（载入权重时一次性完成）。
- batch 函数已内联到逐元素调用，但你可以在外层做更粗粒度的并行（FreeRTOS 任务、Cache-friendly 循环展开）。
