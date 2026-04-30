# 说明

!!! note "说明"

    重采样是信号处理中的重要步骤，用于改变信号的采样率，广泛应用于音频、视频等信号处理场景。本库提供三个层次的重采样功能：基于保留/跳过模式的下采样、基于零插入的上采样，以及基于线性插值的任意因子重采样。


---


## 1. 算法原理

### 1.1 保留/跳过模式的下采样

不同于简单的等距抽取（keep=1, skip=N），本库实现了 **保留-跳过模式** ：用户可以指定每个周期内保留多少连续样本（ `keep` ），跳过多少连续样本（ `skip` ）。

**模式示例**：
```
输入:  [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]
keep=1, skip=1  →  输出 = [s0, s2, s4, s6, s8]        (步长2)
keep=2, skip=1  →  输出 = [s0, s1, s3, s4, s6, s7, s9] (周期3)
keep=3, skip=2  →  输出 = [s0, s1, s2, s5, s6, s7]     (周期5)
```

该设计包含简单抽取（`keep=1`），同时允许分组保留，适用于分块平均预处理等场景。

**输出长度**：动态确定，近似公式

\[
L_{out} \approx \left\lceil L_{in} \cdot \frac{keep}{keep + skip} \right\rceil
\]

### 1.2 零插入上采样

给定整数扩展因子 \(F = target\_len / input\_len\)，将原始样本放置在输出中 \(F\) 的整数倍位置，其余位置填零：

\[
output[i] = \begin{cases}
input[i / F] & \text{当 } i \bmod F = 0 \text{ 且 } i/F < input\_len \\
0 & \text{其他情况}
\end{cases}
\]

当 `target_len` 不是 `input_len` 的精确整数倍时，超出最后一个有效源样本的尾部位置将被填零。

### 1.3 任意因子线性插值重采样

对于任意升/降采样（非整数倍率），库采用 **线性插值** ：

1. 计算比率：\(r = \frac{\mathrm{target}_{\mathrm{len}}}{\mathrm{input}_{\mathrm{len}}}\)
2. 对每个输出索引 \(i\)，找到在输入中对应的浮点位置：
   \[
   pos = i / r
   \]
3. 拆分为整数索引 \(idx = \lfloor pos \rfloor\) 和小数部分 \(frac = pos - idx\)。
4. 线性混合最近的两个输入样本：
   \[
   output[i] = input[idx] \cdot (1 - frac) + input[idx+1] \cdot frac
   \]
5. 末端钳制：如果 \(idx \ge \mathrm{input}_{\mathrm{len}} - 1\)，使用 `input[input_len - 1]`。


该方法为 O(N) 轻量级算法，**不包含** 抗混叠滤波——详见注意事项章节。

---

## 2. 代码设计理念

### 2.1 灵活的保留/跳过下采样

简单的等距抽取（`keep=1`）会丢弃大块样本而不考虑信号结构。保留-跳过模式允许用户保留连续分组，在以下场景中很有用：

- 信号的每个"块"具有特定含义（如分包传输数据）
- 模拟非矩形窗口后抽取的效果
- 控制短时事件保留程度

### 2.2 零插入作为构建模块

零插入被有意地与插值滤波分离。这使用户可以控制：

- 后续应用何种插值滤波器（如低通 FIR 核）
- 是否与 `tiny_conv_f32` 级联实现完整插值
- 保持上采样步骤本身无内存分配且速度快

### 2.3 线性插值追求简洁性

在资源受限的 MCU 上，完整的多相重采样代价高昂。线性插值提供：

- O(target_len) 时间，O(1) 辅助内存
- 当输入相对其带宽充分过采样时可接受的质量
- 可预测的性能特性（无动态分配）

### 2.4 边界保护

三个函数均包含下溢/溢出防护逻辑：

- `tiny_downsample_skip_f32`：`copy_n = min(keep, input_len - in_idx)` 防止过读
- `tiny_upsample_zero_f32`：`src < input_len` 防止 `target_len` 非精确倍数时的越界访问
- `tiny_resample_f32`：`index >= input_len - 1` 钳制防止读取数组末尾之外

---

## 3. API 接口 — 函数

### 3.1 `tiny_downsample_skip_f32`

```c
/**
 * @name tiny_downsample_skip_f32
 * @brief Downsample a signal by a given factor using skipping
 *
 * @param input pointer to the input signal array
 * @param input_len length of the input signal array
 * @param output pointer to the output signal array
 * @param output_len pointer to the length of the output signal array
 * @param keep number of samples to keep
 * @param skip number of samples to skip
 *
 * @return tiny_error_t
 */

tiny_error_t tiny_downsample_skip_f32(const float *input, int input_len,
                                       float *output, int *output_len,
                                       int keep, int skip)
{
    if (!input || !output || !output_len)
        return TINY_ERR_DSP_NULL_POINTER;

    if (input_len <= 0 || keep <= 0 || skip <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;



    int in_idx = 0;
    int out_idx = 0;


    while (in_idx < input_len)
    {

        // 保留 'keep' 个样本
        int copy_n = keep;
        if (in_idx + copy_n > input_len)
            copy_n = input_len - in_idx;

        for (int i = 0; i < copy_n; i++)
        {
            output[out_idx++] = input[in_idx++];
        }

        // 跳过 'skip' 个样本
        in_idx += skip;
    }

    *output_len = out_idx;

    return TINY_OK;
}

```

**描述**：


通过交替复制 `keep` 个连续样本并跳过 `skip` 个连续样本来下采样信号。重复直到输入耗尽。

**特点**：


- 通用保留-跳过模式（非简单等距抽取）
- 妥善处理不完整尾部块
- 通过 `output_len` 返回输出长度



**参数**：


| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `const float*` | 输入信号数组指针 |
| `input_len` | `int` | 输入信号长度（> 0） |
| `output` | `float*` | 输出缓冲区指针。调用者需分配至少 `ceil(input_len * keep / (keep + skip))` 的空间 |
| `output_len` | `int*` | [出参] 实际写入的样本数量 |
| `keep` | `int` | 每周期保留的连续样本数（≥ 1） |
| `skip` | `int` | 每周期跳过的连续样本数（≥ 1） |


**返回值**：


| 码值 | 含义 |
|------|------|
| `TINY_OK` | 下采样成功 |
| `TINY_ERR_DSP_NULL_POINTER` | `input`、`output` 或 `output_len` 为空 |
| `TINY_ERR_DSP_INVALID_PARAM` | `input_len ≤ 0`、`keep ≤ 0` 或 `skip ≤ 0` |


---


### 3.2 `tiny_upsample_zero_f32`







```c
/**
 * @name tiny_upsample_zero_f32
 * @brief Upsample a signal using zero-insertion between samples
 *
 * @param input pointer to the input signal array
 * @param input_len length of the input signal array
 * @param output pointer to the output signal array
 * @param target_len target length for the output signal array
 * @return tiny_error_t
 */

tiny_error_t tiny_upsample_zero_f32(const float *input, int input_len,
                                     float *output, int target_len)
{
    if (!input || !output)
        return TINY_ERR_DSP_NULL_POINTER;

    if (input_len <= 0 || target_len <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    int factor = target_len / input_len;
    if (factor <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    for (int i = 0; i < target_len; i++)
    {

        if (i % factor == 0)
        {
            int src = i / factor;
            if (src < input_len)
                output[i] = input[src];
            else
                output[i] = 0.0f;  // 超出末尾 → 填零
        }
        else
        {
            output[i] = 0.0f;
        }
    }

    return TINY_OK;
}
```

**描述**：


通过在原始样本间插入零来上采样信号。扩展因子为 `target_len / input_len`（整数除法）。当 `target_len` 不是 `input_len` 的精确整数倍时，源索引超出输入末尾的尾部位置将被填零。


**特点**：


- 整数倍零插入
- 边界安全：非精确倍数时保护越界读取
- 无内存分配



**参数**：


| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `const float*` | 输入信号数组指针 |
| `input_len` | `int` | 输入信号长度（> 0） |
| `output` | `float*` | 输出缓冲区指针。大小至少为 `target_len` |
| `target_len` | `int` | 输出信号目标长度（> 0） |


**返回值**：


| 码值 | 含义 |
|------|------|
| `TINY_OK` | 上采样成功 |
| `TINY_ERR_DSP_NULL_POINTER` | `input` 或 `output` 为空 |
| `TINY_ERR_DSP_INVALID_PARAM` | `input_len ≤ 0`、`target_len ≤ 0` 或 `factor = target_len / input_len ≤ 0` |


---

### 3.3 `tiny_resample_f32`





```c
/**
 * @name: tiny_resample_f32
 * @brief Resample a signal to a target length
 *
 * @param input pointer to the input signal array
 * @param input_len length of the input signal array
 * @param output pointer to the output signal array
 * @param target_len target length for the output signal array
 * @return tiny_error_t
 */
tiny_error_t tiny_resample_f32(const float *input,
                               int input_len,
                               float *output,
                               int target_len)
{
    if (!input || !output)
        return TINY_ERR_DSP_NULL_POINTER;

    if (input_len <= 0 || target_len <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    float ratio = (float)(target_len) / (float)(input_len);

    for (int i = 0; i < target_len; i++)
    {
        float pos = i / ratio;
        int index = (int)floorf(pos);
        float frac = pos - index;

        if (index >= input_len - 1)

            output[i] = input[input_len - 1]; // 末端钳制
        else
            output[i] = input[index] * (1.0f - frac) + input[index + 1] * frac;
    }

    return TINY_OK;
}


```

**描述**：


使用线性插值将信号重采样到目标长度。支持任意升采样和降采样比例——非整数倍率由插值公式自然处理。

**特点**：


- 非整数倍重采样
- 线性插值（计算开销低）
- 无动态内存分配
- 末端钳制防止越界访问



**参数**：


| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `const float*` | 输入信号数组指针 |
| `input_len` | `int` | 输入信号长度（> 0） |
| `output` | `float*` | 输出缓冲区指针。大小至少为 `target_len` |
| `target_len` | `int` | 输出信号目标长度（> 0） |


**返回值**：


| 码值 | 含义 |
|------|------|
| `TINY_OK` | 重采样成功 |
| `TINY_ERR_DSP_NULL_POINTER` | `input` 或 `output` 为空 |
| `TINY_ERR_DSP_INVALID_PARAM` | `input_len ≤ 0` 或 `target_len ≤ 0` |


---

## 4. 函数对比

| 特性 | `tiny_downsample_skip_f32` | `tiny_upsample_zero_f32` | `tiny_resample_f32` |
|------|---------------------------|-------------------------|---------------------|
| **方向** | 降低采样率 | 提高采样率 | 双向均可 |
| **因子类型** | 整数（keep+skip 模式） | 整数（factor = target/input） | 任意（允许非整数） |
| **插值方式** | 无（跳过样本） | 无（零插入） | 线性插值 |
| **抗混叠** | 不含 | 不含 | 不含 |
| **内存分配** | 无 | 无 | 无 |
| **输出长度** | 动态返回 | 用户指定 `target_len` | 用户指定 `target_len` |
| **边界处理** | 截断不完整周期 | 尾部填零 | 末端钳制 |
| **适用场景** | 已知保留/跳过模式 | 需要零填充版本以便后续滤波 | 快速重采样到任意长度 |

### 何时使用 `tiny_downsample_skip_f32`

- 需要降低采样率同时保留样本块
- 需要非均匀抽取模式（如每 3 个中保留 2 个）
- 保留-跳过模式与数据结构对齐

### 何时使用 `tiny_upsample_zero_f32`

- 需要提高采样率作为插值的第一步
- 计划后续应用重建滤波器
- 扩展因子为整数

### 何时使用 `tiny_resample_f32`

- 比率为非整数
- 需要一步完成重采样，无需外部滤波
- 信号已充分过采样（可避免混叠伪影）

---

## 5. ⚠️ 重要注意事项

### 5.1 下采样 — 混叠风险

`tiny_downsample_skip_f32` 执行 **纯选取** ，不含抗混叠滤波。如果输入信号中存在高于新奈奎斯特频率（\(f_s' / 2 = f_s / (2 \cdot stride)\)）的频率分量，将发生混叠。如果信号有显著高频内容，**调用此函数前先用低通滤波器预处理**（如 `tiny_fir_filter_f32`）。

### 5.2 上采样 — 零填充频谱

`tiny_upsample_zero_f32` 插入零会创建频谱镜像。通常需要后续应用重建低通滤波器（插值滤波器）。此函数设计为构建模块；与 `tiny_conv_f32` 或 `tiny_fir_filter_f32` 配合使用可完成完整的插值。

### 5.3 重采样 — 线性插值的局限

`tiny_resample_f32` 使用纯线性插值：

- **不含抗混叠**：降采样可能引入混叠伪影。
- **不含抗镜像**：升采样可能有阶梯效应。
- 当信号充分过采样（如 Nyquist 率的 10 倍以上）时质量可接受。
- 高质量重采样时，请先用适当的 FIR/IIR 低通滤波器进行预滤波（降采样）或后滤波（升采样）。

### 5.4 输出缓冲区大小

| 函数 | 最小输出缓冲区大小 |
|------|-------------------|
| `tiny_downsample_skip_f32` | `ceil(input_len * keep / (keep + skip))` |
| `tiny_upsample_zero_f32` | `target_len` |
| `tiny_resample_f32` | `target_len` |

对于 `tiny_downsample_skip_f32`，精确输出长度通过 `*output_len` 返回。

### 5.5 因子有效性

- `tiny_downsample_skip_f32`：`keep ≥ 1` 且 `skip ≥ 1`。
- `tiny_upsample_zero_f32`：`target_len / input_len ≥ 1`（必须 ≥ 1；仅支持零插入，不支持缩小）。
- `tiny_resample_f32`：任意 `target_len > 0`（升采样或降采样均可）。

### 5.6 平台独立性
三个函数均为**平台无关**的纯 C 实现，不含 `#if ESP32` 分支。在所有支持的 MCU 平台上行为一致。

