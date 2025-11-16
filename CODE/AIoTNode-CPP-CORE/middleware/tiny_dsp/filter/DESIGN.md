# FIR & IIR Filter Design Document

## 设计理念

### 1. **模块化设计**
- **FIR** 和 **IIR** 分离为独立模块，便于维护和扩展
- 每个模块包含：设计函数、应用函数、实时处理支持
- 遵循 `tiny_dsp` 统一的 API 风格

### 2. **双模式支持**
- **批量处理模式**：处理整个数组（适合离线分析）
- **实时处理模式**：逐样本处理（适合实时系统、传感器数据流）

### 3. **平台优化**
- ESP32：使用 ESP-DSP 库优化（`dsps_fir_f32`, `dsps_biquad_f32`）
- 通用平台：提供 C 语言实现

---

## FIR 滤波器设计

### 核心特点
- ✅ **总是稳定**（只有零点，没有极点）
- ✅ **可实现线性相位**（对称系数）
- ✅ **实现简单**（基于卷积，可复用 `tiny_conv`）

### API 设计

#### 1. 滤波器设计函数
```c
// 低通滤波器设计
tiny_fir_design_lowpass(cutoff_freq, num_taps, window, coefficients);

// 高通、带通、带阻类似
```

**设计方法**：
- **Window Method（窗函数法）**：简单实用，适合大多数场景
  - 矩形窗、Hamming、Hanning、Blackman
  - 未来可扩展：Kaiser 窗（可调参数）
- **未来扩展**：Equiripple（Parks-McClellan）、频率采样法

#### 2. 滤波器应用函数
```c
// 批量处理（使用卷积）
tiny_fir_filter_f32(input, input_len, coefficients, num_taps, output, padding_mode);

// 实时处理（使用状态结构）
tiny_fir_init(&filter, coefficients, num_taps);
output = tiny_fir_process_sample(&filter, input);
```

**实现方式**：
- 批量处理：直接调用 `tiny_conv_ex_f32`（复用现有代码）
- 实时处理：维护延迟线（delay line）状态

### 设计要点
1. **系数对称性**：线性相位 FIR 需要对称系数
2. **阶数选择**：通常选择奇数阶（`num_taps` 为奇数）
3. **归一化频率**：`0.0` 到 `0.5`（`0.5` = Nyquist 频率）

---

## IIR 滤波器设计

### 核心特点
- ⚠️ **可能不稳定**（有极点，需要稳定性检查）
- ✅ **计算效率高**（相同规格下比 FIR 需要更少系数）
- ✅ **过渡带更陡峭**（相同阶数下性能更好）

### API 设计

#### 1. 滤波器设计函数
```c
// 低通滤波器设计
tiny_iir_design_lowpass(cutoff_freq, order, design_method, ripple_db, b_coeffs, a_coeffs);
```

**设计方法**：
- **Butterworth**：最大平坦响应，无纹波
- **Chebyshev Type I**：通带等纹波，过渡带更陡
- **Chebyshev Type II**：阻带等纹波
- **未来扩展**：Elliptic、Bessel

#### 2. 滤波器结构
- **Direct Form II Transposed**：最常用的实现结构
  - 状态变量少
  - 数值稳定性好
  - 适合定点实现

- **Biquad（二阶节）**：用于高阶滤波器级联
  - 每个 biquad 是二阶 IIR
  - 可以级联多个 biquad 实现高阶滤波器
  - 更灵活，便于优化

#### 3. 滤波器应用函数
```c
// 批量处理
tiny_iir_filter_f32(input, input_len, b_coeffs, num_b, a_coeffs, num_a, output, initial_state);

// 实时处理
tiny_iir_init(&filter, b_coeffs, num_b, a_coeffs, num_a);
output = tiny_iir_process_sample(&filter, input);
```

### 设计要点
1. **稳定性检查**：设计后需要验证极点是否在单位圆内
2. **系数归一化**：通常 `a[0] = 1.0`（归一化形式）
3. **初始条件**：批量处理时可指定初始状态

---

## 使用场景对比

### 选择 FIR 的场景
- ✅ 需要线性相位（音频处理、通信系统）
- ✅ 需要保证稳定性（关键系统）
- ✅ 滤波器阶数不高（< 100 阶）
- ✅ 实时性要求不高

### 选择 IIR 的场景
- ✅ 需要陡峭的过渡带（相同阶数下性能更好）
- ✅ 计算资源有限（需要更少系数）
- ✅ 滤波器阶数较高（> 50 阶）
- ✅ 可以接受非线性相位

---

## 实现优先级

### Phase 1: 基础功能（推荐先实现）
1. **FIR**：
   - ✅ Window 方法设计（Hamming, Hanning, Blackman）
   - ✅ 低通、高通滤波器设计
   - ✅ 批量处理（基于卷积）
   - ✅ 实时处理（状态结构）

2. **IIR**：
   - ✅ Butterworth 设计
   - ✅ 低通、高通滤波器设计
   - ✅ 批量处理
   - ✅ 实时处理（Direct Form II）

### Phase 2: 扩展功能
1. **FIR**：
   - 带通、带阻设计
   - Kaiser 窗（可调参数）
   - 系数对称性检查

2. **IIR**：
   - Chebyshev 设计
   - 带通、带阻设计
   - Biquad 级联支持
   - 稳定性检查

### Phase 3: 高级功能
1. **FIR**：
   - Equiripple 设计（Parks-McClellan）
   - 频率采样法

2. **IIR**：
   - Elliptic 设计
   - Bessel 设计
   - 自适应滤波器

---

## 代码组织

```
filter/
├── tiny_fir.h          # FIR 滤波器头文件
├── tiny_fir.c          # FIR 滤波器实现
├── tiny_fir_test.h     # FIR 测试头文件
├── tiny_fir_test.c     # FIR 测试实现
├── tiny_iir.h          # IIR 滤波器头文件
├── tiny_iir.c          # IIR 滤波器实现
├── tiny_iir_test.h     # IIR 测试头文件
├── tiny_iir_test.c     # IIR 测试实现
└── DESIGN.md           # 本文档
```

---

## 测试策略

### FIR 测试
1. **设计测试**：验证设计的滤波器系数是否正确
2. **频率响应测试**：使用 FFT 验证频率响应
3. **实时处理测试**：验证状态管理正确性
4. **边界测试**：测试极端参数（如 `cutoff_freq = 0.0` 或 `0.5`）

### IIR 测试
1. **设计测试**：验证系数计算正确性
2. **稳定性测试**：验证极点位置（应在单位圆内）
3. **频率响应测试**：验证通带、阻带特性
4. **数值稳定性测试**：长时间运行测试

---

## 注意事项

1. **内存管理**：
   - FIR/IIR 状态结构需要动态分配内存
   - 使用后必须调用 `deinit` 释放内存

2. **数值精度**：
   - 浮点运算可能累积误差
   - IIR 滤波器需要特别注意数值稳定性

3. **实时处理**：
   - 状态结构在多次调用间保持
   - 需要正确初始化延迟线

4. **平台差异**：
   - ESP32 使用优化库
   - 其他平台使用通用 C 实现

