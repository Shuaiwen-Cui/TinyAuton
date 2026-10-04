# TinyAuton

**Decisions where the data is.** A self-contained C/C++ stack that puts numerics, signal
processing and neural networks — including training — inside a microcontroller, so a node can
decide on its own instead of shipping raw data somewhere else.

[![build](https://github.com/Shuaiwen-Cui/TinyAuton/actions/workflows/build.yml/badge.svg)](https://github.com/Shuaiwen-Cui/TinyAuton/actions/workflows/build.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![docs](https://img.shields.io/badge/docs-EN%20%7C%20ZH-brightgreen.svg)](http://www.cuishuaiwen.com:9300/)

TinyAuton ships as ESP-IDF components with no external runtime dependency. Every kernel has a
portable C implementation and, where the platform offers one, dispatches to a vendor-accelerated
path instead. Forward *and* backward propagation run in firmware — the framework trains on the
device, it does not merely execute a model exported from a host.

---

## Modules

| Module | What it does | Contents |
|---|---|---|
| `tiny_math` | Linear algebra foundation | vectors, matrices, decompositions, eigenproblems |
| `tiny_dsp` | Signal processing | FIR/IIR filters, convolution, correlation, resampling, FFT, DWT, ICA |
| `tiny_ai` | Neural networks & on-device training | tensors, Dense/Conv/Pool/BatchNorm/LayerNorm/Attention, SGD & Adam, loss, Trainer, Dataset, FP8/INT8 quantisation |
| `tiny_toolbox` | Platform adaptation | timing, HAL access; the only module that changes when porting |

Higher modules depend only on the module below them, so `tiny_math` and `tiny_dsp` can be used
without pulling in `tiny_ai`.

---

## Quick start

Run the bundled training example on the MCU. The docs distinguish float evaluation from quantization API demonstrations:

```cpp
#include "tiny_ai.h"

extern "C" void app_main(void)
{
    example_mlp();
}
```

Three ready-to-flash ESP-IDF projects live under `CODE/`, each pulling in progressively more of
the stack:

| Project | Modules included |
|---|---|
| `AIoTNode-TinyAuton-MATH` | math, dsp, toolbox |
| `AIoTNode-TinyAuton-DSP` | math, dsp, toolbox |
| `AIoTNode-TinyAuton-AI` | math, dsp, ai, toolbox |

```bash
cd CODE/AIoTNode-TinyAuton-AI
idf.py set-target esp32s3
idf.py build flash monitor
```

Current projects and CI use ESP-IDF v6.0 on ESP32-S3. See [Getting started](DOC/docs/GETTING_STARTED/getting_started.en.md) and [Prerequisites](DOC/docs/PREREQUISITE/prerequisite.en.md)
for the toolchain setup.

---

## Platform support

| Platform | Status | Acceleration |
|---|---|---|
| ESP32 / ESP32-S3 | **Primary target** — developed and tested here | ESP-DSP / ESP-DL |
| Generic kernels | Available for selected functions; full stack port unverified | portable reference paths |
| STM32 | Adaptation layer reserved, **not yet implemented** | CMSIS-DSP planned |
| RISC-V | Adaptation layer reserved, **not yet implemented** | — |

Platform selection is configured in `tiny_math_config.h`. A complete port also needs build dependencies, allocation and direct platform calls adapted. See [architecture](DOC/docs/ARCHITECTURE/architecture.en.md).

---

## Tests

Each module includes an on-target test harness. MATH currently enables G1/G2/G3 matrix quality checks; DSP enables FFT/DWT/ICA; AI runs all three examples. Historical outputs are distinguished from current coverage.

```
tiny_math   tiny_vec_test, tiny_mat_test, tiny_matrix_test
tiny_dsp    tiny_fir_test, tiny_iir_test, tiny_conv_test, tiny_corr_test,
            tiny_resample_test, tiny_view_test, tiny_fft_test, tiny_dwt_test, tiny_ica_test
tiny_ai     example_mlp, example_cnn, example_attention  (train → quantise → evaluate)
```

Vector and matrix kernels are cross-checked against the ESP-DSP reference implementations, and
every test reports iteration timing so regressions show up as slowdowns, not just as wrong
answers. Full results and expected output: [test documentation](http://www.cuishuaiwen.com:9300/).

---

## Documentation

A bilingual (English / 中文) API reference is built with MkDocs Material and covers every module,
header and test.

```bash
python -m pip install -r DOC/requirements.txt
python DOC/tools/check_preservation.py
python -m mkdocs serve -f DOC/mkdocs.yml
```

Start with [getting started](DOC/docs/GETTING_STARTED/getting_started.en.md), [projects](DOC/docs/PROJECTS/projects.en.md) and [documentation maintenance](DOC/README.md).

---

## Repository layout

```
CODE/       three ESP-IDF example projects; middleware/ holds the tiny_* modules
DOC/        MkDocs sources for the API reference (EN / ZH)
ARCHIVED/   earlier iterations, kept for provenance
REF/        reference firmware and host-side scripts
```

---

## Background

TinyAuton is the general-purpose branch of work done during my Ph.D. at Nanyang Technological
University on edge intelligence for wireless sensing. Domain-specific modules (system
identification, measurement, damage assessment) are maintained separately and are not
open-sourced.

Related publications are listed on [my site](https://www.cuishuaiwen.com).

---

## License

MIT — see [LICENSE](LICENSE).
