# Projects and versions

This page describes projects and entries in the current working tree. Directory names are not release guarantees. The projects are configured for ESP32-S3 and ESP-IDF 6.0.

## Project composition

| Project | Middleware | Default entry |
|---|---|---|
| `AIoTNode-TinyAuton-MATH` | Toolbox, Math, DSP | C++ matrix tests |
| `AIoTNode-TinyAuton-DSP` | Toolbox, Math, DSP | FFT, DWT and ICA tests |
| `AIoTNode-TinyAuton-AI` | Toolbox, Math, DSP, AI | MLP, CNN1D and Attention examples |

Projects live under `CODE/`. MATH also contains DSP: its name describes the test focus rather than strict component removal.

## Implementation sources for documentation

| Section | Primary source |
|---|---|
| Math | MATH project; `tiny_math` files currently match across all three projects |
| DSP | DSP project; its `tiny_dsp` files currently match the AI project |
| AI | AI project |
| Toolbox | AI project; `tiny_toolbox` files currently match across all three projects |

The MATH project's DSP copy differs from DSP/AI in 21 source, header or test files. The newer multilevel DWT decomposition adds `cD_lens_out/cD_total_len`, and reconstruction requires `cD_lens`. Check the local headers when using DSP from MATH; do not mix these versions. This documentation update did not synchronize firmware copies.

## Relationship to TinySHM

At comparison time, the AI project's TinyAI, TinyDSP and TinyToolbox source files matched TinySHM Core, allowing reuse of some documentation organization and interpretation. TinySHM's TinyMath adds separate C modules including `linalg/cfloat/decomp/eigen/iterative`; those directories are absent here. TinyAuton's C++ `tiny::Mat` still provides its own decomposition, solving and eigenvalue methods.

TinyMeasurement, TinySysid, TinyDamage, TinyBench and TinyOrch are absent from these TinyAuton projects. Structural dynamics examples in matrix tests illustrate numerical methods; they do not constitute a complete SHM application module.

## Historical and reference directories

`ARCHIVED/` contains earlier projects, and `REF/` contains reference firmware and host scripts. Current API guidance follows the listed `CODE/` projects. Historical source excerpts and logs remain intact and are identified separately from current calls.

[Build and verify](../GETTING_STARTED/getting_started.md)
