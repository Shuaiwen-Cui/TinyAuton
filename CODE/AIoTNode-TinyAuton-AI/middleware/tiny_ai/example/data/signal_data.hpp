/**
 * @file signal_data.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Synthetic 1-D signal dataset embedded as C++ arrays.
 *
 *  3 classes, 50 samples each = 150 total.
 *  Each sample is 64 float32 time-domain values (sampling rate 64 Hz, 1 s).
 *
 *  Class 0: Low-frequency  sinusoid  f = 4 Hz  + Gaussian noise (σ = 0.10)
 *  Class 1: Medium-frequency sinusoid f = 8 Hz  + Gaussian noise (σ = 0.10)
 *  Class 2: High-frequency  sinusoid  f = 16 Hz + Gaussian noise (σ = 0.10)
 *
 *  The noise seeds are deterministic (LCG), so the dataset is fully
 *  reproducible and requires no file-system access.
 *
 * @version 1.0
 * @date 2026-04-30
 * @copyright Copyright (c) 2026
 */

#pragma once

#include <cmath>
#include <cstdint>

#ifdef __cplusplus

namespace tiny_data
{

// ============================================================================
// Synthetic signal dataset constants
// ============================================================================

static constexpr int SIG_N_SAMPLES   = 150;
static constexpr int SIG_N_CLASSES   = 3;
static constexpr int SIG_SIGNAL_LEN  = 64;    ///< Samples per signal
static constexpr int SIG_SAMPLES_PER_CLASS = 50;

static const char * const SIG_CLASS_NAMES[SIG_N_CLASSES] = {
    "LowFreq-4Hz", "MidFreq-8Hz", "HighFreq-16Hz"
};

// ============================================================================
// On-demand signal generator — avoids embedding 9600 floats in Flash
// ============================================================================

/**
 * @brief Fill a [SIG_N_SAMPLES][SIG_SIGNAL_LEN] buffer with the synthetic dataset.
 *        Called once before using the CNN example.
 *
 * @param out_X    Float buffer of size SIG_N_SAMPLES × SIG_SIGNAL_LEN
 * @param out_Y    Int buffer of size SIG_N_SAMPLES
 */
static inline void generate_signal_dataset(float *out_X, int *out_Y)
{
    // Frequencies for each class (Hz)
    const float freqs[SIG_N_CLASSES] = {4.0f, 8.0f, 16.0f};
    const float fs    = 64.0f;   // sampling rate (Hz)
    const float noise_std = 0.10f;

    // LCG parameters (Numerical Recipes)
    uint32_t seed = 2718281828u;
    auto lcg_rand = [&]() -> float {
        seed = seed * 1664525u + 1013904223u;
        // Box-Muller transform to get Gaussian sample
        seed = seed * 1664525u + 1013904223u;
        float u1 = (float)(seed)       / 4294967296.0f + 1e-7f;
        seed = seed * 1664525u + 1013904223u;
        float u2 = (float)(seed >> 1) / 2147483648.0f;
        return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    };

    for (int cls = 0; cls < SIG_N_CLASSES; cls++)
    {
        float freq = freqs[cls];
        for (int s = 0; s < SIG_SAMPLES_PER_CLASS; s++)
        {
            int sample_idx = cls * SIG_SAMPLES_PER_CLASS + s;
            // Phase offset per sample for variety
            float phase = (float)s * 0.1f;
            float *sig = out_X + sample_idx * SIG_SIGNAL_LEN;
            for (int t = 0; t < SIG_SIGNAL_LEN; t++)
            {
                float time = (float)t / fs;
                sig[t] = sinf(2.0f * 3.14159265f * freq * time + phase)
                         + noise_std * lcg_rand();
            }
            out_Y[sample_idx] = cls;
        }
    }
}

} // namespace tiny_data

#endif // __cplusplus
