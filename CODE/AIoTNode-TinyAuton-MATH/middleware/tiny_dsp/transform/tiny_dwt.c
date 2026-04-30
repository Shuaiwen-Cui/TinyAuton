/**
 * @file tiny_dwt.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_dwt | code | source
 * @version 1.0
 * @date 2025-04-29
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_dwt.h" // TinyDWT Header

/* DWT FILTERS */
const float db1_lo_d[2] = {
    0.70710678f, 0.70710678f};

const float db1_hi_d[2] = {
    -0.70710678f, 0.70710678f};

const float db1_lo_r[2] = {
    0.70710678f, 0.70710678f};

const float db1_hi_r[2] = {
    0.70710678f, -0.70710678f};

const float db2_lo_d[4] = {
    -0.12940952f, 0.22414387f, 0.83651630f, 0.48296291f};

const float db2_hi_d[4] = {
    -0.48296291f, 0.83651630f, -0.22414387f, -0.12940952f};

const float db2_lo_r[4] = {
    0.48296291f, 0.83651630f, 0.22414387f, -0.12940952f};

const float db2_hi_r[4] = {
    -0.12940952f, -0.22414387f, 0.83651630f, -0.48296291f};

const float db3_lo_d[6] = {
    0.03522629f, -0.08544127f, -0.13501102f, 0.45987750f,
    0.80689151f, 0.33267055f};

const float db3_hi_d[6] = {
    -0.33267055f, 0.80689151f, -0.45987750f, -0.13501102f,
    0.08544127f, 0.03522629f};

const float db3_lo_r[6] = {
    0.33267055f, 0.80689151f, 0.45987750f, -0.13501102f,
    -0.08544127f, 0.03522629f};

const float db3_hi_r[6] = {
    0.03522629f, 0.08544127f, -0.13501102f, -0.45987750f,
    0.80689151f, -0.33267055f};

const float db4_lo_d[8] = {
    -0.01059740f, 0.03288301f, 0.03084138f, -0.18703481f,
    -0.02798377f, 0.63088077f, 0.71484657f, 0.23037781f};

const float db4_hi_d[8] = {
    -0.23037781f, 0.71484657f, -0.63088077f, -0.02798377f,
    0.18703481f, 0.03084138f, -0.03288301f, -0.01059740f};

const float db4_lo_r[8] = {
    0.23037781f, 0.71484657f, 0.63088077f, -0.02798377f,
    -0.18703481f, 0.03084138f, 0.03288301f, -0.01059740f};

const float db4_hi_r[8] = {
    -0.01059740f, -0.03288301f, 0.03084138f, 0.18703481f,
    -0.02798377f, -0.63088077f, 0.71484657f, -0.23037781f};

const float db5_lo_d[10] = {
    0.00333573f, -0.01258075f, -0.00624149f, 0.07757149f,
    -0.03224487f, -0.24229489f, 0.13842815f, 0.72430853f,
    0.60382927f, 0.16010240f};

const float db5_hi_d[10] = {
    -0.16010240f, 0.60382927f, -0.72430853f, 0.13842815f,
    0.24229489f, -0.03224487f, -0.07757149f, -0.00624149f,
    0.01258075f, 0.00333573f};

const float db5_lo_r[10] = {
    0.16010240f, 0.60382927f, 0.72430853f, 0.13842815f,
    -0.24229489f, -0.03224487f, 0.07757149f, -0.00624149f,
    -0.01258075f, 0.00333573f};

const float db5_hi_r[10] = {
    0.00333573f, 0.01258075f, -0.00624149f, -0.07757149f,
    -0.03224487f, 0.24229489f, 0.13842815f, -0.72430853f,
    0.60382927f, -0.16010240f};

const float db6_lo_d[12] = {
    -0.00107730f, 0.00477726f, 0.00055384f, -0.03158204f,
    0.02752287f, 0.09750161f, -0.12976687f, -0.22626469f,
    0.31525035f, 0.75113391f, 0.49462389f, 0.11154074f};

const float db6_hi_d[12] = {
    -0.11154074f, 0.49462389f, -0.75113391f, 0.31525035f,
    0.22626469f, -0.12976687f, -0.09750161f, 0.02752287f,
    0.03158204f, 0.00055384f, -0.00477726f, -0.00107730f};

const float db6_lo_r[12] = {
    0.11154074f, 0.49462389f, 0.75113391f, 0.31525035f,
    -0.22626469f, -0.12976687f, 0.09750161f, 0.02752287f,
    -0.03158204f, 0.00055384f, 0.00477726f, -0.00107730f};

const float db6_hi_r[12] = {
    -0.00107730f, -0.00477726f, 0.00055384f, 0.03158204f,
    0.02752287f, -0.09750161f, -0.12976687f, 0.22626469f,
    0.31525035f, -0.75113391f, 0.49462389f, -0.11154074f};

const float db7_lo_d[14] = {
    0.00035371f, -0.00180164f, 0.00042958f, 0.01255100f,
    -0.01657454f, -0.03802994f, 0.08061261f, 0.07130922f,
    -0.22403618f, -0.14390600f, 0.46978229f, 0.72913209f,
    0.39653932f, 0.07785205f};

const float db7_hi_d[14] = {
    -0.07785205f, 0.39653932f, -0.72913209f, 0.46978229f,
    0.14390600f, -0.22403618f, -0.07130922f, 0.08061261f,
    0.03802994f, -0.01657454f, -0.01255100f, 0.00042958f,
    0.00180164f, 0.00035371f};

const float db7_lo_r[14] = {
    0.07785205f, 0.39653932f, 0.72913209f, 0.46978229f,
    -0.14390600f, -0.22403618f, 0.07130922f, 0.08061261f,
    -0.03802994f, -0.01657454f, 0.01255100f, 0.00042958f,
    -0.00180164f, 0.00035371f};

const float db7_hi_r[14] = {
    0.00035371f, 0.00180164f, 0.00042958f, -0.01255100f,
    -0.01657454f, 0.03802994f, 0.08061261f, -0.07130922f,
    -0.22403618f, 0.14390600f, 0.46978229f, -0.72913209f,
    0.39653932f, -0.07785205f};

const float db8_lo_d[16] = {
    -0.00011748f, 0.00067545f, -0.00039174f, -0.00487035f,
    0.00874609f, 0.01398103f, -0.04408825f, -0.01736930f,
    0.12874743f, 0.00047248f, -0.28401554f, -0.01582911f,
    0.58535468f, 0.67563074f, 0.31287159f, 0.05441584f};

const float db8_hi_d[16] = {
    -0.05441584f, 0.31287159f, -0.67563074f, 0.58535468f,
    0.01582911f, -0.28401554f, -0.00047248f, 0.12874743f,
    0.01736930f, -0.04408825f, -0.01398103f, 0.00874609f,
    0.00487035f, -0.00039174f, -0.00067545f, -0.00011748f};

const float db8_lo_r[16] = {
    0.05441584f, 0.31287159f, 0.67563074f, 0.58535468f,
    -0.01582911f, -0.28401554f, 0.00047248f, 0.12874743f,
    -0.01736930f, -0.04408825f, 0.01398103f, 0.00874609f,
    -0.00487035f, -0.00039174f, 0.00067545f, -0.00011748f};

const float db8_hi_r[16] = {
    -0.00011748f, -0.00067545f, -0.00039174f, 0.00487035f,
    0.00874609f, -0.01398103f, -0.04408825f, 0.01736930f,
    0.12874743f, -0.00047248f, -0.28401554f, 0.01582911f,
    0.58535468f, -0.67563074f, 0.31287159f, -0.05441584f};

const float db9_lo_d[18] = {
    0.00003935f, -0.00025196f, 0.00023039f, 0.00184765f,
    -0.00428150f, -0.00472320f, 0.02236166f, 0.00025095f,
    -0.06763283f, 0.03072568f, 0.14854075f, -0.09684078f,
    -0.29327378f, 0.13319739f, 0.65728808f, 0.60482312f,
    0.24383467f, 0.03807795f};

const float db9_hi_d[18] = {
    -0.03807795f, 0.24383467f, -0.60482312f, 0.65728808f,
    -0.13319739f, -0.29327378f, 0.09684078f, 0.14854075f,
    -0.03072568f, -0.06763283f, -0.00025095f, 0.02236166f,
    0.00472320f, -0.00428150f, -0.00184765f, 0.00023039f,
    0.00025196f, 0.00003935f};

const float db9_lo_r[18] = {
    0.03807795f, 0.24383467f, 0.60482312f, 0.65728808f,
    0.13319739f, -0.29327378f, -0.09684078f, 0.14854075f,
    0.03072568f, -0.06763283f, 0.00025095f, 0.02236166f,
    -0.00472320f, -0.00428150f, 0.00184765f, 0.00023039f,
    -0.00025196f, 0.00003935f};

const float db9_hi_r[18] = {
    0.00003935f, 0.00025196f, 0.00023039f, -0.00184765f,
    -0.00428150f, 0.00472320f, 0.02236166f, -0.00025095f,
    -0.06763283f, -0.03072568f, 0.14854075f, 0.09684078f,
    -0.29327378f, -0.13319739f, 0.65728808f, -0.60482312f,
    0.24383467f, -0.03807795f};

const float db10_lo_d[20] = {
    -0.00001326f, 0.00009359f, -0.00011647f, -0.00068586f,
    0.00199241f, 0.00139535f, -0.01073318f, 0.00360655f,
    0.03321267f, -0.02945754f, -0.07139415f, 0.09305736f,
    0.12736934f, -0.19594627f, -0.24984642f, 0.28117234f,
    0.68845904f, 0.52720119f, 0.18817680f, 0.02667006f};

const float db10_hi_d[20] = {
    -0.02667006f, 0.18817680f, -0.52720119f, 0.68845904f,
    -0.28117234f, -0.24984642f, 0.19594627f, 0.12736934f,
    -0.09305736f, -0.07139415f, 0.02945754f, 0.03321267f,
    -0.00360655f, -0.01073318f, -0.00139535f, 0.00199241f,
    0.00068586f, -0.00011647f, -0.00009359f, -0.00001326f};

const float db10_lo_r[20] = {
    0.02667006f, 0.18817680f, 0.52720119f, 0.68845904f,
    0.28117234f, -0.24984642f, -0.19594627f, 0.12736934f,
    0.09305736f, -0.07139415f, -0.02945754f, 0.03321267f,
    0.00360655f, -0.01073318f, 0.00139535f, 0.00199241f,
    -0.00068586f, -0.00011647f, 0.00009359f, -0.00001326f};

const float db10_hi_r[20] = {
    -0.00001326f, -0.00009359f, -0.00011647f, 0.00068586f,
    0.00199241f, -0.00139535f, -0.01073318f, -0.00360655f,
    0.03321267f, 0.02945754f, -0.07139415f, -0.09305736f,
    0.12736934f, 0.19594627f, -0.24984642f, -0.28117234f,
    0.68845904f, -0.52720119f, 0.18817680f, -0.02667006f};

// Lookup tables for decomposition and reconstruction filters
const float *tiny_lo_d_table[TINY_WAVELET_COUNT] = {
    db1_lo_d, db2_lo_d, db3_lo_d, db4_lo_d, db5_lo_d,
    db6_lo_d, db7_lo_d, db8_lo_d, db9_lo_d, db10_lo_d};

const float *tiny_hi_d_table[TINY_WAVELET_COUNT] = {
    db1_hi_d, db2_hi_d, db3_hi_d, db4_hi_d, db5_hi_d,
    db6_hi_d, db7_hi_d, db8_hi_d, db9_hi_d, db10_hi_d};

const float *tiny_lo_r_table[TINY_WAVELET_COUNT] = {
    db1_lo_r, db2_lo_r, db3_lo_r, db4_lo_r, db5_lo_r,
    db6_lo_r, db7_lo_r, db8_lo_r, db9_lo_r, db10_lo_r};

const float *tiny_hi_r_table[TINY_WAVELET_COUNT] = {
    db1_hi_r, db2_hi_r, db3_hi_r, db4_hi_r, db5_hi_r,
    db6_hi_r, db7_hi_r, db8_hi_r, db9_hi_r, db10_hi_r};

const int tiny_filter_length_table[TINY_WAVELET_COUNT] = {
    2, 4, 6, 8, 10,
    12, 14, 16, 18, 20};

/* FUNCTION DEFINITIONS */

/**
 * @name tiny_dwt_decompose_f32
 * @brief Perform single-level discrete wavelet decomposition
 */
tiny_error_t tiny_dwt_decompose_f32(const float *input, int input_len,
                                    tiny_wavelet_type_t wavelet,
                                    float *cA, float *cD,
                                    int *cA_len, int *cD_len)
{
    if (!input || !cA || !cD || !cA_len || !cD_len)
        return TINY_ERR_DSP_NULL_POINTER;

    int filter_len = TINY_WAVELET_GET_LEN(wavelet);
    const float *lo_d = TINY_WAVELET_GET_LO_D(wavelet);
    const float *hi_d = TINY_WAVELET_GET_HI_D(wavelet);

    // TINY_CONV_CENTER mode internally performs FULL convolution first
    // Need buffer size: input_len + filter_len - 1 for FULL convolution
    int conv_full_len = input_len + filter_len - 1;
    float *temp_conv_full = (float *)calloc(conv_full_len, sizeof(float));
    if (!temp_conv_full)
        return TINY_ERR_DSP_MEMORY_ALLOC;

    // Temporary buffer for CENTER mode output (input_len samples)
    float *temp_conv = (float *)calloc(input_len, sizeof(float));
    if (!temp_conv)
    {
        free(temp_conv_full);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }

    // Low-pass filter convolution
    tiny_error_t err = tiny_conv_ex_f32(input, input_len, lo_d, filter_len, temp_conv_full,
                                        TINY_PADDING_SYMMETRIC, TINY_CONV_FULL);
    if (err != TINY_OK)
    {
        free(temp_conv_full);
        free(temp_conv);
        return err;
    }
    
    // Extract center portion (equivalent to TINY_CONV_CENTER mode)
    int center_start = (filter_len - 1) / 2;
    for (int i = 0; i < input_len; i++)
    {
        temp_conv[i] = temp_conv_full[center_start + i];
    }
    
    err = tiny_downsample_skip_f32(temp_conv, input_len, cA, cA_len, 1, 2);
    if (err != TINY_OK)
    {
        free(temp_conv_full);
        free(temp_conv);
        return err;
    }

    // High-pass filter convolution
    err = tiny_conv_ex_f32(input, input_len, hi_d, filter_len, temp_conv_full,
                           TINY_PADDING_SYMMETRIC, TINY_CONV_FULL);
    if (err != TINY_OK)
    {
        free(temp_conv_full);
        free(temp_conv);
        return err;
    }
    
    // Extract center portion
    for (int i = 0; i < input_len; i++)
    {
        temp_conv[i] = temp_conv_full[center_start + i];
    }
    
    err = tiny_downsample_skip_f32(temp_conv, input_len, cD, cD_len, 1, 2);
    if (err != TINY_OK)
    {
        free(temp_conv_full);
        free(temp_conv);
        return err;
    }

    free(temp_conv_full);
    free(temp_conv);
    return TINY_OK;
}

/**
 * @name tiny_dwt_reconstruct_f32
 * @brief Perform single-level discrete wavelet reconstruction
 */
tiny_error_t tiny_dwt_reconstruct_f32(const float *cA, const float *cD, int coeff_len,
                                      tiny_wavelet_type_t wavelet,
                                      float *output, int *output_len)
{
    if (!cA || !cD || !output || !output_len)
        return TINY_ERR_DSP_NULL_POINTER;

    int filter_len = TINY_WAVELET_GET_LEN(wavelet);
    const float *lo_r = TINY_WAVELET_GET_LO_R(wavelet);
    const float *hi_r = TINY_WAVELET_GET_HI_R(wavelet);

    int up_len = coeff_len * 2;
    int conv_len = up_len + filter_len - 1;
    // Correct offset: for FULL convolution, center extraction starts at (filter_len - 1) / 2
    // This aligns with TINY_CONV_CENTER mode logic
    int offset = (filter_len - 1) / 2;

    // Allocate memory for upsampled signals (only need up_len, not conv_len)
    float *upA = (float *)calloc(up_len, sizeof(float));
    float *upD = (float *)calloc(up_len, sizeof(float));
    if (!upA || !upD)
    {
        free(upA);
        free(upD);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }

    tiny_error_t err = tiny_upsample_zero_f32(cA, coeff_len, upA, up_len);
    if (err != TINY_OK)
    {
        free(upA);
        free(upD);
        return err;
    }
    err = tiny_upsample_zero_f32(cD, coeff_len, upD, up_len);
    if (err != TINY_OK)
    {
        free(upA);
        free(upD);
        return err;
    }

    // Allocate memory for convolution results (FULL mode output length)
    float *recA = (float *)calloc(conv_len, sizeof(float));
    float *recD = (float *)calloc(conv_len, sizeof(float));
    if (!recA || !recD)
    {
        free(upA);
        free(upD);
        free(recA);
        free(recD);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }

    err = tiny_conv_ex_f32(upA, up_len, lo_r, filter_len, recA,
                           TINY_PADDING_SYMMETRIC, TINY_CONV_FULL);
    if (err != TINY_OK)
        goto cleanup;

    err = tiny_conv_ex_f32(upD, up_len, hi_r, filter_len, recD,
                           TINY_PADDING_SYMMETRIC, TINY_CONV_FULL);
    if (err != TINY_OK)
        goto cleanup;

    // Extract center portion from FULL convolution result
    // With offset = (filter_len - 1) / 2, we can safely extract up_len samples
    // because: offset + up_len - 1 = (filter_len - 1) / 2 + up_len - 1
    //         = (filter_len - 1) / 2 + up_len - 1
    //         < (filter_len - 1) + up_len - 1
    //         = filter_len + up_len - 2
    //         < up_len + filter_len - 1 = conv_len
    for (int i = 0; i < up_len; i++)
    {
        int idx = i + offset;
        if (idx < conv_len)
        {
            output[i] = recA[idx] + recD[idx];
        }
        else
        {
            // This shouldn't happen with correct offset, but handle gracefully
            output[i] = 0.0f;
        }
    }

    *output_len = up_len;

cleanup:
    free(upA);
    free(upD);
    free(recA);
    free(recD);
    return err;
}

/**
 * @name tiny_dwt_multilevel_decompose_f32
 * @brief Perform multi-level DWT decomposition
 */
tiny_error_t tiny_dwt_multilevel_decompose_f32(const float *input, int input_len,
                                               tiny_wavelet_type_t wavelet, int levels,
                                               float **cA_out, float **cD_out, int *len_out)
{
    if (!input || !cA_out || !cD_out || !len_out || levels <= 0)
        return TINY_ERR_DSP_NULL_POINTER;

    float *current = (float *)malloc(sizeof(float) * input_len);
    if (!current)
        return TINY_ERR_DSP_MEMORY_ALLOC;
    memcpy(current, input, sizeof(float) * input_len);

    float *cD_all = (float *)malloc(sizeof(float) * input_len);
    int *cD_lens = (int *)malloc(sizeof(int) * levels);
    int cD_pos = 0;

    int current_len = input_len;
    for (int l = 0; l < levels; l++)
    {
        int cA_len = 0, cD_len = 0;
        float *cA_temp = (float *)malloc(sizeof(float) * current_len);
        float *cD_temp = (float *)malloc(sizeof(float) * current_len);
        if (!cA_temp || !cD_temp)
        {
            free(cA_temp);
            free(cD_temp);
            free(current);
            free(cD_all);
            free(cD_lens);
            return TINY_ERR_DSP_MEMORY_ALLOC;
        }

        tiny_error_t err = tiny_dwt_decompose_f32(current, current_len, wavelet, cA_temp, cD_temp, &cA_len, &cD_len);
        if (err != TINY_OK)
        {
            free(current);
            free(cD_all);
            free(cD_lens);
            free(cA_temp);
            free(cD_temp);
            return err;
        }

        memcpy(cD_all + cD_pos, cD_temp, sizeof(float) * cD_len);
        cD_lens[l] = cD_len;
        cD_pos += cD_len;

        free(current);
        current = cA_temp;
        current_len = cA_len;
        free(cD_temp);
    }

    *cA_out = current;
    *cD_out = cD_all;
    *len_out = current_len;

    free(cD_lens); // optional if not passed out
    return TINY_OK;
}

/**
 * @name tiny_dwt_coeffs_process
 * @brief Placeholder for user-defined coefficient processing
 */
void tiny_dwt_coeffs_process(float *cA, float *cD, int cA_len, int cD_len, int levels)
{
    // Currently does nothing, can be extended by user
    (void)cA;
    (void)cD;
    (void)cA_len;
    (void)cD_len;
    (void)levels;
}

/**
 * @name tiny_dwt_multilevel_reconstruct_f32
 * @brief Perform multi-level DWT reconstruction
 */
tiny_error_t tiny_dwt_multilevel_reconstruct_f32(const float *cA_init, const float *cD_all,
                                                 int final_len, tiny_wavelet_type_t wavelet, int levels,
                                                 float *output)
{
    if (!cA_init || !cD_all || !output)
        return TINY_ERR_DSP_NULL_POINTER;

    float *current = (float *)malloc(sizeof(float) * final_len);
    if (!current)
        return TINY_ERR_DSP_MEMORY_ALLOC;
    memcpy(current, cA_init, sizeof(float) * final_len);

    int cA_len = final_len;
    int cD_pos = 0;

    for (int l = 0; l < levels; l++)
    {
        int cD_len = cA_len;
        float *cD = (float *)malloc(sizeof(float) * cD_len);
        if (!cD)
        {
            free(current);
            return TINY_ERR_DSP_MEMORY_ALLOC;
        }

        memcpy(cD, cD_all + cD_pos, sizeof(float) * cD_len);
        cD_pos += cD_len;

        int out_len = 0;
        float *recon = (float *)malloc(sizeof(float) * cD_len * 2);
        if (!recon)
        {
            free(current);
            free(cD);
            return TINY_ERR_DSP_MEMORY_ALLOC;
        }

        tiny_error_t err = tiny_dwt_reconstruct_f32(current, cD, cD_len, wavelet, recon, &out_len);
        free(current);
        free(cD);
        if (err != TINY_OK)
        {
            free(recon);
            return err;
        }

        current = recon;
        cA_len = out_len;
    }

    memcpy(output, current, sizeof(float) * cA_len);
    free(current);

    return TINY_OK;
}