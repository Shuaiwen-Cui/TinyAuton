#include "dl_esp32p4_s16.S"
#include "dl_esp32p4_common.S"


#void dl_esp32p4_s16_notq4d_bchw_w1_8_w2_8_simdnotq(int16_t *output_ptr, int16_t *input0_ptr,  int lenght);

    .align 2
    .text
    .global dl_esp32p4_s16_notq4d_bchw_w1_8_w2_8_simdnotq
    .type   dl_esp32p4_s16_notq4d_bchw_w1_8_w2_8_simdnotq, @function
    #.section .iram1
dl_esp32p4_s16_notq4d_bchw_w1_8_w2_8_simdnotq:
    .align 2

    # a0: int16_t *output_ptr
    # a1: int16_t *input0_ptr
    # a2: length

    lw a4, 44(a2)
    srai a3, a4, 3

    li t0, 0
loop:
    beq t0, a3, end
    esp.vld.128.ip q0, a1, 16
    esp.notq q2, q0
    esp.vst.128.ip q2, a0, 16
    addi t0, t0, 1
    j loop
end:
    ret





    .align 2
    .text
    .global dl_esp32p4_s16_notq4d_bchw_w1_8_w2_8_simdnotq_unaligned
    .type   dl_esp32p4_s16_notq4d_bchw_w1_8_w2_8_simdnotq_unaligned, @function
    #.section .iram1
dl_esp32p4_s16_notq4d_bchw_w1_8_w2_8_simdnotq_unaligned:
    .align 2
    esp32p4_push_128_stacks_4r  s0, s1, s8, s9

    # a0: int16_t *output_ptr
    # a1: int16_t *input0_ptr
    # a2: int16_t *input1_ptr no use
    # a3: void *args
    # a4: c_div_x_1
    # a5: input_shift
    # t3: output_scale
    # t4: output_shift
    # t5: c_remainder

    lw a4, 64(a2)
    lw t5, 76(a2)
    #lw a5, 88(a3)

    #bgez a5, dl_esp32p4_s16_unaligned_rescale_add2d_11c

# input0 exp = input1 exp = output exp

    esp.ld.128.usar.ip q5, a0, 0 #get output_ptr sar_byte
    esp.movx.r.sar.bytes s1

    bltz a4, dl_esp32p4_s16_notq4d_bchw_w1_8_w2_8_simdnotq_unaligned_small_remainder # channel < 16

    esp.ld.128.usar.ip q0, a1, 16
    #esp.ld.128.usar.ip q3, a2, 16
    esp.ld.128.usar.ip q1, a1, 16

    beqz s1, dl_esp32p4_s16_notq4d_bchw_w1_8_w2_8_simdnotq_unaligned_0
    li t0, 8
    beq s1, t0, dl_esp32p4_s16_notq4d_bchw_w1_8_w2_8_simdnotq_unaligned_1

    add t0, a4, x0
    blez t0, 1f
    0:
        esp.src.q.qup q2, q0, q1
        #esp.ld.128.usar.ip q4, a2, 16
        #esp.src.q.qup q5, q3, q4
        esp.notq q2, q2
        esp.ld.128.usar.ip q1, a1, 16
        dl_esp32p4_128b_unaligned_store0 q2, a0, s1
        addi t0, t0, -1
        bgtz t0, 0b
    1:
    addi a1, a1, -16
    add a1, a1, t5
    esp.movx.r.sar.bytes t6
    esp.src.q.qup q2, q0, q1

    #esp.ld.128.usar.xp q4, a2, t5
    #esp.movx.r.sar.bytes s0
    #esp.src.q.qup q5, q3, q4

    esp.notq q2, q2
    dl_esp32p4_128b_unaligned_store0 q2, a0, s1
    j dl_esp32p4_s16_notq4d_bchw_w1_8_w2_8_simdnotq_unaligned_remainder

    #output sar = 0
dl_esp32p4_s16_notq4d_bchw_w1_8_w2_8_simdnotq_unaligned_0:
    add t0, a4, x0
    blez t0, 3f
    2:
        esp.src.q.qup q2, q0, q1

        #esp.ld.128.usar.ip q4, a2, 16
        #esp.src.q.qup q5, q3, q4

        esp.notq q2, q2

        esp.ld.128.usar.ip q1, a1, 16
        esp.vst.128.ip q2, a0, 16
        addi t0, t0, -1
        bgtz t0, 2b
    3:
    addi a1, a1, -16
    add a1, a1, t5
    esp.movx.r.sar.bytes t6
    esp.src.q.qup q2, q0, q1

    #esp.ld.128.usar.xp q4, a2, t5
    #esp.movx.r.sar.bytes s0
    #esp.src.q.qup q5, q3, q4

    esp.notq q2, q2
    esp.vst.128.ip q2, a0, 16
    j dl_esp32p4_s16_notq4d_bchw_w1_8_w2_8_simdnotq_unaligned_remainder

    #output sar = 8
dl_esp32p4_s16_notq4d_bchw_w1_8_w2_8_simdnotq_unaligned_1:
    add t0, a4, x0
    blez t0, 5f
    4:
        esp.src.q.qup q2, q0, q1

        #esp.ld.128.usar.ip q4, a2, 16
        #esp.src.q.qup q5, q3, q4

        esp.notq q2, q2

        esp.ld.128.usar.ip q1, a1, 16
        dl_esp32p4_128b_unaligned_store1 q2, a0
        addi t0, t0, -1
        bgtz t0, 4b
    5:
    addi a1, a1, -16
    add a1, a1, t5
    esp.movx.r.sar.bytes t6
    esp.src.q.qup q2, q0, q1

    #esp.ld.128.usar.xp q4, a2, t5
    #esp.movx.r.sar.bytes s0
    #esp.src.q.qup q5, q3, q4

    esp.notq q2, q2
    dl_esp32p4_128b_unaligned_store1 q2, a0
    j dl_esp32p4_s16_notq4d_bchw_w1_8_w2_8_simdnotq_unaligned_remainder

dl_esp32p4_s16_notq4d_bchw_w1_8_w2_8_simdnotq_unaligned_small_remainder:
    esp.ld.128.usar.xp q0, a1, t5
    esp.movx.r.sar.bytes t6

    #esp.ld.128.usar.xp q3, a2, t5
    #esp.movx.r.sar.bytes s0

dl_esp32p4_s16_notq4d_bchw_w1_8_w2_8_simdnotq_unaligned_remainder:

    beqz t5, dl_esp32p4_s16_unaligned_add2d_end

    esp.ld.128.usar.ip q1, a1, 0
    esp.movx.w.sar.bytes t6
    esp.src.q q2, q0, q1

    #esp.ld.128.usar.ip q4, a2, 0
    #esp.movx.w.sar.bytes s0
    #esp.src.q q5, q3

    esp.notq q2, q2

    srli t5, t5, 1
    dl_esp32p4_s16_store_remainder q2, t5, s1, a0

dl_esp32p4_s16_unaligned_add2d_end:

    esp32p4_pop_128_stacks_4r  s0, s1, s8, s9
    ret
