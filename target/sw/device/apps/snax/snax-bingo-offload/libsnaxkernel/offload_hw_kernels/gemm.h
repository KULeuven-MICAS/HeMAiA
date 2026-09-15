// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
//
// Core-level GEMM kernels (bingo-hw flow). Hand-maintained plain C header.
// Paired with device/runtime/snax/versacore/gemm_shapes.h, which holds the per-shape
// parameter table and shape-invariant widths. Both headers are validated
// against the hwcfg at build time by the versacore validate_shapes.py —
// any drift fails `make sw`.
//
// Kernels exposed:
//   - __snax_bingo_kernel_gemm_full       (configures streamer+versacore and runs)
//   - __snax_bingo_kernel_gemm_minimal    (reuses prior config, just starts/waits)
// Both return uint32_t (BINGO_RET_SUCC / BINGO_RET_FAIL). Shape-dependent
// parameters are read via `bingo_gemm_shape_params[array_shape_idx]` —
// a typed const struct from gemm_shapes.h. A single bounds check at the top of
// __snax_bingo_kernel_gemm_full handles out-of-range indices.

#pragma once

#include "../macros.h"
#include <snax_versacore_lib.h>
#include <gemm_shapes.h>

// =============================================================
// Core-level GEMM kernels (bingo-hw)
// =============================================================
//
// -------------------------------------------------------------
// Precision support & how to configure it
// -------------------------------------------------------------
// The MAC datatype is fixed by the hardware build (this is the int8
// `snax_versacore_to_cluster` config): the VersaCore array multiplies
// INT8 x INT8 and accumulates into an INT32 C/D. Element widths come from
// gemm_shapes.h: BINGO_A_ELEM_LEN = BINGO_B_ELEM_LEN = 8,
// BINGO_C_ELEM_LEN = BINGO_D32_ELEM_LEN = 32. Changing the *MAC* datatype
// (e.g. FP8/FP16 multiply) needs a different RTL build (e.g. the FP8
// `snax_versacore_dse_cluster`) and cannot be selected at runtime here.
//
// On this int8 build, precision is selected per call via the
// __snax_bingo_kernel_gemm_full_args_t fields below (no rebuild needed); they
// flow host -> device as: params.hjson -> the workload datagen -> these kernel
// args. The flags only change operand packing and the output stage:
//   - int4_a_enable / int4_b_enable : pack operand A / B to 4-bit
//       (a_elem_len / b_elem_len 8 -> 4). Inputs only; the MAC still runs on
//       the int8 array. The A/B channel-enable masks are rebuilt dynamically.
//   - quantization_enable (+ shift_i, multiplier_i, input_zp_i, output_zp_i):
//       requantize the INT32 accumulator down to an INT8 output
//       (d_elem_len 32 -> 8).
//   - int32tofp16_enable : convert the INT32 accumulator to an FP16 output
//       (d_elem_len 32 -> 16).
//   - none set : INT32 output (d_elem_len = 32, the baseline).
// quantization_enable and int32tofp16_enable are mutually exclusive (the only
// two output-stage extensions); both set -> BINGO_RET_FAIL.
//
// The 6 reachable precision modes (input_A x input_B -> output_D), i.e. the
// GEMM_PRECISIONS suites in
// target/sim/automation/sweep/versacore/testing_workload_gen.py. Each has a
// clean precision-named wrapper kernel below (preferred); gemm_full stays the
// low-level escape hatch that exposes all flags:
//   int8 x int8 -> int32   (all flags 0)                    -> gemm_i8i8_i32
//   int8 x int4 -> int32   (int4_b_enable)                  -> gemm_i8i4_i32
//   int4 x int4 -> int32   (int4_a_enable + int4_b_enable)  -> gemm_i4i4_i32
//   int8 x int8 -> int8    (quantization_enable + requant)  -> gemm_i8i8_i8
//   int8 x int4 -> fp16    (int4_b_enable + int32tofp16)    -> gemm_i8i4_f16
//   int8 x int8 -> fp16    (int32tofp16)                    -> gemm_i8i8_f16
// (B is the weight operand; int4_b packs B to 4-bit.)
// -------------------------------------------------------------

// -------------------------------------------------------------
// The "+C" term and accumPrevC  (D = A*B + C)
// How many output blocks did the writer actually land? The array retires blocks in
// order, so comparing the LAST word of each block against a snapshot taken before the
// dispatch gives the prefix it reached. Poison-filled L1 makes this reliable: an
// unwritten block still holds whatever was there, and a written one almost never
// reproduces it exactly.
static inline uint32_t bingo_blocks_changed(volatile int32_t *d, const int32_t *before,
                                            uint32_t blk_words, uint32_t nblk)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < nblk; i++)
        if (d[i * blk_words + blk_words - 1] != before[i]) n = i + 1;
    return n;
}

// -------------------------------------------------------------
// Every GEMM here computes D = A*B + C. The accumPrevC flag selects WHERE the
// "+C" addend comes from: a C operand in memory, or the value already sitting
// in the VersaCore accumulator register from the previous compute. There are
// three mutually-exclusive C-handling modes (the host datagen asserts exactly
// one is active):
//
//   1. add a real C (accumPrevC=0, C_addr != 0)  -> D = A*B + C_mem
//        C is streamed from L1 using the shape's channel_en_C mask.
//        Internally addNonZeroC=1 is inferred below.
//   2. add zero  (accumPrevC=0, C_addr == 0)      -> D = A*B
//        C is forced to 0: the C reader uses bingo_channel_en_C_null, so the
//        accumulator is seeded with the bare product. addNonZeroC=0.
//   3. accumulate previous (accumPrevC=1)         -> D = A*B + prev_reg
//        NO C is streamed (Ctlbound0=0, channel_en_C_null). The array adds the
//        new product on top of whatever its accumulator register still holds
//        from the prior kernel call. addNonZeroC=0.
//
// How it reaches the hardware:
//   set_versacore_csr(accumPrevC == 0, ...) writes the OVERWRITE_ACCUM CSR =
//   take_in_new_c. In the array (chisel_acc .../versacore/{VersaCore,Accumulator,
//   Array}.scala) this drives accAddExtIn, which is asserted only on the FIRST
//   of the K accumulation steps of each output tile:
//     accAddExtIn = (computeFireCounter==0) && take_in_new_c && busy
//   On that first step the accumulator register is initialized to:
//     take_in_new_c=1 (accumPrevC=0): reg := product[0] + C_external  (fresh: C or 0)
//     take_in_new_c=0 (accumPrevC=1): reg := product[0] + reg_old     (keeps prior sum)
//   Steps 1..K-1 always do reg += product[k] (the normal K reduction). The
//   register is NOT cleared between kernel calls, which is exactly what lets a
//   later accumPrevC=1 call continue a sum left by an earlier call.
//
// CONSTRAINT: accumPrevC=1 requires M == 1 && N == 1. The accumulator register
// holds only ONE output tile (meshRow x meshCol). With M>1 or N>1 the array
// walks to other tiles and overwrites the register, so "previous C" is only
// well-defined for a single tile. (The host datagen asserts M==N==1 here.)
//
// Worked example — on-chip K-split / chained accumulation of one output tile,
// computing  D = A0*B0 + A1*B1 + A2*B2  without ever loading/storing a partial
// C to L1 between passes (M=N=1, same D_addr each call):
//
//   // pass 0: seed the register with A0*B0 (no C, accumPrevC=0)
//   gemm(A0,B0, C=0,    D, M=1,K,N=1, accumPrevC=0); // reg = A0*B0,           D=reg
//   // pass 1: add A1*B1 onto the kept register (accumPrevC=1, C ignored)
//   gemm(A1,B1, C=0/ign,D, M=1,K,N=1, accumPrevC=1); // reg = A0*B0+A1*B1,     D=reg
//   // pass 2: add A2*B2 onto the kept register
//   gemm(A2,B2, C=0/ign,D, M=1,K,N=1, accumPrevC=1); // reg = A0*B0+A1*B1+A2*B2,D=reg
//
// After pass 2, D holds the full sum. Versus mode 1, this saves the C-load +
// C-store memory traffic on every pass by keeping the running sum on-chip.
// (To instead seed pass 0 with a bias term, use accumPrevC=0 with C_addr != 0.)
// -------------------------------------------------------------

// -------------------------------------------------------------
// Shared GEMM core: configure the streamer + VersaCore and run.
// All precision/quant fields are explicit parameters, so the precision-named
// wrappers (and gemm_full) below select them. Emits the shared GEMM_FULL_RUN trace
// markers for every caller, so the sweep/gemm cycle LUT works for all precisions.
// Caller must already have done the core-0 check + SW guard + arg parse.
// -------------------------------------------------------------
static uint32_t __bingo_gemm_run(
    uint32_t A_addr, uint32_t B_addr, uint32_t C_addr, uint32_t D_addr,
    uint32_t M, uint32_t K, uint32_t N, uint32_t array_shape_idx,
    uint32_t transpose_A, uint32_t transpose_B, uint32_t accumPrevC,
    uint32_t quantization_enable, uint32_t shift_i, uint32_t multiplier_i,
    int32_t input_zp_i, int32_t output_zp_i, int32_t int32tofp16_enable,
    int32_t int4_a_enable, int32_t int4_b_enable,
    bingo_kernel_scratchpad_t *sp)
{
    VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Bingo GEMM run A=0x%08x B=0x%08x C=0x%08x D=0x%08x\r\n",
                          snrt_cluster_idx(), snrt_cluster_core_idx(),
                          A_addr, B_addr, C_addr, D_addr);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    // Bounds-check array_shape_idx against the hwcfg, then grab the per-shape
    // parameter block. All shape-dependent values below are read from `shape`.
    if (array_shape_idx >= BINGO_NUM_ARRAY_SHAPES)
    {
        VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Error! array_shape_idx=%d invalid (only %d shapes in hwcfg)\r\n",
                              snrt_cluster_idx(), snrt_cluster_core_idx(),
                              array_shape_idx, BINGO_NUM_ARRAY_SHAPES);
        return BINGO_RET_FAIL;
    }
    const bingo_gemm_shape_params_t *shape = &bingo_gemm_shape_params[array_shape_idx];
    const uint32_t meshRow = shape->meshRow;
    const uint32_t tileSize = shape->tileSize;
    const uint32_t meshCol = shape->meshCol;

    if (quantization_enable && int32tofp16_enable)
    {
        VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Error! quantization and int32tofp16 cannot both be enabled\r\n",
                              snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    // WHICH OUTPUT-STAGE EXTENSIONS THIS CLUSTER ACTUALLY HAS.
    //
    // set_versacore_streamer_csr() writes a FIXED seven user CSRs at
    // READER_WRITER_EXTENSION_1_CSR_BASE -- the layout of a D write host carrying a
    // rescale unit AND the FP16 converter -- and puts the converter's enable in bit 1.
    // A host that carries only the converter owns TWO registers and puts its enable in
    // bit 0, so on such a cluster both facts are wrong: the five extra writes land on the
    // streamer's own registers (base+2 is STREAMER_START_CSR, which launches the streamer
    // mid-configuration) and the converter is never armed.
    //
    // Refuse rather than mis-program. A GEMM that silently writes INT32 where the caller
    // asked for FP16 -- at twice the beats, into a buffer sized for half -- corrupts
    // whatever follows it in L1 and reports success.
    //
    // FlashAttention needs the converter on exactly this cluster and gets it from
    // offload_hw_kernels/gemm_fa.h, which programs the host directly. Extending the
    // generic path means teaching the shared versacore library the window size, which is
    // an upstream change (the same function has the same fixed seven writes there).
#if defined(READER_WRITER_EXTENSION_1_CSR_BASE) && READER_WRITER_EXTENSION_1_CSR_NUM < 7
    // UPDATE: the shared library no longer walks off the end of the window -- it now
    // writes exactly READER_WRITER_EXTENSION_1_CSR_NUM user CSRs and places the converter
    // enable at the bit its own extension list gives it. What a two-CSR port still cannot
    // do is RESCALE: there is no quantisation extension on it to arm, and no zero
    // point/multiplier/shift register to put the caller's values in. Refuse that alone.
    if (quantization_enable)
    {
        printf_safe("[Cluster %d Core %d]: Error! gemm_full cannot quantise on this "
                    "cluster: the D write host owns %d user CSRs and carries only the "
                    "INT32->FP16 converter, with no rescale stage.\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(),
                    READER_WRITER_EXTENSION_1_CSR_NUM);
        return BINGO_RET_FAIL;
    }
#endif

    const uint32_t a_elem_len = int4_a_enable ? 4u : BINGO_A_ELEM_LEN;
    const uint32_t b_elem_len = int4_b_enable ? 4u : BINGO_B_ELEM_LEN;
    const uint32_t d_elem_len = quantization_enable ? 8u
                                                    : (int32tofp16_enable ? 16u : BINGO_D32_ELEM_LEN);
    const uint32_t has_d_extension = quantization_enable || int32tofp16_enable;
    const uint32_t one_output_tile_bits = meshRow * meshCol * d_elem_len;
    uint32_t channel_en_A_dyn[BINGO_A_CSR_NUM] = {0};
    uint32_t channel_en_B_dyn[BINGO_B_CSR_NUM] = {0};
    uint32_t channel_en_D32_dyn[BINGO_D32_CSR_NUM] = {0};

    // dynamically construct channel_en masks
    const uint32_t *channel_en_A_ptr = shape->channel_en_A;
    if (int4_a_enable)
    {
        uint32_t bits = (meshRow * tileSize * a_elem_len + BINGO_BANK_WIDTH - 1) /
                        BINGO_BANK_WIDTH;
        if (bits == 0)
        {
            bits = 1;
        }
        for (uint32_t i = 0; i < bits; i++)
        {
            uint32_t idx = i / 32;
            if (idx < BINGO_A_CSR_NUM)
            {
                // high location of channel_en_A_dyn relates to low channel index
                channel_en_A_dyn[BINGO_A_CSR_NUM - 1 - idx] |=
                    (uint32_t)(1u << (i % 32));
            }
        }
        channel_en_A_ptr = channel_en_A_dyn;
    }

    const uint32_t *channel_en_B_ptr = shape->channel_en_B;
    if (int4_b_enable)
    {
        uint32_t bits = (meshCol * tileSize * b_elem_len + BINGO_BANK_WIDTH - 1) /
                        BINGO_BANK_WIDTH;
        if (bits == 0)
        {
            bits = 1;
        }
        for (uint32_t i = 0; i < bits; i++)
        {
            uint32_t idx = i / 32;
            if (idx < BINGO_B_CSR_NUM)
            {
                channel_en_B_dyn[BINGO_B_CSR_NUM - 1 - idx] |=
                    (uint32_t)(1u << (i % 32));
            }
        }
        channel_en_B_ptr = channel_en_B_dyn;
    }

    const uint32_t *channel_en_D32_ptr = shape->channel_en_D32;
    if (has_d_extension)
    {
        uint32_t bits = (BINGO_SERIAL_C_D_WIDTH + BINGO_BANK_WIDTH - 1) /
                        BINGO_BANK_WIDTH;
        for (uint32_t i = 0; i < bits; i++)
        {
            uint32_t idx = i / 32;
            if (idx < BINGO_D32_CSR_NUM)
            {
                channel_en_D32_dyn[BINGO_D32_CSR_NUM - 1 - idx] |=
                    (uint32_t)(1u << (i % 32));
            }
        }
        channel_en_D32_ptr = channel_en_D32_dyn;
    }

    // some inferenced args
    uint32_t addNonZeroC;
    if (accumPrevC)
    {
        addNonZeroC = 0;
    }
    else if (C_addr != 0)
    {
        addNonZeroC = 1;
    }
    else
    {
        addNonZeroC = 0;
    }
    // Configuration the Steamer and Versacore
    //////////////////////////////////////////////////////////////
    // Streamer cfg for A
    //////////////////////////////////////////////////////////////

    // Aslstride0
    uint32_t Aslstride0 = BINGO_BANK_WIDTH / 8;
    // Atlbound0~5
    uint32_t Atlbound[6];
    // Atlbound0
    Atlbound[0] = K;
    // Atlbound1
    Atlbound[1] = N;
    // Atlbound2
    Atlbound[2] = M;
    // Atlbound3
    Atlbound[3] = 1;
    // Atlbound4
    Atlbound[4] = 1;
    // Atlbound5
    Atlbound[5] = 1;
    uint32_t Atlstride[6];
    // Atlstride0 — A K-tile pitch in bytes. The A reader's sparse interconnect
    // wires read-port i only to banks of parity (i % BINGO_GRANULARITY_A), so
    // the K-tile stride (in banks) must be a multiple of BINGO_GRANULARITY_A or
    // a later K step walks port 0 onto an odd bank (unroutable -> fatal). For
    // int8 A the tile is already an even number of banks for every shape; for
    // int4 A at array_shape 1 the tile is a single bank, so round the pitch UP
    // to BINGO_GRANULARITY_A banks. The datagen pads each int4 A K-tile to the
    // same width (gemm_sim_utils.py), so the pad bytes land in the skipped bank
    // and the VersaCore still consumes the valid tile (channel_en is unchanged).
    uint32_t a_tile_banks =
        (a_elem_len * meshRow * tileSize + BINGO_BANK_WIDTH - 1) / BINGO_BANK_WIDTH;
    if (a_tile_banks == 0)
    {
        a_tile_banks = 1;
    }
    uint32_t a_tile_banks_aligned =
        ((a_tile_banks + BINGO_GRANULARITY_A - 1) / BINGO_GRANULARITY_A) *
        BINGO_GRANULARITY_A;
    Atlstride[0] = a_tile_banks_aligned * (BINGO_BANK_WIDTH / 8);
    // Atlstride1
    Atlstride[1] = 0;
    // Atlstride2 — one full K-run, using the (granularity-aligned) K-tile pitch.
    Atlstride[2] = Atlstride[0] * K;
    // Atlstride3
    Atlstride[3] = 0;
    // Atlstride4
    Atlstride[4] = 0;
    // Atlstride5
    Atlstride[5] = 0;
    uint32_t set_addr_remap_index_A = 0;
    //////////////////////////////////////////////////////////////
    // Streamer cfg for B
    //////////////////////////////////////////////////////////////
    // Bslstride0
    uint32_t Bslstride0 = BINGO_BANK_WIDTH / 8;
    // Btlbound0~2
    uint32_t Btlbound[3];
    // Btlbound0
    Btlbound[0] = K;
    // Btlbound1
    Btlbound[1] = N;
    // Btlbound2
    Btlbound[2] = M;
    uint32_t Btlstride[3];
    // Btlstride0
    Btlstride[0] = b_elem_len * tileSize * meshCol / 8;
    // Btlstride1
    Btlstride[1] = b_elem_len * tileSize * meshCol * K / 8;
    // Btlstride2
    Btlstride[2] = 0;
    uint32_t set_addr_remap_index_B = 0;
    //////////////////////////////////////////////////////////////
    // Streamer cfg for C
    //////////////////////////////////////////////////////////////
    // C spatial strides -- ONE PER DECLARED SPATIAL DIMENSION.
    //
    // The streamer writes S_STRIDE_NUM_READER_WRITER_0 of these whatever the caller
    // passed, so handing it a scalar leaves dimension 1 reading the next thing on this
    // stack frame. The ordinary GEMM layout wants the channels laid end to end, which is
    // the nest stride sl[i] = sl[i-1] * bound[i-1] -- for [4, 8] that is 8 B then 32 B,
    // and the 32 channels cover one contiguous 256 B serialised beat exactly as the old
    // single-dimension [[16]] port did. FlashAttention's interleaved layout needs a
    // different sl1 and has its own kernels; see __snax_bingo_kernel_gemm_fa_*.
#if BINGO_CD_SPATIAL_NUM > 2
#error "extend the C/D spatial stride nest: only 1 or 2 dimensions are derived here"
#endif
    uint32_t Cslstride[BINGO_CD_SPATIAL_NUM];
    Cslstride[0] = BINGO_BANK_WIDTH / 8;
#if BINGO_CD_SPATIAL_NUM > 1
    Cslstride[1] = (BINGO_BANK_WIDTH / 8) * BINGO_CD_SPATIAL_BOUND0;
#endif
    // Ctlbound0~3
    uint32_t Ctlbound[4];
    Ctlbound[0] = (accumPrevC == 1) ? 0 : shape->Ctlbound0;
    // Ctlbound1
    Ctlbound[1] = N;
    // Ctlbound2
    Ctlbound[2] = M;
    // Ctlbound3
    Ctlbound[3] = 1;
    uint32_t Ctlstride[4];
    // Ctlstride0
    Ctlstride[0] = shape->Ctlstride0;
    // Ctlstride1
    Ctlstride[1] = BINGO_C_ELEM_LEN * meshRow *
                   meshCol / 8;
    // Ctlstride2
    Ctlstride[2] = N * BINGO_C_ELEM_LEN *
                   meshRow *
                   meshCol / 8;
    // Ctlstride3
    Ctlstride[3] = 0;
    uint32_t set_addr_remap_index_C = 0;
    // Pick the shape's channel_en_C mask, or the shape-invariant zero mask when
    // accumPrevC or !addNonZeroC.
    const uint32_t *channel_en_C_ptr = (accumPrevC == 1 || addNonZeroC == 0)
                                           ? bingo_channel_en_C_null
                                           : shape->channel_en_C;
    //////////////////////////////////////////////////////////////
    // Streamer cfg for D
    //////////////////////////////////////////////////////////////
    // D32 spatial strides -- same nest as C above (the two halves of one port).
    uint32_t D32slstride[BINGO_CD_SPATIAL_NUM];
    D32slstride[0] = BINGO_BANK_WIDTH / 8;
#if BINGO_CD_SPATIAL_NUM > 1
    D32slstride[1] = (BINGO_BANK_WIDTH / 8) * BINGO_CD_SPATIAL_BOUND0;
#endif
    // D32tlbound0~3
    uint32_t D32tlbound[4];
    uint32_t D32tlstride[4];
    if (has_d_extension)
    {
        if (one_output_tile_bits > BINGO_SERIAL_C_D_WIDTH)
        {
            // one output tile is larger than the streamer width, need to split into multiple stores
            if ((one_output_tile_bits % BINGO_SERIAL_C_D_WIDTH) != 0)
            {
                VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Error! D extension output width is not streamer-aligned\r\n",
                                      snrt_cluster_idx(), snrt_cluster_core_idx());
                return BINGO_RET_FAIL;
            }

            // D32tlbound[0] times store for one tile
            D32tlbound[0] = one_output_tile_bits / BINGO_SERIAL_C_D_WIDTH;
            // in total N * M tile
            D32tlbound[1] = N * M;
            D32tlbound[2] = 1;
            D32tlbound[3] = 1;
            D32tlstride[0] = BINGO_SERIAL_C_D_WIDTH / 8;
            // D32tlbound[1] times 1 tile store
            D32tlstride[1] = d_elem_len * meshRow *
                             meshCol / 8;
            D32tlstride[2] = 0;
            D32tlstride[3] = 0;
        }
        else
        {
            // one output tile fits within the streamer width, will pack multiple tiles into one store
            if ((BINGO_SERIAL_C_D_WIDTH % one_output_tile_bits) != 0)
            {
                VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Error! D extension output width does not divide streamer width\r\n",
                                      snrt_cluster_idx(), snrt_cluster_core_idx());
                return BINGO_RET_FAIL;
            }
            // The D-extension output serializer PACKS output_matrix_per_store
            // narrowed tiles into one BINGO_SERIAL_C_D_WIDTH beat and only emits a
            // beat once it is full, so M*N must fill whole beats. If it does not
            // (e.g. array_shape 1 small GEMMs where one narrowed tile <
            // BINGO_SERIAL_C_D_WIDTH and M*N < output_matrix_per_store) the
            // serializer would stall waiting for tiles that never come (sim hang).
            // Bail cleanly; the sweep workload drops these unsupported configs up
            // front (util/sim/gemm/gemm_psweep_lib.py) so this guard is never hit
            // there. (Padding M/N to fill a beat would change the measured GEMM.)
            if (((M * N * one_output_tile_bits) % BINGO_SERIAL_C_D_WIDTH) != 0)
            {
                VERSACORE_DEBUG_PRINT("[Cluster %d Core %d]: Error! D extension output does not fill streamer stores cleanly\r\n",
                                      snrt_cluster_idx(), snrt_cluster_core_idx());
                return BINGO_RET_FAIL;
            }
            const uint32_t output_matrix_per_store = BINGO_SERIAL_C_D_WIDTH / one_output_tile_bits;
            // D32tlbound0 is 1 (many output tiles are packed into one store)
            D32tlbound[0] = 1;
            // how many times store (each time stores output_matrix_per_store tiles)
            D32tlbound[1] = N * M / output_matrix_per_store;
            D32tlbound[2] = 1;
            D32tlbound[3] = 1;
            // Each D writer step is one full serial C/D beat.
            D32tlstride[0] = BINGO_SERIAL_C_D_WIDTH / 8;
            D32tlstride[1] = BINGO_SERIAL_C_D_WIDTH / 8;
            D32tlstride[2] = 0;
            D32tlstride[3] = 0;
        }
    }
    else
    {
        D32tlbound[0] = shape->D32tlbound0;
        // D32tlbound1
        D32tlbound[1] = N;
        // D32tlbound2
        D32tlbound[2] = M;
        // D32tlbound3
        D32tlbound[3] = 1;
        // D32tlstride0
        D32tlstride[0] = shape->D32tlstride0;
        // D32tlstride1
        // /8 for bit to byte
        D32tlstride[1] = BINGO_D32_ELEM_LEN * meshRow *
                         meshCol / 8;
        // D32tlstride2
        D32tlstride[2] = N * BINGO_D32_ELEM_LEN *
                         meshRow *
                         meshCol / 8;
        // D32tlstride3
        D32tlstride[3] = 0;
    }
    uint32_t set_addr_remap_index_D32 = 0;
    //////////////////////////////////////////////////////////////
    // Configuration the Steamer and Versacore
    //////////////////////////////////////////////////////////////
    VERSACORE_DEBUG_PRINT(
        "Bingo GEMM Full Kernel Compute Streamer Cfg Start!\r\n");
    set_versacore_streamer_csr(
        A_addr,                         // A_addr
        &Aslstride0,                    // Aslstride[] base
        Atlbound,                       // Atlbound[] base
        Atlstride,                      // Atlstride[] base
        set_addr_remap_index_A,         // set_addr_remap_index_A
        transpose_A,                    // transpose_A
        (uint32_t *)channel_en_A_ptr,   // channel_en_A []
        B_addr,                         // B_addr
        &Bslstride0,                    // Bslstride[] base
        Btlbound,                       // Btlbound[] base
        Btlstride,                      // Btlstride[] base
        set_addr_remap_index_B,         // set_addr_remap_index_B
        transpose_B,                    // transpose_B
        (uint32_t *)channel_en_B_ptr,   // channel_en_B []
        C_addr,                         // C_addr
        Cslstride,                      // Cslstride[] base
        Ctlbound,                       // Ctlbound[] base
        Ctlstride,                      // Ctlstride[] base
        set_addr_remap_index_C,         // set_addr_remap_index_C
        (uint32_t *)channel_en_C_ptr,   // channel_en_C []
        D_addr,                         // D_addr
        D32slstride,                    // D32slstride[] base
        D32tlbound,                     // D32tlbound[] base
        D32tlstride,                    // D32tlstride[] base
        set_addr_remap_index_D32,       // set_addr_remap_index_D32
        (uint32_t *)channel_en_D32_ptr, // channel_en_D32 []
        array_shape_idx,
        quantization_enable,
        shift_i,
        multiplier_i,
        input_zp_i,
        output_zp_i,
        int32tofp16_enable,
        int4_a_enable,
        int4_b_enable);

    set_versacore_csr(
        // accPrevC means takes new C
        accumPrevC == 0,
        K,
        N * M,
        0,
        array_shape_idx,
        0);
    VERSACORE_DEBUG_PRINT(
        "Bingo GEMM Full Kernel Streamer Configuration Done!\r\n");
    // Snapshot the last INT32 of every output block BEFORE anything starts, so the
    // timeout diagnostic below can say how far the writer actually got.
    const uint32_t blk_words = meshRow * meshCol;
    const uint32_t nblk = (M * N < 8u) ? M * N : 8u;
    int32_t d_before[8];
    for (uint32_t i = 0; i < nblk; i++)
        d_before[i] = ((volatile int32_t *)(uintptr_t)D_addr)[i * blk_words
                                                              + blk_words - 1];
    // Set CSR to start Streamer
    start_versacore_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    // Poll until Streamer and GEMM accelerator finish.
    //
    // DIAGNOSTIC (2026-09-15): wait_versacore_and_streamer() is an unbounded pair of
    // spin loops, so a dispatch that never retires wedges the core with nothing printed.
    // It also polls busy immediately after the START writes -- with csrw_ss folded to a
    // single csrw those are ~6 cycles apart, the accelerator has not raised busy yet and
    // the VersaCore loop falls straight through, leaving only the streamer loop to spin.
    // Bound it, and on expiry report what each engine thinks it is doing AND whether the
    // writer ever landed anything in D: an untouched D means the array never produced an
    // output block (an A/B/C feed problem), a partly written D means it produced some and
    // the writer's bounds ran out (a count problem). Those are different bugs.
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    {
        // Probe the LAST int32 of every output block. The array retires blocks in order,
        // so the highest block that changed is how far it actually got -- which is the
        // one thing the busy bits cannot tell us. A block is meshRow*meshCol INT32.
        // The snapshot is taken by the caller BEFORE the START writes -- see the
        // d_before[] above start_versacore_and_streamer(). Sampling it here would race
        // the dispatch, which retires in tens of cycles, and six TCDM loads are not
        // faster than that.
        volatile int32_t *dprobe = (volatile int32_t *)(uintptr_t)D_addr;
        csrw_ss(STREAMER_START_CSR, 0);
        csrw_ss(STREAMER_START_CSR, 0);
        for (uint32_t g = 0; g < 64u; g++) {
            if (csrr_ss(VERSACORE_BUSY) || csrr_ss(STREAMER_BUSY_CSR)) break;
        }
        uint32_t spin = 0;
        while (csrr_ss(VERSACORE_BUSY) || csrr_ss(STREAMER_BUSY_CSR)) {
            if (++spin >= 200000u) {
                printf_safe(
                    "[Cluster %d Core %d]: Error! gemm_full timed out (M=%d K=%d N=%d "
                    "accumPrevC=%d addNonZeroC=%d): versacore_busy=%d streamer_busy=%d "
                    "versacore_cc=%d streamer_cc=%d blocks_written=%d/%d\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), (int)M, (int)K, (int)N,
                    (int)accumPrevC, (int)addNonZeroC,
                    (int)csrr_ss(VERSACORE_BUSY), (int)csrr_ss(STREAMER_BUSY_CSR),
                    (int)csrr_ss(VERSACORE_PERFORMANCE_COUNTER),
                    (int)csrr_ss(STREAMER_PERFORMANCE_COUNTER_CSR),
                    (int)bingo_blocks_changed(dprobe, d_before, blk_words, nblk),
                    (int)(M * N));
                break;
            }
        }
        csrw_ss(VERSACORE_START_CSR, 0);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
    VERSACORE_DEBUG_PRINT("Bingo GEMM Full Kernel Compute Done!\r\n");
    sp->return_value = D_addr;
    sp->num_return_values = M * N;
    return BINGO_RET_SUCC;
}

// -------------------------------------------------------------
// gemm_full: low-level entry point exposing every precision/quant flag.
// Thin arg parser over __bingo_gemm_run.
// -------------------------------------------------------------
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_gemm_full(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_gemm_full_args_t);
    if (snrt_cluster_core_idx() != 0)
    {
        printf_safe("[Cluster %d Core %d]: Error! Bingo GEMM full should be called from core 0!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_gemm_full_args_t *a =
        (const __snax_bingo_kernel_gemm_full_args_t *)arg;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_gemm_full_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    return __bingo_gemm_run(
        a->input_A_addr, a->input_B_addr, a->input_C_addr, a->output_D_addr,
        a->M, a->K, a->N, a->array_shape_idx,
        a->transpose_A, a->transpose_B, a->accumPrevC,
        a->quantization_enable, a->shift_i, a->multiplier_i,
        a->input_zp_i, a->output_zp_i, a->int32tofp16_enable,
        a->int4_a_enable, a->int4_b_enable, sp);
}

// -------------------------------------------------------------
// Clean per-precision wrappers (preferred user interface). Each fixes the
// precision flags; the user passes only operands/shape (the quantized one
// also takes requant params). All delegate to __bingo_gemm_run, which emits
// the shared GEMM_FULL_RUN trace markers.
// -------------------------------------------------------------

// Shared prologue for the "plain" (non-requant) wrappers. NOTE: contains early
// returns (SW guard + core-0 check), so it must expand inside the kernel body.
// Declares `a` (args) and `sp` (scratchpad) for the run macro below.
#define BINGO_GEMM_PLAIN_PROLOGUE()                                                            \
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_gemm_args_t);                                 \
    if (snrt_cluster_core_idx() != 0)                                                           \
    {                                                                                          \
        printf_safe("[Cluster %d Core %d]: Error! Bingo GEMM should be called from core 0!\r\n", \
                    snrt_cluster_idx(), snrt_cluster_core_idx());                               \
        return BINGO_RET_FAIL;                                                                  \
    }                                                                                          \
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);                                     \
    const __snax_bingo_kernel_gemm_args_t *a =                                                  \
        (const __snax_bingo_kernel_gemm_args_t *)arg;                                           \
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_gemm_args_t);         \
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END)

// __bingo_gemm_run call for a plain wrapper with the given precision flags
// (no requantization: quant=0, shift/mult/zp=0).
#define BINGO_GEMM_PLAIN_RUN(int4_a, int4_b, int32tofp16)                                      \
    __bingo_gemm_run(a->input_A_addr, a->input_B_addr, a->input_C_addr, a->output_D_addr,       \
                     a->M, a->K, a->N, a->array_shape_idx,                                      \
                     a->transpose_A, a->transpose_B, a->accumPrevC,                             \
                     0, 0, 0, 0, 0, (int32tofp16), (int4_a), (int4_b), sp)

// int8 x int8 -> int32 (baseline).
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_gemm_i8i8_i32(void *arg)
{
    BINGO_GEMM_PLAIN_PROLOGUE();
    return BINGO_GEMM_PLAIN_RUN(/*int4_a*/ 0, /*int4_b*/ 0, /*int32tofp16*/ 0);
}

// int8 x int4 -> int32 (weight operand B packed to 4-bit).
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_gemm_i8i4_i32(void *arg)
{
    BINGO_GEMM_PLAIN_PROLOGUE();
    return BINGO_GEMM_PLAIN_RUN(/*int4_a*/ 0, /*int4_b*/ 1, /*int32tofp16*/ 0);
}

// int4 x int4 -> int32 (both operands packed to 4-bit).
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_gemm_i4i4_i32(void *arg)
{
    BINGO_GEMM_PLAIN_PROLOGUE();
    return BINGO_GEMM_PLAIN_RUN(/*int4_a*/ 1, /*int4_b*/ 1, /*int32tofp16*/ 0);
}

// int8 x int4 -> fp16 (weight B int4; int32 accumulator converted to fp16).
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_gemm_i8i4_f16(void *arg)
{
    BINGO_GEMM_PLAIN_PROLOGUE();
    return BINGO_GEMM_PLAIN_RUN(/*int4_a*/ 0, /*int4_b*/ 1, /*int32tofp16*/ 1);
}

// int8 x int8 -> fp16 (no input packing; int32 accumulator converted to fp16).
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_gemm_i8i8_f16(void *arg)
{
    BINGO_GEMM_PLAIN_PROLOGUE();
    return BINGO_GEMM_PLAIN_RUN(/*int4_a*/ 0, /*int4_b*/ 0, /*int32tofp16*/ 1);
}

// int8 x int8 -> int8 (int32 accumulator requantized to int8). Takes the
// requant params (shift/multiplier/zero-points) in its dedicated args struct.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_gemm_i8i8_i8(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_gemm_quant_args_t);
    if (snrt_cluster_core_idx() != 0)
    {
        printf_safe("[Cluster %d Core %d]: Error! Bingo GEMM should be called from core 0!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_gemm_quant_args_t *a =
        (const __snax_bingo_kernel_gemm_quant_args_t *)arg;
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_gemm_quant_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    return __bingo_gemm_run(
        a->input_A_addr, a->input_B_addr, a->input_C_addr, a->output_D_addr,
        a->M, a->K, a->N, a->array_shape_idx,
        a->transpose_A, a->transpose_B, a->accumPrevC,
        /*quant*/ 1, a->shift_i, a->multiplier_i, a->input_zp_i, a->output_zp_i,
        /*int32tofp16*/ 0, /*int4_a*/ 0, /*int4_b*/ 0, sp);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_gemm_minimal(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_gemm_minimal_args_t);
    // This kernel will only start the versacore and streamer with pre-configured CSRs
    if (snrt_cluster_core_idx() != 0)
    {
        printf_safe("[Cluster %d Core %d]: Error! Bingo GEMM minimal should be called from core 0!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t A_addr = ((uint32_t *)arg)[0];
    uint32_t B_addr = ((uint32_t *)arg)[1];
    uint32_t C_addr = ((uint32_t *)arg)[2];
    uint32_t D_addr = ((uint32_t *)arg)[3];
    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_gemm_minimal_args_t);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_CFG_START);
    set_minimal_streamer_cfg(
        A_addr,
        B_addr,
        C_addr,
        D_addr);
    start_versacore_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_RUN_START);
    wait_versacore_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_RUN_END);
    VERSACORE_DEBUG_PRINT("Bingo GEMM Minimal Kernel Compute Done!\r\n");
    sp->return_value = D_addr;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}
