// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Driver for the cluster's SIMD block: hart 1's stream-operator engine.
//
// THIS IS NOT A DMA. The block reads a stream, applies a chain of operators to it, and
// writes a stream. Addresses and strides are the shape of the ITERATION SPACE, not a
// transfer descriptor; a copy is the degenerate case with no operator. Everything
// cross-cluster is absent: no multicast, no remote destination, no chained transfer, no
// AXI. Every pointer is a local TCDM address.
//
// THE OPERATOR CHAIN IS A FIXED LINEAR ORDER, set by the cluster hjson:
//
//     EW0 -> Map -> Reduce -> EW1 -> Fp16ToInt8
//
// so what can be fused into ONE pass over memory is a property of the HARDWARE, not of
// this library. A combine that must happen BEFORE the pointwise transform uses EW0; one
// that must happen after uses EW1. The generated header names the two instances
// SIMD_EXT_STREAMELEMENTWISE_0 / _1 -- there is no bare name, and picking the wrong one
// is silent.
//
// Addresses are absolute and arrive from the caller: under BINGO the L1 arena belongs to
// the allocator, not to the kernel, so nothing here derives a base from snrt_l1_next().

#pragma once

#include <stdbool.h>

#include "snrt.h"
#include "stdint.h"

// CSR map, generated from the active cluster cfg and picked up off the
// `sw/snax/*/include` wildcard in device/apps/common.mk -- the same route the versacore
// lib takes for streamer_csr_addr_map.h.
#include "snax-simd-addr.h"

// Base of this hart's SNAX CSR window. Every engine uses the same number: csrw_ss
// addresses the accelerator attached to THIS hart, and each block sits on a hart of its
// own, so there is no conflict -- and no way to reach another engine's bank.
#define SNAX_SIMD_CFG_ADDR 960

// Bytes moved per beat, and bytes per lane within a beat.
#define SIMD_BEAT_BYTES SIMD_WIDTH
#define SIMD_LANE_BYTES (SIMD_WIDTH / SIMD_SPATIAL_CHAN)

// Deepest nested sweep the hardware supports, and the widest operator CSR block.
#define SIMD_MAX_DIM SIMD_SRC_TEMP_DIM
#define SIMD_MAX_OP_CSR 4

#ifdef SIMD_DEBUG
#define SIMD_DEBUG_PRINT(...) printf_safe(__VA_ARGS__)
#else
#define SIMD_DEBUG_PRINT(...)
#endif

// always_inline so a compile-time-constant `addr` propagates into the csrr_ss/csrw_ss
// switch and constant-folds to a single direct `csrr/csrw <imm>`. Without it every CSR
// access pays a jump-table load from L2 plus an indirect jump -- measured to be the
// dominant cost of accelerator configuration on this core.
__attribute__((always_inline)) static inline uint32_t snax_read_simd_cfg_reg(
    uint32_t addr) {
    return csrr_ss(SNAX_SIMD_CFG_ADDR + addr);
}
__attribute__((always_inline)) static inline void snax_write_simd_cfg_reg(
    uint32_t addr, uint32_t value) {
    csrw_ss(SNAX_SIMD_CFG_ADDR + addr, value);
}

// ============================================================== shapes

// The iteration space of one side of a task.
//
// `bound`/`stride` are the nested loops, innermost first, in BEATS and BYTES
// respectively. `lane_stride` is the distance between adjacent lanes inside one beat --
// SIMD_LANE_BYTES for contiguous data. `lane_mask` selects participating lanes;
// `byte_mask` (output only) selects which bytes of each word are written.
typedef struct {
    void *base;  // must be SIMD_BEAT_BYTES aligned
    uint32_t lane_stride;
    uint32_t lane_mask;
    uint32_t byte_mask;
    uint32_t dim;
    uint32_t bound[SIMD_MAX_DIM];
    uint32_t stride[SIMD_MAX_DIM];
} snax_simd_shape_t;

// NOTE: no `= {0}` anywhere below. On a struct this size the compiler lowers it to a
// memset() call, and this is a freestanding runtime with no libc -- it links as
// `undefined symbol: memset`. Every field is set explicitly.
static inline void snax_simd_shape_clear(snax_simd_shape_t *s) {
    s->base = 0;
    s->lane_stride = 0;
    s->lane_mask = 0;
    s->byte_mask = 0;
    s->dim = 0;
    // bound 1 / stride 0 is the NEUTRAL loop -- one iteration, no address advance -- so
    // an unused dimension can be written to the hardware unconditionally. Zero would
    // make the whole sweep empty, and the block DROPS an empty task rather than
    // hanging, which shows up only as the sticky bad-config status bit.
    for (uint32_t i = 0; i < SIMD_MAX_DIM; i++) {
        s->bound[i] = 1;
        s->stride[i] = 0;
    }
}

// `beats` consecutive beats from `base`.
static inline void snax_simd_shape_flat(snax_simd_shape_t *s, void *base,
                                        uint32_t beats) {
    snax_simd_shape_clear(s);
    s->base = base;
    s->lane_stride = SIMD_LANE_BYTES;
    s->lane_mask = 0xFFFFFFFFu;
    s->byte_mask = 0xFFFFFFFFu;
    s->dim = 1;
    s->bound[0] = beats;
    s->stride[0] = SIMD_BEAT_BYTES;
}

// `rows` rows of `beats_per_row` beats each, rows `row_stride` bytes apart. This is the
// shape a row-wise reduction sweeps, and the one a TAP-padded tensor needs (row_stride
// = (beats+1)*SIMD_BEAT_BYTES skips the appended scalar beat).
static inline void snax_simd_shape_rows(snax_simd_shape_t *s, void *base,
                                        uint32_t rows, uint32_t beats_per_row,
                                        uint32_t row_stride) {
    snax_simd_shape_clear(s);
    s->base = base;
    s->lane_stride = SIMD_LANE_BYTES;
    s->lane_mask = 0xFFFFFFFFu;
    s->byte_mask = 0xFFFFFFFFu;
    s->dim = 2;
    s->bound[0] = beats_per_row;
    s->stride[0] = SIMD_BEAT_BYTES;
    s->bound[1] = rows;
    s->stride[1] = row_stride;
}

// A 2-D sweep with strides given outright, innermost first, in BYTES. The two patterns
// the rows/flat helpers cannot express both live here:
//   stride0 == 0    re-present one beat bound0 times (broadcast)
//   bound0 == 2, stride0 == d
//                   interleave two operand streams d apart, which is how a 2-operand
//                   StreamElementwise is fed
// `d` must be POSITIVE: the AGU stride is unsigned, so a second operand placed BELOW
// the first wraps the address, reads outside TCDM, and writes X while the task still
// reports complete. Callers that cannot control operand order must swap and rely on the
// op being commutative -- see snax_simd_ew2_base() below.
static inline void snax_simd_shape_2d(snax_simd_shape_t *s, void *base,
                                      uint32_t bound0, uint32_t stride0,
                                      uint32_t bound1, uint32_t stride1) {
    snax_simd_shape_clear(s);
    s->base = base;
    s->lane_stride = SIMD_LANE_BYTES;
    s->lane_mask = 0xFFFFFFFFu;
    s->byte_mask = 0xFFFFFFFFu;
    s->dim = 2;
    s->bound[0] = bound0;
    s->stride[0] = stride0;
    s->bound[1] = bound1;
    s->stride[1] = stride1;
}

// One beat, presented `beats` times. Stride 0 drives the reader's repeat path, which is
// how a per-row scalar is broadcast against a full row without materialising it.
static inline void snax_simd_shape_broadcast(snax_simd_shape_t *s, void *base,
                                             uint32_t beats) {
    snax_simd_shape_flat(s, base, beats);
    s->stride[0] = 0;
}

// Total beats a shape sweeps (the product of its bounds).
static inline uint32_t snax_simd_shape_beats(const snax_simd_shape_t *s) {
    uint32_t n = 1;
    for (uint32_t i = 0; i < s->dim; i++) n *= s->bound[i];
    return n;
}

// ============================================================== operator modes

// Operator ids come from the generated header as SIMD_EXT_<NAME>; which ones exist
// depends on the cluster cfg. The CSR layouts below mirror the Chisel:
//   StreamMap          csr0 = a (FP32 bits), csr1 = b (FP32 bits), csr2[1:0] = func
//   StreamReduce       csr0 = operandCount,  csr1 = mode
//   StreamElementwise  csr0 = operandCount,  csr1 = mode
//   Fp16ToInt8         csr0 = inv_scale (FP32 bits)

// StreamMap: out = func(a * x + b), elementwise.
#define SIMD_FUNC_LINEAR 0u
#define SIMD_FUNC_EXP 1u
#define SIMD_FUNC_SILU 2u
// out = 1/sqrt(a*x + b), for the per-row normalisations. `a` is where the division by the
// row length goes: a reduce emits SUM(x^2) and rmsnorm wants 1/sqrt(SUM/D), so a = 1/D and
// b = 0 -- and rather than costing a pass, this REPLACES the identity multiply on the
// broadcast pass a per-row scalar already needs. A non-positive or non-finite input gives
// 0, so an all-zero row normalises to zero instead of poisoning itself with +Inf. Accurate
// to 1 FP16 ULP, which is better than the core's integer sqrt+reciprocal.
//
// UNLIKE EVERY OTHER CONSTANT HERE, THIS ONE IS NOT ALWAYS BUILT. The generated
// snax-simd-addr.h names each EXTENSION (SIMD_EXT_STREAMMAP) but nothing about which
// `func` values that extension elaborated -- SimdTopGen emits one macro per extension and
// none per func. So a cfg whose HasStreamMap.func list omits RSQRT_FP16 still defines
// SIMD_EXT_STREAMMAP, still accepts func = 3, and silently computes the LINEAR result.
// Kernels gate on BINGO_SIMD_HAS_RSQRT (offload_hw_kernels/simd.h) rather than on the
// presence of this define.
#define SIMD_FUNC_RSQRT 3u

// StreamReduce: fold a row of `operand_beats` down to one scalar beat.
#define SIMD_RED_MAX 0u
#define SIMD_RED_ADD 1u
#define SIMD_RED_SUMSQ 2u
// Pass the row through AND emit the scalar as a trailing beat, so one task yields both a
// transformed row and its reduction: N beats in, N+1 out.
#define SIMD_RED_TAP 0x100u
// Emit the scalar as FP32 rather than narrowing to the transport grid -- needed when the
// sum would overflow FP16 (the narrow WRAPS to garbage, it does not saturate to inf).
#define SIMD_RED_FP32OUT 0x200u
// Emit the per-lane partials as one beat -- the reduction ACROSS beats -- instead of
// folding them to a scalar.
//
// The fold, its treeBuf serialisation and the scalar drain all exist to turn the lane
// partials into ONE number, and they are what makes a short row expensive: a 2-beat row
// costs ~40 cycles on the folding path, nearly all of it bubble. The partials are
// already the across-beat reduction, computed for free in the accumulator as the row
// streams past. Orient the data so the axis you want to reduce runs along BEATS and this
// returns exactly what you want, with no fold at all.
#define SIMD_RED_LANEWISE 0x400u

// StreamElementwise: combine `operand_beats` interleaved operands into one.
// Fp16ToInt8 csr(1): pass the row's LAST beat through unquantised, so a TAP pass can
// narrow its tile and keep its scalar. 0 disables it and is bit-identical to a build
// without the feature. MUST be a multiple of 2 -- the pack ratio -- so the pack the tail
// interrupts is always complete.
//
// STICKY: snax_simd_use1() writes csr(0) only. Any task that arms Fp16ToInt8 without
// naming a tail inherits the previous task's. Use snax_simd_use2() and say 0.
#define SIMD_QUANT_TAIL(beats) ((uint32_t)(beats))

#define SIMD_EW_MUL 0u
#define SIMD_EW_ADD 1u
// Latch the FIRST beat of the task as operand B and combine every later beat against it,
// instead of taking both operands from the stream. Use with operand_beats = 1, so each
// beat is its own row. Without it a broadcast operand has to be physically replicated
// once per data beat -- a whole extra pass to write it and a doubled read stream to
// consume it -- and the AGU cannot avoid that, because one affine address stream cannot
// hold one operand's address fixed while the other walks.
#define SIMD_EW_STICKY_B 0x100u

// FP32 bit patterns. NO `float` ANYWHERE IN THIS API, DELIBERATELY: the cores that drive
// the SIMD block are rv32ima -- integer only, no FPU -- so a `float` parameter compiles
// to an FP instruction and traps with an illegal instruction the moment the kernel runs.
#define SIMD_F32_ZERO 0x00000000u
#define SIMD_F32_ONE 0x3F800000u
#define SIMD_F32_NEG_ONE 0xBF800000u

// Flip the sign of an FP32 bit pattern -- integer-only, as the kernels do it.
static inline uint32_t snax_simd_f32_neg(uint32_t bits) {
    return bits ^ 0x80000000u;
}

// f16_to_f32bits lives in snax/snax_fp16_math.h and is shared, so it is not duplicated
// here; include that header when a per-row FP16 scalar has to become an operator
// coefficient.

// ============================================================== operator arming

typedef struct {
    uint8_t id;
    uint8_t csr_num;
    uint32_t csr[SIMD_MAX_OP_CSR];
} snax_simd_op_t;

// Arm an operator by id with its raw CSR words. Walks SIMD_EXT_CUSTOM_CSR_NUM at run
// time, so every write it makes has a COMPUTED address and pays the jump-table dispatch.
// Correct, general, and the slow path -- prefer the snax_simd_use* macros in a loop, and
// keep this for bring-up and for kernels whose operator id is a variable.
static inline int32_t snax_simd_enable_ext(uint8_t ext, uint32_t *csr_value) {
#if SIMD_EXT_NUM == 0
    (void)ext;
    (void)csr_value;
    return -1;
#else
    if (ext >= SIMD_EXT_NUM) return -1;
    uint8_t custom_csr_list[SIMD_EXT_NUM] = SIMD_EXT_CUSTOM_CSR_NUM;
    uint32_t csr_offset = SIMD_EXT_CSR_PTR;
    for (uint8_t i = 0; i < ext; i++) csr_offset += custom_csr_list[i];

    snax_write_simd_cfg_reg(
        SIMD_EXT_ENABLE_PTR,
        snax_read_simd_cfg_reg(SIMD_EXT_ENABLE_PTR) | (1u << ext));
    for (uint8_t i = 0; i < custom_csr_list[ext]; i++) {
        snax_write_simd_cfg_reg(csr_offset + i, csr_value[i]);
    }
    return 0;
#endif
}

static inline int32_t snax_simd_disable_ext(uint8_t ext) {
#if SIMD_EXT_NUM == 0
    (void)ext;
    return 0;
#else
    if (ext >= SIMD_EXT_NUM) return 0;
    snax_write_simd_cfg_reg(
        SIMD_EXT_ENABLE_PTR,
        snax_read_simd_cfg_reg(SIMD_EXT_ENABLE_PTR) & ~(1u << ext));
    return 0;
#endif
}

static inline void snax_simd_disable_all_ext(void) {
#if SIMD_EXT_NUM != 0
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, 0);
#endif
}

// Arm exactly one operator and disable every other, in a SINGLE csrw.
//
// The arm* family below read-modify-writes the enable mask, and a read is 5.1 cycles
// against a write's 1.0 (measured: a csrw is posted -- the ReqRspManager answers
// combinationally -- while a csrr stalls the core until writeback). Every call site that
// wants one operator can compute the mask as `1 << id` with no read at all, which is
// what these do. `base` must be a compile-time constant: pass the generated
// SIMD_EXT_<NAME>_CSR, never a computed value, or the whole point is lost.
#define snax_simd_use0(id) \
    snax_write_simd_cfg_reg(SIMD_EXT_ENABLE_PTR, 1u << (id))
#define snax_simd_use1(id, base, c0)               \
    do {                                           \
        snax_simd_use0(id);                        \
        snax_write_simd_cfg_reg((base) + 0, (c0)); \
    } while (0)
#define snax_simd_use2(id, base, c0, c1)           \
    do {                                           \
        snax_simd_use1(id, base, c0);              \
        snax_write_simd_cfg_reg((base) + 1, (c1)); \
    } while (0)
#define snax_simd_use3(id, base, c0, c1, c2)       \
    do {                                           \
        snax_simd_use2(id, base, c0, c1);          \
        snax_write_simd_cfg_reg((base) + 2, (c2)); \
    } while (0)

// Same, but ADDING to whatever is already armed -- for chaining two or more operators
// into one pass. Pays the read once per operator; a kernel that arms a fixed chain every
// iteration should write the mask itself with one csrw and then set the CSRs.
#define snax_simd_arm0(id)                                                     \
    snax_write_simd_cfg_reg(                                                   \
        SIMD_EXT_ENABLE_PTR,                                                   \
        snax_read_simd_cfg_reg(SIMD_EXT_ENABLE_PTR) | (1u << (id)))
#define snax_simd_arm1(id, base, c0)               \
    do {                                           \
        snax_simd_arm0(id);                        \
        snax_write_simd_cfg_reg((base) + 0, (c0)); \
    } while (0)
#define snax_simd_arm2(id, base, c0, c1)           \
    do {                                           \
        snax_simd_arm1(id, base, c0);              \
        snax_write_simd_cfg_reg((base) + 1, (c1)); \
    } while (0)
#define snax_simd_arm3(id, base, c0, c1, c2)       \
    do {                                           \
        snax_simd_arm2(id, base, c0, c1);          \
        snax_write_simd_cfg_reg((base) + 2, (c2)); \
    } while (0)

// Write one operator CSR at a CONSTANT address -- the cheap way to change a coefficient
// between passes: one `csrw imm` rather than a jump-table dispatch.
#define snax_simd_set_op_csr(base, idx, val) \
    snax_write_simd_cfg_reg((base) + (idx), (val))

// ============================================================== task programming

// EVERY CSR ADDRESS BELOW IS A COMPILE-TIME CONSTANT, DELIBERATELY.
//
// csrw_ss is a switch over the CSR number, because the RISC-V csrw instruction takes an
// IMMEDIATE. When the compiler can fold the address it emits one `csrw imm`; when it
// cannot, the write becomes a jump-table load out of L2 plus an indirect jump -- roughly
// 20 cycles instead of 1. At ~30 geometry CSRs per task that is the difference between a
// few tens of cycles and several hundred, which on a short task is the whole task.
#if SIMD_SRC_TEMP_DIM > 6 || SIMD_DST_TEMP_DIM > 6
#error "snax_simd_lib: extend the unrolled CSR writes for this AGU depth"
#endif

// The full program: geometry only, no operators, no validation, no branches.
//
// It trusts the caller completely: every dimension is written unconditionally, so the
// shape's unused loops must hold the neutral bound=1/stride=0 that
// snax_simd_shape_clear() installs. Every snax_simd_shape_* constructor does.
static inline void snax_simd_program_fast(const snax_simd_shape_t *in,
                                          const snax_simd_shape_t *out) {
    snax_write_simd_cfg_reg(SIMD_SRC_ADDR_PTR_LSB, (uint32_t)(uintptr_t)in->base);
    snax_write_simd_cfg_reg(SIMD_SRC_ADDR_PTR_MSB, 0);
    snax_write_simd_cfg_reg(SIMD_SRC_SPATIAL_STRIDE_PTR, in->lane_stride);
#if SIMD_SRC_TEMP_DIM > 0
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 0, in->bound[0]);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 0, in->stride[0]);
#endif
#if SIMD_SRC_TEMP_DIM > 1
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 1, in->bound[1]);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 1, in->stride[1]);
#endif
#if SIMD_SRC_TEMP_DIM > 2
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 2, in->bound[2]);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 2, in->stride[2]);
#endif
#if SIMD_SRC_TEMP_DIM > 3
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 3, in->bound[3]);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 3, in->stride[3]);
#endif
#if SIMD_SRC_TEMP_DIM > 4
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 4, in->bound[4]);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 4, in->stride[4]);
#endif
#if SIMD_SRC_TEMP_DIM > 5
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 5, in->bound[5]);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 5, in->stride[5]);
#endif
    snax_write_simd_cfg_reg(SIMD_SRC_ENABLED_CHAN_PTR, in->lane_mask);
    snax_write_simd_cfg_reg(SIMD_DST_ADDR_PTR_LSB, (uint32_t)(uintptr_t)out->base);
    snax_write_simd_cfg_reg(SIMD_DST_ADDR_PTR_MSB, 0);
    snax_write_simd_cfg_reg(SIMD_DST_SPATIAL_STRIDE_PTR, out->lane_stride);
#if SIMD_DST_TEMP_DIM > 0
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 0, out->bound[0]);
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 0, out->stride[0]);
#endif
#if SIMD_DST_TEMP_DIM > 1
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 1, out->bound[1]);
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 1, out->stride[1]);
#endif
#if SIMD_DST_TEMP_DIM > 2
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 2, out->bound[2]);
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 2, out->stride[2]);
#endif
#if SIMD_DST_TEMP_DIM > 3
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 3, out->bound[3]);
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 3, out->stride[3]);
#endif
#if SIMD_DST_TEMP_DIM > 4
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 4, out->bound[4]);
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 4, out->stride[4]);
#endif
#if SIMD_DST_TEMP_DIM > 5
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 5, out->bound[5]);
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 5, out->stride[5]);
#endif
    snax_write_simd_cfg_reg(SIMD_DST_ENABLED_CHAN_PTR, out->lane_mask);
    snax_write_simd_cfg_reg(SIMD_DST_ENABLED_BYTE_PTR, out->byte_mask);
}

// The 1-D steady state: only the six CSRs a 1-D task actually varies. Address high word,
// spatial stride, channel/byte masks and temporal dims 1+ are left exactly as an earlier
// snax_simd_program_fast() set them -- so the caller MUST have run one, and every shape
// since must have kept bound[1] == 1. 6 CSR writes instead of 21.
static inline void snax_simd_program_1d(const snax_simd_shape_t *in,
                                        const snax_simd_shape_t *out) {
    snax_write_simd_cfg_reg(SIMD_SRC_ADDR_PTR_LSB, (uint32_t)(uintptr_t)in->base);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_BOUND_PTR + 0, in->bound[0]);
    snax_write_simd_cfg_reg(SIMD_SRC_TEMP_STRIDE_PTR + 0, in->stride[0]);
    snax_write_simd_cfg_reg(SIMD_DST_ADDR_PTR_LSB, (uint32_t)(uintptr_t)out->base);
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_BOUND_PTR + 0, out->bound[0]);
    snax_write_simd_cfg_reg(SIMD_DST_TEMP_STRIDE_PTR + 0, out->stride[0]);
}

// ============================================================== launch and wait

// Submit the staged task and return immediately -- no counter read at all.
//
// This is what lets the 2-entry taskQueue do its job. SimdTop snapshots the WHOLE task
// (both AGU configs, the extension enable mask and the operator CSRs) into the queue on
// the start pulse, so the core can arm and program task i+1 while task i is still
// running, and the CSR write itself back-pressures when the queue is full -- the core
// stalls on exactly the right cycle without polling.
static inline void snax_simd_fire(void) {
    snax_write_simd_cfg_reg(SIMD_START_PTR, 1);
}

// Launch and confirm the block took the configuration. Costs a blocking read per call;
// use snax_simd_fire() in a batch and retire the batch with snax_simd_wait_all().
static inline uint32_t snax_simd_launch(void) {
    uint32_t submitted = snax_read_simd_cfg_reg(SIMD_SUBMITTED_TASK_PTR);
    snax_write_simd_cfg_reg(SIMD_START_PTR, 1);
    while (snax_read_simd_cfg_reg(SIMD_SUBMITTED_TASK_PTR) == submitted) {
        // The task queue was full; the write is held until it drains.
    }
    return snax_read_simd_cfg_reg(SIMD_SUBMITTED_TASK_PTR);
}

static inline uint32_t snax_simd_submitted(void) {
    return snax_read_simd_cfg_reg(SIMD_SUBMITTED_TASK_PTR);
}

static inline uint32_t snax_simd_finished(void) {
    return snax_read_simd_cfg_reg(SIMD_FINISHED_TASK_PTR);
}

// Block until the task with this id has retired.
static inline void snax_simd_wait(uint32_t task_id) {
    while (snax_read_simd_cfg_reg(SIMD_FINISHED_TASK_PTR) < task_id) {
    }
}

// Block until every submitted task has retired.
static inline void snax_simd_wait_all(void) {
    snax_simd_wait(snax_read_simd_cfg_reg(SIMD_SUBMITTED_TASK_PTR));
}

// Bounded drain. A hand-synchronised engine CAN wedge, and an unbounded poll in a
// Verilator or VCS run burns wall-clock with no diagnosis. Returns 0 on a clean drain,
// non-zero on timeout -- every BINGO kernel below turns that into BINGO_RET_FAIL.
#ifndef SNAX_SIMD_DRAIN_SPINS
#define SNAX_SIMD_DRAIN_SPINS 200000u
#endif
static inline uint32_t snax_simd_wait_all_bounded(void) {
    uint32_t want = snax_read_simd_cfg_reg(SIMD_SUBMITTED_TASK_PTR);
    uint32_t spins = 0;
    while (snax_read_simd_cfg_reg(SIMD_FINISHED_TASK_PTR) < want) {
        if (++spins > SNAX_SIMD_DRAIN_SPINS) return 1u;
    }
    return 0u;
}

// ============================================================== status

static inline uint32_t snax_simd_last_task_cycle(void) {
    return snax_read_simd_cfg_reg(SIMD_PERF_CTR_TASK);
}
static inline uint32_t snax_simd_last_read_cycle(void) {
    return snax_read_simd_cfg_reg(SIMD_PERF_CTR_READER);
}
static inline uint32_t snax_simd_last_write_cycle(void) {
    return snax_read_simd_cfg_reg(SIMD_PERF_CTR_WRITER);
}

// Cumulative engine-busy cycles since reset, free-running. Use THIS, not
// snax_simd_last_task_cycle(), to measure a MULTI-task sequence: the per-task counters
// restart on every start pulse, so reading them forces a wait after each task and
// destroys the overlap the task queue exists to provide.
static inline uint32_t snax_simd_busy_cycles(void) {
    return snax_read_simd_cfg_reg(SIMD_BUSY_CYCLES);
}

static inline bool snax_simd_busy(void) {
    return (snax_read_simd_cfg_reg(SIMD_STATUS) & 0x1) != 0;
}

// Sticky since the last launch: a task was started whose AGU never became busy, i.e. the
// geometry was degenerate. The block DROPS such a task rather than hanging, so this bit
// is the only evidence it happened -- check it after a task that produced nothing.
static inline bool snax_simd_bad_config(void) {
    return (snax_read_simd_cfg_reg(SIMD_STATUS) & 0x2) != 0;
}

// ============================================================== two-operand helper

// The reader AGU strides FORWARD only (the temporal-stride CSR is zero-extended to the
// 48-bit address, so a "negative" stride wraps, reads outside TCDM and STALLS the task
// forever -- a hang with no error). A 2-operand interleave must therefore be based at
// the LOWER of the two operands.
//
// This returns the base and the positive stride, swapping if needed. Swapping is only
// valid because the StreamElementwise ops in use (MUL, ADD) are COMMUTATIVE: op(a,b) ==
// op(b,a) per beat and the output beat order is unchanged. A non-commutative op would
// need real per-operand AGU bases.
//
// (The order-preserving fix is a one-line HW change: SIGN-extend the temporal-stride CSR
// to the full address width instead of zero-extending it. A negative stride would then
// carry the right two's-complement bits and wrap onto the lower operand WITHOUT
// reordering, so any op and any layout would work. Once that lands, drop the swap and
// pass src_b - src_a unconditionally.)
static inline void *snax_simd_ew2_base(void *src_a, void *src_b,
                                       uint32_t *operand_stride) {
    uint32_t a_lo = (uint32_t)(uintptr_t)src_a;
    uint32_t b_lo = (uint32_t)(uintptr_t)src_b;
    if (b_lo >= a_lo) {
        *operand_stride = b_lo - a_lo;
        return src_a;
    }
    *operand_stride = a_lo - b_lo;
    return src_b;
}
