// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Which cluster core owns which engine.
//
//   hart 0  GEMM   the matmul accelerator and its streamer
//   hart 1  SIMD   the stream-operator block: EW0, Map, Reduce, EW1, Fp16ToInt8
//   hart 2  xDMA   transfer engine: Transposer, Memset, the writer junctions, AXI
//   hart 3  DM     the classic iDMA, and the cluster's singleton-init core
//
// Each engine's CSR bank is reachable ONLY from its own hart: csrw_ss addresses the
// accelerator attached to the core executing it, and there is no memory-mapped path to
// any of them. So a kernel that runs on the wrong core does not fault -- it programs a
// different bank at the same CSR offsets and reports success. Use the predicates below
// rather than a core index, and assert the role at kernel entry with
// BINGO_REQUIRE_CORE.
//
// snRuntime's own notion of roles is unchanged: SNRT_CLUSTER_DM_CORE_NUM stays 1, so
// snrt_is_dm_core() still selects exactly hart 3. That matters because snrt uses it for
// singleton work -- the L1/L3 allocator init in alloc.h, global-barrier participation in
// sync.h -- which must happen on one core and only one. Harts 1 and 2 are ordinary
// compute cores to snRuntime; these helpers are what distinguishes them.
//
// TODO: the layout below is HARDCODED to the four-engine cluster. It is a property of
// the cluster hjson -- which core carries snax_acc_cfg, snax_simd_cfg, snax_xdma_cfg --
// and the generator already knows it. The right fix is to emit the assignment into
// occamy.h alongside N_CORES_PER_CLUSTER and have both this header and the bingo
// mini-compiler read it from there, so a cluster with a different core order cannot
// silently disagree with the kernels. Until that lands, a cfg change means editing this
// file.

#pragma once

#include "snrt.h"

// ------------------------------------------------------------------ role predicates

#define SNAX_CORE_GEMM 0
#define SNAX_CORE_SIMD 1
#define SNAX_CORE_XDMA 2
#define SNAX_CORE_DM 3

static inline int snax_is_gemm_core(void) {
    return snrt_cluster_core_idx() == SNAX_CORE_GEMM;
}
static inline int snax_is_simd_core(void) {
    return snrt_cluster_core_idx() == SNAX_CORE_SIMD;
}
static inline int snax_is_xdma_core(void) {
    return snrt_cluster_core_idx() == SNAX_CORE_XDMA;
}

// ------------------------------------------------------------------ role assertion
//
// A BINGO node carries its core placement from the mini-compiler. If the graph puts a
// SIMD kernel on the xDMA core, the kernel would otherwise configure the wrong
// accelerator and report success. This turns that into a loud failure at the first
// dispatch.
//
// Usage, as the first statement of a role-bound kernel body:
//     BINGO_REQUIRE_CORE(snax_is_simd_core(), "simd_stream_reduce", "SIMD");
#define BINGO_REQUIRE_CORE(_pred, _kname, _rolename)                               \
    do {                                                                           \
        if (!(_pred)) {                                                            \
            printf_safe("[Cluster %d Core %d]: Error! " _kname                     \
                        " must run on the " _rolename                              \
                        " core; the DFG placed it here. Kernel not executed.\r\n", \
                        snrt_cluster_idx(), snrt_cluster_core_idx());              \
            return BINGO_RET_FAIL;                                                 \
        }                                                                          \
    } while (0)
