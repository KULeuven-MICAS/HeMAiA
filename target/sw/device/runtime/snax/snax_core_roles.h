// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Which cluster core owns which engine -- the HeMAiA-side predicates over the map that
// snax_cluster now GENERATES.
//
// The map itself lives in snax_core_roles_defs.h, mirrored from
//   $(SNITCH_ROOT)/target/snitch_cluster/sw/runtime/common/snax-core-roles-defs.h
// by `make snax-sw-gen` (target/sw/Makefile). Upstream derives it from the cluster
// hjson: snax_acc_cfg -> gemm, snax_simd_cfg -> simd, snax_xdma_cfg -> xdma, and the
// `xdma:` boolean (the Snitch DMA ISA, a different key) -> idma. So it tracks $(CFG)
// automatically -- on snax_split_cluster it is GEMM 0 / SIMD 1 / xDMA 2 / iDMA 3, on the
// two-core snax_versacore_to_cluster SIMD, xDMA and iDMA all resolve to hart 1.
//
// Each engine's CSR bank is reachable ONLY from its own hart: csrw_ss addresses the
// accelerator attached to the core executing it, and there is no memory-mapped path to
// any of them. So a kernel that runs on the wrong core does not fault -- it programs a
// different bank at the same CSR offsets and reports success. Use the predicates below
// rather than a core index, and assert the role at kernel entry with
// BINGO_REQUIRE_CORE.
//
// snRuntime's own notion of roles is unchanged: SNRT_CLUSTER_DM_CORE_NUM stays 1, so
// snrt_is_dm_core() still selects exactly the last hart -- the one the generated map
// calls SNAX_CORE_IDMA (upstream rejects any cfg that puts the DMA ISA elsewhere). That
// matters because snrt uses it for singleton work -- the L1/L3 allocator init in
// alloc.h, global-barrier participation in sync.h -- which must happen on one core and
// only one. The remaining harts are ordinary compute cores to snRuntime; these helpers
// are what distinguishes them.
//
// NOTE the spelling change against the old hand-written block: SNAX_CORE_DM is now
// SNAX_CORE_IDMA, because upstream names the ENGINE (idma = the classic Snitch DMA,
// xdma = the SNAX transfer engine) rather than snRuntime's scheduling term, so one core
// can carry both.

#pragma once

#include "snax_core_roles_defs.h"  // generated; mirrored by `make snax-sw-gen`
#include "snrt.h"

// ------------------------------------------------------------------ role predicates
//
// Guarded on SNAX_HAS_<ROLE>_CORE: where the cluster genuinely lacks the engine,
// SNAX_CORE_<ROLE> is not defined at all, so building a SIMD kernel against a cluster
// with no SIMD block must FAIL rather than land the kernel on hart 0.
//
// The absent case expands to an undeclared identifier rather than merely leaving the
// function undeclared. Dropping the declaration alone is only a -Wimplicit-function-
// declaration WARNING on older toolchains, and the device build (toolchain.mk) carries
// no -Werror -- so the kernel would still link and still run on the wrong hart. An
// undeclared identifier is a hard error in every C compiler, and it names the reason.

#if SNAX_HAS_GEMM_CORE
static inline int snax_is_gemm_core(void) {
    return snrt_cluster_core_idx() == SNAX_CORE_GEMM;
}
#else
#define snax_is_gemm_core() SNAX_THIS_CLUSTER_HAS_NO_GEMM_CORE
#endif

#if SNAX_HAS_SIMD_CORE
static inline int snax_is_simd_core(void) {
    return snrt_cluster_core_idx() == SNAX_CORE_SIMD;
}
#else
#define snax_is_simd_core() SNAX_THIS_CLUSTER_HAS_NO_SIMD_CORE
#endif

#if SNAX_HAS_XDMA_CORE
static inline int snax_is_xdma_core(void) {
    return snrt_cluster_core_idx() == SNAX_CORE_XDMA;
}
#else
#define snax_is_xdma_core() SNAX_THIS_CLUSTER_HAS_NO_XDMA_CORE
#endif

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
