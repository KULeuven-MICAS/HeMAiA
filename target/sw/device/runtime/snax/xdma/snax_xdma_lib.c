// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>

#include <stdint.h>

extern int32_t xdma_memcpy_nd_full_addr(
    uint64_t src, uint64_t dst, uint32_t spatial_stride_src,
    uint32_t spatial_stride_dst, uint32_t temp_dim_src,
    uint32_t* temp_stride_src, uint32_t* temp_bound_src, uint32_t temp_dim_dst,
    uint32_t* temp_stride_dst, uint32_t* temp_bound_dst,
    uint32_t enabled_chan_src, uint32_t enabled_chan_dst,
    uint32_t enabled_byte_dst);

extern int32_t xdma_memcpy_nd(void* src, void* dst, uint32_t spatial_stride_src,
                              uint32_t spatial_stride_dst,
                              uint32_t temp_dim_src, uint32_t* temp_stride_src,
                              uint32_t* temp_bound_src, uint32_t temp_dim_dst,
                              uint32_t* temp_stride_dst,
                              uint32_t* temp_bound_dst,
                              uint32_t enabled_chan_src,
                              uint32_t enabled_chan_dst,
                              uint32_t enabled_byte_dst);

extern int32_t xdma_retask_1d(void* src, void* dst, uint32_t dst_bound0);

extern int32_t xdma_memcpy_1d_full_addr(uint64_t src, uint64_t dst,
                                        uint32_t size);

extern int32_t xdma_memcpy_1d(void* src, void* dst, uint32_t size);

extern int32_t xdma_multicast_nd_full_address(
    uint64_t src, uint64_t* dst, uint32_t dst_num, uint32_t spatial_stride_src,
    uint32_t spatial_stride_dst, uint32_t temp_dim_src,
    uint32_t* temp_stride_src, uint32_t* temp_bound_src, uint32_t temp_dim_dst,
    uint32_t* temp_stride_dst, uint32_t* temp_bound_dst,
    uint32_t enabled_chan_src, uint32_t enabled_chan_dst,
    uint32_t enabled_byte_dst);

extern int32_t xdma_multicast_nd(
    void* src, void** dst, uint32_t dst_num, uint32_t spatial_stride_src,
    uint32_t spatial_stride_dst, uint32_t temp_dim_src,
    uint32_t* temp_stride_src, uint32_t* temp_bound_src, uint32_t temp_dim_dst,
    uint32_t* temp_stride_dst, uint32_t* temp_bound_dst,
    uint32_t enabled_chan_src, uint32_t enabled_chan_dst,
    uint32_t enabled_byte_dst);
extern int32_t xdma_multicast_1d_full_address(uint64_t src, uint64_t* dst,
                                              uint32_t dst_num, uint32_t size);

extern int32_t xdma_multicast_1d(void* src, void** dst, uint32_t dst_num,
                                 uint32_t size);

extern int32_t xdma_enable_src_ext(uint8_t ext, uint32_t* csr_value);
extern int32_t xdma_disable_src_ext(uint8_t ext);
extern int32_t xdma_enable_dst_ext(uint8_t ext, uint32_t* csr_value);
extern int32_t xdma_disable_dst_ext(uint8_t ext);

extern xdma_task_t xdma_start();
extern void xdma_local_wait(uint32_t task_id);
extern void xdma_remote_wait(uint32_t task_id);
extern void xdma_wait_task(xdma_task_t task);
extern void xdma_disable_all_extensions();
extern uint32_t xdma_last_task_cycle();
extern uint32_t xdma_last_read_cycle();
extern uint32_t xdma_last_write_cycle();

// ChainGather / writer-junction entry points. These live behind XDMA_DST_JCT_ENABLE_PTR in
// snax_xdma_lib.h, so instantiate them only when the generated xDMA header actually
// declares a junction region -- a configuration built without junctions has no definition
// to emit and would fail to link here.
//
// Like every other entry point in this file, they are `inline` in the header, which in C99
// emits NO out-of-line copy on its own. Whether that matters depends on whether the
// compiler chooses to inline every call site, so omitting the extern works right up until
// it silently does not: the device apps are built with clang/ld.lld, which left
// xdma_chain_gather_1d_full_address undefined at link time.
#ifdef XDMA_DST_JCT_ENABLE_PTR
extern int32_t xdma_enable_dst_junction(uint8_t jct, uint32_t* csr_value);
extern int32_t xdma_disable_dst_junction(uint8_t jct);
extern int32_t xdma_chain_gather_1d_full_address(uint64_t local_src, uint64_t* chain,
                                                 uint32_t chain_num, uint32_t size,
                                                 uint8_t junction, uint32_t jct_csr0);
#endif  // XDMA_DST_JCT_ENABLE_PTR
