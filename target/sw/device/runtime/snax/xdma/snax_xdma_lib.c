// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>

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

extern uint32_t xdma_start();
extern void xdma_local_wait(uint32_t task_id);
extern void xdma_remote_wait(uint32_t task_id);
extern uint32_t xdma_last_task_cycle();
extern uint32_t xdma_last_read_cycle();
extern uint32_t xdma_last_write_cycle();
