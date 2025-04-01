// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>

#pragma once
#include <stdbool.h>
#include "stdint.h"
// Define the CSR address of xdma, should be generated by scala
#include "snax-xdma-csr-addr.h"

// Data Copy Task
int32_t xdma_memcpy_nd_full_addr(
    uint64_t src, uint64_t dst, uint32_t spatial_stride_src,
    uint32_t spatial_stride_dst, uint32_t temp_dim_src,
    uint32_t* temp_stride_src, uint32_t* temp_bound_src, uint32_t temp_dim_dst,
    uint32_t* temp_stride_dst, uint32_t* temp_bound_dst,
    uint32_t enabled_chan_src, uint32_t enabled_chan_dst,
    uint32_t enabled_byte_dst);

int32_t xdma_memcpy_nd(uint8_t* src, uint8_t* dst, uint32_t spatial_stride_src,
                       uint32_t spatial_stride_dst, uint32_t temp_dim_src,
                       uint32_t* temp_stride_src, uint32_t* temp_bound_src,
                       uint32_t temp_dim_dst, uint32_t* temp_stride_dst,
                       uint32_t* temp_bound_dst, uint32_t enabled_chan_src,
                       uint32_t enabled_chan_dst, uint32_t enabled_byte_dst);

int32_t xdma_memcpy_1d_full_addr(uint64_t src, uint64_t dst, uint32_t size);

int32_t xdma_memcpy_1d(uint8_t* src, uint8_t* dst, uint32_t size);

// Multicast Task
int32_t xdma_multicast_nd_full_address(
    uint64_t src, uint64_t* dst, uint32_t dst_num, uint32_t spatial_stride_src,
    uint32_t spatial_stride_dst, uint32_t temp_dim_src,
    uint32_t* temp_stride_src, uint32_t* temp_bound_src, uint32_t temp_dim_dst,
    uint32_t* temp_stride_dst, uint32_t* temp_bound_dst,
    uint32_t enabled_chan_src, uint32_t enabled_chan_dst,
    uint32_t enabled_byte_dst);

int32_t xdma_multicast_nd(uint8_t* src, uint8_t** dst, uint32_t dst_num,
                          uint32_t spatial_stride_src,
                          uint32_t spatial_stride_dst, uint32_t temp_dim_src,
                          uint32_t* temp_stride_src, uint32_t* temp_bound_src,
                          uint32_t temp_dim_dst, uint32_t* temp_stride_dst,
                          uint32_t* temp_bound_dst, uint32_t enabled_chan_src,
                          uint32_t enabled_chan_dst, uint32_t enabled_byte_dst);

int32_t xdma_multicast_1d_full_address(uint64_t src, uint64_t* dst,
                                       uint32_t dst_num, uint32_t size);

int32_t xdma_multicast_1d(uint8_t* src, uint8_t** dst, uint32_t dst_num,
                          uint32_t size);

// Extension
int32_t xdma_enable_src_ext(uint8_t ext, uint32_t* csr_value);
int32_t xdma_disable_src_ext(uint8_t ext);
int32_t xdma_enable_dst_ext(uint8_t ext, uint32_t* csr_value);
int32_t xdma_disable_dst_ext(uint8_t ext);

// Start
uint32_t xdma_start();

// Check if xdma is finished
static inline bool xdma_local_is_finished(uint32_t task_id);
static inline bool xdma_remote_is_finished(uint32_t task_id);

void xdma_local_wait(uint32_t task_id);
void xdma_remote_wait(uint32_t task_id);
