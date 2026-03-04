// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

extern volatile uint64_t *sys_dma_src_ptr(uint8_t chip_id);
extern volatile uint64_t *sys_dma_dst_ptr(uint8_t chip_id);
extern volatile uint64_t *sys_dma_num_bytes_ptr(uint8_t chip_id);
extern volatile uint64_t *sys_dma_conf_ptr(uint8_t chip_id);
extern volatile uint64_t *sys_dma_status_ptr(uint8_t chip_id);
extern volatile uint64_t *sys_dma_nextid_ptr(uint8_t chip_id);
extern volatile uint64_t *sys_dma_done_ptr(uint8_t chip_id);

extern uint64_t sys_dma_memcpy(uint8_t chip_id, uint64_t dst, uint64_t src, uint64_t size);
extern void sys_dma_blk_memcpy(uint8_t chip_id, uint64_t dst, uint64_t src, uint64_t size);
