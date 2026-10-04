// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "idma_reg64_1d.h"
#include "occamy_memory_map.h"
#include "chip_id.h"

// A system iDMA exposes a single stream (NumStreams = 1), so software always uses
// register index 0 for status / next_id / done_id.

#define IDMA_CONF_ADDR \
    (SYS_IDMA_CFG_BASE_ADDR + IDMA_REG64_1D_CONF_REG_OFFSET)
#define IDMA_STATUS_ADDR \
    (SYS_IDMA_CFG_BASE_ADDR + IDMA_REG64_1D_STATUS_0_REG_OFFSET)
#define IDMA_NEXTID_ADDR \
    (SYS_IDMA_CFG_BASE_ADDR + IDMA_REG64_1D_NEXT_ID_0_REG_OFFSET)
#define IDMA_DONE_ADDR \
    (SYS_IDMA_CFG_BASE_ADDR + IDMA_REG64_1D_DONE_ID_0_REG_OFFSET)
#define IDMA_DST_ADDR \
    (SYS_IDMA_CFG_BASE_ADDR + IDMA_REG64_1D_DST_ADDR_LOW_REG_OFFSET)
#define IDMA_SRC_ADDR \
    (SYS_IDMA_CFG_BASE_ADDR + IDMA_REG64_1D_SRC_ADDR_LOW_REG_OFFSET)
#define IDMA_LENGTH_ADDR \
    (SYS_IDMA_CFG_BASE_ADDR + IDMA_REG64_1D_LENGTH_LOW_REG_OFFSET)

// conf register (idma_reg64_1d) bit fields:
//   [0]     decouple_aw   : R-AW coupling. The RTL (idma_channel_coupler.sv) is
//                           authoritative; the idma_pkg.sv comment is inverted.
//                           0 = COUPLED  : the write address (AW) is held back until
//                               the first beat of the matching read arrives, so a
//                               data-starved write can never grab and hold a shared
//                               write mux.
//                           1 = DECOUPLED: AW is issued eagerly, before any read data
//                               returns -- a starved write can then hold the mux.
//   [1]     decouple_rw   : R and W datapaths fully decoupled (can deadlock)
//   [2]     src_reduce_len   [3] dst_reduce_len
//   [6:4]   src_max_llen     [9:7] dst_max_llen
//   [10]    enable_nd     : N-D (2D+) transfer mode
//   [13:11] src_protocol     [16:14] dst_protocol   (0 = AXI)
// All fields 0 => plain AXI->AXI 1D copy with decouple_aw=0 (R-AW COUPLED). Coupling
// avoids the GEMM parallel-DMA deadlock
#define IDMA_CONF_AXI_MEMCPY (0u)

// The 64-bit src/dst/length each occupy two adjacent 32-bit registers
// (_LOW then _HIGH); the returned pointer indexes [0]=low, [1]=high.
inline volatile uint32_t *sys_dma_dst_ptr(uint8_t chip_id) {
    return (volatile uint32_t *)(chiplet_addr_transform_full(chip_id, IDMA_DST_ADDR));
}
inline volatile uint32_t *sys_dma_src_ptr(uint8_t chip_id) {
    return (volatile uint32_t *)(chiplet_addr_transform_full(chip_id, IDMA_SRC_ADDR));
}
inline volatile uint32_t *sys_dma_length_ptr(uint8_t chip_id) {
    return (volatile uint32_t *)(chiplet_addr_transform_full(chip_id, IDMA_LENGTH_ADDR));
}
inline volatile uint32_t *sys_dma_conf_ptr(uint8_t chip_id) {
    return (volatile uint32_t *)(chiplet_addr_transform_full(chip_id, IDMA_CONF_ADDR));
}
inline volatile uint32_t *sys_dma_status_ptr(uint8_t chip_id) {
    return (volatile uint32_t *)(chiplet_addr_transform_full(chip_id, IDMA_STATUS_ADDR));
}
inline volatile uint32_t *sys_dma_nextid_ptr(uint8_t chip_id) {
    return (volatile uint32_t *)(chiplet_addr_transform_full(chip_id, IDMA_NEXTID_ADDR));
}
inline volatile uint32_t *sys_dma_done_ptr(uint8_t chip_id) {
    return (volatile uint32_t *)(chiplet_addr_transform_full(chip_id, IDMA_DONE_ADDR));
}

// A memory chip has several engines (MEM_CHIP_NUM_SYS_IDMA_<k> in occamy.h, up to four),
// each with its own registers SYS_IDMA_CFG_STRIDE apart, its own port into the HBM and its
// own local port on the D2D link: engines pushing to different neighbours run in parallel.
// Engine 0 is at SYS_IDMA_CFG_BASE_ADDR, where a compute chip's one engine is, so the
// functions without an engine argument drive engine 0. Transfer ids count per engine.
#ifndef SYS_IDMA_CFG_STRIDE
#define SYS_IDMA_CFG_STRIDE 0x1000
#endif

static inline volatile uint32_t *sys_dma_engine_reg(uint8_t chip_id, uint32_t engine,
                                                    uint32_t addr) {
    return (volatile uint32_t *)(chiplet_addr_transform_full(
        chip_id, addr + (uint64_t)engine * SYS_IDMA_CFG_STRIDE));
}

// Launch a copy on engine `engine` of chip `chip_id` and return its transfer id.
//
// The launch is a READ (next_id) that follows posted register WRITES, and AXI does not order
// a read after earlier writes. Across the D2D link it really overtakes them: the link answers
// each write at the sender, and its framer arbitrates a waiting AR against waiting AWs
// round-robin, so with the link busy (the previous push still draining) the launch can reach
// the engine before the last register writes do and start with the PREVIOUS transfer's
// values. On the weight prefetcher that turned a 64 B flag push into the preceding 128 KiB
// data length, written from the flag address up over the device code.
//
// So the length's low word is written LAST and read back until it holds the new value:
// writes to one engine land in order, so then every register does. The low word is cleared
// after the launch, so a copy with the same length as the previous one cannot pass the check
// on the stale value. The fence first pushes the writes out ahead of the check's read, which
// then normally succeeds at once: the cost is one register round trip per launch. (A size
// whose low word is 0, i.e. a multiple of 4 GiB, is not supported.)
static inline uint32_t sys_dma_engine_memcpy(uint8_t chip_id, uint32_t engine, uint64_t dst,
                                             uint64_t src, uint64_t size) {
    volatile uint32_t *dst_ptr = sys_dma_engine_reg(chip_id, engine, IDMA_DST_ADDR);
    volatile uint32_t *src_ptr = sys_dma_engine_reg(chip_id, engine, IDMA_SRC_ADDR);
    volatile uint32_t *len_ptr = sys_dma_engine_reg(chip_id, engine, IDMA_LENGTH_ADDR);
    dst_ptr[0] = (uint32_t)dst;
    dst_ptr[1] = (uint32_t)(dst >> 32);
    src_ptr[0] = (uint32_t)src;
    src_ptr[1] = (uint32_t)(src >> 32);
    len_ptr[1] = (uint32_t)(size >> 32);
    *sys_dma_engine_reg(chip_id, engine, IDMA_CONF_ADDR) = IDMA_CONF_AXI_MEMCPY;
    len_ptr[0] = (uint32_t)size;
    asm volatile("fence" ::: "memory");
    while (len_ptr[0] != (uint32_t)size) {
    }
    // Reading next_id launches the transfer and returns its id.
    const uint32_t id = *sys_dma_engine_reg(chip_id, engine, IDMA_NEXTID_ADDR);
    len_ptr[0] = 0;
    return id;
}

// Queued launch, for an engine with a descriptor-queue frontend: a transfer is WRITES alone --
// src, dst, length, then a doorbell -- and the engine runs its queue back to back. No read
// crosses the link, so the issuer waits neither for a round trip nor for a half-duplex link to
// turn around while the engine is still pushing; and writes to one engine land in order, so
// the doorbell always sees this transfer's registers. Two queued transfers run in the order
// queued: a flag queued after its data lands after it. (For now the frontend is a testbench
// model, snax_idma_queue_model in soc_probes.sv; an engine is switched over once, idle, by
// sys_dma_engine_queue_enable.)
#define SYS_DMA_QUEUE_ADDR (SYS_IDMA_CFG_BASE_ADDR + 0x800)

static inline void sys_dma_engine_queue_enable(uint8_t chip_id, uint32_t engine) {
    *sys_dma_engine_reg(chip_id, engine, IDMA_CONF_ADDR) = IDMA_CONF_AXI_MEMCPY;
    *sys_dma_engine_reg(chip_id, engine, SYS_DMA_QUEUE_ADDR + 0x1c) = 1;
}

static inline void sys_dma_engine_queue(uint8_t chip_id, uint32_t engine, uint64_t dst,
                                        uint64_t src, uint64_t size) {
    volatile uint64_t *q =
        (volatile uint64_t *)sys_dma_engine_reg(chip_id, engine, SYS_DMA_QUEUE_ADDR);
    q[0] = src;
    q[1] = dst;
    q[2] = size;
    q[3] = 0;   // the doorbell
}

// The id of the last transfer engine `engine` of chip `chip_id` completed. On a memory
// chip, done means the last write LEFT it (the link answers writes at the sender), not
// that it arrived.
static inline uint32_t sys_dma_engine_done_id(uint8_t chip_id, uint32_t engine) {
    return *sys_dma_engine_reg(chip_id, engine, IDMA_DONE_ADDR);
}

static inline void sys_dma_engine_wait(uint8_t chip_id, uint32_t engine, uint32_t tf_id) {
    while (sys_dma_engine_done_id(chip_id, engine) != tf_id) {
        asm volatile("nop");
    }
}

static inline uint64_t sys_dma_memcpy(uint8_t chip_id, uint64_t dst, uint64_t src, uint64_t size) {
    return sys_dma_engine_memcpy(chip_id, 0, dst, src, size);
}

static inline void sys_dma_blk_memcpy(uint8_t chip_id, uint64_t dst, uint64_t src, uint64_t size) {
    sys_dma_engine_wait(chip_id, 0, sys_dma_engine_memcpy(chip_id, 0, dst, src, size));
}
