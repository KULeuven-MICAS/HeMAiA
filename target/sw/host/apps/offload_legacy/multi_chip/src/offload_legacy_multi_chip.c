// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>

#include "host.h"
#include "hemaia_clk_rst_controller.h"

// Multi-chip offload entry point: the SAME binary runs on every compute chip.
// Each chip brings up its own D2D links and a zero-initialised communication
// buffer (so the device-side chip_barrier checkpoints start clean), then wakes
// ITS OWN Snitch cluster and waits. The device kernel performs the cross-chip
// data movement and synchronisation (e.g. gemm-msplit-1cluster-4chip).
int main() {
    uint32_t current_chip_id = get_current_chip_id();
    init_uart(get_current_chip_baseaddress(), 32, 1);
    // Enable vector extension
    enable_vec();
    printf("[HeMAiA] Multi-chip Offload Legacy Main (chip %x%x)\r\n",
           current_chip_id >> 4, current_chip_id & 0x0F);

    // Bring up the D2D links so cross-chip transfers / barriers work.
    hemaia_d2d_link_initialize_4c1m(current_chip_id);

    // Communication buffer lives at the local narrow SPM, identical address on
    // every chip so the device chip_barrier broadcast lands at the same offset.
    volatile comm_buffer_t* comm_buffer_ptr =
        (comm_buffer_t*)chiplet_addr_transform((uint64_t)&__narrow_spm_start);
    // Zero the whole buffer (incl. chip_level_checkpoint) before the device runs.
    initialize_comm_buffer((comm_buffer_t*)comm_buffer_ptr);

    enable_sw_interrupts();
    comm_buffer_ptr->lock = 0;
    comm_buffer_ptr->chip_id = current_chip_id;
    program_snitches(current_chip_id, comm_buffer_ptr);
    asm volatile("fence" ::: "memory");

    printf("[HeMAiA] Calling snitch cluster on chip %x%x to execute the task\r\n",
           current_chip_id >> 4, current_chip_id & 0x0F);

    // Start the local Snitch cluster
    wakeup_snitches_cl(current_chip_id);

    int ret = wait_snitches_done(current_chip_id);

    printf("[HeMAiA] Snitch cluster done with exit code %d\r\n", ret);

    // Wait for job done and return Snitch exit code
    return ret;
}
