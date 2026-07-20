// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Fanchen Kong <fanchen.kong@kuleuven.be>

// 4-chiplet DVFS validation app.
//
// Same SPMD binary runs on all four compute chiplets ((0,0)/(0,1)/(1,0)/(1,1)),
// differentiated at runtime by get_current_chip_id(). Unlike the offload_bingo_hw
// bring-up test (which armed DVFS only on chip(0,0)), here EVERY chiplet arms its
// own DVFS host-notify path, so each chiplet independently:
//   - programs its PM into DVFS mode + installs the trap handler (dvfs_init),
//   - runs the DVFS-stress DAG (kernel_execution) whose cluster cores cycle
//     busy<->idle several times,
//   - services each idle<->busy doorbell in dvfs_trap_handler (records silently),
//   - narrates the serviced RAISE/LOWER sequence via dvfs_dump_log().
// dvfs_init targets get_current_chip_baseaddress()'s CLINT, so arming on every
// chiplet is per-chip correct (each rings ITS OWN host).

#include "offload_bingo_hw.h"
// DVFS runtime (trap handler + dvfs_init). Included here (not in host.h) so only
// this test links the DVFS handler + its printf footprint; other apps are unaffected.
#include "dvfs.h"

int main() {
    uint8_t current_chip_id = get_current_chip_id();
    uint8_t x = get_current_chip_loc_x();
    uint8_t y = get_current_chip_loc_y();
    // Program the Chiplet Topology
    hemaia_d2d_link_initialize_4c1m(current_chip_id);
    // Init the uart for printf
    init_uart(get_current_chip_baseaddress(), 32, 1);

    ///////////////////////////////
    // 1. Init the Allocator
    ///////////////////////////////
    if (bingo_hemaia_system_mmap_init() < 0) {
        printf("[dvfs][chip(%x,%x)] Error initializing allocator\n", x, y);
        return -1;
    }

    ///////////////////////////////
    // 2. Wake up all the clusters
    ///////////////////////////////
    uint64_t comm_buffer_ptr = bingo_get_l2_comm_buffer(current_chip_id);
    enable_sw_interrupts();
    ((comm_buffer_t *)comm_buffer_ptr)->lock = 0;
    ((comm_buffer_t *)comm_buffer_ptr)->chip_id = current_chip_id;
    program_snitches(current_chip_id, (comm_buffer_t *)comm_buffer_ptr);
    wakeup_snitches_cl(current_chip_id);
    asm volatile("fence" ::: "memory");

    ///////////////////////////////
    // 3. Arm DVFS on THIS chiplet (all four do it)
    ///////////////////////////////
    // Arm BEFORE the workload so the PM is already in DVFS mode when the workload
    // enables EN_IDLE_PM (see dvfs.h: pm_mode must be set before EN_IDLE_PM). boot_level
    // = normal seeds DVFS_ACK so the chip starts at "normal" (no spurious initial RAISE);
    // it only rings once the clusters actually go idle.
    dvfs_init(N_CLUSTERS_PER_CHIPLET + 1, BINGO_PM_NORMAL_POWER_LEVEL);
    printf("[dvfs][chip(%x,%x)] DVFS armed before workload\n", x, y);

    ///////////////////////////////
    // 4. Run the DVFS-stress workload
    ///////////////////////////////
    // The DAG's cluster cores go busy (gemm) then idle (host-check gap) several times;
    // this chiplet's PM rings its host doorbell on each transition and dvfs_trap_handler
    // records it.
    int ret = kernel_execution();
    clear_host_sw_interrupt(current_chip_id);

    ///////////////////////////////
    // 5. Narrate the serviced DVFS transitions
    ///////////////////////////////
    // Stop further DVFS traps before printing (the ISR records silently; printing happens
    // here, outside interrupt context, to stay reentrant-safe).
    asm volatile("csrc mstatus, %0" ::"r"(1 << 3));  // clear mstatus.MIE
    printf("[dvfs][chip(%x,%x)] workload done\n", x, y);
    dvfs_dump_log();

    // Keep all four compute chiplets alive at a rendezvous so none returns early
    // (which would end the multi-chip simulation before the others finish).
    chip_barrier((volatile comm_buffer_t *)comm_buffer_ptr, 0x00, 0x11, 1);
    return ret;
}
