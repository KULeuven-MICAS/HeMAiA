// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>
// Fanchen Kong <fanchen.kong@kueleuven.be>
// Xiaoling Yi <xiaoling.yi@kuleuven.be>

// The following code will be run on all hosts for the chiplet system
// We use the chiplet_id to identify each chiplet
// The Main Chiplet has the chiplet_id=0
// For the host main, the goal is to init the system and manage the offload tasks to the snax clusters
// For the chiplets:
// 1. Init the clk manager
// 2. Init the allocator
// 3. Wake up the chiplet local clusters
//    a. This is done in parallel by all the chiplets, the clusters will wait for the main host core to start
//    b. The cluster in each chiplet will polling their local mailbox in L2
// 4. Running the bingo runtime
//    a. Notice all the chiplets will run the same bingo runtime code, which means they will have the same DFG
//       hence the same task list in the chiplet local memory
//    b. Based on the assigned_chiplet_id in the DFG, each chiplet takes the corresponding tasks and execute them
//       (see details in the bingo_runtime_schedule function in bingo_api.c)

#include "offload_bingo_hw.h"
// DVFS runtime (trap handler + dvfs_init). Included here (not in host.h) so only
// this test links the DVFS handler + its printf footprint; other apps are unaffected.
#include "dvfs.h"

int main() {
    // Bear in mind that all the function calls here will be executed by all the chiplets
    // The chip id and chip address prefix is needed to differentiate the chiplets
    uint8_t current_chip_id = get_current_chip_id();
    // Program the Chiplet Topology
    hemaia_d2d_link_initialize(current_chip_id);
    // Init the uart for printf
    init_uart(get_current_chip_baseaddress(), 32, 1);
    // Enable vector extension
    enable_vec();
    OFFLOAD_BINGO_HW_DEBUG_PRINT_SAFE("Multi-chip Offload HW Bingo Main\r\n");
    OFFLOAD_BINGO_HW_DEBUG_PRINT_SAFE("Chip(%x, %x): [Host] Start Offloading Program\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());

    ///////////////////////////////
    // 2. Init the Allocator
    ///////////////////////////////
    if(bingo_hemaia_system_mmap_init() < 0){
        OFFLOAD_BINGO_HW_DEBUG_PRINT_SAFE("Chip(%x, %x): [Host] Error when initializing Allocator\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    } else {
        OFFLOAD_BINGO_HW_DEBUG_PRINT_SAFE("Chip(%x, %x): [Host] Init Allocator Success\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    }
    ///////////////////////////////
    // 3. Wake up all the clusters
    ///////////////////////////////

    // 3.1 The pointer to the communication buffer
    uint64_t local_l3_heap_manager = bingo_get_l3_heap_manager(current_chip_id);
    uint64_t comm_buffer_ptr = bingo_get_l2_comm_buffer(current_chip_id);

    enable_sw_interrupts();

    // 3.2 Program Snitch entry point and communication buffer
    // Initialize communication buffer
    ((comm_buffer_t *)comm_buffer_ptr)->lock = 0;
    ((comm_buffer_t *)comm_buffer_ptr)->chip_id = current_chip_id;
    program_snitches(current_chip_id, (comm_buffer_t *)comm_buffer_ptr);

    // 3.3 Start Snitches
    wakeup_snitches_cl(current_chip_id);
    asm volatile("fence" ::: "memory");
    // printf("Chip(%x, %x): [Host] Wake up clusters\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    ///////////////////////////////
    // 4. DVFS bring-up test
    ///////////////////////////////
    // Only chip(0,0) exercises the scheduler PM + new DVFS notify path. Its clusters
    // were woken above and now block on the (empty) ready queue, so once we arm DVFS the
    // PM sees the whole chiplet idle and rings the host MSIP doorbell -> dvfs_trap_handler
    // prints the (mimicked) voltage/frequency control sequence and acks the PM. The other
    // chiplets run no workload; they only sit at the barrier below so they do not return
    // early (which would end the multi-chip simulation before chip00 finishes).
    int ret = 0;
    // Arm DVFS on chip(0,0) BEFORE the workload so its PM monitors chip00's busy<->idle
    // transitions during execution and rings the host doorbell. Only chip00 enables DVFS;
    // the other chiplets run the workload normally without DVFS.
    if (current_chip_id == 0x00) {
        // boot_level = normal: seed DVFS_ACK so the PM starts at "normal" and only rings
        // once chip00 actually goes idle (no spurious initial RAISE).
        dvfs_init(N_CLUSTERS_PER_CHIPLET + 1, BINGO_PM_NORMAL_POWER_LEVEL);
        printf("[dvfs][chip0] DVFS armed before workload\n");
    }
    // All chiplets run their part of the (cross-chiplet) int32_add workload. This releases
    // the clusters and produces real busy<->idle transitions; chip00's PM then rings the
    // DVFS doorbell and dvfs_trap_handler prints the mimicked V/F control sequence.
    ret = kernel_execution();
    clear_host_sw_interrupt(current_chip_id);
    if (current_chip_id == 0x00) {
        // Stop further DVFS traps before printing (the ISR records silently; printing
        // happens here, outside interrupt context, to stay reentrant-safe).
        asm volatile("csrc mstatus, %0" ::"r"(1 << 3));  // clear mstatus.MIE
        printf("[dvfs][chip0] workload done\n");
        dvfs_dump_log();
    }
    // Keep all four compute chiplets alive at a rendezvous so none returns early.
    chip_barrier((volatile comm_buffer_t *)comm_buffer_ptr, 0x00, 0x11, 1);
    OFFLOAD_BINGO_HW_DEBUG_PRINT_SAFE("Chip(%x, %x): [Host] Offload Finish\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    return ret;
}
