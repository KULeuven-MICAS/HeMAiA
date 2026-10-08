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
//     after initializing its TPS6287x and establishing the idle voltage,
//   - runs the DVFS-stress DAG (kernel_execution) whose cluster cores cycle
//     busy<->idle several times,
//   - applies each idle<->busy transition in dvfs_trap_handler and logs it,
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
    // 2. Configure THIS chiplet's PMIC and arm DVFS before waking the clusters.
    ///////////////////////////////
    // Every chiplet has its own I2C controller. The same PMIC address is valid
    // on these independent buses. The dvfs.h defines use millivolts, while the
    // PM's normal/idle levels are clock divisors (5/7 in the current workload).
    const dvfs_config_t config = {
        .pmic_address = DVFS_PMIC_ADDR,
        .peripheral_clock_hz = DVFS_PERIPH_HZ,
        .normal_level = BINGO_PM_NORMAL_POWER_LEVEL,
        .idle_level = BINGO_PM_IDLE_POWER_LEVEL,
        .normal_mv = DVFS_NORMAL_MV,
        .idle_mv = DVFS_IDLE_MV,
        .settle_us = DVFS_SETTLE_US,
    };
    int rc = dvfs_init(N_CLUSTERS_PER_CHIPLET + 1, &config);
    if (rc < 0) {
        printf("[dvfs][chip(%x,%x)] PMIC/DVFS init failed: %d. "
               "Check DVFS_PMIC_ADDR/PERIPH_HZ/NORMAL_MV/IDLE_MV/SETTLE_US.\n",
               x, y, rc);
        return rc;
    }
    printf("[dvfs][chip(%x,%x)] TPS6287x 0x%02x initialized; DVFS armed\n",
           x, y, (unsigned)config.pmic_address);

    ///////////////////////////////
    // 3. Wake up all the clusters
    ///////////////////////////////
    uint64_t comm_buffer_ptr = bingo_get_l2_comm_buffer(current_chip_id);
    enable_sw_interrupts();
    ((comm_buffer_t *)comm_buffer_ptr)->lock = 0;
    ((comm_buffer_t *)comm_buffer_ptr)->chip_id = current_chip_id;
    program_snitches(current_chip_id, (comm_buffer_t *)comm_buffer_ptr);
    wakeup_snitches_cl(current_chip_id);
    asm volatile("fence" ::: "memory");

    ///////////////////////////////
    // 4. Run the DVFS-stress workload
    ///////////////////////////////
    // The DAG's cluster cores go busy (gemm) then idle (host-check gap) several times;
    // this chiplet's PM rings its host doorbell on each transition and dvfs_trap_handler
    // applies the voltage/frequency transition and records it.
    int ret = kernel_execution();
    clear_host_sw_interrupt(current_chip_id);

    ///////////////////////////////
    // 5. Narrate the serviced DVFS transitions
    ///////////////////////////////
    // The workload has stopped. Disable automatic requests and return to idle.
    asm volatile("csrc mstatus, %0" ::"r"(1 << 3));  // clear mstatus.MIE
    writew(0, (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_enable_idle_pm_addr()));
    fence();
    rc = dvfs_apply_request((uint32_t)config.idle_level << DVFS_REQ_LEVEL_SHIFT);
    if (rc < 0) return rc;
    writew(config.idle_level,
           (uintptr_t)chiplet_addr_transform((uint64_t)quad_ctrl_dvfs_ack_addr()));
    printf("[dvfs][chip(%x,%x)] workload done; idle %d mV /%u\n",
           x, y, rc, (unsigned)config.idle_level);
    dvfs_dump_log();

    // Keep all four compute chiplets alive at a rendezvous so none returns early
    // (which would end the multi-chip simulation before the others finish).
    chip_barrier((volatile comm_buffer_t *)comm_buffer_ptr, 0x00, 0x11, 1);
    return ret;
}
