// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "host.h"

// Global Variables for communication buffer
volatile comm_buffer_t* comm_buffer_ptr = (comm_buffer_t*)0;

int main() {
    // Reset and ungate all quadrants, deisolate
    uint32_t current_chip_id = get_current_chip_id();
    init_uart(get_current_chip_baseaddress(), 32, 1);
    printf("[HeMAiA] Single-chip Offload Legacy Main\r\n");

    // This app deliberately does not call hemaia_d2d_link_initialize_1c1m().
    // That routine is for a 1-compute + 1-memory-chiplet topology: it enables the
    // four D2D PHY clock domains and programs the D2D registers. A design built
    // with single_chip has neither -- its clock/reset controller has only the host
    // and cluster domains -- and writing a clock-domain valid bit past the end
    // wedges the peripheral bus, hanging the boot before the snitches are woken.
    //
    // A single-chip scenario has nothing to reach over D2D, so the link is simply
    // left alone here. A workload that does need the memory chiplet belongs in the
    // multi_chip app, which initialises the link itself.
    //
    // Keep the host and cluster clock divisions the routine used to set, so this
    // does not silently fall back to the slower reset default.
    enable_clk_domain(0, 7);  // host CPU
    for (uint8_t i = 0; i < N_CLUSTERS_PER_CHIPLET; i++) {
        enable_clk_domain(1 + i, 7);  // cluster i
    }
    comm_buffer_ptr = (comm_buffer_t*)chiplet_addr_transform(((uint64_t)&__narrow_spm_start));
    enable_sw_interrupts();
    comm_buffer_ptr->lock = 0;
    comm_buffer_ptr->chip_id = current_chip_id;
    program_snitches(current_chip_id, comm_buffer_ptr);

    printf("[HeMAiA] Calling snitch cluster on chip %d to execute the task\r\n",
           current_chip_id);

    // Start Snitches
    wakeup_snitches_cl(current_chip_id);

    int ret = wait_snitches_done(current_chip_id);

    printf("[HeMAiA] Snitch cluster done with exit code %d\r\n", ret);

    // Wait for job done and return Snitch exit code
    return ret;
}
