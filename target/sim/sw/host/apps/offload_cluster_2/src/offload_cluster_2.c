// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "host.c"

int main() {
    // Reset and ungate all quadrants, deisolate
    init_uart(32, 1);
    print_uart("[OCC] Offload Cluster 2 Only \r\n");

    // To enable the cluster gating
    // uncore c3 c2 c1 c0
    // 1      0  1  0  0
    // When using single cluster, uncomment the following line
    reset_and_ungate_quadrants(0x14);
    // reset_and_ungate_quadrants_all();
    deisolate_all();

    // Enable interrupts to receive notice of job termination
    enable_sw_interrupts();

    // Program Snitch entry point and communication buffer
    program_snitches();

    // Compiler fence to ensure Snitch entry point is
    // programmed before Snitches are woken up
    asm volatile("" ::: "memory");

    print_uart("[OCC] Calling Snitches to execute task \r\n");

    // Start Snitches
    // wakeup_snitches_cl();
    // When using single cluster, uncomment the following line
    wakeup_cluster(2);

    // Wait for job done and return Snitch exit code
    return wait_snitches_done();
}
