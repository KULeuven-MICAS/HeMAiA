// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#include "offload_bingo.h"

int main() {
    
    uintptr_t current_chip_address_prefix =
        (uintptr_t)get_current_chip_baseaddress();
    uint8_t current_chip_id = get_current_chip_id();

    init_uart(current_chip_address_prefix, 32, 1);
    printf("Single-chip Offload Bingo Main\r\n");
    printf("Chip(%x, %x): [Host] Start Offloading Program\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());

    ///////////////////////////////
    // 1. Init the clk manager
    ///////////////////////////////
    // Set clk manager to 1 division for a faster simulation time
    enable_clk_domain(0, 1);
    enable_clk_domain(1, 1);
    enable_clk_domain(2, 1);
    enable_clk_domain(3, 1);
    enable_clk_domain(4, 1);
    enable_clk_domain(5, 1);
    printf("Chip(%x, %x): [Host] Init CLK Manager\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());

    ///////////////////////////////
    // 2. Init the mailbox
    ///////////////////////////////
    host_init_local_dev();
    printf("Chip(%x, %x): [Host] Init MailBox\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    ///////////////////////////////
    // 3. Wake up all the clusters
    ///////////////////////////////

    // 3.1 The pointer to the communication buffer
    volatile comm_buffer_t* comm_buffer_ptr = (comm_buffer_t*)0;
    initialize_comm_buffer((comm_buffer_t*)comm_buffer_ptr);
    comm_buffer_ptr = (comm_buffer_t*)chiplet_addr_transform(((uint64_t)&__narrow_spm_start));
    enable_sw_interrupts();

    // 3.2 Program Snitch entry point and communication buffer
    comm_buffer_ptr->lock = 0;
    comm_buffer_ptr->chip_id = current_chip_id;
    program_snitches(current_chip_id, comm_buffer_ptr);

    // 3.3 Start Snitches
    wakeup_snitches_cl(current_chip_id);
    asm volatile("fence" ::: "memory");
    printf("Chip(%x, %x): [Host] Wake up clusters\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());

    ///////////////////////////////
    // 4. Run the bingo runtime
    ///////////////////////////////

    printf("Chip(%x, %x): [Host] Start Bingo\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    int ret = kernel_execution();
    // By default the clusters will pull up the interrupt line once the tasks are done
    // So we clean up the interrupt line here
    clear_host_sw_interrupt(current_chip_id);
    printf("Chip(%x, %x): [Host] Offload Finish with ret = %d\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), ret);

    // Wait for job done and return Snitch exit code
    return ret;
}
