// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>
// Fanchen Kong <fanchen.kong@kueleuven.be>

// The following code will be run on all hosts for the chiplet system
// We use the chiplet_id to identify each chiplet
// The Main Chiplet has the chiplet_id=0
// For the host main, the goal is to init the system and manage the offload tasks to the snax clusters
// For the chiplets:
// 1. Init the clk manager
// 2. Init the mailbox for the bingo runtime
// 3. Wake up the chiplet local clusters
//    a. This is done in parallel by all the chiplets, the clusters will wait for the main host core to start
//    b. The cluster in each chiplet will polling their local mailbox in L2
// After this step, there are diversions of the main host core and the other chiplet host cores:
// Other chiplets:
// 4. Set flag signals to the soc ctrl
// 5. Put itself to sleep
// Main chiplet:
// 4. Get all the mailbox addresses for all the other chiplets
// 5. Offload the tasks to the chiplets according to the dependency graph
//    a. The local offloading is based on the mailbox in local L2
//    b. The remote offloading is based on the mailbox in the remote L2 memory

#include "offload_bingo_multichip.h"
int main() {
    printf("Multi-chip Offload Bingo Main\r\n");
    // Bear in mind that all the function calls here will be executed by all the chiplets
    // The chip id and chip address prefix is needed to differentiate the chiplets
    uintptr_t current_chip_address_prefix = (uintptr_t)get_current_chip_baseaddress();
    uint8_t current_chip_id = get_current_chip_id();
    // Init the uart for printf
    init_uart(current_chip_address_prefix, 32, 1);
    printf("Chip(%x, %x): [Host] Start Offloading Program\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());

    ///////////////////////////////
    // 1. Init the clk manager
    ///////////////////////////////
    // Set clk manager to 1 division for a faster simulation time
    
    enable_clk_domain(0, 1);
    enable_clk_domain(1, 1);
    printf("Chip(%x, %x): [Host] Init CLK Manager\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    ///////////////////////////////
    // 2. Init the mailbox
    ///////////////////////////////

    host_init_local_dev();
    printf("Chip(%x, %x): [Host] Init MailBox\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    ///////////////////////////////
    // 3. Wake up all the clusters
    ///////////////////////////////

    // 3.1 Reset and ungate all quadrants, deisolate
    reset_and_ungate_quadrants_all(current_chip_id);
    deisolate_all(current_chip_id);

    // 3.2 The pointer to the communication buffer
    volatile comm_buffer_t* comm_buffer_ptr = (comm_buffer_t*)0;
    initialize_comm_buffer((comm_buffer_t*)comm_buffer_ptr);
    comm_buffer_ptr = (comm_buffer_t*)chiplet_addr_transform(((uint64_t)&__narrow_spm_start));
    enable_sw_interrupts();

    // 3.3 Program Snitch entry point and communication buffer
    comm_buffer_ptr->lock = 0;
    comm_buffer_ptr->chip_id = current_chip_id;
    program_snitches(current_chip_id, comm_buffer_ptr);

    // 3.4 Start Snitches
    wakeup_snitches_cl(current_chip_id);
    asm volatile("fence" ::: "memory");
    printf("Chip(%x, %x): [Host] Wake up clusters\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    ///////////////////////////////
    // 4. Run the bingo runtime
    ///////////////////////////////
    // We need to first sync all the chiplets
    uint8_t sync_checkpoint = 1;
    chip_barrier(comm_buffer_ptr, 0x00, 0x10, sync_checkpoint);
    sync_checkpoint++;
    int ret = 0;
    // Only the main chiplet will perform the offloading
    if (current_chip_id == 0) {
        printf("Chip(%x, %x): [Host] Start Bingo\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
        ret = kernel_execution();
        // By default the clusters will pull up the interrupt line once the tasks are done
        // So we clean up the interrupt line here
        clear_host_sw_interrupt(current_chip_id);
        printf("Chip(%x, %x): [Host] Offload Finish\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    } else
    {
        printf("Chip(%x, %x): [Host] Enter WFI\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
        wait_sw_interrupt();
        // By default the clusters will pull up the interrupt line once the tasks are done
        // So we clean up the interrupt line here
        clear_host_sw_interrupt(current_chip_id);
    }
    return ret;
}
