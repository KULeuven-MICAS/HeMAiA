// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#include "offload_bingo_ci.h"
// Used only for the CI
int main() {
    
    uintptr_t current_chip_address_prefix =
        (uintptr_t)get_current_chip_baseaddress();
    uint8_t current_chip_id = get_current_chip_id();

    init_uart(current_chip_address_prefix, 32, 1);
    // printf("Single-chip Offload Bingo Main\r\n");
    // printf("Chip(%x, %x): [Host] Start Offloading Program\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());

    ///////////////////////////////
    // 2. Init the Allocator
    ///////////////////////////////
    if(bingo_hemaia_system_mmap_init() < 0){
        printf("Chip(%x, %x): [Host] Error when initializing Allocator\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    } else {
        // printf("Chip(%x, %x): [Host] Allocator Init Success\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    }
    ///////////////////////////////
    // 3. Wake up all the clusters
    ///////////////////////////////

    // 3.1 The pointer to the communication buffer
    uint64_t local_l3_heap_manager = bingo_get_l3_heap_manager(current_chip_id);
    uint64_t comm_buffer_ptr = bingo_get_l2_comm_buffer(current_chip_id);
    enable_sw_interrupts();

    // 3.2 Program Snitch entry point and communication buffer
    ((comm_buffer_t *)comm_buffer_ptr)->lock = 0;
    ((comm_buffer_t *)comm_buffer_ptr)->chip_id = current_chip_id;
    program_snitches(current_chip_id, (comm_buffer_t *)comm_buffer_ptr);

    // 3.3 Start Snitches
    wakeup_snitches_cl(current_chip_id);
    asm volatile("fence" ::: "memory");
    // printf("Chip(%x, %x): [Host] Wake up clusters\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());

    ///////////////////////////////
    // 4. Run the bingo runtime
    ///////////////////////////////

    // printf("Chip(%x, %x): [Host] Start Bingo\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    int ret = kernel_execution();
    // By default the clusters will pull up the interrupt line once the tasks are done
    // So we clean up the interrupt line here
    clear_host_sw_interrupt(current_chip_id);
    // printf("Chip(%x, %x): [Host] Offload Finish with ret = %d\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), ret);
    // printf("Chip(%x, %x): [Host] End Offloading Program\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());   
    return ret;
}
