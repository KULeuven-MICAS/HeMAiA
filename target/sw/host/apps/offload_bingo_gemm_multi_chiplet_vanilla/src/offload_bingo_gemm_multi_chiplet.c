// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>

#include "offload_bingo_gemm_multi_chiplet.h"

int main() {
    
    // Bear in mind that all the function calls here will be executed by all the chiplets
    // The chip id and chip address prefix is needed to differentiate the chiplets
    uintptr_t current_chip_address_prefix = (uintptr_t)get_current_chip_baseaddress();
    uint8_t current_chip_id = get_current_chip_id();
    // Init the uart for printf
    init_uart(current_chip_address_prefix, 32, 1);
    printf("Multi-chip Offload Bingo Main\r\n");
    printf("Chip(%x, %x): [Host] Start Offloading Program\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());

    ///////////////////////////////
    // 2. Init the Allocator
    ///////////////////////////////
    if(bingo_hemaia_system_mmap_init() < 0){
        printf("Chip(%x, %x): [Host] Error when initializing Allocator\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    } else {
        printf("Chip(%x, %x): [Host] Init Allocator Success\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    }
    ///////////////////////////////
    // 3. Wake up all the clusters
    ///////////////////////////////

    // 3.1 The pointer to the communication buffer
    O1HeapInstance *local_l3_heap_manager = bingo_get_l3_heap_manager(current_chip_id);
    comm_buffer_t* comm_buffer_ptr = bingo_get_l2_comm_buffer(current_chip_id);

    enable_sw_interrupts();

    // 3.2 Program Snitch entry point and communication buffer
    // Initialize communication buffer
    initialize_comm_buffer(comm_buffer_ptr);
    comm_buffer_ptr->lock = 0;
    comm_buffer_ptr->chip_id = current_chip_id;
    program_snitches(current_chip_id, comm_buffer_ptr);

    // 3.3 Start Snitches
    wakeup_snitches_cl(current_chip_id);
    asm volatile("fence" ::: "memory");
    // printf("Chip(%x, %x): [Host] Wake up clusters\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    ///////////////////////////////
    // 4. Run the bingo runtime
    ///////////////////////////////
    // We need to first sync all the chiplets
    uint8_t sync_checkpoint = 1;
    // chip_barrier(comm_buffer_ptr, 0x00, 0x11, sync_checkpoint);
    // sync_checkpoint++;
    // printf("Chip(%x, %x): [Host] All chiplets synced, start Bingo\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    int ret = 0;
    ret = kernel_execution();
    clear_host_sw_interrupt(current_chip_id);
    printf("Chip(%x, %x): [Host] Offload Finish\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    // Sync before exit
    chip_barrier(comm_buffer_ptr, 0x00, 0x11, sync_checkpoint);
    sync_checkpoint++;
    return ret;
}
