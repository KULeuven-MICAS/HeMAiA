// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#include "libbingo/bingo_api.h"
#include "host.h"
#include "hemaia_clk_rst_controller.h"
int main(){
    // Bear in mind that all the function calls here will be executed by all the chiplets
    // The chip id and chip address prefix is needed to differentiate the chiplets
    uint8_t current_chip_id = get_current_chip_id();
    // Program the Chiplet Topology
    hemaia_d2d_link_initialize(current_chip_id);
    // Init the uart for printf
    init_uart(get_current_chip_baseaddress(), 32, 1);
    printf("Multi-chip HW Mailbox\r\n");
    printf("Chip(%x, %x): [Host] Start Program\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
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
    // Communication buffer pointer
    volatile comm_buffer_t* comm_buffer_ptr;
    comm_buffer_ptr = (comm_buffer_t*)chiplet_addr_transform(((uint64_t)&__narrow_spm_start));
    initialize_comm_buffer((comm_buffer_t*)comm_buffer_ptr);
    enable_sw_interrupts();
    // Make sure the comm_buffer is visible
    asm volatile("fence" ::: "memory");

    // We need to first sync all the chiplets
    uint8_t sync_checkpoint = 1;
    chip_barrier(comm_buffer_ptr, 0x00, 0x10, sync_checkpoint);
    sync_checkpoint++;
    // let chip 0 write to chip 1's H2H mailbox
    if (current_chip_id == 0) {
        uint8_t target_chip_loc_x = 1;
        uint8_t target_chip_loc_y = 0;
        uint8_t target_chip_id = chip_loc_to_chip_id(target_chip_loc_x, target_chip_loc_y);
        printf("Chip(%x, %x): [Host] Write to Chip(%x, %x)'s H2H Mailbox\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), target_chip_loc_x, target_chip_loc_y);
        int wr_ret;
        uint32_t wr_retries = 0;
        wr_ret = bingo_write_h2h_mailbox(target_chip_id, 0xdeadbeefcafebabe, 0 /* wait forever */, &wr_retries);
        // Notice here are no chiplet barriers, the writes and reads are asynchronous and blocking
    } else {
        printf("Chip(%x, %x): [Host] Read from its own H2H Mailbox\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
        uint64_t read_data = 0;
        uint32_t rd_retries;
        int rd_ret = bingo_read_h2h_mailbox(&read_data, 0 /* wait forever */, &rd_retries);
    }
    printf("Chip(%x, %x): [Host] End Program\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    return 0;
}