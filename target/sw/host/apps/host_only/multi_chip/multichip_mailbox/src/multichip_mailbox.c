// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#include "libbingo/bingo_api.h"
#include "host.h"
#include "hemaia_clk_rst_controller.h"

#define TARGET_CHIP_LOC_X 1
#define TARGET_CHIP_LOC_Y 0
#define TARGET_CHIP_ID chip_loc_to_chip_id(TARGET_CHIP_LOC_X, TARGET_CHIP_LOC_Y)

// Compute-chiplet rectangle used by chip_barrier (top-left .. bottom-right).
#define BARRIER_TL_CHIP_ID 0x00
#define BARRIER_BR_CHIP_ID 0x11

int main(){
    // Bear in mind that all the function calls here will be executed by all the chiplets
    // The chip id and chip address prefix is needed to differentiate the chiplets
    uint8_t current_chip_id = get_current_chip_id();
    // Program the Chiplet Topology
    hemaia_d2d_link_initialize(current_chip_id);
    // Init the uart for printf
    init_uart(get_current_chip_baseaddress(), 32, 1);
    // Enable vector extension
    enable_vec();
    printf("Multi-chip HW Mailbox\r\n");
    printf("Chip(%x, %x): [Host] Start Program\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    // Communication buffer pointer
    comm_buffer_t* comm_buf =
        (comm_buffer_t*)bingo_get_l2_comm_buffer(current_chip_id);
    enable_sw_interrupts();
    // Make sure the comm_buffer is visible
    asm volatile("fence" ::: "memory");

    // We need to first sync all the chiplets
    uint8_t sync_checkpoint = 1;
    chip_barrier(comm_buf, BARRIER_TL_CHIP_ID, BARRIER_BR_CHIP_ID, sync_checkpoint);
    sync_checkpoint++;
    // let chip 0 write to chip 1's H2H mailbox
    if (current_chip_id == 0) {
        uint8_t target_chip_loc_x = TARGET_CHIP_LOC_X;
        uint8_t target_chip_loc_y = TARGET_CHIP_LOC_Y;
        uint8_t target_chip_id = TARGET_CHIP_ID;
        printf("Chip(%x, %x): [Host] Write to Chip(%x, %x)'s H2H Mailbox\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), target_chip_loc_x, target_chip_loc_y);
        int wr_ret;
        uint32_t wr_retries = 0;
        uint64_t write_data = 0xdeadbeefcafebabe;
        wr_ret = bingo_write_h2h_mailbox(target_chip_id, write_data, 0 /* wait forever */, &wr_retries);
        printf("Chip(%x, %x): [Host] Write %lx, return: %d, retries: %d\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), write_data, wr_ret, wr_retries);
    } else if (current_chip_id == TARGET_CHIP_ID) {
        printf("Chip(%x, %x): [Host] Read from its own H2H Mailbox\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
        uint64_t read_data = 0;
        uint32_t rd_retries;
        int rd_ret = bingo_read_h2h_mailbox(&read_data, 0 /* wait forever */, &rd_retries);
        printf("Chip(%x, %x): [Host] Read data: %lx\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), read_data);
    } else {
        printf("Chip(%x, %x): [Host] Do nothing\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    }
    printf("Chip(%x, %x): [Host] End Program\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    return 0;
}