// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "host.h"

// Hardware IDs are (x << 4) | y. The board with the EAST connector is
// (0,0); the board with the WEST connector is (1,0), i.e. 0x10, NOT 0x01.
#define FPGA_EAST_CHIP 0x00u
#define FPGA_WEST_CHIP 0x10u
#define FPGA_LOCAL_ADDR_MASK UINT64_C(0xffffffffff)

enum {
    FPGA_BOOT_IDLE = 0x46504741u,
    FPGA_BOOT_READY = 0x52454144u,
    FPGA_BOOT_GO = 0x474f2121u,
    FPGA_BOOT_ACK = 0x41434b21u,
};

typedef struct {
    uint32_t ready;
    uint32_t go;
    uint32_t ack;
} fpga_boot_mailbox_t;

// UART loads this initialized L3 data on BOTH boards BEFORE either is run.
// Do not put it in .bss or clear it in main: a peer can write READY before
// this host reaches _start, whose initialize_bss() would erase the message.
// Explicit .data also works with older host linker scripts lacking .data.*.
static volatile fpga_boot_mailbox_t boot_mailbox
    __attribute__((section(".data"), aligned(64))) = {
        FPGA_BOOT_IDLE, FPGA_BOOT_IDLE, FPGA_BOOT_IDLE};

static volatile fpga_boot_mailbox_t *mailbox_on(uint8_t chip_id) {
    // PC-relative symbol addresses may already contain this chip's prefix.
    uint64_t offset = (uint64_t)(uintptr_t)&boot_mailbox & FPGA_LOCAL_ADDR_MASK;
    return (volatile fpga_boot_mailbox_t *)(uintptr_t)
        chiplet_addr_transform_full(chip_id, offset);
}

static void wait_for_peer(uint8_t chip_id) {
    volatile fpga_boot_mailbox_t *local = mailbox_on(chip_id);
    uint8_t peer_id = chip_id == FPGA_EAST_CHIP ? FPGA_WEST_CHIP
                                              : FPGA_EAST_CHIP;
    volatile fpga_boot_mailbox_t *peer = mailbox_on(peer_id);

    printf("[FPGA %02x] Waiting for peer %02x; start its loaded image via UART\r\n",
           chip_id, peer_id);

    // Either UART run order is allowed, with an arbitrarily long gap.
    // 0x10 announces READY, 0x00 releases it with GO, and 0x10 acknowledges.
    // All local initialization must precede this protocol; none follows it.
    if (chip_id == FPGA_WEST_CHIP) {
        peer->ready = FPGA_BOOT_READY;
        asm volatile("fence iorw, iorw" ::: "memory");
        while (local->go != FPGA_BOOT_GO) {
            asm volatile("nop" ::: "memory");
        }
        asm volatile("fence iorw, iorw" ::: "memory");
        peer->ack = FPGA_BOOT_ACK;
        asm volatile("fence iorw, iorw" ::: "memory");
    } else {
        while (local->ready != FPGA_BOOT_READY) {
            asm volatile("nop" ::: "memory");
        }
        asm volatile("fence iorw, iorw" ::: "memory");
        peer->go = FPGA_BOOT_GO;
        asm volatile("fence iorw, iorw" ::: "memory");
        while (local->ack != FPGA_BOOT_ACK) {
            asm volatile("nop" ::: "memory");
        }
        asm volatile("fence iorw, iorw" ::: "memory");
    }
}

int main(void) {
    uint8_t chip_id = get_current_chip_id();
    printf("[HeMAiA] Two-FPGA legacy offload, chip %02x\r\n", chip_id);

    if (chip_id != FPGA_EAST_CHIP && chip_id != FPGA_WEST_CHIP) {
        printf("[FPGA] Expected EAST board ID 00 or WEST board ID 10\r\n");
        return 1;
    }

    printf("[FPGA %02x] Using existing UART/D2D setup\r\n", chip_id);

    volatile comm_buffer_t *comm_buffer = (volatile comm_buffer_t *)(uintptr_t)
        chiplet_addr_transform((uint64_t)(uintptr_t)&__narrow_spm_start);
    printf("[FPGA %02x] Preparing local Snitch launch\r\n", chip_id);
    initialize_comm_buffer((comm_buffer_t *)comm_buffer);
    comm_buffer->chip_id = chip_id;
    enable_sw_interrupts();
    program_snitches(chip_id, comm_buffer);
    asm volatile("fence iorw, iorw" ::: "memory");

    // Both SPM regions are cached in occamy_cva6.sv. Preserve the boot flow's
    // cache state and bypass cached mailbox reads only for the host handshake.
    printf("[FPGA %02x] Entering host handshake\r\n", chip_id);
    uint16_t dcache_enabled = get_d_cache_enable();
    set_d_cache_enable(0);
    asm volatile("fence iorw, iorw" ::: "memory");
    wait_for_peer(chip_id);
    asm volatile("fence iorw, iorw" ::: "memory");
    set_d_cache_enable(dcache_enabled);
    asm volatile("fence iorw, iorw" ::: "memory");
    printf("[FPGA %02x] Host handshake complete; D-cache state restored\r\n",
           chip_id);
    printf("[FPGA %02x] Both hosts ready; starting local Snitch workload\r\n",
           chip_id);
    wakeup_snitches_cl(chip_id);
    int ret = wait_snitches_done(chip_id);
    printf("[FPGA %02x] GEMM finished, exit code %d\r\n", chip_id, ret);
    return ret;
}
