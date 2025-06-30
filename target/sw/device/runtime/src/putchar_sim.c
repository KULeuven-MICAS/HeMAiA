// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Provide an implementation for putchar.
// void _putchar(char character) {}
extern uintptr_t volatile tohost, fromhost;
static uintptr_t volatile *occamy_tohost = (volatile uintptr_t *)0x800004c0;
static uintptr_t volatile *occamy_fromhost = (volatile uintptr_t *)0x80000500;
// Rudimentary string buffer for putc calls.
extern uint32_t _edram;
#define PUTC_BUFFER_LEN (1024 - sizeof(size_t))
struct putc_buffer_header {
    size_t size;
    uint64_t syscall_mem[8];
};
static volatile struct putc_buffer {
    struct putc_buffer_header hdr;
    char data[PUTC_BUFFER_LEN];
} *const putc_buffer = (void *)&_edram;

// Provide an implementation for putchar.
void _putchar(char character) {
    volatile struct putc_buffer *buf = &putc_buffer[snrt_hartid()];
    buf->data[buf->hdr.size++] = character;
    if (buf->hdr.size == PUTC_BUFFER_LEN || character == '\n') {
        buf->hdr.syscall_mem[0] = 64;  // sys_write
        buf->hdr.syscall_mem[1] = 1;   // file descriptor (1 = stdout)
        buf->hdr.syscall_mem[2] = (uintptr_t)&buf->data;  // buffer
        buf->hdr.syscall_mem[3] = buf->hdr.size;          // length

        *occamy_tohost = (uintptr_t)buf->hdr.syscall_mem;
        while (*occamy_fromhost == 0)
            ;
        *occamy_fromhost = 0;

        buf->hdr.size = 0;
    }
}
