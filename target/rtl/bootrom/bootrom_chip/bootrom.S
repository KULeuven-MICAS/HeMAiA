# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

#define MSTATUS_MIE 8

.global _start

_start:
    # Clear all registers
    li x1, 0
    li x2, 0
    li x3, 0
    li x4, 0
    li x5, 0
    li x6, 0
    li x7, 0
    li x8, 0
    li x9, 0
    li x10, 0
    li x11, 0
    li x12, 0
    li x13, 0
    li x14, 0
    li x15, 0
    li x16, 0
    li x17, 0
    li x18, 0
    li x19, 0
    li x20, 0
    li x21, 0
    li x22, 0
    li x23, 0
    li x24, 0
    li x25, 0
    li x26, 0
    li x27, 0
    li x28, 0
    li x29, 0
    li x30, 0
    li x31, 0
    # Park non-64-bit cores: inspect MISA MSB
    csrr    t0, misa
    bltz    t0, _cva6_init

# WARNING: Do *not* use rv64 instructions in this option block!
# Check the dumps to be certain!
.option push
.option norvc

_snitch_park:
    # Prepare arguments for snRuntime
    csrr  a0, mhartid
    # Set trap vector
    la      t0, _snitch_resume
    csrw    mtvec, t0
    # Enable software and cluster interrupts
    csrsi   mstatus, MSTATUS_MIE
    lui     t0, 0x80  # (1 << 19) cluster interrupts
    addi    t0, t0, 8 # (1 << 3) software interrupts
    csrw    mie, t0
    wfi

_snitch_resume:
    la      t0, __soc_ctrl_scratch1
    lw      t0, 0(t0)
    jalr    ra, 0(t0)
    j       _snitch_park

.option pop

# You may use rv64 instuctions in this block.
_cva6_init:
    # Previously was 0xf15
    # But 0xf15 is used by MCONFIGGPTR in new cva6
    # So we use 0xf16 instead
    csrr    s0, 0xf16
    # Transfer as mask by left shift 40 bits
    slli    s0, s0, 40
    # Set global pointer at the local
    .option push
    .option norelax
    la      gp, __global_pointer$
    or      gp, gp, s0
    .option pop
    # Set stack pointer at the local
    la      sp, __stack_pointer$
    or      sp, sp, s0
    # Set the interrupt vector
    la      t1, exception
    csrw    mtvec, t1
    call    bootrom
	li      t0, 0x80000000
    or      t0, t0, s0
    jalr    t0
    j       _start
exception:
    wfi
    j     exception
