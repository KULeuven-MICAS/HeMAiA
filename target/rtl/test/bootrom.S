# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

#define MSTATUS_MIE 8

.global _start

_start:
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
    la    a1, device_tree
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
    # Load the Chip ID to the register
    csrr    s0, 0xf15
    # Transfer as mask by left shift 40 bits
    slli    s0, s0, 40
    # Set global pointer
    .option push
    .option norelax
    la      gp, __global_pointer$
    or      gp, gp, s0
    .option pop
    # Set stack pointer
    la      sp, __stack_pointer$
    or      sp, sp, s0
    # Set stack pointer at the local
    la      t1, exception
    csrw    mtvec, t1
    la      t0, entry_addr
    jalr    t0
exception:
    wfi
    j     exception

# Create a new section to place at the end of binary,
# as expected by test/common_lib.cc
.pushsection end, "ax"
entry_addr:
.dword      0x80000000
device_tree:
