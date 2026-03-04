# Copyright 2023 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
MK_DIR   = $(shell pwd)
$(info Making in $(MK_DIR))
APP     = snax-bingo-offload
SRCS    = src/snax-bingo-offload.c

INCDIRS += libsnaxkernel

ifeq ($(HOST_APP_TYPE), offload_bingo_sw)
    RISCV_CFLAGS += -DOFFLOAD_BINGO_SW
endif

ifeq ($(HOST_APP_TYPE), offload_bingo_hw)
    RISCV_CFLAGS += -DOFFLOAD_BINGO_HW
endif

include ./libsnaxkernel/Makefile
include ../../common.mk

# llvm-objdump -t target/sw/device/apps/snax/snax-bingo-offload/build/snax-bingo-offload.elf | grep __snax_kernel_dummy | cut -c1-8
# get the kernel address
