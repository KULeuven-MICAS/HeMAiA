# Copyright 2023 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

<<<<<<<< HEAD:target/sim/sw/device/apps/snax/snax-data-reshuffler/Makefile
APP     = snax-data-reshuffler
========
APP     = snax-gemmx-conv
>>>>>>>> 32685b4 (update gemmx sw (#69)):target/sim/sw/device/apps/snax/snax-gemmx-conv/Makefile

INCDIRS = data

INCDIRS += ../../../snax/data-reshuffler/include

# Include this binary in the final build
RISCV_LDFLAGS += ../../../snax/data-reshuffler/build/snax-data-reshuffler-lib.o

<<<<<<<< HEAD:target/sim/sw/device/apps/snax/snax-data-reshuffler/Makefile
SRCS    = src/snax-data-reshuffler.c
========
SRCS    = src/snax-gemmx-conv.c
>>>>>>>> 32685b4 (update gemmx sw (#69)):target/sim/sw/device/apps/snax/snax-gemmx-conv/Makefile

include ./data/Makefile
include ../../common.mk

$(DEP): $(DATA_H)
