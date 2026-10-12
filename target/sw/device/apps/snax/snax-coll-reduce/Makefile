# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# The reduce collective of E2 (offload_legacy, multi_chip): ../snax-coll/src/snax-coll.c built
# with COLL_PRIM=2; the inputs and goldens are ../snax-coll/data.

APP     = snax-coll-reduce

INCDIRS += ../snax-coll/data

SRCS    = ../snax-coll/src/snax-coll.c

# before common.mk: its per-host-app rules expand the flags when it is included
RISCV_CFLAGS += -DCOLL_PRIM=2

include ../snax-coll/data/Makefile
include ../../common.mk


$(DEP): $(DATA_H)
