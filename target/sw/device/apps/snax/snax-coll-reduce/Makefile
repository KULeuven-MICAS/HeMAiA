# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# The E2 reduce (offload_legacy, multi_chip): the in-fabric xDMA chains against software
# reductions; the inputs and goldens are generated from data/params.hjson.

APP     = snax-coll-reduce

INCDIRS += data

SRCS    = src/snax-coll-reduce.c

include ./data/Makefile
include ../../common.mk

$(DEP): $(DATA_H)
