# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
HOST_APP_TYPE = offload_legacy
CHIP_TYPE     = multi_chip

DEVICE_APPS_WITH_MEMPOOL_IMAGE = gemm-msplit-1cluster-4chip gemm-msplit-4cluster-1chip

SRCS = src/offload_legacy_multi_chip.c

BINGO_HOST ?= 0 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 1 # True to include the device binary

# Include more dev apps here
DEVICE_APPS = snax-xdma-multicast-chiplet
DEVICE_APPS += gemm-msplit-1cluster-4chip
DEVICE_APPS += snax-test-chip-barrier
DEVICE_APPS += snax-coll-reduce

# The E2 reduce runs with the D2D links in DDR, as the layer experiments on the same
# platform do.
# `override`: the top-level make passes USER_FLAGS on the command line.
ifneq ($(filter snax-coll-%,$(DEV_APP)),)
override USER_FLAGS += -DHEMAIA_D2D_DDR=1
endif
include ../../common.mk

# The gemm-msplit device apps preload their input matrices into the
# memory chip through a mempool.bin image produced by its datagen. The target/sim
# apps build expects that image at $(BUILDDIR)/mempool.bin (see APP_MEM in
# target/sim/apps/Makefile), so copy it across once the device build has produced
# it (it exists by the finalize-build stage of single-sw). Gated on DEV_APP.
ifneq ($(filter $(DEV_APP),$(DEVICE_APPS_WITH_MEMPOOL_IMAGE)),)
MEMPOOL_SRC = $(DEVICE_DIR)/apps/snax/$(DEV_APP)/data/mempool.bin
finalize-build: $(BUILDDIR)/mempool.bin
$(BUILDDIR)/mempool.bin: $(MEMPOOL_SRC) | $(BUILDDIR)
	cp -f $< $@
endif
