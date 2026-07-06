# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# Workloads to build separate ELFs for
APPS = gemm_msplit_4chiplet_1cluster
APPS += gemm_msplit_4chiplet_2cluster
APPS += gemm_parallel_dma_4chiplet_1cluster
APPS += gemm_ksplit_4chiplet_1cluster
APPS += dma_for_gemm_msplit_4chiplet_1cluster
APPS += int32_add_4chiplet_1cluster
APPS += dma_read_from_chip00_4chiplet_1cluster
APPS += dma_read_from_chip00_4chiplet_2cluster
APPS += dma_write_from_other_chip_4chiplet_1cluster
APPS += dvfs_cross_trigger_4chiplet_1cluster

SUBDIRS = $(addprefix workloads/,$(APPS))

TARGET ?= all

.PHONY: all clean partial-build finalize-build $(SUBDIRS)

all partial-build finalize-build: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET)

clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
