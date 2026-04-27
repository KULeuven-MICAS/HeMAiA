# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# Workloads to build separate ELFs for
APPS = dummy_1cluster
APPS+= xdma_1d_copy_1cluster
APPS+= gemm_merge_load_and_compute_1cluster
APPS+= gemm_seperate_load_and_compute_1cluster
APPS+= gemm_tiled_1cluster
APPS+= gemm_stacked_1cluster

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