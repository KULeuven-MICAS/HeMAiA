# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# Workloads to build separate ELFs for
APPS = concurrent_dma_1cluster
APPS +=  gemm_double_buffer_1cluster
APPS += gemm_serial_dma_1cluster
APPS += gemm_parallel_dma_1cluster
APPS += gemm_mem_chip_1cluster
APPS += gemm_stacked_1cluster
APPS += gemm_sweep_1cluster
APPS += xdma_all_ops_1cluster
APPS += xdma_sweep_1cluster
APPS += moe4_4cluster
APPS += basic_quantize_host
APPS += basic_dequantize_host
APPS += basic_int32_add_host
APPS += gemm_ksplit_4cluster
APPS += basic_softmax_host
APPS += attn_reference_4cluster
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
