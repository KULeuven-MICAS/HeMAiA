# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# Workloads to build separate ELFs for
APPS =  gemm_double_buffer_1cluster
APPS += gemm_seperate_load_and_compute_parallel_1cluster
APPS += gemm_seperate_load_and_compute_serial_1cluster
APPS += gemm_stacked_1cluster
APPS += gemm_sweep_1cluster
APPS += xdma_all_ops_1cluster
APPS += xdma_sweep_1cluster
APPS += moe4_4cluster
APPS += attn_test_single_gemm_4cluster
APPS += attn_test_quantize_4cluster
APPS += attn_test_dequantize_4cluster
APPS += attn_test_int32_add_4cluster
APPS += attn_test_ksplit_gemm_4cluster
APPS += attn_test_softmax_4cluster
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
