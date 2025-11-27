# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>
# Fanchen Kong <fanchen.kong@kuleuven.be>
# Yunhao Deng <yunhao.deng@kuleuven.be>

# Add user applications to APPS variable

# Host only applications
APPS  = hello_world
APPS += hemaia_clk_rst_configurator
APPS += hemaia_system_xdma_copy
# # Host + Device applications
APPS += offload
APPS += offload_bingo
APPS += offload_bingo_ci
APPS += offload_bingo_gemm_intra_chiplet
APPS += offload_bingo_gemm_multi_chiplet_vanilla
APPS += offload_bingo_gemm_multi_chiplet_broadcast
# # Host only applications
APPS += hemaia_mailbox
APPS += multichip_sync
APPS += hemaia_d2d_link_configurator
# # Host + Device applications
APPS += offload_multichip
APPS += offload_bingo_multichip
APPS += offload_bingo_mempool_xdma

TARGET ?= all

LIB_BINGO_DIR = runtime/libbingo


SUBDIRS = $(addprefix apps/,$(APPS))

.PHONY: all libs $(SUBDIRS)

all: $(SUBDIRS)

libs:
	$(MAKE) -C $(LIB_BINGO_DIR) gen-lib

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET)
