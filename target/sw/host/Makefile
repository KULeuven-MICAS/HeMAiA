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
# XDMA applications
APPS += hemaia_system_xdma_copy
# Offload applications
APPS += offload
APPS += offload_bingo
APPS += offload_bingo_ci

ifneq ($(findstring multichip,$(CFG_OVERRIDE)),)
# If chiplet cfg is used, offload_multichip is compiled with the support to execute applications on cores at different chips
# Host only applications
APPS += hemaia_d2d_link_configurator
APPS += hemaia_mailbox
APPS += multichip_sync
# Offload applications
APPS += offload_multichip
APPS += offload_bingo_multichip
endif

TARGET ?= all

LIB_BINGO_DIR = runtime/libbingo


SUBDIRS = $(addprefix apps/,$(APPS))

.PHONY: all libs $(SUBDIRS)

all: $(SUBDIRS)

libs:
	$(MAKE) -C $(LIB_BINGO_DIR) gen-lib

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET)
