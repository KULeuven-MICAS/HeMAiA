# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>
# Fanchen Kong <fanchen.kong@kuleuven.be>
# Yunhao Deng <yunhao.deng@kuleuven.be>

# Add user applications to APPS variable
APPS = hemaia_mailbox
# APPS  = hello_world
# APPS += hemaia_clk_rst_configurator
# APPS += hemaia_mailbox
# ifneq ($(findstring multichip,$(CFG_OVERRIDE)),)
# # If chiplet cfg is used, offloaf_multichip is compiled with the support to execute applications on cores at different chips
# # host only applications
# APPS += hemaia_d2d_link_configurator
# APPS += multichip_sync
# # offload applications
# # The legacy offload
# APPS += offload_multichip
# # The bingo offload
# APPS += offload_bingo_multichip
# else
# # Otherwise, single chip offload is compiled
# # The legacy offload
# APPS += offload
# # The bingo offload
# APPS += offload_bingo
# endif

# XDMA copy test application
APPS += hemaia_system_xdma_copy

TARGET ?= all

LIB_HERO_DIR = runtime/libhero
LIB_BINGO_DIR = runtime/libbingo


SUBDIRS = $(addprefix apps/,$(APPS))

.PHONY: all libs $(SUBDIRS)

all: $(SUBDIRS)

libs:
	$(MAKE) -C $(LIB_HERO_DIR) gen-lib
	$(MAKE) -C $(LIB_BINGO_DIR) gen-lib

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET)
