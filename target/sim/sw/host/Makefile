# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

# Add user applications to APPS variable
APPS  = hello_world
ifneq ($(findstring chiplet,$(CFG_OVERRIDE)),)
# If chiplet cfg is used, offloaf_multichip is compiled with the support to execute applications on cores at different chips
APPS += offload_multichip
else
# Otherwise, simple offload is compiled, which is mainly used for ci. 
APPS += offload
endif

TARGET ?= all

SUBDIRS = $(addprefix apps/,$(APPS))

.PHONY: all $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET)
