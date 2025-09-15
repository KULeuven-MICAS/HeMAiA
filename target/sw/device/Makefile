# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# Add user applications to APPS variable automatically. This is located in the apps directory with depth of 2.

# Those apps are standalone apps
# Will be gradually moved to the kernels
APPDIR  = apps/snax/snax-test-integration
APPDIR += apps/snax/snax-xdma-maxpool
APPDIR += apps/snax/snax-xdma-memset
APPDIR += apps/snax/snax-xdma-multicast
APPDIR += apps/snax/snax-xdma-copy
APPDIR += apps/snax/snax-xdma-hazard
APPDIR += apps/snax/snax-xdma-transpose
APPDIR += apps/snax/snax-xdma-local-transpose
APPDIR += apps/snax/snax-xdma-reshape
APPDIR += apps/snax/snax-xdma-multicast-exploration
APPDIR += apps/snax/snax-xdma-overhead
APPDIR += apps/snax/snax-axi-torture
APPDIR += apps/snax/snax-axi-torture-devil
APPDIR += apps/snax/snax-xdma-reshape-multicast
# In the future we only need this offload bingo app
APPDIR += apps/snax/snax-bingo-offload

TARGET ?= all

RTDIR    = math
RTDIR   += runtime


ALL_HOST_LIST   = host_app_list.tmp
ALL_ORIGIN_LIST = host_app_origin.tmp
ALL_DEV_LIST    = dev_app_list.tmp
ALL_TMPS       = $(ALL_HOST_LIST) $(ALL_ORIGIN_LIST) $(ALL_DEV_LIST)

aggregate-tmps: $(ALL_TMPS)

$(ALL_HOST_LIST): $(wildcard host_app_list.*.tmp)
# Make an empty file first, and then concatenate all the tmp files into one
	> $@
	$(foreach f,$^,cat $(f) >> $@;)

$(ALL_ORIGIN_LIST): $(wildcard host_app_origin.*.tmp)
# Make an empty file first, and then concatenate all the tmp files into one
	> $@
	$(foreach f,$^,cat $(f) >> $@;)

$(ALL_DEV_LIST): $(wildcard dev_app_list.*.tmp)
# Make an empty file first, and then concatenate all the tmp files into one
	> $@
	$(foreach f,$^,cat $(f) >> $@;)

.PHONY: all $(RTDIR) $(APPDIR) aggregate-tmps

all: $(APPDIR)

$(RTDIR):
	$(MAKE) -C $@ $(TARGET)

define APP_template
.PHONY: $(1)
$(1): $(RTDIR) $(ALL_TMPS)
	$$(MAKE) -C $(1) $$(TARGET)
endef

$(foreach app,$(APPDIR),$(eval $(call APP_template,$(app))))


