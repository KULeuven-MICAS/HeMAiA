# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# Add user applications to APPS variable automatically. This is located in the apps directory with depth of 2.

# Those apps are standalone apps
# Will be gradually moved to the kernels
APPDIR  = apps/snax/snax-test-integration
APPDIR += apps/snax/snax-test-global-barrier
APPDIR += apps/snax/snax-test-bingo-alloc
APPDIR += apps/snax/snax-xdma-maxpool
APPDIR += apps/snax/snax-xdma-memset
APPDIR += apps/snax/snax-xdma-multicast
APPDIR += apps/snax/snax-xdma-copy
APPDIR += apps/snax/snax-xdma-hazard
APPDIR += apps/snax/snax-xdma-transpose
APPDIR += apps/snax/snax-xdma-local-transpose
APPDIR += apps/snax/snax-xdma-reshape
APPDIR += apps/snax/snax-xdma-multicast-chiplet
APPDIR += apps/snax/snax-xdma-multicast-exploration
APPDIR += apps/snax/snax-xdma-overhead
APPDIR += apps/snax/snax-xdma-reshape-multicast
APPDIR += apps/snax/versacore-matmul-profile-1cluster-4chip
APPDIR += apps/snax/versacore-matadd-profile-1cluster-4chip
APPDIR += apps/snax/gemm-msplit-1cluster-4chip
APPDIR += apps/snax/gemm-msplit-4cluster-1chip
# In the future we only need this offload bingo app
APPDIR += apps/snax/snax-bingo-offload

# If DEVICE_APPS is specified, filter APPDIR to only include those apps.
# This is useful for single-sw builds to avoid building everything.
ifdef DEVICE_APPS
    # Filter APPDIR entries that end with any of the names in DEVICE_APPS
    SELECTED_APPS = $(foreach target,$(DEVICE_APPS),$(filter %/$(target),$(APPDIR)))
    # If filtering returned something, use it; otherwise fallback to nothing or specific error
    # To be safe, if DEVICE_APPS is 'None' or empty, we build nothing in APPDIR
    ifeq ($(DEVICE_APPS), None)
        SELECTED_APPS =
    else ifeq ($(DEVICE_APPS),)
        SELECTED_APPS =
    else
        # If no match found but it wasn't None/empty, maybe it's a full path or we just use what was given
        ifeq ($(SELECTED_APPS),)
            SELECTED_APPS = $(foreach app,$(DEVICE_APPS),apps/snax/$(app))
        endif
    endif
else
    SELECTED_APPS = $(APPDIR)
endif

TARGET ?= all

RTDIR    = math
RTDIR   += runtime

ALL_HOST_LIST   = host_app_list.tmp
ALL_ORIGIN_LIST = host_app_origin.tmp
ALL_DEV_LIST    = dev_app_list.tmp
ALL_TMPS       = $(ALL_HOST_LIST) $(ALL_ORIGIN_LIST) $(ALL_DEV_LIST)

aggregate-tmps: $(ALL_TMPS)

# Concatenate the per-app registration files into one list, but only replace the
# aggregate when the concatenation actually differs. apps/common.mk hangs every
# origin_<host>.ld off host_app_origin.tmp, so a gratuitous mtime bump here relinks
# and re-dumps every device ELF. Build into a .new temp (NOT .tmp -- that suffix is
# globbed back into $^ below) and mv only on a real change.
define concat_if_changed
	@> $@.new
	@$(foreach f,$^,cat $(f) >> $@.new;)
	@if cmp -s $@.new $@; then rm -f $@.new; else mv -f $@.new $@; fi
endef

$(ALL_HOST_LIST): $(wildcard host_app_list.*.tmp)
	$(concat_if_changed)

$(ALL_ORIGIN_LIST): $(wildcard host_app_origin.*.tmp)
	$(concat_if_changed)

$(ALL_DEV_LIST): $(wildcard dev_app_list.*.tmp)
	$(concat_if_changed)

.PHONY: all $(RTDIR) $(APPDIR) aggregate-tmps

all: $(SELECTED_APPS)

$(RTDIR):
	$(MAKE) -C $@ $(TARGET)

define APP_template
.PHONY: $(1)
$(1): $(RTDIR) $(ALL_TMPS)
	$$(MAKE) -C $(1) $$(TARGET)
endef

$(foreach app,$(APPDIR),$(eval $(call APP_template,$(app))))
