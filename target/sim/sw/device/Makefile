# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

# The host partial-build creates an origin.ld for every device application it
# embeds. Build that exact set here so unregistered applications are skipped
# and every host dependency receives a device binary.
ORIGIN_LDS := $(wildcard apps/snax/*/build/origin.ld)
APPDIR     := $(patsubst %/build/origin.ld,%,$(ORIGIN_LDS))
TARGET ?= all


RTDIR    = math
RTDIR   += runtime
RTDIR   += snax


# RTDIR   += $(shell find snax -mindepth 1 -maxdepth 1 -type d)

.PHONY: all $(RTDIR) $(APPDIR)

all: $(APPDIR)

$(RTDIR):
	$(MAKE) -C $@ $(TARGET)

$(APPDIR): $(RTDIR)
	$(MAKE) -C $@ $(TARGET)
