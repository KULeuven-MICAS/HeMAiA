# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

# Add user applications to APPS variable automatically. This is located in the apps directory with depth of 2.

# Now we only compile the snax apps
# APPDIR  = $(shell find apps/snax -mindepth 1 -maxdepth 1 -type d)
APPDIR = apps/snax/snax-nvsa
# APPDIR = apps/snax/snax-cgra-cmatmul
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
